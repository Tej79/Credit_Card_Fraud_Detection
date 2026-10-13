"""Train and evaluate credit card fraud models.

Fixes the following on top of the original .ipynb notebook:
  * time-ordered train/val/test split instead of a random shuffle
  * scaling fit on the training data only (inside the pipeline)
  * every model evaluated on the same, realistically imbalanced data
  * PR-AUC + a cost-based decision threshold instead of accuracy / 0.5 cutoff
  * experiments tracked in MLflow

Usage:
    python train.py --data creditcard.csv --out artifacts
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURES, V_COLS, make_features
from drift import build_reference

SEED = 1
REVIEW_COST = 5.0  # ASSUMPTION: $ cost of manually reviewing one flagged transaction


def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    df = df.sort_values("Time").reset_index(drop=True)
    i = int(len(df) * train_frac)
    j = int(len(df) * (train_frac + val_frac))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def total_cost(y, amount, flagged) -> float:
    """Missed fraud costs its amount; every flagged transaction costs a review."""
    missed = (y == 1) & ~flagged
    return float(amount[missed].sum() + REVIEW_COST * flagged.sum())


def best_threshold(y, amount, scores) -> float:
    grid = np.linspace(0.0, 1.0, 1001)
    costs = [total_cost(y, amount, scores >= t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def bootstrap_ci(y, amount, scores, threshold, n_boot=1000, seed=SEED) -> dict:
    """Resample the test set with replacement n_boot times and recompute
    precision/recall/f1/cost each time, to see how much they'd plausibly
    move on a different sample of the same size. Doesn't retrain anything."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {"precision": [], "recall": [], "f1": [], "cost": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # sample n indices with replacement
        m = evaluate(y[idx], amount[idx], scores[idx], threshold)
        for k in boot:
            boot[k].append(m[k])
    ci = {}
    for k, vals in boot.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci[k] = {"point": float(np.median(vals)), "lo": float(lo), "hi": float(hi)}
    return ci


def evaluate(y, amount, scores, threshold) -> dict:
    flagged = scores >= threshold
    p, r, f1, _ = precision_recall_fscore_support(
        y, flagged.astype(int), average="binary", zero_division=0
    )
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "cost": total_cost(y, amount, flagged),
        "cost_no_model": float(amount[y == 1].sum()),  # baseline: flag nothing
    }


def main(data: str, out: str) -> None:
    df = pd.read_csv(data)
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {n_before - len(df)} duplicate rows")

    train, val, test = time_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s}: {len(part):>7d} rows, {int(part['Class'].sum()):>4d} frauds")

    X_tr, y_tr = make_features(train), train["Class"].to_numpy()
    X_va, y_va = make_features(val), val["Class"].to_numpy()
    X_te, y_te = make_features(test), test["Class"].to_numpy()
    a_va, a_te = val["Amount"].to_numpy(), test["Amount"].to_numpy()

    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=SEED,
        ),
    }

    mlflow.set_experiment("fraud-detection")
    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if name == "xgboost":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                model.fit(X_tr, y_tr)
            scores = model.predict_proba(X_va)[:, 1]
            thr = best_threshold(y_va, a_va, scores)
            metrics = evaluate(y_va, a_va, scores, thr)
            mlflow.log_params({"model": name, "threshold": thr, "review_cost": REVIEW_COST})
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            results[name] = (model, thr, metrics)

    print("\nValidation results (threshold chosen to minimise cost):")
    print(pd.DataFrame({k: v[2] for k, v in results.items()}).T.round(3).to_string())

    # Pick by validation PR-AUC, then test with test set.
    best = max(results, key=lambda k: results[k][2]["pr_auc"])
    model, thr, _ = results[best]
    test_scores = model.predict_proba(X_te)[:, 1]
    test_metrics = evaluate(y_te, a_te, test_scores, thr)
    test_ci = bootstrap_ci(y_te, a_te, test_scores, thr)
    print(f"\nSelected model: {best} (threshold={thr:.3f})")
    print("Test results:", json.dumps({k: round(v, 3) for k, v in test_metrics.items()}, indent=2))
    print(f"\n{len(y_te)} test rows, {int(y_te.sum())} frauds — 95% bootstrap CI ({1000} resamples):")
    for k, v in test_ci.items():
        print(f"  {k:9s}: {v['point']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")

    reference = build_reference(X_tr)
    (out_dir / "feature_distribution.json").write_text(json.dumps(reference))

    meta = {
        "model": best,
        "threshold": thr,
        "features": FEATURES,
        "review_cost": REVIEW_COST,
        "test_metrics": test_metrics,
        "test_metrics_95ci": test_ci,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    with mlflow.start_run(run_name=f"final_{best}"):
        mlflow.log_params({"model": best, "threshold": thr, "review_cost": REVIEW_COST})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        for k, v in test_ci.items():
            mlflow.log_metrics({f"test_{k}_ci_lo": v["lo"], f"test_{k}_ci_hi": v["hi"]})
        mlflow.log_artifacts(str(out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    main(args.data, args.out)
