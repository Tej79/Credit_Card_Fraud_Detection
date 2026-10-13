"""PSI-based feature drift detection.

Used in two places, at two different times:
  * train.py freezes the TRAINING distribution of each feature right
    after training, and saves it to artifacts/feature_distribution.json.
  * drift_check.py later compares a NEW batch of data against that frozen
    reference, as a separate, scheduled job.
"""
import numpy as np
import pandas as pd

N_BINS = 10
PSI_WARN = 0.10    # worth watching
PSI_ALERT = 0.25   # investigate / consider retraining


def bin_edges(train_col: pd.Series, n_bins: int = N_BINS) -> list:
    """Quantile-based bin edges from the training distribution.

    Quantile bins (not equal-width) so each training bin starts with
    roughly equal mass.
    """
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(train_col.quantile(qs).to_numpy())
    # extend the outer edges to +/- inf so a new batch's extreme values
    # (e.g. a transaction larger than anything seen in training) still
    # land in the outermost bin instead of being dropped as NaN
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.tolist()


def bin_proportions(col: pd.Series, edges: list) -> list:
    counts = pd.cut(col, bins=edges, include_lowest=True).value_counts(sort=False)
    props = (counts / max(len(col), 1)).to_numpy()
    # PSI takes a log of a ratio; floor at a small value so an empty bin
    # (0%) never produces log(0) / divide-by-zero
    return np.clip(props, 1e-4, None).tolist()


def psi(train_props: list, new_props: list) -> float:
    t, n = np.array(train_props), np.array(new_props)
    return float(np.sum((n - t) * np.log(n / t)))


def build_reference(X_train: pd.DataFrame) -> dict:
    """Training time: freeze each feature's bin edges + proportions."""
    reference = {}
    for col in X_train.columns:
        edges = bin_edges(X_train[col])
        reference[col] = {
            "edges": edges,
            "train_props": bin_proportions(X_train[col], edges),
        }
    return reference


def check_drift(X_new: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Check time: compare a new batch against the frozen reference."""
    rows = []
    for col, ref in reference.items():
        if col not in X_new.columns:
            continue
        new_props = bin_proportions(X_new[col], ref["edges"])
        score = psi(ref["train_props"], new_props)
        status = "ALERT" if score >= PSI_ALERT else ("WARN" if score >= PSI_WARN else "OK")
        rows.append({"feature": col, "psi": round(score, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
