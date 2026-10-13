"""Compare a new batch of transactions against the training distribution
frozen at train time, and report per-feature PSI.

This project has no live production traffic yet, so there's no real
"new batch" to check against. As a stand-in, this script takes in any
CSV of transactions (e.g. the held-out test set, or a later slice of
the full dataset) to demonstrate the mechanism end to end.

Usage:
    python drift_check.py --reference artifacts/feature_distribution.json --data new_batch.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from features import make_features
from drift import check_drift, PSI_ALERT


def main(reference_path: str, data_path: str) -> int:
    reference = json.loads(Path(reference_path).read_text())
    df = pd.read_csv(data_path)
    X_new = make_features(df)

    report = check_drift(X_new, reference)
    print(f"Checked {len(X_new)} rows against the training reference "
          f"({len(reference)} features):\n")
    print(report.to_string(index=False))

    alerts = report[report["status"] == "ALERT"]
    if len(alerts):
        print(f"\n{len(alerts)} feature(s) at or above the PSI alert "
              f"threshold ({PSI_ALERT}): {', '.join(alerts['feature'])}")
        print("Consider investigating or retraining.")
        return 1  # non-zero exit so a scheduler/alerting job can detect this
    print("\nNo features show significant drift.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="artifacts/feature_distribution.json")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    sys.exit(main(args.reference, args.data))
