"""Generate synthetic 'new batch' files to test drift_check.py against.

Since this project has no live production traffic yet, this script creates
two stand-in batches from your own dataset:
  * a batch with NO artificial drift (a recent slice, as a sanity check --
    drift_check.py should report mostly OK, aside from the documented
    `hour` caveat for short windows)
  * a batch WITH artificial drift (Amount inflated, a couple of V columns
    shifted), to confirm drift_check.py actually catches a real shift

Usage:
    python make_test_batch.py --data creditcard.csv
"""
import argparse

import numpy as np
import pandas as pd


def main(data_path: str, n: int) -> None:
    df = pd.read_csv(data_path).sort_values("Time")

    # no_drift_batch: just the most recent n rows, unmodified
    no_drift = df.iloc[-n:].copy()
    no_drift.to_csv("no_drift_batch.csv", index=False)
    print(f"Wrote no_drift_batch.csv ({len(no_drift)} rows, unmodified)")

    # drift_batch: same rows, but with Amount and a couple of V columns
    # shifted, simulating a real change in transaction patterns
    rng = np.random.default_rng(0)
    drifted = no_drift.copy()
    drifted["Amount"] = drifted["Amount"] * 4 + 40
    drifted["V1"] = drifted["V1"] + 3.5
    drifted["V4"] = drifted["V4"] - 2.5
    drifted["V4"] += rng.normal(0, 0.5, len(drifted))  # add a little noise too
    drifted.to_csv("drift_batch.csv", index=False)
    print(f"Wrote drift_batch.csv ({len(drifted)} rows, "
          f"Amount/V1/V4 artificially shifted)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--n", type=int, default=5000, help="rows per batch")
    args = parser.parse_args()
    main(args.data, args.n)
