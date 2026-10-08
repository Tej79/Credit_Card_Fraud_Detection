"""Raw transaction columns -> model input features.

Imported by both train.py (training) and app/main.py (serving) so the
exact same transformation runs in both places avoiding any
training/serving skew.
"""
import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["log_amount", "hour"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[V_COLS].copy()
    X["log_amount"] = np.log1p(df["Amount"])
    # `Time` is seconds elapsed since the first transaction in the dataset,
    # so it never repeats in future data. Hour-of-day is the part of it
    # that generalizes to new transactions.
    X["hour"] = (df["Time"] // 3600) % 24
    return X[FEATURES]
