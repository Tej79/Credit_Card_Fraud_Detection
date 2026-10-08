"""Serve the trained fraud model over HTTP.

Run locally:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health   liveness + which model/threshold is loaded
    POST /predict  score one transaction
"""
import json
import sys
import time
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# this is so that `from features import make_features` works whether this is run from
# the repo root or from inside app/
sys.path.append(str(Path(__file__).resolve().parent.parent))
from features import V_COLS, make_features

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

app = FastAPI(title="Fraud Detection API", version="1.0")

_model = None
_threshold = None
_metadata = None


@app.on_event("startup")
def load_model() -> None:
    """Load the model once at process startup, not per-request."""
    global _model, _threshold, _metadata
    _model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    _metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    _threshold = _metadata["threshold"]
    print(f"Loaded {_metadata['model']} model, threshold={_threshold:.3f}")


class Transaction(BaseModel):
    """One row of the raw Kaggle schema. V1-V28 are the PCA-anonymized features."""

    Time: float = Field(..., description="Seconds elapsed since the first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V: List[float] = Field(..., min_length=28, max_length=28, description="V1..V28, in order")

    def to_frame(self) -> pd.DataFrame:
        row = {col: val for col, val in zip(V_COLS, self.V)}
        row["Time"] = self.Time
        row["Amount"] = self.Amount
        return pd.DataFrame([row])


class PredictResponse(BaseModel):
    fraud_probability: float
    flagged: bool
    threshold: float
    model: str
    latency_ms: float


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model": _metadata["model"],
        "threshold": _threshold,
        "test_metrics": _metadata.get("test_metrics"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(txn: Transaction):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    X = make_features(txn.to_frame())
    proba = float(_model.predict_proba(X)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        fraud_probability=proba,
        flagged=proba >= _threshold,
        threshold=_threshold,
        model=_metadata["model"],
        latency_ms=round(latency_ms, 3),
    )
