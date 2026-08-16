"""Shared helpers for loading the trained model and running predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
FEATURE_COLUMNS = ["feature_1", "feature_2", "feature_3", "feature_4"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH)


def predict(model, features: dict) -> dict:
    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
    label = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    confidence = float(max(proba))
    return {"label": str(label), "confidence": round(confidence, 4)}
