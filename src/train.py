"""Train a classifier on the sample dataset and save it for serving."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.model_utils import FEATURE_COLUMNS, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_data.csv"


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> float:
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    logger.info("Held-out accuracy: %.3f", accuracy)

    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)
    return accuracy


if __name__ == "__main__":
    train()
