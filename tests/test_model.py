from pathlib import Path

import pytest

from src.train import train
from src.model_utils import load_model, predict, MODEL_PATH


@pytest.fixture(scope="module", autouse=True)
def trained_model():
    accuracy = train()
    assert accuracy > 0.7
    yield
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()


def test_predict_returns_label_and_confidence():
    model = load_model()
    result = predict(model, {"feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4, "feature_4": 0.2})
    assert result["label"] == "class_a"
    assert 0.0 < result["confidence"] <= 1.0


def test_predict_distinguishes_classes():
    model = load_model()
    class_a = predict(model, {"feature_1": 4.9, "feature_2": 3.1, "feature_3": 1.5, "feature_4": 0.1})
    class_c = predict(model, {"feature_1": 6.7, "feature_2": 2.5, "feature_3": 5.8, "feature_4": 1.8})
    assert class_a["label"] != class_c["label"]
