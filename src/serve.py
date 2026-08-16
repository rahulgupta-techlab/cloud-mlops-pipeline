"""FastAPI app serving predictions from the trained model."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.model_utils import load_model, predict

app = FastAPI(title="Cloud MLOps Pipeline")
model = None


class PredictRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float


@app.on_event("startup")
def load() -> None:
    global model
    model = load_model()


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> dict:
    assert model is not None, "Model not loaded - run src/train.py first"
    return predict(model, request.model_dump())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
