# Cloud MLOps Pipeline

A compact MLOps reference project: train a scikit-learn model on sample data,
serve predictions through a FastAPI endpoint, containerize it with Docker, and
deploy it with a Kubernetes manifest - with CI running tests on every push.

Built to show the full loop from training to a deployable service, not just a
notebook: reproducible training script, a served model with input validation,
a container image, a k8s deployment/service pair, and automated tests wired into
GitHub Actions.

## Stack

- Python 3.11, scikit-learn, FastAPI
- Docker for packaging
- Kubernetes manifests for deployment (`k8s/`)
- GitHub Actions for CI (`.github/workflows/ci.yml`)

## Project layout

```
src/
  train.py        # trains a classifier on data/sample_data.csv, saves model.joblib
  model_utils.py   # shared loading/prediction helpers
  serve.py         # FastAPI app exposing POST /predict
data/
  sample_data.csv
k8s/
  deployment.yaml  # Deployment + Service for the serving container
tests/
  test_model.py
```

## Running it

```bash
pip install -r requirements.txt
python -m src.train                 # writes model.joblib
uvicorn src.serve:app --reload      # serves POST /predict
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4, "feature_4": 0.2}'
```

### With Docker

```bash
docker build -t cloud-mlops-pipeline .
docker run --rm -p 8000:8000 cloud-mlops-pipeline
```

### Deploying to Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

## Tests / CI

```bash
pytest tests/
```

`.github/workflows/ci.yml` runs the test suite on every push and pull request.
