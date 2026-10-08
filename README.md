# Credit Card Fraud Detection — End-to-End ML Service

A production-style fraud detection pipeline: a model trained with leakage-free,
time-aware evaluation, served over a validated REST API, and deployed as a
containerized service on Google Cloud Run.

Started from exploratory work in a Colab notebook (model comparison on the
[Kaggle `mlg-ulb/creditcardfraud` dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)),
then rebuilt as a proper pipeline to fix evaluation issues in that notebook
(see [Evaluation methodology](#evaluation-methodology)) and to demonstrate
production ML and deployment skills end to end.

## Status

- [x] Model training & evaluation
- [x] Serving API (FastAPI)
- [x] Containerized, deployed to Cloud Run
- [ ] Load testing
- [ ] Infra as code (Terraform) + CI/CD pipeline

## Repo structure

```
train.py                  # training pipeline: split, train, evaluate, select, save
features.py                # raw columns -> model features (shared by train + serve)
app/main.py                 # FastAPI service: /predict, /health
Dockerfile                  # serving image
requirements-serve.txt      # deps the serving image actually needs
requirements-train.txt      # requirements-serve.txt + MLflow
artifacts/                  # model.joblib + metadata.json (not committed — see below)
```

`creditcard.csv` and `mlruns/` are intentionally not committed (raw data and
local MLflow run history — see `.gitignore`). Download the dataset from
Kaggle and point `train.py` at it to reproduce.

## Problem

Fraud is ~0.17% of transactions in this dataset making it a realistic, heavily
imbalanced classification problem. The two failure modes have very different
costs: a missed fraud costs the transaction amount; a false alarm costs a
fixed review fee. The model and its decision threshold are both built around
that asymmetry rather than optimizing accuracy or a flat F1 score.

## Evaluation methodology

A few choices were made deliberately, after finding that the original
exploratory notebook's approach overstated performance:

- **No undersampling at evaluation time.** Models are trained with class
  weights (`class_weight="balanced"` / `scale_pos_weight`) rather than by
  undersampling the training set, and every model is scored on the real,
  imbalanced validation/test distribution. Scoring on an artificially
  rebalanced set inflates precision in a way that doesn't reflect production.
- **Time-ordered split** (70/15/15, train/val/test) instead of a random
  shuffle: a deployed model only ever predicts the future, so it should
  only ever be evaluated on it.
- **Scaling fit inside a pipeline, on training data only**: avoids leaking
  validation/test statistics into preprocessing.
- **PR-AUC**, not ROC-AUC or accuracy, used for model selection: ROC-AUC is
  insensitive to the absolute number of false positives when negatives vastly
  outnumber positives, which is exactly the situation here.
- **Cost-based threshold selection.** Rather than the default 0.5 cutoff, the
  threshold is chosen to minimize `sum(missed fraud $ amount) + review_cost *
  num_flagged`, searched over a grid of candidate thresholds on the
  validation set. `review_cost = $5` is an assumption (set in `train.py`),
  not a measured figure.
- **Test set touched exactly once**: used only to evaluate the single model
  selected by validation PR-AUC, not to pick among models.
- **Bootstrap confidence intervals** on all test metrics (1000 resamples,
  with replacement): the test set has only ~52 fraud cases, so point
  estimates alone would be misleading without a sense of their uncertainty.

## Results

Three models compared (logistic regression, random forest, XGBoost);
**random forest** selected by validation PR-AUC.

| Metric (test set) | Point estimate | 95% bootstrap CI |
|---|---|---|
| Precision | 0.83 | [0.72, 0.93] |
| Recall | 0.76 | [0.63, 0.86] |
| F1 | 0.79 | [0.69, 0.87] |
| PR-AUC | 0.771 | — |
| ROC-AUC | 0.934 | — |

**Cost at chosen threshold (0.184):** $2,607 vs. $6,169 if nothing were
flagged which is a ~58% reduction in fraud + review cost on the test period.
The cost interval is wide ($494–$5,515) because a handful of large-amount
frauds dominate the dollar total; this is flagged explicitly rather than
hidden, and treated as a directional number, not a precise one, at this
data scale (test set covers a few hours of a 2-day dataset).

Full numbers and run history are in `artifacts/metadata.json` and the local
MLflow store (`mlflow ui` against `mlruns/`).

## Running it

### Train

```bash
pip install -r requirements-train.txt
python train.py --data creditcard.csv --out artifacts
```

Trains all three models, prints the validation comparison and test metrics
with confidence intervals, and writes `artifacts/model.joblib` +
`artifacts/metadata.json`.

### Serve locally

```bash
pip install -r requirements-serve.txt
uvicorn app.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 50000, "Amount": 120.5, "V": [ ...28 values... ]}'
```

Interactive docs at `http://localhost:8000/docs`.

### Run in Docker

```bash
docker build -t fraud-api .
docker run -p 8080:8080 fraud-api
```

### Deploy to Cloud Run

```bash
gcloud artifacts repositories create fraud-detection \
  --repository-format=docker --location=$REGION

gcloud auth configure-docker ${REGION}-docker.pkg.dev

docker tag fraud-api ${REGION}-docker.pkg.dev/${PROJECT_ID}/fraud-detection/fraud-api:v1
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/fraud-detection/fraud-api:v1

gcloud run deploy fraud-api \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/fraud-detection/fraud-api:v1 \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=512Mi \
  --min-instances=0 \
  --max-instances=3
```

Currently deployed and reachable as a public HTTPS endpoint at https://fraud-api-314643089891.us-central1.run.app.

## API

**`GET /health`** — status, loaded model name, threshold, test metrics.

**`POST /predict`**
```json
{"Time": 50000, "Amount": 120.5, "V": [v1, v2, ..., v28]}
```
→
```json
{
  "fraud_probability": 0.786,
  "flagged": true,
  "threshold": 0.184,
  "model": "random_forest",
  "latency_ms": 61.2
}
```

### Sample request body
```bash
curl -X POST https://fraud-api-314643089891.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 154278, "Amount": 1504.93, "V": [-1.60021129907252,-3.48813018118561,-6.45930280667571,3.24681566349703,-1.61460769860713,-1.26037454854632,0.288223321921388,-0.0489639910109505,-0.734974617920653,-4.44148408070017,2.94437484242405,-3.80546898374331,-2.10222683492697,-6.1061832734711,-0.641736399701777,-1.55596289626148,-2.08406707082134,0.394247352869229,0.0833797227600835,3.18935489592753,1.19117480721903,-0.96714144500067,-1.46342052446637,-0.624230851495284,-0.176461532068143,0.400348351070197,0.152946864261687,0.477774924576796]}'
```

## Known limitations

- **Review cost ($5) is an assumption**,
  stated explicitly so the cost numbers are read as illustrative.
- **Dataset spans only ~2 days**, so the test set is a few hours of data and
  metric confidence intervals are correspondingly wide (see Results). More
  data would narrow these.
- **Cold starts**: Cloud Run scales to zero by default (`--min-instances=0`),
  so the first request after idle time is noticeably slower (~950ms observed
  locally) than a warm request (~60ms observed locally). No formal load test
  has been run yet against the deployed service.
- **Features are mostly PCA-anonymized** (`V1`–`V28`), limiting feature
  engineering to `Amount` and `Time`-derived fields.

## Next steps

- Drift monitoring (e.g. PSI on key features) triggering retraining.
- Load test the deployed endpoint for real p50/p95 latency under concurrency.
- Terraform for the Cloud Run service + Artifact Registry (currently deployed
  manually via `gcloud`).
- GitLab CI pipeline: lint/test → build → push → deploy on commit.
