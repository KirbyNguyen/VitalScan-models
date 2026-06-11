# Testing the VitalScan Group 3 API

## 1. Start the server (one terminal)
```bash
cd /Users/ducnguyen/Desktop/VitalScan-models
.venv/bin/uvicorn src.api:app --port 8000
```
Leave it running. You should see `Uvicorn running on http://127.0.0.1:8000`.

## 2. Three ways to test

### A) Browser (easiest, no commands) — Swagger UI
Open **http://localhost:8000/docs** → click an endpoint → **Try it out** → paste a JSON
body (from the files here) → **Execute**. You see the response inline.

### B) curl (one endpoint at a time)
```bash
# health check
curl -s http://localhost:8000/health

# best model per group
curl -s http://localhost:8000/models/best | python3 -m json.tool

# a prediction, from an example file
curl -s -X POST http://localhost:8000/predict/heart/deployable \
     -H "Content-Type: application/json" \
     -d @examples/heart_deployable.json | python3 -m json.tool

# pick a specific model with ?algo=
curl -s -X POST "http://localhost:8000/predict/diabetes/deployable?algo=xgboost" \
     -H "Content-Type: application/json" \
     -d @examples/diabetes_deployable.json | python3 -m json.tool

# the integration endpoint
curl -s -X POST http://localhost:8000/predict/risk \
     -H "Content-Type: application/json" \
     -d @examples/predict_risk.json | python3 -m json.tool
```

### C) Run the whole smoke test at once
```bash
bash examples/test_api.sh
```
Hits every endpoint with the example files and pretty-prints each response.

## 3. The example files

| File | Endpoint | Notes |
|---|---|---|
| `heart_deployable.json` | `POST /predict/heart/deployable` | scan + profile features |
| `diabetes_deployable.json` | `POST /predict/diabetes/deployable` | **CDC** features: `HighBP`, `Age`=bucket 1–13, `GenHlth` 1–5, rest 0/1 |
| `heart_full.json` | `POST /predict/heart/full` | all 13 clinical features (evaluation only) |
| `diabetes_full.json` | `POST /predict/diabetes/full` | Pima features incl. `Glucose` (evaluation only) |
| `predict_risk.json` | `POST /predict/risk` | full biomarker contract + profile |
| `predict_risk_with_labs.json` | `POST /predict/risk` | adds the optional `clinical` block → makes the `_full` models meaningful |
| `predict_risk_partial.json` | `POST /predict/risk` | only BP + heart rate → shows imputation + lower confidence |

**`/predict/risk` response now has two layers:** the headline `diabetes_risk`/`hypertension_risk`
+ `_detail` are the **deployable** (app-obtainable, trustworthy) scores; `_full` adds the
**full-clinical** model outputs — only meaningful if you send the optional `clinical` labs
(glucose, cholesterol, …), otherwise `_full` shows `confidence: low` and a long `imputed_features`.

## 4. Reading the response
```json
{
  "risk": 0.83,            // 0–1 risk score
  "at_risk": true,         // crossed the recall-tuned threshold?
  "threshold": 0.54,
  "confidence": "high",    // "low" if too many fields were imputed
  "imputed_features": [],  // fields the API had to fill in
  "model_metrics": { "roc_auc": 0.875, "precision": 0.80, "recall": 0.81, "f1": 0.81, ... }
}
```
`/predict/risk` returns `{ diabetes_risk, hypertension_risk, confidence, _detail, _notes }`.

## Tips
- Every field is **optional** — omitted fields are median-imputed and listed in
  `imputed_features` (which lowers `confidence`). Send the listed fields for a trustworthy score.
- Add `?algo=<name>` to any `/predict/...` endpoint to use a specific model
  (names from `GET /models`). Unknown names return a 400 listing valid choices.
- `?algo=torch_mlp` serves the **PyTorch** neural net (needs `src/train_torch.py` to have run;
  `setup.sh` does this). Example:
  ```bash
  curl -s -X POST "http://localhost:8000/predict/heart/deployable?algo=torch_mlp" \
       -H "Content-Type: application/json" -d @examples/heart_deployable.json | python3 -m json.tool
  ```
