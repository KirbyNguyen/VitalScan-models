#!/usr/bin/env bash
# Smoke-test every VitalScan Group 3 API endpoint with the example JSON files.
# Usage:  bash examples/test_api.sh           (server must be running on :8000)
#         BASE=http://localhost:8000 bash examples/test_api.sh
set -e
BASE="${BASE:-http://localhost:8000}"
DIR="$(dirname "$0")"
j() { python3 -m json.tool; }   # pretty-print JSON

echo "============ GET /health ============"
curl -s "$BASE/health" | j

echo "============ GET /models/best ============"
curl -s "$BASE/models/best" | j

echo "============ POST /predict/heart/deployable ============"
curl -s -X POST "$BASE/predict/heart/deployable" \
     -H "Content-Type: application/json" -d @"$DIR/heart_deployable.json" | j

echo "============ POST /predict/heart/deployable?algo=xgboost ============"
curl -s -X POST "$BASE/predict/heart/deployable?algo=xgboost" \
     -H "Content-Type: application/json" -d @"$DIR/heart_deployable.json" | j

echo "============ POST /predict/diabetes/deployable (CDC, glucose-free) ============"
curl -s -X POST "$BASE/predict/diabetes/deployable" \
     -H "Content-Type: application/json" -d @"$DIR/diabetes_deployable.json" | j

echo "============ POST /predict/heart/full (evaluation only) ============"
curl -s -X POST "$BASE/predict/heart/full" \
     -H "Content-Type: application/json" -d @"$DIR/heart_full.json" | j

echo "============ POST /predict/diabetes/full (Pima, evaluation only) ============"
curl -s -X POST "$BASE/predict/diabetes/full" \
     -H "Content-Type: application/json" -d @"$DIR/diabetes_full.json" | j

echo "============ POST /predict/risk (full contract) ============"
curl -s -X POST "$BASE/predict/risk" \
     -H "Content-Type: application/json" -d @"$DIR/predict_risk.json" | j

echo "============ POST /predict/risk (with clinical labs -> _full becomes meaningful) ============"
curl -s -X POST "$BASE/predict/risk" \
     -H "Content-Type: application/json" -d @"$DIR/predict_risk_with_labs.json" | j

echo "============ POST /predict/risk (partial -> imputation + lower confidence) ============"
curl -s -X POST "$BASE/predict/risk" \
     -H "Content-Type: application/json" -d @"$DIR/predict_risk_partial.json" | j

echo "============ done ============"
