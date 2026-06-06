#!/usr/bin/env bash
# One-command setup for the VitalScan Group 3 project.
# After cloning:  bash setup.sh   (needs Python 3.11+; developed on 3.14)
set -e
cd "$(dirname "$0")"

echo "[1/3] Creating virtual environment (.venv) ..."
python3 -m venv .venv

echo "[2/3] Installing dependencies (this can take a few minutes) ..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "[3/3] Training the served models (builds models/ — the API needs these) ..."
.venv/bin/python src/train_compare.py >/dev/null

echo
echo "Setup complete. Start the API with:"
echo "    .venv/bin/uvicorn src.api:app --port 8000"
echo "then open http://localhost:8000/docs"
echo
echo "Optional extras (not required for the API):"
echo "    .venv/bin/python src/visualize.py        # EDA / comparison / ROC figures"
echo "    .venv/bin/python src/shap_analysis.py    # SHAP explainability"
echo "    .venv/bin/python src/fetch_nhanes.py && .venv/bin/python src/nn_study.py  # NN study"
