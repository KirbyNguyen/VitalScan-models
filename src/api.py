"""
VitalScan Group 3 — Risk Scoring API (FastAPI).

Routes
------
GET  /health
GET  /models                              list conditions/variants/algos + metrics
POST /predict/{condition}/{variant}       condition: diabetes|heart, variant: full|deployable
       ?algo=<name>                       optional; defaults to best. Any of the 14 sklearn/boosting
                                           models OR 'torch_mlp' (the PyTorch net, if trained).
POST /predict/risk                        shared biomarker contract -> both deployable models

Most models are a full sklearn Pipeline (impute -> scale/encode -> classifier) saved as .pkl, so
missing inputs are imputed automatically and reported. 'torch_mlp' is a PyTorch model served from
a .pt bundle (weights + fitted preprocessor) — see src/train_torch.py.

Run:  .venv/bin/uvicorn src.api:app --reload --port 8000     (docs at /docs)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + xgboost OpenMP clash (macOS)
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
VARIANT_DIR = {"full": "full_clinical", "deployable": "deployable"}
CONDITIONS = ["diabetes", "heart"]

app = FastAPI(title="VitalScan – Group 3 Risk API", version="1.0")

# ---------------------------------------------------------------- registry
_META: Dict[tuple, dict] = {}
_CACHE: Dict[tuple, dict] = {}  # (cond, variant, algo) -> bundle

_TORCH: Dict[tuple, tuple] = {}   # (cond, variant) -> (torch model, saved bundle)

def _meta(cond: str, variant: str) -> dict:
    key = (cond, variant)
    if key not in _META:
        p = MODELS_DIR / VARIANT_DIR[variant] / cond / "_meta.json"
        if not p.exists():
            raise HTTPException(404, f"no models for {cond}/{variant}")
        _META[key] = json.loads(p.read_text())
    return _META[key]

def _bundle(cond: str, variant: str, algo: Optional[str]) -> dict:
    meta = _meta(cond, variant)
    algo = algo or meta["best_algo"]
    if algo not in meta["models"]:
        raise HTTPException(400, f"unknown algo '{algo}'. choose: {list(meta['models'])}")
    key = (cond, variant, algo)
    if key not in _CACHE:
        _CACHE[key] = joblib.load(MODELS_DIR / VARIANT_DIR[variant] / cond / f"{algo}.pkl")
    return _CACHE[key]

def _age_to_cdc_bucket(age):
    """Years -> BRFSS _AGEG5YR bucket (1=18-24, 2=25-29, … 13=80+)."""
    if age is None:
        return None
    if age < 25:
        return 1
    if age >= 80:
        return 13
    return min(13, 2 + (int(age) - 25) // 5)

def _highbp_from_bp(systolic, diastolic):
    """ACC/AHA 2017 high-BP flag from a measured reading; None if no BP supplied."""
    if systolic is None and diastolic is None:
        return None
    return int((systolic or 0) >= 130 or (diastolic or 0) >= 80)

def _confidence(prob: float, provided_ratio: float) -> str:
    if provided_ratio < 0.5:
        return "low"            # too much was imputed to trust the score
    margin = abs(prob - 0.5) * 2
    return "high" if margin > 0.5 else "medium"

def _build_row(feats: Dict[str, Any], cols):
    """1-row DataFrame with all model columns; missing -> NaN (the pipeline imputes them)."""
    provided = {k: v for k, v in feats.items() if v is not None and k in cols}
    imputed = [c for c in cols if c not in provided]
    X = pd.DataFrame([{c: provided.get(c, np.nan) for c in cols}], columns=cols)
    return provided, imputed, X

def _response(cond, variant, resolved, prob, thr, provided, cols, imputed, meta):
    s = meta["models"][resolved]
    return dict(condition=cond, variant=variant, algo=resolved,
                risk=round(prob, 4), at_risk=bool(prob >= thr), threshold=round(thr, 2),
                confidence=_confidence(prob, len(provided) / len(cols)),
                imputed_features=imputed,
                # how good the model that produced this score is (measured on the test set)
                model_metrics=dict(roc_auc=s["test_auc"], cv_auc=s["cv_auc"],
                                   precision=s["precision"], recall=s["recall"], f1=s["f1"],
                                   recall_at_threshold=s["recall_at_threshold"],
                                   precision_at_threshold=s["precision_at_recall80"]))

def _torch_predict(cond, variant, feats, meta):
    """Serve the PyTorch MLP: reload weights + fitted preprocessor, forward pass, sigmoid."""
    import torch                       # lazy — API startup stays torch-free
    import torch.nn as nn
    key = (cond, variant)
    if key not in _TORCH:
        p = MODELS_DIR / VARIANT_DIR[variant] / cond / "torch_mlp.pt"
        if not p.exists():
            raise HTTPException(400, "torch_mlp not trained for this group — run src/train_torch.py")
        b = torch.load(p, weights_only=False)   # bundle holds a pickled sklearn preprocessor
        # architecture MUST match src/torch_nn.py MLP (state_dict keys net.0/3/6.*)
        class _MLP(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
                                         nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                                         nn.Linear(32, 1))
            def forward(self, x):
                return self.net(x)
        m = _MLP(b["input_dim"]); m.load_state_dict(b["state_dict"]); m.eval()
        _TORCH[key] = (m, b)
    model, b = _TORCH[key]
    cols = b["features"]
    provided, imputed, X = _build_row(feats, cols)
    Z = b["preprocessor"].transform(X)
    Z = (Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)).astype("float32")
    with torch.no_grad():
        prob = float(torch.sigmoid(model(torch.tensor(Z)))[0, 0])
    return _response(cond, variant, "torch_mlp", prob, float(b["threshold"]),
                     provided, cols, imputed, meta)

def _predict(cond: str, variant: str, algo: Optional[str], feats: Dict[str, Any]) -> dict:
    meta = _meta(cond, variant)
    resolved = algo or meta["best_algo"]
    if resolved == "torch_mlp":                 # PyTorch path (not a sklearn .pkl Pipeline)
        return _torch_predict(cond, variant, feats, meta)
    bundle = _bundle(cond, variant, algo)
    cols = bundle["features"]
    provided, imputed, X = _build_row(feats, cols)
    prob = float(bundle["pipeline"].predict_proba(X)[0, 1])
    return _response(cond, variant, resolved, prob, float(bundle["threshold"]),
                     provided, cols, imputed, meta)

# ---------------------------------------------------------------- schemas
class DiabetesFull(BaseModel):
    Pregnancies: Optional[float] = None
    Glucose: Optional[float] = None
    BloodPressure: Optional[float] = Field(None, description="diastolic mmHg")
    SkinThickness: Optional[float] = None
    Insulin: Optional[float] = None
    BMI: Optional[float] = None
    DiabetesPedigreeFunction: Optional[float] = None
    Age: Optional[float] = None

class DiabetesDeployable(BaseModel):
    # CDC/BRFSS glucose-free model. Raw CDC coding (use /predict/risk for friendly mapping).
    HighBP: Optional[float] = Field(None, description="1 if high BP (≥130/80), else 0 — derive from Group 1 BP")
    BMI: Optional[float] = None
    Age: Optional[float] = Field(None, description="CDC age bucket 1–13 (1=18-24 … 13=80+)")
    Sex: Optional[float] = Field(None, description="1=male, 0=female")
    Smoker: Optional[float] = Field(None, description="0/1 — Group 4 questionnaire")
    PhysActivity: Optional[float] = Field(None, description="0/1 physically active — Group 4")
    GenHlth: Optional[float] = Field(None, description="self-rated general health 1=excellent … 5=poor")
    DiffWalk: Optional[float] = Field(None, description="0/1 difficulty walking — Group 4")

class HeartFull(BaseModel):
    age: Optional[float] = None
    sex: Optional[float] = None
    cp: Optional[float] = None
    trestbps: Optional[float] = None
    chol: Optional[float] = None
    fbs: Optional[float] = None
    restecg: Optional[float] = None
    thalach: Optional[float] = None
    exang: Optional[float] = None
    oldpeak: Optional[float] = None
    slope: Optional[float] = None
    ca: Optional[float] = None
    thal: Optional[float] = None

class HeartDeployable(BaseModel):
    age: Optional[float] = None
    trestbps: Optional[float] = Field(None, description="resting systolic BP (from Group 1)")
    thalach: Optional[float] = Field(None, description="heart rate (Group 1 resting HR proxy)")
    sex: Optional[float] = None
    cp: Optional[float] = Field(None, description="chest pain type (Group 4 self-report)")
    exang: Optional[float] = None

class BloodPressure(BaseModel):
    systolic: Optional[float] = None
    diastolic: Optional[float] = None

class Biomarkers(BaseModel):
    heart_rate: Optional[float] = None
    hrv_sdnn: Optional[float] = None
    stress_index: Optional[float] = None
    blood_pressure: BloodPressure = BloodPressure()

class Profile(BaseModel):
    age: Optional[float] = None          # years (mapped to CDC bucket for diabetes)
    sex: Optional[float] = None          # 1=male, 0=female
    bmi: Optional[float] = None
    cp: Optional[float] = None           # heart: chest-pain type 1–4
    exang: Optional[float] = None        # heart: exercise angina 0/1
    smoker: Optional[float] = None       # diabetes: 0/1
    phys_activity: Optional[float] = None  # diabetes: 0/1
    gen_health: Optional[float] = None   # diabetes: self-rated 1=excellent … 5=poor
    diff_walk: Optional[float] = None    # diabetes: difficulty walking 0/1

class Clinical(BaseModel):
    """Optional lab/clinical values — enable the FULL models in /predict/risk. If omitted,
    the full prediction imputes them and is flagged low-confidence (the app can't measure these)."""
    # diabetes-full (Pima)
    glucose: Optional[float] = None
    insulin: Optional[float] = None
    skin_thickness: Optional[float] = None
    pregnancies: Optional[float] = None
    diabetes_pedigree: Optional[float] = None
    # heart-full (UCI)
    chol: Optional[float] = None
    fbs: Optional[float] = None
    restecg: Optional[float] = None
    oldpeak: Optional[float] = None
    slope: Optional[float] = None
    ca: Optional[float] = None
    thal: Optional[float] = None

class RiskRequest(BaseModel):
    biomarkers: Biomarkers
    profile: Profile = Profile()
    clinical: Clinical = Clinical()    # optional — powers the _full section
    conditions: Optional[list] = None

# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    out = {}
    for variant in VARIANT_DIR:
        for cond in CONDITIONS:
            try:
                m = _meta(cond, variant)
            except HTTPException:
                continue
            out.setdefault(cond, {})[variant] = dict(
                features=m["features"], best_algo=m["best_algo"],
                algos=list(m["models"]), metrics=m["models"])
    return out

@app.get("/models/best")
def best_models():
    """Best model + its metrics for every condition x variant (the winners only)."""
    out = {}
    for cond in CONDITIONS:
        for variant in VARIANT_DIR:
            try:
                m = _meta(cond, variant)
            except HTTPException:
                continue
            algo = m["best_algo"]
            s = m["models"][algo]
            out.setdefault(cond, {})[variant] = dict(
                best_algo=algo, model=s["model"],
                cv_auc=s["cv_auc"], test_auc=s["test_auc"],
                precision=s["precision"], recall=s["recall"], f1=s["f1"],
                threshold=s["threshold"], recall_at_threshold=s["recall_at_threshold"],
                precision_at_recall80=s["precision_at_recall80"],
                tuned_params=s.get("tuned_params"), features=m["features"])
    return out

@app.post("/predict/diabetes/full")
def predict_diabetes_full(body: DiabetesFull, algo: Optional[str] = Query(None)):
    return _predict("diabetes", "full", algo, body.model_dump())

@app.post("/predict/diabetes/deployable")
def predict_diabetes_deployable(body: DiabetesDeployable, algo: Optional[str] = Query(None)):
    return _predict("diabetes", "deployable", algo, body.model_dump())

@app.post("/predict/heart/full")
def predict_heart_full(body: HeartFull, algo: Optional[str] = Query(None)):
    return _predict("heart", "full", algo, body.model_dump())

@app.post("/predict/heart/deployable")
def predict_heart_deployable(body: HeartDeployable, algo: Optional[str] = Query(None)):
    return _predict("heart", "deployable", algo, body.model_dump())

@app.post("/predict/risk")
def predict_risk(req: RiskRequest, algo: Optional[str] = Query(None)):
    """Shared biomarker contract -> both deployable models (the Week-5 integration path)."""
    bm, pr = req.biomarkers, req.profile
    sys_bp = bm.blood_pressure.systolic
    dia_bp = bm.blood_pressure.diastolic
    heart_feats = dict(age=pr.age, sex=pr.sex, cp=pr.cp, exang=pr.exang,
                       trestbps=sys_bp, thalach=bm.heart_rate)  # HR is resting-HR proxy
    # diabetes = CDC glucose-free model: HighBP derived from BP, age -> CDC bucket, rest from profile
    diab_feats = dict(HighBP=_highbp_from_bp(sys_bp, dia_bp), BMI=pr.bmi,
                      Age=_age_to_cdc_bucket(pr.age), Sex=pr.sex, Smoker=pr.smoker,
                      PhysActivity=pr.phys_activity, GenHlth=pr.gen_health, DiffWalk=pr.diff_walk)
    heart = _predict("heart", "deployable", algo, heart_feats)
    diab = _predict("diabetes", "deployable", algo, diab_feats)

    # FULL-clinical models (extra data): meaningful only if `clinical` labs are supplied,
    # otherwise the lab features impute and the result is flagged low-confidence.
    cl = req.clinical
    diab_full_feats = dict(Glucose=cl.glucose, Insulin=cl.insulin, SkinThickness=cl.skin_thickness,
                           BloodPressure=dia_bp, BMI=pr.bmi, Age=pr.age,
                           Pregnancies=cl.pregnancies, DiabetesPedigreeFunction=cl.diabetes_pedigree)
    heart_full_feats = dict(age=pr.age, sex=pr.sex, cp=pr.cp, trestbps=sys_bp, thalach=bm.heart_rate,
                            exang=pr.exang, chol=cl.chol, fbs=cl.fbs, restecg=cl.restecg,
                            oldpeak=cl.oldpeak, slope=cl.slope, ca=cl.ca, thal=cl.thal)
    diab_full = _predict("diabetes", "full", None, diab_full_feats)
    heart_full = _predict("heart", "full", None, heart_full_feats)

    # contract keys; hypertension served by the UCI heart-disease model (proxy) for now
    conf = "low" if "low" in (heart["confidence"], diab["confidence"]) else \
           ("high" if heart["confidence"] == diab["confidence"] == "high" else "medium")
    return {
        "diabetes_risk": diab["risk"],
        "hypertension_risk": heart["risk"],
        "confidence": conf,
        "_detail": {"diabetes": diab, "hypertension": heart},      # deployable (served, trustworthy)
        "_full": {"diabetes": diab_full, "hypertension": heart_full},  # full-clinical (needs labs)
        "_notes": [
            "Headline diabetes_risk/hypertension_risk and _detail use the DEPLOYABLE models (the "
            "trustworthy, app-obtainable scores). _full uses the full-clinical models and is only "
            "meaningful when `clinical` labs are supplied — otherwise its imputed_features list is "
            "long and confidence is 'low'.",
            "diabetes (deployable) uses the CDC/BRFSS glucose-free model; HighBP derived from BP (≥130/80).",
            "hypertension is the UCI heart-disease model as a proxy; train on cardio.csv for true hypertension.",
            "thalach is fed Group 1's resting heart_rate (the training feature is exercise max-HR).",
        ],
    }
