"""
DEEP-LEARNING STUDY (NN-only — does NOT touch the served models or the API).

Answers: "does more data make the neural net better, and can it beat gradient boosting?"
  1. Head-to-head per large dataset: PyTorch MLP vs TabNet vs XGBoost (XGB = reference yardstick).
  2. Scaling curve on CDC (253k): MLP vs XGBoost ROC-AUC as training size grows 1k -> 200k.

Datasets (all large, glucose-allowed — this is a learning study, not the deployable model):
  CDC/BRFSS (253k) · iammustafatz 100k · cardiovascular 70k · NHANES (~15.5k).

Run:  .venv/bin/python src/nn_study.py
Outputs: reports/nn_study.csv, reports/nn_scaling.csv, reports/figures/nn_scaling.png
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + xgboost OpenMP clash (macOS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import warnings, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from torch_nn import MLP, preprocessor   # reuse the 64->32->1 architecture + preprocessing

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "data" / "extra"
RNG = 42
torch.manual_seed(RNG); np.random.seed(RNG)

# ----------------------------------------------------------------- datasets
def cdc():
    df = pd.read_csv(EX / "cdc_diabetes.csv").drop(columns=["ID"])
    return df, [c for c in df.columns if c != "Diabetes_binary"], [], "Diabetes_binary"

def iammustafatz():
    df = pd.read_csv(EX / "diabetes_100k.csv").drop_duplicates()
    return (df, ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease"],
            ["gender", "smoking_history"], "diabetes")

def cardio():
    df = pd.read_csv(EX / "cardio.csv", sep=";").drop(columns=["id"])
    return df, [c for c in df.columns if c != "cardio"], [], "cardio"

def nhanes():
    df = pd.read_csv(EX / "nhanes_diabetes.csv")
    return df, ["age", "sex", "bmi", "systolic", "diastolic"], [], "diabetes"

DATASETS = {"CDC_253k": cdc, "iammustafatz_100k": iammustafatz,
            "cardio_70k": cardio, "NHANES_15k": nhanes}

def _dense(Z):
    return (Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)).astype("float32")

def prep(df, cont, cat, target, n=None, seed=RNG, pre=None, Xte_raw=None):
    """Return train/val/test dense arrays. If pre+Xte_raw given, reuse them (scaling study)."""
    if n and n < len(df):
        df = df.sample(n, random_state=seed)
    y = df[target].astype(int).values
    X = df[cont + cat]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    pre = preprocessor(cont, cat).fit(Xtr2)
    return (_dense(pre.transform(Xtr2)), ytr2, _dense(pre.transform(Xval)), yval,
            _dense(pre.transform(Xte)), yte)

# ----------------------------------------------------------------- models
def fit_mlp(Xtr, ytr, Xval, yval, epochs=200, patience=15, batch=256, lr=1e-3):
    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xv, yv = torch.tensor(Xval), torch.tensor(yval, dtype=torch.float32).view(-1, 1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                         batch_size=batch, shuffle=True)
    best, best_state, wait = 1e9, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), yv).item()
        if v < best - 1e-4:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model

def auc_mlp(Xtr, ytr, Xval, yval, Xte, yte):
    m = fit_mlp(Xtr, ytr, Xval, yval)
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor(Xte))).numpy().ravel()
    return roc_auc_score(yte, p)

def auc_xgb(Xtr, ytr, Xte, yte):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                        eval_metric="logloss", scale_pos_weight=spw, random_state=RNG)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

def auc_tabnet(Xtr, ytr, Xval, yval, Xte, yte):
    clf = TabNetClassifier(seed=RNG, verbose=0)
    clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric=["auc"],
            max_epochs=40, patience=8, batch_size=1024, virtual_batch_size=256)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

# ----------------------------------------------------------------- experiments
def head_to_head(sample=20000):
    print("=== head-to-head (sample %d/dataset): MLP vs TabNet vs XGBoost ===" % sample)
    rows = []
    for name, loader in DATASETS.items():
        df, cont, cat, tgt = loader()
        Xtr, ytr, Xval, yval, Xte, yte = prep(df, cont, cat, tgt, n=sample)
        r = dict(dataset=name, n=min(sample, len(df)),
                 MLP=round(auc_mlp(Xtr, ytr, Xval, yval, Xte, yte), 3),
                 TabNet=round(auc_tabnet(Xtr, ytr, Xval, yval, Xte, yte), 3),
                 XGBoost_ref=round(auc_xgb(Xtr, ytr, Xte, yte), 3))
        rows.append(r); print("  ", r)
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "nn_study.csv", index=False)
    return rows

def scaling_curve():
    print("\n=== scaling on CDC: MLP vs XGBoost AUC as train size grows ===")
    df, cont, cat, tgt = cdc()
    y = df[tgt].astype(int).values
    Xtr_raw, Xte_raw, ytr_raw, yte = train_test_split(df[cont + cat], y, test_size=0.2,
                                                      stratify=y, random_state=RNG)
    pre = preprocessor(cont, cat).fit(Xtr_raw)
    Xte = _dense(pre.transform(Xte_raw))
    sizes = [1000, 5000, 20000, 80000, len(Xtr_raw)]
    rows = []
    for n in sizes:
        idx = np.random.RandomState(RNG).permutation(len(Xtr_raw))[:n]
        Xs_raw, ys = Xtr_raw.iloc[idx], ytr_raw[idx]
        Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(Xs_raw, ys, test_size=0.2,
                                                        stratify=ys, random_state=RNG)
        Xtr, Xval = _dense(pre.transform(Xs_tr)), _dense(pre.transform(Xs_val))
        mlp = auc_mlp(Xtr, ys_tr, Xval, ys_val, Xte, yte)
        xgb = auc_xgb(_dense(pre.transform(Xs_raw)), ys, Xte, yte)
        rows.append(dict(train_size=n, MLP=round(mlp, 3), XGBoost=round(xgb, 3),
                         gap=round(xgb - mlp, 3)))
        print("  ", rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports" / "nn_scaling.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.plot(res.train_size, res.MLP, "o-", label="PyTorch MLP")
    plt.plot(res.train_size, res.XGBoost, "s-", label="XGBoost (reference)")
    plt.xscale("log"); plt.xlabel("training rows (log scale)"); plt.ylabel("test ROC-AUC")
    plt.title("CDC diabetes: NN vs XGBoost as data grows"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "figures" / "nn_scaling.png", dpi=120)
    plt.close()
    return rows

def main():
    head_to_head()
    scaling_curve()
    print("\n-> reports/nn_study.csv, reports/nn_scaling.csv, reports/figures/nn_scaling.png")

if __name__ == "__main__":
    main()
