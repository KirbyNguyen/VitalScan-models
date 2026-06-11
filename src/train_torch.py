"""
Train + save a PyTorch MLP for each of the 4 SERVED groups so it's selectable in the API via
`?algo=torch_mlp`. Unlike the sklearn models (self-contained Pipelines), a torch model needs its
weights + the fitted preprocessor + input dim saved together — that's what `torch_mlp.pt` holds.

Adds a `torch_mlp` entry to each group's `_meta.json` (does NOT change `best_algo` — the default
stays the sklearn winner; torch_mlp is opt-in). Run AFTER train_compare.py:

    .venv/bin/python src/train_torch.py        ->  models/{full_clinical,deployable}/{diabetes,heart}/torch_mlp.pt
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + xgboost OpenMP clash (macOS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import json, copy, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

from train_compare import variant_spec, tuned_threshold, RNG
from torch_nn import MLP, preprocessor          # same 64->32->1 architecture used in the NN study

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
VARIANT_DIR = {"full": "full_clinical", "deployable": "deployable"}
torch.manual_seed(RNG); np.random.seed(RNG)

def _dense(Z):
    return (Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)).astype("float32")

def fit(Xtr, ytr, Xval, yval, epochs=300, patience=20, batch=128, lr=1e-3):
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

def proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy().ravel()

def run(cond, variant):
    spec = variant_spec(cond, variant)
    df, target, cont, cat = spec["df"], spec["target"], spec["cont"], spec["cat"]
    y = df[target].values
    X = df[cont + cat]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
    Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=RNG)
    pre = preprocessor(cont, cat).fit(Xtr2)
    Xtr2d, Xvald, Xted = _dense(pre.transform(Xtr2)), _dense(pre.transform(Xval)), _dense(pre.transform(Xte))

    model = fit(Xtr2d, ytr2, Xvald, yval)
    p, pv = proba(model, Xted), proba(model, Xvald)
    auc, vauc = roc_auc_score(yte, p), roc_auc_score(yval, pv)
    t = tuned_threshold(yte, p)

    out = ROOT / "models" / VARIANT_DIR[variant] / cond
    meta_path = out / "_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing — run src/train_compare.py first.")
    # weights + fitted preprocessor + input dim, all in one file the API reloads
    torch.save({"state_dict": model.state_dict(), "preprocessor": pre,
                "input_dim": Xtr2d.shape[1], "features": cont + cat,
                "threshold": float(t), "test_auc": float(auc), "cv_auc": float(vauc),
                "model": "PyTorch_MLP"}, out / "torch_mlp.pt")

    meta = json.loads(meta_path.read_text())
    meta["models"]["torch_mlp"] = dict(
        model="PyTorch_MLP", cv_auc=round(vauc, 3), test_auc=round(auc, 3),
        precision=round(precision_score(yte, p >= .5, zero_division=0), 3),
        recall=round(recall_score(yte, p >= .5), 3),
        f1=round(f1_score(yte, p >= .5, zero_division=0), 3),
        threshold=round(t, 2),
        recall_at_threshold=round(recall_score(yte, p >= t), 3),
        precision_at_recall80=round(precision_score(yte, p >= t, zero_division=0), 3),
        tuned_params=None)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  {cond}/{variant}: torch_mlp test_auc={auc:.3f} val_auc={vauc:.3f}  -> saved + _meta updated")

def main():
    for variant in ["full", "deployable"]:
        for cond in ["diabetes", "heart"]:
            run(cond, variant)
    print("\ntorch_mlp added to all 4 served groups. Restart the API to pick it up (?algo=torch_mlp).")

if __name__ == "__main__":
    main()
