"""
Task 5 (stretch) — neural network vs tree-based models, in PyTorch.

A 3-layer fully-connected net (input -> 64 -> 32 -> 1, sigmoid output), trained with Adam +
binary cross-entropy and EARLY STOPPING on validation loss, compared head-to-head against the
tuned RandomForest and XGBoost on the SAME test split (full-clinical feature set).

Run:  .venv/bin/python src/torch_nn.py   ->  reports/nn_comparison.csv
"""
import os
# torch + xgboost/lightgbm each ship an OpenMP runtime; on macOS the duplicate load
# segfaults. Allow it (set before any OpenMP-linked import loads).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import warnings, copy
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from train_compare import load_diabetes, load_heart, DIABETES, HEART

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RNG = 42
torch.manual_seed(RNG); np.random.seed(RNG)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))           # logits; sigmoid applied at eval / via BCEWithLogits

    def forward(self, x):
        return self.net(x)

def preprocessor(cont, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), cont),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])

def train_mlp(Xtr, ytr, Xval, yval, epochs=500, patience=25):
    d = Xtr.shape[1]
    model = MLP(d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # class imbalance handled via pos_weight in BCEWithLogitsLoss
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xval_t, yval_t = torch.tensor(Xval, dtype=torch.float32), torch.tensor(yval, dtype=torch.float32).view(-1, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)

    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xval_t), yval_t).item()
        if v < best_val - 1e-4:
            best_val, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:            # early stopping on validation loss
                break
    model.load_state_dict(best_state)
    return model, ep + 1

def run(cond, df, spec):
    cont, cat = spec["full"]["cont"], spec["full"]["cat"]
    X, y = df[cont + cat], df[spec["target"]].values
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
    Xtr2_raw, Xval_raw, ytr2, yval = train_test_split(Xtr_raw, ytr, test_size=0.2,
                                                      stratify=ytr, random_state=RNG)
    pre = preprocessor(cont, cat).fit(Xtr2_raw)
    dense = lambda Z: (Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)).astype("float32")
    Xtr2, Xval, Xte = dense(pre.transform(Xtr2_raw)), dense(pre.transform(Xval_raw)), dense(pre.transform(Xte_raw))

    model, n_ep = train_mlp(Xtr2, ytr2, Xval, yval)
    with torch.no_grad():
        p_nn = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy().ravel()
    auc_nn = roc_auc_score(yte, p_nn)

    # tree baselines on the SAME test split (their pipelines preprocess internally)
    out = {"condition": cond, "PyTorch_MLP": round(auc_nn, 3), "epochs": n_ep}
    for algo, label in [("randomforest", "RandomForest"), ("xgboost", "XGBoost")]:
        b = joblib.load(ROOT / "models" / "full_clinical" / cond / f"{algo}.pkl")
        out[label] = round(roc_auc_score(yte, b["pipeline"].predict_proba(Xte_raw)[:, 1]), 3)
    return out

def main():
    rows = [run("diabetes", load_diabetes(), DIABETES), run("heart", load_heart(), HEART)]
    res = pd.DataFrame(rows)[["condition", "PyTorch_MLP", "RandomForest", "XGBoost", "epochs"]]
    print(res.to_string(index=False))
    res.to_csv(ROOT / "reports" / "nn_comparison.csv", index=False)
    print("\n-> reports/nn_comparison.csv")

if __name__ == "__main__":
    main()
