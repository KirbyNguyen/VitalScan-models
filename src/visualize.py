"""
Visualizations for VitalScan Group 3 — covers Task 1 (EDA) and Task 2 (model comparison).
Outputs PNGs to reports/figures/.

Run:  .venv/bin/python src/visualize.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: write files, don't open windows
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from train_compare import load_diabetes, load_heart, variant_spec  # reuse loaders + variant source of truth

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = 42
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. EDA
def eda(name, df, target):
    num = df.select_dtypes("number").drop(columns=[c for c in [target] if c in df], errors="ignore")
    # correlation heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.select_dtypes("number").corr(), annot=False, cmap="coolwarm", center=0)
    plt.title(f"{name}: feature correlation")
    plt.tight_layout(); plt.savefig(FIG / f"eda_corr_{name}.png", dpi=120); plt.close()
    # feature distributions split by outcome
    cols = list(num.columns)[:9]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10)); axes = axes.ravel()
    for i, col in enumerate(cols):
        for cls, sub in df.groupby(target):
            axes[i].hist(sub[col].dropna(), bins=25, alpha=0.55, label=f"{target}={cls}")
        axes[i].set_title(col); axes[i].legend(fontsize=7)
    for j in range(len(cols), len(axes)): axes[j].axis("off")
    fig.suptitle(f"{name}: feature distributions by outcome", y=1.02)
    plt.tight_layout(); plt.savefig(FIG / f"eda_dist_{name}.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"  EDA written for {name}")

# ---------------------------------------------------------------- 2. model comparison
def comparison_bars():
    res = pd.read_csv(ROOT / "reports" / "full_vs_deployable.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, cond in zip(axes, ["diabetes", "heart"]):
        sub = res[res.condition == cond]
        piv = sub.pivot(index="model", columns="feature_set", values="test_roc_auc").sort_values("full")
        piv.plot(kind="barh", ax=ax)
        ax.axvline(0.80, color="red", ls="--", lw=1, label="rubric 0.80")
        ax.set_title(f"{cond}: ROC-AUC by model"); ax.set_xlabel("test ROC-AUC"); ax.set_xlim(0.6, 0.95)
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "model_comparison_auc.png", dpi=120); plt.close()
    print("  model_comparison_auc.png written")

# ---------------------------------------------------------------- 3. ROC curves (best models)
def roc_curves():
    import json
    plt.figure(figsize=(8, 7))
    for cond in ["diabetes", "heart"]:
        for fs, vdir in [("full", "full_clinical"), ("deployable", "deployable")]:
            spec = variant_spec(cond, fs)        # correct dataset/features per variant (CDC for diabetes/deployable)
            df, target = spec["df"], spec["target"]
            X, y = df[spec["cont"] + spec["cat"]], df[target].values
            _, Xte, _, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
            meta = json.loads((ROOT / "models" / vdir / cond / "_meta.json").read_text())
            bundle = joblib.load(ROOT / "models" / vdir / cond / f"{meta['best_algo']}.pkl")
            p = bundle["pipeline"].predict_proba(Xte)[:, 1]
            fpr, tpr, _ = roc_curve(yte, p)
            plt.plot(fpr, tpr, label=f"{cond}/{fs} ({meta['best_algo']}, AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
    plt.title("ROC curves — best model per group"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "roc_curves.png", dpi=120); plt.close()
    print("  roc_curves.png written")

def main():
    print("EDA:"); eda("diabetes", load_diabetes(), "Outcome"); eda("heart", load_heart(), "target")
    print("Comparison:"); comparison_bars()
    print("ROC:"); roc_curves()
    print(f"\nAll figures -> {FIG}")

if __name__ == "__main__":
    main()
