"""
SHAP explainability — VitalScan Group 3, Task 3 (core, 20% of grade).

Explains WHY the model predicts risk, using the full-clinical XGBoost model per condition
(tree model -> fast shap.TreeExplainer). Produces:
  * summary (beeswarm) plot  — which features drive predictions overall
  * bar plot                 — mean |SHAP| feature importance
  * waterfall plots          — one high-risk and one low-risk individual, feature-by-feature

Outputs PNGs to reports/figures/. Run:  .venv/bin/python src/shap_analysis.py
"""
from pathlib import Path
import warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, shap
from sklearn.model_selection import train_test_split

from train_compare import load_diabetes, load_heart, DIABETES, HEART

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = 42

def explain(cond, df, spec, algo="xgboost"):
    cont, cat = spec["full"]["cont"], spec["full"]["cat"]
    X, y = df[cont + cat], df[spec["target"]].values
    _, Xte, _, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)

    bundle = joblib.load(ROOT / "models" / "full_clinical" / cond / f"{algo}.pkl")
    pipe = bundle["pipeline"]
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]

    # transform inputs to the encoded space the classifier actually sees
    Xt = pre.transform(Xte)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]  # strip num__/cat__
    Xt_df = pd.DataFrame(Xt, columns=names)

    explainer = shap.TreeExplainer(clf)
    sv = explainer(Xt_df)

    # 1) beeswarm summary
    plt.figure()
    shap.summary_plot(sv, Xt_df, show=False, max_display=12)
    plt.title(f"{cond}: SHAP summary (full-clinical {algo})")
    plt.tight_layout(); plt.savefig(FIG / f"shap_summary_{cond}.png", dpi=120, bbox_inches="tight"); plt.close()

    # 2) mean|SHAP| bar
    plt.figure()
    shap.plots.bar(sv, show=False, max_display=12)
    plt.title(f"{cond}: SHAP feature importance")
    plt.tight_layout(); plt.savefig(FIG / f"shap_bar_{cond}.png", dpi=120, bbox_inches="tight"); plt.close()

    # 3) waterfalls for the 3 highest-risk and 3 lowest-risk patients (Task 3 spec)
    proba = clf.predict_proba(Xt)[:, 1]
    order = np.argsort(proba)
    for tag, idxs in {"highrisk": order[-3:][::-1], "lowrisk": order[:3]}.items():
        for k, idx in enumerate(idxs, 1):
            plt.figure()
            shap.plots.waterfall(sv[int(idx)], show=False, max_display=10)
            plt.title(f"{cond}: {tag} #{k} (predicted risk={proba[int(idx)]:.2f})")
            plt.tight_layout()
            plt.savefig(FIG / f"shap_waterfall_{cond}_{tag}_{k}.png", dpi=120, bbox_inches="tight")
            plt.close()

    top = pd.Series(np.abs(sv.values).mean(0), index=names).sort_values(ascending=False).head(8)
    print(f"  {cond}: top features -> {', '.join(top.index[:5])}")
    return {k: round(float(v), 4) for k, v in top.items()}

def main():
    tops = {"diabetes": explain("diabetes", load_diabetes(), DIABETES),
            "heart": explain("heart", load_heart(), HEART)}
    (ROOT / "reports" / "shap_top_features.json").write_text(json.dumps(tops, indent=2))
    print(f"\nSHAP figures -> {FIG}\ntop-features -> reports/shap_top_features.json")

if __name__ == "__main__":
    main()
