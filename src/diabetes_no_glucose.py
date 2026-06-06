"""
Can we predict diabetes WITHOUT glucose? Test on CDC/BRFSS Diabetes Health Indicators
(data/extra/cdc_diabetes.csv) — a survey dataset with NO glucose feature.

Compare two feature sets:
  ALL          — every indicator (incl. lab/history features like HighChol, Stroke).
  DEPLOYABLE   — only what VitalScan can obtain:
                 HighBP  <- derived from Group 1's blood_pressure
                 BMI     <- Group 4 / profile
                 Age,Sex <- profile
                 Smoker,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,
                 GenHlth,MentHlth,PhysHlth,DiffWalk  <- Group 4 questionnaire
"""
from pathlib import Path
import warnings, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RNG = 42

df = pd.read_csv(ROOT / "data" / "extra" / "cdc_diabetes.csv").drop(columns=["ID"])
# speed: stratified 80k sample (full 253k gives ~same AUC, just slower)
parts = [g.sample(min(len(g), 40000), random_state=RNG)
         for _, g in df.groupby("Diabetes_binary")]
df = pd.concat(parts).sample(frac=1, random_state=RNG).reset_index(drop=True)

assert "Glucose" not in df.columns and "glucose" not in [c.lower() for c in df.columns]
y = df["Diabetes_binary"].astype(int).values
X_all = df.drop(columns=["Diabetes_binary"])

DEPLOYABLE = ["HighBP", "BMI", "Age", "Sex", "Smoker", "PhysActivity",
              "Fruits", "Veggies", "HvyAlcoholConsump",
              "GenHlth", "MentHlth", "PhysHlth", "DiffWalk"]

FEATURE_SETS = {"all": list(X_all.columns), "deployable": DEPLOYABLE}

def models():
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                               random_state=RNG, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, eval_metric="logloss",
                                 scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                                 random_state=RNG),
    }

def tuned_thr(y_true, p, target=0.80):
    t = 0.5
    for c in np.linspace(0.01, 0.99, 99):
        if recall_score(y_true, p >= c) >= target:
            t = c
    return t

rows = []
for fs, cols in FEATURE_SETS.items():
    X = X_all[cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    for mname, clf in models().items():
        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        t = tuned_thr(yte, p)
        rows.append(dict(feature_set=fs, n_features=len(cols), model=mname,
                         cv_roc_auc=round(cv_auc, 3), test_roc_auc=round(roc_auc_score(yte, p), 3),
                         recall_0p5=round(recall_score(yte, p >= .5), 3),
                         precision_0p5=round(precision_score(yte, p >= .5, zero_division=0), 3),
                         thr_recall80=round(t, 2),
                         precision_at_recall80=round(precision_score(yte, p >= t, zero_division=0), 3)))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 20)
print(f"CDC/BRFSS sample: {len(df):,} rows, prevalence={y.mean():.1%} diabetic, NO glucose feature\n")
print(res.to_string(index=False))
print("\nbest AUC  all:", res[res.feature_set=='all'].test_roc_auc.max(),
      " deployable:", res[res.feature_set=='deployable'].test_roc_auc.max())
res.to_csv(ROOT / "reports" / "diabetes_no_glucose.csv", index=False)
