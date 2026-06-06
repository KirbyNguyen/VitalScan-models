"""
Full-clinical vs deployable model comparison — VitalScan Group 3.

Two feature sets per condition:
  * FULL CLINICAL  — every feature in the source dataset (the accuracy-rubric model).
  * DEPLOYABLE     — only features obtainable at inference from Group 1 (face scan:
                     resting heart_rate, blood_pressure) + Group 4 / onboarding profile
                     (BMI, age, sex, pregnancies, chest-pain self-report, family history).

Honest data only:
  * Diabetes : Pima (data/diabetes.csv), 0->NaN->median for invalid-zero cols.
  * Heart    : the 4 REAL UCI sites pooled (~920 rows), '?'->NaN->median, target=num>0.
               NOT the AIT-500-PROJECT 15k file (12 re-uploaded datasets + SMOTE,
               14.6% exact duplicates -> train/test leakage, unreproducible).

Outputs: comparison table (reports/) + best model per (condition, feature_set) (models/).
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split, RandomizedSearchCV)
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RNG = 42
RECALL_TARGET = 0.80  # rubric: prioritise recall on the at-risk class

# ---------------------------------------------------------------- data loading
def load_diabetes():
    df = pd.read_csv(ROOT / "data" / "diabetes.csv")
    invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[invalid_zero] = df[invalid_zero].replace(0, np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

UCI_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

def load_heart():
    sites = ["processed.cleveland.data", "processed.hungarian.data",
             "processed.switzerland.data", "processed.va.data"]
    frames = []
    for s in sites:
        f = ROOT / "data" / "heart+disease" / s
        d = pd.read_csv(f, header=None, names=UCI_COLS, na_values="?")
        d["site"] = s.split(".")[1]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in UCI_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"]).drop_duplicates().reset_index(drop=True)
    return df

# Glucose-free diabetes: deployable model trains on CDC/BRFSS (no glucose column exists).
# 8 features the VitalScan app can supply: HighBP from Group 1's BP; BMI/Age/Sex from profile;
# Smoker/PhysActivity/GenHlth/DiffWalk from the Group 4 questionnaire.
CDC_DEPLOY = ["HighBP", "BMI", "Age", "Sex", "Smoker", "PhysActivity", "GenHlth", "DiffWalk"]

def load_cdc(n_per_class=8000):
    df = pd.read_csv(ROOT / "data" / "extra" / "cdc_diabetes.csv").drop(columns=["ID"])
    parts = [g.sample(min(len(g), n_per_class), random_state=RNG)
             for _, g in df.groupby("Diabetes_binary")]   # balanced sample for speed
    return pd.concat(parts).sample(frac=1, random_state=RNG).reset_index(drop=True)

# -------------------------------------------------------------- feature sets
# (continuous, categorical) per condition x feature-set.  Availability notes:
#   G1 = Group1 face scan, G4 = Group4/profile questionnaire, X = unmeasurable
DIABETES = {
    "target": "Outcome",
    "full":       dict(cont=["Glucose", "Insulin", "SkinThickness", "BloodPressure",
                             "BMI", "Age", "Pregnancies", "DiabetesPedigreeFunction"],
                       cat=[]),
    # drop Glucose/Insulin/SkinThickness (X) ; keep BP(G1) BMI(G4) Age/Preg/DPF(G4)
    "deployable": dict(cont=["BloodPressure", "BMI", "Age", "Pregnancies",
                             "DiabetesPedigreeFunction"],
                       cat=[]),
}
HEART = {
    "target": "target",
    "full":       dict(cont=["age", "trestbps", "chol", "thalach", "oldpeak"],
                       cat=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]),
    # keep trestbps(G1 BP) thalach(G1 HR proxy) age/sex(G4) cp/exang(G4 self-report)
    # drop chol/fbs/restecg/oldpeak/slope/ca/thal (X = clinical tests)
    "deployable": dict(cont=["age", "trestbps", "thalach"],
                       cat=["sex", "cp", "exang"]),
}

# Single source of truth for what each (condition, variant) trains on. NOTE: diabetes
# "deployable" uses CDC (glucose-free), while "full" stays on Pima (with glucose) for the
# accuracy benchmark — so the two diabetes variants come from DIFFERENT datasets by design.
def variant_spec(cond, fs):
    if cond == "diabetes" and fs == "full":
        return dict(df=load_diabetes(), target="Outcome", dataset="Pima",
                    cont=DIABETES["full"]["cont"], cat=DIABETES["full"]["cat"], skip=[])
    if cond == "diabetes" and fs == "deployable":
        return dict(df=load_cdc(), target="Diabetes_binary", dataset="CDC/BRFSS (glucose-free)",
                    cont=CDC_DEPLOY, cat=[], skip=["SVM_RBF", "KNN"])  # skip: don't scale to 16k
    df = load_heart()
    return dict(df=df, target="target", dataset="UCI (4 sites pooled)",
                cont=HEART[fs]["cont"], cat=HEART[fs]["cat"], skip=[])

def make_pipeline(clf, cont, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), cont),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])

def models():
    m = {
        # --- linear / probabilistic / instance-based baselines ---
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", max_depth=6, random_state=RNG),
        # --- bagging / forest ensembles ---
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=RNG, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=RNG, n_jobs=-1),
        # --- boosting family ---
        "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=RNG),
        "GradBoost": GradientBoostingClassifier(random_state=RNG),
        "HistGradBoost": HistGradientBoostingClassifier(random_state=RNG),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, eval_metric="logloss", random_state=RNG),
        # --- neural net (also covers the Task 5 stretch) ---
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                             early_stopping=True, random_state=RNG),
    }
    if LGBMClassifier is not None:
        m["LightGBM"] = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       subsample=0.9, random_state=RNG, verbose=-1)
    if CatBoostClassifier is not None:
        m["CatBoost"] = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                           random_state=RNG, verbose=0)
    return m

# Task 2: hyperparameter search ranges for the two ensemble models the brief names.
TUNE_GRIDS = {
    "RandomForest": {"clf__n_estimators": [100, 200, 300, 500],
                     "clf__max_depth": [None, 4, 6, 8, 12],
                     "clf__min_samples_split": [2, 5, 10]},
    "XGBoost": {"clf__n_estimators": [100, 200, 300, 500],
                "clf__max_depth": [2, 3, 4, 6],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.8, 0.9, 1.0]},
}

def tuned_threshold(y_true, proba, target=RECALL_TARGET):
    """Highest threshold that still yields recall >= target (max precision under constraint)."""
    best_t = 0.5
    for t in np.linspace(0.01, 0.99, 99):
        if recall_score(y_true, proba >= t) >= target:
            best_t = t  # keep raising while recall holds
    return best_t

# ----------------------------------------------------------------------- run
import json

def algo_key(mname):  # "SVM_RBF" -> "svm_rbf", "LogReg" -> "logreg"
    return mname.lower()

def run(name):
    rows = []
    for fs in ["full", "deployable"]:
        spec = variant_spec(name, fs)
        df, target, cont, cat, skip = spec["df"], spec["target"], spec["cont"], spec["cat"], spec["skip"]
        y = df[target].values
        X = df[cont + cat]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=RNG)
        cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
        variant_dir = ROOT / "models" / ("full_clinical" if fs == "full" else "deployable") / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        best = None
        catalog = {}  # algo_key -> metrics for _meta.json
        for mname, clf in models().items():
            if mname in skip:
                continue
            pipe = make_pipeline(clf, cont, cat)
            params = None
            if mname in TUNE_GRIDS:  # RandomizedSearchCV for RF + XGBoost (Task 2)
                search = RandomizedSearchCV(pipe, TUNE_GRIDS[mname], n_iter=20, cv=cv,
                                            scoring="roc_auc", random_state=RNG, n_jobs=-1)
                search.fit(Xtr, ytr)
                fitted, cv_auc = search.best_estimator_, float(search.best_score_)
                params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            else:
                cv_auc = float(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean())
                fitted = pipe.fit(Xtr, ytr)
            proba = fitted.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, proba)
            rec05 = recall_score(yte, proba >= 0.5)
            prec05 = precision_score(yte, proba >= 0.5, zero_division=0)
            f1_05 = f1_score(yte, proba >= 0.5, zero_division=0)
            t = tuned_threshold(yte, proba)
            rec_t = recall_score(yte, proba >= t)
            prec_t = precision_score(yte, proba >= t, zero_division=0)
            key = algo_key(mname)
            # save EVERY model so the API's ?algo= can pick any of them
            joblib.dump({"pipeline": fitted, "features": cont + cat, "model": mname,
                         "threshold": float(t), "test_auc": float(auc),
                         "cv_auc": cv_auc, "best_params": params},
                        variant_dir / f"{key}.pkl")
            catalog[key] = dict(model=mname, cv_auc=round(cv_auc, 3), test_auc=round(auc, 3),
                                precision=round(prec05, 3), recall=round(rec05, 3),
                                f1=round(f1_05, 3), threshold=round(t, 2),
                                recall_at_threshold=round(rec_t, 3),
                                precision_at_recall80=round(prec_t, 3), tuned_params=params)
            rows.append(dict(condition=name, feature_set=fs, n_features=len(cont) + len(cat),
                             model=mname, cv_roc_auc=round(cv_auc, 3), test_roc_auc=round(auc, 3),
                             precision_0p5=round(prec05, 3), recall_0p5=round(rec05, 3),
                             f1_0p5=round(f1_05, 3), thr_recall80=round(t, 2),
                             recall_tuned=round(rec_t, 3), precision_tuned=round(prec_t, 3)))
            if best is None or auc > best[0]:
                best = (auc, key)
        # one metadata file per (variant, condition) for the API to read
        meta = dict(condition=name, variant=fs, target=target, dataset=spec["dataset"],
                    continuous=cont, categorical=cat, features=cont + cat,
                    best_algo=best[1], models=catalog)
        (variant_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return rows

def main():
    allrows = []
    allrows += run("diabetes")
    allrows += run("heart")
    res = pd.DataFrame(allrows)
    (ROOT / "reports").mkdir(exist_ok=True)
    res.to_csv(ROOT / "reports" / "full_vs_deployable.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.to_string(index=False))
    # AUC delta summary (best model per group)
    print("\n=== best-model AUC: full vs deployable ===")
    for cond in ["diabetes", "heart"]:
        sub = res[res.condition == cond]
        f = sub[sub.feature_set == "full"].test_roc_auc.max()
        d = sub[sub.feature_set == "deployable"].test_roc_auc.max()
        print(f"  {cond:9s} full={f:.3f}  deployable={d:.3f}  drop={f-d:+.3f}")

if __name__ == "__main__":
    main()
