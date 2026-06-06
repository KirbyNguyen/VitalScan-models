"""
Extra full-clinical diabetes benchmark on the iammustafatz "Diabetes prediction dataset"
(data/extra/diabetes_100k.csv). This is a SEPARATE benchmark, NOT merged with Pima — the two
datasets share only age/BMI/glucose, so they can't be row-combined (see DATASETS.md note).

It's benchmark-only (NOT served by the API): its top feature, HbA1c, is near-diagnostic and
unmeasurable by a face scan, so this informs the writeup but does not change the deployable model.

Run:  .venv/bin/python src/diabetes_hba1c_benchmark.py  ->  models/full_clinical/diabetes_hba1c/
"""
from pathlib import Path
import warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

from train_compare import (models, TUNE_GRIDS, make_pipeline, tuned_threshold, algo_key,
                           RandomizedSearchCV, RNG)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "full_clinical" / "diabetes_hba1c"
SKIP = ["SVM_RBF", "KNN"]            # don't scale to a large sample
CONT = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease"]
CAT = ["gender", "smoking_history"]

def load():
    df = pd.read_csv(ROOT / "data" / "extra" / "diabetes_100k.csv")
    df = df.drop_duplicates().reset_index(drop=True)          # 3,854 exact dups removed
    return df.sample(n=min(len(df), 20000), random_state=RNG).reset_index(drop=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    y = df["diabetes"].values
    X = df[CONT + CAT]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    rows, catalog, best = [], {}, None
    for mname, clf in models().items():
        if mname in SKIP:
            continue
        pipe = make_pipeline(clf, CONT, CAT)
        params = None
        if mname in TUNE_GRIDS:
            s = RandomizedSearchCV(pipe, TUNE_GRIDS[mname], n_iter=20, cv=cv,
                                   scoring="roc_auc", random_state=RNG, n_jobs=-1).fit(Xtr, ytr)
            fitted, cv_auc = s.best_estimator_, float(s.best_score_)
            params = {k.replace("clf__", ""): v for k, v in s.best_params_.items()}
        else:
            cv_auc = float(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean())
            fitted = pipe.fit(Xtr, ytr)
        p = fitted.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        t = tuned_threshold(yte, p)
        rec05, prec05, f105 = (recall_score(yte, p >= .5), precision_score(yte, p >= .5, zero_division=0),
                               f1_score(yte, p >= .5, zero_division=0))
        key = algo_key(mname)
        joblib.dump({"pipeline": fitted, "features": CONT + CAT, "model": mname,
                     "threshold": float(t), "test_auc": float(auc), "cv_auc": cv_auc,
                     "best_params": params}, OUT / f"{key}.pkl")
        catalog[key] = dict(model=mname, cv_auc=round(cv_auc, 3), test_auc=round(auc, 3),
                            precision=round(prec05, 3), recall=round(rec05, 3), f1=round(f105, 3),
                            threshold=round(t, 2), recall_at_threshold=round(recall_score(yte, p >= t), 3),
                            precision_at_recall80=round(precision_score(yte, p >= t, zero_division=0), 3),
                            tuned_params=params)
        rows.append((mname, round(cv_auc, 3), round(auc, 3), round(f105, 3)))
        if best is None or auc > best[0]:
            best = (auc, key)
    meta = dict(condition="diabetes_hba1c", variant="full", target="diabetes",
                dataset="iammustafatz 100k (HbA1c+glucose)", note="benchmark only; not API-served",
                continuous=CONT, categorical=CAT, features=CONT + CAT,
                best_algo=best[1], models=catalog)
    (OUT / "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"sample: {len(df):,} rows ({y.mean()*100:.1f}% diabetic), {len(catalog)} models")
    print(f"{'model':14s} cv_auc test_auc  f1")
    for m, c, a, f in sorted(rows, key=lambda r: -r[2]):
        print(f"  {m:14s} {c:.3f}  {a:.3f}  {f:.3f}")
    print(f"\nbest: {best[1]}  AUC={best[0]:.3f}   (Pima full benchmark = 0.823)")

if __name__ == "__main__":
    main()
