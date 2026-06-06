# Full-Clinical vs Deployable Models — Group 3

**Question:** how much accuracy do we lose when we restrict the model to features the
*deployed* app can actually obtain — Group 1's face scan (resting `heart_rate`,
`blood_pressure`) + Group 4 / onboarding profile (BMI, age, sex, pregnancies, chest-pain
self-report, family history) — versus a model that uses every clinical feature in the dataset?

**Data (honest, leakage-free):**
- Diabetes — Pima (`data/diabetes.csv`), 768 rows; invalid zeros → median imputed.
- Heart — 4 real UCI sites pooled (`data/heart+disease/processed.*`), ~920 rows; `?` → median; target = `num>0`.
- **Deliberately NOT** the AIT-500-PROJECT `clean_heart_disease_all.csv` (15k): 12 mostly-duplicate datasets + SMOTE, 14.6% exact duplicate rows → train/test leakage, and no build script exists (unreproducible).

Repro: `.venv/bin/python src/train_compare.py` → `reports/full_vs_deployable.csv`.
5-fold CV on a 75/25 split; `class_weight=balanced`; threshold tuned for recall ≥ 0.80.

## Headline result (best of 14 models per group)

| Condition | Full-clinical AUC | Deployable AUC | Passes rubric (AUC>0.80)? |
|---|---|---|---|
| **Heart** | 0.915 (UCI) | **0.875** (UCI) | ✅ yes |
| **Diabetes** | 0.823 (Pima, with glucose) | **0.808** (CDC, glucose-free) | ✅ yes |

> The deployable diabetes model trains on **CDC/BRFSS** (no glucose), not Pima — because Pima's
> top feature (glucose) is unmeasurable by the app. Pima-minus-glucose only reached 0.771 (❌);
> CDC reaches 0.808 with features the app can supply. Full-clinical diabetes stays on Pima.

## Model sweep — 14 algorithms

LogReg · GaussianNB · KNN · SVM(RBF) · DecisionTree · RandomForest · ExtraTrees ·
AdaBoost · GradientBoosting · HistGradientBoosting · XGBoost · LightGBM · CatBoost · MLP.

Three findings:
- **Gradient boosting does NOT dominate at this scale.** On 768/920-row tabular data,
  LogReg / SVM / AdaBoost match or beat XGBoost/LightGBM/CatBoost. The "boosting wins on
  tabular" effect only appears on the 75k-row CDC set (there XGBoost tops at 0.829). The
  effect is **sample-size dependent** — a strong Task-5 discussion point.
- **No algorithm rescued deployable-diabetes on Pima** (all 14 landed 0.71–0.77). It was a
  *data* problem, not a model problem → so we switched the dataset (to glucose-free CDC), not the
  classifier. That switch is now implemented (see below).
- **Trust 5-fold CV over the single-split test AUC for heart** (~230-row test set is noisy;
  test AUC ≈ 0.91 sits above CV AUC ≈ 0.88). Quote the CV number.

## What this means

**Heart / hypertension — deployability is cheap (−0.037).**
Dropping the 7 clinical-test features (`chol, fbs, restecg, oldpeak, slope, ca, thal`) and
keeping only `age, sex, trestbps (BP), thalach (HR), cp, exang` costs almost nothing — the
deployable model still hits **0.875 AUC**, comfortably past the rubric's 0.80 bar. Reason:
blood pressure + heart rate + age carry most of the cardiovascular signal, and the dropped
tests are partly correlated with what we keep. **A deployable heart-risk model is viable.**

**Diabetes — Pima's glucose can't be replaced, so we switched datasets.** On Pima, removing
glucose (unmeasurable by a face scan, plus Insulin/SkinThickness) drops AUC to 0.77 and no
algorithm recovers it. Instead of shipping a weak model, we **train the deployable diabetes model
on the glucose-free CDC/BRFSS dataset** (BP + BMI + lifestyle indicators), which reaches **0.808
AUC** — clearing the bar. *Caveat:* it's mostly questionnaire-driven (the scan contributes the
`HighBP` flag), so it's honestly a "questionnaire + BP screen," not a scan-derived prediction.

## Recall (the false-negatives-matter clause)

We tune each model's threshold to meet **recall ≥ 0.80** on the at-risk class and report the
precision cost:

| | recall@0.5 → tuned | precision at tuned threshold |
|---|---|---|
| Heart deployable (LogReg)         | 0.811 → 0.803 | 0.810 |
| Diabetes deployable (CDC, CatBoost) | 0.786 → 0.804 | 0.712 |

Recall ≥ 0.80 is achievable on both, and the CDC diabetes model holds **~0.71 precision** at that
recall — far better than the abandoned Pima-minus-glucose model (~0.50). Still disclose to users
that a positive flag is a screening signal, not a diagnosis.

## Recommendations

1. **Ship both deployable models** — heart (0.875) and diabetes (CDC, 0.808) both clear the
   rubric and run on app-obtainable inputs.
2. **Disclose the diabetes model's nature** — questionnaire + BP screen, not glucose-grade.
3. **For true hypertension** (next step), train on `cardio.csv` / `framingham.prevalentHyp`, or
   threshold Group 1's BP directly — currently `hypertension_risk` is the UCI heart-disease model.
4. **Report both variants in the writeup.** The full-vs-deployable gap *is* the clinical-
   implications finding the rubric asks for — show it, don't hide it.

## Saved artifacts
- `models/full_clinical/{diabetes,heart}/<algo>.pkl` + `_meta.json` — diabetes = Pima (w/ glucose)
- `models/deployable/{diabetes,heart}/<algo>.pkl` + `_meta.json` — diabetes = CDC (glucose-free)
- each pickle = `{pipeline, features, model}` (sklearn Pipeline incl. impute+scale+encode)
