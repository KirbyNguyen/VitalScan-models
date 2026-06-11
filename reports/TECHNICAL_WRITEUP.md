# VitalScan — Group 3: Biomarker Risk Prediction
### Technical Writeup

**Scope.** Group 3 owns the risk-prediction stage: given biomarkers from Group 1's face scan
plus profile data, output diabetes and hypertension/heart-disease risk for the downstream
recommendation engine. This document covers our data-preprocessing decisions, model selection
rationale, the clinical implications of false negatives, and our explainability findings.

---

## 1. Data and preprocessing

**Datasets.** Two required public datasets plus several added for feature alignment, benchmarking,
and the deep-learning study (full details in `data/DATASETS.md`):

| Dataset | Rows | Target | Role in this project |
|---|---|---|---|
| Pima Diabetes | 768 | `Outcome` | **full-clinical** diabetes benchmark (with glucose) |
| UCI Heart Disease (4 sites pooled) | ~920 | `num>0` | heart model — both variants |
| CDC/BRFSS Diabetes Indicators | 253,680 | `Diabetes_binary` | **served** deployable diabetes (glucose-free) |
| iammustafatz 100k | 100,000 | `diabetes` | extra full-clinical benchmark (HbA1c) — *not served* |
| Framingham | 4,240 | `TenYearCHD` / `prevalentHyp` | reference: resting HR+BP, real hypertension label |
| Cardiovascular | 70,000 | `cardio` | reference / NN study |
| NHANES (assembled) | 15,547 | `diabetes` | NN study only — measured vitals |

**Preprocessing decisions (justified).**
- *Missing values.* Pima encodes missing physiology as `0` (e.g. BMI=0); we convert these to
  NaN and **median-impute**, chosen over MICE for transparency and because tree models are
  robust to it. UCI uses `?` → NaN → median. All imputation lives inside the model Pipeline so
  it is applied identically at train and inference time (no leakage).
- *Scaling.* Continuous features are StandardScaled; categoricals one-hot encoded
  (`handle_unknown="ignore"` so unseen categories don't break inference).
- *Heart pooling.* We pool the four real UCI sites (~920 rows) and explicitly **reject** the
  pre-existing `clean_heart_disease_all.csv` (15k): it merges 12 mostly-duplicate datasets and
  SMOTE-augments to 15k, contains 14.6% exact-duplicate rows (train/test leakage), and has no
  reproducible build script. Honesty over inflated numbers.

---

## 2. The central design decision: full-clinical vs deployable models

The features that make the best model are not the features the deployed app can collect. A face
scan yields resting heart rate and blood pressure; it cannot measure glucose, cholesterol, or
stress-test results. We therefore train **two models per condition**:

- **Full-clinical** — every dataset feature (accuracy benchmark / evaluation only).
- **Deployable** — only features obtainable from Group 1 (BP, HR) + Group 4 profile (BMI, age,
  sex, chest-pain self-report, family history).

This split is the project's key finding and frames everything below.

---

## 3. Model training, tuning, and selection

We trained **14 algorithms** per condition × variant (LogReg, GaussianNB, KNN, SVM-RBF,
DecisionTree, RandomForest, ExtraTrees, AdaBoost, GradientBoosting, HistGradientBoosting,
XGBoost, LightGBM, CatBoost, MLP) under 5-fold stratified cross-validation — 12 for the large
CDC deployable-diabetes set, where SVM/KNN don't scale and are skipped. RandomForest and
XGBoost were **hyperparameter-tuned** with `RandomizedSearchCV` (RF: n_estimators/max_depth/
min_samples_split; XGB: n_estimators/max_depth/learning_rate/subsample); tuned params are
recorded in each `_meta.json`. Class imbalance handled via `class_weight="balanced"` /
`scale_pos_weight`. Full table (ROC-AUC, Precision, Recall, F1) in `reports/full_vs_deployable.csv`.

**Best model per group (test ROC-AUC, F1, precision at recall ≥ 0.80):**

| Condition / variant | Best model | CV AUC | Test AUC | F1 | Precision @ recall 0.80 |
|---|---|---|---|---|---|
| Diabetes — full (Pima, with glucose) | Logistic Regression | 0.833 | 0.823 | 0.66 | 0.58 |
| Diabetes — deployable (CDC, glucose-free) | CatBoost | 0.815 | 0.808 | 0.75 | 0.71 |
| Heart — full | CatBoost | 0.873 | 0.915 | 0.85 | 0.90 |
| Heart — deployable | Logistic Regression | 0.859 | 0.875 | 0.81 | 0.81 |

**Selection rationale.** We select on **cross-validated** ROC-AUC, not the single-split test AUC,
because the heart test set (~230 rows) is noisy (test AUC sits above CV AUC). Two findings drove
selection: (1) on these small datasets, **simple models (LogReg/SVM) match or beat the gradient
boosters** — boosting only wins at the 75k-row CDC scale, so we don't default to XGBoost; (2)
hyperparameter tuning helped the boosters most on the deployable sets (e.g. XGBoost
deployable-diabetes 0.736 → 0.769). The deployed models are the *deployable*-variant winners.

**The diabetes problem and its fix (now implemented).** Deployable-diabetes on Pima reached only
**0.771** — below the 0.80 bar — because Pima's dominant feature is **Glucose**, which a face scan
cannot measure, and no algorithm rescued it (all 14 land 0.71–0.77 → a *data* problem, not a model
problem). We therefore **train the deployable diabetes model on the CDC/BRFSS dataset**, which has
no glucose column at all and predicts from BP + BMI + lifestyle indicators. Using 8 app-obtainable
features (`HighBP` derived from Group 1's BP; `BMI/Age/Sex` from profile; `Smoker/PhysActivity/
GenHlth/DiffWalk` from the Group 4 questionnaire), the served CatBoost model reaches **0.808 AUC,
F1 0.75, precision 0.71 at recall 0.80** — clearing the rubric bar and far better-calibrated than
the Pima-minus-glucose model (precision 0.51). The full-clinical diabetes model stays on Pima
(with glucose) as the accuracy benchmark. *Caveat:* most of CDC's signal is questionnaire-driven —
the scan contributes mainly the `HighBP` flag — so the diabetes output is honestly a
"questionnaire + BP" screen, not a scan-derived prediction.

**A third diabetes benchmark (context only).** For completeness we also benchmarked the
iammustafatz 100k dataset (`src/diabetes_hba1c_benchmark.py`), which reaches **0.98 AUC** — but
that is *too easy*: its top feature, HbA1c, is essentially the clinical definition of diabetes, so
the model is partly reading the label. We keep it as a labelled benchmark only; it is not served
and does not change the deployable story. So we have three diabetes views: deployable (CDC, 0.808,
*served*), full/Pima (glucose, 0.823), and full/iammustafatz (HbA1c, 0.980).

---

## 4. Clinical implications: why false negatives dominate

In a screening app a **false negative** (telling an at-risk user they're fine) is far more
dangerous than a false positive. We therefore optimize **recall on the at-risk class**, tuning
each model's decision threshold to achieve **recall ≥ 0.80** rather than using the default 0.5,
and report the precision cost. The trade-off is real: heart-deployable holds precision ~0.81 at
recall 0.80 (acceptable), but diabetes-deployable on Pima drops to ~0.51 — half its alarms are
false. That is defensible for screening (catching cases matters more than annoying the worried-
well) but **must be disclosed to the user**, and is another reason to prefer the CDC diabetes
model, where precision at recall 0.80 is ~0.70.

---

## 5. Explainability (SHAP)

We applied `shap.TreeExplainer` to the full-clinical models (`src/shap_analysis.py`), producing
summary, bar, and per-patient waterfall plots (3 highest-risk + 3 lowest-risk each) in
`reports/figures/`. Clinical interpretation in `reports/shap_interpretation.md`. Headline:

- **Diabetes** is driven by **Glucose** (dominant), then **BMI** and **Age** — physiologically
  correct, and a direct illustration of the deployment gap (the #1 feature is unmeasurable).
- **Heart disease** is driven by **asymptomatic chest-pain type**, **exercise-induced angina**,
  and **ST depression (oldpeak)** — two of which are stress-test findings, again unavailable to
  a passive scan.

SHAP thus does more than rank features: it makes the deployability gap *feature-specific* and
explains why the deployable models, while clinically valid, must shed their strongest predictors.

---

## 6. Serving and integration (API)

Models are joblib-serialized as full Pipelines and served via FastAPI (`src/api.py`):
`POST /predict/{condition}/{variant}` (with `?algo=` to select any of the 14 — or `torch_mlp`,
the PyTorch net served from a `.pt` weights+preprocessor bundle, see `src/train_torch.py`), and
`POST /predict/risk` consuming Group 1's biomarker contract and returning
`{diabetes_risk, hypertension_risk, confidence}` from the deployable models, plus a `_full`
section with the full-clinical model outputs (meaningful only when an optional `clinical` lab
block is supplied; otherwise imputed and flagged low-confidence). Missing inputs are imputed and
reported, with `confidence` downgraded when too much is missing.

**Two honesty caveats** (also surfaced in the API response): `hypertension_risk` is currently a
proxy from the UCI heart-disease model — a true hypertension target exists (`framingham.prevalentHyp`)
or can be read directly by thresholding Group 1's BP (ACC/AHA: ≥130/80); and the heart model's
`thalach` feature was trained on exercise-max HR but receives resting HR at inference.

---

## 7. Stretch — neural networks and a deep-learning study

**7a. Baseline comparison (`reports/nn_discussion.md`).** A PyTorch MLP (64→32, Adam, BCE, early
stopping) on the small clinical datasets ties the trees on heart (0.907 vs 0.909) and trails on
diabetes (0.808 vs 0.819) — the textbook "gradient boosting wins on small tabular data" result.

**7b. Does more data help? (`reports/nn_scaling.md`, NN-only — does not affect served models).**
We pushed further: trained the MLP, **TabNet** (a tabular-specialized deep net), and a reference
XGBoost on the large datasets, and plotted AUC vs training size on CDC (1k→200k rows). Findings:
- **More data brings the NN to parity.** On the large sets the MLP *matches* gradient boosting
  (both ≈0.826 on CDC at full size); the earlier "trees win" gap was a small-sample effect.
  **Sample size, not algorithm family, decides.**
- It reaches **parity, not dominance** — the two track within ±0.01 at scale.
- **TabNet underperformed a plain MLP** on every dataset — the fancy architecture didn't help.
- We also added **NHANES** (~15.5k adults, real measured vitals) for this study — the dataset
  closest to VitalScan's actual inputs.
- Honest note: deep learning would genuinely *beat* trees only on a different modality — Group 1's
  raw rPPG **waveforms** — not on these tabular features.

**7c. The PyTorch model is servable.** `src/train_torch.py` trains and saves a PyTorch MLP for all
four served groups (weights + fitted preprocessor in a `.pt` bundle); the API exposes it via
`?algo=torch_mlp`. It performs on par with the sklearn models (e.g. heart-deployable ≈0.87) but is
*not* the default — the sklearn/CatBoost winners are faster and as accurate, so `torch_mlp` is
opt-in for comparison rather than the production choice.

---

## 8. Summary of deliverables

Trained/serialized models (14 × 4 groups, best flagged); model comparison with ROC-AUC/Precision/
Recall/F1 and tuned hyperparameters; SHAP summary + 3+3 waterfalls + clinical interpretation;
`/predict/risk` + per-condition/variant REST endpoints; EDA/comparison/ROC/SHAP figures; the
stretch PyTorch NN comparison; an extra HbA1c benchmark; and a deep-learning scaling study
(MLP vs TabNet vs XGBoost across data sizes, incl. NHANES). All reproducible from `README.md`.
