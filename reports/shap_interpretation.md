# SHAP Explainability — Clinical Interpretation (Task 3)

SHAP (SHapley Additive exPlanations) attributes each prediction to its input features: a
positive SHAP value pushes the predicted risk **up**, negative pushes it **down**. Computed
with `shap.TreeExplainer` on the full-clinical models. Plots in `reports/figures/`
(`shap_summary_*`, `shap_bar_*`, and six `shap_waterfall_*` per condition: 3 highest-risk,
3 lowest-risk patients). Mean |SHAP| rankings are in `reports/shap_top_features.json`.

## Diabetes — top 3 features (clinical reading)

1. **Glucose** (mean |SHAP| 0.79 — dominant). Plasma glucose concentration is the *defining*
   biomarker of diabetes, so its dominance is expected and reassuring: the model keys on the
   physiologically correct signal. High glucose drives risk sharply up in every high-risk
   waterfall. **Deployment consequence:** this is also the feature a face scan cannot measure,
   which is exactly why the deployable Pima model fails and we pivot to the glucose-free CDC
   dataset (see `full_vs_deployable.md`).
2. **BMI** (0.52). Body-mass index is the leading *modifiable* risk factor for type-2 diabetes —
   higher adiposity drives insulin resistance. The model raises risk steadily with BMI, and BMI
   *is* obtainable in deployment (Group 4 profile), so it carries much of the deployable signal.
3. **Age** (0.25). Risk accumulates with age as β-cell function declines and insulin resistance
   rises; SHAP shows older patients pushed toward higher risk. Also deployment-available.

(Runners-up: Pregnancies 0.18 — parity/gestational-diabetes history; DiabetesPedigreeFunction
0.16 — family-history genetic score.)

## Heart disease — top 3 features (clinical reading)

1. **Chest-pain type = asymptomatic** (`cp_4.0`, 0.59). The strongest driver. In the Cleveland
   cohort, *asymptomatic* presentation (type 4) is paradoxically the biggest red flag — these
   patients have angiographic disease without classic chest pain, so the model correctly treats
   "no typical pain" as high-risk rather than reassuring.
2. **Exercise-induced angina** (`exang_0.0`, 0.42). When `exang=0` (no angina on exertion) the
   SHAP value is protective (risk down); patients who *do* get exertional angina are pushed up.
   Clinically sound — exertional angina signals inducible ischemia.
3. **ST depression / `oldpeak`** (0.41). Exercise-induced ST-segment depression is a direct ECG
   marker of myocardial ischemia; larger `oldpeak` → higher risk in every high-risk waterfall.

(Runners-up: cholesterol `chol` 0.40; number of fluoroscopy-visible vessels `ca` 0.26 — `ca=0`
is protective; age 0.26.)

## Why this matters for VitalScan

Two of heart disease's top-3 (`oldpeak`, and `cp`/`exang` as exercise-test findings) come from a
**clinical stress test**, not a passive face scan — mirroring the diabetes/glucose problem. SHAP
makes the deployment gap concrete and feature-specific: the models are clinically valid, but
their most informative features are precisely the ones the deployed app cannot capture. This is
the core honest finding for the false-negative / clinical-implications discussion.
