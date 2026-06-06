# VitalScan – Group 3 Datasets

All paths relative to `data/`. The **contract features** we actually receive at inference
(from Group 1's face scan) are: `heart_rate`, `hrv_sdnn`, `stress_index`,
`blood_pressure.systolic`, `blood_pressure.diastolic`. `BMI` is available via Group 4.
Datasets are ranked below by how well their columns overlap that contract.

## Required (provided)

### `diabetes.csv` — Pima Indians Diabetes
- **Rows / cols:** 768 / 9 · **Target:** `Outcome` (0/1)
- **Cols:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
- **Contract overlap:** BloodPressure (diastolic), BMI. No heart rate / HRV / stress.
- **Caveat:** zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI are *missing*, not real → impute.

### `heart+disease/` — UCI Heart Disease (id=45)
- Four sites: `processed.cleveland.data` (303), `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`. **Pool all four ≈ 920 rows.**
- **Target:** `num` (0 = no disease, 1–4 = disease; binarize to >0).
- **Cols (13):** age, sex, cp, trestbps (resting BP), chol, fbs, restecg, **thalach (max heart rate)**, exang, oldpeak, slope, ca, thal
- **Contract overlap:** trestbps (BP), thalach (HR — note: *max* HR under stress, not resting).
- **Caveat:** Switzerland site has `chol=0` throughout (missing); `?` marks missing values.

## Added (verified & downloaded 2026-06-05) — under `extra/`

### `extra/framingham.csv` — Framingham Heart Study ⭐ best contract alignment
- **Rows / cols:** 4,240 / 16 · **Target:** `TenYearCHD` (10-yr coronary heart disease)
- **Cols:** male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, **prevalentHyp** (hypertension label), diabetes, totChol, **sysBP**, **diaBP**, **BMI**, **heartRate**, glucose, TenYearCHD
- **Why it matters:** the ONLY common dataset with `heartRate` **and** `sysBP`/`diaBP` together — trainable directly on our real input features. `prevalentHyp` is a usable hypertension target.
- **Caveats:** `NA` missing values present; was CR-only line endings → normalized to LF on download.
- **Source:** https://raw.githubusercontent.com/matackett/sta210/master/data/framingham.csv (subset of FHS; also on Kaggle `aasheesh200/framingham-heart-study-dataset`)

### `extra/cardio.csv` — Cardiovascular Disease (sulianova, 70k) ⭐ hypertension
- **Rows / cols:** 70,000 / 13 · **Delimiter: `;`** · **Target:** `cardio` (0/1)
- **Cols:** id, age (**in DAYS**), gender, height, weight, **ap_hi** (systolic), **ap_lo** (diastolic), cholesterol (1–3), gluc (1–3), smoke, alco, active, cardio
- **Why it matters:** large, balanced (~35k/35k), BP-driven target → strong for the **hypertension** model. Derive BMI from height/weight.
- **Caveats:** read with `sep=';'`; convert `age` days→years (`/365.25`); `ap_hi`/`ap_lo` have known data-entry outliers (e.g. negatives, values >10000) → clip to plausible ranges.
- **Source:** https://raw.githubusercontent.com/caravanuden/cardio/master/cardio_train.csv (orig: Kaggle `sulianova/cardiovascular-disease-dataset`)

### `extra/cdc_diabetes.csv` — CDC Diabetes Health Indicators (UCI id=891)
- **Rows / cols:** 253,680 / 23 · **Target:** `Diabetes_binary` (0/1)
- **Cols:** ID, Diabetes_binary, **HighBP**, HighChol, CholCheck, **BMI**, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
- **Why it matters:** **now the training set for the served deployable diabetes model** — it has
  no glucose column, so it predicts diabetes from features the app can actually obtain. Deployable
  model uses 8 of these: `HighBP` (from Group 1 BP), `BMI, Age, Sex` (profile), `Smoker,
  PhysActivity, GenHlth, DiffWalk` (Group 4 questionnaire) → CatBoost 0.808 AUC. Pima stays the
  full-clinical (with-glucose) benchmark.
- **Caveats:** `HighBP` is a 0/1 indicator, **not** a measured BP value — does not map cleanly to the contract's numeric BP. `Age` is bucketed (1–13), not years. No heart rate / HRV.
- **Source:** https://archive.ics.uci.edu/static/public/891/data.csv (DOI 10.24432/C53919)

### `extra/diabetes_100k.csv` — iammustafatz "Diabetes prediction dataset" (extra full-clinical benchmark)
- **Rows / cols:** 100,000 / 9 · **Target:** `diabetes` (0/1, ~8.5% positive)
- **Cols:** gender, age, hypertension, heart_disease, smoking_history, BMI, **HbA1c_level**, **blood_glucose_level**, diabetes
- **Why it's here:** a *second, independent* full-clinical diabetes benchmark (`src/diabetes_hba1c_benchmark.py` → `models/full_clinical/diabetes_hba1c/`). **Benchmark only — NOT served by the API.**
- **Caveats:** **3,854 exact duplicate rows** (deduped on load); `smoking_history` has a `'No Info'` bucket; `gender` includes `'Other'`. Mislabeled as "CDC" in some places — it is **not** the CDC/BRFSS set; it's a Kaggle dataset and partly synthetic.
- **Source:** https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset (mirror: raw.githubusercontent.com/mezkymy/diabetes-prediction)

#### iammustafatz 100k vs Pima — why we keep them separate
| | Pima (`diabetes.csv`) | iammustafatz (`diabetes_100k.csv`) |
|---|---|---|
| Rows | 768 | 100,000 (3,854 dups) |
| Target prevalence | ~35% | ~8.5% |
| Key features | Glucose, Insulin, SkinThickness, Pregnancies, DPF, BP, BMI, Age | **HbA1c**, blood glucose, BMI, age, hypertension, heart_disease, smoking, gender |
| Shared features | only **age, BMI, glucose** | — |
| Full-model AUC | **0.823** | **0.980** (XGBoost) |

- **Can't be merged row-wise** — they overlap on only 3 columns, so use as *parallel* benchmarks, not a combined dataset.
- **The 0.98 is "too easy".** `HbA1c ≥ 6.5%` is essentially the clinical *definition* of diabetes, so a model reading HbA1c is partly reading the label. Pima's 0.823 (driven by a single glucose snapshot, no HbA1c) is the harder, more realistic benchmark.
- **Neither helps the deployable model** — both lean on lab values (glucose/HbA1c) a face scan can't measure. The deployable diabetes model stays on the glucose-free CDC/BRFSS set.

### `extra/nhanes_diabetes.csv` — NHANES (assembled, NN study only)
- **Rows / cols:** 15,547 / 6 · **Target:** `diabetes` (0/1, ~16.2% positive)
- **Features:** age, sex, bmi, systolic, diastolic (glucose-free vitals + demographics)
- **Built by** `src/fetch_nhanes.py`: pools 3 CDC NHANES cycles (2013-14, 2015-16, 2017-18),
  merges DEMO+BMX+BPX+GHB+DIQ on `SEQN`. Label = HbA1c ≥ 6.5 **or** doctor-diagnosed; adults 18+.
- **Why it's here:** a real, *measured-vitals* dataset for the **NN study** (`reports/nn_scaling.md`).
  **Not used by the served models or API.** It best matches VitalScan's actual inputs (BP, age).
- **Source:** CDC NHANES, https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/ (raw XPT cached in `extra/nhanes_raw/`)

## Pointer (not downloaded)
- **MCD-rPPG** (600 subjects + biomarkers, HF `kyegorov/mcd_rppg`) — listed in Group 1's brief. The only source that natively pairs rPPG-derived vitals with health data; best candidate for closing the train↔inference gap. Larger/auth-gated — pull if we pursue the alignment work.

## Open gap
No public dataset contains `hrv_sdnn` or `stress_index` paired with diabetes/hypertension
labels. Options: (a) train only on the overlapping features (HR + BP), or (b) treat HRV/stress
as auxiliary inputs handled in the Task 4 feature-mapping layer. Decide before Task 2.
