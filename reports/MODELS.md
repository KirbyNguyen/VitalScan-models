# Model Reference — how each model works, its parameters, and its data

One place that documents every model in the project: what it is, the exact hyperparameters we
configured, and which dataset/features it trains on. Source of truth: `src/train_compare.py`
(`models()`, `TUNE_GRIDS`, `variant_spec()`), `src/torch_nn.py`, `src/train_torch.py`. The
*actual selected* tuned parameters and scores for any group live in its
`models/<variant>/<condition>/_meta.json`.

---

## 1. Shared preprocessing & training protocol (applies to all sklearn/boosting models)

Every model is wrapped in a scikit-learn **Pipeline** so cleaning is learned on train data only
(no leakage) and reapplied identically at inference:

| Step | Numeric features | Categorical features |
|---|---|---|
| Impute missing | median | most-frequent |
| Encode/scale | `StandardScaler` (mean 0, sd 1) | `OneHotEncoder(handle_unknown="ignore")` |

- **Split:** 75% train / 25% test, stratified, `random_state=42`.
- **Validation:** 5-fold stratified cross-validation (the reported `cv_auc`).
- **Class imbalance:** `class_weight="balanced"` where supported; `scale_pos_weight` for XGBoost;
  `pos_weight` for the PyTorch net.
- **Decision threshold:** tuned per model so **recall ≥ 0.80** on the at-risk class (then we
  report the precision cost). The chosen threshold is saved with each model.

---

## 2. Datasets & features per served group

| Served group | Dataset | Rows | Target | Features (count) |
|---|---|---|---|---|
| `diabetes / full` | **Pima** | 768 | `Outcome` | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age (8) |
| `diabetes / deployable` | **CDC/BRFSS** (balanced 16k sample) | 253,680 | `Diabetes_binary` | HighBP, BMI, Age, Sex, Smoker, PhysActivity, GenHlth, DiffWalk (8) |
| `heart / full` | **UCI Heart** (4 sites) | ~920 | `num>0` | age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal (13) |
| `heart / deployable` | **UCI Heart** (4 sites) | ~920 | `num>0` | age, sex, cp, trestbps, thalach, exang (6) |
| `diabetes_hba1c` (benchmark, not served) | **iammustafatz 100k** | 100,000 | `diabetes` | gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level (8) |

Full dataset provenance, caveats, and the full-vs-deployable rationale are in `data/DATASETS.md`.

---

## 3. The model roster — how each works + configured hyperparameters

| Model | Family | How it works (one line) | Configured hyperparameters | Tuned? |
|---|---|---|---|---|
| **LogReg** | linear | weighted linear combination → sigmoid; the interpretable baseline | `max_iter=1000, class_weight=balanced` | no |
| **GaussianNB** | probabilistic | Bayes' rule assuming each feature is independent & Gaussian | defaults | no |
| **KNN** | instance-based | vote of the 15 nearest training patients | `n_neighbors=15` | no |
| **SVM_RBF** | kernel | maximum-margin boundary in an RBF-kernel space | `kernel=rbf, probability=True, class_weight=balanced` | no |
| **DecisionTree** | tree | one flowchart of yes/no feature splits | `max_depth=6, class_weight=balanced` | no |
| **RandomForest** | bagging | average of many decorrelated trees on bootstrap samples | `n_estimators=300, class_weight=balanced` (base) | **yes** |
| **ExtraTrees** | bagging | like RF but random split thresholds (more variance reduction) | `n_estimators=300, class_weight=balanced` | no |
| **AdaBoost** | boosting | sequential stumps, re-weighting the misclassified | `n_estimators=300, learning_rate=0.5` | no |
| **GradBoost** | boosting | sequential trees fitting the previous residuals (sklearn) | defaults | no |
| **HistGradBoost** | boosting | histogram-binned gradient boosting (fast on big data) | defaults | no |
| **XGBoost** | boosting | regularized gradient boosting | `n_estimators=300, max_depth=4, lr=0.05, subsample=0.9` (base) | **yes** |
| **LightGBM** | boosting | leaf-wise gradient boosting, very fast | `n_estimators=300, max_depth=4, lr=0.05, subsample=0.9` | no |
| **CatBoost** | boosting | ordered boosting, strong defaults | `iterations=300, depth=4, lr=0.05` | no |
| **MLP** (sklearn) | neural net | 2-hidden-layer perceptron (scikit-learn) | `hidden_layer_sizes=(64,32), max_iter=600, early_stopping=True` | no |
| **torch_mlp** | neural net | PyTorch 2-hidden-layer net, served via `?algo=torch_mlp` | see §5 | no |

All sklearn/boosting models use `random_state=42`. The "base" hyperparameters for RF/XGBoost are
only the starting point — they are **hyperparameter-searched** (see §4).

---

## 4. Hyperparameter search (Task 2) — `RandomizedSearchCV`, 20 candidates, 5-fold CV

| Model | Search space |
|---|---|
| **RandomForest** | `n_estimators ∈ {100,200,300,500}`, `max_depth ∈ {None,4,6,8,12}`, `min_samples_split ∈ {2,5,10}` |
| **XGBoost** | `n_estimators ∈ {100,200,300,500}`, `max_depth ∈ {2,3,4,6}`, `learning_rate ∈ {0.01,0.05,0.1,0.2}`, `subsample ∈ {0.7,0.8,0.9,1.0}` |

The **actual chosen** values per group are saved in `_meta.json` under `models.<algo>.tuned_params`
(e.g. `GET /models/best` returns them live).

---

## 5. PyTorch model (`torch_mlp`)

Architecture (`src/torch_nn.py`, mirrored in `src/api.py` for serving):
```
input (encoded features)
  → Linear(→64) → ReLU → Dropout(0.3)
  → Linear(→32) → ReLU → Dropout(0.3)
  → Linear(→1)  → sigmoid  (probability)
```
Training (`src/train_torch.py`): **Adam** (`lr=1e-3, weight_decay=1e-4`), **BCEWithLogitsLoss**
with `pos_weight` for imbalance, mini-batches of 128, **early stopping** on validation loss
(patience 20). Saved as a `.pt` bundle (weights + the fitted preprocessor + input dim) per served
group; the API reloads it and runs a forward pass for `?algo=torch_mlp`. It performs on par with
the sklearn models (not better) — opt-in, not the default.

---

## 6. Where the actual numbers live
- **Per-model scores + tuned params:** `models/<variant>/<condition>/_meta.json` (or `GET /models`).
- **Best model per group:** `GET /models/best`.
- **Full 14-model comparison table:** `reports/full_vs_deployable.csv`.
- **SHAP feature importances:** `reports/shap_top_features.json`, `reports/shap_interpretation.md`.
- **Neural-net studies:** `reports/nn_discussion.md`, `reports/nn_scaling.md`.
