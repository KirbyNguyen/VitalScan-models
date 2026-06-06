# Stretch — Neural Network vs Tree-Based Models (Task 5)

> This is the **small-data** comparison (Pima 768, UCI ~920). For the large-data follow-up
> (MLP vs TabNet vs XGBoost + an AUC-vs-data-size scaling curve), see `reports/nn_scaling.md`.

**Setup.** A 3-layer fully-connected network (input → 64 → 32 → 1, ReLU + dropout, sigmoid
output) trained in PyTorch with the Adam optimizer, binary cross-entropy loss (`BCEWithLogitsLoss`
with `pos_weight` for class imbalance), and **early stopping on validation loss** (20% of train
held out, patience 25). Compared against the tuned RandomForest and XGBoost on the identical
test split, full-clinical features. Code: `src/torch_nn.py`. Results: `reports/nn_comparison.csv`.

## Results (test ROC-AUC)

| Condition | PyTorch MLP | RandomForest | XGBoost | epochs to stop |
|---|---|---|---|---|
| Diabetes | 0.808 | **0.819** | 0.813 | 75 |
| Heart    | 0.907 | 0.907 | **0.909** | 33 |

The neural net is **competitive but never best**: it ties on heart and trails the trees by
~0.01 on diabetes.

## Discussion: the "gradient boosting wins on tabular data" phenomenon

This result is the textbook outcome, and it is worth understanding *why* rather than just
noting it. Across the literature (e.g. Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022),
gradient-boosted decision trees remain the default winner on small-to-medium tabular data, for
several structural reasons that all apply here:

- **Sample size.** With only 768 (diabetes) and ~920 (heart) rows, a neural network has far too
  few examples to learn good representations from scratch; it overfits and leans heavily on
  regularization (dropout, weight decay, early stopping) just to stay competitive. Trees, by
  contrast, are efficient learners on little data. The effect is *sample-size dependent* — and we
  tested that directly in a follow-up scaling study (`reports/nn_scaling.md`): on the large CDC
  dataset the MLP **catches up to parity** with gradient boosting (≈0.826 each at 200k rows). So
  more data closes this gap — it just doesn't let the NN decisively *beat* boosting on tabular data.
- **Feature geometry.** Tabular features are heterogeneous (a one-hot chest-pain flag and a
  continuous cholesterol value live on different scales and have no spatial/sequential
  structure). Trees split on individual features and are invariant to monotonic rescaling; MLPs
  must learn that structure through dense layers, which is harder and needs more data.
- **Irregular decision boundaries.** Boosting builds sharp axis-aligned, piecewise-constant
  boundaries that fit tabular targets well; MLPs are biased toward smooth functions, a poorer
  match for threshold-like clinical rules ("BP > 140 → flag").
- **Tuning effort.** The trees here were hyperparameter-searched; the MLP used sensible defaults.
  Even with more NN tuning, the expected ceiling is "match," not "beat," at this scale.

## When *would* we prefer the neural network?

When the inputs stop being clean tabular rows: raw rPPG **waveforms** (Group 1's actual signal),
images, or text; when the dataset is large (tens of thousands of rows or more); when we want one
model to fuse modalities (vitals + signal + image) end-to-end; or when we need transfer learning
from a pretrained backbone. For VitalScan's *tabular risk-scoring* stage specifically, the
evidence says ship the tuned tree model — it is more accurate, faster, cheaper to serve, and
(with SHAP) more interpretable. The neural net is the right tool one stage upstream, inside
Group 1's signal-extraction pipeline.
