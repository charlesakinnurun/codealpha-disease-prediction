# Model Card

This card documents the three task-specific models produced by this repository.
All numbers below were produced by `python scripts/train.py` and independently
re-derived by `python scripts/evaluate.py` (seed 42). Reproduce them with those
commands; do not hand-edit results.

> **Educational and research use only.** These models are not medical devices and
> are not clinically validated. They must not be used to diagnose, treat, or rule
> out any disease. See `docs/responsible_use.md`.

## Common protocol

- **Splits:** stratified train / validation / test using seed 42
  (heart 181/61/61, diabetes 312/104/104, breast cancer 341/114/114).
- **Preprocessing:** fitted inside the pipeline on training folds only —
  median imputation + standardisation for numeric features, most-frequent
  imputation for binary features, most-frequent imputation + one-hot encoding for
  categorical features.
- **Model selection:** `RandomizedSearchCV` with stratified 5-fold CV on the train
  split; champion chosen by **validation ROC-AUC** (threshold-free).
- **Calibration:** sigmoid (Platt) calibration fitted on the **validation** split
  for the champion.
- **Threshold:** chosen on the **validation** split maximising Youden's J
  (`sensitivity + specificity − 1`). The test split is used only for reporting.
- **Metrics reported:** test-split values. ROC-AUC is accompanied by a percentile
  bootstrap 95% CI (1000 resamples).

---

## Model 1 — Heart Disease — `logistic_regression`

| | |
| --- | --- |
| **Task id** | `heart_disease` |
| **Champion** | Logistic Regression (calibrated, sigmoid) |
| **Artifact version** | `0.1.0.20261012162753` |
| **Dataset** | UCI Heart Disease, Cleveland subset (303 rows, 14 attributes) |
| **Source / licence** | <https://archive.ics.uci.edu/dataset/45/heart+disease> · CC BY 4.0 |
| **Target** | Presence of angiographic coronary artery disease (`num > 0`) |
| **Intended use** | Research/teaching; comparing explainable models on tabular clinical data |
| **Out of scope** | Any clinical decision, triage, or patient communication |

**Held-out test metrics (n = 61, threshold = 0.5495)**

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Sensitivity (recall) | 0.9286 |
| Specificity | 0.8485 |
| F1 | 0.8814 |
| ROC-AUC | 0.9589 (95% CI 0.9110–0.9909) |
| PR-AUC | 0.9530 |
| Brier score | 0.0842 |
| Confusion matrix | TN 28 · FP 5 · FN 2 · TP 26 |

**Error trade-offs.** 2 false negatives (missed disease) and 5 false positives.
In a screening framing a missed case is usually costlier, and the champion's
threshold (chosen on validation) leans toward higher sensitivity.

**Explanations.** Logistic-regression coefficients are available on the
standardised/one-hot scale; permutation importance is computed on the test split.
`thal`, `ca`, `cp`, `oldpeak` and `sex` dominate. Coefficients are **not** causal:
`thal` and `ca` encode downstream clinical findings, so they correlate with the
target by construction.

---

## Model 2 — Diabetes — `svm`

| | |
| --- | --- |
| **Task id** | `diabetes` |
| **Champion** | RBF SVM (Platt-scaled, then sigmoid-calibrated on validation) |
| **Artifact version** | `0.1.0.20261012163158` |
| **Dataset** | UCI Early Stage Diabetes Risk Prediction (520 rows) |
| **Source / licence** | <https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset> · CC BY 4.0 |
| **Target** | Questionnaire-based risk classification (Positive/Negative) |
| **Intended use** | Research/teaching on symptom-questionnaire tabular data |
| **Out of scope** | Any diagnostic or screening decision for real individuals |

**Held-out test metrics (n = 104, threshold = 0.0556)**

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9712 |
| Precision | 0.9552 |
| Sensitivity (recall) | 1.0000 |
| Specificity | 0.9250 |
| F1 | 0.9771 |
| ROC-AUC | 0.9883 (95% CI 0.9584–1.0000) |
| PR-AUC | 0.9908 |
| Brier score | 0.0220 |
| Confusion matrix | TN 37 · FP 3 · FN 0 · TP 64 |

**Error trade-offs.** No false negatives at this operating point, at the cost of
3 false positives. The **low threshold (0.056)** is a direct consequence of the
Youden objective on validation plus calibration: it should be read as "the
operating point that favours sensitivity", not as a probability cut-off with
clinical meaning.

**Data caveats.** The dataset contains **269 exact duplicate feature rows**
(52%), a documented property of the questionnaire release. This caps how much
independent signal exists and inflates apparent performance; treat the high
metrics with scepticism. The label is *not* a laboratory diagnosis.

**Explanations.** As an RBF SVM has no native coefficients, permutation
importance (and SHAP where feasible) is used. `Polyuria`, `Polydipsia`,
`Age` and `sudden_weight_loss` dominate.

---

## Model 3 — Breast Cancer — `svm`

| | |
| --- | --- |
| **Task id** | `breast_cancer` |
| **Champion** | RBF SVM (Platt-scaled, then sigmoid-calibrated on validation) |
| **Artifact version** | `0.1.0.20261012163346` |
| **Dataset** | Breast Cancer Wisconsin (Diagnostic), 569 rows, 30 features |
| **Source / licence** | <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic> · CC BY 4.0 / bundled with scikit-learn (BSD-3-Clause) |
| **Target** | Malignant breast mass (positive class) |
| **Intended use** | Research/teaching on high-dimensional tabular classification |
| **Out of scope** | Any clinical use; results depend on one clinic's imaging protocol |

**Held-out test metrics (n = 114, threshold = 0.2646)**

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9825 |
| Precision | 0.9762 |
| Sensitivity (recall) | 0.9762 |
| Specificity | 0.9861 |
| F1 | 0.9762 |
| ROC-AUC | 0.9967 (95% CI 0.9872–1.0000) |
| PR-AUC | 0.9954 |
| Brier score | 0.0148 |
| Confusion matrix | TN 71 · FP 1 · FN 1 · TP 41 |

**Error trade-offs.** 1 false negative and 1 false positive. Even at very high
accuracy the CI is wide because the test split has only 42 malignant cases.

**Explanations.** The 30 features are the mean/error/worst triplets of ten
nuclei measurements and are **highly correlated**. Attributions therefore
concentrate on a handful of features and can flip between near-duplicate
predictors; do not read single-feature importance as a biological mechanism.

---

## Ethical considerations and limitations

- **Representativeness.** Small, old, single-region or single-clinic datasets.
  Benchmark performance is not evidence of transfer to other populations,
  equipment or workflows.
- **Labels.** The diabetes target is a questionnaire classification, not a
  diagnosis; the heart target is an angiographic outcome used as a proxy for
  clinical disease.
- **Subgroup performance.** Not reported. The datasets are too small for
  adequately powered subgroup estimates, and reporting them without sufficient
  data would be misleading. Any subgroup evaluation must be legally and
  ethically appropriate.
- **Fairness/regulatory claims.** None are made. This is not a regulated device.
- **Calibration.** Calibrated on small validation splits; may not transfer.
  Uncalibrated models are labelled as such by the API and dashboard.
- **Explanations.** Attributions describe the model, not the biology, and are
  unstable under correlated inputs.

## Prohibited uses

Clinical diagnosis, triage, or treatment decisions; patient-facing risk
communication without clinician oversight; insurance/employment decisions; any
use with real identifiable patient data without legal basis and safeguards.
