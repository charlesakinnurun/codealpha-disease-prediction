# Model comparison — heart_disease

- Split sizes: {'train': 181, 'validation': 61, 'test': 61}
- Random seed: 42
- Selection metric: **roc_auc** (validation)
- Champion: **logistic_regression**
- Rationale: Selected `logistic_regression` by highest validation roc_auc. Selection used validation data only; the test set was not consulted. Runner-up was `svm` (Δroc_auc = 0.0065); the margin should be read alongside cross-validation variability. Dummy baseline roc_auc = 0.5000; champion = 0.9470.

## Validation metrics

| Model | CV ROC-AUC | CV std | Val ROC-AUC | Val PR-AUC | Val F1 | Sens | Spec | Brier | Threshold | Calibrated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Logistic Regression | 0.8873 | 0.0469 | 0.9470 | 0.9477 | 0.8462 | 0.7857 | 0.9394 | 0.0912 | 0.549 | True |
| Support Vector Machine (RBF) | 0.8904 | 0.0421 | 0.9405 | 0.9428 | 0.8627 | 0.7857 | 0.9697 | 0.1071 | 0.611 | False |
| XGBoost | 0.8724 | 0.0466 | 0.9264 | 0.9330 | 0.8571 | 0.7500 | 1.0000 | 0.1117 | 0.620 | False |
| Random Forest | 0.8850 | 0.0418 | 0.9113 | 0.9168 | 0.8000 | 0.7143 | 0.9394 | 0.1333 | 0.608 | False |
| Dummy (prior) baseline | nan | nan | 0.5000 | 0.4905 | 0.0000 | 0.0000 | 1.0000 | 0.2483 | 0.500 | False |

## Held-out test metrics (champion)

| Metric | Value |
| --- | ---: |
| threshold | 0.5495 |
| n | 61 |
| accuracy | 0.8852459016393442 |
| precision | 0.8387096774193549 |
| recall | 0.9285714285714286 |
| sensitivity | 0.9285714285714286 |
| specificity | 0.8484848484848485 |
| f1 | 0.8813559322033898 |
| roc_auc | 0.9588744588744589 |
| pr_auc | 0.9529999635725468 |
| brier | 0.08420697716037195 |
| positive_rate | 0.45901639344262296 |
| roc_auc_ci95 | [0.9109831029185868, 0.9908557346543617] |
| pr_auc_ci95 | [0.8856020500319356, 0.9906614882727868] |

Confusion matrix — TN=28, FP=5, FN=2, TP=26
