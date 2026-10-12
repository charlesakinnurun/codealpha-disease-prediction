# Model comparison — diabetes

- Split sizes: {'train': 312, 'validation': 104, 'test': 104}
- Random seed: 42
- Selection metric: **roc_auc** (validation)
- Champion: **svm**
- Rationale: Selected `svm` by highest validation roc_auc. Selection used validation data only; the test set was not consulted. Runner-up was `xgboost` (Δroc_auc = 0.0109); the margin should be read alongside cross-validation variability. Dummy baseline roc_auc = 0.5000; champion = 0.9738.

## Validation metrics

| Model | CV ROC-AUC | CV std | Val ROC-AUC | Val PR-AUC | Val F1 | Sens | Spec | Brier | Threshold | Calibrated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Support Vector Machine (RBF) | 0.9970 | 0.0035 | 0.9738 | 0.9821 | 0.9697 | 1.0000 | 0.9000 | 0.0369 | 0.056 | True |
| XGBoost | 0.9959 | 0.0050 | 0.9629 | 0.9773 | 0.9457 | 0.9531 | 0.9000 | 0.0656 | 0.109 | False |
| Random Forest | 0.9974 | 0.0025 | 0.9613 | 0.9714 | 0.9697 | 1.0000 | 0.9000 | 0.0529 | 0.352 | False |
| Logistic Regression | 0.9822 | 0.0145 | 0.9301 | 0.9620 | 0.9206 | 0.9062 | 0.9000 | 0.0923 | 0.489 | False |
| Dummy (prior) baseline | nan | nan | 0.5000 | 0.6271 | 0.7619 | 1.0000 | 0.0000 | 0.2367 | 0.500 | False |

## Held-out test metrics (champion)

| Metric | Value |
| --- | ---: |
| threshold | 0.0556 |
| n | 104 |
| accuracy | 0.9711538461538461 |
| precision | 0.9552238805970149 |
| recall | 1.0 |
| sensitivity | 1.0 |
| specificity | 0.925 |
| f1 | 0.9770992366412213 |
| roc_auc | 0.98828125 |
| pr_auc | 0.9908163759225568 |
| brier | 0.022043871379563453 |
| positive_rate | 0.6153846153846154 |
| roc_auc_ci95 | [0.9584352233132721, 1.0] |
| pr_auc_ci95 | [0.9657357272950108, 1.0] |

Confusion matrix — TN=37, FP=3, FN=0, TP=64
