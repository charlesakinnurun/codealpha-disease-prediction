# Model comparison — breast_cancer

- Split sizes: {'train': 341, 'validation': 114, 'test': 114}
- Random seed: 42
- Selection metric: **roc_auc** (validation)
- Champion: **svm**
- Rationale: Selected `svm` by highest validation roc_auc. Selection used validation data only; the test set was not consulted. Runner-up was `logistic_regression` (Δroc_auc = 0.0007); the margin should be read alongside cross-validation variability. Dummy baseline roc_auc = 0.5000; champion = 0.9971.

## Validation metrics

| Model | CV ROC-AUC | CV std | Val ROC-AUC | Val PR-AUC | Val F1 | Sens | Spec | Brier | Threshold | Calibrated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Support Vector Machine (RBF) | 0.9941 | 0.0087 | 0.9971 | 0.9958 | 0.9655 | 0.9767 | 0.9718 | 0.0159 | 0.265 | True |
| Logistic Regression | 0.9939 | 0.0082 | 0.9964 | 0.9953 | 0.9767 | 0.9767 | 0.9859 | 0.0255 | 0.634 | False |
| XGBoost | 0.9925 | 0.0100 | 0.9957 | 0.9942 | 0.9767 | 0.9767 | 0.9859 | 0.0265 | 0.599 | False |
| Random Forest | 0.9891 | 0.0122 | 0.9902 | 0.9884 | 0.9647 | 0.9535 | 0.9859 | 0.0344 | 0.625 | False |
| Dummy (prior) baseline | nan | nan | 0.5000 | 0.3786 | 0.0000 | 0.0000 | 1.0000 | 0.2349 | 0.500 | False |

## Held-out test metrics (champion)

| Metric | Value |
| --- | ---: |
| threshold | 0.2646 |
| n | 114 |
| accuracy | 0.9824561403508771 |
| precision | 0.9761904761904762 |
| recall | 0.9761904761904762 |
| sensitivity | 0.9761904761904762 |
| specificity | 0.9861111111111112 |
| f1 | 0.9761904761904762 |
| roc_auc | 0.9966931216931217 |
| pr_auc | 0.9954212454212453 |
| brier | 0.01477584336290129 |
| positive_rate | 0.3684210526315789 |
| roc_auc_ci95 | [0.9871794871794872, 1.0] |
| pr_auc_ci95 | [0.981802007829647, 1.0] |

Confusion matrix — TN=71, FP=1, FN=1, TP=41
