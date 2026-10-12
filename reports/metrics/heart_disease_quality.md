# Data Quality Report — heart_disease

- **Rows:** 303
- **Columns:** 14
- **Status:** PASS
- **Duplicate feature rows:** 0
- **Target counts:** {'0': 164, '1': 139}

## Feature profile

| Feature | Kind | Missing % | Unique | Min | Max | Out-of-range |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| age | numeric | 0.00 | 41 | 29 | 77 | 0 |
| sex | binary | 0.00 | 2 | — | — | 0 |
| cp | categorical | 0.00 | 4 | — | — | 0 |
| trestbps | numeric | 0.00 | 50 | 94 | 200 | 0 |
| chol | numeric | 0.00 | 152 | 126 | 564 | 0 |
| fbs | binary | 0.00 | 2 | — | — | 0 |
| restecg | categorical | 0.00 | 3 | — | — | 0 |
| thalach | numeric | 0.00 | 91 | 71 | 202 | 0 |
| exang | binary | 0.00 | 2 | — | — | 0 |
| oldpeak | numeric | 0.00 | 40 | 0 | 6.2 | 0 |
| slope | categorical | 0.00 | 3 | — | — | 0 |
| ca | categorical | 1.32 | 4 | — | — | 0 |
| thal | categorical | 0.66 | 3 | — | — | 0 |

## Target distribution

- Positive rate: **0.459** ({'0': 164, '1': 139}) — positive means *presence of angiographic coronary artery disease*.
