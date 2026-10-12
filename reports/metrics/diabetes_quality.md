# Data Quality Report — diabetes

- **Rows:** 520
- **Columns:** 18
- **Status:** PASS
- **Duplicate feature rows:** 269
- **Target counts:** {'0': 200, '1': 320}

## Warnings
- 269 exact duplicate feature row(s) detected — check whether they represent distinct patients.

## Feature profile

| Feature | Kind | Missing % | Unique | Min | Max | Out-of-range |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Age | numeric | 0.00 | 51 | 16 | 90 | 0 |
| Gender | categorical | 0.00 | 2 | — | — | 0 |
| Polyuria | binary | 0.00 | 2 | — | — | 0 |
| Polydipsia | binary | 0.00 | 2 | — | — | 0 |
| sudden_weight_loss | binary | 0.00 | 2 | — | — | 0 |
| weakness | binary | 0.00 | 2 | — | — | 0 |
| Polyphagia | binary | 0.00 | 2 | — | — | 0 |
| Genital_thrush | binary | 0.00 | 2 | — | — | 0 |
| visual_blurring | binary | 0.00 | 2 | — | — | 0 |
| Itching | binary | 0.00 | 2 | — | — | 0 |
| Irritability | binary | 0.00 | 2 | — | — | 0 |
| delayed_healing | binary | 0.00 | 2 | — | — | 0 |
| partial_paresis | binary | 0.00 | 2 | — | — | 0 |
| muscle_stiffness | binary | 0.00 | 2 | — | — | 0 |
| Alopecia | binary | 0.00 | 2 | — | — | 0 |
| Obesity | binary | 0.00 | 2 | — | — | 0 |

## Target distribution

- Positive rate: **0.615** ({'0': 200, '1': 320}) — positive means *screening-classified diabetes risk (Positive/Negative)*.
