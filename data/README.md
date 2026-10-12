# Data

This directory holds datasets used by the project. Large or licence-restricted
files are **not** committed to Git and are regenerated with commands below.

```
data/
├── heart+disease/            # UCI Cleveland heart data (vendored, small, kept in Git)
│   └── processed.cleveland.data
├── raw/                      # Downloaded archives/files (gitignored)
│   └── diabetes/
│       ├── diabetes_data_upload.csv
│       └── provenance.json    # URL, retrieval time, size, SHA-256
├── processed/                # Generated profiles (gitignored)
│   └── <task>/profile.json
└── README.md
```

## Sources and licences

| Task | Dataset | Source | Licence |
| --- | --- | --- | --- |
| `heart_disease` | Heart Disease (Cleveland, 303 rows) | [UCI id 45](https://archive.ics.uci.edu/dataset/45/heart+disease) | CC BY 4.0 |
| `diabetes` | Early Stage Diabetes Risk Prediction (520 rows) | [UCI id 529](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) | CC BY 4.0 |
| `breast_cancer` | Breast Cancer Wisconsin (Diagnostic, 569 rows) | [UCI id 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) / bundled with scikit-learn | CC BY 4.0 / BSD-3-Clause |

See `docs/data_dictionary.md` for schema and target definitions, and each task's
citation for the requested attribution.

## Reproduction

```bash
# Fetch/verify all datasets (records provenance.json for downloads)
python scripts/download_data.py

# Validate, write quality reports and EDA figures
python scripts/validate_data.py
```

The Cleveland file is vendored for offline use and to satisfy the dataset's
publication request in `heart+disease.names`. The diabetes file is downloaded
from UCI and its SHA-256 recorded. WDBC is loaded from scikit-learn so no network
access is required.

## Ground rules

- Do not commit downloaded files that are not clearly redistributable.
- Do not add real patient data. Ever.
- Record retrieval date and checksum for any new dataset you add.
