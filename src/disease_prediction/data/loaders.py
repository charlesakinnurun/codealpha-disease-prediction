"""Dataset loaders.

Each loader turns a raw artefact into a *canonical* frame: the schema's feature
columns (typed, missing values as ``NaN``) plus a single integer ``target``
column where ``1`` means the documented positive class. Target semantics are
never silently changed between datasets — see :mod:`disease_prediction.data.registry`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from disease_prediction.config import Settings, get_settings
from disease_prediction.data.registry import get_schema

# --- Raw file locations -----------------------------------------------------
HEART_RAW_FILE = Path("heart+disease") / "processed.cleveland.data"
DIABETES_RAW_REL = Path("raw") / "diabetes" / "diabetes_data_upload.csv"

_HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Raw diabetes headers contain spaces; map them onto the registry's canonical names.
_DIABETES_COLUMN_MAP = {
    "Age": "Age",
    "Gender": "Gender",
    "Polyuria": "Polyuria",
    "Polydipsia": "Polydipsia",
    "sudden weight loss": "sudden_weight_loss",
    "weakness": "weakness",
    "Polyphagia": "Polyphagia",
    "Genital thrush": "Genital_thrush",
    "visual blurring": "visual_blurring",
    "Itching": "Itching",
    "Irritability": "Irritability",
    "delayed healing": "delayed_healing",
    "partial paresis": "partial_paresis",
    "muscle stiffness": "muscle_stiffness",
    "Alopecia": "Alopecia",
    "Obesity": "Obesity",
    "class": "class",
}


def heart_raw_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_dir / HEART_RAW_FILE


def diabetes_raw_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_dir / DIABETES_RAW_REL


# ---------------------------------------------------------------------------
# Heart Disease
# ---------------------------------------------------------------------------
def load_heart(settings: Settings | None = None) -> pd.DataFrame:
    """Load and canonicalise the UCI Cleveland heart disease data."""
    path = heart_raw_path(settings)
    if not path.exists():
        raise FileNotFoundError(
            f"Heart disease data not found at {path}. "
            "Run `python scripts/download_data.py --task heart_disease`."
        )
    df = pd.read_csv(
        path,
        header=None,
        names=_HEART_COLUMNS,
        na_values=["?"],
        comment=None,
        skipinitialspace=True,
    )
    for col in _HEART_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Binary target: presence of disease (num > 0).
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns=["num"])


# ---------------------------------------------------------------------------
# Diabetes
# ---------------------------------------------------------------------------
def load_diabetes(settings: Settings | None = None) -> pd.DataFrame:
    """Load and canonicalise the UCI early-stage diabetes risk dataset."""
    path = diabetes_raw_path(settings)
    if not path.exists():
        raise FileNotFoundError(
            f"Diabetes data not found at {path}. "
            "Run `python scripts/download_data.py --task diabetes`."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    unknown = set(df.columns) - set(_DIABETES_COLUMN_MAP)
    if unknown:
        raise ValueError(
            f"Unexpected diabetes columns {sorted(unknown)}. "
            "The upstream file may have changed; update the registry and loader."
        )
    df = df.rename(columns=_DIABETES_COLUMN_MAP)

    # Normalise categorical/boolean encodings to the schema's numeric codes.
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    binary_cols = [c for c in df.columns if c not in {"Age", "Gender", "class", "target"}]
    for col in binary_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
        )
    df["target"] = (
        df["class"].astype(str).str.strip().str.lower().map({"positive": 1, "negative": 0})
    )
    if df["target"].isna().any():
        bad = sorted(df.loc[df["target"].isna(), "class"].unique())
        raise ValueError(f"Unrecognised diabetes class labels: {bad}")
    df["target"] = df["target"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Breast Cancer (bundled with scikit-learn — no network required)
# ---------------------------------------------------------------------------
def load_breast_cancer(settings: Settings | None = None) -> pd.DataFrame:
    """Load the UCI WDBC data from scikit-learn and canonicalise the target."""
    from sklearn.datasets import load_breast_cancer as _load

    bunch = _load(as_frame=True)
    df = bunch.frame.copy()
    # scikit-learn: 0 = malignant, 1 = benign. Positive class = malignant.
    df["target"] = (df["target"] == 0).astype(int)
    schema = get_schema("breast_cancer")
    keep = schema.feature_names + ["target"]
    missing = set(keep) - set(df.columns)
    if missing:
        raise ValueError(f"scikit-learn breast cancer frame is missing columns: {sorted(missing)}")
    return df[keep]


_LOADERS = {
    "heart_disease": load_heart,
    "diabetes": load_diabetes,
    "breast_cancer": load_breast_cancer,
}


def load_dataset(task: str, settings: Settings | None = None) -> pd.DataFrame:
    """Load the canonical frame for *task*.

    Returns a frame with exactly ``schema.feature_names + ["target"]`` columns
    and an integer target in ``{0, 1}``. Feature dtypes follow the schema:
    numeric features are ``float64``/``int64`` with ``NaN`` for missing values.
    """
    if task not in _LOADERS:
        raise KeyError(f"No loader for task {task!r}")
    df = _LOADERS[task](settings)
    schema = get_schema(task)
    expected = schema.feature_names + ["target"]
    extra = [c for c in df.columns if c not in expected]
    ordered = df[expected].copy()
    # Coerce feature dtypes deterministically according to their declared kind.
    for spec in schema.features:
        if spec.kind.value == "numeric":
            ordered[spec.name] = pd.to_numeric(ordered[spec.name], errors="coerce").astype(float)
        elif spec.allowed_values is None or all(
            isinstance(v, int) and not isinstance(v, bool) for v in spec.allowed_values
        ):
            # Binary / integer-coded categorical feature.
            ordered[spec.name] = pd.to_numeric(ordered[spec.name], errors="coerce")
        else:
            # String categorical feature — keep as text, do not coerce.
            ordered[spec.name] = ordered[spec.name].astype("string")
    ordered["target"] = ordered["target"].astype(int)
    if extra:
        # `class` from the diabetes loader is retained only for reporting.
        for col in extra:
            if col == "class":
                ordered[col] = df[col].values
    return ordered


def target_counts(df: pd.DataFrame) -> dict[int, int]:
    """Return a plain {label: count} mapping for the target column."""
    counts = df["target"].value_counts().to_dict()
    return {int(k): int(v) for k, v in counts.items()}
