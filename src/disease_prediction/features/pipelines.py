"""Feature preprocessing pipelines.

Preprocessing lives *inside* scikit-learn pipelines so that imputation,
encoding, scaling and any feature selection are fitted on training folds only.
Each dataset gets its own ``ColumnTransformer`` because the three datasets have
different feature types and missing-value behaviour.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from disease_prediction.data.schemas import DatasetSchema, FeatureKind


def build_preprocessor(schema: DatasetSchema) -> ColumnTransformer:
    """Build a leakage-safe ``ColumnTransformer`` for *schema*.

    - **Numeric** features: median imputation then standardisation.
    - **Binary** features: most-frequent imputation, passed through unchanged.
    - **Categorical** features: most-frequent imputation then one-hot encoding
      with ``handle_unknown="ignore"`` so unseen categories at inference time do
      not crash the pipeline.

    Output is always dense to keep downstream explainability simple.
    """
    numeric = schema.numeric_features()
    binary = schema.binary_features()
    categorical = [f.name for f in schema.features if f.kind is FeatureKind.CATEGORICAL]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if binary:
        transformers.append(
            (
                "binary",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="most_frequent"))],
                ),
                binary,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )

    if not transformers:
        raise ValueError(f"Schema {schema.task!r} declares no features.")

    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def build_pipeline(estimator: BaseEstimator, schema: DatasetSchema) -> Pipeline:
    """Wrap an estimator with the schema's preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(schema)),
            ("model", estimator),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the output feature names of a fitted pipeline's preprocessor."""
    pre: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(pre.get_feature_names_out())
