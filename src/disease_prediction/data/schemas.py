"""Explicit, declarative schemas for every supported dataset.

The schemas drive three things at once:

1. **Raw-data validation** (types, categories, ranges, missing markers).
2. **Inference-time input validation** (allowed categories and numeric ranges
   surfaced through the API and dashboard).
3. **Documentation** (``docs/data_dictionary.md`` is generated from them).

They are intentionally independent per dataset: the three datasets do not share
feature names, target conventions or missing-value encodings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class FeatureKind(StrEnum):
    """The kind of a feature, used to select preprocessing and validation."""

    NUMERIC = "numeric"
    BINARY = "binary"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class FeatureSpec:
    """A single, documented input feature."""

    name: str
    kind: FeatureKind
    description: str = ""
    unit: str | None = None
    #: Inclusive plausibility bounds. ``None`` means "no hard bound; only
    #: finiteness is checked". Bounds are deliberately wider than the observed
    #: dataset range so that legitimate out-of-sample patients are accepted.
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[int | str, ...] | None = None
    #: True when the raw dataset legitimately contains missing values.
    optional: bool = False


@dataclass(frozen=True)
class DatasetSchema:
    """Everything needed to load, validate and document one dataset."""

    task: str
    display_name: str
    source_url: str
    license: str
    citation: str
    target_column: str
    positive_class: int | str
    negative_class: int | str
    target_meaning: str
    features: tuple[FeatureSpec, ...]
    target_description: str = ""
    id_columns: tuple[str, ...] = ()
    notes: str = ""
    missing_markers: tuple[str, ...] = field(default=("?",))

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def feature(self, name: str) -> FeatureSpec:
        for spec in self.features:
            if spec.name == name:
                return spec
        raise KeyError(f"Unknown feature {name!r} for task {self.task!r}")

    def categorical_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind is not FeatureKind.NUMERIC]

    def numeric_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind is FeatureKind.NUMERIC]

    def binary_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind is FeatureKind.BINARY]


def make_feature(
    name: str,
    kind: FeatureKind,
    description: str,
    *,
    unit: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    allowed_values: tuple[int | str, ...] | None = None,
    optional: bool = False,
) -> FeatureSpec:
    """Terse constructor used to keep the registry tables readable."""
    return FeatureSpec(
        name=name,
        kind=kind,
        description=description,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        allowed_values=allowed_values,
        optional=optional,
    )
