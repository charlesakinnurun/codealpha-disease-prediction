"""Feature-schema objects used at inference time.

The schema stored inside a model artifact is the contract the API validates
against: ordered feature names, their kinds, allowed categories and the numeric
bounds observed during training. Loading an artifact whose schema does not match
the incoming request fails loudly rather than silently producing a prediction
from mis-ordered columns.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InferenceFeature(BaseModel):
    """A single feature as seen by a trained artifact."""

    name: str
    kind: str  # "numeric" | "binary" | "categorical"
    unit: str | None = None
    allowed_values: list[int | str] | None = None
    minimum: float | None = None
    maximum: float | None = None
    observed_min: float | None = None
    observed_max: float | None = None


class InferenceSchema(BaseModel):
    """The full ordered feature contract of a trained artifact."""

    task: str
    features: list[InferenceFeature]
    target_column: str = "target"
    classes: list[int] = Field(default_factory=lambda: [0, 1])
    positive_class: int | str = 1
    target_meaning: str = ""

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def feature(self, name: str) -> InferenceFeature | None:
        for f in self.features:
            if f.name == name:
                return f
        return None

    def validate_row(self, row: dict[str, Any], *, require_all: bool = True) -> list[str]:
        """Validate a single feature mapping, returning human-readable problems.

        This is the same logic the API uses, so unit tests here directly cover
        request validation. It never mutates *row* and never fills defaults.
        """
        problems: list[str] = []
        known = set(self.feature_names)
        for key in row:
            if key not in known:
                problems.append(f"Unknown feature '{key}'.")

        for spec in self.features:
            if spec.name not in row:
                if require_all and spec.kind == "numeric" and spec.observed_min is None:
                    problems.append(f"Missing required feature '{spec.name}'.")
                continue
            value = row[spec.name]
            if value is None:
                continue
            if spec.kind == "numeric":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    problems.append(f"Feature '{spec.name}' must be numeric.")
                elif spec.allowed_values is None:
                    lo = spec.minimum if spec.minimum is not None else spec.observed_min
                    hi = spec.maximum if spec.maximum is not None else spec.observed_max
                    if lo is not None and value < lo:
                        problems.append(f"Feature '{spec.name}' below minimum {lo}.")
                    if hi is not None and value > hi:
                        problems.append(f"Feature '{spec.name}' above maximum {hi}.")
            elif spec.allowed_values is not None and value not in spec.allowed_values:
                problems.append(
                    f"Feature '{spec.name}' must be one of {spec.allowed_values}, got {value!r}."
                )
        return problems
