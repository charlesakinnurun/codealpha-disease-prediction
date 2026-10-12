"""Dataset validation and data-quality profiling.

Produces a :class:`ValidationReport` that separates **errors** (blocking
problems that make a dataset unfit for training) from **warnings** (suspicious
but tolerable conditions). The report doubles as the data-quality artefact
written by ``scripts/validate_data.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from disease_prediction.data.schemas import DatasetSchema, FeatureKind


@dataclass
class ColumnProfile:
    """Per-column quality statistics."""

    name: str
    kind: str
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    unexpected_values: list[str] = field(default_factory=list)
    out_of_range: int = 0


@dataclass
class ValidationReport:
    """Aggregate validation outcome for one dataset."""

    task: str
    n_rows: int
    n_columns: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    columns: list[ColumnProfile] = field(default_factory=list)
    duplicate_rows: int = 0
    target_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data

    def format_text(self) -> str:
        lines = [
            f"Dataset: {self.task}",
            f"Rows: {self.n_rows}  Columns: {self.n_columns}",
            f"Target counts: {self.target_counts}",
            f"Exact duplicate rows: {self.duplicate_rows}",
            f"Status: {'PASS' if self.ok else 'FAIL'}",
        ]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def _unexpected(values: pd.Series, allowed: tuple[int | str, ...]) -> list[str]:
    observed = set(values.dropna().unique().tolist())
    return sorted(str(v) for v in observed - set(allowed))


def validate_dataframe(
    df: pd.DataFrame,
    schema: DatasetSchema,
    *,
    allow_missing_optional: bool = True,
) -> ValidationReport:
    """Validate *df* against *schema* and return a structured report."""
    report = ValidationReport(task=schema.task, n_rows=len(df), n_columns=df.shape[1])

    expected = schema.feature_names
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        report.errors.append(f"Missing required feature columns: {missing_cols}")
    if "target" not in df.columns:
        report.errors.append("Missing target column 'target'.")

    if len(df) == 0:
        report.errors.append("Dataset is empty.")

    for spec in schema.features:
        if spec.name not in df.columns:
            continue
        series = df[spec.name]
        profile = ColumnProfile(
            name=spec.name,
            kind=spec.kind.value,
            dtype=str(series.dtype),
            n_missing=int(series.isna().sum()),
            pct_missing=float(series.isna().mean() * 100.0),
            n_unique=int(series.nunique(dropna=True)),
        )
        non_null_pct = float(series.notna().mean() * 100.0)

        if spec.kind is FeatureKind.NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                profile.minimum = float(numeric.min())
                profile.maximum = float(numeric.max())
                profile.mean = float(numeric.mean())
            if spec.minimum is not None:
                below = numeric < spec.minimum
                profile.out_of_range += int(below.sum())
            if spec.maximum is not None:
                above = numeric > spec.maximum
                profile.out_of_range += int(above.sum())
            if profile.out_of_range:
                report.warnings.append(
                    f"{spec.name}: {profile.out_of_range} value(s) outside documented "
                    f"range [{spec.minimum}, {spec.maximum}]."
                )
        else:
            if spec.allowed_values is not None:
                profile.unexpected_values = _unexpected(series, spec.allowed_values)
                if profile.unexpected_values:
                    report.errors.append(
                        f"{spec.name}: unexpected categories {profile.unexpected_values}; "
                        f"allowed {list(spec.allowed_values)}."
                    )

        # Missing-value policy.
        if profile.n_missing and not spec.optional and allow_missing_optional:
            report.errors.append(
                f"{spec.name}: {profile.n_missing} missing value(s) in a required feature."
            )
        if spec.optional and profile.n_missing and (100.0 - non_null_pct) > 40.0:
            report.warnings.append(
                f"{spec.name}: {profile.pct_missing:.1f}% missing — imputation may be unreliable."
            )

        report.columns.append(profile)

    # Target checks.
    if "target" in df.columns:
        target = df["target"]
        if target.isna().any():
            report.errors.append(f"Target contains {int(target.isna().sum())} missing value(s).")
        unique = set(pd.to_numeric(target, errors="coerce").dropna().unique().tolist())
        if not unique.issubset({0, 1}):
            report.errors.append(f"Target must be binary {{0, 1}}; found {sorted(unique)}.")
        report.target_counts = {
            str(int(k)): int(v) for k, v in target.value_counts().sort_index().items()
        }
        counts = target.value_counts()
        if len(counts) == 2:
            ratio = counts.min() / counts.max()
            if ratio < 0.3:
                report.warnings.append(
                    f"Class imbalance detected (minority/majority ratio = {ratio:.2f})."
                )

    # Duplicate detection. Full-row duplicates can indicate merged registries
    # or repeated patients; we warn rather than fail because some datasets
    # legitimately contain identical feature rows.
    feature_cols = [c for c in expected if c in df.columns]
    report.duplicate_rows = int(df.duplicated(subset=feature_cols).sum())
    if report.duplicate_rows:
        report.warnings.append(
            f"{report.duplicate_rows} exact duplicate feature row(s) detected — "
            "check whether they represent distinct patients."
        )

    # Non-finite guard for numeric columns.
    for spec in schema.features:
        if spec.kind is FeatureKind.NUMERIC and spec.name in df.columns:
            numeric = pd.to_numeric(df[spec.name], errors="coerce")
            n_inf = int(np.isinf(numeric).sum())
            if n_inf:
                report.errors.append(f"{spec.name}: {n_inf} non-finite (inf) value(s).")

    return report
