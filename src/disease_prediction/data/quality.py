"""Data-quality and EDA reporting.

Produces machine-readable profiles (``profile.json``) used downstream for
input-range validation and human-readable Markdown summaries. Charts are
generated separately by the EDA notebooks and ``scripts/validate_data.py``
so that this module stays dependency-light and testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from disease_prediction.config import Settings, get_settings
from disease_prediction.data.schemas import DatasetSchema, FeatureKind
from disease_prediction.data.validate import ValidationReport


def profile_frame(df: pd.DataFrame, schema: DatasetSchema) -> dict[str, Any]:
    """Build a JSON-serialisable profile of a canonical frame."""
    columns: dict[str, Any] = {}
    for spec in schema.features:
        if spec.name not in df.columns:
            continue
        series = df[spec.name]
        entry: dict[str, Any] = {
            "kind": spec.kind.value,
            "description": spec.description,
            "unit": spec.unit,
            "dtype": str(series.dtype),
            "n_missing": int(series.isna().sum()),
            "pct_missing": round(float(series.isna().mean() * 100.0), 3),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if spec.kind is FeatureKind.NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce")
            entry.update(
                {
                    "min": _clean(numeric.min()),
                    "max": _clean(numeric.max()),
                    "mean": _clean(numeric.mean()),
                    "median": _clean(numeric.median()),
                    "std": _clean(numeric.std()),
                    "p01": _clean(numeric.quantile(0.01)),
                    "p99": _clean(numeric.quantile(0.99)),
                    "allowed_min": spec.minimum,
                    "allowed_max": spec.maximum,
                }
            )
        else:
            entry["allowed_values"] = list(spec.allowed_values or [])
            entry["observed_values"] = sorted(
                (int(v) if isinstance(v, (int, np.integer)) else str(v))
                for v in series.dropna().unique().tolist()
            )
        columns[spec.name] = entry

    target_counts = {
        str(int(k)): int(v) for k, v in df["target"].value_counts().sort_index().items()
    }
    return {
        "task": schema.task,
        "display_name": schema.display_name,
        "n_rows": int(len(df)),
        "n_features": len(schema.features),
        "target": {
            "column": schema.target_column,
            "positive_class": schema.positive_class,
            "negative_class": schema.negative_class,
            "meaning": schema.target_meaning,
            "counts": target_counts,
            "positive_rate": round(float(df["target"].mean()), 4),
        },
        "columns": columns,
    }


def save_profile(
    task: str,
    df: pd.DataFrame,
    schema: DatasetSchema,
    settings: Settings | None = None,
) -> Path:
    """Write ``data/processed/<task>/profile.json`` and return its path."""
    settings = settings or get_settings()
    out_dir = settings.processed_dir / task
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "profile.json"
    path.write_text(json.dumps(profile_frame(df, schema), indent=2), encoding="utf-8")
    return path


def write_quality_report(
    report: ValidationReport,
    path: Path,
    *,
    profile: dict[str, Any] | None = None,
) -> Path:
    """Render a validation report (optionally with a profile) to Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# Data Quality Report — {report.task}", ""]
    lines.append(f"- **Rows:** {report.n_rows}")
    lines.append(f"- **Columns:** {report.n_columns}")
    lines.append(f"- **Status:** {'PASS' if report.ok else 'FAIL'}")
    lines.append(f"- **Duplicate feature rows:** {report.duplicate_rows}")
    lines.append(f"- **Target counts:** {report.target_counts}")
    lines.append("")

    if report.errors:
        lines.append("## Errors")
        lines.extend(f"- {e}" for e in report.errors)
        lines.append("")
    if report.warnings:
        lines.append("## Warnings")
        lines.extend(f"- {w}" for w in report.warnings)
        lines.append("")

    lines.append("## Feature profile")
    lines.append("")
    lines.append("| Feature | Kind | Missing % | Unique | Min | Max | Out-of-range |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for col in report.columns:
        lines.append(
            f"| {col.name} | {col.kind} | {col.pct_missing:.2f} | {col.n_unique} | "
            f"{_fmt(col.minimum)} | {_fmt(col.maximum)} | {col.out_of_range} |"
        )
    lines.append("")

    if profile is not None:
        lines.append("## Target distribution")
        lines.append("")
        tgt = profile["target"]
        lines.append(
            f"- Positive rate: **{tgt['positive_rate']:.3f}** "
            f"({tgt['counts']}) — positive means *{tgt['meaning']}*."
        )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _clean(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4g}"
