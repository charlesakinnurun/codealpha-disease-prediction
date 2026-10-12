"""Model comparison reporting.

Turns a :class:`~disease_prediction.models.train.TrainingResult` into
human-readable Markdown and machine-readable JSON. No metric is ever invented
here — everything comes from the objects produced by actual training runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from disease_prediction.config import Settings, get_settings

if TYPE_CHECKING:  # avoid a runtime import cycle
    from disease_prediction.models.train import TrainingResult


def comparison_markdown(result: TrainingResult) -> str:
    rows = result.comparison_table()
    lines = [
        f"# Model comparison — {result.task}",
        "",
        f"- Split sizes: {result.split_sizes}",
        f"- Random seed: {result.random_seed}",
        f"- Selection metric: **{result.selection_metric}** (validation)",
        f"- Champion: **{result.champion}**",
        f"- Rationale: {result.rationale}",
        "",
        "## Validation metrics",
        "",
        "| Model | CV ROC-AUC | CV std | Val ROC-AUC | Val PR-AUC | Val F1 | Sens | Spec | Brier | Threshold | Calibrated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        lines.append(
            "| {display_name} | {cv_roc_auc:.4f} | {cv_std:.4f} | {val_roc_auc:.4f} | "
            "{val_pr_auc:.4f} | {val_f1:.4f} | {val_sensitivity:.4f} | {val_specificity:.4f} | "
            "{val_brier:.4f} | {threshold:.3f} | {calibrated} |".format(**row)
        )
    lines.append("")
    lines.append("## Held-out test metrics (champion)")
    lines.append("")
    champ = result.results.get(result.champion or "")
    if champ is not None:
        m = champ.test_metrics
        lines.append(
            "| Metric | Value |\n| --- | ---: |\n"
            + "\n".join(f"| {k} | {v} |" for k, v in m.items() if k not in {"tn", "fp", "fn", "tp"})
        )
        lines.append("")
        lines.append(
            f"Confusion matrix — TN={m.get('tn')}, FP={m.get('fp')}, "
            f"FN={m.get('fn')}, TP={m.get('tp')}"
        )
    lines.append("")
    return "\n".join(lines)


def write_comparison_table(
    result: TrainingResult,
    settings: Settings | None = None,
) -> Path:
    settings = settings or get_settings()
    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    path = settings.metrics_dir / f"{result.task}_comparison.md"
    path.write_text(comparison_markdown(result), encoding="utf-8")
    return path


def write_metrics_json(
    result: TrainingResult,
    settings: Settings | None = None,
) -> Path:
    settings = settings or get_settings()
    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    path = settings.metrics_dir / f"{result.task}_metrics.json"
    payload: dict[str, Any] = {
        "task": result.task,
        "champion": result.champion,
        "selection_metric": result.selection_metric,
        "rationale": result.rationale,
        "split_sizes": result.split_sizes,
        "random_seed": result.random_seed,
        "models": {
            name: {
                "display_name": r.display_name,
                "best_params": r.best_params,
                "cv_score": None if not _finite(r.cv_score) else r.cv_score,
                "cv_std": None if not _finite(r.cv_std) else r.cv_std,
                "threshold": r.threshold,
                "calibrated": r.calibrated,
                "validation_metrics": r.validation_metrics,
                "test_metrics": r.test_metrics,
            }
            for name, r in result.results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _finite(value: float) -> bool:
    import math

    return isinstance(value, (int, float)) and math.isfinite(value)


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)
