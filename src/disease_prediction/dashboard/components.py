"""Dashboard helper functions.

Deliberately free of Streamlit imports so they can be unit-tested directly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from disease_prediction.api.dependencies import ModelService, PredictionOutcome
from disease_prediction.config import TASK_LABELS

DISCLAIMER = (
    "⚠️ **Educational and research use only.** This application estimates "
    "statistical risk from benchmark models. It is **not** a diagnostic tool, "
    "must not be used to diagnose, treat, or rule out any disease, and does not "
    "replace a qualified healthcare professional."
)

EXAMPLE_NOTICE = (
    "Any example values are synthetic or drawn from public benchmark datasets. "
    "Never enter real patient information."
)


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task)


def prediction_label(outcome: PredictionOutcome) -> str:
    """Human-readable class label for a prediction outcome."""
    return str(outcome.positive_class) if outcome.label == 1 else str(outcome.negative_class)


def probability_frame(outcome: PredictionOutcome) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": [str(outcome.positive_class), str(outcome.negative_class)],
            "probability": [outcome.probability_positive, outcome.probability_negative],
        }
    )


def contributions_frame(outcome: PredictionOutcome) -> pd.DataFrame | None:
    if not outcome.contributions:
        return None
    df = pd.DataFrame(outcome.contributions)
    df["abs_contribution"] = df["contribution"].abs()
    return df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)


def confusion_counts(metrics: dict[str, Any]) -> list[list[int]]:
    """Return a 2x2 confusion matrix [[TN, FP], [FN, TP]] from a metric dict."""
    return [
        [int(metrics.get("tn", 0)), int(metrics.get("fp", 0))],
        [int(metrics.get("fn", 0)), int(metrics.get("tp", 0))],
    ]


def metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Format a metric dict as a tidy two-column table for display."""
    keys = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
        "threshold",
        "n",
    ]
    rows = [{"metric": k, "value": metrics.get(k)} for k in keys if k in metrics]
    return pd.DataFrame(rows)


def model_service() -> ModelService:
    """Create a model service for the dashboard (cheap; models load lazily)."""
    return ModelService()


def run_prediction(
    service: ModelService,
    task: str,
    features: dict[str, Any],
    *,
    explain: bool = False,
) -> tuple[PredictionOutcome | None, str | None]:
    """Run a prediction, returning ``(outcome, error_message)``."""
    try:
        return service.predict(task, features, explain=explain), None
    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
        return None, str(exc)


def champion_metrics(service: ModelService, task: str) -> dict[str, Any] | None:
    """Return the champion model's held-out test metrics, if available."""
    try:
        loaded = service.load(task)
    except Exception:  # noqa: BLE001
        return None
    return dict(loaded.metadata.test_metrics) or None


def champion_metadata(service: ModelService, task: str) -> dict[str, Any] | None:
    try:
        return service.load(task).metadata.model_dump()
    except Exception:  # noqa: BLE001
        return None
