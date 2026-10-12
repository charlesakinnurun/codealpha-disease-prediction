from __future__ import annotations

import pandas as pd

from disease_prediction.api.dependencies import PredictionOutcome
from disease_prediction.dashboard.components import (
    confusion_counts,
    contributions_frame,
    metrics_table,
    prediction_label,
    probability_frame,
)


def _outcome(**overrides) -> PredictionOutcome:
    defaults = {
        "task": "heart_disease",
        "model_name": "logistic_regression",
        "model_version": "0.1.0",
        "label": 1,
        "probability_positive": 0.72,
        "probability_negative": 0.28,
        "threshold": 0.55,
        "calibrated": True,
        "positive_class": 1,
        "negative_class": 0,
        "target_meaning": "presence of disease",
        "contributions": [
            {"feature": "age", "contribution": 0.31},
            {"feature": "chol", "contribution": -0.12},
        ],
        "warnings": [],
    }
    defaults.update(overrides)
    return PredictionOutcome(**defaults)


def test_prediction_label_reflects_class():
    assert prediction_label(_outcome(label=1)) == "1"
    assert prediction_label(_outcome(label=0)) == "0"


def test_probability_frame_sums_to_one():
    frame = probability_frame(_outcome())
    assert isinstance(frame, pd.DataFrame)
    assert abs(frame["probability"].sum() - 1.0) < 1e-9


def test_confusion_counts_layout():
    matrix = confusion_counts({"tn": 5, "fp": 2, "fn": 1, "tp": 9})
    assert matrix == [[5, 2], [1, 9]]


def test_metrics_table_contains_expected_rows():
    table = metrics_table({"accuracy": 0.9, "roc_auc": 0.95, "n": 100})
    assert set(table["metric"]) == {"accuracy", "roc_auc", "n"}


def test_contributions_frame_sorted_by_absolute_value():
    frame = contributions_frame(_outcome())
    assert frame is not None
    assert list(frame["feature"])[0] == "age"
    assert frame["abs_contribution"].iloc[0] >= frame["abs_contribution"].iloc[1]


def test_contributions_frame_none_without_contributions():
    assert contributions_frame(_outcome(contributions=None)) is None
