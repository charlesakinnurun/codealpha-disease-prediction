from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from disease_prediction.evaluation.metrics import (
    binary_metrics,
    bootstrap_ci,
    threshold_for_objective,
)

Y_TRUE = np.array([0, 0, 1, 1])
Y_SCORE = np.array([0.1, 0.4, 0.35, 0.8])


def test_binary_metrics_confusion_and_scores():
    m = binary_metrics(Y_TRUE, Y_SCORE, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 1, 1)
    assert np.isclose(m["accuracy"], 0.75)
    assert np.isclose(m["precision"], 1.0)
    assert np.isclose(m["recall"], 0.5)
    assert np.isclose(m["specificity"], 1.0)
    assert np.isclose(m["roc_auc"], roc_auc_score(Y_TRUE, Y_SCORE))
    assert 0.0 <= m["brier"] <= 1.0


def test_metric_keys_present():
    m = binary_metrics(Y_TRUE, Y_SCORE)
    for key in (
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
    ):
        assert key in m


def test_single_class_returns_nan_auc_without_raising():
    m = binary_metrics(np.array([1, 1, 1]), np.array([0.2, 0.6, 0.9]))
    assert np.isnan(m["roc_auc"])


def test_bootstrap_ci_is_ordered_and_bounded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.1, 200), 0, 1)
    lo, hi = bootstrap_ci(y, score, "roc_auc", n_boot=200, seed=1)
    assert 0.0 <= lo <= hi <= 1.0


def test_threshold_improves_on_a_crafted_case():
    # Well-separated probabilities: default 0.5 is already optimal-ish; the
    # objective must never return a threshold worse than the 0.5 baseline.
    y = np.array([0, 0, 1, 1, 1, 0])
    score = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.15])
    thr = threshold_for_objective(y, score, objective="f1")
    assert binary_metrics(y, score, threshold=thr)["f1"] >= binary_metrics(y, score)["f1"]


def test_numpy_metrics_match_sklearn():
    """Our vectorised metrics must agree with scikit-learn's reference values."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rng = np.random.default_rng(123)
    y = rng.integers(0, 2, 400)
    score = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.2, 400), 0, 1)
    pred = (score >= 0.5).astype(int)
    m = binary_metrics(y, score, threshold=0.5)

    assert np.isclose(m["accuracy"], accuracy_score(y, pred))
    assert np.isclose(m["precision"], precision_score(y, pred, zero_division=0))
    assert np.isclose(m["recall"], recall_score(y, pred, zero_division=0))
    assert np.isclose(m["f1"], f1_score(y, pred, zero_division=0))
    assert np.isclose(m["roc_auc"], roc_auc_score(y, score))
    assert np.isclose(m["brier"], brier_score_loss(y, score))
    assert np.isclose(m["pr_auc"], average_precision_score(y, score), atol=0.02)


def test_unknown_objective_raises():
    try:
        threshold_for_objective(Y_TRUE, Y_SCORE, objective="not_a_metric")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
