"""Evaluation metrics with explicit handling of edge cases.

These functions are deliberately implemented with vectorised NumPy rather than
calling scikit-learn's metric functions in a loop. Profiling showed that
scikit-learn's per-call input validation dominates bootstrap confidence
intervals and threshold search (thousands of calls), so we compute the metrics
directly. The NumPy results are checked against scikit-learn in the unit tests.

All functions are pure and take ``y_true`` / ``y_score`` arrays, which keeps
them cheap to unit-test and reuse from the API, the notebooks and the CLI.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), handling ties like ``scipy.stats.rankdata``."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return ranks


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (tie-aware)."""
    n = y_true.size
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(y_score)
    sum_pos_ranks = ranks[y_true == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(min(max(auc, 0.0), 1.0))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == y_true.size:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ys = y_true[order].astype(float)
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    prev_recall = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - prev_recall) * precision))
    return float(min(max(ap, 0.0), 1.0))


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute a comprehensive metric set for binary classification.

    ``specificity`` (true-negative rate) is reported alongside sensitivity
    because in a screening context the two error types carry different costs.
    ``roc_auc``/``pr_auc`` fall back to ``nan`` when a split contains a single
    class, rather than raising.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = y_true.size

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0

    return {
        "threshold": float(threshold),
        "n": int(n),
        "accuracy": float((tp + tn) / n) if n else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": _roc_auc(y_true, y_score),
        "pr_auc": _average_precision(y_true, y_score),
        "brier": float(np.mean((y_true - y_score) ** 2)) if n else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "positive_rate": float(y_true.mean()) if n else float("nan"),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "roc_auc",
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a metric.

    Returns ``(nan, nan)`` when the metric cannot be computed for a resample
    (for example a bootstrap draw with a single class).
    """
    if metric not in {"roc_auc", "pr_auc", "accuracy", "f1", "brier"}:
        raise ValueError(f"Unsupported metric for bootstrap: {metric!r}")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))

    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats.append(binary_metrics(yt, ys)[metric])  # type: ignore[arg-type]
    if not stats:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def threshold_for_objective(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    objective: str = "f1",
    min_specificity: float | None = None,
) -> float:
    """Select a decision threshold on *validation* data.

    Objectives:

    - ``f1``: maximise F1.
    - ``youden``: maximise sensitivity + specificity - 1.
    - ``sensitivity_at_specificity``: highest sensitivity subject to a minimum
      specificity (screening-style operating point).

    The caller is responsible for passing validation (never test) data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    grid = np.unique(np.round(y_score, 4))
    if len(grid) < 2:
        return 0.5

    best_threshold, best_score = 0.5, -np.inf
    for thr in grid:
        m = binary_metrics(y_true, y_score, threshold=float(thr))
        if objective == "f1":
            score = m["f1"]
        elif objective == "youden":
            score = m["sensitivity"] + m["specificity"] - 1
        elif objective == "sensitivity_at_specificity":
            if min_specificity is not None and m["specificity"] < min_specificity:
                continue
            score = m["sensitivity"]
        else:
            raise ValueError(f"Unknown objective {objective!r}")
        if score > best_score:
            best_score, best_threshold = score, float(thr)
    return best_threshold
