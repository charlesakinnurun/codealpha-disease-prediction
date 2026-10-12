"""Evaluation figures: ROC, precision-recall, calibration and confusion matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _finish(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(metrics: dict, path: Path, title: str = "Confusion matrix") -> Path:
    matrix = np.array(
        [[metrics.get("tn", 0), metrics.get("fp", 0)], [metrics.get("fn", 0), metrics.get("tp", 0)]]
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    disp = ConfusionMatrixDisplay(matrix, display_labels=["negative", "positive"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    return _finish(fig, path)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, path: Path, title: str = "ROC curve") -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _finish(fig, path)


def plot_pr(
    y_true: np.ndarray, y_score: np.ndarray, path: Path, title: str = "Precision-Recall"
) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, label=f"AP = {average_precision_score(y_true, y_score):.3f}")
        baseline = float(np.mean(y_true))
        ax.axhline(baseline, linestyle="--", color="grey", label=f"prevalence = {baseline:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return _finish(fig, path)


def plot_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    title: str = "Calibration curve",
    n_bins: int = 8,
) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if len(np.unique(y_true)) == 2:
        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_score, n_bins=n_bins, strategy="quantile"
            )
            ax.plot(prob_pred, prob_true, marker="o", label="model")
        except ValueError:
            pass
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _finish(fig, path)


def plot_importance(features: list[str], values: list[float], path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, max(3, 0.28 * len(features))))
    order = np.argsort(values)
    ax.barh(np.array(features)[order], np.array(values)[order], color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel("importance")
    return _finish(fig, path)
