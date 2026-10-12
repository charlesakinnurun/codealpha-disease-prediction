"""Reusable EDA plotting.

Every figure answers a specific question (class balance, missingness, feature
distributions, redundancy). Figures are written to ``reports/figures/``. Uses
matplotlib's non-interactive ``Agg`` backend so it runs headless in CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - must precede pyplot import
import logging as _logging  # noqa: E402

_logging.getLogger("matplotlib").setLevel(_logging.WARNING)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from disease_prediction.data.schemas import DatasetSchema  # noqa: E402


def _finish(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_distribution(df: pd.DataFrame, schema: DatasetSchema, path: Path) -> Path:
    counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar([str(i) for i in counts.index], counts.values, color=["#4c72b0", "#c44e52"])
    ax.set_title(f"{schema.display_name}: class balance")
    ax.set_xlabel(f"target (1 = {schema.target_meaning})")
    ax.set_ylabel("count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    return _finish(fig, path)


def plot_missingness(df: pd.DataFrame, schema: DatasetSchema, path: Path) -> Path:
    cols = [c for c in schema.feature_names if c in df.columns]
    pct = (df[cols].isna().mean() * 100).sort_values(ascending=False)
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if pct.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(pct.index[::-1], pct.values[::-1], color="#dd8452")
        ax.set_xlabel("% missing")
        ax.set_title(f"{schema.display_name}: missingness")
    return _finish(fig, path)


def plot_numeric_distributions(df: pd.DataFrame, schema: DatasetSchema, path: Path) -> Path:
    numeric = [c for c in schema.numeric_features() if c in df.columns]
    numeric = numeric[:12]  # keep the figure readable
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    axes = axes.ravel()
    for ax, col in zip(axes, numeric, strict=False):
        ax.hist(df[col].dropna(), bins=25, color="#4c72b0", alpha=0.85)
        ax.set_title(col, fontsize=9)
    for ax in axes[len(numeric) :]:
        ax.axis("off")
    fig.suptitle(f"{schema.display_name}: numeric feature distributions", y=1.01)
    return _finish(fig, path)


def plot_correlation(df: pd.DataFrame, schema: DatasetSchema, path: Path) -> Path:
    numeric = [c for c in schema.numeric_features() if c in df.columns][:20]
    if len(numeric) < 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Not enough numeric features", ha="center", va="center")
        ax.axis("off")
        return _finish(fig, path)
    corr = df[numeric].corr()
    fig, ax = plt.subplots(figsize=(0.45 * len(numeric) + 3, 0.45 * len(numeric) + 2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric)))
    ax.set_xticklabels(numeric, rotation=90, fontsize=7)
    ax.set_yticks(range(len(numeric)))
    ax.set_yticklabels(numeric, fontsize=7)
    ax.set_title(f"{schema.display_name}: feature correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _finish(fig, path)


def plot_feature_target_relationship(df: pd.DataFrame, schema: DatasetSchema, path: Path) -> Path:
    """Compare the strongest numeric feature between target classes."""
    numeric = [c for c in schema.numeric_features() if c in df.columns]
    if not numeric:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis("off")
        return _finish(fig, path)
    target = df["target"]
    best, best_gap = numeric[0], -np.inf
    for col in numeric:
        gap = abs(df[col].corr(target))
        if np.isfinite(gap) and gap > best_gap:
            best, best_gap = col, gap
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for label, color in ((0, "#4c72b0"), (1, "#c44e52")):
        ax.hist(
            df.loc[target == label, best].dropna(),
            bins=25,
            alpha=0.6,
            label=f"target={label}",
            color=color,
            density=True,
        )
    ax.set_title(f"{schema.display_name}: {best} by target (|r|={best_gap:.2f})")
    ax.set_xlabel(best)
    ax.legend()
    return _finish(fig, path)


def generate_eda_figures(
    task: str,
    df: pd.DataFrame,
    schema: DatasetSchema,
    figures_dir: Path,
) -> list[Path]:
    """Generate the standard EDA figure set for a task."""
    task_dir = figures_dir / task
    return [
        plot_target_distribution(df, schema, task_dir / "target_distribution.png"),
        plot_missingness(df, schema, task_dir / "missingness.png"),
        plot_numeric_distributions(df, schema, task_dir / "numeric_distributions.png"),
        plot_correlation(df, schema, task_dir / "correlation.png"),
        plot_feature_target_relationship(df, schema, task_dir / "top_feature_by_target.png"),
    ]
