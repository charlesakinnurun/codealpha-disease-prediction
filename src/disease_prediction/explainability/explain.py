"""Model explainability.

Four complementary techniques, chosen automatically from the estimator's
capabilities:

- **Coefficients** for regularised linear models (on the *transformed* scale).
- **Impurity feature importance** for tree ensembles.
- **Permutation importance** as a model-agnostic, held-out-set attribution.
- **SHAP** for local and global additive attributions where feasible.

Important caveats, echoed in the docs and dashboard:

- Feature importance is **not** evidence of causation, nor a clinical finding.
- Correlated features (e.g. the mean/error/worst triplets in WDBC) split and
  destabilise attribution; rankings can flip between correlated inputs.
- Coefficients apply to standardised/one-hot-encoded features, not raw units.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from disease_prediction.features.pipelines import transformed_feature_names
from disease_prediction.logging_config import get_logger

logger = get_logger(__name__)

CAVEAT = (
    "Explanations describe the model's internal behaviour on benchmark data. "
    "They are not medical reasoning, do not establish causation, and can be "
    "unstable when input features are correlated."
)


class ExplanationError(RuntimeError):
    """Raised when no explanation method applies to a model."""


def unwrap_base_pipeline(model: Any) -> Pipeline:
    """Return the underlying preprocessing+estimator pipeline.

    Calibrated models wrap the original pipeline (possibly via a
    ``FrozenEstimator``); explanations should be computed on the base estimator
    because the calibration layer is monotonic and does not change feature
    attributions.
    """
    current = model
    for _ in range(5):
        if isinstance(current, CalibratedClassifierCV):
            base = getattr(current, "estimator", None)
            if base is None:  # older scikit-learn
                base = current.calibrated_classifiers_[0].estimator
            current = base
            continue
        if type(current).__name__ == "FrozenEstimator" and hasattr(current, "estimator"):
            current = current.estimator
            continue
        break
    return current


def _base_model(pipeline: Pipeline) -> Any:
    return pipeline.named_steps["model"]


def linear_coefficients(model: Any) -> pd.DataFrame:
    """Coefficient table for a linear model, on the transformed feature scale."""
    pipeline = unwrap_base_pipeline(model)
    estimator = _base_model(pipeline)
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        raise ExplanationError("Estimator has no coefficients.")
    coef = np.asarray(coef).ravel()
    names = transformed_feature_names(pipeline)
    if len(names) != len(coef):
        names = [f"f{i}" for i in range(len(coef))]
    df = pd.DataFrame({"feature": names, "coefficient": coef})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def tree_feature_importance(model: Any) -> pd.DataFrame:
    """Impurity-based feature importance for tree ensembles."""
    pipeline = unwrap_base_pipeline(model)
    estimator = _base_model(pipeline)
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None:
        raise ExplanationError("Estimator has no feature_importances_.")
    names = transformed_feature_names(pipeline)
    if len(names) != len(importance):
        names = [f"f{i}" for i in range(len(importance))]
    df = pd.DataFrame({"feature": names, "importance": np.asarray(importance)})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance_scores(
    model: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 10,
    seed: int = 42,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Model-agnostic permutation importance on the *original* feature scale."""
    result = permutation_importance(
        model,
        x,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": list(x.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _shap_explainer(model: Any, background: pd.DataFrame | None = None):
    """Build a SHAP explainer for the base estimator, or return ``None``."""
    try:
        import shap
    except ImportError:  # pragma: no cover
        return None

    pipeline = unwrap_base_pipeline(model)
    estimator = _base_model(pipeline)
    pre = pipeline.named_steps["preprocessor"]

    class _Transformed:
        """Small wrapper so SHAP sees transformed features but predicts probabilities."""

        def __init__(self) -> None:
            self._pre = pre
            self._est = estimator

        def __call__(self, data: np.ndarray) -> np.ndarray:
            return self._est.predict_proba(data)

        @property
        def feature_names(self) -> list[str]:
            return list(pre.get_feature_names_out())

    wrapped = _Transformed()
    try:
        if hasattr(estimator, "feature_importances_"):
            return shap.TreeExplainer(estimator), wrapped
        if hasattr(estimator, "coef_"):
            if background is None:
                return None
            bg = pre.transform(background)
            return shap.LinearExplainer(estimator, bg), wrapped
        # Generic fallback with a small background sample.
        if background is not None:
            return shap.Explainer(wrapped, pre.transform(background)), wrapped
    except Exception as exc:  # noqa: BLE001 - SHAP is best-effort
        logger.warning("SHAP explainer construction failed: %s", exc)
        return None
    return None


def shap_explanations(
    model: Any,
    x: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    max_rows: int = 200,
) -> pd.DataFrame | None:
    """Return mean |SHAP| per transformed feature, or ``None`` if unavailable."""
    built = _shap_explainer(model, background)
    if built is None:
        return None
    explainer, wrapped = built
    pipeline = unwrap_base_pipeline(model)
    sample = x.head(max_rows)
    transformed = pipeline.named_steps["preprocessor"].transform(sample)
    try:
        values = explainer.shap_values(transformed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP value computation failed: %s", exc)
        return None

    values = _collapse_shap(values)
    if values is None:
        return None
    mean_abs = np.abs(values).mean(axis=0)
    names = wrapped.feature_names
    if len(names) != len(mean_abs):
        names = [f"f{i}" for i in range(len(mean_abs))]
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _collapse_shap(values: Any) -> np.ndarray | None:
    """Normalise SHAP output across versions to a 2-D array of shape (n, features)."""
    if values is None:
        return None
    if isinstance(values, list):  # older API: list per class
        values = values[-1] if len(values) > 1 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:  # (n, features, classes)
        arr = arr[..., -1]
    if arr.ndim != 2:
        return None
    return arr


def global_explanations(
    model: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int = 42,
    n_repeats: int = 10,
    background: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute every explanation that applies, keyed by method name."""
    out: dict[str, pd.DataFrame] = {}
    out["permutation"] = permutation_importance_scores(model, x, y, n_repeats=n_repeats, seed=seed)

    pipeline = unwrap_base_pipeline(model)
    estimator = _base_model(pipeline)
    if hasattr(estimator, "coef_"):
        with contextlib.suppress(ExplanationError):
            out["coefficients"] = linear_coefficients(model)
    if hasattr(estimator, "feature_importances_"):
        with contextlib.suppress(ExplanationError):
            out["tree_importance"] = tree_feature_importance(model)

    shap_df = shap_explanations(model, x, background=background)
    if shap_df is not None:
        out["shap"] = shap_df
    return out


def local_explanation(
    model: Any,
    row: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Return per-feature contributions for a single row, or ``None``.

    Uses SHAP when available; otherwise falls back to a linear
    ``coefficient × transformed_value`` contribution for linear models.
    """
    pipeline = unwrap_base_pipeline(model)
    estimator = _base_model(pipeline)
    pre = pipeline.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())

    built = _shap_explainer(model, background)
    if built is not None:
        explainer, _ = built
        try:
            transformed = pre.transform(row)
            values = _collapse_shap(explainer.shap_values(transformed))
            if values is not None:
                contributions = values[0]
                return (
                    pd.DataFrame({"feature": names, "contribution": contributions})
                    .assign(abs_contribution=lambda d: d["contribution"].abs())
                    .sort_values("abs_contribution", ascending=False)
                    .reset_index(drop=True)
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local SHAP failed: %s", exc)

    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        transformed = np.asarray(pre.transform(row)).ravel()
        contributions = np.asarray(coef).ravel() * transformed
        return (
            pd.DataFrame({"feature": names, "contribution": contributions})
            .assign(abs_contribution=lambda d: d["contribution"].abs())
            .sort_values("abs_contribution", ascending=False)
            .reset_index(drop=True)
        )
    return None
