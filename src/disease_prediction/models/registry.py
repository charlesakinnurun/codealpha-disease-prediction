"""Model registry: estimator factories, search spaces and capabilities.

Every candidate model is declared here with its hyper-parameter search space and
a capability declaration that the explainability layer uses to pick an
appropriate attribution method. Adding a model is a single entry, which keeps
training, evaluation and documentation in sync.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:  # XGBoost is a hard dependency but keep the import defensive for clarity.
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class ModelSpec:
    """A candidate model plus the metadata needed to explain and document it."""

    name: str
    display_name: str
    factory: Callable[..., BaseEstimator]
    param_distributions: dict[str, list] = field(default_factory=dict)
    supports_coefficients: bool = False
    supports_feature_importance: bool = False
    is_tree: bool = False
    is_linear: bool = False
    description: str = ""

    def build(self, seed: int, *, scale_pos_weight: float | None = None) -> BaseEstimator:
        return self.factory(seed=seed, scale_pos_weight=scale_pos_weight)


def _dummy(seed: int, scale_pos_weight: float | None = None) -> BaseEstimator:
    # `prior` gives the majority-class reference: any real model must beat this.
    return DummyClassifier(strategy="prior", random_state=seed)


def _logistic(seed: int, scale_pos_weight: float | None = None) -> BaseEstimator:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )


def _svm(seed: int, scale_pos_weight: float | None = None) -> BaseEstimator:
    # sklearn >= 1.9 deprecates SVC(probability=True); wrap an uncalibrated SVC
    # in CalibratedClassifierCV as recommended upstream. This also gives us
    # genuine (Platt-scaled) probability estimates for the SVM.
    from sklearn.svm import SVC as _SVC

    return CalibratedClassifierCV(
        _SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        method="sigmoid",
        cv=3,
        ensemble=False,
    )


def _random_forest(seed: int, scale_pos_weight: float | None = None) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def _xgboost(seed: int, scale_pos_weight: float | None = None) -> BaseEstimator:
    if XGBClassifier is None:  # pragma: no cover
        raise ImportError("xgboost is required for the XGBoost model.")
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight if scale_pos_weight else 1.0,
    )


_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="dummy",
        display_name="Dummy (prior) baseline",
        factory=_dummy,
        description="Predicts the training prior; establishes the value of any real model.",
    ),
    ModelSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        factory=_logistic,
        is_linear=True,
        supports_coefficients=True,
        param_distributions={"model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]},
        description="Regularised linear model; coefficients are directly interpretable on the transformed scale.",
    ),
    ModelSpec(
        name="svm",
        display_name="Support Vector Machine (RBF)",
        factory=_svm,
        param_distributions={
            "model__estimator__C": [0.1, 1.0, 3.0, 10.0],
            "model__estimator__gamma": ["scale", 0.01, 0.1, 1.0],
        },
        description="RBF-kernel SVM with Platt scaling for probabilities.",
    ),
    ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        factory=_random_forest,
        is_tree=True,
        supports_feature_importance=True,
        param_distributions={
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        description="Bagged decision trees; impurity importance is biased toward high-cardinality features.",
    ),
    ModelSpec(
        name="xgboost",
        display_name="XGBoost",
        factory=_xgboost,
        is_tree=True,
        supports_feature_importance=True,
        param_distributions={
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [2, 3, 4, 6],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
        description="Gradient-boosted trees; strong tabular performer with built-in regularisation.",
    ),
]


def all_model_specs() -> list[ModelSpec]:
    return list(_SPECS)


def model_spec(name: str) -> ModelSpec:
    for spec in _SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model {name!r}. Available: {[s.name for s in _SPECS]}")


def model_specs(names: list[str] | None = None) -> list[ModelSpec]:
    if names is None:
        return list(_SPECS)
    return [model_spec(n) for n in names]


#: Metric used to select the champion model per task.
#: ROC-AUC is threshold-free and robust to class imbalance; PR-AUC and
#: sensitivity-at-fixed-specificity are reported alongside it for transparency.
DEFAULT_SELECTION_METRIC = "roc_auc"
