"""Training, model selection, calibration and artifact creation.

Protocol (per task), designed to keep the test set untouched until the very end:

1. Stratified split into **train / validation / test** (test held out).
2. Hyper-parameter search with **stratified K-fold CV on train only**.
3. Refit the best configuration on train, evaluate on **validation**.
4. Select the champion by validation ROC-AUC (documented, threshold-free).
5. Fit a probability **calibrator on validation**, choose the decision
   **threshold on validation** for a documented objective.
6. Report final, honest metrics on the **untouched test set**.

Every model that trains successfully is persisted so the API and dashboard can
compare them; the champion is recorded in ``champion.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from disease_prediction import __version__
from disease_prediction.config import Settings, get_settings
from disease_prediction.data.fingerprint import dataset_fingerprint
from disease_prediction.data.loaders import load_dataset
from disease_prediction.data.registry import get_schema
from disease_prediction.data.schemas import DatasetSchema
from disease_prediction.evaluation.metrics import (
    binary_metrics,
    bootstrap_ci,
    threshold_for_objective,
)
from disease_prediction.features.pipelines import build_pipeline
from disease_prediction.features.schema import InferenceFeature, InferenceSchema
from disease_prediction.logging_config import get_logger
from disease_prediction.models.metadata import ModelArtifactMetadata
from disease_prediction.models.persistence import save_artifact, set_champion
from disease_prediction.models.registry import ModelSpec, model_specs

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for one training run."""

    task: str
    models: list[str] | None = None
    test_size: float = 0.2
    validation_size: float = 0.25  # fraction of the non-test data
    cv_folds: int = 5
    n_iter: int = 12
    scoring: str = "roc_auc"
    selection_metric: str = "roc_auc"
    threshold_objective: str = "youden"
    calibration_method: str | None = "sigmoid"
    seed: int = 42
    save: bool = True


@dataclass
class ModelResult:
    name: str
    display_name: str
    best_params: dict[str, Any]
    cv_score: float
    cv_std: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5
    calibrated: bool = False
    pipeline: Pipeline | None = None  # not serialised in reports


@dataclass
class TrainingResult:
    task: str
    selection_metric: str
    champion: str | None
    results: dict[str, ModelResult]
    rationale: str
    split_sizes: dict[str, int]
    random_seed: int

    def comparison_table(self) -> list[dict[str, Any]]:
        """Flat table of validation metrics, ordered by the selection metric."""
        rows: list[dict[str, Any]] = []
        for r in self.results.values():
            m = r.validation_metrics
            rows.append(
                {
                    "model": r.name,
                    "display_name": r.display_name,
                    "cv_roc_auc": round(r.cv_score, 4),
                    "cv_std": round(r.cv_std, 4),
                    "val_roc_auc": round(float(m.get("roc_auc", float("nan"))), 4),
                    "val_pr_auc": round(float(m.get("pr_auc", float("nan"))), 4),
                    "val_f1": round(float(m.get("f1", float("nan"))), 4),
                    "val_sensitivity": round(float(m.get("sensitivity", float("nan"))), 4),
                    "val_specificity": round(float(m.get("specificity", float("nan"))), 4),
                    "val_brier": round(float(m.get("brier", float("nan"))), 4),
                    "threshold": round(r.threshold, 4),
                    "calibrated": r.calibrated,
                }
            )
        rows.sort(key=lambda row: row[f"val_{self.selection_metric}"], reverse=True)
        return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _positive_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(getattr(model, "classes_", [0, 1]))
    if len(classes) == 1:
        return np.full(len(x), float(classes[0]))
    return proba[:, classes.index(1)]


def _make_calibrator(estimator: Any, method: str) -> Any:
    """Return a prefit calibrator compatible with several scikit-learn versions."""
    try:  # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except (ImportError, TypeError):
        return CalibratedClassifierCV(estimator, cv="prefit", method=method)


def build_inference_schema(
    schema: DatasetSchema,
    train_frame: pd.DataFrame,
) -> InferenceSchema:
    """Create the inference contract, using training-only observed ranges."""
    features: list[InferenceFeature] = []
    for spec in schema.features:
        column = train_frame[spec.name]
        observed_min = observed_max = None
        if spec.kind.value == "numeric" and column.notna().any():
            observed_min = float(column.min())
            observed_max = float(column.max())
        features.append(
            InferenceFeature(
                name=spec.name,
                kind=spec.kind.value,
                unit=spec.unit,
                allowed_values=list(spec.allowed_values) if spec.allowed_values else None,
                minimum=spec.minimum,
                maximum=spec.maximum,
                observed_min=observed_min,
                observed_max=observed_max,
            )
        )
    return InferenceSchema(
        task=schema.task,
        features=features,
        target_column=schema.target_column,
        classes=[0, 1],
        positive_class=schema.positive_class,
        target_meaning=schema.target_meaning,
    )


def split_data(
    df: pd.DataFrame, config: TrainingConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split with a fixed seed."""
    train_val, test = train_test_split(
        df,
        test_size=config.test_size,
        stratify=df["target"],
        random_state=config.seed,
    )
    train, validation = train_test_split(
        train_val,
        test_size=config.validation_size,
        stratify=train_val["target"],
        random_state=config.seed,
    )
    return train, validation, test


def _fit_single_model(
    spec: ModelSpec,
    config: TrainingConfig,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    schema: DatasetSchema,
    scale_pos_weight: float | None,
) -> tuple[Pipeline, dict[str, Any], float, float]:
    """Fit one model with CV search (or directly for the dummy baseline)."""
    estimator = spec.build(config.seed, scale_pos_weight=scale_pos_weight)
    pipeline = build_pipeline(estimator, schema)

    if spec.name == "dummy" or not spec.param_distributions:
        pipeline.fit(x_train, y_train)
        return pipeline, {}, float("nan"), float("nan")

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=spec.param_distributions,
        n_iter=min(config.n_iter, _grid_size(spec.param_distributions)),
        scoring=config.scoring,
        cv=cv,
        random_state=config.seed,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    best_index = search.best_index_
    std = float(search.cv_results_["std_test_score"][best_index])
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_), std


def _grid_size(grid: dict[str, list]) -> int:
    size = 1
    for values in grid.values():
        size *= max(len(values), 1)
    return size


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def train_task(
    config: TrainingConfig,
    settings: Settings | None = None,
    df: pd.DataFrame | None = None,
) -> TrainingResult:
    """Run the full protocol for one task and (optionally) persist artifacts."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    schema = get_schema(config.task)
    frame = load_dataset(config.task, settings) if df is None else df
    fingerprint = dataset_fingerprint(config.task, settings) if df is None else None

    feature_names = schema.feature_names
    train, validation, test = split_data(frame, config)
    x_train, y_train = train[feature_names], train["target"].to_numpy()
    x_val, y_val = validation[feature_names], validation["target"].to_numpy()
    x_test, y_test = test[feature_names], test["target"].to_numpy()

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = (neg / pos) if pos else None

    specs = model_specs(config.models)
    results: dict[str, ModelResult] = {}
    best_params_by_model: dict[str, dict[str, Any]] = {}

    for spec in specs:
        logger.info("training task=%s model=%s", config.task, spec.name)
        pipeline, best_params, cv_score, cv_std = _fit_single_model(
            spec, config, x_train, y_train, schema, scale_pos_weight
        )
        best_params_by_model[spec.name] = best_params

        val_proba = _positive_proba(pipeline, x_val)
        threshold = (
            0.5
            if spec.name == "dummy"
            else threshold_for_objective(y_val, val_proba, objective=config.threshold_objective)
        )
        val_metrics = binary_metrics(y_val, val_proba, threshold=threshold)
        results[spec.name] = ModelResult(
            name=spec.name,
            display_name=spec.display_name,
            best_params=best_params,
            cv_score=cv_score,
            cv_std=cv_std,
            validation_metrics=val_metrics,
            threshold=threshold,
            pipeline=pipeline,
        )
        logger.info(
            "trained task=%s model=%s val_roc_auc=%.4f val_f1=%.4f",
            config.task,
            spec.name,
            val_metrics["roc_auc"],
            val_metrics["f1"],
        )

    # --- Champion selection (validation only) ----------------------------
    real_models = {k: v for k, v in results.items() if k != "dummy"}
    pool = real_models or results
    champion_name = max(
        pool,
        key=lambda k: (
            pool[k].validation_metrics.get(config.selection_metric, float("-inf")),
            pool[k].cv_score if np.isfinite(pool[k].cv_score) else float("-inf"),
        ),
    )
    rationale = _build_rationale(config, results, champion_name, pool)

    # --- Calibration of the champion (fit on validation) -----------------
    champion = results[champion_name]
    if config.calibration_method and champion.pipeline is not None and champion_name != "dummy":
        try:
            calibrator = _make_calibrator(champion.pipeline, config.calibration_method)
            calibrator.fit(x_val, y_val)
            champion.pipeline = calibrator  # type: ignore[assignment]
            champion.calibrated = True
        except Exception as exc:  # noqa: BLE001 - calibration is best-effort
            logger.warning(
                "calibration failed task=%s model=%s err=%s", config.task, champion_name, exc
            )

    # --- Threshold + final honest test evaluation ------------------------
    champion_val_proba = _positive_proba(champion.pipeline, x_val)
    champion.threshold = (
        0.5
        if champion_name == "dummy"
        else threshold_for_objective(
            y_val, champion_val_proba, objective=config.threshold_objective
        )
    )
    champion.validation_metrics = binary_metrics(
        y_val, champion_val_proba, threshold=champion.threshold
    )

    inference_schema = build_inference_schema(schema, train)

    for result in results.values():
        if result.pipeline is None:
            continue
        test_proba = _positive_proba(result.pipeline, x_test)
        result.test_metrics = binary_metrics(y_test, test_proba, threshold=result.threshold)
        result.test_metrics["roc_auc_ci95"] = list(
            bootstrap_ci(y_test, test_proba, "roc_auc", seed=config.seed)
        )
        result.test_metrics["pr_auc_ci95"] = list(
            bootstrap_ci(y_test, test_proba, "pr_auc", seed=config.seed)
        )

    # Champion test metrics after calibration + threshold, taken from the loop above
    if config.save:
        _persist_all(
            config=config,
            schema=schema,
            results=results,
            best_params_by_model=best_params_by_model,
            inference_schema=inference_schema,
            champion_name=champion_name,
            rationale=rationale,
            fingerprint=fingerprint,
            settings=settings,
        )

    return TrainingResult(
        task=config.task,
        selection_metric=config.selection_metric,
        champion=champion_name,
        results=results,
        rationale=rationale,
        split_sizes={
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        random_seed=config.seed,
    )


def _build_rationale(
    config: TrainingConfig,
    results: dict[str, ModelResult],
    champion: str,
    pool: dict[str, ModelResult],
) -> str:
    ranked = sorted(
        pool.items(),
        key=lambda kv: kv[1].validation_metrics.get(config.selection_metric, float("-inf")),
        reverse=True,
    )
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    dummy = results.get("dummy")
    parts = [
        f"Selected `{champion}` by highest validation {config.selection_metric}.",
        "Selection used validation data only; the test set was not consulted.",
    ]
    if runner_up:
        delta = ranked[0][1].validation_metrics.get(config.selection_metric, 0) - ranked[1][
            1
        ].validation_metrics.get(config.selection_metric, 0)
        parts.append(
            f"Runner-up was `{runner_up}` (Δ{config.selection_metric} = {delta:.4f}); "
            "the margin should be read alongside cross-validation variability."
        )
    if dummy is not None:
        dummy_score = dummy.validation_metrics.get(config.selection_metric, float("nan"))
        champion_score = pool[champion].validation_metrics.get(
            config.selection_metric, float("nan")
        )
        parts.append(
            f"Dummy baseline {config.selection_metric} = {dummy_score:.4f}; "
            f"champion = {champion_score:.4f}."
        )
    return " ".join(parts)


def _persist_all(
    *,
    config: TrainingConfig,
    schema: DatasetSchema,
    results: dict[str, ModelResult],
    best_params_by_model: dict[str, dict[str, Any]],
    inference_schema: InferenceSchema,
    champion_name: str,
    rationale: str,
    fingerprint: str | None,
    settings: Settings,
) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    for name, result in results.items():
        if result.pipeline is None:
            continue
        is_champion = name == champion_name
        metadata = ModelArtifactMetadata(
            task=config.task,
            model_name=name,
            model_version=f"{__version__}.{stamp}",
            display_name=result.display_name,
            dataset_id=schema.display_name,
            dataset_source=schema.source_url,
            dataset_license=schema.license,
            dataset_fingerprint=fingerprint,
            random_seed=config.seed,
            training_config={
                "test_size": config.test_size,
                "validation_size": config.validation_size,
                "cv_folds": config.cv_folds,
                "n_iter": config.n_iter,
                "scoring": config.scoring,
                "selection_metric": config.selection_metric,
                "threshold_objective": config.threshold_objective,
                "calibration_method": config.calibration_method,
            },
            best_params=best_params_by_model.get(name, {}),
            cv_metric=config.scoring,
            cv_score=None if not np.isfinite(result.cv_score) else float(result.cv_score),
            validation_metrics=result.validation_metrics,
            test_metrics=result.test_metrics,
            threshold=result.threshold,
            calibrated=result.calibrated,
            calibration_method=config.calibration_method if result.calibrated else None,
            feature_schema=inference_schema,
            target_mapping={
                "positive_class": schema.positive_class,
                "negative_class": schema.negative_class,
                "meaning": schema.target_meaning,
            },
            selection_rationale=rationale if is_champion else "Not selected as champion.",
            environment=ModelArtifactMetadata.capture_environment(),
            notes=schema.notes,
        )
        save_artifact(result.pipeline, metadata, settings)
    set_champion(
        config.task,
        champion_name,
        settings,
        rationale=rationale,
        model_version=f"{__version__}.{stamp}",
    )
