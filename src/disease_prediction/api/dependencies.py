"""Model loading and prediction business logic.

Kept deliberately free of FastAPI imports so it can be unit-tested directly and
reused by the Streamlit dashboard. Models are loaded once and cached; artifacts
are validated before use and never loaded from untrusted sources.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from disease_prediction.config import Settings, get_settings
from disease_prediction.explainability.explain import local_explanation
from disease_prediction.logging_config import get_logger
from disease_prediction.models.persistence import (
    ArtifactError,
    LoadedModel,
    get_champion,
    load_artifact,
)

logger = get_logger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when a task has no trained, servable model artifact."""


@dataclass
class PredictionOutcome:
    """The result of running a single prediction through the service."""

    task: str
    model_name: str
    model_version: str
    label: int
    probability_positive: float
    probability_negative: float
    threshold: float
    calibrated: bool
    positive_class: int | str
    negative_class: int | str
    target_meaning: str
    contributions: list[dict[str, Any]] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction": self.label,
            "probability_positive": self.probability_positive,
            "probability_negative": self.probability_negative,
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "contributions": self.contributions,
            "warnings": self.warnings,
        }


class ModelService:
    """Loads, caches and serves model artifacts for every task."""

    def __init__(self, settings: Settings | None = None, max_cache: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.max_cache = max_cache or self.settings.model_cache_size
        self._cache: dict[tuple[str, str], LoadedModel] = {}
        self._lock = threading.Lock()

    # -- loading ----------------------------------------------------------
    def load(self, task: str, model_name: str | None = None, *, force: bool = False) -> LoadedModel:
        """Load (and cache) the requested model, defaulting to the champion."""
        name = model_name or get_champion(task, self.settings)
        if name is None:
            raise ModelUnavailableError(
                f"No trained model is registered for task '{task}'. "
                "Run `python scripts/train.py --task {task}`."
            )
        key = (task, name)
        with self._lock:
            if not force and key in self._cache:
                return self._cache[key]
            loaded = load_artifact(task, name, self.settings)
            if len(self._cache) >= self.max_cache:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = loaded
            return loaded

    def is_ready(self, task: str) -> bool:
        try:
            self.load(task)
            return True
        except (ArtifactError, ModelUnavailableError):
            return False

    def ready_tasks(self) -> tuple[list[str], list[str]]:
        from disease_prediction.config import TASKS

        ready = [t for t in TASKS if self.is_ready(t)]
        missing = [t for t in TASKS if t not in ready]
        return ready, missing

    # -- prediction -------------------------------------------------------
    def predict(
        self,
        task: str,
        features: dict[str, Any],
        *,
        explain: bool = False,
        model_name: str | None = None,
    ) -> PredictionOutcome:
        """Run a validated prediction for *task*."""
        loaded = self.load(task, model_name)
        schema = loaded.metadata.feature_schema

        require_all = not self.settings.allow_optional_imputation
        problems = schema.validate_row(features, require_all=require_all)
        if problems:
            raise ValueError("; ".join(problems))

        # Build a single-row frame in the exact training column order.
        row = {name: features.get(name) for name in schema.feature_names}
        frame = pd.DataFrame([row], columns=schema.feature_names)

        proba = loaded.pipeline.predict_proba(frame)[0]
        classes = list(getattr(loaded.pipeline, "classes_", [0, 1]))
        pos_index = classes.index(1)
        proba_pos = float(proba[pos_index])
        proba_neg = float(proba[classes.index(0)])
        label = int(proba_pos >= loaded.threshold)

        warnings: list[str] = []
        contributions = None
        if explain:
            contributions, warn = self._explain(loaded, frame)
            if warn:
                warnings.append(warn)

        if loaded.metadata.calibrated is False:
            warnings.append(
                "Predicted probabilities are not calibrated for this model; "
                "treat them as relative scores, not absolute risks."
            )

        return PredictionOutcome(
            task=task,
            model_name=loaded.metadata.model_name,
            model_version=loaded.model_version,
            label=label,
            probability_positive=proba_pos,
            probability_negative=proba_neg,
            threshold=loaded.threshold,
            calibrated=bool(loaded.metadata.calibrated),
            positive_class=loaded.metadata.target_mapping.get("positive_class", 1),
            negative_class=loaded.metadata.target_mapping.get("negative_class", 0),
            target_meaning=loaded.metadata.target_mapping.get("meaning", ""),
            contributions=contributions,
            warnings=warnings,
        )

    def _explain(
        self, loaded: LoadedModel, frame: pd.DataFrame
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        try:
            table = local_explanation(loaded.pipeline, frame)
        except Exception as exc:  # noqa: BLE001 - explanation must never break serving
            logger.warning("explanation failed task=%s err=%s", loaded.task, exc)
            return None, "Feature explanation could not be computed for this model."
        if table is None:
            return None, "No explanation method applies to this model."
        top = table.head(10)
        return (
            [
                {"feature": str(r.feature), "contribution": round(float(r.contribution), 5)}
                for r in top.itertuples()
            ],
            None,
        )
