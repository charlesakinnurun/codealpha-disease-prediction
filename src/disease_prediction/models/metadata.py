"""Model artifact metadata — the contract stored alongside every model."""

from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from disease_prediction.features.schema import InferenceSchema

ARTIFACT_VERSION = "1"


class ModelArtifactMetadata(BaseModel):
    """Everything needed to reproduce, audit and safely serve a model."""

    artifact_version: str = ARTIFACT_VERSION
    task: str
    model_name: str
    model_version: str
    display_name: str

    # Provenance
    dataset_id: str
    dataset_source: str
    dataset_license: str
    dataset_fingerprint: str | None = None
    trained_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    package_version: str = "0.1.0"
    random_seed: int = 42

    # Training
    training_config: dict[str, Any] = Field(default_factory=dict)
    best_params: dict[str, Any] = Field(default_factory=dict)
    cv_metric: str = "roc_auc"
    cv_score: float | None = None

    # Evaluation
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)
    threshold: float = 0.5
    calibrated: bool = False
    calibration_method: str | None = None

    # Contract
    feature_schema: InferenceSchema
    target_mapping: dict[str, Any] = Field(default_factory=dict)

    selection_rationale: str = ""
    environment: dict[str, str] = Field(default_factory=dict)
    notes: str = ""

    @staticmethod
    def capture_environment() -> dict[str, str]:
        import sklearn
        import xgboost

        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        }
