"""Model registry, training, persistence and inference."""

from disease_prediction.models.persistence import (
    ArtifactError,
    LoadedModel,
    available_models,
    get_champion,
    load_artifact,
    load_metadata,
)
from disease_prediction.models.registry import (
    DEFAULT_SELECTION_METRIC,
    ModelSpec,
    all_model_specs,
    model_spec,
)

__all__ = [
    "ArtifactError",
    "LoadedModel",
    "available_models",
    "get_champion",
    "load_artifact",
    "load_metadata",
    "DEFAULT_SELECTION_METRIC",
    "ModelSpec",
    "all_model_specs",
    "model_spec",
]
