from __future__ import annotations

import json

import numpy as np
import pytest

from disease_prediction.data.registry import get_schema
from disease_prediction.features.pipelines import build_pipeline
from disease_prediction.models.metadata import ModelArtifactMetadata
from disease_prediction.models.persistence import (
    ArtifactError,
    artifact_paths,
    available_models,
    load_artifact,
    save_artifact,
    set_champion,
)
from disease_prediction.models.registry import model_spec
from disease_prediction.models.train import build_inference_schema
from tests.conftest import make_synthetic_frame


def _fitted_artifact(settings, model_name: str = "logistic_regression"):
    schema = get_schema("heart_disease")
    frame = make_synthetic_frame("heart_disease", n=80, seed=5)
    x, y = frame[schema.feature_names], frame["target"].to_numpy()
    pipeline = build_pipeline(model_spec(model_name).build(11), schema)
    pipeline.fit(x, y)
    metadata = ModelArtifactMetadata(
        task="heart_disease",
        model_name=model_name,
        model_version="test.1",
        display_name="Test model",
        dataset_id=schema.display_name,
        dataset_source=schema.source_url,
        dataset_license=schema.license,
        feature_schema=build_inference_schema(schema, frame),
        target_mapping={"positive_class": 1, "negative_class": 0, "meaning": schema.target_meaning},
    )
    save_artifact(pipeline, metadata, settings)
    return pipeline, x, metadata


def test_save_and_load_roundtrip_predictions_match(settings):
    pipeline, x, _ = _fitted_artifact(settings)
    loaded = load_artifact("heart_disease", "logistic_regression", settings)
    assert np.allclose(pipeline.predict_proba(x), loaded.pipeline.predict_proba(x))
    assert loaded.metadata.model_version == "test.1"


def test_missing_artifact_raises_clear_error(settings):
    with pytest.raises(ArtifactError):
        load_artifact("heart_disease", "does_not_exist", settings)


def test_incompatible_artifact_version_is_rejected(settings):
    _fitted_artifact(settings)
    _, meta_path = artifact_paths("heart_disease", "logistic_regression", settings)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["artifact_version"] = "999"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_artifact("heart_disease", "logistic_regression", settings)


def test_available_models_and_champion_tracking(settings):
    _fitted_artifact(settings)
    assert "logistic_regression" in available_models("heart_disease", settings)
    set_champion("heart_disease", "logistic_regression", settings)
    from disease_prediction.models.persistence import get_champion

    assert get_champion("heart_disease", settings) == "logistic_regression"


def test_metadata_captures_environment(settings):
    _, _, metadata = _fitted_artifact(settings)
    assert metadata.random_seed == 42
    env = ModelArtifactMetadata.capture_environment()
    assert "scikit_learn" in env and "python" in env
