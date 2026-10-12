"""Model artifact persistence and safe loading.

Layout::

    artifacts/models/<task>/
        <model_name>.joblib        # fitted preprocessing+estimator pipeline
        <model_name>.meta.json     # ModelArtifactMetadata
        champion.json              # which model is currently served

Artifacts are produced locally by ``scripts/train.py``. **Never** load a
``.joblib`` file from an untrusted source: joblib deserialises arbitrary Python
objects. Loading therefore checks that a matching, schema-valid metadata file
exists next to the artifact, and rejects artifacts whose ``artifact_version`` or
feature schema is incompatible with this code base.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from disease_prediction.config import Settings, get_settings
from disease_prediction.logging_config import get_logger
from disease_prediction.models.metadata import ARTIFACT_VERSION, ModelArtifactMetadata

logger = get_logger(__name__)

CHAMPION_FILENAME = "champion.json"


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, incompatible or unsafe to load."""


@dataclass
class LoadedModel:
    """A fitted pipeline bundled with its validated metadata."""

    pipeline: Any
    metadata: ModelArtifactMetadata
    path: Path
    model_version: str

    @property
    def task(self) -> str:
        return self.metadata.task

    @property
    def threshold(self) -> float:
        return self.metadata.threshold


def model_dir(task: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.models_dir / task


def artifact_paths(
    task: str, model_name: str, settings: Settings | None = None
) -> tuple[Path, Path]:
    base = model_dir(task, settings) / model_name
    return base.with_suffix(".joblib"), base.with_suffix(".meta.json")


def save_artifact(
    pipeline: Any,
    metadata: ModelArtifactMetadata,
    settings: Settings | None = None,
) -> tuple[Path, Path]:
    """Persist a fitted pipeline and its metadata, returning both paths."""
    settings = settings or get_settings()
    directory = model_dir(metadata.task, settings)
    directory.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = artifact_paths(metadata.task, metadata.model_name, settings)

    joblib.dump(pipeline, model_path)
    meta_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "saved artifact task=%s model=%s path=%s", metadata.task, metadata.model_name, model_path
    )
    return model_path, meta_path


def load_metadata(
    task: str, model_name: str, settings: Settings | None = None
) -> ModelArtifactMetadata:
    """Load and version-check the metadata for an artifact."""
    _, meta_path = artifact_paths(task, model_name, settings)
    if not meta_path.exists():
        raise ArtifactError(f"No metadata found at {meta_path}. Train the model first.")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = ModelArtifactMetadata.model_validate(raw)
    if metadata.artifact_version != ARTIFACT_VERSION:
        raise ArtifactError(
            f"Artifact {task}/{model_name} has version {metadata.artifact_version!r}; "
            f"this build expects {ARTIFACT_VERSION!r}. Retrain the model."
        )
    return metadata


def load_artifact(task: str, model_name: str, settings: Settings | None = None) -> LoadedModel:
    """Load a fitted pipeline after validating its metadata contract."""
    settings = settings or get_settings()
    model_path, meta_path = artifact_paths(task, model_name, settings)
    if not model_path.exists():
        raise ArtifactError(
            f"Model artifact not found at {model_path}. "
            "Run `python scripts/train.py --task {task}` first."
        )
    metadata = load_metadata(task, model_name, settings)
    pipeline = joblib.load(model_path)
    # Fail fast if the loaded object is not a usable estimator with predict_proba.
    if not hasattr(pipeline, "predict_proba"):
        raise ArtifactError(f"Artifact {model_path} has no predict_proba; refusing to serve it.")
    logger.info(
        "loaded artifact task=%s model=%s version=%s", task, model_name, metadata.model_version
    )
    return LoadedModel(
        pipeline=pipeline, metadata=metadata, path=model_path, model_version=metadata.model_version
    )


def set_champion(
    task: str, model_name: str, settings: Settings | None = None, **extra: Any
) -> Path:
    """Record which model should be served for *task*."""
    settings = settings or get_settings()
    directory = model_dir(task, settings)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CHAMPION_FILENAME
    payload = {"task": task, "model_name": model_name, **extra}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def get_champion(task: str, settings: Settings | None = None) -> str | None:
    """Return the champion model name for *task*, if one has been recorded."""
    settings = settings or get_settings()
    path = model_dir(task, settings) / CHAMPION_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("model_name")


def available_models(task: str, settings: Settings | None = None) -> list[str]:
    """List model names with both an artifact and metadata present."""
    settings = settings or get_settings()
    directory = model_dir(task, settings)
    if not directory.exists():
        return []
    names = []
    for meta in sorted(directory.glob("*.meta.json")):
        name = meta.name[: -len(".meta.json")]
        if artifact_paths(task, name, settings)[0].exists():
            names.append(name)
    return names
