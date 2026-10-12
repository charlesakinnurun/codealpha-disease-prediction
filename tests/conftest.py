"""Shared pytest fixtures.

Most tests use small, deterministic **synthetic** frames built from the dataset
schemas so they are fast and independent of downloads. Tests that need the real
datasets are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from disease_prediction.config import REPO_ROOT, Settings
from disease_prediction.data.registry import get_schema
from disease_prediction.data.schemas import FeatureKind


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated settings pointing every output at a temporary directory."""
    return Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        config_dir=REPO_ROOT / "configs",
        random_seed=7,
        allow_optional_imputation=True,
    )


def make_synthetic_frame(
    task: str, n: int = 80, seed: int = 0, *, separable: bool = False
) -> pd.DataFrame:
    """Build a schema-valid synthetic frame with a deterministic pattern."""
    schema = get_schema(task)
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    signal = np.zeros(n)
    for spec in schema.features:
        if spec.kind is FeatureKind.NUMERIC:
            lo = spec.minimum if spec.minimum is not None else 0.0
            hi = spec.maximum if spec.maximum is not None else 50.0
            values = rng.uniform(lo, hi, n)
            signal += values - values.mean()
            if spec.optional and rng.random() < 0.1:
                values = values.astype(float)
                values[rng.random(n) < 0.2] = np.nan
            data[spec.name] = values.astype(float)
        elif spec.allowed_values is not None:
            data[spec.name] = rng.choice(list(spec.allowed_values), size=n)
        else:
            data[spec.name] = rng.choice(["Male", "Female"], size=n)

    target = (signal > np.median(signal)).astype(int) if separable else rng.integers(0, 2, n)
    data["target"] = target
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame("heart_disease")


@pytest.fixture
def trained_artifacts(settings: Settings):
    """Train a tiny heart-disease model into the temporary artifact store."""
    from disease_prediction.models.train import TrainingConfig, train_task

    frame = make_synthetic_frame("heart_disease", n=120, seed=3)
    config = TrainingConfig(
        task="heart_disease",
        models=["dummy", "logistic_regression"],
        n_iter=2,
        cv_folds=2,
        seed=7,
        calibration_method="sigmoid",
        save=True,
    )
    result = train_task(config, settings, df=frame)
    return settings, result, frame
