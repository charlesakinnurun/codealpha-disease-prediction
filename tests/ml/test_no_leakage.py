"""ML-specific guarantees: leakage prevention, determinism and output contracts."""

from __future__ import annotations

import numpy as np

from disease_prediction.data.registry import get_schema
from disease_prediction.features.pipelines import build_pipeline
from disease_prediction.models.registry import model_spec
from disease_prediction.models.train import TrainingConfig, split_data
from tests.conftest import make_synthetic_frame


def test_splits_are_disjoint_and_preserve_all_rows():
    frame = make_synthetic_frame("heart_disease", n=200, seed=1)
    train, validation, test = split_data(frame, TrainingConfig(task="heart_disease", seed=42))
    index_sets = [set(train.index), set(validation.index), set(test.index)]
    assert index_sets[0].isdisjoint(index_sets[1])
    assert index_sets[0].isdisjoint(index_sets[2])
    assert index_sets[1].isdisjoint(index_sets[2])
    assert sum(len(i) for i in index_sets) == len(frame)


def test_preprocessing_statistics_come_from_train_only():
    frame = make_synthetic_frame("heart_disease", n=150, seed=3)
    train, _, test = split_data(frame, TrainingConfig(task="heart_disease", seed=42))
    schema = get_schema("heart_disease")
    pipeline = build_pipeline(model_spec("logistic_regression").build(0), schema)
    pipeline.fit(train[schema.feature_names], train["target"])

    scaler = (
        pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
    )
    # The scaler must reflect the *training* distribution, never the test set.
    assert not np.allclose(scaler.mean_, np.nanmean(test[schema.numeric_features()], axis=0))
    assert np.allclose(scaler.mean_, np.nanmean(train[schema.numeric_features()], axis=0))


def test_features_exclude_target():
    schema = get_schema("diabetes")
    assert "target" not in schema.feature_names
    assert set(schema.feature_names).isdisjoint({"target", "class"})
