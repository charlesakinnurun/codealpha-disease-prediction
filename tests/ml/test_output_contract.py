from __future__ import annotations

import numpy as np
import pytest

from disease_prediction.config import TASKS
from disease_prediction.data.registry import get_schema
from disease_prediction.evaluation.metrics import binary_metrics
from disease_prediction.features.pipelines import build_pipeline
from disease_prediction.models.registry import model_spec
from tests.conftest import make_synthetic_frame


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("model_name", ["logistic_regression", "svm", "random_forest", "xgboost"])
def test_probability_output_contract(task: str, model_name: str):
    schema = get_schema(task)
    frame = make_synthetic_frame(task, n=60, seed=2)
    x, y = frame[schema.feature_names], frame["target"].to_numpy()
    pipeline = build_pipeline(model_spec(model_name).build(0), schema)
    pipeline.fit(x, y)

    proba = pipeline.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.isfinite(proba).all()
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(pipeline.classes_) == [0, 1]


def test_trained_model_beats_dummy_on_separable_data():
    """Sanity check: a real model must beat the prior baseline on learnable data."""
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import roc_auc_score

    schema = get_schema("heart_disease")
    frame = make_synthetic_frame("heart_disease", n=200, seed=11, separable=True)
    x, y = frame[schema.feature_names], frame["target"].to_numpy()

    model = build_pipeline(model_spec("logistic_regression").build(0), schema)
    model.fit(x, y)
    dummy = DummyClassifier(strategy="prior").fit(x, y)

    model_auc = roc_auc_score(y, model.predict_proba(x)[:, 1])
    dummy_auc = roc_auc_score(y, dummy.predict_proba(x)[:, 1])
    assert model_auc > dummy_auc
    assert model_auc > 0.8


def test_metrics_are_bounded_on_realistic_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 300), 0, 1)
    m = binary_metrics(y, score)
    for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
        assert 0.0 <= m[key] <= 1.0
