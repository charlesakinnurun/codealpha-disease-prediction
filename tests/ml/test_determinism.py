from __future__ import annotations

import numpy as np

from disease_prediction.data.registry import get_schema
from disease_prediction.features.pipelines import build_pipeline
from disease_prediction.models.registry import model_spec
from disease_prediction.models.train import TrainingConfig, train_task
from tests.conftest import make_synthetic_frame


def test_fixed_seed_gives_identical_predictions():
    frame = make_synthetic_frame("heart_disease", n=120, seed=9)
    schema = get_schema("heart_disease")
    x, y = frame[schema.feature_names], frame["target"].to_numpy()

    first = build_pipeline(model_spec("random_forest").build(123), schema)
    second = build_pipeline(model_spec("random_forest").build(123), schema)
    first.fit(x, y)
    second.fit(x, y)
    assert np.allclose(first.predict_proba(x), second.predict_proba(x))


def test_repeated_training_is_reproducible(settings):
    frame = make_synthetic_frame("heart_disease", n=120, seed=4)

    def run():
        return train_task(
            TrainingConfig(
                task="heart_disease",
                models=["dummy", "logistic_regression"],
                n_iter=2,
                cv_folds=2,
                seed=5,
                save=False,
            ),
            settings,
            df=frame,
        )

    a, b = run(), run()
    assert a.champion == b.champion
    assert np.isclose(
        a.results[a.champion].test_metrics["roc_auc"],
        b.results[b.champion].test_metrics["roc_auc"],
    )
