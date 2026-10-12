from __future__ import annotations

import pytest

from disease_prediction.config import TASKS
from disease_prediction.models.persistence import available_models, get_champion
from disease_prediction.models.train import TrainingConfig, train_task
from tests.conftest import make_synthetic_frame


@pytest.mark.parametrize("task", TASKS)
def test_training_protocol_end_to_end(task: str, settings):
    frame = make_synthetic_frame(task, n=100, seed=1)
    config = TrainingConfig(
        task=task,
        models=["dummy", "logistic_regression", "random_forest"],
        n_iter=2,
        cv_folds=2,
        seed=7,
        save=True,
    )
    result = train_task(config, settings, df=frame)

    assert result.champion in result.results
    assert "dummy" in result.results
    assert sum(result.split_sizes.values()) == len(frame)

    champion = result.results[result.champion]
    assert 0.0 <= champion.test_metrics["roc_auc"] <= 1.0
    assert 0.0 <= champion.test_metrics["brier"] <= 1.0

    assert get_champion(task, settings) == result.champion
    assert result.champion in available_models(task, settings)


def test_comparison_table_is_sorted_by_selection_metric(settings):
    frame = make_synthetic_frame("heart_disease", n=90, seed=2)
    result = train_task(
        TrainingConfig(
            task="heart_disease",
            models=["dummy", "logistic_regression"],
            n_iter=2,
            cv_folds=2,
            seed=1,
        ),
        settings,
        df=frame,
    )
    scores = [row["val_roc_auc"] for row in result.comparison_table()]
    assert scores == sorted(scores, reverse=True)


def test_label_order_consistency(settings):
    """The persisted inference schema must match the training feature order."""
    from disease_prediction.data.registry import get_schema

    frame = make_synthetic_frame("breast_cancer", n=80, seed=4)
    train_task(
        TrainingConfig(
            task="breast_cancer", models=["logistic_regression"], n_iter=1, cv_folds=2, seed=1
        ),
        settings,
        df=frame,
    )
    from disease_prediction.models.persistence import load_artifact

    loaded = load_artifact("breast_cancer", "logistic_regression", settings)
    assert loaded.metadata.feature_schema.feature_names == get_schema("breast_cancer").feature_names
