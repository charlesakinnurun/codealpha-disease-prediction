from __future__ import annotations

import numpy as np

from disease_prediction.data.registry import get_schema
from disease_prediction.features.pipelines import build_pipeline, build_preprocessor
from disease_prediction.models.registry import model_spec
from tests.conftest import make_synthetic_frame


def _x_y(task: str, n: int = 80):
    frame = make_synthetic_frame(task, n=n, seed=1)
    schema = get_schema(task)
    return frame[schema.feature_names].copy(), frame["target"].to_numpy(), schema


def test_preprocessor_output_is_dense_and_named():
    x, y, schema = _x_y("heart_disease")
    pipeline = build_pipeline(model_spec("logistic_regression").build(0), schema)
    pipeline.fit(x, y)
    out = pipeline.named_steps["preprocessor"].transform(x)
    assert out.ndim == 2
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    assert out.shape[1] == len(names)
    # One-hot encoding must have expanded the categorical features.
    assert len(names) > len(schema.feature_names)


def test_scaler_is_fitted_on_training_data_only():
    """Imputation/scaling statistics must come from training rows only."""
    x, _, schema = _x_y("heart_disease", n=100)
    train = x.iloc[:70]
    pre = build_preprocessor(schema)
    pre.fit(train)
    scaler = pre.named_transformers_["numeric"].named_steps["scaler"]
    age_index = schema.numeric_features().index("age")
    assert np.isclose(scaler.mean_[age_index], np.nanmean(train["age"]))


def test_unseen_category_does_not_crash():
    x, _, schema = _x_y("heart_disease")
    pre = build_preprocessor(schema)
    pre.fit(x)
    mutated = x.copy()
    mutated.loc[mutated.index[0], "thal"] = 99  # never seen during fit
    out = pre.transform(mutated)
    assert np.isfinite(out).all()


def test_missing_values_are_imputed():
    x, _, schema = _x_y("heart_disease")
    x = x.copy()
    x.loc[x.index[0], "ca"] = np.nan
    x.loc[x.index[1], "trestbps"] = np.nan
    pre = build_preprocessor(schema)
    out = pre.fit_transform(x)
    assert np.isfinite(out).all()


def test_pipeline_fits_and_predicts_probabilities():
    x, y, schema = _x_y("diabetes")
    pipeline = build_pipeline(model_spec("logistic_regression").build(0), schema)
    pipeline.fit(x, y)
    proba = pipeline.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.all((proba >= 0) & (proba <= 1))
