from __future__ import annotations

from disease_prediction.data.registry import get_schema
from disease_prediction.features.pipelines import build_pipeline
from disease_prediction.models.registry import model_spec
from disease_prediction.models.train import build_inference_schema
from tests.conftest import make_synthetic_frame


def test_inference_schema_uses_training_ranges():
    schema = get_schema("heart_disease")
    train = make_synthetic_frame("heart_disease", n=50, seed=2)
    inference = build_inference_schema(schema, train)
    age = inference.feature("age")
    assert age is not None
    assert age.observed_min == train["age"].min()
    assert age.observed_max == train["age"].max()


def test_inference_schema_preserves_feature_order():
    schema = get_schema("diabetes")
    train = make_synthetic_frame("diabetes", n=30)
    inference = build_inference_schema(schema, train)
    assert inference.feature_names == schema.feature_names


def test_model_builds_via_registry_factories():
    schema = get_schema("breast_cancer")
    frame = make_synthetic_frame("breast_cancer", n=40, seed=1)
    x, y = frame[schema.feature_names], frame["target"].to_numpy()
    for name in ("logistic_regression", "random_forest", "svm", "xgboost"):
        pipeline = build_pipeline(model_spec(name).build(0), schema)
        pipeline.fit(x, y)
        proba = pipeline.predict_proba(x)
        assert proba.shape == (len(x), 2)
