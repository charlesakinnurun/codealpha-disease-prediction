from __future__ import annotations

import pytest

from disease_prediction.config import TASKS
from disease_prediction.data.registry import all_schemas, get_schema
from disease_prediction.data.schemas import FeatureKind


def test_all_tasks_have_schemas():
    schemas = all_schemas()
    assert set(schemas) == set(TASKS)


@pytest.mark.parametrize("task", TASKS)
def test_schema_is_internally_consistent(task: str):
    schema = get_schema(task)
    assert schema.feature_names, "schema must declare features"
    assert len(schema.feature_names) == len(set(schema.feature_names)), "duplicate feature names"
    assert schema.positive_class != schema.negative_class
    for spec in schema.features:
        if spec.kind is FeatureKind.NUMERIC:
            assert spec.allowed_values is None
        else:
            assert spec.allowed_values, f"{spec.name} must declare allowed values"
        if spec.minimum is not None and spec.maximum is not None:
            assert spec.minimum < spec.maximum


def test_feature_lookup_and_unknown_name():
    schema = get_schema("heart_disease")
    assert schema.feature("age").name == "age"
    with pytest.raises(KeyError):
        schema.feature("does_not_exist")


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        get_schema("not_a_task")


def test_feature_kind_partition():
    schema = get_schema("heart_disease")
    parts = schema.numeric_features() + schema.categorical_features()
    assert sorted(parts) == sorted(schema.feature_names)
