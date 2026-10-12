from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from disease_prediction.api.schemas import (
    BREAST_CANCER_REQUEST,
    DIABETES_REQUEST,
    HEART_DISEASE_REQUEST,
    REQUEST_MODELS,
)
from disease_prediction.data.registry import get_schema
from disease_prediction.data.schemas import FeatureKind


def valid_payload(task: str) -> dict[str, Any]:
    schema = get_schema(task)
    payload: dict[str, Any] = {}
    for spec in schema.features:
        if spec.kind is FeatureKind.NUMERIC:
            lo = spec.minimum if spec.minimum is not None else 0.0
            hi = spec.maximum if spec.maximum is not None else lo + 10
            payload[spec.name] = (lo + hi) / 2
        else:
            payload[spec.name] = spec.allowed_values[0]
    return payload


@pytest.mark.parametrize("task", ["heart_disease", "diabetes", "breast_cancer"])
def test_valid_payloads_are_accepted(task: str):
    model = REQUEST_MODELS[task]
    model.model_validate(valid_payload(task))


def test_heart_request_rejects_unexpected_category():
    payload = valid_payload("heart_disease")
    payload["cp"] = 99  # not in {1,2,3,4}
    with pytest.raises(ValidationError):
        HEART_DISEASE_REQUEST.model_validate(payload)


def test_heart_request_rejects_out_of_range_numeric():
    payload = valid_payload("heart_disease")
    payload["trestbps"] = 5000
    with pytest.raises(ValidationError):
        HEART_DISEASE_REQUEST.model_validate(payload)


def test_extra_fields_are_forbidden():
    payload = valid_payload("heart_disease")
    payload["patient_name"] = "Jane Doe"
    with pytest.raises(ValidationError):
        HEART_DISEASE_REQUEST.model_validate(payload)


def test_missing_required_field_is_rejected():
    payload = valid_payload("heart_disease")
    del payload["age"]
    with pytest.raises(ValidationError):
        HEART_DISEASE_REQUEST.model_validate(payload)


def test_optional_heart_features_may_be_omitted():
    payload = valid_payload("heart_disease")
    payload.pop("ca", None)
    payload.pop("thal", None)
    HEART_DISEASE_REQUEST.model_validate(payload)


def test_diabetes_gender_enum_accepts_documented_values():
    payload = valid_payload("diabetes")
    payload["Gender"] = "Female"
    DIABETES_REQUEST.model_validate(payload)
    payload["Gender"] = "female"  # not a documented category
    with pytest.raises(ValidationError):
        DIABETES_REQUEST.model_validate(payload)


def test_breast_cancer_exposes_thirty_features():
    payload = valid_payload("breast_cancer")
    assert len(payload) == 30
    BREAST_CANCER_REQUEST.model_validate(payload)
