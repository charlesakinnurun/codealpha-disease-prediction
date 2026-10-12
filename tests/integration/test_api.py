from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from disease_prediction.api.main import create_app
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


def test_health_endpoint(trained_artifacts):
    settings, _, _ = trained_artifacts
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint_reports_loaded_and_missing(trained_artifacts):
    settings, _, _ = trained_artifacts
    with TestClient(create_app(settings)) as client:
        body = client.get("/ready").json()
    assert body["ready"] is True
    assert "heart_disease" in body["loaded_tasks"]
    assert "diabetes" in body["missing_tasks"]


def test_models_endpoint_lists_champion(trained_artifacts):
    settings, _, _ = trained_artifacts
    with TestClient(create_app(settings)) as client:
        body = client.get("/v1/models").json()
    assert body["tasks"]
    heart = [m for m in body["models"] if m["task"] == "heart_disease"]
    assert heart and any(m["champion"] for m in heart)


def test_predict_returns_structured_response(trained_artifacts):
    settings, _, _ = trained_artifacts
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict/heart-disease", json=valid_payload("heart_disease"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "heart_disease"
    assert body["prediction"]["label"] in (0, 1)
    assert 0.0 <= body["probability"]["positive"] <= 1.0
    assert body["request_id"]
    assert "NOT a diagnosis" in body["disclaimer"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_predict_with_explanations(trained_artifacts):
    settings, _, _ = trained_artifacts
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/predict/heart-disease",
            json=valid_payload("heart_disease"),
            params={"explain": "true"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["contributions"]


def test_predict_rejects_invalid_category(trained_artifacts):
    settings, _, _ = trained_artifacts
    payload = valid_payload("heart_disease")
    payload["cp"] = 42
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict/heart-disease", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "validation_error"
    assert "cp" in " ".join(detail["fields"])


def test_predict_rejects_extra_fields(trained_artifacts):
    settings, _, _ = trained_artifacts
    payload = valid_payload("heart_disease")
    payload["mrn"] = "12345"
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict/heart-disease", json=payload)
    assert response.status_code == 422


def test_predict_missing_model_returns_503(trained_artifacts):
    settings, _, _ = trained_artifacts
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict/diabetes", json=valid_payload("diabetes"))
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_unavailable"


def test_payload_size_limit_returns_413(trained_artifacts):
    settings, _, _ = trained_artifacts
    settings.api_max_body_bytes = 64
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/predict/heart-disease", json=valid_payload("heart_disease"))
    assert response.status_code == 413


def test_metrics_endpoint_snapshot(trained_artifacts):
    settings, _, _ = trained_artifacts
    with TestClient(create_app(settings)) as client:
        client.post("/v1/predict/heart-disease", json=valid_payload("heart_disease"))
        body = client.get("/v1/metrics").json()
    assert "counters" in body
    assert body["counters"].get("http_requests_total", 0) >= 1
    assert any(k.startswith("heart_disease|") for k in body["prediction_counts"])
