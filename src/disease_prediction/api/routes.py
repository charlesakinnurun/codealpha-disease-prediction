"""Versioned API routes."""

from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from disease_prediction import __version__
from disease_prediction.api.dependencies import ModelService, ModelUnavailableError
from disease_prediction.api.schemas import (
    BREAST_CANCER_REQUEST,
    DIABETES_REQUEST,
    HEART_DISEASE_REQUEST,
    FeatureContribution,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    Prediction,
    PredictionResponse,
    Probability,
    ReadyResponse,
)
from disease_prediction.config import TASK_LABELS, TASKS
from disease_prediction.logging_config import get_logger
from disease_prediction.models.persistence import available_models, get_champion, load_metadata
from disease_prediction.monitoring.metrics import metrics

logger = get_logger(__name__)

router = APIRouter()


def _service(request: Request) -> ModelService:
    return request.app.state.model_service


def _clean_payload(payload: Any) -> dict[str, Any]:
    """Convert a validated request model into plain Python values."""
    raw = payload.model_dump()
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Enum):
            cleaned[key] = value.value
        else:
            cleaned[key] = value
    return cleaned


def _respond(request: Request, task: str, payload: Any, explain: bool) -> PredictionResponse:
    request_id = getattr(request.state, "request_id", "-")
    features = _clean_payload(payload)
    try:
        outcome = _service(request).predict(task, features, explain=explain)
    except ModelUnavailableError as exc:
        metrics.incr(f"predict_unavailable_total|{task}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "model_unavailable", "message": str(exc), "fields": []},
        ) from exc
    except ValueError as exc:
        metrics.incr(f"predict_validation_error_total|{task}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "schema_mismatch", "message": str(exc), "fields": []},
        ) from exc

    metrics.record_prediction(task, outcome.model_name, outcome.probability_positive, outcome.label)
    return PredictionResponse(
        request_id=request_id,
        task=task,
        model_name=outcome.model_name,
        model_version=outcome.model_version,
        prediction=Prediction(
            label=outcome.label,
            label_name=(
                str(outcome.positive_class) if outcome.label == 1 else str(outcome.negative_class)
            ),
            positive_class=outcome.positive_class,
            negative_class=outcome.negative_class,
        ),
        probability=Probability(
            positive=round(outcome.probability_positive, 6),
            negative=round(outcome.probability_negative, 6),
            threshold=outcome.threshold,
            calibrated=outcome.calibrated,
        ),
        contributions=(
            [FeatureContribution(**c) for c in outcome.contributions]
            if outcome.contributions
            else None
        ),
        warnings=outcome.warnings,
    )


# ---------------------------------------------------------------------------
# Operational endpoints
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — the process is up and serving."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(request: Request) -> ReadyResponse:
    """Readiness probe — at least one model is loadable and servable."""
    service = _service(request)
    ready_tasks, missing = service.ready_tasks()
    details = {t: ("ready" if t in ready_tasks else "no artifact") for t in TASKS}
    return ReadyResponse(
        ready=bool(ready_tasks),
        loaded_tasks=ready_tasks,
        missing_tasks=missing,
        details=details,
    )


@router.get("/v1/metrics", tags=["ops"])
def metrics_snapshot() -> dict[str, Any]:
    """In-process metrics snapshot (latency, errors, prediction distribution)."""
    return metrics.snapshot()


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
@router.get("/v1/models", response_model=ModelsResponse, tags=["models"])
def list_models(request: Request) -> ModelsResponse:
    """List trained models and which one is served per task."""
    service = _service(request)
    infos: list[ModelInfo] = []
    for task in TASKS:
        champion = get_champion(task, service.settings)
        for name in available_models(task, service.settings):
            try:
                meta = load_metadata(task, name, service.settings)
            except Exception:  # noqa: BLE001
                continue
            infos.append(
                ModelInfo(
                    task=task,
                    model_name=name,
                    display_name=meta.display_name,
                    model_version=meta.model_version,
                    champion=name == champion,
                    calibrated=meta.calibrated,
                    test_metrics=meta.test_metrics,
                )
            )
    return ModelsResponse(models=infos, tasks=list(TASKS))


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/v1/predict/heart-disease",
    response_model=PredictionResponse,
    tags=["predict"],
    summary=f"Predict {TASK_LABELS['heart_disease']}",
)
def predict_heart_disease(
    request: Request,
    payload: HEART_DISEASE_REQUEST,  # type: ignore[valid-type]
    explain: bool = Query(False, description="Include per-feature contributions (slower)."),
) -> PredictionResponse:
    return _respond(request, "heart_disease", payload, explain)


@router.post(
    "/v1/predict/diabetes",
    response_model=PredictionResponse,
    tags=["predict"],
    summary=f"Predict {TASK_LABELS['diabetes']}",
)
def predict_diabetes(
    request: Request,
    payload: DIABETES_REQUEST,  # type: ignore[valid-type]
    explain: bool = Query(False, description="Include per-feature contributions (slower)."),
) -> PredictionResponse:
    return _respond(request, "diabetes", payload, explain)


@router.post(
    "/v1/predict/breast-cancer",
    response_model=PredictionResponse,
    tags=["predict"],
    summary=f"Predict {TASK_LABELS['breast_cancer']}",
)
def predict_breast_cancer(
    request: Request,
    payload: BREAST_CANCER_REQUEST,  # type: ignore[valid-type]
    explain: bool = Query(False, description="Include per-feature contributions (slower)."),
) -> PredictionResponse:
    return _respond(request, "breast_cancer", payload, explain)
