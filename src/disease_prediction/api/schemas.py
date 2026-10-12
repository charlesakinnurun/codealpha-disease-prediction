"""Pydantic request/response schemas for the API.

Request models are generated from the dataset registry so that the web
interface, the OpenAPI docs and the validators can never drift from the
documented schema. Categorical/binary fields become enums and numeric fields
carry inclusive bounds.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from disease_prediction.data.registry import get_schema
from disease_prediction.data.schemas import FeatureKind

DISCLAIMER = (
    "Educational and research use only. This output is a statistical estimate "
    "from a benchmark model and is NOT a diagnosis. It must not be used to "
    "diagnose, treat, or rule out any disease. Always consult a qualified "
    "healthcare professional."
)


def _make_enum(name: str, values: tuple[Any, ...]) -> type[Enum]:
    return cast(
        "type[Enum]",
        Enum(
            name,
            {f"VALUE_{str(v)}": v for v in values},
            type=int if all(isinstance(v, int) for v in values) else str,
        ),
    )


def _build_request_model(task: str) -> type[BaseModel]:
    schema = get_schema(task)
    fields: dict[str, Any] = {}

    for spec in schema.features:
        if spec.kind is FeatureKind.NUMERIC:
            description = spec.description + (f" [{spec.unit}]" if spec.unit else "")
            fields[spec.name] = (
                float,
                Field(..., ge=spec.minimum, le=spec.maximum, description=description),
            )
        else:
            enum_cls = _make_enum(f"{task}_{spec.name}", spec.allowed_values or ())
            if spec.optional:
                fields[spec.name] = (
                    enum_cls | None,
                    Field(None, description=spec.description + " (nullable; imputed if omitted)"),
                )
            else:
                fields[spec.name] = (enum_cls, Field(..., description=spec.description))

    model = create_model(
        f"{task.title().replace('_', '')}PredictionRequest",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    model.model_config = ConfigDict(extra="forbid")
    return model


HEART_DISEASE_REQUEST = _build_request_model("heart_disease")
DIABETES_REQUEST = _build_request_model("diabetes")
BREAST_CANCER_REQUEST = _build_request_model("breast_cancer")

REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "heart_disease": HEART_DISEASE_REQUEST,
    "diabetes": DIABETES_REQUEST,
    "breast_cancer": BREAST_CANCER_REQUEST,
}


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class Prediction(BaseModel):
    label: int = Field(..., ge=0, le=1)
    label_name: str
    positive_class: str | int
    negative_class: str | int


class Probability(BaseModel):
    positive: float = Field(..., ge=0.0, le=1.0)
    negative: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    calibrated: bool


class PredictionResponse(BaseModel):
    request_id: str
    task: str
    model_name: str
    model_version: str
    prediction: Prediction
    probability: Probability
    contributions: list[FeatureContribution] | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class ModelInfo(BaseModel):
    task: str
    model_name: str
    display_name: str
    model_version: str
    champion: bool
    calibrated: bool
    test_metrics: dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    tasks: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    loaded_tasks: list[str]
    missing_tasks: list[str]
    details: dict[str, str] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    code: str
    message: str
    fields: list[str] = Field(default_factory=list)
