"""FastAPI service."""

from disease_prediction.api.dependencies import ModelService, PredictionOutcome
from disease_prediction.api.main import app, create_app

__all__ = ["app", "create_app", "ModelService", "PredictionOutcome"]
