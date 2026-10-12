"""FastAPI application factory.

Separation of concerns:

- :mod:`disease_prediction.api.routes` — HTTP routing and response shaping.
- :mod:`disease_prediction.api.dependencies` — model loading and prediction logic.
- :mod:`disease_prediction.api.schemas` — request/response contracts.
- this module — application wiring, middleware and error handling.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from disease_prediction import __version__
from disease_prediction.api.dependencies import ModelService
from disease_prediction.api.middleware import RequestContextMiddleware
from disease_prediction.api.routes import router
from disease_prediction.config import Settings, get_settings
from disease_prediction.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = ModelService(settings)
        app.state.model_service = service
        ready, missing = service.ready_tasks()
        logger.info(
            "api_startup version=%s ready_tasks=%s missing_tasks=%s", __version__, ready, missing
        )
        if missing:
            logger.warning(
                "api_startup_no_model tasks=%s — run scripts/train.py to produce artifacts", missing
            )
        yield
        logger.info("api_shutdown")

    app = FastAPI(
        title="Disease Prediction API",
        version=__version__,
        description=(
            "Explainable machine-learning predictions from structured patient data.\n\n"
            "**Educational and research use only.** Outputs are statistical "
            "estimates from benchmark datasets and are **not** diagnoses."
        ),
        lifespan=lifespan,
    )

    app.state.settings = settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.api_max_body_bytes)

    app.include_router(router)

    _register_error_handlers(app)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Return structured, information-minimal errors (no stack traces)."""

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted(
            {".".join(str(p) for p in err.get("loc", ()) if p != "body") for err in exc.errors()}
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "validation_error",
                    "message": "One or more fields are missing or invalid.",
                    "fields": [f for f in fields if f],
                    "errors": [
                        {
                            "field": ".".join(str(p) for p in e.get("loc", ()) if p != "body"),
                            "message": e.get("msg", ""),
                        }
                        for e in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = {"detail": detail}
        else:
            body = {
                "detail": {"code": f"http_{exc.status_code}", "message": str(detail), "fields": []}
            }
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("unhandled_error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "internal_error",
                    "message": "An internal error occurred. Reference the request id when reporting.",
                    "fields": [],
                    "request_id": request_id,
                }
            },
        )


app = create_app()
