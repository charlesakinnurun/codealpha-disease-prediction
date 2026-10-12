"""HTTP middleware: request IDs, body-size limits, timing and structured logs."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from disease_prediction.logging_config import get_logger
from disease_prediction.monitoring.metrics import metrics

logger = get_logger("disease_prediction.api")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID, enforce a body-size limit and record latency.

    Request bodies are never logged (patient data minimisation).
    """

    def __init__(self, app, max_body_bytes: int = 32 * 1024) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_body_bytes:
                    metrics.incr("http_requests_rejected_size_total")
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": {
                                "code": "payload_too_large",
                                "message": f"Request body exceeds {self.max_body_bytes} bytes.",
                                "fields": [],
                            }
                        },
                        headers={REQUEST_ID_HEADER: request_id},
                    )
            except ValueError:
                pass

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - re-raised after logging
            duration = time.perf_counter() - start
            metrics.incr("http_requests_errors_total")
            logger.exception(
                "request_failed method=%s path=%s request_id=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                request_id,
                duration * 1000,
            )
            raise

        duration = time.perf_counter() - start
        metrics.incr("http_requests_total")
        metrics.observe_latency("http_request", duration)
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration * 1000,
            request_id,
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration * 1000:.1f}"
        return response
