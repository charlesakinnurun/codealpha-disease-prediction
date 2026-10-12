# syntax=docker/dockerfile:1
# Multi-stage-free, slim image for the FastAPI inference service.
# Model artifacts are NOT baked in: mount ./artifacts at runtime (see Makefile).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    DP_API_HOST=0.0.0.0 \
    DP_API_PORT=8000

# Non-root user for least privilege.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

# Install uv (pinned) and resolve runtime dependencies first for layer caching.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --no-dev --frozen

# Drop privileges.
RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uv", "run", "--no-dev", "uvicorn", "disease_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
