# Disease Prediction — developer shortcuts.
# Requires `uv` (https://docs.astral.sh/uv/) and, for Docker targets, Docker.
# On Windows, run these with `make` under Git Bash/WSL, or copy the commands.

.DEFAULT_GOAL := help
PY := uv run python

.PHONY: help install data validate train evaluate api dashboard test test-fast lint format typecheck docker-build docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the environment and install all dependencies (incl. dev)
	uv sync --extra dev

data: ## Download datasets, validate them and write profiles + EDA figures
	$(PY) scripts/download_data.py
	$(PY) scripts/validate_data.py

validate: ## Validate datasets and regenerate quality reports + figures
	$(PY) scripts/validate_data.py

train: ## Train all tasks (configs/train.yaml), write artifacts and metrics
	$(PY) scripts/train.py

evaluate: ## Evaluate champions on the held-out test split and write figures
	$(PY) scripts/evaluate.py

api: ## Run the FastAPI service (http://localhost:8000/docs)
	uv run uvicorn disease_prediction.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Run the Streamlit dashboard (http://localhost:8501)
	uv run streamlit run src/disease_prediction/dashboard/app.py

test: ## Run the full test suite
	$(PY) -m pytest

test-fast: ## Run everything except slow, real-data tests
	$(PY) -m pytest -m "not slow"

lint: ## Lint with Ruff
	uv run ruff check .

format: ## Auto-format and auto-fix with Ruff
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Static type checking of the package
	uv run mypy

docker-build: ## Build the API container image
	docker build -t disease-prediction:latest .

docker-run: ## Run the API container (mounts trained artifacts and data read-only)
	docker run --rm -p 8000:8000 -v "$(CURDIR)/artifacts:/app/artifacts:ro" -v "$(CURDIR)/data:/app/data:ro" disease-prediction:latest

clean: ## Remove caches and generated reports (keeps data and artifacts)
	-rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	-rm -rf reports/figures/* reports/metrics/* reports/tmp
