"""Centralised, typed application configuration.

All runtime configuration flows through :class:`Settings`, sourced from
environment variables (prefix ``DP_``) with sensible defaults so the project
runs on a fresh developer machine without any ``.env`` file. Secrets never
carry defaults and are never logged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root is three levels up from this file: src/disease_prediction/config.py
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Canonical prediction task identifiers used across data, models, API and UI.
TASKS: tuple[str, ...] = ("heart_disease", "diabetes", "breast_cancer")

#: Human-readable labels for the tasks.
TASK_LABELS: dict[str, str] = {
    "heart_disease": "Heart Disease",
    "diabetes": "Diabetes",
    "breast_cancer": "Breast Cancer",
}


class Settings(BaseSettings):
    """Runtime settings. Instantiate through :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="DP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Filesystem -------------------------------------------------------
    data_dir: Path = Field(default=Path("data"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    reports_dir: Path = Field(default=Path("reports"))
    config_dir: Path = Field(default=Path("configs"))

    # --- Reproducibility --------------------------------------------------
    random_seed: int = 42
    log_level: str = "INFO"

    # --- API --------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_max_body_bytes: int = 32 * 1024
    api_cors_origins: str = ""
    model_cache_size: int = 8
    allow_optional_imputation: bool = True

    # --- Derived paths ----------------------------------------------------
    @field_validator("data_dir", "artifacts_dir", "reports_dir", "config_dir")
    @classmethod
    def _resolve_relative(cls, value: Path) -> Path:
        """Interpret relative paths against the repository root."""
        return value if value.is_absolute() else (REPO_ROOT / value)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.reports_dir / "metrics"

    @property
    def cors_origins(self) -> list[str]:
        """Parse the CORS origins environment variable into a list."""
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        """Create all runtime directories if they do not exist."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.runs_dir,
            self.figures_dir,
            self.metrics_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
