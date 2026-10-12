from __future__ import annotations

from pathlib import Path

from disease_prediction.config import REPO_ROOT, Settings, get_settings


def test_default_paths_resolve_against_repo_root():
    s = Settings()
    assert s.data_dir == REPO_ROOT / "data"
    assert s.models_dir == REPO_ROOT / "artifacts" / "models"


def test_absolute_paths_are_preserved(tmp_path: Path):
    s = Settings(data_dir=tmp_path / "d")
    assert s.data_dir == tmp_path / "d"
    assert s.raw_dir == tmp_path / "d" / "raw"


def test_derived_directories_and_ensure_dirs(tmp_path: Path):
    s = Settings(data_dir=tmp_path / "d", artifacts_dir=tmp_path / "a", reports_dir=tmp_path / "r")
    s.ensure_dirs()
    for directory in (
        s.raw_dir,
        s.processed_dir,
        s.models_dir,
        s.runs_dir,
        s.figures_dir,
        s.metrics_dir,
    ):
        assert directory.is_dir()


def test_cors_origins_parsing(tmp_path: Path):
    s = Settings(api_cors_origins="https://a.example, https://b.example ,")
    assert s.cors_origins == ["https://a.example", "https://b.example"]
    assert Settings(api_cors_origins="").cors_origins == []


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
