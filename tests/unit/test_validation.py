from __future__ import annotations

import pandas as pd

from disease_prediction.data.registry import get_schema
from disease_prediction.data.validate import validate_dataframe
from tests.conftest import make_synthetic_frame


def test_valid_frame_passes():
    frame = make_synthetic_frame("heart_disease", n=60)
    report = validate_dataframe(frame, get_schema("heart_disease"))
    assert report.ok, report.errors
    assert report.n_rows == 60


def test_missing_column_is_an_error():
    frame = make_synthetic_frame("heart_disease", n=20).drop(columns=["age"])
    report = validate_dataframe(frame, get_schema("heart_disease"))
    assert not report.ok
    assert any("Missing required feature columns" in e for e in report.errors)


def test_unexpected_category_is_an_error():
    frame = make_synthetic_frame("heart_disease", n=20)
    frame.loc[0, "cp"] = 99
    report = validate_dataframe(frame, get_schema("heart_disease"))
    assert not report.ok
    assert any("cp" in e for e in report.errors)


def test_non_binary_target_is_an_error():
    frame = make_synthetic_frame("heart_disease", n=20)
    frame.loc[0, "target"] = 3
    report = validate_dataframe(frame, get_schema("heart_disease"))
    assert not report.ok


def test_out_of_range_numeric_produces_warning():
    frame = make_synthetic_frame("heart_disease", n=20)
    frame.loc[0, "trestbps"] = 500  # above documented maximum
    report = validate_dataframe(frame, get_schema("heart_disease"))
    assert report.ok  # warnings do not fail validation
    assert any("trestbps" in w for w in report.warnings)


def test_duplicate_rows_are_reported():
    frame = make_synthetic_frame("heart_disease", n=10)
    duplicated = pd.concat([frame, frame], ignore_index=True)
    report = validate_dataframe(duplicated, get_schema("heart_disease"))
    assert report.duplicate_rows == 10


def test_report_round_trips_to_dict():
    frame = make_synthetic_frame("diabetes", n=30)
    report = validate_dataframe(frame, get_schema("diabetes"))
    data = report.to_dict()
    assert data["ok"] is True
    assert "columns" in data and len(data["columns"]) == len(get_schema("diabetes").features)
