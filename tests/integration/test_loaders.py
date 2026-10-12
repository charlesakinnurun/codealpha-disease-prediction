from __future__ import annotations

import pytest

from disease_prediction.data.loaders import diabetes_raw_path, load_dataset, target_counts
from disease_prediction.data.registry import get_schema
from disease_prediction.data.validate import validate_dataframe


def test_heart_disease_loads_offline():
    df = load_dataset("heart_disease")
    assert len(df) == 303
    assert "target" in df.columns
    assert set(target_counts(df)) <= {0, 1}
    assert df["ca"].isna().any()  # documented missing values preserved as NaN


def test_breast_cancer_loads_and_maps_malignant_positive():
    df = load_dataset("breast_cancer")
    assert len(df) == 569
    # 212 malignant cases in the classic WDBC dataset.
    assert target_counts(df)[1] == 212


@pytest.mark.skipif(not diabetes_raw_path().exists(), reason="diabetes data not downloaded")
def test_diabetes_loads_and_validates():
    df = load_dataset("diabetes")
    assert len(df) == 520
    report = validate_dataframe(df, get_schema("diabetes"))
    assert report.ok, report.errors


def test_loaded_frames_validate_against_their_schemas():
    for task in ("heart_disease", "breast_cancer"):
        df = load_dataset(task)
        report = validate_dataframe(df, get_schema(task))
        assert report.ok, (task, report.errors)
