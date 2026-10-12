from __future__ import annotations

from disease_prediction.features.schema import InferenceFeature, InferenceSchema


def _schema() -> InferenceSchema:
    return InferenceSchema(
        task="toy",
        features=[
            InferenceFeature(name="age", kind="numeric", minimum=18, maximum=120),
            InferenceFeature(name="sex", kind="binary", allowed_values=[0, 1]),
            InferenceFeature(
                name="thal",
                kind="categorical",
                allowed_values=[3, 6, 7],
            ),
            InferenceFeature(name="bp", kind="numeric", minimum=80, maximum=220),
        ],
    )


def test_valid_row_has_no_problems():
    schema = _schema()
    assert schema.validate_row({"age": 50, "sex": 1, "thal": 6, "bp": 120}) == []


def test_out_of_range_and_unknown_category_detected():
    schema = _schema()
    problems = schema.validate_row({"age": 200, "sex": 1, "thal": 9, "bp": 120})
    assert any("age" in p for p in problems)
    assert any("thal" in p for p in problems)


def test_unknown_feature_detected():
    schema = _schema()
    problems = schema.validate_row({"age": 50, "sex": 1, "thal": 6, "bp": 120, "extra": 1})
    assert any("Unknown feature" in p for p in problems)


def test_missing_required_numeric_detected():
    schema = _schema()
    problems = schema.validate_row({"sex": 1, "thal": 6}, require_all=True)
    assert any("Missing required feature 'age'" in p for p in problems)


def test_wrong_type_detected():
    schema = _schema()
    problems = schema.validate_row(
        {"age": "fifty", "sex": 1, "thal": 6, "bp": 120}, require_all=False
    )
    assert any("must be numeric" in p for p in problems)


def test_feature_lookup():
    schema = _schema()
    assert schema.feature("age").kind == "numeric"
    assert schema.feature("missing") is None
    assert schema.feature_names == ["age", "sex", "thal", "bp"]
