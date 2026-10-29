"""Unit tests for input validation and prediction."""

import pandas as pd
import pytest

from src import predict as predict_module
from src.data_preprocessing import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERICAL_FEATURES
from src.predict import (
    EXPECTED_FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    Prediction,
    ValidationError,
    load_artifacts,
    predict_applicant,
    validate_applicant,
)


def make_valid_applicant(metadata: dict) -> dict:
    """Build a valid applicant: midpoint of each numeric range, first allowed category."""
    applicant = {}
    for col in MODEL_NUMERICAL_FEATURES:
        limits = metadata["numerical_features"][col]
        applicant[col] = int((limits["min"] + limits["max"]) // 2)
    for col in MODEL_CATEGORICAL_FEATURES:
        applicant[col] = metadata["categorical_features"][col][0]
    return applicant


@pytest.fixture
def valid_applicant(metadata):
    return make_valid_applicant(metadata)


def test_valid_applicant_returns_consistent_prediction(trained_pipeline, metadata, valid_applicant):
    result = predict_applicant(valid_applicant, trained_pipeline, metadata)
    assert isinstance(result, Prediction)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.higher_risk == (result.risk_score >= metadata["threshold"])
    assert result.warnings == []


def test_prediction_matches_direct_pipeline_call(trained_pipeline, metadata, valid_applicant):
    """The app path and a direct call must use the same pipeline and give the same score."""
    result = predict_applicant(valid_applicant, trained_pipeline, metadata)
    row = pd.DataFrame([valid_applicant])[EXPECTED_FEATURES]
    expected = trained_pipeline.predict_proba(row)[0, 1]
    assert result.risk_score == pytest.approx(expected)


def test_missing_field_is_rejected(metadata, valid_applicant):
    del valid_applicant["age"]
    with pytest.raises(ValidationError, match="Missing"):
        validate_applicant(valid_applicant, metadata)


def test_sensitive_attribute_is_rejected(metadata, valid_applicant):
    valid_applicant["personal_status_sex"] = "A93"
    with pytest.raises(ValidationError, match="Unexpected"):
        validate_applicant(valid_applicant, metadata)


@pytest.mark.parametrize(
    "field, value",
    [
        ("age", 5),
        ("age", 150),
        ("duration_months", 0),
        ("credit_amount", -100),
        ("age", "forty"),
        ("age", None),
        ("age", float("nan")),
        ("age", float("inf")),
        ("age", True),
    ],
)
def test_invalid_numeric_values_are_rejected(metadata, valid_applicant, field, value):
    valid_applicant[field] = value
    with pytest.raises(ValidationError):
        validate_applicant(valid_applicant, metadata)


def test_unknown_category_is_rejected(metadata, valid_applicant):
    valid_applicant["housing"] = "A999"
    with pytest.raises(ValidationError):
        validate_applicant(valid_applicant, metadata)


def test_value_outside_training_range_gives_warning_not_error(metadata, valid_applicant):
    valid_applicant["age"] = 90  # above the training maximum, but within allowed bounds
    warnings = validate_applicant(valid_applicant, metadata)
    assert any("age" in message for message in warnings)


def test_threshold_controls_the_flag(trained_pipeline, metadata, valid_applicant):
    low_threshold = {**metadata, "threshold": 0.0}
    high_threshold = {**metadata, "threshold": 1.01}
    assert predict_applicant(valid_applicant, trained_pipeline, low_threshold).higher_risk is True
    assert predict_applicant(valid_applicant, trained_pipeline, high_threshold).higher_risk is False


def test_missing_model_file_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError, match="save_model"):
        load_artifacts()


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and METADATA_PATH.exists()),
    reason="Run `python -m src.save_model` first to create the model files.",
)
def test_saved_model_loads_and_predicts():
    pipeline, metadata = load_artifacts()
    result = predict_applicant(make_valid_applicant(metadata), pipeline, metadata)
    assert 0.0 <= result.risk_score <= 1.0
    assert 0.0 < metadata["threshold"] < 1.0