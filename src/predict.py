"""Load the saved pipeline and predict for ONE applicant, with input validation.

Run a small demo from the project root:
    python -m src.predict
"""

import json
import logging
import math
from dataclasses import dataclass, field
from numbers import Real

import joblib
import pandas as pd

from src.data_preprocessing import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    PROJECT_ROOT,
    get_features_and_target,
    load_raw_data,
)

logger = logging.getLogger(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

EXPECTED_FEATURES = MODEL_NUMERICAL_FEATURES + MODEL_CATEGORICAL_FEATURES

# Hard limits reject impossible values. The training range only triggers a warning.
HARD_BOUNDS = {
    "duration_months": (1, 120),
    "credit_amount": (1, 1_000_000),
    "installment_rate": (1, 4),
    "residence_since": (1, 4),
    "age": (18, 100),
    "existing_credits": (0, 10),
    "num_dependents": (0, 10),
}


class ValidationError(ValueError):
    """Raised when the applicant input is missing, unknown, or invalid."""


@dataclass
class Prediction:
    risk_score: float          # model output for "higher risk"; NOT a verified probability
    higher_risk: bool          # risk_score >= threshold
    threshold: float
    warnings: list = field(default_factory=list)


def load_artifacts():
    """Load the saved pipeline and its metadata."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.save_model")
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


def validate_applicant(applicant: dict, metadata: dict) -> list:
    """Raise ValidationError for invalid input; return a list of warnings."""
    missing = [f for f in EXPECTED_FEATURES if f not in applicant]
    unexpected = [f for f in applicant if f not in EXPECTED_FEATURES]
    if missing:
        raise ValidationError(f"Missing fields: {missing}")
    if unexpected:
        raise ValidationError(f"Unexpected fields (not used by the model): {unexpected}")

    warnings = []
    for col in MODEL_NUMERICAL_FEATURES:
        value = applicant[col]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValidationError(f"{col} must be a finite number, got {value!r}")
        low, high = HARD_BOUNDS[col]
        if not low <= value <= high:
            raise ValidationError(f"{col} must be between {low} and {high}, got {value}")
        train_range = metadata["numerical_features"][col]
        if not train_range["min"] <= value <= train_range["max"]:
            warnings.append(
                f"{col}={value} is outside the training range "
                f"({train_range['min']:.0f} to {train_range['max']:.0f}); "
                "the estimate may be unreliable."
            )

    for col in MODEL_CATEGORICAL_FEATURES:
        allowed = metadata["categorical_features"][col]
        if str(applicant[col]) not in allowed:
            raise ValidationError(f"{col} must be one of {allowed}, got {applicant[col]!r}")
    return warnings


def predict_applicant(applicant: dict, pipeline=None, metadata=None) -> Prediction:
    """Validate the input, then score it with the SAME pipeline used in training."""
    if pipeline is None or metadata is None:
        pipeline, metadata = load_artifacts()
    warnings = validate_applicant(applicant, metadata)
    row = pd.DataFrame([applicant])[EXPECTED_FEATURES]
    score = float(pipeline.predict_proba(row)[0, 1])
    threshold = float(metadata["threshold"])
    return Prediction(score, score >= threshold, threshold, warnings)


def main() -> None:
    """Demo: score the first 3 rows of the public dataset and show the real label."""
    pipeline, metadata = load_artifacts()
    df = load_raw_data()
    X, y = get_features_and_target(df)
    for i in range(3):
        result = predict_applicant(X.iloc[i].to_dict(), pipeline, metadata)
        label = "higher risk" if y.iloc[i] == 1 else "lower risk"
        print(f"Row {i}: score={result.risk_score:.3f} | "
              f"predicted {'HIGHER' if result.higher_risk else 'LOWER'} risk "
              f"(threshold {result.threshold:.2f}) | dataset label: {label}")

    print("\nValidation demo (age = 5 should be rejected):")
    bad_input = X.iloc[0].to_dict()
    bad_input["age"] = 5
    try:
        predict_applicant(bad_input, pipeline, metadata)
    except ValidationError as error:
        print("Rejected as expected ->", error)


if __name__ == "__main__":
    main()