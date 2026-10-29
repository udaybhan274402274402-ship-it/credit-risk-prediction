"""Shared test fixtures. They use small synthetic data, so tests need no downloads."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    get_features_and_target,
    split_data,
)
from src.train import build_pipeline

N_ROWS = 200

# Value ranges for the fake numeric columns.
NUMERIC_RANGES = {
    "duration_months": (4, 72),
    "credit_amount": (250, 18000),
    "installment_rate": (1, 4),
    "residence_since": (1, 4),
    "age": (19, 75),
    "existing_credits": (1, 4),
    "num_dependents": (1, 2),
}


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """A fake dataset with the same columns as german.data (raw coding: 1 = good, 2 = bad)."""
    rng = np.random.default_rng(0)
    data = {}
    for col in NUMERICAL_FEATURES:
        low, high = NUMERIC_RANGES[col]
        data[col] = rng.integers(low, high + 1, size=N_ROWS)
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice([f"{col}_a", f"{col}_b", f"{col}_c"], size=N_ROWS)
    df = pd.DataFrame(data)
    # Make risk depend a little on duration, so both classes exist.
    noise = rng.normal(0, 10, size=N_ROWS)
    df[TARGET_COLUMN] = np.where(df["duration_months"] + noise > 45, 2, 1)
    return df


@pytest.fixture(scope="session")
def split(synthetic_df):
    """(X_train, X_test, y_train, y_test) from the same split function the project uses."""
    X, y = get_features_and_target(synthetic_df)
    return split_data(X, y)


@pytest.fixture(scope="session")
def trained_pipeline(split):
    """A small real pipeline (preprocessing + Logistic Regression) fitted on fake data."""
    X_train, _, y_train, _ = split
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture(scope="session")
def metadata(split) -> dict:
    """Metadata in the same structure that src/save_model.py writes."""
    X_train = split[0]
    return {
        "threshold": 0.3,
        "numerical_features": {
            col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
            for col in MODEL_NUMERICAL_FEATURES
        },
        "categorical_features": {
            col: sorted(str(v) for v in X_train[col].unique())
            for col in MODEL_CATEGORICAL_FEATURES
        },
    }