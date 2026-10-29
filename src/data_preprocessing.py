"""Data loading and preprocessing utilities for the German Credit dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german.data"

RANDOM_STATE = 42  # fixed seed so results are reproducible
TEST_SIZE = 0.2

# german.data has no header row, so we name the columns ourselves
# (order follows german.doc, attributes 1 to 20, then the target).
COLUMN_NAMES = [
    "checking_account_status",  # 1  categorical
    "duration_months",          # 2  numerical
    "credit_history",           # 3  categorical
    "purpose",                  # 4  categorical
    "credit_amount",            # 5  numerical
    "savings_account",          # 6  categorical
    "employment_since",         # 7  categorical
    "installment_rate",         # 8  numerical (% of disposable income)
    "personal_status_sex",      # 9  categorical (SENSITIVE)
    "other_debtors",            # 10 categorical
    "residence_since",          # 11 numerical
    "property",                 # 12 categorical
    "age",                      # 13 numerical (SENSITIVE)
    "other_installment_plans",  # 14 categorical
    "housing",                  # 15 categorical
    "existing_credits",         # 16 numerical
    "job",                      # 17 categorical
    "num_dependents",           # 18 numerical
    "telephone",                # 19 categorical
    "foreign_worker",           # 20 categorical (SENSITIVE)
    "credit_risk",              # target: 1 = good, 2 = bad
]

TARGET_COLUMN = "credit_risk"

NUMERICAL_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

# Sensitive attributes that are NOT used as model inputs (fairness decision).
# They stay in the raw data so we can audit model behaviour across groups later.
EXCLUDED_FEATURES = ["personal_status_sex", "foreign_worker"]

MODEL_NUMERICAL_FEATURES = [c for c in NUMERICAL_FEATURES if c not in EXCLUDED_FEATURES]
MODEL_CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c not in EXCLUDED_FEATURES]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load german.data into a DataFrame with proper column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m src.download_data"
        )

    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns but found {df.shape[1]}."
        )
    return df


def make_binary_target(df: pd.DataFrame) -> pd.Series:
    """Convert raw labels (1 = good, 2 = bad) to 0 = lower risk, 1 = higher risk."""
    return (df[TARGET_COLUMN] == 2).astype(int).rename("higher_risk")


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model input features (without excluded columns) and the binary target."""
    X = df[MODEL_NUMERICAL_FEATURES + MODEL_CATEGORICAL_FEATURES].copy()
    y = make_binary_target(df)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Reproducible, stratified train-test split (keeps the 70/30 ratio in both parts)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) preprocessing step.

    Numerical: fill missing with median, then standardize.
    Categorical: fill missing with most frequent, then one-hot encode.
    Unknown categories at prediction time are ignored instead of crashing.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, MODEL_NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, MODEL_CATEGORICAL_FEATURES),
        ]
    )