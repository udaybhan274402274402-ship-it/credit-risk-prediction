"""Data loading utilities for the German Credit dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german.data"

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