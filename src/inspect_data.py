"""Print a first inspection of the raw German Credit dataset.

Run from the project root:
    python -m src.inspect_data
"""

import pandas as pd

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_raw_data,
)


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)

    df = load_raw_data()

    print("=== SHAPE (rows, columns) ===")
    print(df.shape)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES (total) ===")
    print(int(df.isna().sum().sum()))

    print("\n=== DUPLICATE ROWS ===")
    print(int(df.duplicated().sum()))

    print("\n=== TARGET DISTRIBUTION (1 = good, 2 = bad) ===")
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    percent = (df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100).round(1)
    print(pd.DataFrame({"count": counts, "percent": percent}))

    print("\n=== NUMERICAL SUMMARY ===")
    print(df[NUMERICAL_FEATURES].describe().T.round(2))

    print("\n=== CATEGORICAL FEATURES: number of unique values ===")
    print(df[CATEGORICAL_FEATURES].nunique())


if __name__ == "__main__":
    main()