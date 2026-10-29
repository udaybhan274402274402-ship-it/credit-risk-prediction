"""Check the train/test split and the preprocessing pipeline.

Run from the project root:
    python -m src.check_preprocessing
"""

import numpy as np

from src.data_preprocessing import (
    EXCLUDED_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    build_preprocessor,
    get_features_and_target,
    load_raw_data,
    split_data,
)


def main() -> None:
    df = load_raw_data()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("=== SPLIT SIZES ===")
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print("Higher-risk rate in train:", round(y_train.mean() * 100, 1), "%")
    print("Higher-risk rate in test: ", round(y_test.mean() * 100, 1), "%")

    print("\n=== FAIRNESS CHECK ===")
    print("Excluded from model:", EXCLUDED_FEATURES)
    print("Any excluded column in model input?",
          any(col in X.columns for col in EXCLUDED_FEATURES))

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)  # learn ONLY from train
    X_test_t = preprocessor.transform(X_test)        # apply only, never fit

    print("\n=== AFTER PREPROCESSING ===")
    print("Train matrix:", X_train_t.shape, "| Test matrix:", X_test_t.shape)
    print("NaN values (train, test):",
          int(np.isnan(X_train_t).sum()), int(np.isnan(X_test_t).sum()))

    n_num = len(MODEL_NUMERICAL_FEATURES)
    print("\nScaled numeric means, TRAIN (should be about 0):")
    print(X_train_t[:, :n_num].mean(axis=0).round(3))
    print("Scaled numeric means, TEST (close to 0 but NOT exactly 0):")
    print(X_test_t[:, :n_num].mean(axis=0).round(3))


if __name__ == "__main__":
    main()