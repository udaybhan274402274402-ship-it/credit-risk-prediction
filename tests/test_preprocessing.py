"""Unit tests for loading, target creation, splitting and the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    EXCLUDED_FEATURES,
    MODEL_NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    get_features_and_target,
    load_raw_data,
    make_binary_target,
    split_data,
)


def test_binary_target_mapping():
    df = pd.DataFrame({TARGET_COLUMN: [1, 2, 1, 2, 2]})
    assert make_binary_target(df).tolist() == [0, 1, 0, 1, 1]


def test_features_exclude_sensitive_columns_and_target(synthetic_df):
    for col in EXCLUDED_FEATURES:
        assert col in synthetic_df.columns  # the raw data has them...
    X, y = get_features_and_target(synthetic_df)
    for col in EXCLUDED_FEATURES:
        assert col not in X.columns  # ...but the model input must not
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == len(synthetic_df)
    assert set(y.unique()) == {0, 1}


def test_split_is_reproducible(synthetic_df):
    X, y = get_features_and_target(synthetic_df)
    first = split_data(X, y)
    second = split_data(X, y)
    assert first[0].index.equals(second[0].index)
    assert first[1].index.equals(second[1].index)


def test_split_sizes_and_no_overlap(split):
    X_train, X_test, _, _ = split
    assert len(X_train) == 160
    assert len(X_test) == 40
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_is_stratified(split):
    _, _, y_train, y_test = split
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_preprocessor_learns_only_from_training_data(split):
    X_train, X_test, _, _ = split
    preprocessor = build_preprocessor().fit(X_train)
    learned_mean = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_
    train_mean = X_train[MODEL_NUMERICAL_FEATURES].mean().to_numpy()
    full_mean = pd.concat([X_train, X_test])[MODEL_NUMERICAL_FEATURES].mean().to_numpy()
    assert np.allclose(learned_mean, train_mean)
    assert not np.allclose(learned_mean, full_mean, rtol=1e-9, atol=0)


def test_preprocessor_output_has_no_missing_values_and_fixed_width(split):
    X_train, X_test, _, _ = split
    preprocessor = build_preprocessor()
    train_out = preprocessor.fit_transform(X_train)
    test_out = preprocessor.transform(X_test)
    assert train_out.shape[1] == test_out.shape[1]
    assert not np.isnan(train_out).any()
    assert not np.isnan(test_out).any()


def test_preprocessor_ignores_unseen_category(split):
    X_train, X_test, _, _ = split
    preprocessor = build_preprocessor().fit(X_train)
    normal_row = X_test.iloc[[0]]
    odd_row = normal_row.copy()
    odd_row["purpose"] = "never_seen_before"
    assert preprocessor.transform(odd_row).shape == preprocessor.transform(normal_row).shape


def test_preprocessor_fills_missing_numeric_values(split):
    X_train, X_test, _, _ = split
    preprocessor = build_preprocessor().fit(X_train)
    row = X_test.iloc[[0]].astype({"age": float})
    row["age"] = np.nan
    assert not np.isnan(preprocessor.transform(row)).any()


def test_load_raw_data_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.data")