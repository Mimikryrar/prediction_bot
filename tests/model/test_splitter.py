import pandas as pd
import pytest

from src.model.features import build_feature_dataframe
from src.model.splitter import temporal_split


def test_no_test_date_in_train_window(sample_records):
    """Temporal integrity: no test-set date appears in the training window."""
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df)

    train_dates = set(pd.to_datetime(split.train["date"]).dt.date)
    test_dates = set(pd.to_datetime(split.test["date"]).dt.date)
    overlap = train_dates & test_dates
    assert len(overlap) == 0, f"Test dates found in train: {overlap}"


def test_no_val_date_in_train_window(sample_records):
    """No val-set date appears in training window."""
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df)

    train_dates = set(pd.to_datetime(split.train["date"]).dt.date)
    val_dates = set(pd.to_datetime(split.val["date"]).dt.date)
    overlap = train_dates & val_dates
    assert len(overlap) == 0, f"Val dates found in train: {overlap}"


def test_split_sizes_sum_to_total(sample_records):
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df)
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(df)


def test_train_end_date_before_val(sample_records):
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df)
    assert split.train_end_date <= split.val_end_date


def test_temporal_order_respected(sample_records):
    """All train dates < all val dates < all test dates."""
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df)

    max_train = pd.to_datetime(split.train["date"]).max()
    min_val = pd.to_datetime(split.val["date"]).min()
    max_val = pd.to_datetime(split.val["date"]).max()
    min_test = pd.to_datetime(split.test["date"]).min()

    assert max_train <= min_val, "Train bleeds into val"
    assert max_val <= min_test, "Val bleeds into test"


def test_custom_fractions(sample_records):
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    n = len(df)
    assert len(split.train) == int(n * 0.6)
    assert len(split.val) == int(n * (0.6 + 0.2)) - int(n * 0.6)


def test_invalid_fractions_raises(sample_records):
    df = build_feature_dataframe(sample_records).reset_index()
    with pytest.raises(ValueError):
        temporal_split(df, train_frac=0.8, val_frac=0.3)
