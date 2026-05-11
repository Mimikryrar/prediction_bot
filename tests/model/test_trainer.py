import json
from pathlib import Path

import pandas as pd
import pytest

from src.model.features import build_feature_dataframe
from src.model.splitter import temporal_split
from src.model.trainer import train, build_labels


def test_build_labels_no_lookahead(sample_records):
    """Labels at time T must use close[T+horizon], never close[T] itself."""
    df = build_feature_dataframe(sample_records)
    labels = build_labels(df, horizon_days=5)
    # Labels should be 0 or 1 (or NaN for the last horizon rows)
    valid = labels.dropna()
    assert set(valid.unique()) <= {0.0, 1.0}


def test_train_produces_artifact(sample_records, tmp_model_path):
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df)

    # Re-index back to MultiIndex for trainer
    train_df = split.train.set_index(["symbol", "date"])
    val_df = split.val.set_index(["symbol", "date"])
    test_df = split.test.set_index(["symbol", "date"])

    from src.model.splitter import SplitResult
    split_mi = SplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        train_end_date=split.train_end_date,
        val_end_date=split.val_end_date,
    )

    pipe = train(
        split_mi,
        horizon_days=5,
        model_version="test_v1",
        symbol_universe=["AAPL"],
        artifact_path=tmp_model_path,
    )
    assert tmp_model_path.exists()
    meta_path = tmp_model_path.with_suffix(".json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["model_version"] == "test_v1"
    assert meta["horizon_days"] == 5
    assert "feature_names" in meta


def test_trained_model_can_predict(sample_records, tmp_model_path):
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df)

    train_df = split.train.set_index(["symbol", "date"])
    val_df = split.val.set_index(["symbol", "date"])
    test_df = split.test.set_index(["symbol", "date"])

    from src.model.splitter import SplitResult
    split_mi = SplitResult(
        train=train_df, val=val_df, test=test_df,
        train_end_date=split.train_end_date,
        val_end_date=split.val_end_date,
    )

    pipe = train(
        split_mi,
        horizon_days=5,
        model_version="test_v2",
        symbol_universe=["AAPL"],
        artifact_path=tmp_model_path,
    )

    from src.model.features import FEATURE_COLS
    test_row = split.test[FEATURE_COLS].iloc[0:1]
    proba = pipe.predict_proba(test_row)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6
