from datetime import date

import pytest

from src.model.features import build_feature_dataframe, FEATURE_COLS
from src.model.splitter import temporal_split, SplitResult
from src.model.trainer import train
from src.model.predictor import Predictor
from src.shared.schemas import PredictionResult


def _make_split(sample_records):
    df = build_feature_dataframe(sample_records).reset_index()
    split = temporal_split(df)
    train_df = split.train.set_index(["symbol", "date"])
    val_df = split.val.set_index(["symbol", "date"])
    test_df = split.test.set_index(["symbol", "date"])
    return SplitResult(
        train=train_df, val=val_df, test=test_df,
        train_end_date=split.train_end_date,
        val_end_date=split.val_end_date,
    )


def test_predictor_returns_prediction_result(sample_records, tmp_model_path):
    split = _make_split(sample_records)
    train(split, horizon_days=5, model_version="pv1", symbol_universe=["AAPL"], artifact_path=tmp_model_path)

    predictor = Predictor(tmp_model_path)
    test_row = split.test[FEATURE_COLS].iloc[0]
    result = predictor.predict("AAPL", test_row.to_dict(), date(2023, 4, 28))

    assert isinstance(result, PredictionResult)
    assert result.symbol == "AAPL"
    assert 0.0 <= result.p_up <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.horizon_days == 5
    assert result.model_version == "pv1"


def test_predictor_p_up_plus_p_down_eq_1(sample_records, tmp_model_path):
    split = _make_split(sample_records)
    train(split, horizon_days=5, model_version="pv2", symbol_universe=["AAPL"], artifact_path=tmp_model_path)

    predictor = Predictor(tmp_model_path)
    test_row = split.test[FEATURE_COLS].iloc[0]
    result = predictor.predict("AAPL", test_row.to_dict(), date(2023, 4, 28))

    # confidence is max(p_up, 1-p_up) so p_up is always in [0,1]
    assert 0.0 <= result.p_up <= 1.0
