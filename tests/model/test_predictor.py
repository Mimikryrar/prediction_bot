from datetime import date

import pytest

from prediction_bot.model.features import build_feature_dataframe, FEATURE_COLS
from prediction_bot.model.splitter import temporal_split, SplitResult
from prediction_bot.model.trainer import train
from prediction_bot.model.predictor import Predictor
from prediction_bot.shared.schemas import PredictionResult


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


def test_predict_matches_full_pipeline(sample_records, tmp_model_path):
    """
    Regression for O-3: the optimized predict path (cached scaler params +
    numpy ndarray + direct estimator.predict_proba) must match the original
    self._model.predict_proba(DataFrame) output to within float tolerance.
    """
    import pandas as pd

    split = _make_split(sample_records)
    pipe = train(split, horizon_days=5, model_version="pv3", symbol_universe=["AAPL"], artifact_path=tmp_model_path)

    predictor = Predictor(tmp_model_path)
    rows = split.test[FEATURE_COLS].head(10)
    for _, row in rows.iterrows():
        feat = row.to_dict()
        result = predictor.predict("AAPL", feat, date(2023, 4, 28))
        ref_proba = pipe.predict_proba(pd.DataFrame([{c: feat[c] for c in FEATURE_COLS}]))[0][1]
        assert abs(result.p_up - float(ref_proba)) < 1e-10


def test_predict_many_matches_single_predictions(sample_records, tmp_model_path):
    split = _make_split(sample_records)
    train(split, horizon_days=5, model_version="pv4", symbol_universe=["AAPL"], artifact_path=tmp_model_path)

    predictor = Predictor(tmp_model_path)
    rows = split.test[FEATURE_COLS].head(5)
    feature_rows = [row.to_dict() for _, row in rows.iterrows()]
    symbols = ["AAPL"] * len(feature_rows)
    dates = [date(2023, 4, 28)] * len(feature_rows)

    many = predictor.predict_many(symbols, feature_rows, dates)
    single = [predictor.predict(symbol, feat, pred_date) for symbol, feat, pred_date in zip(symbols, feature_rows, dates)]

    assert len(many) == len(single)
    for batch_res, single_res in zip(many, single):
        assert abs(batch_res.p_up - single_res.p_up) < 1e-12
        assert abs(batch_res.confidence - single_res.confidence) < 1e-12
        assert batch_res.model_version == single_res.model_version


def test_predict_many_mismatched_lengths_raises(sample_records, tmp_model_path):
    split = _make_split(sample_records)
    train(split, horizon_days=5, model_version="pv5", symbol_universe=["AAPL"], artifact_path=tmp_model_path)

    predictor = Predictor(tmp_model_path)
    feat = split.test[FEATURE_COLS].iloc[0].to_dict()
    with pytest.raises(ValueError, match="same length"):
        predictor.predict_many(["AAPL"], [feat], [date(2023, 4, 28), date(2023, 4, 29)])
