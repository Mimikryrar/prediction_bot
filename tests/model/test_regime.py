from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_bot.model.regime import (
    attach_regime_features,
    compute_market_returns,
    fit_regime_hmm,
    predict_regimes,
    regime_feature_columns,
)


def _synthetic_two_regime_returns(n_per_regime: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    calm = rng.normal(loc=0.001, scale=0.005, size=n_per_regime)
    crisis = rng.normal(loc=-0.002, scale=0.030, size=n_per_regime)
    values = np.concatenate([calm, crisis, calm])
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=dates)


def test_fit_predict_regimes_two_state():
    returns = _synthetic_two_regime_returns()
    hmm = fit_regime_hmm(returns, n_states=2)
    states = predict_regimes(hmm, returns)
    assert states.notna().all()
    assert set(states.unique()) <= {0.0, 1.0}
    assert states.index.equals(returns.index)


def test_fit_rejects_too_few_observations():
    returns = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError):
        fit_regime_hmm(returns, n_states=2)


def test_fit_rejects_invalid_n_states():
    returns = _synthetic_two_regime_returns()
    with pytest.raises(ValueError):
        fit_regime_hmm(returns, n_states=1)


def test_predict_preserves_nan():
    returns = _synthetic_two_regime_returns()
    hmm = fit_regime_hmm(returns, n_states=2)
    masked = returns.copy()
    masked.iloc[:5] = np.nan
    states = predict_regimes(hmm, masked)
    assert states.iloc[:5].isna().all()
    assert states.iloc[5:].notna().all()


def test_compute_market_returns_flat_df():
    df = pd.DataFrame({
        "symbol": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
        "log_return_1d": [0.01, 0.02, -0.01, 0.04],
    })
    means = compute_market_returns(df)
    assert means.loc[pd.Timestamp("2020-01-01")] == pytest.approx(0.0)
    assert means.loc[pd.Timestamp("2020-01-02")] == pytest.approx(0.03)


def test_compute_market_returns_multiindex_df():
    idx = pd.MultiIndex.from_tuples(
        [("A", pd.Timestamp("2020-01-01")), ("A", pd.Timestamp("2020-01-02")),
         ("B", pd.Timestamp("2020-01-01")), ("B", pd.Timestamp("2020-01-02"))],
        names=["symbol", "date"],
    )
    df = pd.DataFrame({"log_return_1d": [0.01, 0.02, -0.01, 0.04]}, index=idx)
    means = compute_market_returns(df)
    assert means.loc[pd.Timestamp("2020-01-02")] == pytest.approx(0.03)


def test_regime_feature_columns():
    assert regime_feature_columns(2) == ["regime_1"]
    assert regime_feature_columns(3) == ["regime_1", "regime_2"]


def test_attach_regime_features_lags_by_one_day():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    feature_df = pd.DataFrame({
        "symbol": ["A"] * 5,
        "date": dates,
        "log_return_1d": [0.01, 0.02, -0.01, 0.04, 0.0],
    }).set_index(["symbol", "date"])

    # regime at each date: alternating 0,1,0,1,0
    regime_states = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], index=dates, dtype=float)

    attached = attach_regime_features(feature_df, regime_states, n_states=2)

    # First row gets NaN lagged regime → dropped
    assert len(attached) == 4
    # regime_1 column == 1 when lagged regime == 1
    # lagged regime for the 4 surviving rows is [0, 1, 0, 1]
    assert attached["regime_1"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_attach_regime_features_three_states():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    feature_df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=dates)
    regime_states = pd.Series([0.0, 1.0, 2.0, 0.0], index=dates, dtype=float)
    attached = attach_regime_features(feature_df, regime_states, n_states=3)
    assert list(attached.columns) == ["close", "regime_1", "regime_2"]
    # lagged regimes for the 3 surviving rows: [0, 1, 2]
    assert attached["regime_1"].tolist() == [0.0, 1.0, 0.0]
    assert attached["regime_2"].tolist() == [0.0, 0.0, 1.0]
