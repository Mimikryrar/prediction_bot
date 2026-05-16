from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


DEFAULT_RANDOM_STATE = 42


def compute_market_returns(feature_df: pd.DataFrame) -> pd.Series:
    """
    Mean log_return_1d across symbols per date.

    `feature_df` is the output of build_feature_dataframe, either flat (with
    symbol and date columns) or MultiIndexed by (symbol, date). Returns a
    Series indexed by date, sorted ascending.
    """
    if isinstance(feature_df.index, pd.MultiIndex) and "date" in feature_df.index.names:
        grouped = feature_df["log_return_1d"].groupby(level="date").mean()
    else:
        grouped = feature_df.groupby("date")["log_return_1d"].mean()
    return grouped.sort_index()


def fit_regime_hmm(
    returns: pd.Series,
    n_states: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> GaussianHMM:
    """
    Fit a GaussianHMM on a 1-D return series. Caller is responsible for
    restricting `returns` to the train window only; this function does no
    windowing of its own.
    """
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    series = returns.dropna()
    if len(series) < n_states * 10:
        raise ValueError(f"Need at least {n_states * 10} observations to fit a {n_states}-state HMM; got {len(series)}")
    x = series.to_numpy(dtype=float).reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=random_state,
    )
    hmm.fit(x)
    return hmm


def predict_regimes(hmm: GaussianHMM, returns: pd.Series) -> pd.Series:
    """
    Decode the most-likely state sequence for `returns` using a fitted HMM.
    NaN inputs are preserved as NaN outputs (no extrapolation).
    """
    out = pd.Series(np.nan, index=returns.index, dtype=float)
    mask = returns.notna()
    if mask.any():
        x = returns[mask].to_numpy(dtype=float).reshape(-1, 1)
        states = hmm.predict(x)
        out.loc[mask] = states.astype(float)
    return out


def regime_feature_columns(n_states: int) -> List[str]:
    """Names of the one-hot regime columns (drop-first to avoid collinearity)."""
    return [f"regime_{i}" for i in range(1, n_states)]


def attach_regime_features(
    feature_df: pd.DataFrame,
    regime_states: pd.Series,
    n_states: int,
) -> pd.DataFrame:
    """
    Add lagged one-hot regime columns to feature_df.

    The regime at date D used as a feature is the regime at D-1 (consistent
    with every other feature being lagged), so the model has no peek at the
    contemporaneous return.

    Rows where the lagged regime is NaN (start of the series) are dropped.
    """
    if n_states < 2:
        raise ValueError("n_states must be >= 2")

    cols = regime_feature_columns(n_states)
    lagged = regime_states.sort_index().shift(1)

    if isinstance(feature_df.index, pd.MultiIndex) and "date" in feature_df.index.names:
        dates = feature_df.index.get_level_values("date")
    elif isinstance(feature_df.index, pd.DatetimeIndex):
        dates = feature_df.index
    else:
        dates = pd.to_datetime(feature_df["date"])

    state_per_row = pd.Series(lagged.reindex(dates).to_numpy(), index=feature_df.index)
    out = feature_df.copy()
    for i, col in enumerate(cols, start=1):
        out[col] = (state_per_row == float(i)).astype(float)
    out.loc[state_per_row.isna(), cols] = np.nan
    return out.dropna(subset=cols)
