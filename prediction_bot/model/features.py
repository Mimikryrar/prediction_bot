from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from prediction_bot.shared.schemas import AssetOHLCV


def build_feature_dataframe(records: List[AssetOHLCV]) -> pd.DataFrame:
    """
    Given a sorted list of AssetOHLCV records (one or more symbols),
    compute per-asset features using only lagged data (no future leak).

    Features per row:
      - log_return_1d       : log(close[t-1] / close[t-2])   (1-day return, fully lagged)
      - momentum_5d         : log(close[t-1] / close[t-6])   (5-day, fully lagged)
      - momentum_20d        : log(close[t-1] / close[t-21])
      - volatility_20d      : rolling std of log_return_1d over last 20 days (shift 1)
      - sma_cross_5_20      : 1 if SMA5[t-1] > SMA20[t-1] else 0
      - volume_zscore       : (volume[t-1] - mean_vol_20[t-1]) / std_vol_20[t-1]

    All rolling windows shift by 1 so that the feature at date t does not
    incorporate the close price at date t (which would be the label source).

    Returns a DataFrame indexed by (symbol, date) with NaN rows for
    insufficient history dropped.
    """
    if not records:
        return pd.DataFrame()

    # Build only the columns features actually need — model_dump() per record
    # serializes every field through pydantic and is ~12% of total runtime.
    df = pd.DataFrame({
        "symbol": [r.symbol for r in records],
        "date": pd.to_datetime([r.date for r in records]),
        "close": [r.close for r in records],
        "volume": [r.volume for r in records],
    })
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    by_sym = df.groupby("symbol", sort=False)
    close = df["close"]

    # Single-shift lagged columns (native groupby.shift — vectorized in C)
    lagged_close = by_sym["close"].shift(1)
    close_shift6 = by_sym["close"].shift(6)
    close_shift21 = by_sym["close"].shift(21)
    lagged_vol = by_sym["volume"].shift(1)

    # log_ret[t] = log(close[t]/close[t-1]); lagged_log_ret[t] = log_ret[t-1]
    log_ret = np.log(close / lagged_close)
    lagged_log_ret = log_ret.groupby(df["symbol"], sort=False).shift(1)

    momentum_5d = np.log(lagged_close / close_shift6)
    momentum_20d = np.log(lagged_close / close_shift21)

    # Native grouped rolling — keeps the work in C, no python-level apply per group.
    # The result has a (symbol, original_index) MultiIndex; drop the symbol level
    # so it aligns back to df's RangeIndex.
    volatility_20d = lagged_log_ret.groupby(df["symbol"], sort=False).rolling(20).std().droplevel(0)
    sma5 = lagged_close.groupby(df["symbol"], sort=False).rolling(5).mean().droplevel(0)
    sma20 = lagged_close.groupby(df["symbol"], sort=False).rolling(20).mean().droplevel(0)
    sma_cross = (sma5 > sma20).astype(float)

    vol_mean20 = lagged_vol.groupby(df["symbol"], sort=False).rolling(20).mean().droplevel(0)
    vol_std20 = lagged_vol.groupby(df["symbol"], sort=False).rolling(20).std().droplevel(0)
    volume_zscore = (lagged_vol - vol_mean20) / vol_std20.replace(0, np.nan)

    out = pd.DataFrame({
        "symbol": df["symbol"].values,
        "date": df["date"].values,
        "log_return_1d": lagged_log_ret.values,
        "momentum_5d": momentum_5d.values,
        "momentum_20d": momentum_20d.values,
        "volatility_20d": volatility_20d.values,
        "sma_cross_5_20": sma_cross.values,
        "volume_zscore": volume_zscore.values,
        "close": close.values,
    })
    out = out.set_index(["symbol", "date"])
    out = out.dropna(subset=["momentum_20d", "volatility_20d", "volume_zscore"])
    return out


FEATURE_COLS = [
    "log_return_1d",
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "sma_cross_5_20",
    "volume_zscore",
]
