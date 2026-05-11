from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from src.shared.schemas import AssetOHLCV


def build_feature_dataframe(records: List[AssetOHLCV]) -> pd.DataFrame:
    """
    Given a sorted list of AssetOHLCV records (one or more symbols),
    compute per-asset features using only lagged data (no future leak).

    Features per row:
      - log_return_1d       : log(close[t] / close[t-1])
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

    df = pd.DataFrame([r.model_dump() for r in records])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    results = []
    for symbol, grp in df.groupby("symbol", sort=False):
        g = grp.set_index("date").sort_index().copy()

        log_ret = np.log(g["close"] / g["close"].shift(1))

        # All features lagged by 1 so feature[t] uses data up to t-1
        lagged_close = g["close"].shift(1)
        lagged_log_ret = log_ret.shift(1)

        momentum_5d = np.log(lagged_close / lagged_close.shift(5))
        momentum_20d = np.log(lagged_close / lagged_close.shift(20))
        volatility_20d = lagged_log_ret.rolling(20).std()

        sma5 = lagged_close.rolling(5).mean()
        sma20 = lagged_close.rolling(20).mean()
        sma_cross = (sma5 > sma20).astype(float)

        vol_mean20 = g["volume"].shift(1).rolling(20).mean()
        vol_std20 = g["volume"].shift(1).rolling(20).std()
        volume_zscore = (g["volume"].shift(1) - vol_mean20) / vol_std20.replace(0, np.nan)

        feat = pd.DataFrame(
            {
                "symbol": symbol,
                "log_return_1d": log_ret,
                "momentum_5d": momentum_5d,
                "momentum_20d": momentum_20d,
                "volatility_20d": volatility_20d,
                "sma_cross_5_20": sma_cross,
                "volume_zscore": volume_zscore,
                "close": g["close"],
            }
        )
        results.append(feat)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results)
    out = out.reset_index().rename(columns={"index": "date"})
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
