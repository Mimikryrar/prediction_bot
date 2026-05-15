from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prediction_bot.model.features import FEATURE_COLS
from prediction_bot.model.model_store import save_model
from prediction_bot.model.splitter import SplitResult


def build_labels(df: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    """
    For each (symbol, date) row, label = 1 if close[t+horizon] > close[t] else 0.
    Rows where the future close is unavailable are dropped (NaN label).

    df must have 'close' as a column and 'date' either as a column or MultiIndex level.
    """
    close = df["close"]
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        future_close = close.groupby(level="symbol", sort=False).shift(-horizon_days)
    else:
        future_close = close.shift(-horizon_days)

    labels = (future_close > close).astype(float)
    labels[future_close.isna()] = float("nan")
    return labels


def train(
    split: SplitResult,
    *,
    horizon_days: int = 5,
    model_version: str = "v1",
    symbol_universe: List[str],
    artifact_path: Path,
) -> Pipeline:
    """
    Fit a logistic regression on the train split, persist the artifact.
    Returns the fitted sklearn Pipeline (scaler + logistic regression).
    """
    train_df = split.train.copy()
    labels = build_labels(train_df, horizon_days=horizon_days)

    valid_mask = labels.notna()
    X_train = train_df.loc[valid_mask, FEATURE_COLS]
    y_train = labels[valid_mask]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    # Determine train date range
    if isinstance(train_df.index, pd.MultiIndex) and "date" in train_df.index.names:
        dates = train_df.index.get_level_values("date")
    else:
        dates = pd.to_datetime(train_df["date"])

    train_start = pd.Timestamp(dates.min()).date()
    train_end = pd.Timestamp(dates.max()).date()

    save_model(
        pipe,
        artifact_path,
        model_version=model_version,
        symbol_universe=symbol_universe,
        feature_names=FEATURE_COLS,
        train_start=train_start,
        train_end=train_end,
        extra={"horizon_days": horizon_days},
    )

    return pipe
