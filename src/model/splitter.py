from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end_date: pd.Timestamp
    val_end_date: pd.Timestamp


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitResult:
    """
    Split df by sorted date order into train / val / test with no shuffling.

    df must have 'date' as a column or as part of a MultiIndex.
    Rows are sorted by date; cutoffs are determined by row count fractions.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    # Extract dates for sorting
    if "date" in df.columns:
        dates = df["date"]
    elif isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        dates = df.index.get_level_values("date")
    else:
        raise ValueError("DataFrame must have a 'date' column or MultiIndex level")

    sorted_df = df.copy()
    sorted_df = sorted_df.iloc[pd.Series(dates).argsort().values]

    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = sorted_df.iloc[:train_end]
    val = sorted_df.iloc[train_end:val_end]
    test = sorted_df.iloc[val_end:]

    def _max_date(part: pd.DataFrame) -> pd.Timestamp:
        if "date" in part.columns:
            return pd.Timestamp(part["date"].max())
        return pd.Timestamp(part.index.get_level_values("date").max())

    return SplitResult(
        train=train,
        val=val,
        test=test,
        train_end_date=_max_date(train),
        val_end_date=_max_date(val) if len(val) > 0 else _max_date(train),
    )
