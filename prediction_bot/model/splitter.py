from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end_date: pd.Timestamp
    val_end_date: pd.Timestamp


def _extract_dates(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        return pd.Series(pd.to_datetime(df["date"]).to_numpy(), index=df.index)
    if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        return pd.Series(pd.to_datetime(df.index.get_level_values("date")).to_numpy(), index=df.index)
    raise ValueError("DataFrame must have a 'date' column or MultiIndex level")


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitResult:
    """
    Split df by sorted unique dates into train / val / test with no shuffling.

    df must have 'date' as a column or as part of a MultiIndex.
    All rows for a given date are kept in the same partition.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    dates = _extract_dates(df)
    unique_dates = pd.Index(sorted(pd.unique(dates)))
    if len(unique_dates) == 0:
        raise ValueError("Cannot split an empty DataFrame")

    n_dates = len(unique_dates)
    train_end = max(1, int(n_dates * train_frac))
    val_end = max(train_end + 1, int(n_dates * (train_frac + val_frac)))
    train_end = min(train_end, n_dates)
    val_end = min(val_end, n_dates)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    sorted_df = df.loc[dates.sort_values().index]
    sorted_dates = dates.loc[sorted_df.index]

    train = sorted_df.loc[sorted_dates.isin(train_dates)]
    val = sorted_df.loc[sorted_dates.isin(val_dates)]
    test = sorted_df.loc[sorted_dates.isin(test_dates)]

    def _max_date(part: pd.DataFrame) -> pd.Timestamp:
        part_dates = _extract_dates(part)
        return pd.Timestamp(part_dates.max())

    return SplitResult(
        train=train,
        val=val,
        test=test,
        train_end_date=_max_date(train),
        val_end_date=_max_date(val) if len(val) > 0 else _max_date(train),
    )
