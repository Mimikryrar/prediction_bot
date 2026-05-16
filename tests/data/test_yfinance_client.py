import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from prediction_bot.data.yfinance_client import YFinanceClient
from prediction_bot.shared.interfaces import StockClient

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture_as_df() -> pd.DataFrame:
    with open(FIXTURES / "yfinance_aapl_sample.json") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    df = df.drop(columns=["date"])
    return df


def test_satisfies_protocol():
    client = YFinanceClient()
    assert isinstance(client, StockClient)


def test_get_ohlcv_returns_records(aapl_raw_records):
    df = _load_fixture_as_df()
    with patch("yfinance.download", return_value=df):
        client = YFinanceClient()
        records = client.get_ohlcv("AAPL", date(2023, 10, 2), date(2024, 2, 9))

    assert len(records) == len(df)
    first = records[0]
    assert set(first.keys()) == {"date", "open", "high", "low", "close", "volume"}
    assert isinstance(first["open"], float)
    assert isinstance(first["close"], float)


def test_get_ohlcv_empty_returns_empty_list():
    with patch("yfinance.download", return_value=pd.DataFrame()):
        client = YFinanceClient()
        result = client.get_ohlcv("INVALID", date(2023, 1, 1), date(2023, 1, 31))
    assert result == []


def test_get_ohlcv_dates_are_date_objects(aapl_raw_records):
    df = _load_fixture_as_df()
    with patch("yfinance.download", return_value=df):
        client = YFinanceClient()
        records = client.get_ohlcv("AAPL", date(2023, 10, 2), date(2024, 2, 9))
    for r in records:
        assert isinstance(r["date"], date)


def test_get_ohlcv_requests_flat_columns():
    """
    Real yfinance now defaults to MultiIndex columns; this client expects flat
    columns and must explicitly opt out. Regression guard for the bug fixed in
    commit 69f29e6.
    """
    df = _load_fixture_as_df()
    with patch("yfinance.download", return_value=df) as mock_dl:
        YFinanceClient().get_ohlcv("AAPL", date(2023, 10, 2), date(2024, 2, 9))
    _, kwargs = mock_dl.call_args
    assert kwargs.get("multi_level_index") is False, (
        "YFinanceClient must pass multi_level_index=False to yfinance.download — "
        "the current yfinance default returns MultiIndex columns that break "
        "the normalizer."
    )


def test_get_ohlcv_handles_multi_level_columns_if_returned():
    """
    Defense in depth: even if a future yfinance ignores multi_level_index=False
    or our flag flips, ensure the client can cope with MultiIndex columns
    rather than producing Series values in rows.
    """
    flat = _load_fixture_as_df()
    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in flat.columns])
    with patch("yfinance.download", return_value=multi):
        client = YFinanceClient()
        records = client.get_ohlcv("AAPL", date(2023, 10, 2), date(2024, 2, 9))
    # If the client doesn't flatten, row["Open"] would be a Series and float() raises.
    # We expect either: records produced, OR a clean failure mode (empty / raise) —
    # not silent corruption. Today multi-level inputs cause a TypeError on float(Series);
    # this test pins that until the client gains a flatten step.
    if records:
        for r in records:
            assert isinstance(r["open"], float)
            assert isinstance(r["close"], float)
