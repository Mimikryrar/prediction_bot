import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.yfinance_client import YFinanceClient
from src.shared.interfaces import StockClient

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
