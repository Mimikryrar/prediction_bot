from datetime import date

import pytest

from prediction_bot.data.normalizer import normalize_crypto_records, normalize_stock_records, to_dataframe
from prediction_bot.shared.schemas import AssetOHLCV


def _make_row(d=date(2024, 1, 2), price=100.0, vol=1_000_000.0):
    return {"date": d, "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": vol}


def test_normalize_stock_basic(aapl_raw_records):
    result = normalize_stock_records(aapl_raw_records, "AAPL")
    assert len(result) == len(aapl_raw_records)
    for r in result:
        assert isinstance(r, AssetOHLCV)
        assert r.symbol == "AAPL"
        assert r.asset_type == "stock"
        assert r.source == "yfinance"


def test_normalize_stock_skips_nan_row():
    rows = [_make_row(), {"date": date(2024, 1, 3), "open": float("nan"), "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6}]
    result = normalize_stock_records(rows, "AAPL")
    assert len(result) == 1


def test_normalize_stock_skips_missing_field():
    rows = [_make_row(), {"date": date(2024, 1, 3), "open": 100.0}]
    result = normalize_stock_records(rows, "AAPL")
    assert len(result) == 1


def test_normalize_crypto_basic(btc_coingecko_response):
    from prediction_bot.data.coingecko_client import CoinGeckoClient
    client = CoinGeckoClient(base_url="http://unused")
    raw = client._parse(btc_coingecko_response, "bitcoin")
    result = normalize_crypto_records(raw, "bitcoin")
    assert len(result) >= 90
    for r in result:
        assert r.asset_type == "crypto"
        assert r.source == "coingecko"


def test_to_dataframe_sorted():
    rows = [
        _make_row(date(2024, 1, 4)),
        _make_row(date(2024, 1, 2)),
        _make_row(date(2024, 1, 3)),
    ]
    records = normalize_stock_records(rows, "AAPL")
    df = to_dataframe(records)
    dates = list(df["date"])
    assert dates == sorted(dates)


def test_to_dataframe_empty():
    df = to_dataframe([])
    assert df.empty


def test_normalize_stock_all_fields_present():
    row = _make_row()
    result = normalize_stock_records([row], "MSFT")
    r = result[0]
    assert r.open == row["open"]
    assert r.high == row["high"]
    assert r.low == row["low"]
    assert r.close == row["close"]
    assert r.volume == row["volume"]
    assert r.date == row["date"]
