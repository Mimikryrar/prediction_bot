import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prediction_bot.data.cache import DiskCache
from prediction_bot.data.pipeline import DataPipeline
from prediction_bot.shared.schemas import AssetOHLCV

FIXTURES = Path(__file__).parent / "fixtures"

START = date(2023, 10, 2)
END = date(2024, 2, 9)


def _load_aapl_records():
    with open(FIXTURES / "yfinance_aapl_sample.json") as f:
        rows = json.load(f)
    for row in rows:
        row["date"] = date.fromisoformat(row["date"])
    return rows


def _load_btc_coingecko():
    with open(FIXTURES / "coingecko_btc_sample.json") as f:
        return json.load(f)


def test_get_ohlcv_stock(tmp_path):
    mock_stock = MagicMock()
    mock_stock.get_ohlcv.return_value = _load_aapl_records()
    mock_crypto = MagicMock()
    pipeline = DataPipeline(
        stock_client=mock_stock,
        crypto_client=mock_crypto,
        cache=DiskCache(cache_dir=str(tmp_path)),
        stock_symbols=["AAPL"],
        crypto_symbols=[],
    )
    results = pipeline.get_ohlcv(["AAPL"], START, END)
    assert len(results) == 95
    assert all(isinstance(r, AssetOHLCV) for r in results)
    assert all(r.symbol == "AAPL" for r in results)
    assert all(r.asset_type == "stock" for r in results)


def test_get_ohlcv_crypto(tmp_path):
    from prediction_bot.data.coingecko_client import CoinGeckoClient
    mock_crypto = MagicMock(spec=CoinGeckoClient)
    raw_data = _load_btc_coingecko()
    client = CoinGeckoClient(base_url="http://unused")
    mock_crypto.get_ohlcv.return_value = client._parse(raw_data, "bitcoin")
    pipeline = DataPipeline(
        stock_client=MagicMock(),
        crypto_client=mock_crypto,
        cache=DiskCache(cache_dir=str(tmp_path)),
        stock_symbols=[],
        crypto_symbols=["bitcoin"],
    )
    results = pipeline.get_ohlcv(["bitcoin"], START, END)
    assert len(results) >= 90
    assert all(r.asset_type == "crypto" for r in results)


def test_get_ohlcv_uses_cache(tmp_path):
    mock_stock = MagicMock()
    mock_stock.get_ohlcv.return_value = _load_aapl_records()
    pipeline = DataPipeline(
        stock_client=mock_stock,
        crypto_client=MagicMock(),
        cache=DiskCache(cache_dir=str(tmp_path)),
        stock_symbols=["AAPL"],
        crypto_symbols=[],
    )
    pipeline.get_ohlcv(["AAPL"], START, END)
    pipeline.get_ohlcv(["AAPL"], START, END)
    assert mock_stock.get_ohlcv.call_count == 1  # second call hits cache


def test_get_ohlcv_sorted(tmp_path):
    mock_stock = MagicMock()
    records = _load_aapl_records()
    import random
    random.shuffle(records)
    mock_stock.get_ohlcv.return_value = records
    pipeline = DataPipeline(
        stock_client=mock_stock,
        crypto_client=MagicMock(),
        cache=DiskCache(cache_dir=str(tmp_path)),
        stock_symbols=["AAPL"],
        crypto_symbols=[],
    )
    results = pipeline.get_ohlcv(["AAPL"], START, END)
    dates = [r.date for r in results]
    assert dates == sorted(dates)


def test_get_ohlcv_unknown_symbol_logs_warning(tmp_path, caplog):
    import logging
    pipeline = DataPipeline(
        stock_client=MagicMock(),
        crypto_client=MagicMock(),
        cache=DiskCache(cache_dir=str(tmp_path)),
        stock_symbols=["AAPL"],
        crypto_symbols=["bitcoin"],
    )
    with caplog.at_level(logging.WARNING):
        results = pipeline.get_ohlcv(["UNKNOWN_XYZ"], START, END)
    assert results == []
    assert "UNKNOWN_XYZ" in caplog.text


def test_get_ohlcv_empty_symbols(tmp_path):
    pipeline = DataPipeline(
        stock_client=MagicMock(),
        crypto_client=MagicMock(),
        cache=DiskCache(cache_dir=str(tmp_path)),
        stock_symbols=["AAPL"],
        crypto_symbols=["bitcoin"],
    )
    results = pipeline.get_ohlcv([], START, END)
    assert results == []


def test_pipeline_resolves_env_at_construction(monkeypatch, tmp_path):
    """
    Regression for O-6: env vars must be read at __init__ time, not module
    import time, so test code (and future runtime reconfiguration) can set
    STOCK_SYMBOLS / CRYPTO_SYMBOLS via os.environ without restarting.
    """
    monkeypatch.setenv("STOCK_SYMBOLS", "FOO,BAR")
    monkeypatch.setenv("CRYPTO_SYMBOLS", "baz,qux")
    pipeline = DataPipeline(
        stock_client=MagicMock(),
        crypto_client=MagicMock(),
        cache=DiskCache(cache_dir=str(tmp_path)),
    )
    assert pipeline._stock_symbols == ["FOO", "BAR"]
    assert pipeline._crypto_symbols == ["baz", "qux"]
