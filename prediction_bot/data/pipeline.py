import logging
import os
from datetime import date

import pandas as pd

from prediction_bot.data.cache import DiskCache
from prediction_bot.data.coingecko_client import CoinGeckoClient
from prediction_bot.data.normalizer import normalize_crypto_records, normalize_stock_records
from prediction_bot.data.yfinance_client import YFinanceClient
from prediction_bot.shared.interfaces import StockClient
from prediction_bot.shared.schemas import AssetOHLCV

logger = logging.getLogger(__name__)

_DEFAULT_STOCK_SYMBOLS = "AAPL,MSFT,GOOGL,AMZN,NVDA"
_DEFAULT_CRYPTO_SYMBOLS = "bitcoin,ethereum,solana,bnb,ripple"
_DEFAULT_CACHE_DIR = ".cache"
_DEFAULT_COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


class DataPipeline:
    def __init__(
        self,
        stock_client: StockClient | None = None,
        crypto_client: CoinGeckoClient | None = None,
        cache: DiskCache | None = None,
        stock_symbols: list[str] | None = None,
        crypto_symbols: list[str] | None = None,
    ) -> None:
        coingecko_base_url = os.getenv("COINGECKO_BASE_URL", _DEFAULT_COINGECKO_BASE_URL)
        cache_dir = os.getenv("CACHE_DIR", _DEFAULT_CACHE_DIR)
        self._stock_client = stock_client or YFinanceClient()
        self._crypto_client = crypto_client or CoinGeckoClient(base_url=coingecko_base_url)
        self._cache = cache or DiskCache(cache_dir=cache_dir)
        if stock_symbols is None:
            stock_symbols = [s.strip() for s in os.getenv("STOCK_SYMBOLS", _DEFAULT_STOCK_SYMBOLS).split(",")]
        if crypto_symbols is None:
            crypto_symbols = [s.strip() for s in os.getenv("CRYPTO_SYMBOLS", _DEFAULT_CRYPTO_SYMBOLS).split(",")]
        self._stock_symbols = stock_symbols
        self._crypto_symbols = crypto_symbols

    def get_ohlcv(self, symbols: list[str], start: date, end: date) -> list[AssetOHLCV]:
        stock_syms = [s for s in symbols if s in self._stock_symbols]
        crypto_syms = [s for s in symbols if s in self._crypto_symbols]
        unknown = [s for s in symbols if s not in self._stock_symbols and s not in self._crypto_symbols]
        if unknown:
            logger.warning("Unknown symbols (not in stock or crypto universe): %s", unknown)

        results: list[AssetOHLCV] = []
        for sym in stock_syms:
            results.extend(self._fetch_stock(sym, start, end))
        for sym in crypto_syms:
            results.extend(self._fetch_crypto(sym, start, end))
        return sorted(results, key=lambda r: (r.symbol, r.date))

    def _fetch_stock(self, symbol: str, start: date, end: date) -> list[AssetOHLCV]:
        cached = self._cache.load(symbol, start, end)
        if cached is not None:
            return _df_to_ohlcv(cached, symbol, "stock", "yfinance")
        raw = self._stock_client.get_ohlcv(symbol, start, end)
        normalized = normalize_stock_records(raw, symbol)
        if normalized:
            self._cache.save(symbol, start, end, _ohlcv_to_df(normalized))
        return normalized

    def _fetch_crypto(self, symbol: str, start: date, end: date) -> list[AssetOHLCV]:
        cached = self._cache.load(symbol, start, end)
        if cached is not None:
            return _df_to_ohlcv(cached, symbol, "crypto", "coingecko")
        raw = self._crypto_client.get_ohlcv(symbol, start, end)
        normalized = normalize_crypto_records(raw, symbol)
        if normalized:
            self._cache.save(symbol, start, end, _ohlcv_to_df(normalized))
        return normalized


def _ohlcv_to_df(records: list[AssetOHLCV]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [r.date for r in records],
        "open": [r.open for r in records],
        "high": [r.high for r in records],
        "low": [r.low for r in records],
        "close": [r.close for r in records],
        "volume": [r.volume for r in records],
    })


def _df_to_ohlcv(df: pd.DataFrame, symbol: str, asset_type: str, source: str) -> list[AssetOHLCV]:
    result: list[AssetOHLCV] = []
    for row in df.itertuples(index=False):
        d = row.date.date() if hasattr(row.date, "date") else row.date
        result.append(AssetOHLCV(
            symbol=symbol,
            asset_type=asset_type,
            date=d,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
            source=source,
        ))
    return result
