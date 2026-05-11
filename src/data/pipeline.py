import logging
import os
from datetime import date

from src.data.cache import DiskCache
from src.data.coingecko_client import CoinGeckoClient
from src.data.normalizer import normalize_crypto_records, normalize_stock_records
from src.data.yfinance_client import YFinanceClient
from src.shared.interfaces import StockClient
from src.shared.schemas import AssetOHLCV

logger = logging.getLogger(__name__)

_STOCK_SYMBOLS = [s.strip() for s in os.getenv("STOCK_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,NVDA").split(",")]
_CRYPTO_SYMBOLS = [s.strip() for s in os.getenv("CRYPTO_SYMBOLS", "bitcoin,ethereum,solana,bnb,ripple").split(",")]
_CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
_COINGECKO_BASE_URL = os.getenv("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")


class DataPipeline:
    def __init__(
        self,
        stock_client: StockClient | None = None,
        crypto_client: CoinGeckoClient | None = None,
        cache: DiskCache | None = None,
        stock_symbols: list[str] | None = None,
        crypto_symbols: list[str] | None = None,
    ) -> None:
        self._stock_client = stock_client or YFinanceClient()
        self._crypto_client = crypto_client or CoinGeckoClient(base_url=_COINGECKO_BASE_URL)
        self._cache = cache or DiskCache(cache_dir=_CACHE_DIR)
        self._stock_symbols = stock_symbols if stock_symbols is not None else _STOCK_SYMBOLS
        self._crypto_symbols = crypto_symbols if crypto_symbols is not None else _CRYPTO_SYMBOLS

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
        import pandas as pd
        cached = self._cache.load(symbol, start, end)
        if cached is not None:
            return _df_to_ohlcv(cached, symbol, "stock", "yfinance")
        raw = self._stock_client.get_ohlcv(symbol, start, end)
        normalized = normalize_stock_records(raw, symbol)
        if normalized:
            df = _ohlcv_to_df(normalized)
            self._cache.save(symbol, start, end, df)
        return normalized

    def _fetch_crypto(self, symbol: str, start: date, end: date) -> list[AssetOHLCV]:
        cached = self._cache.load(symbol, start, end)
        if cached is not None:
            return _df_to_ohlcv(cached, symbol, "crypto", "coingecko")
        raw = self._crypto_client.get_ohlcv(symbol, start, end)
        normalized = normalize_crypto_records(raw, symbol)
        if normalized:
            df = _ohlcv_to_df(normalized)
            self._cache.save(symbol, start, end, df)
        return normalized


def _ohlcv_to_df(records: list[AssetOHLCV]):
    import pandas as pd
    rows = [r.model_dump() for r in records]
    return pd.DataFrame(rows)


def _df_to_ohlcv(df, symbol: str, asset_type: str, source: str) -> list[AssetOHLCV]:
    result = []
    for _, row in df.iterrows():
        d = row["date"]
        if hasattr(d, "date"):
            d = d.date()
        result.append(AssetOHLCV(
            symbol=symbol,
            asset_type=asset_type,
            date=d,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            source=source,
        ))
    return result
