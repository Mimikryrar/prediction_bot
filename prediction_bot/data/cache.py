import logging
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TTL_HOURS = 24
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")


class DiskCache:
    """TTL-aware disk cache for OHLCV DataFrames stored as Parquet files."""

    def __init__(self, cache_dir: str = ".cache", ttl_hours: float = _DEFAULT_TTL_HOURS) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, symbol: str, start: date, end: date) -> Path:
        if not _SYMBOL_PATTERN.match(symbol):
            raise ValueError(f"Invalid symbol for cache key: {symbol!r}")
        safe = symbol.replace("/", "_").replace("-", "_")
        return self.cache_dir / f"{safe}__{start.isoformat()}__{end.isoformat()}.parquet"

    def load(self, symbol: str, start: date, end: date) -> pd.DataFrame | None:
        path = self._key_path(symbol, start, end)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.ttl_seconds:
            logger.debug("Cache expired for %s (age=%.0fs)", symbol, age)
            path.unlink(missing_ok=True)
            return None
        logger.debug("Cache hit for %s", symbol)
        return pd.read_parquet(path)

    def save(self, symbol: str, start: date, end: date, df: pd.DataFrame) -> None:
        path = self._key_path(symbol, start, end)
        df.to_parquet(path, index=False)
        logger.debug("Cached %d rows for %s", len(df), symbol)
