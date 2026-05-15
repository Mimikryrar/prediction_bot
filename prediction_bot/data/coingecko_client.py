import logging
import time
from datetime import date, datetime, timezone

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.coingecko.com/api/v3"
_MAX_RETRIES = 4
_INITIAL_BACKOFF = 2.0


class CoinGeckoClient:
    """Fetches crypto OHLCV data from CoinGecko public REST API."""

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def get_ohlcv(self, coin_id: str, start: date, end: date) -> list[dict]:
        from_ts = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
        to_ts = int(datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params = {"vs_currency": "usd", "from": from_ts, "to": to_ts}

        backoff = _INITIAL_BACKOFF
        for attempt in range(_MAX_RETRIES):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(url, params=params)
                if response.status_code == 429:
                    logger.warning("CoinGecko rate limited (attempt %d); sleeping %.1fs", attempt + 1, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                data = response.json()
                return self._parse(data, coin_id)
            except httpx.HTTPStatusError as exc:
                logger.error("CoinGecko HTTP error for %s: %s", coin_id, exc)
                raise
        raise RuntimeError(f"CoinGecko rate limit not resolved after {_MAX_RETRIES} retries for {coin_id}")

    def _parse(self, data: dict, coin_id: str) -> list[dict]:
        prices = data.get("prices", [])
        volumes = {int(ts): v for ts, v in data.get("total_volumes", [])}
        records: dict[str, dict] = {}
        for ts_ms, price in prices:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
            key = dt.isoformat()
            if key not in records:
                records[key] = {
                    "date": dt,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volumes.get(int(ts_ms), 0.0),
                }
            else:
                records[key]["high"] = max(records[key]["high"], price)
                records[key]["low"] = min(records[key]["low"], price)
                records[key]["close"] = price
        return sorted(records.values(), key=lambda r: r["date"])
