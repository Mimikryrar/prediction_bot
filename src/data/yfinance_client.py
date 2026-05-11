import logging
from datetime import date

import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceClient:
    """Fetches stock OHLCV data via yfinance. Implements StockClient Protocol."""

    def get_ohlcv(self, symbol: str, start: date, end: date) -> list[dict]:
        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            logger.warning("yfinance returned empty data for %s", symbol)
            return []
        df = df.reset_index()
        records = []
        for _, row in df.iterrows():
            records.append({
                "date": row["Date"].date() if hasattr(row["Date"], "date") else row["Date"],
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            })
        return records
