import logging
from datetime import date

import pandas as pd

from src.shared.schemas import AssetOHLCV

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {"date", "open", "high", "low", "close", "volume"}


def _is_valid_row(row: dict) -> bool:
    if not _REQUIRED_FIELDS.issubset(row.keys()):
        return False
    for field in ("open", "high", "low", "close", "volume"):
        val = row[field]
        if val is None or (isinstance(val, float) and (val != val)):  # NaN check
            return False
    return True


def normalize_stock_records(records: list[dict], symbol: str) -> list[AssetOHLCV]:
    result = []
    for row in records:
        if not _is_valid_row(row):
            logger.warning("Skipping invalid stock row for %s: %s", symbol, row)
            continue
        result.append(AssetOHLCV(
            symbol=symbol,
            asset_type="stock",
            date=row["date"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            source="yfinance",
        ))
    return result


def normalize_crypto_records(records: list[dict], symbol: str) -> list[AssetOHLCV]:
    result = []
    for row in records:
        if not _is_valid_row(row):
            logger.warning("Skipping invalid crypto row for %s: %s", symbol, row)
            continue
        result.append(AssetOHLCV(
            symbol=symbol,
            asset_type="crypto",
            date=row["date"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            source="coingecko",
        ))
    return result


def to_dataframe(records: list[AssetOHLCV]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = [r.model_dump() for r in records]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df
