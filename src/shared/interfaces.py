from typing import Protocol, runtime_checkable
from datetime import date


@runtime_checkable
class StockClient(Protocol):
    def get_ohlcv(self, symbol: str, start: date, end: date) -> list[dict]:
        ...
