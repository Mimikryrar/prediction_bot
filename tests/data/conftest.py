import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def aapl_raw_records():
    with open(FIXTURES / "yfinance_aapl_sample.json") as f:
        rows = json.load(f)
    from datetime import date
    for row in rows:
        row["date"] = date.fromisoformat(row["date"])
    return rows


@pytest.fixture
def btc_coingecko_response():
    with open(FIXTURES / "coingecko_btc_sample.json") as f:
        return json.load(f)
