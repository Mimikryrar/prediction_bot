import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from prediction_bot.data.coingecko_client import CoinGeckoClient

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "http://test-coingecko.local/api/v3"


def _fixture_data():
    with open(FIXTURES / "coingecko_btc_sample.json") as f:
        return json.load(f)


@respx.mock
def test_get_ohlcv_parses_response(btc_coingecko_response):
    respx.get(f"{BASE_URL}/coins/bitcoin/market_chart/range").mock(
        return_value=httpx.Response(200, json=btc_coingecko_response)
    )
    client = CoinGeckoClient(base_url=BASE_URL)
    records = client.get_ohlcv("bitcoin", date(2023, 10, 2), date(2024, 2, 9))
    assert len(records) >= 90
    first = records[0]
    assert set(first.keys()) == {"date", "open", "high", "low", "close", "volume"}
    assert isinstance(first["close"], float)


@respx.mock
def test_get_ohlcv_sorted_by_date(btc_coingecko_response):
    respx.get(f"{BASE_URL}/coins/bitcoin/market_chart/range").mock(
        return_value=httpx.Response(200, json=btc_coingecko_response)
    )
    client = CoinGeckoClient(base_url=BASE_URL)
    records = client.get_ohlcv("bitcoin", date(2023, 10, 2), date(2024, 2, 9))
    dates = [r["date"] for r in records]
    assert dates == sorted(dates)


@respx.mock
def test_get_ohlcv_retries_on_429(btc_coingecko_response):
    import time
    from unittest.mock import patch

    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(429)
        return httpx.Response(200, json=btc_coingecko_response)

    respx.get(f"{BASE_URL}/coins/bitcoin/market_chart/range").mock(side_effect=handler)
    client = CoinGeckoClient(base_url=BASE_URL)
    with patch("time.sleep"):
        records = client.get_ohlcv("bitcoin", date(2023, 10, 2), date(2024, 2, 9))
    assert len(records) >= 90
    assert call_count == 3


@respx.mock
def test_get_ohlcv_raises_after_max_retries(btc_coingecko_response):
    from unittest.mock import patch

    respx.get(f"{BASE_URL}/coins/bitcoin/market_chart/range").mock(
        return_value=httpx.Response(429)
    )
    client = CoinGeckoClient(base_url=BASE_URL)
    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="rate limit"):
            client.get_ohlcv("bitcoin", date(2023, 10, 2), date(2024, 2, 9))


@respx.mock
def test_get_ohlcv_http_error_raises(btc_coingecko_response):
    respx.get(f"{BASE_URL}/coins/bitcoin/market_chart/range").mock(
        return_value=httpx.Response(500)
    )
    client = CoinGeckoClient(base_url=BASE_URL)
    with pytest.raises(httpx.HTTPStatusError):
        client.get_ohlcv("bitcoin", date(2023, 10, 2), date(2024, 2, 9))
