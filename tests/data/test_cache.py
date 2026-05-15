from datetime import date

import pandas as pd
import pytest

from prediction_bot.data.cache import DiskCache


def test_key_path_accepts_normal_symbol(tmp_path):
    cache = DiskCache(cache_dir=str(tmp_path))
    path = cache._key_path("AAPL", date(2024, 1, 1), date(2024, 1, 2))
    assert path.parent == tmp_path
    assert "AAPL" in path.name


def test_key_path_accepts_hyphenated_symbol(tmp_path):
    cache = DiskCache(cache_dir=str(tmp_path))
    path = cache._key_path("BTC-USD", date(2024, 1, 1), date(2024, 1, 2))
    assert path.parent == tmp_path
    assert "BTC_USD" in path.name


@pytest.mark.parametrize(
    "bad_symbol",
    [
        "../etc/passwd",
        "..\\windows",
        "foo bar",
        "foo;rm",
        "foo\x00bar",
        "",
        "foo/bar",
    ],
)
def test_key_path_rejects_unsafe_symbol(tmp_path, bad_symbol):
    cache = DiskCache(cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid symbol"):
        cache._key_path(bad_symbol, date(2024, 1, 1), date(2024, 1, 2))


def test_save_and_load_roundtrip(tmp_path):
    cache = DiskCache(cache_dir=str(tmp_path), ttl_hours=24)
    df = pd.DataFrame({"date": [date(2024, 1, 1)], "close": [100.0]})
    cache.save("AAPL", date(2024, 1, 1), date(2024, 1, 2), df)
    loaded = cache.load("AAPL", date(2024, 1, 1), date(2024, 1, 2))
    assert loaded is not None
    assert loaded["close"].iloc[0] == 100.0


def test_load_returns_none_when_missing(tmp_path):
    cache = DiskCache(cache_dir=str(tmp_path))
    assert cache.load("AAPL", date(2024, 1, 1), date(2024, 1, 2)) is None


def test_load_expires_stale_entry(tmp_path):
    import os
    import time

    cache = DiskCache(cache_dir=str(tmp_path), ttl_hours=1 / 3600)  # 1 second TTL
    df = pd.DataFrame({"date": [date(2024, 1, 1)], "close": [100.0]})
    cache.save("AAPL", date(2024, 1, 1), date(2024, 1, 2), df)
    # Backdate the file's mtime past the TTL
    path = cache._key_path("AAPL", date(2024, 1, 1), date(2024, 1, 2))
    past = time.time() - 10
    os.utime(path, (past, past))
    assert cache.load("AAPL", date(2024, 1, 1), date(2024, 1, 2)) is None
    assert not path.exists()
