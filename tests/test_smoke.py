"""
End-to-end smoke test (INT-1 / O-2 from Phase 3 audit).

Wires pipeline -> features -> trainer -> predictor on synthetic OHLCV data.
Only the external stock/crypto clients are mocked; every internal stage
(normalizer, cache, feature engineering, training, model store, predictor)
runs for real.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from prediction_bot.data.cache import DiskCache
from prediction_bot.data.pipeline import DataPipeline
from prediction_bot.model.features import FEATURE_COLS, build_feature_dataframe
from prediction_bot.model.predictor import Predictor
from prediction_bot.model.splitter import SplitResult, temporal_split
from prediction_bot.model.trainer import train
from prediction_bot.shared.schemas import PredictionResult


def _synth_ohlcv(symbol: str, n_days: int, seed: int) -> list[dict]:
    """Generate a deterministic synthetic OHLCV series as raw dicts (yfinance-shape)."""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 2)
    close = 100.0
    rows: list[dict] = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        drift = rng.normal(0.0005, 0.015)
        close = max(1.0, close * (1.0 + drift))
        open_ = close * (1.0 + rng.normal(0.0, 0.003))
        high = max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.004)))
        low = min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.004)))
        volume = float(rng.integers(1_000_000, 5_000_000))
        rows.append({
            "date": d,
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": volume,
        })
    return rows


def test_smoke_end_to_end(tmp_path):
    """Stock + crypto: pipeline -> features -> train -> predict, no internal mocks."""
    n_days = 200
    stock_raw = _synth_ohlcv("AAPL", n_days, seed=1)
    crypto_raw = _synth_ohlcv("bitcoin", n_days, seed=2)

    mock_stock = MagicMock()
    mock_stock.get_ohlcv.return_value = stock_raw
    mock_crypto = MagicMock()
    mock_crypto.get_ohlcv.return_value = crypto_raw

    pipeline = DataPipeline(
        stock_client=mock_stock,
        crypto_client=mock_crypto,
        cache=DiskCache(cache_dir=str(tmp_path / "cache")),
        stock_symbols=["AAPL"],
        crypto_symbols=["bitcoin"],
    )

    records = pipeline.get_ohlcv(["AAPL", "bitcoin"], stock_raw[0]["date"], stock_raw[-1]["date"])
    assert len(records) == 2 * n_days
    assert {r.asset_type for r in records} == {"stock", "crypto"}

    feats = build_feature_dataframe(records).reset_index()
    assert not feats.empty
    assert set(FEATURE_COLS).issubset(feats.columns)
    assert feats[FEATURE_COLS].notna().all().all()

    split = temporal_split(feats)
    split_mi = SplitResult(
        train=split.train.set_index(["symbol", "date"]),
        val=split.val.set_index(["symbol", "date"]),
        test=split.test.set_index(["symbol", "date"]),
        train_end_date=split.train_end_date,
        val_end_date=split.val_end_date,
    )

    artifact = tmp_path / "smoke_model.joblib"
    train(
        split_mi,
        horizon_days=5,
        model_version="smoke_v1",
        symbol_universe=["AAPL", "bitcoin"],
        artifact_path=artifact,
    )
    assert artifact.exists()
    assert artifact.with_suffix(".json").exists()

    predictor = Predictor(artifact)
    test_row = split.test[FEATURE_COLS].iloc[0].to_dict()
    result = predictor.predict("AAPL", test_row, date(2024, 1, 1))

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.p_up <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.model_version == "smoke_v1"
    assert result.horizon_days == 5
