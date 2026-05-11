from pydantic import BaseModel
from datetime import date
from typing import Optional


class AssetOHLCV(BaseModel):
    symbol: str
    asset_type: str           # "stock" | "crypto"
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str               # "yfinance" | "coingecko"


class PredictionResult(BaseModel):
    symbol: str
    prediction_date: date
    horizon_days: int
    p_up: float
    confidence: float
    model_version: str
