from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class AssetOHLCV(BaseModel):
    symbol: str = Field(min_length=1)
    asset_type: Literal["stock", "crypto"]
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(ge=0.0)
    source: Literal["yfinance", "coingecko"]


class PredictionResult(BaseModel):
    symbol: str = Field(min_length=1)
    prediction_date: date
    horizon_days: int = Field(gt=0)
    p_up: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str = Field(min_length=1)
