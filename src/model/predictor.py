from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.model.features import FEATURE_COLS
from src.model.model_store import load_model
from src.shared.schemas import PredictionResult


class Predictor:
    def __init__(self, artifact_path: Path) -> None:
        self._model, self._metadata = load_model(Path(artifact_path))
        self._model_version: str = self._metadata["model_version"]
        self._horizon_days: int = self._metadata.get("horizon_days", 5)

    def predict(self, symbol: str, feature_row: Dict[str, float], prediction_date: date) -> PredictionResult:
        """
        Given a dict of feature values (keys = FEATURE_COLS) and the prediction date,
        return a PredictionResult with p_up and confidence.
        """
        X = pd.DataFrame([{col: feature_row[col] for col in FEATURE_COLS}])
        proba = self._model.predict_proba(X)[0]  # [p_down, p_up]
        p_up = float(proba[1])
        confidence = float(np.max(proba))

        return PredictionResult(
            symbol=symbol,
            prediction_date=prediction_date,
            horizon_days=self._horizon_days,
            p_up=p_up,
            confidence=confidence,
            model_version=self._model_version,
        )
