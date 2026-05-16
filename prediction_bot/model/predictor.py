from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from prediction_bot.model.features import FEATURE_COLS
from prediction_bot.model.model_store import load_model, load_regime_hmm
from prediction_bot.shared.schemas import PredictionResult


class Predictor:
    def __init__(self, artifact_path: Path) -> None:
        artifact_path = Path(artifact_path)
        self._model, self._metadata = load_model(artifact_path)
        self._model_version: str = self._metadata["model_version"]
        self._horizon_days: int = self._metadata.get("horizon_days", 5)
        scaler = self._model.named_steps["scaler"]
        self._scaler_mean = scaler.mean_
        self._scaler_scale = scaler.scale_
        self._clf = self._model.named_steps["clf"]
        # Use feature_names from metadata so regime-augmented models score correctly.
        saved_features = self._metadata.get("feature_names") or FEATURE_COLS
        self._feature_cols = tuple(saved_features)
        self._regime_n_states: int = int(self._metadata.get("regime_n_states", 0))
        self._regime_hmm = load_regime_hmm(artifact_path) if self._regime_n_states > 0 else None

    @property
    def feature_cols(self) -> tuple:
        return self._feature_cols

    @property
    def regime_n_states(self) -> int:
        return self._regime_n_states

    @property
    def regime_hmm(self):
        return self._regime_hmm

    def _build_feature_matrix(self, feature_rows: Sequence[Dict[str, float]]) -> np.ndarray:
        return np.asarray(
            [[row[col] for col in self._feature_cols] for row in feature_rows],
            dtype=float,
        )

    def _predict_proba_rows(self, feature_rows: Sequence[Dict[str, float]]) -> np.ndarray:
        x = self._build_feature_matrix(feature_rows)
        x_scaled = (x - self._scaler_mean) / self._scaler_scale
        return self._clf.predict_proba(x_scaled)

    def predict(self, symbol: str, feature_row: Dict[str, float], prediction_date: date) -> PredictionResult:
        """
        Given a dict of feature values (keys = FEATURE_COLS) and the prediction date,
        return a PredictionResult with p_up and confidence.
        """
        proba = self._predict_proba_rows([feature_row])[0]  # [p_down, p_up]
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

    def predict_many(
        self,
        symbols: Sequence[str],
        feature_rows: Sequence[Dict[str, float]],
        prediction_dates: Sequence[date],
    ) -> list[PredictionResult]:
        if not (len(symbols) == len(feature_rows) == len(prediction_dates)):
            raise ValueError("symbols, feature_rows, and prediction_dates must have the same length")

        probas = self._predict_proba_rows(feature_rows)
        results: list[PredictionResult] = []
        for symbol, prediction_date, proba in zip(symbols, prediction_dates, probas):
            p_up = float(proba[1])
            results.append(PredictionResult(
                symbol=symbol,
                prediction_date=prediction_date,
                horizon_days=self._horizon_days,
                p_up=p_up,
                confidence=float(np.max(proba)),
                model_version=self._model_version,
            ))
        return results
