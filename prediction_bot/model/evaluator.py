from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from prediction_bot.shared.schemas import PredictionResult


@dataclass
class EvalMetrics:
    accuracy: float
    auc_roc: float
    brier_score: float
    n_samples: int
    calibration_pred: list[float] = field(default_factory=list)
    calibration_true: list[float] = field(default_factory=list)
    per_quarter: Dict[str, "EvalMetrics"] = field(default_factory=dict)


def _build_metrics(p_ups: np.ndarray, actuals: np.ndarray, n_bins: int) -> EvalMetrics:
    predictions = (p_ups >= 0.5).astype(int)
    auc = float(roc_auc_score(actuals, p_ups)) if len(np.unique(actuals)) > 1 else float("nan")
    cal_true, cal_pred = calibration_curve(actuals, p_ups, n_bins=min(n_bins, len(actuals)), strategy="uniform")
    return EvalMetrics(
        accuracy=float(np.mean(predictions == actuals)),
        auc_roc=auc,
        brier_score=float(brier_score_loss(actuals, p_ups)),
        n_samples=int(len(actuals)),
        calibration_pred=[float(x) for x in cal_pred],
        calibration_true=[float(x) for x in cal_true],
    )


def evaluate(
    results: List[Tuple[PredictionResult, int]],
    *,
    calibration_bins: int = 10,
) -> EvalMetrics:
    """
    Compute accuracy, AUC-ROC, Brier score, and calibration data over
    a list of (PredictionResult, actual_outcome).

    actual_outcome: 1 = asset closed higher, 0 = did not.
    Also breaks down metrics per calendar quarter based on prediction_date.
    """
    if not results:
        raise ValueError("No results to evaluate")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be > 0")

    p_ups = np.array([r.p_up for r, _ in results], dtype=float)
    actuals = np.array([label for _, label in results], dtype=int)
    metrics = _build_metrics(p_ups, actuals, calibration_bins)

    dates = pd.to_datetime([r.prediction_date for r, _ in results])
    quarters = dates.to_period("Q").astype(str)
    per_quarter: Dict[str, EvalMetrics] = {}

    for q in sorted(set(quarters)):
        mask = quarters == q
        per_quarter[q] = _build_metrics(p_ups[mask], actuals[mask], calibration_bins)

    metrics.per_quarter = per_quarter
    return metrics
