from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.shared.schemas import PredictionResult


@dataclass
class EvalMetrics:
    accuracy: float
    auc_roc: float
    brier_score: float
    n_samples: int
    per_quarter: Dict[str, "EvalMetrics"] = field(default_factory=dict)


def evaluate(
    results: List[Tuple[PredictionResult, int]],
) -> EvalMetrics:
    """
    Compute accuracy, AUC-ROC, Brier score over a list of (PredictionResult, actual_outcome).
    actual_outcome: 1 = asset closed higher, 0 = did not.

    Also breaks down metrics per calendar quarter based on prediction_date.
    """
    if not results:
        raise ValueError("No results to evaluate")

    p_ups = np.array([r.p_up for r, _ in results])
    actuals = np.array([label for _, label in results])
    predictions = (p_ups >= 0.5).astype(int)

    accuracy = float(np.mean(predictions == actuals))
    auc = float(roc_auc_score(actuals, p_ups)) if len(np.unique(actuals)) > 1 else float("nan")
    brier = float(brier_score_loss(actuals, p_ups))

    # Per-quarter breakdown
    dates = pd.to_datetime([r.prediction_date for r, _ in results])
    quarters = dates.to_period("Q").astype(str)
    per_quarter: Dict[str, EvalMetrics] = {}

    for q in sorted(set(quarters)):
        mask = quarters == q
        q_pups = p_ups[mask]
        q_actuals = actuals[mask]
        q_preds = (q_pups >= 0.5).astype(int)
        q_auc = (
            float(roc_auc_score(q_actuals, q_pups))
            if len(np.unique(q_actuals)) > 1
            else float("nan")
        )
        per_quarter[q] = EvalMetrics(
            accuracy=float(np.mean(q_preds == q_actuals)),
            auc_roc=q_auc,
            brier_score=float(brier_score_loss(q_actuals, q_pups)),
            n_samples=int(mask.sum()),
        )

    return EvalMetrics(
        accuracy=accuracy,
        auc_roc=auc,
        brier_score=brier,
        n_samples=len(results),
        per_quarter=per_quarter,
    )
