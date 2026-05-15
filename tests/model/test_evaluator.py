from datetime import date, timedelta

import pytest

from prediction_bot.model.evaluator import evaluate, EvalMetrics
from prediction_bot.shared.schemas import PredictionResult


def _make_result(p_up: float, d: date, symbol: str = "AAPL") -> PredictionResult:
    return PredictionResult(
        symbol=symbol,
        prediction_date=d,
        horizon_days=5,
        p_up=p_up,
        confidence=max(p_up, 1 - p_up),
        model_version="test",
    )


def test_perfect_classifier():
    base = date(2023, 1, 3)
    results = [
        (_make_result(0.9, base + timedelta(days=i)), 1)
        for i in range(10)
    ] + [
        (_make_result(0.1, base + timedelta(days=i + 10)), 0)
        for i in range(10)
    ]
    metrics = evaluate(results)
    assert metrics.accuracy == 1.0
    assert abs(metrics.auc_roc - 1.0) < 1e-6
    assert metrics.brier_score < 0.05
    assert metrics.n_samples == 20


def test_random_classifier():
    base = date(2023, 1, 3)
    results = [
        (_make_result(0.5, base + timedelta(days=i)), i % 2)
        for i in range(20)
    ]
    metrics = evaluate(results)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.n_samples == 20


def test_empty_raises():
    with pytest.raises(ValueError):
        evaluate([])


def test_per_quarter_breakdown():
    # Q1 2023: Jan-Mar, Q2 2023: Apr-Jun
    q1_results = [
        (_make_result(0.9, date(2023, 1, 3) + timedelta(days=i)), 1)
        for i in range(5)
    ]
    q2_results = [
        (_make_result(0.1, date(2023, 4, 3) + timedelta(days=i)), 0)
        for i in range(5)
    ]
    metrics = evaluate(q1_results + q2_results)
    assert len(metrics.per_quarter) == 2
    quarters = list(metrics.per_quarter.keys())
    assert any("Q1" in q for q in quarters)
    assert any("Q2" in q for q in quarters)


def test_single_class_auc_is_nan():
    """When all actuals are the same class, AUC should be NaN (not crash)."""
    base = date(2023, 1, 3)
    results = [
        (_make_result(0.7, base + timedelta(days=i)), 1)
        for i in range(10)
    ]
    import math
    metrics = evaluate(results)
    assert math.isnan(metrics.auc_roc)


def test_calibration_output_present():
    base = date(2023, 1, 3)
    results = [
        (_make_result(0.1, base + timedelta(days=0)), 0),
        (_make_result(0.2, base + timedelta(days=1)), 0),
        (_make_result(0.8, base + timedelta(days=2)), 1),
        (_make_result(0.9, base + timedelta(days=3)), 1),
    ]
    metrics = evaluate(results, calibration_bins=4)
    assert len(metrics.calibration_pred) > 0
    assert len(metrics.calibration_pred) == len(metrics.calibration_true)
    assert all(0.0 <= x <= 1.0 for x in metrics.calibration_pred)
    assert all(0.0 <= x <= 1.0 for x in metrics.calibration_true)


def test_invalid_calibration_bins_raises():
    base = date(2023, 1, 3)
    with pytest.raises(ValueError, match="calibration_bins"):
        evaluate([(_make_result(0.5, base), 1)], calibration_bins=0)
