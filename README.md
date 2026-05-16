# Polymarked Prediction Model-Based Bot

A research-oriented stock and crypto direction prediction project.

## Structure
- `prediction_bot/data/`: fetch, cache, normalize OHLCV data
- `prediction_bot/model/`: features, split, train, predict, evaluate
- `prediction_bot/shared/`: shared schemas/contracts
- `tests/`: offline test suite

## Setup
```bash
python3 -m pip install -e .[dev]
```

## Run tests
```bash
python3 -m pytest -q
python3 -m pytest --cov=prediction_bot --cov-report=term-missing -q
```

## Current model flow
1. Fetch normalized OHLCV data via `DataPipeline`
2. Build features with `build_feature_dataframe`
3. Split with `temporal_split`
4. Train with `train`
5. Load and infer with `Predictor`
6. Evaluate with `evaluate`

## CLI
Train:
```bash
prediction-bot train \
  --symbols AAPL,MSFT,bitcoin \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --model-version v1 \
  --output-path models/v1_train_summary.json
```

To add HMM-based market regime detection as a feature (fit on the train window only, persisted alongside the artifact), pass `--regime-states N` where N is the number of regimes (>=2):
```bash
prediction-bot train ... --regime-states 2
```
Downstream `predict`/`evaluate`/`backtest` automatically pick up the regime HMM from the artifact metadata — no extra flags needed.

Predict:
```bash
prediction-bot predict \
  --symbols AAPL \
  --prediction-date 2024-01-15 \
  --artifact-path models/v1.joblib \
  --output-path models/v1_prediction.json
```

Evaluate:
```bash
prediction-bot evaluate \
  --symbols AAPL,MSFT,bitcoin \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --artifact-path models/v1.joblib \
  --partition test \
  --output-path models/v1_test_metrics.json
```

Backtest:
```bash
prediction-bot backtest \
  --symbols AAPL \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --artifact-path models/v1.joblib \
  --partition test \
  --threshold 0.55 \
  --predictions-output-path models/backtest_predictions.csv \
  --output-path models/backtest_summary.json
```

You can also run it as:
```bash
python3 -m prediction_bot ...
```

## Notes
- Temporal splits are date-boundary-based.
- Model artifacts are trusted-local files only — `joblib.load` executes pickle, so only load artifacts produced by `save_model` from a path you trust.
- External API calls are mocked in tests.

## Status
Research project, not production. The `yfinance` data source is used for local research only; deploying this bot against production trading flow would require swapping `yfinance_client.py` for a licensed market data feed.
