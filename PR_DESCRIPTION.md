# PR Description

## Summary
This PR continues the build from the earlier stopping point and upgrades the project into a cleaner, more usable prediction package. It includes package cleanup, temporal split hardening, schema validation improvements, evaluation enhancements, performance optimizations, and a new CLI for train/predict/evaluate workflows.

## What changed

### Package and structure
- Renamed the top-level application package from `src` to `prediction_bot`
- Updated imports across the codebase and tests
- Updated packaging and coverage configuration in `pyproject.toml`

### Data and model improvements
- Changed temporal splitting to use date-boundary semantics instead of row-count-only slicing
- Hardened shared schemas with stronger constraints and explicit allowed values
- Added calibration outputs to model evaluation
- Optimized label generation in training
- Optimized prediction path and added batch scoring with `predict_many()`
- Reduced unnecessary dataframe/object conversion overhead in the data pipeline

### CLI and usability
- Added `prediction_bot/cli.py`
- Added module entrypoint via `prediction_bot/__main__.py`
- Added installable script entrypoint: `prediction-bot`
- Added CLI commands for:
  - `train`
  - `predict`
  - `evaluate`
- Added optional JSON output file support for CLI commands

### Documentation and release readiness
- Added root `README.md`
- Updated historical/docs references after package rename
- Added `release_checklist.md`
- Added code review report files for traceability

### Tests
- Expanded tests for:
  - split behavior
  - evaluator calibration behavior
  - batch prediction behavior
  - CLI commands
  - module entrypoint smoke coverage

## Validation
- `python3 -m pytest -q` → **72 passed**
- `python3 -m pytest --cov=prediction_bot --cov-report=term-missing -q` → **97% coverage**

## Why this matters
These changes improve:
- correctness of temporal validation boundaries
- safety of shared contracts
- usability via CLI workflows
- maintainability of package structure
- performance for training/prediction internals
- release readiness and documentation quality

## Suggested reviewer focus
- `prediction_bot/model/splitter.py`
- `prediction_bot/shared/schemas.py`
- `prediction_bot/model/evaluator.py`
- `prediction_bot/model/predictor.py`
- `prediction_bot/cli.py`
- `pyproject.toml`

## Example usage
Train:
```bash
prediction-bot train \
  --symbols AAPL,MSFT,bitcoin \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --model-version v1 \
  --output-path models/v1_train_summary.json
```

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
