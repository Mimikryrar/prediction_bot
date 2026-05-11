# Stock & Crypto Price Prediction Bot — Unified Build Plan

> Authored by: planner (Phase 1, revised)
> Date: 2026-05-11
> Branch: unified-build
> Supersedes: initial Polymarket-scoped plan (commit: "planner: unified build plan")

---

## 0. Context and Scope

**New scope**: This bot predicts **directional price movements on stocks and crypto assets** over a fixed forward horizon. Polymarket is dropped entirely as a prediction target. The handoff docs in `md_files/` remain useful as historical context (especially the Bayesian aggregation framing, Kelly sizing rationale, and the regression-baseline-first principle), but any Polymarket-specific design (Gamma API, CLOB, funder/signer, `p_0 vs market price`) is superseded by this document.

**Prediction target**: binary direction classification — will the asset close higher than today's close in N trading days? Rationale: direction is a clean, well-defined label; a calibrated probability of UP gives a natural Kelly-sizing input; it avoids the additional complexity of return magnitude estimation while still producing an actionable signal. The horizon N is a tunable config value; the MVP default is **5 trading days**.

**Data sources (resolved)**:
- Stocks: `yfinance` (free, key-less, covers US equities + indices). Recommendation to team lead: confirm this is acceptable. It is Yahoo Finance data, which is suitable for MVP/research but has ToS restrictions on commercial redistribution — flag before any production deployment.
- Crypto: **CoinGecko public REST API** (free tier, no key required for basic OHLCV on major pairs, 30 req/min rate limit). Rationale for choosing CoinGecko over Binance: no account or API key needed for MVP; covers a wide set of coins; rate limits are manageable for a small universe. Binance public REST is an easy upgrade path if higher resolution or lower latency is needed later.

**What this build does NOT include**:
- Execution, order placement, wallet management
- Copy-trading signal ingestion
- Polymarket, Gamma API, FinFeedAPI, Bitquery
- LLM-in-the-loop features
- The Bayesian posterior combination with a copy-trading signal (`p_posterior ∝ w_0*p_0 + w_c*p_c`) — that integration point is deferred until the copy bot is stable

The `py_construction/pred_bot.ipynb` notebook is empty and can be used for experimentation; it is not a production deliverable.

---

## 1. Module Boundaries

### `src/data/` — Data Layer (owned by implementer-data)

Responsible for:
- Fetching OHLCV price history for a configured universe of stock tickers via `yfinance`. Stock fetching MUST go through a `StockClient` Protocol/ABC defined in `src/shared/interfaces.py`. The yfinance-backed concrete class is `YFinanceClient` in `src/data/yfinance_client.py`; any future provider (Alpha Vantage, Polygon) implements the same Protocol.
- Fetching OHLCV price history for a configured universe of crypto pairs via CoinGecko REST
- Normalizing both sources into a canonical time-aligned panel of `AssetOHLCV` records
- Persisting raw and processed data to disk (cache layer with TTL)
- Providing a clean, side-effect-free interface: given a list of symbols and a date range, return a normalized panel ready for feature engineering

Does NOT:
- Define or compute features (that is `src/model/features.py`)
- Run models or produce predictions
- Access any wallet, signing, or execution infrastructure

### `src/model/` — Model Layer (owned by implementer-model)

Responsible for:
- Engineering time-series features from the normalized OHLCV panel (returns, momentum, volatility, moving-average crossovers)
- Constructing train/validation/test splits respecting temporal order (no lookahead)
- Training a logistic regression baseline (scikit-learn) predicting 5-day direction
- Persisting model artifacts (joblib + metadata)
- Producing `PredictionResult` (probability of UP, confidence interval) for a given symbol + feature vector
- Evaluating model quality: accuracy, AUC-ROC, Brier score, calibration curve

Does NOT:
- Fetch or cache data (all data comes from `src/data/`)
- Manage secrets or API credentials
- Implement any execution logic

### `src/shared/` — Shared Contract (owned by implementer-data, written first)

Single file: `src/shared/schemas.py`. Defines the Pydantic models that form the handoff between the two modules. Neither module imports from the other — only from `src/shared/`.

---

## 2. File Ownership Per Module

### implementer-data owns exclusively:

```
src/
  data/
    __init__.py
    yfinance_client.py       # fetch stock OHLCV via yfinance
    coingecko_client.py      # fetch crypto OHLCV via CoinGecko public REST (httpx)
    cache.py                 # disk-based cache (JSON/Parquet), TTL-aware
    normalizer.py            # raw source data -> AssetOHLCV, time-aligned panel
    pipeline.py              # orchestrates: fetch -> normalize -> cache -> return panel
  shared/
    __init__.py
    schemas.py               # AssetOHLCV + PredictionResult Pydantic models (see §1)
    interfaces.py            # StockClient Protocol — swappable stock data provider abstraction

tests/
  data/
    __init__.py
    test_yfinance_client.py
    test_coingecko_client.py
    test_normalizer.py
    test_pipeline.py
    conftest.py
    fixtures/
      yfinance_aapl_sample.json    # 90+ days of AAPL OHLCV
      coingecko_btc_sample.json    # 90+ days of BTC/USD OHLCV
```

### implementer-model owns exclusively:

```
src/
  model/
    __init__.py
    features.py              # time-series feature engineering from AssetOHLCV panel
    splitter.py              # temporal train/val/test split, no lookahead
    trainer.py               # fit logistic regression, persist artifact
    predictor.py             # load artifact, produce PredictionResult
    evaluator.py             # accuracy, AUC-ROC, Brier, calibration
    model_store.py           # versioned joblib save/load with metadata JSON

models/                      # gitignored artifact directory (created at runtime)

tests/
  model/
    __init__.py
    test_features.py
    test_splitter.py
    test_trainer.py
    test_predictor.py
    test_evaluator.py
    conftest.py
    fixtures/
      sample_ohlcv_panel.json      # synthetic 90-day panel for fast tests
      sample_predictions.json
```

### Neither implementer creates (planner sets up before Phase 2 starts):

```
pyproject.toml
.env.example
.gitignore                   # includes models/, .env, __pycache__, *.pyc
src/__init__.py
tests/__init__.py
```

**Ownership rule for `src/shared/schemas.py`**: implementer-data creates and owns this file. Any change to the schema after both implementers have started requires agreement from both parties — raise a question to the planner or team lead before modifying.

---

## 3. Shared Interface (Pydantic schemas in `src/shared/schemas.py`)

```python
from pydantic import BaseModel
from datetime import date
from typing import Optional

class AssetOHLCV(BaseModel):
    symbol: str               # e.g. "AAPL" or "BTC-USD"
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
    prediction_date: date     # date the prediction is made
    horizon_days: int         # e.g. 5
    p_up: float               # probability asset closes higher in horizon_days
    confidence: float         # model confidence (e.g. max class probability)
    model_version: str
```

`src/data/` produces lists of `AssetOHLCV`. `src/model/` consumes them and emits `PredictionResult`.

---

## 4. Test Strategy

### Framework

- **pytest** with `pytest-cov` for coverage
- **pytest-mock** (`mocker` fixture) for patching
- `respx` for mocking `httpx` async HTTP calls (CoinGecko client)
- `unittest.mock.patch` for patching `yfinance.download` (yfinance wraps its own HTTP internally)
- All tests are fully offline — no live network calls

### Mocking strategy

| Client | Mock approach |
|---|---|
| `yfinance_client.py` | Patch `yfinance.download` to return a pre-built DataFrame from fixture |
| `coingecko_client.py` | Use `respx` to intercept `httpx` calls and return fixture JSON |
| `cache.py` | Point at `tmp_path` pytest fixture; no mocking needed |

Fixtures live under `tests/data/fixtures/` and `tests/model/fixtures/`. They must cover at least 90 calendar days so temporal split tests have enough data.

### Coverage targets

- `src/data/`: **85%** line coverage minimum
- `src/model/`: **80%** line coverage minimum
- `src/shared/schemas.py`: **100%** (all Pydantic field validation paths exercised)

### Temporal integrity requirement

Every test that exercises `splitter.py` or `trainer.py` must assert that no test-set date appears in the training window. This is the most critical correctness property of the model layer.

### Integration smoke test

`tests/test_smoke.py` (written after both modules complete, owned by neither): loads fixture OHLCV data through `pipeline.py → features.py → trainer.py → predictor.py` end-to-end without mocks, confirming the two modules compose correctly on a small synthetic dataset.

---

## 5. Risk and Assumption Notes

### Rate limits on free APIs

- **CoinGecko free tier**: ~30 requests/minute. For a small universe (10–20 crypto pairs), this is fine. `coingecko_client.py` must implement a simple backoff/retry on HTTP 429. Risk: free tier may be further restricted without notice.
- **yfinance**: no official rate limit documented, but Yahoo Finance throttles aggressive scrapers. For MVP (fetching once per day), this is not a concern. Risk: Yahoo Finance has ToS restrictions on commercial use — confirm acceptability before any production deployment.

### Lookahead bias in time-series features

The single highest risk in any price-prediction model. Mitigations:
- All features must be computed using only data available at or before `prediction_date`.
- `splitter.py` enforces a strict temporal cutoff: train on rows up to date T, validate on T+1..T+k, test on T+k+1..end. No shuffling across the full dataset.
- The `features.py` implementation must use only `shift(1)` or greater lags when computing rolling statistics — never `shift(0)` on the target's own row.
- A test in `test_features.py` must explicitly assert that features at date D do not incorporate close price at date D (the label).

### Regime shift between train and test windows

A model trained on 90 days may capture a single market regime. Mitigations in the MVP:
- Record the train/test date ranges in the model metadata JSON (persisted by `model_store.py`).
- `evaluator.py` reports performance per calendar quarter so regime drift is visible.
- Accept that the MVP baseline may underperform out-of-sample — the goal is a reproducible benchmark, not immediate alpha.

### yfinance ToS for production

yfinance is a scraper of Yahoo Finance data. It is widely used for research but Yahoo's ToS prohibit commercial redistribution. Flag this before any live deployment. Mitigation: stock fetching is implemented behind a `StockClient` Protocol so a licensed source can be swapped in by adding one concrete class and changing the constructor wiring.

### Small universe risk

Starting with a small symbol universe (recommend: 5 stocks + 5 crypto pairs for MVP) means the model has limited cross-sectional data. The baseline should be evaluated per-asset, not only on the pooled dataset, to detect assets where the model has no edge.

### No existing code to migrate

`py_construction/pred_bot.ipynb` is empty. Implementers start from a clean slate.

---

## 6. Prioritized Task List

Dependencies flow top to bottom. A task must not begin until all tasks it lists as dependencies are marked complete.

### Setup (planner creates before Phase 2 implementers start)

- [SETUP-1] `pyproject.toml` — declare dependencies: `pytest`, `pytest-cov`, `pytest-mock`, `respx`, `httpx`, `pydantic`, `yfinance`, `pandas`, `scikit-learn`, `joblib`, `python-dotenv`
- [SETUP-2] `.env.example` — document `COINGECKO_BASE_URL`, `STOCK_SYMBOLS`, `CRYPTO_SYMBOLS`, `HORIZON_DAYS`, `CACHE_DIR`
- [SETUP-3] `.gitignore` — add `models/`, `.env`, `__pycache__/`, `*.pyc`, `*.egg-info/`, `.pytest_cache/`
- [SETUP-4] Stub `__init__.py` files: `src/`, `src/shared/`, `src/data/`, `src/model/`, `tests/`, `tests/data/`, `tests/model/`

### DATA tasks (implementer-data)

- **[D-1] `src/shared/schemas.py`** — Define `AssetOHLCV` and `PredictionResult` Pydantic models (see §3). **BLOCKER for all MODEL tasks.**
- **[D-1b] `src/shared/interfaces.py`** — Define `StockClient` Protocol (or ABC): `get_ohlcv(symbol: str, start: date, end: date) -> list[dict]`. This interface must be defined before `yfinance_client.py` is implemented. Depends on: D-1.
- **[D-2] `src/data/yfinance_client.py`** — Wraps `yfinance.download`; returns list of `dict` (raw, not normalized). Accepts symbol + date range. Depends on: D-1.
- **[D-3] `src/data/coingecko_client.py`** — Async httpx client for CoinGecko `/coins/{id}/market_chart/range`. Returns raw JSON. Implements exponential backoff on 429. Depends on: D-1.
- **[D-4] `src/data/cache.py`** — `DiskCache` class: save/load Parquet (via pandas) keyed by symbol+date range, with TTL in hours. Depends on: nothing (pure utility).
- **[D-5] `src/data/normalizer.py`** — Converts raw yfinance DataFrame and raw CoinGecko JSON into lists of `AssetOHLCV`. Handles missing/NaN values: log and skip the affected row, never crash. Depends on: D-2, D-3.
- **[D-6] `src/data/pipeline.py`** — Orchestrates: check cache → fetch if stale → normalize → write cache → return sorted `list[AssetOHLCV]`. Public API: `get_ohlcv(symbols: list[str], start: date, end: date) -> list[AssetOHLCV]`. Depends on: D-4, D-5.
- **[D-7] `tests/data/`** — Unit tests for D-2 through D-6; all HTTP mocked. Fixture files cover ≥90 calendar days each. Depends on: D-6.

### MODEL tasks (implementer-model)

All MODEL tasks are **blocked on D-1** (schemas.py must exist before any model code imports from `src/shared/`).

- **[M-1] `src/model/features.py`** — Computes per-asset features from a sorted `list[AssetOHLCV]`: log returns, 5/20-day momentum, 20-day rolling volatility, 5/20-day SMA crossover flag, volume z-score. All using only lagged data (no future leak). Returns a `pd.DataFrame` with symbol+date index. Depends on: D-1.
- **[M-2] `src/model/splitter.py`** — Given a feature DataFrame with dates, produces train/val/test slices with a strict temporal cutoff (no shuffling). Configurable split fractions (default: 70/15/15). Depends on: M-1.
- **[M-3] `src/model/trainer.py`** — Builds label column (1 if close[t+horizon] > close[t] else 0), fits `sklearn.linear_model.LogisticRegression` on train split, persists via `model_store`. Depends on: M-2, M-5 (model_store, can be developed in parallel with M-2 once M-1 is done).
- **[M-4] `src/model/predictor.py`** — Loads persisted model via `model_store`, accepts a feature row, returns `PredictionResult`. Depends on: M-3.
- **[M-5] `src/model/model_store.py`** — Saves/loads joblib model artifact alongside a metadata JSON (train date range, feature names, model version, symbol universe). Can be developed in parallel with M-2. Depends on: D-1.
- **[M-6] `src/model/evaluator.py`** — Computes accuracy, AUC-ROC, Brier score, and a per-quarter performance breakdown over a list of `(PredictionResult, actual_outcome)` pairs. Depends on: M-4.
- **[M-7] `tests/model/`** — Unit tests for M-1 through M-6. Must include a temporal-integrity assertion (no test date in train window). Depends on: M-6.

### Integration (after both DATA and MODEL complete)

- **[INT-1] `tests/test_smoke.py`** — Full-path smoke test on synthetic fixture data, no mocks. Depends on: D-7, M-7.
- **[INT-2] Coverage gate** — Run `pytest --cov=src --cov-fail-under=80`; confirm per-module targets met. Depends on: INT-1.

### Dependency order (ASCII diagram)

```
SETUP-1..4
    └── D-1 (schemas)  ← BLOCKER
            ├── D-2 (yfinance client)
            ├── D-3 (coingecko client)   [D-2, D-3 parallel]
            │       └── D-5 (normalizer)
            │               └── D-6 (pipeline) ← D-4 (cache) [parallel with D-5]
            │                       └── D-7 (data tests)
            └── M-1 (features)
                    └── M-2 (splitter) ─────────────────────┐
                                                             │
                    M-5 (model_store, parallel with M-2) ───┤
                                                             ▼
                                                    M-3 (trainer)
                                                         └── M-4 (predictor)
                                                                  └── M-6 (evaluator)
                                                                           └── M-7 (model tests)
                                                                                    └── INT-1 → INT-2
```

---

## 7. Environment Variables (`.env.example`)

```bash
# CoinGecko base URL (override for testing or if using a Pro key later)
COINGECKO_BASE_URL=https://api.coingecko.com/api/v3

# Symbol universes (comma-separated)
STOCK_SYMBOLS=AAPL,MSFT,GOOGL,AMZN,NVDA
CRYPTO_SYMBOLS=bitcoin,ethereum,solana,bnb,ripple

# Prediction horizon in trading days
HORIZON_DAYS=5

# Cache directory (relative to project root)
CACHE_DIR=.cache

# Minimum history in calendar days for training
MIN_HISTORY_DAYS=90
```

No secrets are required for the MVP. CoinGecko public endpoints and yfinance are both key-less. If yfinance is replaced with a paid provider, add that key here.

---

## 8. Open Decisions (1 remaining)

**Decision needed**: Is `yfinance` acceptable as the stock data source for MVP, given its Yahoo Finance ToS restrictions on commercial use?

- If **yes**: proceed with `yfinance_client.py` as specified.
- If **no**: replace with Alpha Vantage free tier (requires `ALPHAVANTAGE_API_KEY`; 25 req/day on free tier — sufficient for MVP with a small symbol universe). The client interface stays the same; only the implementation changes.

No other open decisions remain.
