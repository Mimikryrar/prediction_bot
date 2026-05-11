# Polymarket Prediction Bot — Unified Build Plan

> Authored by: planner (Phase 1)
> Date: 2026-05-11
> Branch: unified-build

---

## 0. Context and Scope

This plan governs Phase 2 implementation only: building the **prediction bot** (`p_0` emitter) and its data ingestion layer from scratch, in this empty repository. The copy-trading bot described in the handoff docs lives in a separate repo (`polymarked_copy_trading_bot`) and is **out of scope** here.

The three research PDFs in `Res/` (Bayesian regression in finance, adversarial synthesis on market data, hidden Markov model for market regimes) inform the model architecture. They are not required reading before Phase 2 starts but should be consulted when implementer-model chooses the model family.

The `py_construction/pred_bot.ipynb` notebook is empty and can be used as a scratch pad — it is not part of the production deliverable.

---

## 1. Module Boundaries

### `src/data/` — Data Layer (owned by implementer-data)

Responsible for:
- Fetching and caching market metadata (Gamma API)
- Fetching cross-venue OHLCV price data (FinFeedAPI)
- Fetching on-chain resolved-market history (Bitquery GraphQL)
- Normalizing all external data into canonical internal schemas (Pydantic models)
- Persisting raw and processed data to disk (cache layer)
- Providing a clean, side-effect-free interface to `src/model/`

Does NOT:
- Run models or produce probabilities
- Place orders
- Access wallet/signing infrastructure

### `src/model/` — Model Layer (owned by implementer-model)

Responsible for:
- Loading features produced by `src/data/`
- Training and persisting a regression baseline model
- Producing `p_0` (probability) + confidence/variance per market
- Evaluating model quality (Brier score, log-loss) over resolved markets
- Exposing a clean prediction API consumed by future integration code

Does NOT:
- Fetch external data directly (all data comes from `src/data/`)
- Manage secrets or API keys
- Implement execution or order logic

### Shared Interface (contract between the two modules)

The single handoff point is a Pydantic schema defined in `src/shared/schemas.py`:

```python
class MarketFeatures(BaseModel):
    market_id: str
    question: str
    category: str
    end_date: datetime
    current_price_yes: float          # from Gamma API
    cross_venue_prices: dict[str, float]  # venue -> price, from FinFeedAPI
    volume_24h: float
    liquidity: float
    days_to_resolution: float
    resolved: bool
    outcome: Optional[int]            # 1=YES, 0=NO, None if unresolved

class PredictionResult(BaseModel):
    market_id: str
    p_0: float                        # model probability for YES
    variance: float                   # model uncertainty
    timestamp: datetime
    model_version: str
```

`src/data/` produces `MarketFeatures`. `src/model/` consumes `MarketFeatures` and emits `PredictionResult`. Neither module imports from the other.

---

## 2. File Ownership Per Module

### implementer-data owns exclusively:

```
src/
  data/
    __init__.py
    gamma_client.py          # Gamma API market metadata fetcher
    finfeed_client.py        # FinFeedAPI cross-venue OHLCV fetcher
    bitquery_client.py       # Bitquery GraphQL resolved-market history
    cache.py                 # disk-based caching (JSON/SQLite)
    normalizer.py            # raw API responses → MarketFeatures
    pipeline.py              # orchestrates fetch + normalize + cache
  shared/
    __init__.py
    schemas.py               # MarketFeatures + PredictionResult Pydantic models

tests/
  data/
    __init__.py
    test_gamma_client.py
    test_finfeed_client.py
    test_bitquery_client.py
    test_normalizer.py
    test_pipeline.py
    conftest.py              # shared fixtures for data tests
    fixtures/
      gamma_market_sample.json
      finfeed_ohlcv_sample.json
      bitquery_resolved_sample.json
```

### implementer-model owns exclusively:

```
src/
  model/
    __init__.py
    features.py              # feature engineering from MarketFeatures
    trainer.py               # train + persist regression baseline
    predictor.py             # load model, produce PredictionResult
    evaluator.py             # Brier score, log-loss on resolved markets
    model_store.py           # versioned model artifact save/load

tests/
  model/
    __init__.py
    test_features.py
    test_trainer.py
    test_predictor.py
    test_evaluator.py
    conftest.py
    fixtures/
      sample_features.json
      sample_resolved_markets.json
```

### Neither implementer creates:

```
config/
  .env.example               # created by planner in UNIFIED_PLAN.md (this doc)
  settings.py                # created by planner for Phase 2 kickoff
pyproject.toml               # project-wide, created once before Phase 2 starts
```

Note: `src/shared/schemas.py` is created by implementer-data as the first task (task D-1 below), since the model implementer blocks on it. implementer-model must not modify it unilaterally — any schema change requires both implementers to agree.

---

## 3. Test Strategy

### Framework

- **pytest** with `pytest-cov` for coverage
- **pytest-mock** (`mocker` fixture) for patching external API calls
- Test files live under `tests/data/` and `tests/model/` mirroring the source tree

### Mocking strategy for external APIs

All external network calls (Gamma API, FinFeedAPI, Bitquery) are mocked at the HTTP level using `responses` (for `requests`-based clients) or `pytest-httpx` (for `httpx`-based clients). JSON fixtures under `tests/data/fixtures/` provide realistic but static response data. No test should make a live network call.

The `cache.py` module provides a `DiskCache` that can be pointed at a temp directory in tests (`tmp_path` fixture from pytest).

### Coverage targets

- `src/data/`: 85% line coverage minimum
- `src/model/`: 80% line coverage minimum
- `src/shared/schemas.py`: 100% (Pydantic validation paths)

### Test naming convention

`test_<module>_<behavior>` — e.g. `test_normalizer_handles_missing_cross_venue_price`.

### Integration smoke test

One lightweight integration test file `tests/test_pipeline_smoke.py` (owned by neither implementer; written after both modules are done) that loads fixture data through the full `pipeline.py → predictor.py` path without any mocks, confirming the two modules compose correctly.

---

## 4. Risk and Assumption Notes

### API access

- **Gamma API**: free, no key required for public markets. Assumption: rate limits are not a problem for a small market set. Risk: endpoint schema changes; pin the API version if available.
- **FinFeedAPI**: requires a paid subscription (`FINFEED_API_KEY`). Risk: the key may not yet exist. implementer-data should build `finfeed_client.py` to fail loudly (not silently skip) when the key is absent.
- **Bitquery**: requires an API key (`BITQUERY_API_KEY`). Risk: GraphQL schema may differ from examples in the handoff docs; implementer-data should validate against live schema before writing tests.

### Secrets management

- All keys in `.env` (git-ignored). `.env.example` documents required vars with no values.
- Required vars: `FINFEED_API_KEY`, `BITQUERY_API_KEY`, `GAMMA_BASE_URL` (defaultable).
- Do not hardcode any key anywhere in source. `src/data/` reads keys from environment only.

### Model training data availability

- Historical resolved-market data comes from Bitquery. Risk: limited history depth on free tier; implementer-model should document minimum required history (recommend: 90 days of resolved markets).
- The regression baseline does not require a GPU. Risk: if the dataset is too small (<200 resolved markets), Brier score estimates will be noisy; note this in the evaluator output.

### Architecture conflict resolution

The handoff docs describe a Phase 7 service split (copy bot + prediction bot as separate services). That is **future scope**. For this build, the prediction bot is a standalone Python package — no message queue, no service boundary, no integration with the copy bot yet. `PredictionResult` is the output format that will eventually plug into the Bayesian posterior formula (`p_posterior ∝ w_0*p_0 + w_c*p_c`), but the copy-trading side of that equation is not built here.

The master context doc (`polymarket_copytrading_master_context.md`) treats Phase 7 as blocked until the copy bot is live-ready. This build plan builds the prediction bot MVP independently, consistent with the handoff_2 doc's recommendation: "start with a regression baseline first." There is no conflict — both docs agree on regression-first; they differ only in phase numbering convention (master context calls it Phase 7; handoff_2 calls it Phase 4). This plan treats it as the current goal.

### Naming conflict

handoff_2.md uses `p_0` for the prediction bot probability; tools_handoff_1.md uses `p_own`. This plan standardizes on `p_0` (matches `PredictionResult.p_0` above).

### No existing code to preserve

The notebook `py_construction/pred_bot.ipynb` is empty. There is no existing state to migrate. Implementers start from a clean slate.

---

## 5. Prioritized Task List

Dependencies flow top to bottom. No task should begin until all tasks it depends on are marked complete.

### Setup (before Phase 2 implementers start)

- [SETUP-1] Create `pyproject.toml` with `pytest`, `pytest-cov`, `pytest-mock`, `responses`, `pydantic`, `requests` or `httpx`, `python-dotenv`, `scikit-learn` as dependencies.
- [SETUP-2] Create `.env.example` documenting `FINFEED_API_KEY`, `BITQUERY_API_KEY`, `GAMMA_BASE_URL`.
- [SETUP-3] Create `src/__init__.py`, `src/shared/__init__.py`, `tests/__init__.py`, `tests/data/__init__.py`, `tests/model/__init__.py`.

### DATA tasks (implementer-data)

- [D-1] **schemas.py** — Define `MarketFeatures` and `PredictionResult` Pydantic models in `src/shared/schemas.py`. This is the first task and BLOCKS all MODEL tasks.
- [D-2] **gamma_client.py** — Fetch market metadata from Gamma API. Returns list of raw dicts. Depends on: D-1.
- [D-3] **bitquery_client.py** — GraphQL query for resolved-market history per market. Depends on: D-1.
- [D-4] **finfeed_client.py** — Fetch cross-venue OHLCV from FinFeedAPI. Fails loudly if key absent. Depends on: D-1.
- [D-5] **normalizer.py** — Map raw API responses to `MarketFeatures`. Handles missing fields gracefully (log + skip, not crash). Depends on: D-2, D-3, D-4.
- [D-6] **cache.py** — DiskCache class: save/load JSON blobs by key, with TTL. Depends on: nothing (pure utility).
- [D-7] **pipeline.py** — Orchestrates: fetch via clients → normalize → cache. Exposes `get_features(market_ids: list[str]) -> list[MarketFeatures]`. Depends on: D-5, D-6.
- [D-8] **tests/data/** — Unit tests for all data modules with mocked HTTP. Depends on: D-7.

### MODEL tasks (implementer-model)

All MODEL tasks are blocked on D-1 (schemas.py).

- [M-1] **features.py** — Engineer feature vectors from `MarketFeatures` (e.g. cross-venue basis, days-to-resolution, log-price). Depends on: D-1.
- [M-2] **trainer.py** — Train a logistic regression (or ridge regression) baseline on historical resolved markets. Persists model artifact to `models/` dir. Depends on: M-1.
- [M-3] **predictor.py** — Load persisted model, produce `PredictionResult` for a `MarketFeatures` input. Depends on: M-2.
- [M-4] **evaluator.py** — Compute Brier score and log-loss over a list of resolved `PredictionResult` + outcomes. Depends on: M-3.
- [M-5] **model_store.py** — Versioned save/load for model artifacts (joblib + metadata JSON). Depends on: M-2.
- [M-6] **tests/model/** — Unit tests for all model modules with fixture data. Depends on: M-5, M-4.

### Integration (after both DATA and MODEL complete)

- [INT-1] `tests/test_pipeline_smoke.py` — Full path smoke test. Depends on: D-8, M-6.
- [INT-2] Run `pytest --cov=src` and verify coverage targets. Depends on: INT-1.

### Dependency order summary

```
SETUP-1, SETUP-2, SETUP-3
    └── D-1 (schemas)
            ├── D-2, D-3, D-4 (clients) [parallel]
            │       └── D-5 (normalizer)
            │               └── D-7 (pipeline) ← D-6 (cache) [parallel with D-5]
            │                       └── D-8 (data tests)
            └── M-1 (features)
                    └── M-2 (trainer) ← M-5 (model_store) [parallel with M-3]
                            └── M-3 (predictor)
                                    └── M-4 (evaluator)
                                            └── M-6 (model tests)
                                                    └── INT-1 → INT-2
```

---

## 6. Environment Variables (`.env.example`)

```bash
# Gamma API (public, no key needed for basic endpoints)
GAMMA_BASE_URL=https://gamma-api.polymarket.com

# FinFeedAPI — paid subscription required
FINFEED_API_KEY=

# Bitquery — free tier available, key required
BITQUERY_API_KEY=

# Optional: limit which market categories to fetch
TARGET_CATEGORIES=politics,crypto,sports
```

---

## 7. Open Decisions (require team-lead resolution before Phase 2 starts)

1. **Which HTTP client?** `requests` (simpler) vs `httpx` (async-capable). Recommend `httpx` if implementer-data wants async pipeline; `requests` otherwise. Pick one — do not mix.
2. **FinFeedAPI key availability**: if the key does not exist yet, D-4 and M-2 (training) will need to run against fixture data only. Is that acceptable for Phase 2?
3. **Target market category for the baseline**: the handoff docs recommend starting narrow. Which category? (politics / crypto / sports)
4. **Minimum resolved-market history window**: 30 days? 90 days? This affects whether there's enough data to train a non-trivial model.
