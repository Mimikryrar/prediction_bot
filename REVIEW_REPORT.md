# Phase 3 Audit: REVIEW_REPORT.md

**Reviewer**: reviewer  
**Branch**: unified-build (off main)  
**Date**: 2026-05-11  
**Commits reviewed**: d736a6c → 807f41b (5 commits, 38 files, 3758 insertions)

---

## 1. Test Results

**Runner**: `python3 -m pytest tests/ -v --tb=short`

| Suite | Collected | Passed | Failed | Errors |
|---|---|---|---|---|
| tests/data/ | 22 | 22 | 0 | 0 |
| tests/model/ | 23 | 23 | 0 | 0 |
| **Total** | **45** | **45** | **0** | **0** |

**Runtime**: 3.03 s (full suite)

**Coverage** (`pytest --cov=src --cov-report=term-missing`):

| Module | Stmts | Miss | Cover | Plan target |
|---|---|---|---|---|
| src/data/cache.py | 30 | 3 | 90% | 85% ✓ |
| src/data/coingecko_client.py | 46 | 3 | 93% | 85% ✓ |
| src/data/normalizer.py | 38 | 2 | 95% | 85% ✓ |
| src/data/pipeline.py | 66 | 2 | 97% | 85% ✓ |
| src/data/yfinance_client.py | 15 | 0 | 100% | 85% ✓ |
| src/model/evaluator.py | 34 | 0 | 100% | 80% ✓ |
| src/model/features.py | 36 | 1 | 97% | 80% ✓ |
| src/model/model_store.py | 27 | 0 | 100% | 80% ✓ |
| src/model/predictor.py | 20 | 0 | 100% | 80% ✓ |
| src/model/splitter.py | 32 | 4 | 88% | 80% ✓ |
| src/model/trainer.py | 41 | 6 | 85% | 80% ✓ |
| src/shared/schemas.py | 20 | 0 | 100% | 100% ✓ |
| **TOTAL** | **409** | **21** | **95%** | — |

All per-module coverage targets met. The missing lines are non-critical:
- `cache.py:31-33` — the `path.unlink(missing_ok=True)` TTL-expiry branch (covered by logic, not by a TTL-expiry test)
- `coingecko_client.py:61-63` — HTTPStatusError re-raise after exhausted retries
- `normalizer.py:47-48` — NaN float check fallback path
- `pipeline.py:64,84` — empty-result guards in `_fetch_stock`/`_fetch_crypto`
- `splitter.py:35-38,54` — edge-case date-extraction paths for plain `date` columns
- `trainer.py:34-38,72` — non-MultiIndex label-building path and a post-fit branch

**Note**: The `pytest` shell alias installed by the RTK hook filters output and reports "No tests collected." Tests must be invoked via `python3 -m pytest` directly. This is an environment issue, not a code issue — but it means the plan's `pytest --cov-fail-under=80` gate (INT-2) cannot be run via the bare `pytest` command on this machine.

---

## 2. Correctness Findings

### C-1 (MEDIUM): `log_return_1d` is not fully lagged — it uses `close[t]`

**File**: `src/model/features.py:46`

```python
log_ret = np.log(g["close"] / g["close"].shift(1))
```

`log_return_1d` is computed as `log(close[t] / close[t-1])`, which incorporates the current-day close at row `t`. This feature is then placed in the feature vector for date `t` without an additional shift. All *other* features (`momentum_5d`, `momentum_20d`, `volatility_20d`, `sma_cross_5_20`, `volume_zscore`) correctly shift by 1 before use.

The plan states: *"features at date D do not incorporate close price at date D"* and *"never `shift(0)` on the target's own row."* `log_return_1d` violates this because it uses `close[t]` in its numerator.

Impact: minor lookahead; `log_return_1d[t]` encodes `close[t]` which is also the base for the label `close[t+5] > close[t]`. This leaks the denominator of the label into the features, giving the model a statistically unfair advantage.

Fix: shift `log_return_1d` by 1 before inclusion (i.e., `lagged_log_ret` is already computed and used for volatility — use it as `log_return_1d` in the output frame as well).

### C-2 (LOW): `CoinGeckoClient._parse` approximates OHLCV from price ticks

**File**: `src/data/coingecko_client.py:44-58`

The CoinGecko `/market_chart/range` endpoint returns intraday price ticks, not OHLCV candles. The client synthesizes open/high/low/close by treating the first tick of a day as open and the last as close. This is a reasonable approximation for daily data but produces open != exchange open and may produce a single-tick "candle" when the API returns sparse data. This is acknowledged nowhere in the code or plan. Not a bug per se, but a correctness note for future consumers.

### C-3 (LOW): `temporal_split` sorts by row position, not by date value

**File**: `src/model/splitter.py:28-32`

```python
sorted_df = sorted_df.iloc[pd.Series(dates).argsort().values]
```

`pd.Series(dates).argsort()` operates on positional index 0..N-1. If `df` has a non-default integer index (e.g. after a `reset_index()` that preserves an old index), the argsort result may silently misalign. All current callers reset the index before calling `temporal_split`, so this does not currently cause failures, but the code is fragile. A safer implementation would use `df.sort_values("date")`.

### C-4 (LOW): Missing `tests/test_smoke.py` (INT-1 from plan)

The plan requires an integration smoke test (`tests/test_smoke.py`) that runs the full pipeline end-to-end without mocks. This file was not created. The plan explicitly lists it as the final integration deliverable (INT-1) after both implementer tasks complete. All unit tests pass but no end-to-end composition test exists.

### C-5 (LOW): `DataPipeline` reads env vars at import time

**File**: `src/data/pipeline.py:13-16`

Module-level `os.getenv(...)` calls execute when the module is first imported. This means test code that sets `STOCK_SYMBOLS` or `CRYPTO_SYMBOLS` via `monkeypatch.setenv` after import will not affect `_STOCK_SYMBOLS` / `_CRYPTO_SYMBOLS`. Tests currently work around this by injecting `stock_symbols` / `crypto_symbols` via the constructor, but the module-level globals are a footgun for future callers who forget to pass constructor arguments.

---

## 3. Security Findings

### S-1 (LOW): No input validation on symbol strings passed to cache key

**File**: `src/data/cache.py:22-23`

```python
safe = symbol.replace("/", "_").replace("-", "_")
return self.cache_dir / f"{safe}__{start.isoformat()}__{end.isoformat()}.parquet"
```

The sanitization replaces `/` and `-` but does not strip `..`, spaces, or other path characters. A symbol string like `"../../etc/passwd"` would still produce a path traversal. In the current system, symbols come from an env-var-controlled whitelist (`STOCK_SYMBOLS`, `CRYPTO_SYMBOLS`), so exploitation requires control over the environment. However, if `get_ohlcv` is ever exposed to external input without pre-filtering against the whitelist, this becomes a path traversal vulnerability.

Recommendation: assert that the sanitized symbol matches `[A-Za-z0-9_]+` before constructing the path, or use `hashlib.sha256` to hash the cache key.

### S-2 (LOW): `model_store.load_model` deserializes joblib without integrity check

**File**: `src/model/model_store.py:58-63`

`joblib.load` on an untrusted file is equivalent to `pickle.load` — arbitrary code execution. The current use case (loading from a local `models/` directory written by the same process) carries no practical risk. If the artifact path is ever derived from user input or a network path, this becomes critical. No fix needed now, but flag before any API or multi-tenant deployment.

### S-3 (INFO): No secrets present in codebase

`.env.example` contains only base URLs and symbol lists — no API keys, tokens, or credentials. `models/` and `.env` are correctly gitignored. CoinGecko and yfinance are both key-less for the MVP. No hardcoded credentials found.

---

## 4. Plan-Adherence Gaps

| Plan Item | Status | Notes |
|---|---|---|
| SETUP-1..4 (pyproject, .env.example, .gitignore, stubs) | ✓ Complete | All present and correct |
| D-1 `schemas.py` | ✓ Complete | Matches §3 schema exactly |
| D-1b `interfaces.py` | ✓ Complete | `StockClient` Protocol with `runtime_checkable` |
| D-2 `yfinance_client.py` | ✓ Complete | Wraps `yf.download`, returns `list[dict]` |
| D-3 `coingecko_client.py` | ✓ Complete | httpx, exponential backoff on 429 |
| D-4 `cache.py` | ✓ Complete | Parquet + TTL |
| D-5 `normalizer.py` | ✓ Complete | Skips NaN rows with logging |
| D-6 `pipeline.py` | ✓ Complete | Cache → fetch → normalize → return |
| D-7 `tests/data/` | ✓ Complete | All 22 tests pass, fixtures ≥90 days |
| M-1 `features.py` | ~ Partial | All feature columns present; **log_return_1d uses close[t] (C-1)** |
| M-2 `splitter.py` | ✓ Complete | Temporal cutoff enforced |
| M-3 `trainer.py` | ✓ Complete | LogisticRegression + StandardScaler, artifact persisted |
| M-4 `predictor.py` | ✓ Complete | Loads artifact, returns `PredictionResult` |
| M-5 `model_store.py` | ✓ Complete | joblib + metadata JSON with train date range |
| M-6 `evaluator.py` | ✓ Complete | accuracy, AUC-ROC, Brier, per-quarter breakdown |
| M-7 `tests/model/` | ✓ Complete | 23 tests pass; temporal-integrity assertions present |
| INT-1 `tests/test_smoke.py` | ✗ Missing | Not created; required by plan |
| INT-2 Coverage gate | ~ Partial | Coverage meets targets; `pytest --cov-fail-under=80` cannot run via `pytest` shell alias |

---

## 5. Measured Bottlenecks

Profile run: 50 iterations of `build_feature_dataframe` on the 83-row fixture panel, then 1 `train()`, then 100 `predict()` calls. Total wall time: **0.888 s**.

| Rank | Function | Cumulative time | Calls | Notes |
|---|---|---|---|---|
| 1 | `build_feature_dataframe` | 0.658 s (50 calls) | 50 | **13.2 ms/call** — dominant path |
| 2 | `Predictor.predict` | 0.204 s (100 calls) | 100 | **2.0 ms/call** |
| 3 | `Pipeline.predict_proba` (sklearn) | 0.162 s | 100 | 1.6 ms/call — mostly validation overhead |
| 4 | `check_array` (sklearn validation) | 0.144 s | 206 | Called twice per predict |
| 5 | `StandardScaler.transform` | 0.142 s | 101 | 1.4 ms/call |
| 6 | `pd.Series.__init__` | 0.140 s | 1774 | Pandas construction overhead |
| 7 | `pd.DataFrame.__init__` | 0.114 s | 250 | Many small DF constructions |

**Key observations**:

1. `build_feature_dataframe` (13.2 ms/call on 83 rows) will dominate at production scale. The inner loop creates many small `pd.Series` objects per symbol group. At 1000+ rows per symbol, this will scale roughly linearly but with high constant overhead from repeated `pd.Series.__init__` calls (1774 calls for 50 iterations on 83 rows = ~35 per call).

2. `Predictor.predict` (2.0 ms/call) is dominated by sklearn's `_validate_data` / `check_array` which runs input validation on every single call (0.144 s for 100 calls). This is standard sklearn behavior for single-row prediction.

3. Training itself (`train()`) is fast — not in top 15 by cumulative time on this fixture size.

---

## 6. Recommended Optimizations

### O-1 (HIGH priority, correctness): Fix `log_return_1d` lookahead (see C-1)

In `src/model/features.py`, change the output frame to use the already-computed `lagged_log_ret` (which is `log_ret.shift(1)`) instead of `log_ret` for the `log_return_1d` column. This eliminates the close[t] leak. The fix is a one-line change; `lagged_log_ret` is already computed on line 51.

### O-2 (HIGH priority, plan compliance): Add `tests/test_smoke.py` (INT-1)

Create the integration smoke test that wires `pipeline.py → features.py → trainer.py → predictor.py` end-to-end on synthetic fixture data without mocks. This is the final required deliverable per the plan and the only way to confirm cross-module composition is correct.

### O-3 (MEDIUM): Reduce sklearn validation overhead in hot-path prediction

For real-time inference on pre-validated feature rows, wrap `predict_proba` to skip redundant validation:

```python
X = pd.DataFrame([...])
proba = self._model.predict_proba(X)  # 2ms due to check_array each call
```

Alternative: cache a `numpy.ndarray` input shape check at `Predictor.__init__` time and use `self._model[-1].predict_proba(self._model[:-1].transform(X))` with `check_input=False` on the final estimator. This would cut per-call overhead from ~2 ms to ~0.3 ms for batch inference use cases.

### O-4 (MEDIUM): Batch `build_feature_dataframe` for multi-symbol pipelines

Currently each symbol group creates a new `pd.DataFrame` via `model_dump()` for every record. For a universe of 10 symbols × 500 rows, the per-symbol group construction cost adds up. Constructing the full DataFrame once from all records (already done) then grouping is correct — the bottleneck is the per-symbol `pd.Series` construction inside the group loop. Consider pre-allocating numpy arrays per feature before wrapping in a DataFrame at the end.

### O-5 (LOW): Add `assert` or `re.match` guard on symbol in `DiskCache._key_path` (see S-1)

```python
import re
if not re.match(r'^[A-Za-z0-9_.-]+$', symbol):
    raise ValueError(f"Invalid symbol for cache key: {symbol!r}")
```

This is a one-line hardening that eliminates the path-traversal risk if symbols are ever sourced externally.

### O-6 (LOW): Move module-level `os.getenv` calls into `DataPipeline.__init__` (see C-5)

Deferred env resolution makes the class safe for test monkeypatching without constructor injection.
