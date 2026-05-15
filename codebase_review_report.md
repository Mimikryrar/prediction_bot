# Codebase Review Report

## Project
`/Users/mimiimac/Developer/polymarked_prediction_modelbased_bot`

## Review scope
Reviewed primarily:
- `prediction_bot/`
- `tests/`
- `pyproject.toml`
- `.env.example`
- `.gitignore`

Note: `openswarm/` appears to be a separate framework/vendor subtree and was not fully audited as part of this report.

## Verification performed
- Ran test suite: **61 passed**
- Ran coverage: **96% total**

## Executive summary
This is a strong small-to-medium ML/research codebase with clear separation between data, model, and shared contracts. The tests are a major strength. The main weaknesses are package layout, split semantics, schema strictness, and a few gaps between plan/docs and implementation.

## Strengths
- Clean modular structure: `prediction_bot/data`, `prediction_bot/model`, `prediction_bot/shared`
- High test coverage with meaningful regression tests
- Offline/mocked testing for external APIs
- Good use of fixtures
- Straightforward model persistence and prediction flow
- Cache key validation is present and reasonable
- Feature engineering code is mostly clean and vectorized

## Key findings

### 1. Medium: package layout uses `src` as the top-level package
**Files:** `pyproject.toml`, imports across `prediction_bot/*`

Current imports use patterns like:
- `from prediction_bot.data.pipeline import DataPipeline`

The packaging config also exposes `src` itself as the package. This works locally, but it is awkward for installation and downstream reuse.

**Risks:**
- confusing import behavior
- awkward library consumption
- tooling/distribution friction
- tighter coupling to repository layout

**Recommendation:**
Rename the real package to something explicit like:
- `prediction_bot`
- `polymarked_prediction_bot`

Then import via that package name instead of `src`.

---

### 2. Medium: `temporal_split()` is row-fraction-based rather than date-boundary-based
**File:** `prediction_bot/model/splitter.py`

The current implementation sorts by date and then slices by row counts. That is acceptable for a basic baseline, but it is weaker than splitting on sorted unique dates.

**Why it matters:**
For panel data with multiple symbols and uneven coverage, row-based slicing is not the clearest temporal guarantee.

**Recommendation:**
Split by unique ordered dates, then assign all rows for each date to train/val/test.

---

### 3. Medium: evaluator does not fully match planned capabilities
**File:** `prediction_bot/model/evaluator.py`

The implementation provides:
- accuracy
- AUC-ROC
- Brier score
- per-quarter breakdown

But planning/docs mention calibration support as well. That does not currently exist in the evaluator output.

**Recommendation:**
Either:
- add calibration curve / calibration metrics, or
- update docs so implementation and spec match.

---

### 4. Medium: Pydantic schemas are too permissive
**File:** `prediction_bot/shared/schemas.py`

Current models do not strongly constrain values such as:
- `asset_type`
- `source`
- `p_up`
- `confidence`
- `horizon_days`

**Why it matters:**
Prediction pipelines benefit from stronger invariants at module boundaries.

**Recommendation:**
Use enums and numeric constraints, e.g.:
- `asset_type`: `Literal["stock", "crypto"]`
- `source`: `Literal["yfinance", "coingecko"]`
- `p_up`, `confidence`: bounded to `[0, 1]`
- `horizon_days`: positive integer

---

### 5. Low: no root README
There is no top-level `README.md` for the app itself.

**Impact:**
- harder onboarding
- unclear usage flow
- missing setup/run documentation

**Recommendation:**
Add a concise root README covering:
- project purpose
- setup/install
- running tests
- training flow
- prediction flow

---

### 6. Low: minor cleanup opportunities
Examples:
- small implementation/doc mismatches
- minor import cleanup opportunities
- some naming/layout polish still available

Not urgent, but worth doing during refactoring.

---

### 7. Low: `joblib.load()` assumes trusted artifacts
**File:** `prediction_bot/model/model_store.py`

This is acceptable for local trusted model artifacts, but unsafe for untrusted files.

**Recommendation:**
Document that model artifacts must come from trusted local sources only.

## Testing assessment
Testing is one of the strongest parts of the codebase.

Highlights:
- regression-focused tests
- smoke test present
- external APIs mocked
- coverage is high without appearing artificially inflated

Coverage summary observed:
- overall: **96%**
- most modules: high 90s to 100%
- weaker but still acceptable: `splitter.py`, `trainer.py`

## Architecture assessment
Overall architecture is good for an MVP/research system:
- `data/` handles fetching/cache/normalization
- `model/` handles features/training/prediction/evaluation
- `shared/` defines the contract

This separation is a real strength and makes future refactoring feasible.

## Recommended priority order
1. Refactor package name away from `src`
2. Change temporal splitting to unique-date boundaries
3. Tighten schema constraints
4. Align evaluator behavior with docs/plan
5. Add a root README

## Final assessment
This is a solid MVP / research-grade codebase.

### Quick score
- Code quality: **8/10**
- Test quality: **9/10**
- Architecture: **8/10**
- Production readiness: **6.5/10**

## Output path
Saved report to:
`/Users/mimiimac/Developer/polymarked_prediction_modelbased_bot/codebase_review_report.md`
