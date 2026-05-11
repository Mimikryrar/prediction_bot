# Polymarket Copy-Trading Bot: Recommended Optimization Order

This is the recommended execution order for optimizing the current bot based on the latest code review and the existing handoff docs.

## Core Rule

Do not optimize trader selection, ranking, Kelly sizing, Bayesian aggregation, service split, or prediction-bot architecture until the current bot is operationally correct in live conditions.

Right now the highest-risk gaps are not strategy quality. They are control-plane safety, state durability, live identity verification, and incomplete observability.

## Phase 0: Control Plane and State Safety

Goal: make it impossible for the dashboard or a process restart to corrupt or reset a live bot.

Tasks:
- Make `bot.lock` / `bot_mode.json` the source of truth for dashboard state, not only in-memory `BotRunner` state.
- Block `/api/state/reset` unless no active bot is detected from the sentinel files.
- Make dashboard status survive dashboard restarts by reading persistent bot state on every request.
- Add auth or at minimum a shared secret if the dashboard will ever bind beyond localhost.
- Serialize `seen_trades.json` flushes the same way `lots.json` flushes are serialized.
- Add startup checks for malformed `trades.jsonl` / state files and fail loudly on corruption.

Exit criteria:
- A dashboard restart cannot make the UI think the bot is stopped when it is still running.
- State reset is impossible while a real bot process holds the lock.
- `seen_trades.json` writes are atomic and ordered.

## Phase 1: Live-Mode Safety Guards

Goal: make live trading fail closed when the wallet or environment is wrong.

Tasks:
- Make `expected_wallet_address` mandatory when `DRY_RUN=0`.
- Refuse live startup if Bullpen returns no wallet address or the wrong one.
- Validate numeric env vars such as `POLL_INTERVAL_MS`, `RESOLVE_INTERVAL_MS`, `SWEEP_INTERVAL_MS`, and dashboard port values.
- Tighten the handoff docs so they do not normalize storing the signer in plain `.env`; prefer a real secrets manager or OS keychain-backed path.
- Keep the kill switch tied to persistent bot mode, not dashboard-local assumptions.

Exit criteria:
- The bot cannot place live trades against the wrong Bullpen profile.
- Invalid env values fail at startup instead of becoming `NaN` runtime behavior.

## Phase 2: Observability and Error Accounting

Goal: stop treating infrastructure errors as invisible background noise.

Tasks:
- Journal poll timeouts, wallet fetch failures, and unexpected executor exceptions as first-class events.
- Include `buy_skipped_wallet_error` in dashboard skip and failure accounting.
- Expand dashboard failure-rate logic so infrastructure failures show up in alerts.
- Add clear counters for:
  - poll timeouts
  - wallet errors
  - market fetch failures
  - journal parse failures
- Decide which errors should be fatal in live mode instead of only logged.

Exit criteria:
- A “clean” dashboard actually means the system is healthy.
- Dry mode no longer hides important operational failures.

## Phase 3: Execution Layer Migration

Goal: replace Bullpen CLI execution only after the current operational surface is trustworthy.

Tasks:
- Swap the execution layer to the official client (`py-clob-client` or official TS client) behind the same executor interface.
- Keep ranking, sizing, filters, and state format constant in this phase.
- Run old and new executors in parallel dry mode for at least 48h.
- Diff intended orders per trader signal and resolve every mismatch before cutover.

Exit criteria:
- New executor matches old intent for the same signals.
- Execution migration is isolated from strategy changes.

## Phase 4: Signal Correctness and Sell Reconciliation

Goal: prove the activity feed is accurate before trusting any strategy metrics.

Tasks:
- Use Bitquery to reconcile actual copied-trader sell history.
- Compare Bitquery results against the bot's detected activity and journaled behavior.
- Investigate any buy/sell asymmetry before moving to ranking work.
- Build a repeatable reconciliation script, not a one-off manual check.

Exit criteria:
- You can explain the observed buy/sell ratios with confidence.
- Trader activity ingestion is validated against authoritative chain data.

## Phase 5: Trade Safety Logic Hardening

Goal: remove fail-open behavior from live trading decisions.

Tasks:
- Change market-safety checks to fail closed when market metadata is unavailable.
- Revisit timeout behavior so stale poll work does not overlap future cycles.
- Review reserve logic explicitly as a policy choice:
  - current implementation uses cost-basis exposure
  - decide whether production should use cash balance, cost basis, or mark-to-market equity
- Add tests for:
  - duplicate signal handling
  - concurrent buy signals
  - partial sells
  - budget exhaustion
  - restart recovery

Exit criteria:
- Missing market data cannot silently bypass safety filters.
- Budget logic is intentionally defined and tested.

## Phase 6: Strategy Improvements

Only start this after Phases 0 through 5 are green.

Recommended order:
1. Adverse-selection filter on entry
2. Better trader ranking: Brier/log-loss plus shrinkage and minimum sample size
3. Kelly-based sizing with hard exposure caps
4. Trader aggregation / consensus probability
5. Specialization by market category
6. Reflexivity defenses and pump-pattern detection

Rationale:
- These changes can improve edge, but they do not matter if the bot is still operationally unsafe or measuring the wrong thing.

## Phase 7: Service Split and Prediction Bot

Do this last.

Tasks:
- Split copy bot and prediction bot only after the copy bot is stable as a standalone live-trading system.
- Add a prediction bot only after you have a validated baseline and a backtest dataset.
- Do not increase deployment complexity before execution, ingestion, and safety logic are proven.

## What Not To Do Yet

- Do not start with the prediction bot.
- Do not start with the service split.
- Do not start with ranking upgrades before sell reconciliation.
- Do not tune trader selection while control-plane and observability gaps remain.
- Do not combine execution migration with model changes in the same step.

## Immediate Next 5 Tasks

1. Make dashboard state and reset logic read `bot.lock` / `bot_mode.json`.
2. Serialize `seen_trades.json` flushes and add corruption detection on startup.
3. Make live mode require `expected_wallet_address`.
4. Add journaled events and dashboard alerts for wallet errors, poll timeouts, and parse failures.
5. Start the executor diff harness for the Bullpen CLI to official-client migration.

## Decision Summary

The correct order is:

1. Operational correctness
2. Live safety gates
3. Observability
4. Execution migration
5. Signal reconciliation
6. Trade-safety hardening
7. Strategy optimization
8. Multi-service architecture

Anything else is optimizing on top of an unreliable base.
