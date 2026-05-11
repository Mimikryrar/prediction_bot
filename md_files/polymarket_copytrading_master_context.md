# Polymarket Copy-Trading Bot — Master Context, Optimization Roadmap, and Handoff Guide

Last updated: 2026-04-24

## Purpose of This File

This is the master entrypoint for the copy-trading bot optimization process.

Use this file as the canonical overview for:
- what the bot currently is
- what has already been reviewed or fixed
- what order the remaining work should happen in
- which supporting document to read for each topic
- what future sessions should and should not work on

If a new human or AI picks up the project, this is the first file they should read.

## Current Bot Status

The bot is still best classified as a strong prototype, not a live-ready production trading system.

Current strengths:
- The repo has a working dry-run copy-trading loop.
- Buys and sells are now serialized through the executor admission lock.
- `lots.json` and `resolutions.json` use atomic temp-write-plus-rename patterns.
- The bot now uses a single-instance lock plus `bot_mode.json` sentinel state.
- Sell handling now reduces lot cost basis and updates cash flow correctly.
- Dashboard wallet summary now uses persisted bot mode instead of only dashboard env.
- Live wallet errors are no longer misclassified as budget exhaustion.

Current blockers:
- The dashboard is still not fully authoritative about whether a bot is running.
- `seen_trades.json` flushes are not yet serialized like `lots.json`.
- Live Bullpen wallet identity verification is still optional instead of mandatory.
- Operational failures are still undercounted in the dashboard and journal analytics.
- Some safety checks still fail open.
- Execution still depends on Bullpen CLI rather than an official direct client.

Bottom line:
- The next work should focus on operational correctness and live-safety gates.
- Strategy optimization is not the immediate bottleneck.

## What This Bot Does Today

Current functional behavior:
- Watches configured target traders for new Polymarket trades.
- Mirrors qualifying buys with fixed dollar sizing.
- Mirrors sells proportionally.
- Tracks local position state in JSON files.
- Tracks journal activity in `trades.jsonl`.
- Supports dry mode and live mode.
- Uses Bullpen CLI as the current execution and data path.

Current state files:
- `data/state/lots.json`
- `data/state/seen_trades.json`
- `data/state/resolutions.json`
- `data/trades.jsonl`
- `data/state/bot.lock`
- `data/state/bot_mode.json`

## Document Map

These supporting files remain useful, but they serve different purposes.

### 1. [polymarket_bot_handoff_2.md](/Users/mimiimac/Developer/polymarked_copy_trading_bot/polymarket_bot_handoff_2.md)

Role:
- custody and execution migration context
- signer vs funder explanation
- migration-phasing logic
- sell-reconciliation priority

Read this when:
- working on custody
- migrating away from Bullpen CLI
- validating live execution assumptions

Important caution:
- treat this file as infrastructure guidance, not as permission to store sensitive keys casually

### 2. [polymarket_copytrading_optimization_order.md](/Users/mimiimac/Developer/polymarked_copy_trading_bot/polymarket_copytrading_optimization_order.md)

Role:
- clean execution order for the next optimization phases

Read this when:
- deciding what should happen next
- checking whether a task is premature

### 3. [polymarket_copytrading_optimizations.md](/Users/mimiimac/Developer/polymarked_copy_trading_bot/polymarket_copytrading_optimizations.md)

Role:
- later-stage strategy ideas
- ranking, Bayesian aggregation, Kelly sizing, bandits, specialization, reflexivity

Read this when:
- Phases 0 through 5 are already complete
- doing research on trader selection and edge extraction

### 4. [polymarket_tools_handoff_1.md](/Users/mimiimac/Developer/polymarked_copy_trading_bot/polymarket_tools_handoff_1.md)

Role:
- external tools and ecosystem research
- candidate future architecture for copy-bot / prediction-bot split

Read this when:
- evaluating official clients
- planning Bitquery work
- researching future architecture after the copy bot is stable

### 5. [polymarket_copytrading_master_context.md](/Users/mimiimac/Developer/polymarked_copy_trading_bot/polymarket_copytrading_master_context.md)

Role:
- top-level source of truth for current status, phase order, and handoff instructions

Rule:
- if this file and another planning doc disagree, follow this file unless a newer explicit decision supersedes it

## Canonical Order of Work

The order below is the current recommended sequence for the whole optimization process.

## Phase 0 — Control Plane and State Safety

Goal:
- make it impossible for the dashboard or a restart to corrupt or reset a real bot

In scope:
- make `bot.lock` and `bot_mode.json` the dashboard source of truth
- prevent state reset while a real bot is active
- make dashboard state survive dashboard restarts
- serialize `seen_trades.json` flushes
- add corruption detection for journal and state files
- add dashboard auth or local-only guarantees

Why this phase comes first:
- a strategy improvement is worthless if the dashboard can misreport bot state or if resets can wipe state under a live process

Exit criteria:
- dashboard status is derived from persistent bot state
- reset cannot run under an active bot
- `seen_trades.json` writes are atomic and ordered
- journal/state corruption becomes visible immediately

## Phase 1 — Live-Mode Safety Guards

Goal:
- make live trading fail closed if the wallet or environment is wrong

In scope:
- require `expected_wallet_address` when `DRY_RUN=0`
- refuse live startup on Bullpen wallet mismatch
- validate numeric env vars at startup
- tighten key-handling guidance
- keep kill-switch mode tied to persistent bot mode

Why this phase comes second:
- operational correctness is not enough if the wrong Bullpen profile can still place trades

Exit criteria:
- live bot cannot start against the wrong wallet
- invalid runtime env values fail startup
- live-mode assumptions are explicit and enforced

## Phase 2 — Observability and Error Accounting

Goal:
- make hidden failures visible

In scope:
- journal poll timeouts
- journal wallet errors
- journal unexpected executor failures
- count `buy_skipped_wallet_error` in dashboard metrics
- make dashboard alerts reflect infrastructure health, not just trade execution failures
- decide which errors should be fatal in live mode

Why this phase comes before execution migration:
- if the bot cannot measure failures correctly, it cannot safely compare execution paths

Exit criteria:
- a healthy dashboard actually means the system is healthy
- dry mode no longer hides meaningful operational failures

## Phase 3 — Execution Layer Migration

Goal:
- replace Bullpen CLI execution only after the bot is operationally trustworthy

In scope:
- migrate to official execution client
- keep strategy, filters, and state model unchanged during migration
- build a side-by-side diff harness
- compare intended orders for the same signals

Why this phase is isolated:
- execution migration and model changes must not happen in the same step

Exit criteria:
- new executor matches old intent in dry parallel testing
- mismatches are explained before cutover

## Phase 4 — Signal Correctness and Sell Reconciliation

Goal:
- prove trader activity detection is correct

In scope:
- Bitquery-based sell reconciliation
- authoritative comparison between chain activity and bot-detected signals
- investigation of buy/sell asymmetry
- repeatable reconciliation tooling

Why this phase is still pre-strategy:
- ranking work is invalid if the ingestion path is wrong

Exit criteria:
- the team can explain the sell behavior with confidence
- copy-signal ingestion has a trusted reconciliation path

## Phase 5 — Trade Safety Logic Hardening

Goal:
- remove live fail-open behavior and formalize portfolio policy

In scope:
- make market safety checks fail closed when metadata is unavailable
- revisit stale poll overlap and timeout handling
- explicitly define portfolio-cap policy
- test duplicate handling, concurrency, partial sells, restart recovery, and budget checks

Important policy decision:
- the current budget logic uses wallet balance plus cost-basis exposure
- decide explicitly whether production should use:
  - cash balance only
  - cost-basis exposure
  - mark-to-market equity

Exit criteria:
- safety filters do not silently bypass themselves
- budget logic is intentional and tested

## Phase 6 — Strategy Optimization

Only start this after Phases 0 through 5 are complete.

Recommended order:
1. Adverse-selection filter on entry
2. Better trader ranking using Brier/log-loss plus shrinkage
3. Kelly-based sizing with hard exposure caps
4. Consensus aggregation across traders
5. Category specialization
6. Reflexivity and pump-pattern defenses

Reason:
- strategy work should optimize a reliable system, not compensate for broken infrastructure

## Phase 7 — Prediction Bot and Service Split

This is the final phase, not the starting phase.

In scope:
- split copy bot and prediction bot only after the copy bot is stable live
- build a prediction baseline only after execution and ingestion are proven
- increase architecture complexity only when the current monolith is no longer the main risk

## Current Immediate Priorities

These are the next concrete tasks the project should address.

1. Make dashboard state and reset logic use `bot.lock` and `bot_mode.json`.
2. Serialize `seen_trades.json` flushes and add startup corruption checks.
3. Make live mode require `expected_wallet_address`.
4. Add first-class journal events and dashboard reporting for wallet errors, poll timeouts, and parse failures.
5. Build the execution diff harness for Bullpen CLI versus the official client.

## What Is Explicitly Blocked For Now

Do not start these yet:
- prediction bot MVP
- service split scaffolding
- ranking upgrades
- Kelly sizing work
- Bayesian consensus logic
- bandit trader selection
- broad category specialization work

These are blocked because:
- the current bottleneck is still operational correctness, not model sophistication

## Current Known Risks

High-priority unresolved risks:
- dashboard process state can still diverge from real bot state
- `seen_trades.json` is still weaker than `lots.json` in write safety
- Bullpen live identity guard is still optional
- dashboard failure accounting is incomplete
- some live safety checks still fail open

Lower-priority but still relevant:
- journal append format can still lose observability on malformed lines
- env numeric parsing should be hardened
- reserve logic still needs an explicit production policy decision

## Operational Rules for Future Sessions

Use these as hard constraints.

### Change-management rules

- Do not combine execution migration with strategy changes.
- Do not combine ingestion fixes with ranking changes if they affect measured performance.
- Make one class of change at a time so performance deltas stay interpretable.

### Live-safety rules

- Live trading must fail closed, not fail open.
- Wallet identity must be verified before live execution.
- Do not rely on dashboard-local memory for kill/reset decisions.

### Data-integrity rules

- State writes should be atomic.
- If a state file can be corrupted, the bot should detect it loudly.
- Reconciliation should prefer authoritative chain data when ingestion is in doubt.

### Security rules

- Never put secrets in repo files.
- Never paste private keys into docs or chat.
- Prefer secrets manager or OS keychain-backed storage over plain `.env`.
- Treat Bullpen CLI profile selection as part of the live risk surface.

### Tooling rules

- Prefer official clients for execution-critical paths.
- Use Bitquery for sell reconciliation rather than inferring sells only from the poller.
- Treat public whale trackers as data sources, not truth for ranking.

## Recommended Session Workflow

At the start of a new session:
1. Read this file.
2. Identify the current active phase.
3. Read only the supporting docs relevant to that phase.
4. Confirm what is explicitly blocked.
5. Make changes that preserve the phase boundaries.

At the end of a session:
1. Update this file if phase status changed.
2. Update the relevant supporting doc if it contains phase-specific details that changed.
3. Record what was completed, what remains blocked, and what the next session should do.
4. Do not leave ambiguous whether a phase is ready to advance.

## How to Decide Which File to Update

Update this master file when:
- phase order changes
- priorities change
- new blockers are discovered
- a phase becomes complete
- the interpretation of the project changes

Update `polymarket_bot_handoff_2.md` when:
- custody, signer/funder, or execution migration assumptions change

Update `polymarket_copytrading_optimization_order.md` when:
- the recommended task sequence changes but the overall project framing does not

Update `polymarket_copytrading_optimizations.md` when:
- new strategy ideas or research items are added

Update `polymarket_tools_handoff_1.md` when:
- external tooling choices or architecture-research conclusions change

## Definition of “Ready for Live Copy Bot”

Before treating the copy bot as genuinely live-ready, all of the following should be true:
- persistent bot state is authoritative
- reset/kill controls are safe
- live wallet identity verification is mandatory
- infrastructure failures are surfaced clearly
- execution migration is validated by diff harness
- trader activity ingestion is reconciled against chain data
- fail-open market checks are removed
- concurrency and restart behavior are tested

Only after that should strategy optimization become the primary focus.

## Final Decision Summary

The correct macro order for the project is:

1. Control plane and state safety
2. Live-mode safety guards
3. Observability and error accounting
4. Execution migration
5. Signal reconciliation
6. Trade-safety hardening
7. Strategy optimization
8. Prediction-bot / multi-service architecture

Anything that starts lower on the list before the top of the list is complete is probably premature.
