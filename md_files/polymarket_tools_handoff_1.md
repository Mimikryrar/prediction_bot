# Polymarket Tools — Handoff Notes for Copy-Trading & Prediction Bot Split

## Purpose of this doc

Summary of the analysis of [Awesome-Polymarket-Tools](https://github.com/harish-garg/Awesome-Polymarket-Tools), with concrete next actions for splitting our single bot into (1) a copy-trading bot and (2) a prediction bot, and for fixing the known bug from the earlier dashboard audit (1,232 buys / 0 sells).

---

## Framing caveats on the repo

- It's an **awesome-list** (curated links), not code. Only 4 stars, one maintainer.
- Several entries are **placeholder links** — the `github.com/username/...` URLs under "Trading & Execution" and "Copy Trading" are dead. Do not spend time chasing those.
- Treat it as a directory to mine; each interesting entry needs its own evaluation.

---

## Architectural decision: split into two services

### Copy-trading bot
- **Role:** signal ingestion (trader activity) → adverse-selection filter → execution
- **Output:** fills on Polymarket
- **Core loop:** watch selected traders → decide whether to mirror → size → execute

### Prediction bot
- **Role:** features (market metadata, on-chain data, news, cross-venue prices) → model → probability
- **Output:** a single number `p_0` per market with a confidence/variance
- **Core loop:** ingest features → run model → emit probability estimate

### Integration point
The two services reconnect at the Bayesian posterior:

```
p_posterior ∝ w_0 * p_0  +  w_c * p_c
```

where `p_0` is the prediction bot's probability, `p_c` is the copy-trade consensus from the copy bot, and weights are proportional to each signal's precision (inverse variance). That posterior drives Kelly sizing.

Benefits of the split:
- Ship, test, and kill each independently.
- Fall back cleanly if one service is down (`w_0 → 0` when prediction bot is offline).
- Avoids the current monolith's coupling between "who to copy" and "what to trade."

---

## Tool shortlist per bot

### For the COPY-TRADING BOT

| Tool | Why it matters | Action |
|---|---|---|
| `py-clob-client` / `@polymarket/clob-client` (official) | Auth + signing quirks we don't want to reimplement. Community wrappers go stale. | Confirm we're on the official client; migrate if not. |
| `polymarket-apis` (Python, Pydantic) | Unified typed models across CLOB/Gamma/Data/Web3/WS/GraphQL. Kills glue code. | Evaluate as replacement for our current API layer. |
| **Bitquery GraphQL** | On-chain event data for Polymarket contracts. **Directly addresses the 0-sells bug** — query each copied trader's real sell history from chain rather than trusting our poller. | Write a Bitquery query that returns all sells for the current copied trader wallets; reconcile against our bot's recorded activity. Highest priority. |
| `Polymarket/agents` (official framework) | Official agent framework, modular, LLM-integrated. Likely has latency/signing patterns solved. | Clone and read. Don't necessarily adopt wholesale. |
| Whale trackers: PolyTrack, Polywhaler, Poly Whales Tracker | Useful as **data sources**, not as ranking authorities — they all rank by raw PnL, same noisy metric we're moving away from. | Check if any expose an API. If yes, use for trader activity feed. Ignore their rankings. |
| Telegram bots (Polycule, PolyFocus, Polycool) | These are our actual competitors — they do copy-trading as a feature. Part of the reflexivity problem (more bots on the same top traders = faster edge decay). | Sign up briefly for UX/signal recon. Don't integrate. |

### For the PREDICTION BOT

| Tool | Why it matters | Action |
|---|---|---|
| **FinFeedAPI** | Unified OHLCV across Polymarket, Kalshi, Myriad, Manifold. Cross-venue price discrepancies on the same underlying event = cleanest signal in prediction markets. | Integrate early. First feature: cross-venue basis signal. |
| **Gamma API** (official) | Market metadata: categories, resolution rules, liquidity, end dates. Required input for features and for per-category skill modeling. | Wire up as the metadata backbone. |
| **Dune dashboards + Bitquery** | Historical resolved-market data. Needed for backtesting — before trusting any model, check edge over a meaningful span of resolved markets. | Use for backtest dataset. |
| PolyPulse (Chrome extension w/ Perplexity) | Gimmicky as an extension, but the pattern (news → per-market LLM context) is right. | Reference only. Don't install. |

---

## Priority action items (Claude Code / Cowork)

Ordered by impact:

### 1. Fix the 0-sells bug (copy bot)
- Pull wallet addresses of all currently-copied traders.
- Write a Bitquery GraphQL query returning all sell events for those wallets over the last 30 days.
- Reconcile against our bot's internal `sells` table.
- If there's a discrepancy, the bug is in our poller — not in the traders' behavior.

### 2. Service split scaffolding
- Create two services: `copy-bot/` and `prediction-bot/`.
- Define the message contract between them: prediction bot publishes `{market_id, p_0, variance, timestamp}`; copy bot publishes `{market_id, p_c, n_traders, variance, timestamp}`.
- Build a thin `posterior` module that combines the two using the Bayesian formula above and emits Kelly-sized orders.
- Each service owns its own state and can be restarted independently.

### 3. Evaluate `Polymarket/agents` and `polymarket-apis`
- Clone `Polymarket/agents` — read architecture, borrow patterns, don't fork.
- Evaluate whether `polymarket-apis` replaces our current API layer. Criterion: does it reduce our custom glue code by >50% without losing capabilities?

### 4. Prediction bot MVP
- **Do not start with an LLM-in-the-loop model.** Start with a regression baseline so there's something to beat.
- Inputs: Gamma API metadata + FinFeedAPI cross-venue prices.
- Output: `p_0` for markets where we have domain view (start narrow, e.g. one category).
- Evaluate on Brier score over resolved markets.

### 5. Copy-bot ranking upgrade (from earlier conversation)
Already scoped separately, but reminder of the stack in priority order:
1. Brier score on implied probabilities + shrinkage toward the mean.
2. Adverse-selection filter on entry (skip copies where post-trade mid has already moved beyond threshold).
3. Kelly-based bet sizing (fractional, quarter- or half-Kelly).
4. Bayesian aggregation across traders (log-odds averaging).

---

## What NOT to do

- Don't build our own whale tracker. PolyTrack/Polywhaler already exist and we don't need to be in that business.
- Don't treat any public trader ranking as authoritative. The reflexivity problem — smart traders know they're being copied and can exploit it — means visible rankings select for adversarial mimicry.
- Don't adopt community SDKs for execution-critical paths. Use official clients (`py-clob-client`, `@polymarket/clob-client`).
- Don't chase the dead placeholder links in the awesome-list.

---

## Open questions for follow-up

- Which category do we start the prediction bot on? (sports, politics, crypto — our domain view should drive this)
- Do we co-locate the two bots for latency or keep them as separate deploys?
- What's our per-market exposure cap, and do we need a correlation-aware portfolio layer on top of Kelly?

---

*Source analysis: [harish-garg/Awesome-Polymarket-Tools](https://github.com/harish-garg/Awesome-Polymarket-Tools). Prior context: copy-trading optimization conversation (Brier/shrinkage, adverse selection, Kelly, Bayesian aggregation) and dashboard bug audit (0-sells finding).*
