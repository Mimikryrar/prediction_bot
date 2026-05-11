# Polymarket Copy-Trading Bot — Infrastructure & Custody Handoff

> Handoff document for Claude Code / Cowork. Summarizes custody resolution, architectural decisions, and the next concrete work items for the Polymarket copy-trading bot project.

---

## 1. Project Context

Existing system:
- Polymarket copy-trading bot, currently in post-dry-run state (48h dry mode completed, ~1,232 simulated buys, 0 sells recorded — the 0-sells is a known data bug, see §5).
- Execution + wallet interface currently go through the **Bullpen CLI** (`bullpen polymarket …`).
- Funds live in a Bullpen-created embedded wallet on Polygon ($111.49 USDC.e as of this session).

Goal of the re-architecture:
- Decouple execution from the Bullpen wrapper, call Polymarket CLOB directly.
- Split the bot into two services: **copy-trading bot** (signal → filter → execute) and **prediction bot** (features → model → probability).
- Improve ranking/sizing (Brier-based, shrinkage, per-category skill, adverse-selection handling).

Reference repos/packages to evaluate/use:
- https://github.com/Polymarket/agents — official Polymarket agent framework; use as reference for signing flow, WS, rate-limit patterns.
- https://pypi.org/project/polymarket-apis/ — unified typed (Pydantic) Python wrapper across CLOB / Gamma / Data / Web3 / WS / GraphQL.
- https://github.com/Polymarket/py-clob-client — official Python CLOB client.
- https://github.com/Polymarket/safe-wallet-integration — official reference for Safe/proxy-wallet integration patterns.

---

## 2. Custody Resolution (done in this session)

### Key finding
**Bullpen is non-custodial** (per their docs: "Bullpen creates a non-custodial Turnkey wallet for you … you can export your private key at any time. Bullpen is fully non-custodial. Only you have access to your funds."). The signer is fully portable — no migration of funds needed to move off Bullpen's CLI.

### Wallet layout identified in the Bullpen UI (`app.bullpen.fi/wallet/manage`)

| Role | Wallet | Chain | Balance | Purpose |
|---|---|---|---|---|
| **Primary signer (EOA)** | `Bullpen Wallet 0x1b…0666` | EVM / Polygon | $111.49 | Signs Polymarket CLOB orders; owner of the Safe proxy |
| Primary Solana | `Bullpen Wallet GWcZ…GJCY` | Solana | 0.00 | Not relevant to Polymarket |
| Routing → Hyperliquid | sweep wallet | EVM | 0.00 | Internal bridge; not a signer |
| Routing → Solana | sweep wallet | SOL | 0.00 | Internal bridge; not a signer |
| Routing → Polymarket | sweep wallet | EVM | 0.00 | Internal bridge; not a signer |

Export path in UI: on the primary EVM wallet row, the 🔑 icon → **Export Private Key**.

### Signer vs. funder — do not confuse these
- **Signer (EOA)** = `0x1b…0666`. This is the private key. Goes into the bot as `PRIVATE_KEY` env var.
- **Funder (Safe proxy)** = a **different** address, deterministically derived from the signer. This is the Gnosis Safe that actually holds USDC.e and the outcome-token positions. Goes into py-clob-client as the `funder` parameter.
- Orders submitted without the correct `funder` will reject.
- The Safe proxy itself cannot be imported into an external wallet (per Polymarket/Bullpen docs) — only the EOA signer can. Operating directly on the Safe requires going through the signer.

### Verification checklist — RUN BEFORE ANY ARCHITECTURE WORK
All three must pass to confirm full sovereignty over the Polymarket account:

1. **EOA derivation check**: export the private key from Bullpen, import into a throwaway MetaMask profile, confirm the derived address is `0x1b…0666`.
2. **Polymarket login check**: on polymarket.com, log in via WalletConnect using that MetaMask → confirm the $111.49 balance and existing positions are visible, and confirm the funder/proxy address shown at `polymarket.com/settings` matches expectations.
3. **py-clob-client derivation check**:
   ```python
   from py_clob_client.client import ClobClient
   import os
   client = ClobClient(
       host="https://clob.polymarket.com",
       chain=137,
       key=os.getenv("PRIVATE_KEY"),
   )
   creds = client.create_or_derive_api_creds()
   # print funder address and confirm it matches polymarket.com/settings
   ```
   The L2 creds returned (`apiKey`, `secret`, `passphrase`) are separate from the signer key and are used for HMAC auth on authenticated REST endpoints.

If any check fails, STOP and resolve before building new execution paths.

### Key hygiene
- Store `PRIVATE_KEY` in a real secrets manager or an encrypted `.env` that is git-ignored.
- Do **not** paste the key into any chat interface or shared doc (this one included — downstream Claude instances should read the key from env, not be told its value).
- Consider rotating: create a fresh EOA, move USDC.e to the new proxy, use the new EOA as the bot signer, and retire the Bullpen-exported key. This is optional but recommended before going live.
- L2 API creds (apiKey / secret / passphrase) are separate secrets and should also be env-isolated.

---

## 3. Architectural Principles

### Custody / execution separation
- **Signer = sovereign.** Self-custody the EOA; do not treat any frontend as load-bearing for custody.
- **Execution = direct.** Call Polymarket CLOB directly (py-clob-client or polymarket-apis). Don't keep Bullpen CLI on the bot's critical path.
- **Bullpen = optional data layer.** Smart-money feed, WalletScope analytics, CLI discovery commands — consume as inputs if they add value; never as a dependency for order submission.

### Service split
| Service | Input | Output |
|---|---|---|
| Copy-trading bot | Trader activity (on-chain + CLOB) | Filled orders |
| Prediction bot | Market metadata + on-chain + news + cross-venue prices | Single probability $p_0$ per market with a confidence |

They reconnect at the Bayesian posterior:

$$p_{\text{posterior}} \propto w_0 \, p_0 + w_c \, p_c$$

where $p_c$ is the copy-trade consensus and $p_0$ is the prediction bot's output. The posterior drives Kelly sizing. Keep the services independent so each can ship, fail, and be killed without dragging the other down. When the prediction bot is weak or offline, `w_0 → 0` and the copy bot still runs cleanly.

---

## 4. Migration Plan (ordered)

> Rule of thumb: change one variable at a time. Do not combine execution swap with model changes — if performance shifts you won't know which one caused it.

### Phase 0 — Verify custody
Run the three-step verification checklist in §2. Blocker for everything else.

### Phase 1 — Swap execution layer (keep everything else constant)
- Replace all Bullpen CLI calls in the copy bot with `py-clob-client` (recommended for stability) or `polymarket-apis` (recommended if typed Pydantic models across CLOB/Gamma/Data make the codebase cleaner).
- Keep ranking, sizing, state management, config files (`lots.json`, `seen_trades.json`, `resolutions.json`, `trades.jsonl`) untouched in this PR.
- Run the new executor in **dry mode in parallel** with the old Bullpen-based executor for at least 48h.
- Diff their intended orders per trader signal. Any disagreement = bug in one of them; resolve before moving on.
- Do **not** change the dry→live gate logic in this phase. `DRY_RUN=0` still means the same thing.

### Phase 2 — Fix the 0-sells data bug (parallelizable with Phase 1)
- Wire up a **Bitquery GraphQL** query to pull on-chain sell history for each copied trader's proxy wallet.
- Reconcile against the bot's `seen_trades.json` / `trades.jsonl`.
- The 1,232 buys / 0 sells number from dry mode is almost certainly a polling/detection bug, not ground truth. Bitquery gives you the authoritative on-chain record.
- This fix is independent of which execution API is used — it's a correctness issue in the signal-ingestion path.

### Phase 3 — Ranking and filter upgrades
Only start once Phase 1 is stable and Phase 2 has produced clean trader activity data.
- Shrinkage-based PnL ranking (James-Stein or hierarchical Bayesian) to correct for small-sample noise.
- **Brier score / log-loss** on implied entry probabilities as the primary skill metric (PnL is too noisy; Brier is a proper scoring rule on a ground-truth probability).
- Risk-adjusted metrics (Sharpe, Sortino, MDD-adjusted).
- Persistence test: do top-decile traders in window $t$ beat median in window $t+1$? If not, the ranking is noise and lookback / metric need revisiting.
- Per-category skill specialization (cluster markets by category; only copy a trader on categories where they have statistically significant edge).
- Adverse-selection / front-running filter: copying naively buys *after* the trader has moved the book; expected slippage must be baked into EV before sizing.

### Phase 4 — Prediction bot (minimal baseline)
- Start with **Gamma API** (market metadata) + **FinFeedAPI** (cross-venue OHLCV across Polymarket/Kalshi/Myriad/Manifold).
- **Regression baseline first**, not LLM-in-the-loop. Goal is to have something beatable so later iterations can demonstrate actual edge.
- Output: $p_0$ + confidence per market.
- Scope: start with a small set of markets where you have domain view; don't try to cover the full Polymarket universe day one.

### Phase 5 — Operational
- Request Polymarket **whitelist your order-posting server** (free perk for direct integrators; avoids general rate limits and improves reliability during high-volume periods).
- Set up monitoring on: CLOB auth token freshness, signer/funder balance, order reject rate, latency per order, WebSocket reconnects.
- Atomic writes for all state files (write to temp + rename, or use a small SQLite file). Current JSON files are likely not crash-safe.

---

## 5. Known Issues / Open Questions (carry into next session)

- **0-sells bug**: 1,232 buys / 0 sells in 48h dry mode. Highly unlikely to be correct. Root cause unknown; likely in trader-activity polling logic. Bitquery reconciliation (Phase 2) is the canonical fix.
- **Silent error swallowing**: 0 failures across 1,232 dry-mode buys is itself suspicious. Review retry logic and exception handling paths.
- **Race conditions**: concurrent trader signals may hit exposure tracking and budget checks simultaneously. Verify the executor serializes or otherwise handles concurrent buys safely.
- **Dry→live transition gates**: on `DRY_RUN=0`, the bot switches from `dryRunBalanceUsd` in `config/sizing.json` to real wallet queries. Audit for any code paths that still assume dry mode after the flag flips.
- **Portfolio cap**: 75% of wallet balance is currently the spendable cap — re-validate this is still the desired limit post-migration.
- **Key rotation**: decision pending on whether to keep the Bullpen-exported EOA as the production bot signer or rotate to a fresh EOA before going live.

---

## 6. Environment Variables (target state)

```bash
# Signer (Polymarket EOA — exported from Bullpen)
PRIVATE_KEY=0x...                         # never commit, never log

# Polymarket L2 API creds (derived via create_or_derive_api_creds)
POLY_API_KEY=...
POLY_API_SECRET=...
POLY_API_PASSPHRASE=...

# Polymarket funder (Safe proxy address — derived from signer)
POLY_FUNDER=0x...

# Polygon RPC
POLYGON_RPC_URL=https://polygon-rpc.com

# Bot config
DRY_RUN=1                                 # 1=simulated, 0=live
COPY_BUDGET_FRACTION=0.75

# Optional: Bitquery (for sell reconciliation)
BITQUERY_API_KEY=...

# Optional: if retaining Bullpen as data layer
BULLPEN_API_KEY=...
```

---

## 7. First Concrete Tasks for Downstream Claude

In order:

1. **Verify custody (§2 checklist)** — three checks, ~15 minutes. Blocker.
2. **Sketch the py-clob-client executor module** to replace Bullpen CLI calls. Keep the same public interface the rest of the bot expects (so swap is a drop-in).
3. **Build the parallel-diff harness** — run old executor + new executor side-by-side in dry mode, log both `intended_order` payloads, diff per signal.
4. **Draft the Bitquery sell-reconciliation query** for the list of currently copied traders.
5. **Audit crash-safety of state file writes** (atomic write pattern or SQLite migration).

Do **not** start on Phase 3 (ranking upgrades) or Phase 4 (prediction bot) until Phase 1 + 2 are green.

---

## 8. References

- Polymarket CLOB auth: https://docs.polymarket.com/api-reference/authentication
- Polymarket agents framework: https://github.com/Polymarket/agents
- `polymarket-apis` (Pydantic-typed): https://pypi.org/project/polymarket-apis/
- `py-clob-client`: https://github.com/Polymarket/py-clob-client
- Polymarket Safe wallet integration: https://github.com/Polymarket/safe-wallet-integration
- Bullpen wallet/collateral docs: https://docs.bullpen.fi/prediction-markets/wallets-and-collateral
- Bullpen CLI reference: https://cli.bullpen.fi/reference/commands/polymarket/
- Bitquery (on-chain indexing for Polymarket): https://bitquery.io
- FinFeedAPI (cross-venue prediction-market OHLCV): https://finfeedapi.com
