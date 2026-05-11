# Polymarket Copy-Trading Bot: Optimization Notes

Reference document for improving a copy-trading agent on Polymarket. Current baseline policy: copy top 10 traders by PnL over the last 10 days, refresh list daily. Below are axes of optimization, ordered roughly by expected impact.

---

## 1. Trader Ranking (Who to Copy)

Raw PnL over a short window is mostly noise. Improvements:

### 1a. Shrinkage toward the mean
Raw PnL over small samples overstates true skill. Apply James-Stein or hierarchical Bayesian shrinkage so traders with many resolved bets at moderate PnL rank above traders with few resolved bets at high PnL. Prior: most traders are average or worse.

### 1b. Brier score / log-loss on implied probabilities
Prediction markets give you ground-truth resolution (0 or 1). Score each trader's **implied probability at entry** with a proper scoring rule (Brier or log-loss). This separates edge from luck far better than dollar PnL — it is invariant to bet size and market odds.

- Brier: `mean((p_entry - outcome)^2)` — lower is better
- Log-loss: `-mean(outcome * log(p) + (1-outcome) * log(1-p))` — lower is better

### 1c. Risk-adjusted metrics
Sharpe, Sortino, max-drawdown-adjusted PnL. A trader who yolo'd their bankroll and got lucky should not outrank a consistent grinder.

### 1d. Persistence test (sanity check)
Before trusting any ranking metric, verify: do top-decile traders in window `t` outperform median in window `t+1`? If the signal barely persists, the ranking is mostly noise — lengthen the lookback or use a better metric.

### 1e. Specialization detection
A trader great at US politics might be random on sports or crypto. Cluster markets by category, compute per-category skill, and only copy a trader on categories where they have statistically significant edge.

---

## 2. Adverse Selection on Entry (Game Theory #1)

The biggest hidden cost in naive copy trading. When you mirror a trader's entry, you're buying **after** they did, at a worse price, into a book their own trade may have moved. They get the best fill; you get the worst. Over time this structural drag can eat most of the naive alpha.

Mitigations:
- Only copy traders whose typical size is small relative to book depth.
- Skip copy trades where post-trade mid has moved beyond a threshold (e.g. > X bps) from their fill.
- **Best approach:** treat their trade as a *signal*, not an order. Independently re-evaluate whether the *current* price still offers edge given the leaked information.

---

## 3. Reflexivity and the Copy-Trading Commons (Game Theory #2)

Public-ish "top trader" rankings cause many bots to converge on the same names. Effects:
1. Those traders' edge gets arbitraged away by the copier crowd.
2. Smart traders realize they're being copied and exploit it: build reputation with small good bets, then take a large bad position and exit while copiers pile in (Akerlof-style adverse selection).

Defenses:
- Use **private / harder-to-compute rankings** (Brier-based rather than PnL-based).
- Require a **minimum resolved-bet sample size** before a trader can enter the copy list.
- Detect "pump setup" behavior: sudden size increase on a low-conviction market after a clean track record.

---

## 4. Aggregation Across Traders

Currently each trader's trade is executed independently. Richer view: treat each top trader as a **noisy estimator of the true probability**, and aggregate.

- If 7/10 traders are long YES and 3 are short, that's a much stronger signal than "whichever trader acted last."
- Derive a **consensus implied probability** per market, weighted by each trader's posterior skill (from §1).
- Log-odds averaging is a reasonable default (assumes independence). Something more sophisticated can model inter-trader correlations.
- When top traders disagree strongly, the right action is often to **not trade** or to trade smaller.

---

## 5. Integrating Your Own Model (Bayesian)

Clean framing. Let:
- `p_own` = your own model's prediction for the market
- `p_consensus` = skill-weighted consensus from copy signals (§4)
- `p_market` = current market price

Your posterior is a **precision-weighted combination** of `p_own` and `p_consensus` (weights ∝ inverse variance of each). Trade when posterior disagrees meaningfully with `p_market`.

Nice properties:
- When your model is confident and traders disagree, your model dominates — copy becomes a sanity check.
- When your model is uncertain (market outside your competence), the copy signal dominates — exactly when copying is most valuable.

**Advanced:** use the copy-consensus signal as a *feature* in your own prediction model, trained on historical data to learn when trader consensus predicts outcomes vs. when it's noise.

---

## 6. Bet Sizing — Fractional Kelly

Equal-sized copies are almost certainly suboptimal. Kelly sizing:

```
f = edge / variance
```

Scale down to **quarter- or half-Kelly** because edge estimates are themselves uncertain. With the Bayesian setup above you have a natural edge estimate `(p_posterior - p_market)` and can size accordingly.

Sizing often matters more than trader selection. Most copy-trading blow-ups come from over-concentration on a single bet, not from picking the wrong traders.

**Exposure caps:**
- Per-market cap
- Per-category cap
- Total-outstanding cap
- Correlation cap: "10 independent bets" that all resolve on the same underlying event is really one bet.

---

## 7. Dynamic Trader Selection as a Bandit Problem

"Top 10 refreshed daily" is a hard threshold. Softer formulation: treat each candidate trader as an arm in a **multi-armed bandit**, with a posterior over their skill.

- Use **Thompson sampling** or **UCB** to balance exploiting known-good traders against exploring promising newcomers.
- Graceful decay: traders don't fall off the list instantly on a bad week.
- Proper exploration of traders who haven't yet accumulated enough sample to enter the top 10.

---

## 8. Microstructure Details (Polymarket-Specific)

Often matters more than fancy statistics:

- Gas costs and fees relative to bet size (especially for small copy trades).
- Thin-book slippage, particularly on long-tail markets.
- Resolution ambiguity risk — factor into EV, not just probability.
- Time-to-resolution: capital lockup has an opportunity cost; include in edge calculation.
- Latency: how fast can the bot detect and mirror a trade? Every second matters for entry price.

---

## Suggested Implementation Priority

1. **Better trader ranking** — Brier score + shrinkage + minimum sample size (§1b, §1a)
2. **Adverse-selection filter on entry** — re-check current price against their fill before copying (§2)
3. **Kelly-based bet sizing** with exposure caps (§6)
4. **Bayesian aggregation across traders** → consensus probability (§4)
5. **Integrate own model via precision-weighted posterior** (§5)
6. **Bandit-based trader selection** to replace hard top-10 cutoff (§7)
7. **Specialization / per-category skill** (§1e)
8. **Reflexivity defenses** — monitor for pump-setup patterns (§3)

The game-theory items (§2, §3) are the ones most likely to invalidate the whole premise of naive copy trading, so they should be thought through before over-optimizing the statistics layer.

---

## Open Questions for Further Research

- Empirical persistence of skill on Polymarket: what's the correlation between top-decile performance in window `t` and window `t+1` for various window lengths?
- Distribution of book depth vs. top-trader typical size — is adverse selection a first-order or second-order effect at realistic bet sizes?
- How much of "top trader" PnL on Polymarket is from a few resolved tail events vs. consistent edge?
- Are there observable features (bet size patterns, market category, time-of-day) that distinguish skill from variance in-sample?
