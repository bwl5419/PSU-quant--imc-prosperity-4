# IMC Prosperity Round 1 — Research Summary

**Data used:** prices and trades, days −2, −1, 0.  
**Products:** INTARIAN\_PEPPER\_ROOT (trend-following), ASH\_COATED\_OSMIUM (market-making).  
**Date produced:** 2026-04-14.

---

## ASH_COATED_OSMIUM

**What we found.** Osmium is a stable mean-reverting product tightly anchored near 10 000. The fair value of 10 000 is confirmed across all three days (actual mean 10 000.21, range 9 981–10 017). The raw market spread is 16 ticks 63.7% of the time, with occasional widening to 18–19 (another 25%) and rare narrowing to sub-10 (6%).

**What changed.** The single most important fix is the half_spread parameter. The original strategy used half_spread=9, which places quotes at fv±9 against a market that sits at fv±8. This means our quotes were outside the market's best prices 63.7% of the time and fills only occurred during the occasional wider-spread regimes (18–19 ticks). Correcting to half_spread=7 puts us 1 tick inside the market continuously, making our quotes competitive and dramatically increasing fill rate. The second fix is the nonlinear inventory skew. A flat 1-tick skew on a 16-tick-spread product has no meaningful effect on inventory direction. The quadratic formula (`1 * (pos/10)^2`) produces 1 tick of skew at the soft limit (10) and 16 ticks at position 40, which is sufficient to strongly discourage further accumulation at elevated inventory levels without destroying fill probability until position exceeds ~30.

**Confidence level: HIGH** for fair value and spread calibration. **MEDIUM** for phase boundaries (0–20k open, 20k–80k mid, 80k–100k close) — these are reasonable priors but have not been independently validated against a phase-stratified backtest of the live data.

---

## INTARIAN_PEPPER_ROOT

**What we found.** Pepper has a deterministic linear drift of exactly 0.001 ticks per timestamp unit. This slope is perfectly stable: R² = 1.0000 on Day −2, 0.9999 on Days −1 and 0. Intraday thirds analysis shows 0.0–0.2% slope variation within a day. This is almost certainly a synthetic product with a designed constant drift. The slope does not change meaningfully across training days or within a day.

**What changed.** Three fixes were made. First, the timestamp normalization bug: fair value must be computed as `anchor + slope * (ts - first_ts)`, not `anchor + slope * ts`. Using raw timestamps as the x-axis means the model is correct at timestamp 0 but accumulates a large offset as the session progresses if the anchor is the first-tick price rather than the OLS intercept. Since timestamps restart from 0 each day in the training data, this bug had no effect on training-day analysis — but it would cause an error in any environment where timestamps do not start at 0. The PepperEstimator handles this internally. Second, the start price: mean of the first 20 ticks (MAE 1.485) significantly outperforms the first tick alone (MAE 1.996). Averaging 50+ ticks is harmful because the price has drifted too far. Third, the exit mechanism: the original `clear_position_order` placed passive limit orders at a fixed price. In an uptrend, the market ask continuously rises above any fixed clearing level so these orders never fill. The replacement `pepper_exit_orders()` uses aggressive (crossing) orders with three priority tiers: end-of-session linear unwind from ts=80 000, slope-collapse emergency exit, and mid-session circuit breaker at position > 60.

**On the reported live slope of ~0.0005.** Training data shows slope = 0.001 with zero variance. If the live observed slope appeared to be 0.0005, the most likely cause is the elapsed-time normalization bug: if timestamps in the live round do not start at 0 (for example, if they are global session counters continuing from a prior round), a raw-timestamp OLS would underestimate the slope because the intercept absorbs part of the price level. The PepperEstimator class eliminates this by construction.

**Confidence level: HIGH** for slope (0.001, three-day verification with R²≈1). **HIGH** for start-N = 20 (clear MAE minimum). **MEDIUM** for residual wobble (2.2 ticks average, increasing slightly across days: 1.99 → 2.22 → 2.35 — a mild upward trend that may indicate a regime change).

---

## Parameter Table

| Parameter | Value | Derived From | Stable Across Days? |
|-----------|-------|-------------|---------------------|
| OSMIUM fair\_value | 10 000 | Mean mid-price, all 3 days, 27 644 obs | YES |
| OSMIUM half\_spread | 7 | Modal market spread = 16; inside by 1 tick | YES |
| OSMIUM modal\_spread | 16 ticks (63.7%) | Spread distribution, all days | YES |
| OSMIUM soft\_limit | 10 | User live data | — |
| OSMIUM hard\_limit | 80 | User live data | — |
| OSMIUM skew base | 1 tick at pos=10 | Quadratic formula | — |
| OSMIUM open half\_spread | 8 | Conservative (match market) | — |
| OSMIUM close skew\_scale | 2× | Inventory pressure at close | — |
| PEPPER slope | 0.001000 | OLS on elapsed time, all 3 days | YES (0.0% CV) |
| PEPPER intraday slope drift | 0.0–0.2% | Thirds analysis | YES (stable) |
| PEPPER best\_start\_n | 20 | MAE sweep N ∈ {1,5,10,20,50,100} | YES |
| PEPPER best\_window | 1000 | MAE sweep windows ∈ {500..10000} | YES |
| PEPPER mean\_wobble\_std | 2.186 ticks | Rolling-1000 std of residuals | BORDERLINE FLAG |
| PEPPER early\_wobble\_std | 2.174 ticks | First third of session | YES |
| PEPPER late\_wobble\_std | 2.243 ticks | Last third of session | YES |
| PEPPER prev\_day\_close | 13 000.0 | Day 0 final mid-price | — |
| PEPPER day boundary gap | < 5 ticks | First-of-day vs last-of-prior | YES (negligible) |
| PEPPER soft\_limit (circuit breaker) | 60 | User specification | — |
| PEPPER unwind start ts | 80 000 | User specification | — |
| PEPPER slope\_weak\_threshold | 0.0005 | 50% of seed slope | — |

**Flagged parameters (>20% cross-day variation):**
- `mean_wobble_std`: Day −2 = 1.992, Day 0 = 2.351 (18% above Day −2). Borderline. Treat quote-width decisions conservatively and use the Day 0 value (2.35) as the baseline.

---

## Handoff Checklist

- [x] **Slope confirmed from data.** 0.001 ticks/ts, R²=1.000, zero variance across 3 days.
- [x] **Elapsed-time anchor bug fixed.** `PepperEstimator` uses `ts - first_ts` internally.
- [x] **Start price anchor fixed.** Mean of first 20 ticks, not first tick alone.
- [x] **clear\_position\_order replaced.** `pepper_exit_orders()` in `pepper_estimator.py` is the drop-in. Uses aggressive fills, not passive stale limit prices.
- [x] **EOS unwind implemented.** Linear ramp from ts=80 000 to flat at ts=100 000.
- [x] **Circuit breaker at pos=60.** `pepper_should_buy()` blocks new entries.
- [x] **Slope-collapse trigger at 0.0005.** Immediate 1/3-per-tick exit.
- [x] **Osmium half\_spread corrected.** 7 ticks (inside market), not 9 (outside market).
- [x] **Osmium nonlinear skew implemented.** Quadratic formula, base=1 tick at pos=10.
- [x] **Osmium phase switching ready.** open/mid/close parameters in `params.py`.
- [ ] **Live slope validation.** If the live round shows slope ≠ 0.001, investigate whether timestamps start from 0. If timestamps do not start from 0, the PepperEstimator handles it — but verify the slope estimate after the rolling window fills (~1000 ticks).
- [ ] **Osmium half\_spread adverseselection check.** At half\_spread=7, we are always inside the market. If fills are consistently unprofitable (adverse selection), consider raising to 8.
- [ ] **Wobble trend.** Mean wobble increased from 1.99 (Day −2) to 2.35 (Day 0). If this continues in the live round, quote widths may need to expand to 5–6 ticks to avoid over-quoting.
- [ ] **No trader.py in repo.** The original strategy code was not committed. The repo contains only analysis and fix modules. Ensure `pepper_estimator.py` and `params.py` are integrated into the actual submission file before the round.

---

## What Likely Caused the Live 899-Tick Miss

Based on the data analysis, the most probable cause was **the exit mechanism failure, not slope estimation**. The slope of 0.001 is correct and stable. The anchor (if using first-tick or mean-of-first-N) was close to correct since day-boundary gaps are negligible (<5 ticks). The residual wobble (2.2 ticks) is well within the range of normal.

The live session ended +54 long with 16 buys and only 5 sells. This is a liquidity/execution problem: the passive `clear_position_order` placed limit orders at a fixed price that the market ask rose above as the day progressed. The accumulated long position that was never unwound represents the primary source of PnL loss. The fix is `pepper_exit_orders()`.

Secondary risk: the 65-tick dead zone at open (before rolling OLS stabilises) caused a brief period of poor fair-value estimates. With the seeded estimator, this is reduced to the START_N=20 tick anchor period, after which slope estimation begins immediately.
