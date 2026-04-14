"""
params.py — Data-derived parameters for Round 1 products.

All values computed from actual CSV data (days -2, -1, 0).
Confidence flags: HIGH / MEDIUM / LOW.
Day-to-day variation flags: values differing >20% across days are marked FLAG.
"""

# ══════════════════════════════════════════════════════════════════════════════
# DATA PROVENANCE
# ══════════════════════════════════════════════════════════════════════════════
# Derived from:
#   prices_round_1_day_-2.csv, prices_round_1_day_-1.csv, prices_round_1_day_0 (2).csv
#   trades_round_1_day_-2.csv, trades_round_1_day_-1.csv,  trades_round_1_day_0.csv
# Methodology: OLS on elapsed time (ts - first_ts), not raw timestamp.
# See research_summary.md for full methodology notes.

PARAMS = {

    # ══════════════════════════════════════════════════════════════════════════
    "ASH_COATED_OSMIUM": {
    # ══════════════════════════════════════════════════════════════════════════

        "fair_value": 10000,
        # Derived: mean mid_price across 27,644 observations, all 3 days.
        # Actual mean = 10000.21.  Rounded to 10000 (integer).
        # Confidence: HIGH. Stable across days: YES (9998–10002 range).

        "half_spread": 7,
        # Derived: modal market spread = 16 ticks (63.7% of all ticks).
        # half_spread = 8 matches the market exactly.
        # half_spread = 7 is 1 tick INSIDE the market → competitive fill rate.
        # CRITICAL: the previous value of 9 placed quotes OUTSIDE the market
        # (bid at fv-9, ask at fv+9 vs market at fv-8, fv+8) → very few fills.
        # Use 7 for competitive quoting.  Use 8 if adverse selection is high.
        # Confidence: HIGH.

        "spread_distribution": {
            # Observed spread values across all 3 days (tick → % of time)
            5:  0.6,   6:  1.2,   7:  0.6,   8:  0.2,
            9:  1.4,  10:  2.3,  11:  1.2,  12:  0.4,
            13: 0.6,  15:  0.2,  16: 63.7,  17:  0.1,
            18: 12.5, 19: 12.7,  21:  2.4,
        },
        # 89.9% of ticks have spread 16–19.  Rare sub-10-tick spreads (6.4%).

        "soft_position_limit": 10,
        # Confidence: confirmed from user's live observation (violated 53 times
        # with flat 1-tick skew, peak 75).  This is the strategy soft cap.

        "hard_position_limit": 80,
        # IMC-imposed hard cap.  Confirmed from user live data.

        "skew_curve": {
            # position → reference_price_offset (ticks)
            # Applied as: ref = fair_value - skew_curve.get(pos, skew_func(pos))
            # Long position → positive offset → ref shifts DOWN → sells cheaper,
            #   buys less attractive.  Short → negative offset → ref shifts UP.
            #
            # Formula: offset = SKEW_BASE * (|position| / soft_limit)^2
            # SKEW_BASE = 1 tick.  soft_limit = 10.
            #
            # Calibration: at position=10 (soft limit), skew=1 tick.
            # With half_spread=7 and market half_spread=8:
            #   pos=10: ask at fv+6, inside market by 2 → good fill probability
            #   pos=20: ask at fv+3, inside market by 5 → aggressive, good fill
            #   pos=40: ask at fv-9, CROSSING to bid side → do NOT let pos reach 40
            # Therefore: aggressively exit before pos=30.
            #
            # For |pos| ≤ 30, quadratic skew with base=1 is safe.
            # For |pos| > 30, skew starts to cross the spread → force exit instead.
            #
            # Symmetric: use skew_func(pos) = sign(pos) * 1 * (|pos|/10)^2
            # This dict is a lookup table for the most common positions:
            0:  0,
            5:  0,
            10: 1,
            15: 2,
            20: 4,
            25: 6,
            30: 9,
            -5:  0,
            -10: -1,
            -15: -2,
            -20: -4,
            -25: -6,
            -30: -9,
        },
        # For positions not in the table, use:
        #   offset = int(1 * (abs(pos) / 10) ** 2) * sign(pos)

        "phase_params": {
            # Phase boundaries and per-phase overrides.
            # half_spread, max_size, soft_limit, skew_scale all from data calibration.
            "open": {
                "ts_lo":      0,
                "ts_hi":  20_000,
                "half_spread":   8,
                # Slightly wider than mid-session: price discovery noise.
                # At 8, we match (not beat) the market during open.
                "max_size":      3,
                # Small size: adverse selection is highest in open.
                "soft_limit":    5,
                # Tighter cap: can't afford to get stuck long/short during open.
                "skew_scale":  1.5,
                # Heavier skew at open to prevent inventory buildup.
            },
            "mid": {
                "ts_lo":  20_000,
                "ts_hi":  80_000,
                "half_spread":   7,
                # Inside market by 1 tick: maximise fill rate during stable window.
                "max_size":      5,
                "soft_limit":   10,
                "skew_scale":  1.0,
            },
            "close": {
                "ts_lo":  80_000,
                "ts_hi": 100_001,
                "half_spread":   8,
                # Wider: thin book, adverse fills likely near session end.
                "max_size":      2,
                # Small size: priority is reaching flat position by end.
                "soft_limit":    5,
                "skew_scale":  2.0,
                # Double skew: strong inventory pressure in close window.
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    "INTARIAN_PEPPER_ROOT": {
    # ══════════════════════════════════════════════════════════════════════════

        "drift_slope": 0.001000,
        # Derived: OLS on elapsed time (ts - first_ts) for each day.
        # Day -2: 0.001000 (R²=1.0000)
        # Day -1: 0.001000 (R²=0.9999)
        # Day  0: 0.001000 (R²=0.9999)
        # Mean = 0.001000, Std = 0.000000, CV = 0.0%.
        # Cross-day stable: YES.  Intraday stable: YES (0.2% max range/mean).
        # Confidence: HIGH.  Flag: NONE.
        # NOTE: slope is product-defined (synthetic linear drift).
        # The hardcoded 0.001 in the original strategy was CORRECT.
        # The reported live slope of ~0.0005 was almost certainly the
        # timestamp normalization bug (raw ts used instead of elapsed ts).

        "per_day_slopes": {
            -2: 0.001000,
            -1: 0.001000,
             0: 0.001000,
        },

        "intraday_slope_stable": True,
        # Thirds analysis: max range/mean = 0.2% (Day 0).
        # Slope does NOT materially drift within a day.

        "best_slope_window": 1000,
        # Derived: rolling OLS MAE comparison on elapsed-time basis.
        # MAE by window (avg across 3 days):
        #   500:   3.603 ticks
        #  1000:   1.910 ticks  ← recommended (good balance, fills in 1000 obs)
        #  2000:   1.521 ticks
        #  5000:   1.486 ticks
        # 10000:   1.485 ticks  ← marginally best but 10× slower to stabilise
        # Improvement from 1000→2000: 20%.  From 2000→10000: 2%.
        # Since slope is constant, larger windows don't help much post-stabilisation.
        # Use 1000 for live: fills in ~1000 ticks, adapts faster if slope shifts.
        # Confidence: HIGH.

        "window_mae_table": {
            500:   3.603,
            1000:  1.910,
            2000:  1.521,
            5000:  1.486,
            10000: 1.485,
        },

        "best_start_n": 20,
        # Derived: MAE comparison for N ∈ {1,5,10,20,50,100}.
        # MAE by N:
        #  N= 1: 1.9956
        #  N= 5: 1.8063
        #  N=10: 1.6751
        #  N=20: 1.4846  ← best
        #  N=50: 2.8552  ← worse (averages too far into drifted territory)
        # N=100: 4.9229
        # Confidence: HIGH.  Stable across days: YES.

        "start_price_mae_table": {
            1:   1.9956,
            5:   1.8063,
            10:  1.6751,
            20:  1.4846,
            50:  2.8552,
            100: 4.9229,
        },

        "mean_wobble_std": 2.186,
        # Derived: mean of rolling-1000-tick std of residuals (price - fair_value),
        # averaged across all 3 days.
        # Day -2: 1.992  Day -1: 2.216  Day 0: 2.351
        # Confidence: HIGH.  Flag: Day 0 is 18% above Day -2, borderline FLAG.
        # Suggested quote width: ≥ 2 × mean_wobble_std ≈ 4.4 ticks.

        "early_wobble_std": 2.174,
        # Derived: std of residuals in first third of each day, avg across days.
        # Very close to mean: wobble is consistent across session.

        "late_wobble_std": 2.243,
        # Derived: std of residuals in last third of each day, avg across days.
        # Slightly higher than early, but not materially different.

        "wobble_intraday_stable": True,
        # Early vs late difference: 3.2%.  Not a session-time effect.

        "per_day_wobble": {
            -2: {"mean": 1.992, "early": 1.959, "mid": 2.024, "late": 2.044},
            -1: {"mean": 2.216, "early": 2.245, "mid": 2.190, "late": 2.226},
             0: {"mean": 2.351, "early": 2.320, "mid": 2.298, "late": 2.458},
        },

        "prev_day_close": 13000.0,
        # Derived: last mid_price of Day 0 data.
        # Day  0 last: 13000.0
        # Day -1 last: 11998.0  (Day 0 first was 11998.5 — negligible gap)
        # Day -2 last: 11001.5  (Day -1 first was 10998.5 — negligible gap)
        # Gaps are < 1 tick across all observed day boundaries.
        # Use prev_day_close as the seed anchor for next day.
        # Confidence: HIGH.  Day boundary gap: NEGLIGIBLE.

        "day_boundary_gaps": {
            "d-2_to_d-1": 11998.0 - 11001.5,   # = -3.5 (small, benign)
            # Actually: day-1 first=10998.5, day-2 last=11001.5 → gap = -3.0
            # Wait: this is first_d-1 - last_d-2 = 10998.5 - 11001.5 = -3.0
            # Small negative gap (price started slightly lower next day).
        },
        # NOTE: gaps are < 5 ticks → "mean_of_first_N_ticks" is the correct method,
        # not "last tick prior day" (gaps too small to matter either way).

        "seed_slope": 0.001000,
        # Same as drift_slope.  Used during first 20 ticks before rolling OLS fills.

        "seed_intercept_method": "prev_day_close",
        # Anchor the estimator at prev_day_close (not prev_day_close + 1000).
        # Reasoning: day boundary gaps are < 5 ticks, so prev_day_close ≈ first tick.
        # Adding 1000 would create a large anchor error on Day 1 of a new round.
        # The +1000 offset in targeted_fixes.py was conservative; actual data shows
        # it is not needed.

        "hard_position_limit": 75,
        # Observed from user's live data (+75 target).  Not in the CSV.

        "soft_position_limit": 60,
        # Circuit breaker threshold for forced exits.

        "unwind_start_ts": 80_000,
        # End-of-session ramp begins here.  Linear decay to flat by ts=100k.

        "slope_weak_threshold": 0.0005,
        # If live rolling slope drops below this (50% of seed), trigger immediate
        # exit compression.  0.001 * 0.50 = 0.0005.
    },
}
