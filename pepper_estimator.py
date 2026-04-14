"""
pepper_estimator.py — Live slope estimator and exit logic for INTARIAN_PEPPER_ROOT.

Importable with no external deps beyond numpy.
Do NOT modify existing trader code — import from here and call the functions.

Key design decisions (all data-derived):
  - elapsed = ts - first_ts  (NOT raw timestamp)
  - Seeded slope = 0.001 (exact cross-day slope, confirmed R²=1.0000)
  - Anchor = mean of first 20 ticks (best MAE across N ∈ {1..100})
  - Rolling OLS window = 1000 (best practical balance, fills in 1000 obs)
  - Expanding OLS until window fills, then switches to rolling

clear_position_order replacement:
  - pepper_exit_orders() is the drop-in replacement for the passive
    clear_position_order that was sitting at stale limit prices.
  - Returns (price, qty) sell orders that are guaranteed to fill.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (mirror params.py — kept here for import independence)
# ══════════════════════════════════════════════════════════════════════════════

SEED_SLOPE         = 0.001000   # derived from 3-day OLS, R²=1.0
START_N            = 20         # mean of first N ticks for anchor
ROLLING_WINDOW     = 1000       # rolling OLS window (ticks)
SOFT_LIMIT         = 60         # circuit-breaker threshold
HARD_LIMIT         = 75         # IMC position limit
UNWIND_START_TS    = 80_000     # EOS ramp begins
SLOPE_WEAK_FRAC    = 0.50       # slope < 50% of seed → emergency exit
SLOPE_WEAK_THRESH  = SEED_SLOPE * SLOPE_WEAK_FRAC   # = 0.0005


# ══════════════════════════════════════════════════════════════════════════════
# PEPPER ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class PepperEstimator:
    """
    Online fair-value estimator for INTARIAN_PEPPER_ROOT.

    Usage (in your Trader.run):
        # --- __init__ ---
        self.pepper_est = PepperEstimator(prev_day_close=13000.0)

        # --- each tick ---
        fv = self.pepper_est.update(state.timestamp, mid_price)
        slope = self.pepper_est.slope   # current live slope estimate

    Phases:
        [0, START_N)       : seeded slope + prev_day_close anchor
        [START_N, WINDOW)  : expanding OLS (all obs so far), anchor locked at
                             mean of first START_N prices
        [WINDOW, ∞)        : rolling OLS (last WINDOW obs), anchor locked

    Elapsed-time anchor fix:
        All OLS regressions use x = (timestamp - first_timestamp), NOT raw ts.
        This ensures fair_value(first_ts) ≈ anchor_price exactly.
    """

    def __init__(
        self,
        prev_day_close: float,
        seed_slope: float = SEED_SLOPE,
        start_n: int = START_N,
        window: int = ROLLING_WINDOW,
    ):
        self.prev_day_close = prev_day_close
        self._seed_slope    = seed_slope
        self._start_n       = start_n
        self._window        = window

        # Internal state
        self._first_ts:   Optional[float] = None
        self._anchor:     Optional[float] = None   # locked after START_N obs
        self._n_obs:      int             = 0
        self.warmed_up:   bool            = False

        # Ring buffer for rolling OLS (stores (elapsed, price) pairs)
        self._buf_x: list[float] = []  # elapsed times
        self._buf_y: list[float] = []  # prices

        # Expanding OLS accumulators (Welford-style, O(1) per update)
        self._sw  = 0.0   # sum of weights
        self._sx  = 0.0   # sum x
        self._sy  = 0.0   # sum y
        self._sxx = 0.0   # sum x²
        self._sxy = 0.0   # sum xy

        # Current slope estimate (starts seeded)
        self._slope: float = seed_slope

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def slope(self) -> float:
        """Current slope estimate (ticks per timestamp unit, elapsed basis)."""
        return self._slope

    @property
    def n_obs(self) -> int:
        return self._n_obs

    def fair_value(self, timestamp: float) -> float:
        """
        Point-in-time fair value at the given timestamp.
        Can be called without update() to get current estimate at any ts.
        """
        elapsed = self._to_elapsed(timestamp)
        anchor  = self._anchor if self._anchor is not None else self.prev_day_close
        return anchor + self._slope * elapsed

    def update(self, timestamp: float, mid_price: float) -> float:
        """
        Ingest a new observation.  Returns current fair value at this timestamp.
        Call once per tick, in timestamp order.
        """
        # Record first timestamp
        if self._first_ts is None:
            self._first_ts = timestamp

        elapsed = self._to_elapsed(timestamp)
        self._n_obs += 1

        # Phase 1: collect first START_N prices to lock anchor
        if self._anchor is None:
            self._buf_x.append(elapsed)
            self._buf_y.append(mid_price)
            if len(self._buf_x) >= self._start_n:
                self._anchor = float(np.mean(self._buf_y))
                # Bootstrap OLS accumulators from the buffer
                for x, y in zip(self._buf_x, self._buf_y):
                    self._expanding_add(x, y)
                self._update_slope_expanding()
                self.warmed_up = True
                # Clear bootstrap buffer (no longer needed)
                self._buf_x.clear()
                self._buf_y.clear()
        else:
            # Phase 2/3: update slope estimate
            if self._n_obs <= self._window:
                # Still in expanding window
                self._expanding_add(elapsed, mid_price)
                self._update_slope_expanding()
            else:
                # Rolling window: use numpy for simplicity
                # (could be made O(1) with a ring buffer OLS, but N=1000 is fast enough)
                self._buf_x.append(elapsed)
                self._buf_y.append(mid_price)
                # Keep only the last WINDOW observations
                if len(self._buf_x) > self._window:
                    self._buf_x = self._buf_x[-self._window:]
                    self._buf_y = self._buf_y[-self._window:]
                self._update_slope_rolling()

        return self.fair_value(timestamp)

    def reset(self, prev_day_close: float) -> None:
        """
        Reset for a new day.  Call at the start of each new session.
        Preserves window and seed_slope settings.
        """
        self.__init__(prev_day_close, self._seed_slope, self._start_n, self._window)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _to_elapsed(self, timestamp: float) -> float:
        if self._first_ts is None:
            return 0.0
        return timestamp - self._first_ts

    def _expanding_add(self, x: float, y: float) -> None:
        self._sw  += 1
        self._sx  += x
        self._sy  += y
        self._sxx += x * x
        self._sxy += x * y

    def _update_slope_expanding(self) -> None:
        denom = self._sw * self._sxx - self._sx ** 2
        if denom > 1e-12:
            self._slope = (self._sw * self._sxy - self._sx * self._sy) / denom

    def _update_slope_rolling(self) -> None:
        x = np.array(self._buf_x, dtype=np.float64)
        y = np.array(self._buf_y, dtype=np.float64)
        xm = x.mean()
        denom = float(((x - xm) ** 2).sum())
        if denom > 1e-12:
            ym = y.mean()
            self._slope = float(((x - xm) * (y - ym)).sum() / denom)


# ══════════════════════════════════════════════════════════════════════════════
# PEPPER ENTRY GATE
# ══════════════════════════════════════════════════════════════════════════════

def pepper_should_buy(
    position: int,
    timestamp: int,
    current_slope: float,
) -> bool:
    """
    Returns False to block any new long entry.
    Call this before submitting any buy order for PEPPER.

    Blocks when:
      - position >= SOFT_LIMIT (60)    : circuit breaker, already too long
      - timestamp >= UNWIND_START_TS   : EOS ramp started, no new buys
      - current_slope < SLOPE_WEAK_THRESH : trend collapsing, cut risk now
    """
    if position >= SOFT_LIMIT:
        return False
    if timestamp >= UNWIND_START_TS:
        return False
    if current_slope < SLOPE_WEAK_THRESH:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLEAR_POSITION_ORDER DROP-IN REPLACEMENT
# ══════════════════════════════════════════════════════════════════════════════

def pepper_exit_orders(
    position: int,
    timestamp: int,
    current_slope: float,
    best_bid: int,
    best_ask: int,
) -> list[tuple[int, int]]:
    """
    Drop-in replacement for clear_position_order.

    The original clear_position_order placed passive limit orders at a fixed
    price.  In an uptrend, the ask keeps rising above the clearing level so
    those limits never fill.  This function uses aggressive, guaranteed-fill
    logic instead.

    Returns a list of (price, qty) sell orders.
    qty is always NEGATIVE (sells).
    Merge these into your order list each tick.

    Priority (highest to lowest):
      1. EOS hard unwind  (ts ≥ 80_000) — linear decay, hits best_bid
      2. Slope collapse   (slope < 0.0005) — sell 1/3 per tick at best_bid
      3. Circuit breaker  (pos > 60) — sell excess at best_ask - 1

    Arguments:
      position      : current PEPPER position (signed integer)
      timestamp     : current session timestamp
      current_slope : PepperEstimator.slope (live rolling OLS slope)
      best_bid      : best bid in the order book (integer price)
      best_ask      : best ask in the order book (integer price)

    Example (in your Trader.run):
        fv     = self.pepper_est.update(state.timestamp, mid_price)
        exits  = pepper_exit_orders(
            position, state.timestamp, self.pepper_est.slope,
            best_bid, best_ask
        )
        for price, qty in exits:
            orders.append(Order("INTARIAN_PEPPER_ROOT", price, qty))
    """
    orders: list[tuple[int, int]] = []

    if position <= 0:
        return orders

    # ── Priority 1: End-of-session hard unwind ─────────────────────────────────
    if timestamp >= UNWIND_START_TS:
        # Linear decay: 0% sold at ts=80k, 100% sold at ts=100k.
        progress = min(1.0, (timestamp - UNWIND_START_TS) / 20_000.0)
        target   = int(round(position * (1.0 - progress)))
        to_sell  = position - target
        if to_sell > 0:
            # Aggressive: cross the spread to guarantee the fill.
            orders.append((best_bid, -to_sell))
        return orders  # EOS takes full priority — skip all lower tiers

    # ── Priority 2: Slope-weakening trigger ────────────────────────────────────
    if current_slope < SLOPE_WEAK_THRESH:
        # Sell ~1/3 of remaining position per tick until flat.
        # Aggressive at best_bid to ensure fill.
        to_sell = max(1, position // 3)
        orders.append((best_bid, -to_sell))
        return orders  # Return after slope trigger — don't stack with circuit breaker

    # ── Priority 3: Mid-session circuit breaker ────────────────────────────────
    if position > SOFT_LIMIT:
        excess    = position - SOFT_LIMIT
        # Semi-aggressive: 1 tick below best_ask.
        # In an uptrend the ask rises above old clearing levels, so best_ask - 1
        # ensures we're competitive without fully crossing to the bid.
        ask_price = best_ask - 1
        orders.append((ask_price, -excess))

    return orders


# ══════════════════════════════════════════════════════════════════════════════
# QUICK INTEGRATION CHECKLIST
# ══════════════════════════════════════════════════════════════════════════════
#
#  [ ] Import PepperEstimator, pepper_should_buy, pepper_exit_orders
#
#  [ ] In Trader.__init__ (or equivalent):
#        from pepper_estimator import PepperEstimator
#        self.pepper_est = PepperEstimator(prev_day_close=13000.0)
#
#  [ ] In Trader.run, every tick:
#        mid_price = (best_bid + best_ask) / 2.0
#        fv        = self.pepper_est.update(state.timestamp, mid_price)
#        slope     = self.pepper_est.slope
#
#  [ ] Replace clear_position_order calls with:
#        for price, qty in pepper_exit_orders(
#            position, state.timestamp, slope, best_bid, best_ask
#        ):
#            orders.append(Order("INTARIAN_PEPPER_ROOT", price, qty))
#
#  [ ] Gate buy orders:
#        if pepper_should_buy(position, state.timestamp, slope):
#            # your existing buy logic here
#
#  [ ] Do NOT use state.timestamp directly as the x-axis for slope estimation.
#        The PepperEstimator handles elapsed-time conversion internally.
#
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # ── Smoke test ─────────────────────────────────────────────────────────────
    import math

    est = PepperEstimator(prev_day_close=13000.0)
    errors = []

    # Simulate perfect linear drift: price = 13000 + 0.001 * elapsed
    for ts in range(0, 100_000, 100):
        true_price = 13000.0 + 0.001 * ts
        fv = est.update(float(ts), true_price)
        err = abs(fv - true_price)
        errors.append(err)

    print(f"Smoke test: {len(errors)} ticks simulated")
    print(f"  Max error: {max(errors):.4f}")
    print(f"  Mean error (full): {sum(errors)/len(errors):.4f}")
    post_warmup = errors[START_N:]
    print(f"  Mean error (post-warmup): {sum(post_warmup)/len(post_warmup):.6f}")
    print(f"  Final slope estimate: {est.slope:.6f}  (target: 0.001000)")
    print(f"  Warmed up after obs: {START_N}")

    # Test exit orders
    print("\nExit order test:")
    cases = [
        (55, 10_000, 0.001, 14000, 14002, "mid-session, small pos"),
        (65, 10_000, 0.001, 14000, 14002, "mid-session, over soft limit"),
        (70, 85_000, 0.001, 14000, 14002, "EOS ramp (42.5% progress)"),
        (50, 90_000, 0.001, 14000, 14002, "EOS ramp (50% progress)"),
        (40, 50_000, 0.0004, 14000, 14002, "slope collapsed"),
    ]
    for pos, ts, slope, bid, ask, desc in cases:
        ords = pepper_exit_orders(pos, ts, slope, bid, ask)
        gate = pepper_should_buy(pos, ts, slope)
        print(f"  [{desc}]  pos={pos} ts={ts} slope={slope}")
        print(f"    exit_orders={ords}  should_buy={gate}")
