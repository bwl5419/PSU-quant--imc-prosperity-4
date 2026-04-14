"""
IMC Prosperity — Targeted Fixes
Products: INTARIAN_PEPPER_ROOT (trend) + ASH_COATED_OSMIUM (market-making)

Sections
--------
A  Pepper exit + circuit breaker + slope-weakening trigger
B  Osmium nonlinear inventory skew
C  Osmium session phase switching
D  Pepper seeded opening (elapsed-time anchor, no dead zone)
E  Backtester — validates each fix independently on mid_price series
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS  (edit here, propagate everywhere)
# ══════════════════════════════════════════════════════════════════════════════

PEPPER_POS_LIMIT        = 75
PEPPER_SOFT_LIMIT       = 60    # stop new buys above this
PEPPER_UNWIND_START_TS  = 80_000
PEPPER_SEED_SLOPE       = 0.001   # confirmed across 3 training days
PEPPER_START_N          = 20      # ticks averaged for anchor price
PEPPER_SLOPE_WEAK_FRAC  = 0.50   # slope < 50% of seed → compress now

OSMIUM_SOFT_LIMIT  = 10
OSMIUM_HARD_LIMIT  = 80
OSMIUM_BASE_SKEW   = 1   # ticks at exactly the soft limit


# ══════════════════════════════════════════════════════════════════════════════
# A.  PEPPER EXIT + CIRCUIT BREAKER + SLOPE-WEAKENING TRIGGER
# ══════════════════════════════════════════════════════════════════════════════

def pepper_should_buy(
    position: int,
    timestamp: int,
    current_slope: float,
) -> bool:
    """
    Gate: returns False to block any new long entry.
    Call this before submitting any buy order.
    """
    if position >= PEPPER_SOFT_LIMIT:          # circuit breaker
        return False
    if timestamp >= PEPPER_UNWIND_START_TS:    # end-of-session ramp started
        return False
    if current_slope < PEPPER_SEED_SLOPE * PEPPER_SLOPE_WEAK_FRAC:
        return False                            # slope collapsed — no new longs
    return True


def pepper_exit_orders(
    position: int,
    timestamp: int,
    current_slope: float,
    best_bid: int,
    best_ask: int,
) -> list[tuple[int, int]]:
    """
    Returns a list of (price, qty) SELL orders for forced inventory decay.
    qty is always negative (sells).  Call this every tick; merge with your
    existing order list, deduplicate by price, and submit.

    Priority (highest → lowest):
        1. End-of-session hard unwind  (ts >= 80 000)
        2. Slope-weakening trigger     (slope < 50% of seed)
        3. Mid-session circuit breaker (position > soft limit)
    """
    orders: list[tuple[int, int]] = []
    if position <= 0:
        return orders

    # ── 1. End-of-session hard unwind ramp ────────────────────────────────────
    if timestamp >= PEPPER_UNWIND_START_TS:
        progress = min(1.0, (timestamp - PEPPER_UNWIND_START_TS) / 20_000)
        # Linear decay: at ts=80k sell 0%, at ts=100k sell 100%
        target   = int(round(position * (1.0 - progress)))
        to_sell  = position - target
        if to_sell > 0:
            # Aggressive: cross the spread to guarantee fills
            orders.append((best_bid, -to_sell))
        return orders   # EOS takes full priority

    # ── 2. Slope-weakening trigger ─────────────────────────────────────────────
    if current_slope < PEPPER_SEED_SLOPE * PEPPER_SLOPE_WEAK_FRAC:
        # Sell ≈ 1/3 of remaining position per tick until flat
        to_sell = max(1, position // 3)
        orders.append((best_bid, -to_sell))    # aggressive
        return orders

    # ── 3. Mid-session circuit breaker ────────────────────────────────────────
    if position > PEPPER_SOFT_LIMIT:
        excess  = position - PEPPER_SOFT_LIMIT
        # Passive-then-aggressive: try ask-1 first (still crosses if spread ≥ 2)
        ask_price = best_ask - 1
        orders.append((ask_price, -excess))

    return orders


# ══════════════════════════════════════════════════════════════════════════════
# B.  OSMIUM NONLINEAR INVENTORY SKEW
# ══════════════════════════════════════════════════════════════════════════════

def osmium_skew_ticks(position: int) -> int:
    """
    Quadratic skew in ticks.
        pos =  10 →  +1 tick   (at soft limit)
        pos =  40 → +16 ticks
        pos =  70 → +49 ticks
    Positive = long position → skew quotes down to incentivise selling.
    """
    ratio = abs(position) / OSMIUM_SOFT_LIMIT      # 1.0 at soft limit
    magnitude = int(OSMIUM_BASE_SKEW * ratio ** 2)
    return magnitude if position >= 0 else -magnitude


def osmium_quote_prices(
    fair_value: float,
    position: int,
    half_spread: int,
) -> tuple[int, int]:
    """
    Returns (bid_price, ask_price).

    Skew shifts the reference price:
        long  → reference moves down → we sell cheaper, buy higher threshold
        short → reference moves up   → we buy cheaper, sell higher threshold

    The spread stays symmetric around the skewed reference so total width
    is unchanged (we earn the same spread; we just reposition it).
    """
    sk   = osmium_skew_ticks(position)
    ref  = fair_value - sk              # shift reference opposite to position
    bid  = round(ref - half_spread)
    ask  = round(ref + half_spread)
    return bid, ask


# ══════════════════════════════════════════════════════════════════════════════
# C.  OSMIUM SESSION PHASE SWITCHING
# ══════════════════════════════════════════════════════════════════════════════

# Parameter table — tweak values here; logic below reads from it automatically.
#
#  half_spread  : ticks either side of fair value
#  max_size     : maximum order size per side
#  soft_limit   : position threshold before skew kicks in hard
#  skew_scale   : multiplier on the quadratic skew (1.0 = normal)
#
OSMIUM_PHASE_PARAMS: dict[str, dict] = {
    "open":  {
        "ts_lo": 0,      "ts_hi": 20_000,
        "half_spread": 11,          # wider — price discovery still noisy
        "max_size":     3,          # small size until we understand flow
        "soft_limit":   5,          # tighter cap during chaotic open
        "skew_scale":   1.5,        # heavier skew if we do accumulate
    },
    "mid":   {
        "ts_lo": 20_000, "ts_hi": 80_000,
        "half_spread": 9,           # tightest spread — most stable window
        "max_size":     5,
        "soft_limit":  10,
        "skew_scale":   1.0,
    },
    "close": {
        "ts_lo": 80_000, "ts_hi": 100_001,
        "half_spread": 11,          # widen again — thin book, adverse fills
        "max_size":     2,          # small size, priority is flat by EOS
        "soft_limit":   5,
        "skew_scale":   2.0,        # maximum skew pressure to unwind
    },
}


def get_osmium_phase(timestamp: int) -> dict:
    """Returns the active parameter dict for the current timestamp."""
    for name, params in OSMIUM_PHASE_PARAMS.items():
        if params["ts_lo"] <= timestamp < params["ts_hi"]:
            return {"name": name, **params}
    return {"name": "close", **OSMIUM_PHASE_PARAMS["close"]}


def osmium_quote_prices_phased(
    fair_value: float,
    position: int,
    timestamp: int,
) -> tuple[int, int]:
    """
    Full pipeline: phase → skew → quote prices.
    Drop-in replacement for osmium_quote_prices when phase switching is on.
    """
    phase = get_osmium_phase(timestamp)
    sk    = int(osmium_skew_ticks(position) * phase["skew_scale"])
    ref   = fair_value - sk
    bid   = round(ref - phase["half_spread"])
    ask   = round(ref + phase["half_spread"])
    return bid, ask


# ══════════════════════════════════════════════════════════════════════════════
# D.  PEPPER SEEDED OPENING  (elapsed-time anchor, no dead zone)
# ══════════════════════════════════════════════════════════════════════════════
#
# Critical fix: all fair-value calculations use elapsed = ts - first_ts,
# NOT the raw timestamp.  Using raw ts with start_px as the anchor gives
#   fair_value(ts_0) = start_px + slope * ts_0   (wrong — huge offset)
# Using elapsed time gives
#   fair_value(ts_0) = start_px + slope * 0 = start_px  (correct)
#
# The analysis script's predicted = start_px + drift * ts is only correct
# if slope was estimated from a regression that also used raw ts (the OLS
# intercept absorbs the offset).  When you swap in a different anchor
# (start_px ≠ OLS intercept) the two don't cancel and the model is wrong
# from tick 1.


class PepperEstimator:
    """
    Online slope + anchor estimator for PEPPER.

    Before PEPPER_START_N ticks  : seeded slope (0.001) + seeded anchor
    After  PEPPER_START_N ticks  : expanding OLS slope on elapsed time,
                                   anchored at mean of first-N prices
    Slope update: every tick via online Welford-style OLS accumulators
    (O(1) per tick, no stored buffer after warmup).
    """

    def __init__(self, prev_day_close: float):
        self.prev_day_close = prev_day_close

        # Seeded values — used until warmup is complete
        self._slope      = PEPPER_SEED_SLOPE
        self._intercept  = prev_day_close + 1_000   # seed intercept

        # Warm-up buffer
        self._buf: list[tuple[float, float]] = []   # (elapsed, price)
        self._first_ts:  Optional[float] = None
        self._start_px:  Optional[float] = None
        self._n_obs      = 0
        self.warmed_up   = False

        # Online OLS accumulators (Welford, weighted uniformly)
        self._sw   = 0.0   # sum of weights (= n for uniform)
        self._sx   = 0.0   # sum of elapsed
        self._sy   = 0.0   # sum of price
        self._sxx  = 0.0
        self._sxy  = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def slope(self) -> float:
        return self._slope

    def fair_value(self, timestamp: float) -> float:
        """Current fair value estimate at the given timestamp."""
        elapsed = self._elapsed(timestamp)
        if not self.warmed_up:
            # Seeded: anchor at prev_day_close, project forward
            return self._intercept + self._slope * elapsed
        # Warmed up: anchor at start_px (mean of first N), project with live slope
        return self._start_px + self._slope * elapsed

    def update(self, timestamp: float, mid_price: float) -> float:
        """
        Feed a new observation.  Returns the current fair value estimate.
        Call this every tick, use the returned value for your quotes.
        """
        if self._first_ts is None:
            self._first_ts = timestamp

        elapsed = self._elapsed(timestamp)
        self._n_obs += 1

        # Collect first-N prices for anchor
        if len(self._buf) < PEPPER_START_N:
            self._buf.append((elapsed, mid_price))
            if len(self._buf) == PEPPER_START_N:
                self._start_px = float(np.mean([p for _, p in self._buf]))
                # Bootstrap OLS accumulators from the buffer
                for x, y in self._buf:
                    self._add_obs(x, y)
                self.warmed_up = True
                self._update_slope_from_accumulators()
        else:
            # Online update
            self._add_obs(elapsed, mid_price)
            self._update_slope_from_accumulators()

        return self.fair_value(timestamp)

    # ── internals ─────────────────────────────────────────────────────────────

    def _elapsed(self, timestamp: float) -> float:
        if self._first_ts is None:
            return 0.0
        return timestamp - self._first_ts

    def _add_obs(self, x: float, y: float):
        self._sw  += 1
        self._sx  += x
        self._sy  += y
        self._sxx += x * x
        self._sxy += x * y

    def _update_slope_from_accumulators(self):
        denom = self._sw * self._sxx - self._sx ** 2
        if denom > 0:
            self._slope = (self._sw * self._sxy - self._sx * self._sy) / denom


# ══════════════════════════════════════════════════════════════════════════════
# E.  BACKTESTER — validates each fix independently
# ══════════════════════════════════════════════════════════════════════════════
#
# Input : DataFrame with columns [timestamp, mid_price]
#         (one row per tick, one product at a time)
# Output: BacktestResult dataclass with PnL, position trace, fill count
#
# Fill assumption : PASSIVE fills trigger when mid_price crosses the quote.
#   mid >= ask → bid side filled (we bought)   ← note: mid at ask = someone hit us
#   mid <= bid → ask side filled (we sold)
# This is OPTIMISTIC — in reality there is spread and queue position.
# The backtester flags this explicitly.
#
# AGGRESSIVE fills: always assumed to fill at mid (optimistic but acceptable
# as a proxy when the spread is very wide, as with Osmium).


@dataclass
class BacktestResult:
    label:          str
    total_pnl:      float
    realized_pnl:   float
    unrealized_pnl: float
    position_peak:  int
    soft_violations: int       # ticks where |position| > soft limit
    fill_count:     int
    ts_series:      list = field(default_factory=list, repr=False)
    pnl_series:     list = field(default_factory=list, repr=False)
    pos_series:     list = field(default_factory=list, repr=False)


# ── helpers ───────────────────────────────────────────────────────────────────

def _mark_to_market(position: int, avg_cost: float, mid: float) -> float:
    return position * (mid - avg_cost)


def _update_avg_cost(avg_cost: float, position: int,
                     fill_qty: int, fill_price: float) -> tuple[float, float]:
    """Returns (new_avg_cost, realized_pnl)."""
    if fill_qty == 0:
        return avg_cost, 0.0
    new_pos = position + fill_qty
    if new_pos == 0:
        realized = -fill_qty * (fill_price - avg_cost)
        return 0.0, realized
    if position == 0 or (position > 0) == (fill_qty > 0):
        # Same direction: update avg cost
        new_avg = (avg_cost * position + fill_price * fill_qty) / new_pos
        return new_avg, 0.0
    # Partial close
    closed  = min(abs(position), abs(fill_qty))
    sign    = 1 if position > 0 else -1
    realized = sign * closed * (fill_price - avg_cost)
    if abs(fill_qty) <= abs(position):
        return avg_cost, realized
    # Flip: remaining opens at fill price
    return fill_price, realized


# ── A.  Pepper exit backtest ──────────────────────────────────────────────────

def backtest_pepper_exit(
    df: pd.DataFrame,
    prev_day_close: float,
    label: str = "PepperExit",
    entry_threshold: float = 2.0,   # buy when fair_value - mid > threshold
    max_size: int = 5,
) -> BacktestResult:
    """
    Simulates pepper trend strategy WITH all exit fixes.
    Entry: aggressive buy when mid is cheap vs fair value.
    Exit: circuit breaker, slope-weakening trigger, EOS ramp.
    """
    ts_arr  = df["timestamp"].values.astype(float)
    mid_arr = df["mid_price"].values.astype(float)
    n       = len(ts_arr)

    est       = PepperEstimator(prev_day_close)
    position  = 0
    avg_cost  = 0.0
    realized  = 0.0
    fills     = 0
    violations = 0

    ts_out, pnl_out, pos_out = [], [], []

    for i in range(n):
        ts  = ts_arr[i]
        mid = mid_arr[i]
        fv  = est.update(ts, mid)

        # Approximate best_bid/ask from mid (no order book, conservative)
        best_bid = int(mid - 1)
        best_ask = int(mid + 1)

        # ── Forced exit orders (A) ────────────────────────────────────────────
        exit_ords = pepper_exit_orders(
            position, ts, est.slope, best_bid, best_ask
        )
        for price, qty in exit_ords:
            # Aggressive fills assumed at mid
            fill_price = mid
            avg_cost, rp = _update_avg_cost(avg_cost, position, qty, fill_price)
            realized  += rp
            position  += qty
            fills     += 1

        # ── Entry (only if gate allows) ────────────────────────────────────────
        if est.warmed_up and pepper_should_buy(position, ts, est.slope):
            if fv - mid > entry_threshold:
                qty        = min(max_size, PEPPER_POS_LIMIT - position)
                fill_price = mid                    # aggressive taker
                avg_cost, rp = _update_avg_cost(avg_cost, position, qty, fill_price)
                realized  += rp
                position  += qty
                fills     += 1

        # ── Tracking ──────────────────────────────────────────────────────────
        if abs(position) > OSMIUM_SOFT_LIMIT:
            violations += 1
        unrealized = _mark_to_market(position, avg_cost, mid)
        ts_out.append(ts); pnl_out.append(realized + unrealized); pos_out.append(position)

    unrealized_final = _mark_to_market(position, avg_cost, mid_arr[-1])
    return BacktestResult(
        label=label,
        total_pnl=realized + unrealized_final,
        realized_pnl=realized,
        unrealized_pnl=unrealized_final,
        position_peak=max(abs(p) for p in pos_out) if pos_out else 0,
        soft_violations=violations,
        fill_count=fills,
        ts_series=ts_out, pnl_series=pnl_out, pos_series=pos_out,
    )


def backtest_pepper_baseline(
    df: pd.DataFrame,
    prev_day_close: float,
    label: str = "PepperBaseline",
    entry_threshold: float = 2.0,
    max_size: int = 5,
) -> BacktestResult:
    """
    Baseline: same entry, but exit only via passive limit orders (the broken
    clear_position_order logic).  Proxied here as: only sell if mid > fair_value + 1.
    No forced exit, no circuit breaker.
    """
    ts_arr  = df["timestamp"].values.astype(float)
    mid_arr = df["mid_price"].values.astype(float)
    n       = len(ts_arr)

    est       = PepperEstimator(prev_day_close)
    position  = 0
    avg_cost  = 0.0
    realized  = 0.0
    fills     = 0
    violations = 0

    ts_out, pnl_out, pos_out = [], [], []

    for i in range(n):
        ts  = ts_arr[i]
        mid = mid_arr[i]
        fv  = est.update(ts, mid)

        # Passive sell: only when mid drifts above our position cost
        if position > 0 and mid > avg_cost + 1:
            qty        = -min(5, position)
            fill_price = mid
            avg_cost, rp = _update_avg_cost(avg_cost, position, qty, fill_price)
            realized  += rp
            position  += qty
            fills     += 1

        # Entry (no gate — mirrors broken logic)
        if est.warmed_up and position < PEPPER_POS_LIMIT:
            if fv - mid > entry_threshold:
                qty        = min(max_size, PEPPER_POS_LIMIT - position)
                fill_price = mid
                avg_cost, rp = _update_avg_cost(avg_cost, position, qty, fill_price)
                realized  += rp
                position  += qty
                fills     += 1

        if abs(position) > OSMIUM_SOFT_LIMIT:
            violations += 1
        unrealized = _mark_to_market(position, avg_cost, mid)
        ts_out.append(ts); pnl_out.append(realized + unrealized); pos_out.append(position)

    unrealized_final = _mark_to_market(position, avg_cost, mid_arr[-1])
    return BacktestResult(
        label=label,
        total_pnl=realized + unrealized_final,
        realized_pnl=realized,
        unrealized_pnl=unrealized_final,
        position_peak=max(abs(p) for p in pos_out) if pos_out else 0,
        soft_violations=violations,
        fill_count=fills,
        ts_series=ts_out, pnl_series=pnl_out, pos_series=pos_out,
    )


# ── B.  Osmium nonlinear skew backtest ────────────────────────────────────────

def backtest_osmium_skew(
    df: pd.DataFrame,
    fair_value: float,
    half_spread: int,
    label: str,
    use_nonlinear_skew: bool = True,
    soft_limit: int = OSMIUM_SOFT_LIMIT,
    max_size: int = 5,
) -> BacktestResult:
    """
    Passive market-making around a fixed fair value.
    Fill model: if mid_price crosses our quote price, assume passive fill.
    ASSUMPTION: fills are optimistic (first in queue, no adverse selection modelled).
    """
    ts_arr  = df["timestamp"].values.astype(float)
    mid_arr = df["mid_price"].values.astype(float)

    position  = 0
    avg_cost  = 0.0
    realized  = 0.0
    fills     = 0
    violations = 0

    ts_out, pnl_out, pos_out = [], [], []

    for i in range(len(ts_arr)):
        ts  = ts_arr[i]
        mid = mid_arr[i]

        if use_nonlinear_skew:
            bid, ask = osmium_quote_prices(fair_value, position, half_spread)
        else:
            # Flat 1-tick skew (original broken logic)
            sk  = 1 if position > 0 else (-1 if position < 0 else 0)
            bid = round(fair_value - half_spread - sk)
            ask = round(fair_value + half_spread - sk)

        # Passive fill model
        if mid <= bid and position < OSMIUM_HARD_LIMIT:
            qty        = min(max_size, OSMIUM_HARD_LIMIT - position)
            fill_price = float(bid)
            avg_cost, rp = _update_avg_cost(avg_cost, position, qty, fill_price)
            realized  += rp; position += qty; fills += 1

        if mid >= ask and position > -OSMIUM_HARD_LIMIT:
            qty        = -min(max_size, OSMIUM_HARD_LIMIT + position)
            fill_price = float(ask)
            avg_cost, rp = _update_avg_cost(avg_cost, position, qty, fill_price)
            realized  += rp; position += qty; fills += 1

        if abs(position) > soft_limit:
            violations += 1
        unrealized = _mark_to_market(position, avg_cost, mid)
        ts_out.append(ts); pnl_out.append(realized + unrealized); pos_out.append(position)

    unrealized_final = _mark_to_market(position, avg_cost, mid_arr[-1])
    return BacktestResult(
        label=label,
        total_pnl=realized + unrealized_final,
        realized_pnl=realized,
        unrealized_pnl=unrealized_final,
        position_peak=max(abs(p) for p in pos_out) if pos_out else 0,
        soft_violations=violations,
        fill_count=fills,
        ts_series=ts_out, pnl_series=pnl_out, pos_series=pos_out,
    )


# ── C.  Osmium phase switching backtest ───────────────────────────────────────

def backtest_osmium_phases(
    df: pd.DataFrame,
    fair_value: float,
    label: str = "OsmiumPhased",
) -> BacktestResult:
    """
    Market-making with phase-aware parameters and nonlinear skew.
    """
    ts_arr  = df["timestamp"].values.astype(float)
    mid_arr = df["mid_price"].values.astype(float)

    position  = 0
    avg_cost  = 0.0
    realized  = 0.0
    fills     = 0
    violations = 0

    ts_out, pnl_out, pos_out = [], [], []

    for i in range(len(ts_arr)):
        ts    = ts_arr[i]
        mid   = mid_arr[i]
        phase = get_osmium_phase(ts)
        hs    = phase["half_spread"]
        ms    = phase["max_size"]
        sl    = phase["soft_limit"]

        bid, ask = osmium_quote_prices_phased(fair_value, position, ts)

        if mid <= bid and position < OSMIUM_HARD_LIMIT:
            qty        = min(ms, OSMIUM_HARD_LIMIT - position)
            fill_price = float(bid)
            avg_cost, rp = _update_avg_cost(avg_cost, position, qty, fill_price)
            realized  += rp; position += qty; fills += 1

        if mid >= ask and position > -OSMIUM_HARD_LIMIT:
            qty        = -min(ms, OSMIUM_HARD_LIMIT + position)
            fill_price = float(ask)
            avg_cost, rp = _update_avg_cost(avg_cost, position, qty, fill_price)
            realized  += rp; position += qty; fills += 1

        if abs(position) > sl:
            violations += 1
        unrealized = _mark_to_market(position, avg_cost, mid)
        ts_out.append(ts); pnl_out.append(realized + unrealized); pos_out.append(position)

    unrealized_final = _mark_to_market(position, avg_cost, mid_arr[-1])
    return BacktestResult(
        label=label,
        total_pnl=realized + unrealized_final,
        realized_pnl=realized,
        unrealized_pnl=unrealized_final,
        position_peak=max(abs(p) for p in pos_out) if pos_out else 0,
        soft_violations=violations,
        fill_count=fills,
        ts_series=ts_out, pnl_series=pnl_out, pos_series=pos_out,
    )


# ── D.  Pepper seeded opening backtest ────────────────────────────────────────

def backtest_pepper_seed_comparison(
    df: pd.DataFrame,
    prev_day_close: float,
) -> list[BacktestResult]:
    """
    Compares two anchor strategies during the first 5 000 ticks:
        SEED : slope=0.001, anchor=prev_close+1000, start_px=mean(first 20)
        NOSEED: slope=0 (flat), anchor=first tick price
    Returns fair-value MAE series, not PnL (anchor fix affects prediction,
    PnL impact shows up via entry quality in the full pepper exit backtest).
    """
    ts_arr  = df["timestamp"].values.astype(float)
    mid_arr = df["mid_price"].values.astype(float)
    n       = min(5_000, len(ts_arr))

    results = []
    for seeded in [True, False]:
        label = "Seeded" if seeded else "NoSeed"
        est   = PepperEstimator(prev_day_close) if seeded else _NaiveEstimator(mid_arr[0])
        errors = []
        for i in range(n):
            fv  = est.update(ts_arr[i], mid_arr[i])
            errors.append(abs(fv - mid_arr[i]))
        mae = float(np.mean(errors))
        # Pack into BacktestResult (pnl fields carry MAE, violations = ticks > 3 error)
        results.append(BacktestResult(
            label=f"Seed_{label}_MAE",
            total_pnl=-mae,              # negative MAE so "higher is better" applies
            realized_pnl=-mae,
            unrealized_pnl=0,
            position_peak=0,
            soft_violations=int(sum(e > 3.0 for e in errors)),
            fill_count=n,
            ts_series=ts_arr[:n].tolist(),
            pnl_series=[-e for e in errors],
            pos_series=[0] * n,
        ))
    return results


class _NaiveEstimator:
    """Baseline: first-tick anchor, zero slope (pre-fix dead zone proxy)."""
    def __init__(self, first_price: float):
        self._anchor = first_price
        self._first_ts: Optional[float] = None
        self.slope = 0.0
        self.warmed_up = False

    def update(self, ts: float, mid: float) -> float:
        if self._first_ts is None:
            self._first_ts = ts
        return self._anchor  # flat — no slope during dead zone


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER — load data and run all backtests
# ══════════════════════════════════════════════════════════════════════════════

def run_all(
    pepper_df: pd.DataFrame,
    osmium_df: pd.DataFrame,
    osmium_fair_value: float = 10_000.0,
    prev_day_close_pepper: float = 0.0,
    osmium_half_spread: int = 9,
):
    """
    Call this with your actual DataFrames.

    pepper_df  : columns [timestamp, mid_price] for INTARIAN_PEPPER_ROOT
    osmium_df  : columns [timestamp, mid_price] for ASH_COATED_OSMIUM
    osmium_fair_value : known or estimated fair value (e.g. 10 000)
    prev_day_close_pepper : last mid_price of previous day for pepper
    """
    if prev_day_close_pepper == 0.0:
        prev_day_close_pepper = float(pepper_df["mid_price"].iloc[0])

    results: list[BacktestResult] = []

    # ── A: Pepper exit ────────────────────────────────────────────────────────
    r_exit     = backtest_pepper_exit(pepper_df, prev_day_close_pepper,
                                       label="A_PepperExit_Fixed")
    r_baseline = backtest_pepper_baseline(pepper_df, prev_day_close_pepper,
                                           label="A_PepperBaseline")
    results += [r_exit, r_baseline]

    # ── B: Osmium skew ────────────────────────────────────────────────────────
    r_nl  = backtest_osmium_skew(osmium_df, osmium_fair_value, osmium_half_spread,
                                  label="B_Osmium_NonlinearSkew", use_nonlinear_skew=True)
    r_lin = backtest_osmium_skew(osmium_df, osmium_fair_value, osmium_half_spread,
                                  label="B_Osmium_FlatSkew",      use_nonlinear_skew=False)
    results += [r_nl, r_lin]

    # ── C: Osmium phases ──────────────────────────────────────────────────────
    r_ph  = backtest_osmium_phases(osmium_df, osmium_fair_value, label="C_Osmium_Phased")
    r_nph = backtest_osmium_skew(osmium_df, osmium_fair_value, osmium_half_spread,
                                  label="C_Osmium_NoPhase",  use_nonlinear_skew=True)
    results += [r_ph, r_nph]

    # ── D: Pepper seed ────────────────────────────────────────────────────────
    seed_results = backtest_pepper_seed_comparison(pepper_df, prev_day_close_pepper)
    results += seed_results

    # ── Print report ──────────────────────────────────────────────────────────
    _print_report(results)
    return results


def _print_report(results: list[BacktestResult]):
    hdr = f"{'Label':<35} {'TotalPnL':>10} {'RealPnL':>10} {'PosPeak':>8} {'SoftViols':>10} {'Fills':>7}"
    sep = "-" * len(hdr)
    print("\n" + "=" * len(hdr))
    print("BACKTEST REPORT")
    print("=" * len(hdr))
    print(hdr)
    print(sep)
    for r in results:
        print(
            f"{r.label:<35} {r.total_pnl:>10.1f} {r.realized_pnl:>10.1f} "
            f"{r.position_peak:>8d} {r.soft_violations:>10d} {r.fill_count:>7d}"
        )
    print(sep)
    print(
        "\n[!] FILL ASSUMPTIONS ARE OPTIMISTIC:"
        "\n    Passive fills trigger when mid_price touches quote (no queue, no spread cost)."
        "\n    Aggressive fills execute at mid_price (no slippage)."
        "\n    Real fills will be worse — treat absolute PnL numbers as upper bounds."
        "\n    Focus on RELATIVE improvement between fixed and baseline, not absolute values."
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUICK INTEGRATION REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
#
# In your Trader.run() method, for PEPPER:
#
#   # Initialise once (outside run, e.g. in __init__):
#   self.pepper_est = PepperEstimator(prev_day_close=PREV_PEPPER_CLOSE)
#
#   # Every tick:
#   fv = self.pepper_est.update(state.timestamp, mid_price)
#   if pepper_should_buy(position, state.timestamp, self.pepper_est.slope):
#       orders.append(Order(product, bid_price, +size))
#   for price, qty in pepper_exit_orders(position, state.timestamp,
#                                         self.pepper_est.slope, best_bid, best_ask):
#       orders.append(Order(product, price, qty))
#
# For OSMIUM:
#
#   bid, ask = osmium_quote_prices_phased(fair_value, position, state.timestamp)
#   orders.append(Order(product, bid, +size))
#   orders.append(Order(product, ask, -size))
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Standalone test: load the training day CSVs ───────────────────────────
    import os

    PEPPER_FILE = "prices_round_1_day_0 (2).csv"
    OSMIUM_FILE = "prices_round_1_day_0 (2).csv"    # same file, filter below

    if not os.path.exists(PEPPER_FILE):
        print(f"Data file not found: {PEPPER_FILE}")
        print("To run: call run_all(pepper_df, osmium_df, ...) with your DataFrames.")
    else:
        raw = pd.read_csv(PEPPER_FILE, sep=";")

        pepper_df = (raw[raw["product"] == "INTARIAN_PEPPER_ROOT"]
                     [["timestamp", "mid_price"]]
                     .dropna()
                     .sort_values("timestamp")
                     .reset_index(drop=True))

        osmium_df = (raw[raw["product"] == "ASH_COATED_OSMIUM"]
                     [["timestamp", "mid_price"]]
                     .dropna()
                     .sort_values("timestamp")
                     .reset_index(drop=True))

        # Previous day close from day -1 file if available
        prev_close_file = "prices_round_1_day_-1.csv"
        if os.path.exists(prev_close_file):
            prev = pd.read_csv(prev_close_file, sep=";")
            prev = prev[prev["product"] == "INTARIAN_PEPPER_ROOT"]
            prev_close = float(prev["mid_price"].dropna().iloc[-1])
        else:
            prev_close = float(pepper_df["mid_price"].iloc[0])

        osmium_fv = float(osmium_df["mid_price"].mean())

        run_all(
            pepper_df=pepper_df,
            osmium_df=osmium_df,
            osmium_fair_value=osmium_fv,
            prev_day_close_pepper=prev_close,
            osmium_half_spread=9,
        )
