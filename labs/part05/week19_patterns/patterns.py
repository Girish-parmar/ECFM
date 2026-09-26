"""Week 19 — Levels, swings, chart patterns, actions and entry conditions (Part 5, S9–S12).

The rule of the week: anything that needs future bars (a swing high is only a swing once price has turned) carries a
CONFIRMATION INDEX, and signals may use only the confirmation index, never the pivot index.
Fill in every block marked "Your turn", then run:  python -m pytest week19_patterns
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------ S9 levels
def pivot_points(high, low, close, method: str = "classic") -> dict[str, np.ndarray]:
    """Floor pivots for bar t from the PREVIOUS bar's H, L, C (daily bars → today's levels); bar 0 is NaN.
    R = H−L of the previous bar.
      classic   P=(H+L+C)/3, R1=2P−L, S1=2P−H, R2=P+R, S2=P−R
      fibonacci P=(H+L+C)/3, R1=P+0.382R, R2=P+0.618R, R3=P+R, S1..S3 mirror
      camarilla P=(H+L+C)/3, R1..R4 = C + R×1.1/12, /6, /4, /2 ; S1..S4 = C − the same
      woodie    P=(H+L+2C)/4, R1=2P−L, S1=2P−H
    Return a dict of arrays with keys 'P', 'R1', 'S1', ... (exactly the levels listed). Unknown method: ValueError."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def opening_range(index: pd.DatetimeIndex, high, low, minutes: int = 30, tz: str = "America/New_York",
                  session_start: str = "09:30") -> tuple[np.ndarray, np.ndarray]:
    """Opening-range high and low of each session (bars whose OPEN time is within the first `minutes` after
    session_start, local time `tz`). The range is KNOWN only once it is complete, so both outputs are NaN for the
    opening-range bars themselves and for any earlier bars; from the first bar at or after start+minutes to the end of
    that session they hold the range. Index = bar open times (UTC)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def kde_levels(prices, bandwidth: float, n_levels: int = 3, grid: int = 512) -> list[float]:
    """Support/resistance levels as the highest peaks of a Gaussian kernel density of `prices` (e.g. confirmed swing
    prices). Density on a grid of `grid` points from min−3·bw to max+3·bw: d(x) = Σ exp(−½((x−p)/bw)²).
    Peaks = interior grid points strictly greater than both neighbours. Return the `n_levels` peak locations with the
    highest density, sorted by density (highest first)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ---------------------------------------------------------------------- S10 swings
@dataclass(frozen=True)
class Pivot:
    idx: int            # bar where the swing extreme happened
    confirm_idx: int    # first bar where the swing is KNOWN — signals must use this
    price: float
    kind: int           # +1 swing high, −1 swing low


def zigzag(high, low, pct: float) -> list[Pivot]:
    """ZigZag with a percentage reversal threshold. Walk the bars once (t = 1..):
    while trend >= 0 track the highest high (hi_i); while trend <= 0 track the lowest low (lo_i).
    If trend >= 0 and low[t] <= high[hi_i] × (1 − pct): emit swing HIGH Pivot(hi_i, t, high[hi_i], +1), trend = −1,
    lo_i = t. Elif trend <= 0 and high[t] >= low[lo_i] × (1 + pct): emit swing LOW Pivot(lo_i, t, low[lo_i], −1),
    trend = +1, hi_i = t. (Start with trend 0, hi_i = lo_i = 0.)"""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def fractals(high, low, k: int = 2) -> list[Pivot]:
    """Williams fractals: a swing high at i when high[i] is STRICTLY greater than the k highs on each side; a swing low
    at i when low[i] is strictly lower than the k lows on each side. confirm_idx = i + k. Sorted by (idx, kind)
    with lows (−1) before highs (+1) at the same bar."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------- S11 chart patterns
def double_tops_bottoms(pivots: list[Pivot], tol: float, min_sep: int = 5) -> list[dict]:
    """Scan consecutive pivot triples. Double TOP: kinds (+1, −1, +1) with |p1 − p3| <= tol and p3.idx − p1.idx >=
    min_sep; double BOTTOM: kinds (−1, +1, −1), same rules. `tol` is in price units (pass k × ATR at the second pivot).
    Return dicts {"kind": "double_top"|"double_bottom", "first", "neckline" (middle pivot price), "second",
    "known_from" (second pivot's confirm_idx)} in pivot order."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def head_shoulders(pivots: list[Pivot], tol: float) -> list[dict]:
    """Scan consecutive 5-pivot windows. TOP: kinds (+1, −1, +1, −1, +1) = LS, L1, HEAD, L2, RS with HEAD above both
    shoulders and |LS − RS| <= tol. INVERSE: kinds (−1, +1, −1, +1, −1) with the head below both shoulders.
    Neckline through L1 and L2: slope = (L2.price − L1.price)/(L2.idx − L1.idx).
    Return dicts {"kind": "head_shoulders"|"inverse_head_shoulders", "head", "neck": (L1, L2), "slope",
    "known_from": RS.confirm_idx}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------------- S12 actions
def crossover(a, b) -> np.ndarray:
    """a[t] > b[t] and a[t−1] <= b[t−1] (b may be a scalar). A tie counts as NOT above. NaN comparisons are False."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def crossunder(a, b) -> np.ndarray:
    """a[t] < b[t] and a[t−1] >= b[t−1]."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def gap_up(open_, high) -> np.ndarray:
    """Full gap up: open[t] > high[t−1]."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def inside_bar(high, low) -> np.ndarray:
    """high[t] < high[t−1] and low[t] > low[t−1]."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def nr_n(high, low, n: int = 7) -> np.ndarray:
    """NR-n: today's range (H−L) is STRICTLY the narrowest of the last n bars (including today). False in warm-up."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def new_high(x, n: int) -> np.ndarray:
    """x[t] > max(x[t−n .. t−1]) (a new n-bar high); False for the first n bars."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def consecutive(cond) -> np.ndarray:
    """Number of consecutive True values ending at t (0 where cond is False). int array."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def bars_since(cond) -> np.ndarray:
    """Bars since cond was last True (0 on a True bar); NaN before the first True."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


class Condition:
    """Entry rule as a composable, named object (Specification pattern) evaluated on a data context (dict of arrays).
    &, |, ~ combine conditions with names '(a AND b)', '(a OR b)', 'NOT a'.
    within(n): True at t if the condition was True at any of bars t−n+1..t — name 'a within n'.
    confirm(n): True at t if the condition was True on each of the last n bars — name 'a for n bars'."""

    def __init__(self, fn: Callable[[dict], np.ndarray], name: str):
        self.fn, self.name = fn, name

    def __call__(self, ctx: dict) -> np.ndarray:
        return np.asarray(self.fn(ctx), dtype=bool)

    def __and__(self, other: Condition) -> Condition:
        return Condition(lambda x: self(x) & other(x), f"({self.name} AND {other.name})")

    def __or__(self, other: Condition) -> Condition:
        return Condition(lambda x: self(x) | other(x), f"({self.name} OR {other.name})")

    def __invert__(self) -> Condition:
        return Condition(lambda x: ~self(x), f"NOT {self.name}")

    def within(self, n: int) -> Condition:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def confirm(self, n: int) -> Condition:
        raise NotImplementedError("✍️ Your turn: see the docstring")
