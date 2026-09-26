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
    # >>> SOLUTION
    def prev(a):
        a = np.asarray(a, dtype=float)
        return np.concatenate([[np.nan], a[:-1]])
    h, l, c = prev(high), prev(low), prev(close)  # noqa: E741
    r = h - l
    if method == "woodie":
        p = (h + l + 2 * c) / 4
        return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h}
    p = (h + l + c) / 3
    if method == "classic":
        return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + r, "S2": p - r}
    if method == "fibonacci":
        out = {"P": p}
        for i, f in enumerate((0.382, 0.618, 1.0), start=1):
            out[f"R{i}"], out[f"S{i}"] = p + f * r, p - f * r
        return out
    if method == "camarilla":
        out = {"P": p}
        for i, d in enumerate((12, 6, 4, 2), start=1):
            out[f"R{i}"], out[f"S{i}"] = c + r * 1.1 / d, c - r * 1.1 / d
        return out
    raise ValueError(f"unknown method {method!r}")
    # <<< SOLUTION


def opening_range(index: pd.DatetimeIndex, high, low, minutes: int = 30, tz: str = "America/New_York",
                  session_start: str = "09:30") -> tuple[np.ndarray, np.ndarray]:
    """Opening-range high and low of each session (bars whose OPEN time is within the first `minutes` after
    session_start, local time `tz`). The range is KNOWN only once it is complete, so both outputs are NaN for the
    opening-range bars themselves and for any earlier bars; from the first bar at or after start+minutes to the end of
    that session they hold the range. Index = bar open times (UTC)."""
    # >>> SOLUTION
    local = index.tz_convert(tz)
    start = pd.Timedelta(session_start + ":00")
    since = (local - local.normalize()) - start
    in_or = (since >= pd.Timedelta(0)) & (since < pd.Timedelta(minutes=minutes))
    after = since >= pd.Timedelta(minutes=minutes)
    df = pd.DataFrame({"h": np.where(in_or, high, np.nan), "l": np.where(in_or, low, np.nan),
                       "day": local.date}, index=index)
    g = df.groupby("day")
    orh, orl = g["h"].transform("max").to_numpy(), g["l"].transform("min").to_numpy()
    return np.where(after, orh, np.nan), np.where(after, orl, np.nan)
    # <<< SOLUTION


def kde_levels(prices, bandwidth: float, n_levels: int = 3, grid: int = 512) -> list[float]:
    """Support/resistance levels as the highest peaks of a Gaussian kernel density of `prices` (e.g. confirmed swing
    prices). Density on a grid of `grid` points from min−3·bw to max+3·bw: d(x) = Σ exp(−½((x−p)/bw)²).
    Peaks = interior grid points strictly greater than both neighbours. Return the `n_levels` peak locations with the
    highest density, sorted by density (highest first)."""
    # >>> SOLUTION
    p = np.asarray(prices, dtype=float)
    x = np.linspace(p.min() - 3 * bandwidth, p.max() + 3 * bandwidth, grid)
    d = np.exp(-0.5 * ((x[:, None] - p[None, :]) / bandwidth) ** 2).sum(axis=1)
    peaks = np.flatnonzero((d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])) + 1
    best = peaks[np.argsort(d[peaks])[::-1]][:n_levels]
    return [float(x[i]) for i in best]
    # <<< SOLUTION


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
    # >>> SOLUTION
    pivots, trend, hi_i, lo_i = [], 0, 0, 0
    for i in range(1, len(high)):
        if trend >= 0 and high[i] >= high[hi_i]:
            hi_i = i
        if trend <= 0 and low[i] <= low[lo_i]:
            lo_i = i
        if trend >= 0 and low[i] <= high[hi_i] * (1 - pct):
            pivots.append(Pivot(hi_i, i, float(high[hi_i]), +1))
            trend, lo_i = -1, i
        elif trend <= 0 and high[i] >= low[lo_i] * (1 + pct):
            pivots.append(Pivot(lo_i, i, float(low[lo_i]), -1))
            trend, hi_i = +1, i
    return pivots
    # <<< SOLUTION


def fractals(high, low, k: int = 2) -> list[Pivot]:
    """Williams fractals: a swing high at i when high[i] is STRICTLY greater than the k highs on each side; a swing low
    at i when low[i] is strictly lower than the k lows on each side. confirm_idx = i + k. Sorted by (idx, kind)
    with lows (−1) before highs (+1) at the same bar."""
    # >>> SOLUTION
    h, l = np.asarray(high, float), np.asarray(low, float)  # noqa: E741
    out = []
    for i in range(k, len(h) - k):
        nb = np.r_[i - k:i, i + 1:i + k + 1]
        if (l[i] < l[nb]).all():
            out.append(Pivot(i, i + k, float(l[i]), -1))
        if (h[i] > h[nb]).all():
            out.append(Pivot(i, i + k, float(h[i]), +1))
    return out
    # <<< SOLUTION


# ------------------------------------------------------------------- S11 chart patterns
def double_tops_bottoms(pivots: list[Pivot], tol: float, min_sep: int = 5) -> list[dict]:
    """Scan consecutive pivot triples. Double TOP: kinds (+1, −1, +1) with |p1 − p3| <= tol and p3.idx − p1.idx >=
    min_sep; double BOTTOM: kinds (−1, +1, −1), same rules. `tol` is in price units (pass k × ATR at the second pivot).
    Return dicts {"kind": "double_top"|"double_bottom", "first", "neckline" (middle pivot price), "second",
    "known_from" (second pivot's confirm_idx)} in pivot order."""
    # >>> SOLUTION
    out = []
    for a, b, c in zip(pivots, pivots[1:], pivots[2:]):
        kinds = (a.kind, b.kind, c.kind)
        if kinds not in ((1, -1, 1), (-1, 1, -1)) or c.idx - a.idx < min_sep or abs(a.price - c.price) > tol:
            continue
        out.append({"kind": "double_top" if a.kind == 1 else "double_bottom", "first": a, "neckline": b.price,
                    "second": c, "known_from": c.confirm_idx})
    return out
    # <<< SOLUTION


def head_shoulders(pivots: list[Pivot], tol: float) -> list[dict]:
    """Scan consecutive 5-pivot windows. TOP: kinds (+1, −1, +1, −1, +1) = LS, L1, HEAD, L2, RS with HEAD above both
    shoulders and |LS − RS| <= tol. INVERSE: kinds (−1, +1, −1, +1, −1) with the head below both shoulders.
    Neckline through L1 and L2: slope = (L2.price − L1.price)/(L2.idx − L1.idx).
    Return dicts {"kind": "head_shoulders"|"inverse_head_shoulders", "head", "neck": (L1, L2), "slope",
    "known_from": RS.confirm_idx}."""
    # >>> SOLUTION
    out = []
    for w in zip(pivots, pivots[1:], pivots[2:], pivots[3:], pivots[4:]):
        ls, n1, hd, n2, rs = w
        kinds = tuple(p.kind for p in w)
        top = kinds == (1, -1, 1, -1, 1) and hd.price > max(ls.price, rs.price)
        inv = kinds == (-1, 1, -1, 1, -1) and hd.price < min(ls.price, rs.price)
        if not (top or inv) or abs(ls.price - rs.price) > tol:
            continue
        out.append({"kind": "head_shoulders" if top else "inverse_head_shoulders", "head": hd, "neck": (n1, n2),
                    "slope": (n2.price - n1.price) / (n2.idx - n1.idx), "known_from": rs.confirm_idx})
    return out
    # <<< SOLUTION


# --------------------------------------------------------------------------- S12 actions
def crossover(a, b) -> np.ndarray:
    """a[t] > b[t] and a[t−1] <= b[t−1] (b may be a scalar). A tie counts as NOT above. NaN comparisons are False."""
    # >>> SOLUTION
    a = np.asarray(a, dtype=float)
    b = np.broadcast_to(np.asarray(b, dtype=float), a.shape)
    out = np.zeros(a.shape, dtype=bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out
    # <<< SOLUTION


def crossunder(a, b) -> np.ndarray:
    """a[t] < b[t] and a[t−1] >= b[t−1]."""
    # >>> SOLUTION
    a = np.asarray(a, dtype=float)
    b = np.broadcast_to(np.asarray(b, dtype=float), a.shape)
    out = np.zeros(a.shape, dtype=bool)
    out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out
    # <<< SOLUTION


def gap_up(open_, high) -> np.ndarray:
    """Full gap up: open[t] > high[t−1]."""
    # >>> SOLUTION
    out = np.zeros(len(open_), dtype=bool)
    out[1:] = np.asarray(open_)[1:] > np.asarray(high)[:-1]
    return out
    # <<< SOLUTION


def inside_bar(high, low) -> np.ndarray:
    """high[t] < high[t−1] and low[t] > low[t−1]."""
    # >>> SOLUTION
    h, l = np.asarray(high), np.asarray(low)  # noqa: E741
    out = np.zeros(h.shape, dtype=bool)
    out[1:] = (h[1:] < h[:-1]) & (l[1:] > l[:-1])
    return out
    # <<< SOLUTION


def nr_n(high, low, n: int = 7) -> np.ndarray:
    """NR-n: today's range (H−L) is STRICTLY the narrowest of the last n bars (including today). False in warm-up."""
    # >>> SOLUTION
    r = np.asarray(high, float) - np.asarray(low, float)
    out = np.zeros(r.shape, dtype=bool)
    if r.size >= n:
        w = np.lib.stride_tricks.sliding_window_view(r, n)
        out[n - 1:] = (w[:, -1:] < w[:, :-1]).all(axis=1)
    return out
    # <<< SOLUTION


def new_high(x, n: int) -> np.ndarray:
    """x[t] > max(x[t−n .. t−1]) (a new n-bar high); False for the first n bars."""
    # >>> SOLUTION
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape, dtype=bool)
    if x.size > n:
        prev_max = np.lib.stride_tricks.sliding_window_view(x[:-1], n).max(axis=1)
        out[n:] = x[n:] > prev_max
    return out
    # <<< SOLUTION


def consecutive(cond) -> np.ndarray:
    """Number of consecutive True values ending at t (0 where cond is False). int array."""
    # >>> SOLUTION
    out = np.zeros(len(cond), dtype=int)
    run = 0
    for i, c in enumerate(cond):
        run = run + 1 if c else 0
        out[i] = run
    return out
    # <<< SOLUTION


def bars_since(cond) -> np.ndarray:
    """Bars since cond was last True (0 on a True bar); NaN before the first True."""
    # >>> SOLUTION
    out, last = np.full(len(cond), np.nan), -1
    for i, c in enumerate(cond):
        if c:
            last = i
        if last >= 0:
            out[i] = i - last
    return out
    # <<< SOLUTION


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
        # >>> SOLUTION
        def fn(ctx):
            b = bars_since(self(ctx))
            return ~np.isnan(b) & (np.nan_to_num(b, nan=n) < n)
        return Condition(fn, f"{self.name} within {n}")
        # <<< SOLUTION

    def confirm(self, n: int) -> Condition:
        # >>> SOLUTION
        return Condition(lambda ctx: consecutive(self(ctx)) >= n, f"{self.name} for {n} bars")
        # <<< SOLUTION
