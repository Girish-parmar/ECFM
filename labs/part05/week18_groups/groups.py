"""Week 18 — Statistical edge and the advanced indicator groups (Part 5, S5–S8).

Edge testing (next-bar entry, permutation test, Benjamini–Hochberg), direction (KAMA, ADX/DMI, SuperTrend),
volatility estimators (close-to-close, Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang), momentum (CCI, ROC),
price-volume (MFI, A/D, session and anchored VWAP, volume profile) and multi-timeframe alignment without look-ahead.
Golden values from TA-Lib are in golden/groups.npz. Uses your week 17 `sma` and `atr`.
Fill in every block marked "Your turn", then run:  python -m pytest week18_groups
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _loader import load

core = load("week17_core", "core")
sma, atr = core.sma, core.atr


# ----------------------------------------------------------------------- S5 edge testing
def forward_returns(open_: np.ndarray, horizon: int) -> np.ndarray:
    """Log return of a trade that ENTERS AT THE NEXT BAR'S OPEN and exits `horizon` bars later at the open:
    fwd[t] = ln(open[t+1+horizon] / open[t+1]); NaN where t+1+horizon is past the end."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def pattern_edge(signal: np.ndarray, open_: np.ndarray, horizon: int = 5, n_perm: int = 2000,
                 seed: int = 0) -> dict:
    """Event study of a boolean/±1 signal (a bar counts as an event where signal != 0).
    Using forward_returns (drop NaN rows first):
      edge     = mean fwd return over event bars − mean over ALL valid bars
      p_value  = two-sided permutation p-value: draw k = #events bars WITHOUT replacement n_perm times
                 (rng = np.random.default_rng(seed)); null_i = mean(draw) − mean(all);
                 p = (#{|null_i| >= |edge|} + 1) / (n_perm + 1)
      hit_rate = share of event bars with fwd > 0;   n = #events.
    With no events return {"n": 0, "edge": nan, "p_value": 1.0, "hit_rate": nan}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def bh_adjust(p_values) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (same order as the input): sort ascending, q_(i) = p_(i) × m / i,
    then make them non-decreasing from the largest down (running minimum from the end) and cap at 1.
    A test is a discovery at FDR level α when its adjusted p-value <= α."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------- S6 direction
def kama(close: np.ndarray, n: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Kaufman adaptive MA (TA-Lib convention). Efficiency ratio ER[t] = |C[t]−C[t−n]| / Σ|ΔC| over the same n
    changes (0 if the sum is 0); sc = (ER × (2/(fast+1) − 2/(slow+1)) + 2/(slow+1))²;
    KAMA starts from C[n−1] (not reported) and KAMA[t] = KAMA[t−1] + sc × (C[t] − KAMA[t−1]) for t >= n.
    Look-back n."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def adx(high, low, close, n: int = 14):
    """Wilder ADX with +DI and −DI, as TA-Lib (returns adx, plus_di, minus_di).
    +DM = up move if up > down and up > 0 else 0 (up = H[t]−H[t−1], down = L[t−1]−L[t]); −DM the mirror; TR as usual.
    Wilder sums: seed with the sum of the first n−1 values (bars 1..n−1), then S = S − S/n + x for bars n, n+1, …
    +DI = 100 × S(+DM)/S(TR), −DI likewise, both from bar n. DX = 100 × |+DI − −DI| / (+DI + −DI) (0 if the sum is 0).
    ADX[2n−1] = mean(DX[n..2n−1]); then ADX = (ADX_prev × (n−1) + DX) / n. Look-backs: DI n, ADX 2n−1."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def supertrend(high, low, close, n: int = 10, mult: float = 3.0):
    """SuperTrend (path-dependent, needs a loop). Returns (line, direction) with direction +1/−1 (0 in warm-up).
    basic bands: hl2 ± mult × ATR(n). Final upper fu[i] = basic upper if it is lower than fu[i−1] OR C[i−1] > fu[i−1],
    else fu[i−1]; final lower fl mirrors it. Direction: +1 if C[i] > fu[i−1], −1 if C[i] < fl[i−1], else previous
    (start at +1). line = fl when +1, fu when −1. Values from bar n+1 on (NaN before)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------- S7 volatility estimators
def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(x))
    if ok.size and x.size - ok[0] >= n:
        s = ok[0]
        c = np.cumsum(np.insert(x[s:], 0, 0.0))
        out[s + n - 1:] = (c[n:] - c[:-n]) / n
    return out


def realized_vol(o, h, l, c, n: int = 20, method: str = "parkinson", periods: int = 252) -> np.ndarray:  # noqa: E741
    """Rolling annualized volatility sqrt(periods × mean per-bar variance over the last n bars), for method:
      'close'            ln(C/C₋₁)²                              (look-back n)
      'parkinson'        ln(H/L)² / (4 ln 2)                     (look-back n−1)
      'garman_klass'     ½ ln(H/L)² − (2 ln 2 − 1) ln(C/O)²      (look-back n−1)
      'rogers_satchell'  ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)        (look-back n−1)
      'yang_zhang'       σ²_overnight + k σ²_open→close + (1−k) mean(RS), with overnight = ln(O/C₋₁),
                         open→close = ln(C/O), both SAMPLE variances (ddof=1) over the window,
                         k = 0.34 / (1.34 + (n+1)/(n−1))          (look-back n)
    Unknown method -> ValueError. Use _roll_mean (it handles leading NaN)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ---------------------------------------------------------------------------- momentum
def cci(high, low, close, n: int = 20) -> np.ndarray:
    """Commodity Channel Index: TP = (H+L+C)/3; CCI = (TP − SMA(TP, n)) / (0.015 × mean |TP − SMA| over the window).
    0 when the mean deviation is 0. Look-back n−1."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def roc(close: np.ndarray, n: int = 10) -> np.ndarray:
    """Rate of change in percent: 100 × (C[t]/C[t−n] − 1). Look-back n."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------- S8 price–volume–time
def mfi(high, low, close, volume, n: int = 14) -> np.ndarray:
    """Money Flow Index (TA-Lib): TP = (H+L+C)/3, flow = TP × V. A bar's flow is positive if TP > TP₋₁,
    negative if TP < TP₋₁, ignored if equal. MFI[t] = 100 × Σpos / (Σpos + Σneg) over bars t−n+1..t (0 if both 0).
    Look-back n."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def ad_line(high, low, close, volume) -> np.ndarray:
    """Accumulation/Distribution: cumulative sum of ((C−L) − (H−C)) / (H−L) × V (0 for a bar with H == L)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def session_vwap(index: pd.DatetimeIndex, price: np.ndarray, volume: np.ndarray,
                 tz: str = "America/New_York") -> np.ndarray:
    """VWAP that RESETS at each new session (calendar date in `tz`): Σ(p×v)/Σv from the session's first bar
    to the current bar. NaN while the session's cumulative volume is 0."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def anchored_vwap(price: np.ndarray, volume: np.ndarray, anchor: int) -> np.ndarray:
    """VWAP accumulated from bar `anchor` (inclusive), e.g. an earnings day or a confirmed swing low; NaN before it."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def volume_profile(price: np.ndarray, volume: np.ndarray, bins: int = 50, value_area: float = 0.70) -> dict:
    """Histogram of volume by price (np.histogram with `bins`, weights=volume). POC = midpoint of the fullest bin.
    Value area: start at the POC bin and repeatedly add the heavier neighbouring bin (right wins ties; a missing
    neighbour counts as −1) until it holds >= value_area of the volume. Return {"poc", "val", "vah"} where
    val/vah are the lower edge of the lowest and upper edge of the highest bin in the value area."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def align_higher_tf(ltf: pd.DataFrame, fn, rule: str = "1h", offset: str = "30min") -> pd.Series:
    """Compute fn on COMPLETED higher-timeframe closes and map it onto the lower-timeframe bars WITHOUT look-ahead.
    Bars are labelled by their OPEN time. Steps: resample close with label='left', closed='left', offset=offset,
    .last(), drop empty bins; apply fn to the HTF close array; SHIFT BY ONE HTF BAR (a bar is known only after it
    closes); forward-fill onto ltf.index. Returns a Series aligned with ltf.index."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
