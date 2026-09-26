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
    # >>> SOLUTION
    o = np.asarray(open_, dtype=float)
    out = np.full(o.shape, np.nan)
    m = o.size - 1 - horizon
    if m > 0:
        out[:m] = np.log(o[1 + horizon:] / o[1:m + 1])
    return out
    # <<< SOLUTION


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
    # >>> SOLUTION
    fwd = forward_returns(open_, horizon)
    valid = ~np.isnan(fwd)
    sig, fwd = np.asarray(signal)[valid] != 0, fwd[valid]
    k = int(sig.sum())
    if k == 0:
        return {"n": 0, "edge": np.nan, "p_value": 1.0, "hit_rate": np.nan}
    rng = np.random.default_rng(seed)
    base = fwd.mean()
    edge = fwd[sig].mean() - base
    null = np.array([fwd[rng.choice(fwd.size, k, replace=False)].mean() for _ in range(n_perm)]) - base
    p = (np.sum(np.abs(null) >= abs(edge)) + 1) / (n_perm + 1)
    return {"n": k, "edge": float(edge), "p_value": float(p), "hit_rate": float((fwd[sig] > 0).mean())}
    # <<< SOLUTION


def bh_adjust(p_values) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (same order as the input): sort ascending, q_(i) = p_(i) × m / i,
    then make them non-decreasing from the largest down (running minimum from the end) and cap at 1.
    A test is a discovery at FDR level α when its adjusted p-value <= α."""
    # >>> SOLUTION
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out
    # <<< SOLUTION


# ------------------------------------------------------------------------- S6 direction
def kama(close: np.ndarray, n: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Kaufman adaptive MA (TA-Lib convention). Efficiency ratio ER[t] = |C[t]−C[t−n]| / Σ|ΔC| over the same n
    changes (0 if the sum is 0); sc = (ER × (2/(fast+1) − 2/(slow+1)) + 2/(slow+1))²;
    KAMA starts from C[n−1] (not reported) and KAMA[t] = KAMA[t−1] + sc × (C[t] − KAMA[t−1]) for t >= n.
    Look-back n."""
    # >>> SOLUTION
    c = np.asarray(close, dtype=float)
    out = np.full(c.shape, np.nan)
    if c.size <= n:
        return out
    fsc, ssc = 2 / (fast + 1), 2 / (slow + 1)
    absdiff = np.abs(np.diff(c))
    prev = c[n - 1]
    for i in range(n, c.size):
        vol = absdiff[i - n:i].sum()
        er = abs(c[i] - c[i - n]) / vol if vol else 0.0
        sc = (er * (fsc - ssc) + ssc) ** 2
        prev = prev + sc * (c[i] - prev)
        out[i] = prev
    return out
    # <<< SOLUTION


def adx(high, low, close, n: int = 14):
    """Wilder ADX with +DI and −DI, as TA-Lib (returns adx, plus_di, minus_di).
    +DM = up move if up > down and up > 0 else 0 (up = H[t]−H[t−1], down = L[t−1]−L[t]); −DM the mirror; TR as usual.
    Wilder sums: seed with the sum of the first n−1 values (bars 1..n−1), then S = S − S/n + x for bars n, n+1, …
    +DI = 100 × S(+DM)/S(TR), −DI likewise, both from bar n. DX = 100 × |+DI − −DI| / (+DI + −DI) (0 if the sum is 0).
    ADX[2n−1] = mean(DX[n..2n−1]); then ADX = (ADX_prev × (n−1) + DX) / n. Look-backs: DI n, ADX 2n−1."""
    # >>> SOLUTION
    h, l, c = (np.asarray(a, dtype=float) for a in (high, low, close))  # noqa: E741
    N = c.size
    up, dn = h[1:] - h[:-1], l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    pdi, mdi, dx, out = (np.full(N, np.nan) for _ in range(4))
    if N <= 2 * n:
        return out, pdi, mdi
    sp, sm, st = pdm[:n - 1].sum(), mdm[:n - 1].sum(), tr[:n - 1].sum()
    for j in range(n - 1, N - 1):
        sp, sm, st = sp - sp / n + pdm[j], sm - sm / n + mdm[j], st - st / n + tr[j]
        i = j + 1
        pdi[i], mdi[i] = 100 * sp / st, 100 * sm / st
        s = pdi[i] + mdi[i]
        dx[i] = 100 * abs(pdi[i] - mdi[i]) / s if s else 0.0
    out[2 * n - 1] = dx[n:2 * n].mean()
    for i in range(2 * n, N):
        out[i] = (out[i - 1] * (n - 1) + dx[i]) / n
    return out, pdi, mdi
    # <<< SOLUTION


def supertrend(high, low, close, n: int = 10, mult: float = 3.0):
    """SuperTrend (path-dependent, needs a loop). Returns (line, direction) with direction +1/−1 (0 in warm-up).
    basic bands: hl2 ± mult × ATR(n). Final upper fu[i] = basic upper if it is lower than fu[i−1] OR C[i−1] > fu[i−1],
    else fu[i−1]; final lower fl mirrors it. Direction: +1 if C[i] > fu[i−1], −1 if C[i] < fl[i−1], else previous
    (start at +1). line = fl when +1, fu when −1. Values from bar n+1 on (NaN before)."""
    # >>> SOLUTION
    h, l, c = (np.asarray(a, dtype=float) for a in (high, low, close))  # noqa: E741
    a = atr(h, l, c, n)
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fu, fl = ub.copy(), lb.copy()
    line = np.full(c.shape, np.nan)
    d = np.zeros(c.shape, dtype=np.int8)
    for i in range(n + 1, c.size):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if c[i] > fu[i - 1]:
            d[i] = 1
        elif c[i] < fl[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1] if d[i - 1] != 0 else 1
        line[i] = fl[i] if d[i] == 1 else fu[i]
    return line, d
    # <<< SOLUTION


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
    # >>> SOLUTION
    o, h, l, c = (np.asarray(a, dtype=float) for a in (o, h, l, c))  # noqa: E741
    prev = np.concatenate([[np.nan], c[:-1]])
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    if method == "close":
        var = _roll_mean(np.log(c / prev) ** 2, n)
    elif method == "parkinson":
        var = _roll_mean(np.log(h / l) ** 2 / (4 * np.log(2)), n)
    elif method == "garman_klass":
        var = _roll_mean(0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2, n)
    elif method == "rogers_satchell":
        var = _roll_mean(rs, n)
    elif method == "yang_zhang":
        on, oc = np.log(o / prev), np.log(c / o)

        def svar(x):
            m = _roll_mean(x, n)
            return (_roll_mean(x ** 2, n) - m ** 2) * n / (n - 1)
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        var = svar(on) + k * svar(oc) + (1 - k) * _roll_mean(rs, n)
        var[:n] = np.nan
    else:
        raise ValueError(f"unknown method {method!r}")
    return np.sqrt(periods * var)
    # <<< SOLUTION


# ---------------------------------------------------------------------------- momentum
def cci(high, low, close, n: int = 20) -> np.ndarray:
    """Commodity Channel Index: TP = (H+L+C)/3; CCI = (TP − SMA(TP, n)) / (0.015 × mean |TP − SMA| over the window).
    0 when the mean deviation is 0. Look-back n−1."""
    # >>> SOLUTION
    tp = (np.asarray(high, float) + np.asarray(low, float) + np.asarray(close, float)) / 3
    out = np.full(tp.shape, np.nan)
    if tp.size >= n:
        w = np.lib.stride_tricks.sliding_window_view(tp, n)
        m = w.mean(axis=1)
        md = np.abs(w - m[:, None]).mean(axis=1)
        out[n - 1:] = np.divide(tp[n - 1:] - m, 0.015 * md, out=np.zeros(m.shape), where=md > 0)
    return out
    # <<< SOLUTION


def roc(close: np.ndarray, n: int = 10) -> np.ndarray:
    """Rate of change in percent: 100 × (C[t]/C[t−n] − 1). Look-back n."""
    # >>> SOLUTION
    c = np.asarray(close, dtype=float)
    out = np.full(c.shape, np.nan)
    out[n:] = 100 * (c[n:] / c[:-n] - 1)
    return out
    # <<< SOLUTION


# ------------------------------------------------------------------- S8 price–volume–time
def mfi(high, low, close, volume, n: int = 14) -> np.ndarray:
    """Money Flow Index (TA-Lib): TP = (H+L+C)/3, flow = TP × V. A bar's flow is positive if TP > TP₋₁,
    negative if TP < TP₋₁, ignored if equal. MFI[t] = 100 × Σpos / (Σpos + Σneg) over bars t−n+1..t (0 if both 0).
    Look-back n."""
    # >>> SOLUTION
    tp = (np.asarray(high, float) + np.asarray(low, float) + np.asarray(close, float)) / 3
    flow = tp * np.asarray(volume, float)
    d = np.diff(tp)
    pos = np.concatenate([[0.0], np.where(d > 0, flow[1:], 0.0)])
    neg = np.concatenate([[0.0], np.where(d < 0, flow[1:], 0.0)])
    out = np.full(tp.shape, np.nan)
    if tp.size > n:
        sp = np.lib.stride_tricks.sliding_window_view(pos[1:], n).sum(axis=1)
        sn = np.lib.stride_tricks.sliding_window_view(neg[1:], n).sum(axis=1)
        tot = sp + sn
        out[n:] = np.divide(100 * sp, tot, out=np.zeros(tot.shape), where=tot > 0)
    return out
    # <<< SOLUTION


def ad_line(high, low, close, volume) -> np.ndarray:
    """Accumulation/Distribution: cumulative sum of ((C−L) − (H−C)) / (H−L) × V (0 for a bar with H == L)."""
    # >>> SOLUTION
    h, l, c, v = (np.asarray(a, dtype=float) for a in (high, low, close, volume))  # noqa: E741
    rng = h - l
    mfm = np.divide((c - l) - (h - c), rng, out=np.zeros(rng.shape), where=rng > 0)
    return np.cumsum(mfm * v)
    # <<< SOLUTION


def session_vwap(index: pd.DatetimeIndex, price: np.ndarray, volume: np.ndarray,
                 tz: str = "America/New_York") -> np.ndarray:
    """VWAP that RESETS at each new session (calendar date in `tz`): Σ(p×v)/Σv from the session's first bar
    to the current bar. NaN while the session's cumulative volume is 0."""
    # >>> SOLUTION
    df = pd.DataFrame({"pv": np.asarray(price, float) * np.asarray(volume, float), "v": np.asarray(volume, float)},
                      index=index)
    day = index.tz_convert(tz).date
    cum = df.groupby(day).cumsum()
    return np.divide(cum["pv"].to_numpy(), cum["v"].to_numpy(), out=np.full(len(df), np.nan),
                     where=cum["v"].to_numpy() > 0)
    # <<< SOLUTION


def anchored_vwap(price: np.ndarray, volume: np.ndarray, anchor: int) -> np.ndarray:
    """VWAP accumulated from bar `anchor` (inclusive), e.g. an earnings day or a confirmed swing low; NaN before it."""
    # >>> SOLUTION
    p, v = np.asarray(price, float), np.asarray(volume, float)
    out = np.full(p.shape, np.nan)
    cv = np.cumsum(v[anchor:])
    out[anchor:] = np.divide(np.cumsum(p[anchor:] * v[anchor:]), cv, out=np.full(cv.shape, np.nan), where=cv > 0)
    return out
    # <<< SOLUTION


def volume_profile(price: np.ndarray, volume: np.ndarray, bins: int = 50, value_area: float = 0.70) -> dict:
    """Histogram of volume by price (np.histogram with `bins`, weights=volume). POC = midpoint of the fullest bin.
    Value area: start at the POC bin and repeatedly add the heavier neighbouring bin (right wins ties; a missing
    neighbour counts as −1) until it holds >= value_area of the volume. Return {"poc", "val", "vah"} where
    val/vah are the lower edge of the lowest and upper edge of the highest bin in the value area."""
    # >>> SOLUTION
    hist, edges = np.histogram(price, bins=bins, weights=volume)
    poc = int(np.argmax(hist))
    lo = hi = poc
    total, acc = hist.sum(), hist[poc]
    while acc < value_area * total:
        left = hist[lo - 1] if lo > 0 else -1.0
        right = hist[hi + 1] if hi < len(hist) - 1 else -1.0
        if right >= left:
            hi += 1
            acc += hist[hi]
        else:
            lo -= 1
            acc += hist[lo]
    mid = (edges[:-1] + edges[1:]) / 2
    return {"poc": float(mid[poc]), "val": float(edges[lo]), "vah": float(edges[hi + 1])}
    # <<< SOLUTION


def align_higher_tf(ltf: pd.DataFrame, fn, rule: str = "1h", offset: str = "30min") -> pd.Series:
    """Compute fn on COMPLETED higher-timeframe closes and map it onto the lower-timeframe bars WITHOUT look-ahead.
    Bars are labelled by their OPEN time. Steps: resample close with label='left', closed='left', offset=offset,
    .last(), drop empty bins; apply fn to the HTF close array; SHIFT BY ONE HTF BAR (a bar is known only after it
    closes); forward-fill onto ltf.index. Returns a Series aligned with ltf.index."""
    # >>> SOLUTION
    htf_close = ltf["close"].resample(rule, label="left", closed="left", offset=offset).last().dropna()
    htf_val = pd.Series(fn(htf_close.to_numpy()), index=htf_close.index)
    return htf_val.shift(1).reindex(ltf.index, method="ffill")
    # <<< SOLUTION
