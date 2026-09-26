"""Week 23 (S3–S4) — Mean-reversion, range-bound, either-way, volatility, mathematical and statistical strategies.

Every function returns positions DECIDED AT THE CLOSE of each bar (filled next bar by quick_eval), computed only from
data up to that bar. Indicators come from common.py (reference versions of your Part 5 code).
Fill in every block marked "Your turn", then run:  python -m pytest week23_linear_groups
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import adx, bbands, rsi, sma


# -------------------------------------------------------------------- S3 mean reversion
def rsi2_reversion(close, entry: float = 10, exit_: float = 70, trend_n: int = 200, use_trend: bool = True,
                   max_hold: int | None = None) -> np.ndarray:
    """Connors-style long-only: flat → long when RSI(2) < entry (and close > SMA(trend_n) if use_trend);
    long → flat when RSI(2) > exit_ or after max_hold bars in the trade (if given). 0/1 positions."""
    # >>> SOLUTION
    close = np.asarray(close, dtype=float)
    r, m = rsi(close, 2), sma(close, trend_n)
    pos, held = np.zeros(close.size), 0
    for t in range(1, close.size):
        if pos[t - 1] == 0:
            ok = r[t] < entry and (not use_trend or close[t] > m[t])
            pos[t], held = (1.0, 1) if ok else (0.0, 0)
        else:
            held += 1
            done = r[t] > exit_ or (max_hold is not None and held > max_hold)
            pos[t] = 0.0 if done else 1.0
    return pos
    # <<< SOLUTION


def ibs(high, low, close) -> np.ndarray:
    """Internal bar strength (C − L)/(H − L) in [0, 1]; 0.5 on a bar with H == L."""
    # >>> SOLUTION
    h, l, c = (np.asarray(a, dtype=float) for a in (high, low, close))  # noqa: E741
    rng = h - l
    return np.divide(c - l, rng, out=np.full(c.shape, 0.5), where=rng > 0)
    # <<< SOLUTION


def zscore_reversion(close, n: int = 20, entry: float = 2.0, exit_: float = 0.5, max_hold: int = 10) -> np.ndarray:
    """z = (C − SMA(n)) / rolling std (n, ddof=0). Flat → +1 if z < −entry, −1 if z > entry. In a trade → flat when
    |z| < exit_ or after max_hold bars in the trade (a time stop: mean reversion that does not revert is a trend)."""
    # >>> SOLUTION
    c = np.asarray(close, dtype=float)
    m = sma(c, n)
    sd = np.full(c.shape, np.nan)
    if c.size >= n:
        sd[n - 1:] = np.lib.stride_tricks.sliding_window_view(c, n).std(axis=1)
    z = (c - m) / sd
    pos, held = np.zeros(c.size), 0
    for t in range(1, c.size):
        if not np.isfinite(z[t]):
            continue
        if pos[t - 1] == 0:
            pos[t] = 1.0 if z[t] < -entry else (-1.0 if z[t] > entry else 0.0)
            held = 1 if pos[t] else 0
        else:
            held += 1
            pos[t] = 0.0 if (abs(z[t]) < exit_ or held > max_hold) else pos[t - 1]
    return pos
    # <<< SOLUTION


# ---------------------------------------------------------------------- S3 range-bound
def bollinger_fade(high, low, close, n: int = 20, k: float = 2.0, adx_n: int = 14, adx_max: float = 20) -> np.ndarray:
    """Fade the bands ONLY in a range (ADX(adx_n) < adx_max): flat → +1 if close < lower, −1 if close > upper.
    Exit when the close crosses the middle band (long: close >= middle; short: close <= middle) or when ADX rises to
    adx_max or more (the range turned into a trend)."""
    # >>> SOLUTION
    c = np.asarray(close, dtype=float)
    up, mid, lo = bbands(c, n, k)
    a = adx(high, low, c, adx_n)
    pos = np.zeros(c.size)
    for t in range(1, c.size):
        if not (np.isfinite(a[t]) and np.isfinite(mid[t])):
            continue
        prev = pos[t - 1]
        ranging = a[t] < adx_max
        if prev == 0 and ranging:
            pos[t] = 1.0 if c[t] < lo[t] else (-1.0 if c[t] > up[t] else 0.0)
        elif prev == 1:
            pos[t] = 0.0 if (c[t] >= mid[t] or not ranging) else 1.0
        elif prev == -1:
            pos[t] = 0.0 if (c[t] <= mid[t] or not ranging) else -1.0
    return pos
    # <<< SOLUTION


# ------------------------------------------------------------------------ S4 either-way
def orb_trade(session: pd.DataFrame, minutes: int = 30) -> dict | None:
    """Opening-range breakout on ONE session of intraday bars (index = bar open time; columns open/high/low/close).
    Range = high/low of bars that OPEN in the first `minutes`. The first later bar that CLOSES outside the range is the
    signal; enter at the NEXT bar's open (direction +1 above, −1 below), stop at the opposite side of the range.
    Walk the following bars: if a bar's low (long) / high (short) touches the stop, exit at the stop; otherwise exit at
    the last bar's close. Return {"direction", "entry", "exit", "stop", "ret" (direction × (exit/entry − 1)),
    "stopped" (bool)} or None if no breakout (or no bar left to enter on)."""
    # >>> SOLUTION
    start = session.index[0]
    in_rng = session.index < start + pd.Timedelta(minutes=minutes)
    hi, lo = session.loc[in_rng, "high"].max(), session.loc[in_rng, "low"].min()
    after = session[~in_rng]
    for i in range(len(after) - 1):
        c = after["close"].iloc[i]
        if lo <= c <= hi:
            continue
        d = 1 if c > hi else -1
        stop = lo if d == 1 else hi
        entry = after["open"].iloc[i + 1]
        rest = after.iloc[i + 1:]
        for _, b in rest.iterrows():
            if (d == 1 and b["low"] <= stop) or (d == -1 and b["high"] >= stop):
                return {"direction": d, "entry": entry, "exit": stop, "stop": stop,
                        "ret": d * (stop / entry - 1), "stopped": True}
        ex = rest["close"].iloc[-1]
        return {"direction": d, "entry": entry, "exit": ex, "stop": stop, "ret": d * (ex / entry - 1), "stopped": False}
    return None
    # <<< SOLUTION


# ------------------------------------------------------------------------ S4 volatility
def vol_target_overlay(pos, returns, target: float = 0.10, n: int = 20, max_lev: float = 2.0,
                       periods_per_year: int = 252) -> np.ndarray:
    """Scale any position series to constant risk: scale[t] = target / (rolling std of `returns` over bars t−n+1..t,
    ddof=1, annualized), clipped to max_lev; the scaled position is pos × scale (0 where the vol is not known yet).
    `returns[t]` must be the return REALIZED up to bar t (known at its close)."""
    # >>> SOLUTION
    vol = pd.Series(np.asarray(returns, dtype=float)).rolling(n).std().to_numpy() * np.sqrt(periods_per_year)
    scale = np.clip(np.divide(target, vol, out=np.zeros_like(vol), where=vol > 0), 0, max_lev)
    return np.asarray(pos, dtype=float) * np.nan_to_num(scale)
    # <<< SOLUTION


def vix_regime_overlay(pos, vix, vix3m, reduce: float = 0.5) -> np.ndarray:
    """Multiply the position by `reduce` on days when the VIX term structure is inverted (VIX / VIX3M > 1)."""
    # >>> SOLUTION
    inverted = np.asarray(vix, dtype=float) / np.asarray(vix3m, dtype=float) > 1
    return np.where(inverted, np.asarray(pos, dtype=float) * reduce, pos)
    # <<< SOLUTION


# --------------------------------------------------------------------- S4 mathematical
def kalman_level(price, q: float = 1e-5, r: float = 1e-2) -> np.ndarray:
    """Local-level Kalman filter (x = level, p = its variance, start x = price[0], p = 1). Each bar:
    predict p += q; gain k = p/(p + r); update x += k(z − x); p *= (1 − k). Larger q/r = faster, noisier level."""
    # >>> SOLUTION
    price = np.asarray(price, dtype=float)
    x, p = price[0], 1.0
    level = np.empty_like(price)
    for t, z in enumerate(price):
        p += q
        k = p / (p + r)
        x += k * (z - x)
        p *= 1 - k
        level[t] = x
    return level
    # <<< SOLUTION


def hurst_exponent(x, lags=range(2, 64)) -> float:
    """Hurst exponent from the scaling of lagged differences: std(x[t+lag] − x[t]) ∝ lag^H. Fit the slope of
    log(std) on log(lag). ≈ 0.5 random walk, > 0.5 trending (persistent), < 0.5 mean reverting. Pass log prices."""
    # >>> SOLUTION
    x = np.asarray(x, dtype=float)
    lags = np.asarray(list(lags))
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])
    # <<< SOLUTION


# ----------------------------------------------------------------------- S4 statistical
def pairs_positions(y, x, lookback: int = 60, entry: float = 2.0, exit_: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Rolling pairs trade on log prices y, x. At each bar t >= lookback: OLS hedge ratio β on the PREVIOUS lookback
    bars (t−lookback..t−1: no look-ahead), spread s = y − β·x over the same window plus today; z = (s[t] − mean)/std of
    the window's spread (ddof=0). Position in the spread (+1 = long y, short β·x): flat → +1 if z < −entry, −1 if
    z > entry; in a trade → 0 when |z| < exit_. Return (positions, betas) with β NaN during warm-up."""
    # >>> SOLUTION
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    pos, betas = np.zeros(y.size), np.full(y.size, np.nan)
    for t in range(lookback, y.size):
        ys, xs = y[t - lookback:t], x[t - lookback:t]
        beta, alpha = np.polyfit(xs, ys, 1)
        betas[t] = beta
        spread = ys - beta * xs
        sd = spread.std()
        z = (y[t] - beta * x[t] - spread.mean()) / sd if sd > 0 else 0.0
        prev = pos[t - 1]
        if prev == 0:
            pos[t] = 1.0 if z < -entry else (-1.0 if z > entry else 0.0)
        else:
            pos[t] = 0.0 if abs(z) < exit_ else prev
    return pos, betas
    # <<< SOLUTION


def turn_of_month(index: pd.DatetimeIndex, last_days: int = 1, first_days: int = 3) -> np.ndarray:
    """True on the last `last_days` and the first `first_days` BUSINESS days of each calendar month.
    Use the business-day CALENDAR (pd.offsets.BDay), which is known in advance: t is among the last N days of its
    month if t + N business days falls in a later month, and among the first M days if t − M business days falls in
    an earlier month. Never count rows of the data backwards from the month's end: that needs rows that have not
    happened yet (an incomplete final month would flag its last AVAILABLE day). Holidays are ignored here; use an
    exchange calendar in production."""
    # >>> SOLUTION
    idx = index.tz_localize(None) if index.tz is not None else index
    ahead = idx + pd.offsets.BDay(last_days)
    back = idx - pd.offsets.BDay(first_days)
    return np.asarray((ahead.month != idx.month) | (back.month != idx.month))
    # <<< SOLUTION


def overnight_intraday(open_, close) -> tuple[np.ndarray, np.ndarray]:
    """Split close-to-close returns: overnight[t] = open[t]/close[t−1] − 1, intraday[t] = close[t]/open[t] − 1
    (overnight[0] = NaN). (1 + overnight)(1 + intraday) = close[t]/close[t−1]."""
    # >>> SOLUTION
    o, c = np.asarray(open_, dtype=float), np.asarray(close, dtype=float)
    on = np.full(o.shape, np.nan)
    on[1:] = o[1:] / c[:-1] - 1
    return on, c / o - 1
    # <<< SOLUTION
