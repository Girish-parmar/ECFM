"""Shared, complete helpers for the Part 8 labs (nothing to fill in here).

Reference indicators (Part 5), reference strategies and the first-look evaluator (Part 7), and synthetic data generators,
so the Part 8 labs do not depend on your earlier code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------------ indicators (Part 5)
def sma(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.size >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rsi(close, n=14):
    close = np.asarray(close, dtype=float)
    out = np.full(close.shape, np.nan)
    if close.size <= n:
        return out
    d = np.diff(close)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[:n].mean(), loss[:n].mean()
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(n + 1, close.size):
        ag, al = (ag * (n - 1) + gain[i - 1]) / n, (al * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def atr(high, low, close, n=14):
    h, l, c = (np.asarray(a, dtype=float) for a in (high, low, close))  # noqa: E741
    tr = np.full(h.shape, np.nan)
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    out = np.full(h.shape, np.nan)
    if h.size > n:
        out[n] = tr[1:n + 1].mean()
        for i in range(n + 1, h.size):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def adx(high, low, close, n=14):
    h, l, c = (np.asarray(a, dtype=float) for a in (high, low, close))  # noqa: E741
    N = c.size
    up, dn = h[1:] - h[:-1], l[:-1] - l[1:]
    pdm, mdm = np.where((up > dn) & (up > 0), up, 0.0), np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    dx, out = np.full(N, np.nan), np.full(N, np.nan)
    if N <= 2 * n:
        return out
    sp, sm, st = pdm[:n - 1].sum(), mdm[:n - 1].sum(), tr[:n - 1].sum()
    for j in range(n - 1, N - 1):
        sp, sm, st = sp - sp / n + pdm[j], sm - sm / n + mdm[j], st - st / n + tr[j]
        p, m = 100 * sp / st, 100 * sm / st
        dx[j + 1] = 100 * abs(p - m) / (p + m) if p + m else 0.0
    out[2 * n - 1] = dx[n:2 * n].mean()
    for i in range(2 * n, N):
        out[i] = (out[i - 1] * (n - 1) + dx[i]) / n
    return out


def bbands(close, n=20, k=2.0):
    close = np.asarray(close, dtype=float)
    mid = sma(close, n)
    sd = np.full(close.shape, np.nan)
    if close.size >= n:
        sd[n - 1:] = np.lib.stride_tricks.sliding_window_view(close, n).std(axis=1)
    return mid + k * sd, mid, mid - k * sd


# ------------------------------------------------------------------------------- data
def regime_market(n_blocks: int = 24, block: int = 125, seed: int = 0) -> pd.DataFrame:
    """Daily bars built from blocks with a KNOWN regime: `trend` (1 = trending up/down, 0 = range-bound, mean
    reverting) and `high_vol` (1 = 28% annual vol, 0 = 10%). Columns open, high, low, close, volume, trend, high_vol."""
    rng = np.random.default_rng(seed)
    closes, opens, trend_lab, vol_lab = [], [], [], []
    px = 100.0
    for b in range(n_blocks):
        trend = b % 2 == 0 if rng.random() < 0.8 else b % 2 == 1
        high = rng.random() < 0.5
        sig = (0.28 if high else 0.10) / np.sqrt(252)
        drift = rng.choice([-1, 1]) * 0.15 * sig if trend else 0.0          # ~2.4 Sharpe/yr inside a trend block
        anchor = px
        for _ in range(block):
            o = px * np.exp(rng.normal(0, 0.25 * sig))
            if trend:
                lr = drift + rng.normal(0, sig)
            else:
                lr = -0.08 * np.log(o / anchor) + rng.normal(0, sig)             # pull back toward the block's start
            px = o * np.exp(lr)
            opens.append(o)
            closes.append(px)
            trend_lab.append(int(trend))
            vol_lab.append(int(high))
    o, c = np.array(opens), np.array(closes)
    span = np.abs(np.log(c / o)) + rng.gamma(2.0, 0.003, c.size)
    h = np.maximum(o, c) * np.exp(rng.uniform(0.1, 0.6, c.size) * span)
    l = np.minimum(o, c) * np.exp(-rng.uniform(0.1, 0.6, c.size) * span)  # noqa: E741
    idx = pd.bdate_range("2012-01-02", periods=c.size, tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1e6, "trend": trend_lab,
                         "high_vol": vol_lab}, index=idx)


# --------------------------------------------------------------------------- strategies (Part 7)
def tsmom_signal(close, lookback=120, vol_n=20, target_vol=0.10, max_lev=2.0):
    """Reference time-series momentum weights (decided at each close; NaN -> 0)."""
    close = np.asarray(close, dtype=float)
    past = np.full(close.shape, np.nan)
    past[lookback:] = close[lookback:] / close[:-lookback] - 1
    r = np.full(close.shape, np.nan)
    r[1:] = np.diff(np.log(close))
    vol = pd.Series(r).rolling(vol_n).std().to_numpy() * np.sqrt(252)
    return np.nan_to_num(np.clip(np.sign(past) * target_vol / vol, -max_lev, max_lev))


def sma_cross_signal(close, fast=20, slow=100):
    """+1 when SMA(fast) > SMA(slow), -1 when below, 0 in warm-up."""
    f, s = sma(close, fast), sma(close, slow)
    return np.where(np.isnan(s), 0.0, np.where(f > s, 1.0, -1.0))


def rsi2_signal(close, entry=10, exit_=70):
    """Long-only RSI(2) reversion (no trend filter)."""
    r = rsi(close, 2)
    pos = np.zeros(len(close))
    for t in range(1, len(close)):
        pos[t] = (1.0 if r[t] < entry else 0.0) if pos[t - 1] == 0 else (0.0 if r[t] > exit_ else 1.0)
    return pos


def quick_eval_pnl(decided, open_, cost_bps=2.0):
    """Reference Part 7 first-look P&L: decided[t] filled at open[t+1], held to the next open."""
    decided = np.nan_to_num(np.asarray(decided, dtype=float))
    open_ = np.asarray(open_, dtype=float)
    pos = np.zeros_like(decided)
    pos[1:] = decided[:-1]
    ret = np.zeros_like(open_)
    ret[:-1] = open_[1:] / open_[:-1] - 1
    return pos * ret - np.abs(np.diff(pos, prepend=0.0)) * cost_bps / 1e4


def strategy_returns(n_days=2520, sharpes=(0.8, 0.6, 0.5, 0.4, 0.3), vols=(0.10, 0.15, 0.08, 0.20, 0.12),
                     corr=0.2, seed=0, fat_tails=True) -> pd.DataFrame:
    """Daily returns of synthetic strategies with given annual Sharpes, vols and a common pairwise correlation
    (Student-t shocks if fat_tails). Columns S1..Sn."""
    rng = np.random.default_rng(seed)
    n = len(sharpes)
    C = np.full((n, n), corr) + (1 - corr) * np.eye(n)
    L = np.linalg.cholesky(C)
    z = rng.standard_t(5, size=(n_days, n)) / np.sqrt(5 / 3) if fat_tails else rng.standard_normal((n_days, n))
    z = z @ L.T
    vols, sh = np.asarray(vols), np.asarray(sharpes)
    daily = z * vols / np.sqrt(252) + sh * vols / 252
    return pd.DataFrame(daily, index=pd.bdate_range("2015-01-02", periods=n_days),
                        columns=[f"S{i + 1}" for i in range(n)])
