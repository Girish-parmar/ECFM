"""Shared, complete helpers for the Part 7 labs (nothing to fill in here).

Indicators and option pricing are the reference versions of what you built in Parts 5 and 6, so the Part 7 labs do not
depend on your earlier code. `regime_market` builds bars whose TRUE regime is known, for the regime map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
DATA, CONFIGS = ROOT / "data", ROOT / "configs"


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


# --------------------------------------------------------------------------- pricing (Part 6)
def bsm_price(S, K, T, r, q, sigma, cp):
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return cp * (S * np.exp(-q * T) * norm.cdf(cp * d1) - K * np.exp(-r * T) * norm.cdf(cp * d2))


def bsm_greeks(S, K, T, r, q, sigma, cp):
    """Raw units: vega per 1.00 σ, theta per year."""
    sqT, dq, dr = np.sqrt(T), np.exp(-q * T), np.exp(-r * T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    pdf = norm.pdf(d1)
    return {"delta": cp * dq * norm.cdf(cp * d1), "gamma": dq * pdf / (S * sigma * sqT), "vega": S * dq * pdf * sqT,
            "theta": -S * dq * pdf * sigma / (2 * sqT) - cp * r * K * dr * norm.cdf(cp * d2)
            + cp * q * S * dq * norm.cdf(cp * d1)}


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


def implied_vol(price, S, K, T, r, q, cp) -> float:
    """Reference IV (Part 6): NaN outside the no-arbitrage bounds, else Brent on [1e-4, 5]."""
    from scipy.optimize import brentq
    lower = max(cp * (S * np.exp(-q * T) - K * np.exp(-r * T)), 0.0)
    upper = S * np.exp(-q * T) if cp == 1 else K * np.exp(-r * T)
    if not lower < price < upper:
        return np.nan
    return float(brentq(lambda s: bsm_price(S, K, T, r, q, s, cp) - price, 1e-4, 5.0, xtol=1e-12))


def load_chain() -> tuple[pd.DataFrame, dict]:
    """Recorded synthetic SPY chain (quotes only) and its metadata: spot, asof, r, q."""
    import json
    chain = pd.read_csv(DATA / "spy_chain.csv", parse_dates=["expiry"])
    chain["expiry"] = chain["expiry"].dt.date
    return chain, json.loads((DATA / "spy_chain_meta.json").read_text())
