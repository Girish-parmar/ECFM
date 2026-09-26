"""Week 21 (S3–S4) — Black–Scholes–Merton, Black-76, first-order Greeks, implied volatility.

Unit conventions (lesson plan, Section 4) — the library works in RAW units:
  T in years (ACT/365) · σ decimal · vega per 1.00 of σ · theta per YEAR (−∂V/∂T) · rho per 1.00 of r.
Brokers and py_vollib display vega per vol point (÷100), theta per day (÷365), rho per 1% (÷100): see to_display.
Every function takes cp = +1 (call) or −1 (put) and works element-wise on NumPy arrays.
Golden values from py_vollib are stored in golden/vollib.npz (you do not need py_vollib installed).
Fill in every block marked "Your turn", then run:  python -m pytest week21_pricing_iv
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

N, n = norm.cdf, norm.pdf
SECONDS_PER_YEAR = 365 * 86400


def year_fraction(now: datetime, expiry: datetime) -> float:
    """ACT/365 calendar time in years between two timezone-aware datetimes (expiry usually 16:00 New York)."""
    # >>> SOLUTION
    return (expiry - now).total_seconds() / SECONDS_PER_YEAR
    # <<< SOLUTION


def d1d2(S, K, T, r, q, sigma):
    """d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T),  d2 = d1 − σ√T."""
    # >>> SOLUTION
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    return d1, d1 - sigma * sqT
    # <<< SOLUTION


def bsm_price(S, K, T, r, q, sigma, cp):
    """BSM price with continuous dividend yield q:  cp·(S e^{−qT} N(cp·d1) − K e^{−rT} N(cp·d2))."""
    # >>> SOLUTION
    d1, d2 = d1d2(S, K, T, r, q, sigma)
    return cp * (S * np.exp(-q * T) * N(cp * d1) - K * np.exp(-r * T) * N(cp * d2))
    # <<< SOLUTION


def black76_price(F, K, T, r, sigma, cp):
    """Black-76 for options on futures: BSM with S = F and q = r."""
    # >>> SOLUTION
    return bsm_price(F, K, T, r, r, sigma, cp)
    # <<< SOLUTION


def greeks(S, K, T, r, q, sigma, cp) -> dict[str, np.ndarray]:
    """First-order Greeks in RAW units: delta, gamma, vega (per 1.00 σ), theta (per year, = −∂V/∂T), rho (per 1.00 r).
      delta = cp e^{−qT} N(cp d1)            gamma = e^{−qT} n(d1) / (S σ √T)       vega = S e^{−qT} n(d1) √T
      theta = −S e^{−qT} n(d1) σ / (2√T) − cp r K e^{−rT} N(cp d2) + cp q S e^{−qT} N(cp d1)
      rho   = cp K T e^{−rT} N(cp d2)"""
    # >>> SOLUTION
    d1, d2 = d1d2(S, K, T, r, q, sigma)
    sqT, dq, dr = np.sqrt(T), np.exp(-q * T), np.exp(-r * T)
    pdf = n(d1)
    return {
        "delta": cp * dq * N(cp * d1),
        "gamma": dq * pdf / (S * sigma * sqT),
        "vega": S * dq * pdf * sqT,
        "theta": -S * dq * pdf * sigma / (2 * sqT) - cp * r * K * dr * N(cp * d2) + cp * q * S * dq * N(cp * d1),
        "rho": cp * K * T * dr * N(cp * d2),
    }
    # <<< SOLUTION


def to_display(g: dict) -> dict:
    """Convert raw Greeks to display units: vega per vol point (÷100), theta per calendar day (÷365), rho per 1% (÷100).
    delta and gamma are unchanged. Return a NEW dict (do not modify g)."""
    # >>> SOLUTION
    out = dict(g)
    out["vega"], out["theta"], out["rho"] = g["vega"] / 100, g["theta"] / 365, g["rho"] / 100
    return out
    # <<< SOLUTION


def parity_gap(call, put, S, K, T, r, q):
    """Put–call parity residual C − P − (S e^{−qT} − K e^{−rT}); ≈ 0 for European prices from one model."""
    # >>> SOLUTION
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))
    # <<< SOLUTION


def implied_forward(strikes, calls, puts, T, r, n_strikes: int = 3) -> float:
    """Forward implied by parity, robust to unknown (discrete) dividends: for the n_strikes strikes with the smallest
    |C − P| (closest to ATM), F_K = K + e^{rT}(C − P); return the median of those F_K."""
    # >>> SOLUTION
    K, C, P = (np.asarray(a, dtype=float) for a in (strikes, calls, puts))
    idx = np.argsort(np.abs(C - P))[:n_strikes]
    return float(np.median(K[idx] + np.exp(r * T) * (C[idx] - P[idx])))
    # <<< SOLUTION


def price_bounds(S, K, T, r, q, cp) -> tuple[float, float]:
    """No-arbitrage bounds of a European price: lower = max(cp (S e^{−qT} − K e^{−rT}), 0);
    upper = S e^{−qT} for a call, K e^{−rT} for a put."""
    # >>> SOLUTION
    fs, fk = S * np.exp(-q * T), K * np.exp(-r * T)
    return max(cp * (fs - fk), 0.0), (fs if cp == 1 else fk)
    # <<< SOLUTION


def implied_vol(price, S, K, T, r, q, cp, lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-10,
                stats: dict | None = None) -> float:
    """Implied volatility. NaN if price is NOT strictly inside price_bounds. Otherwise Newton–Raphson on σ (start
    σ0 = clip(√(2|ln(S/K) + (r−q)T| / T), 0.05, 2.0), at most 20 steps, stop when |price error| < tol) and fall
    back to brentq on [lo, hi] (xtol=tol) if vega < 1e-8 or σ leaves (lo, hi) or Newton does not converge.
    If `stats` is a dict, set stats['method'] to 'newton' or 'brent' (the tests use it)."""
    # >>> SOLUTION
    lower, upper = price_bounds(S, K, T, r, q, cp)
    if not lower < price < upper:
        return np.nan
    sigma = min(max(np.sqrt(2 * abs(np.log(S / K) + (r - q) * T) / T), 0.05), 2.0)
    for _ in range(20):
        diff = bsm_price(S, K, T, r, q, sigma, cp) - price
        if abs(diff) < tol:
            if stats is not None:
                stats["method"] = "newton"
            return float(sigma)
        vega = greeks(S, K, T, r, q, sigma, cp)["vega"]
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if not lo < sigma < hi:
            break
    if stats is not None:
        stats["method"] = "brent"
    return float(brentq(lambda s: bsm_price(S, K, T, r, q, s, cp) - price, lo, hi, xtol=tol))
    # <<< SOLUTION


def implied_vol_chain(prices, S, K, T, r, q, cp, tol: float = 1e-10) -> np.ndarray:
    """Vectorized IV for a whole chain (arrays broadcast together). Run up to 30 Newton steps on ALL elements at once
    (start σ = 0.3); elements that did not converge (|error| >= tol, vega tiny, or σ outside (1e-4, 5)) are solved
    one by one with implied_vol. Check the no-arbitrage bounds FIRST (vectorized): a price not strictly inside them
    is NaN — otherwise Newton can "find" a σ for a zero price of a far OTM option."""
    # >>> SOLUTION
    price, S, K, T, r, q, cp = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (prices, S, K, T, r, q, cp)))
    fs, fk = S * np.exp(-q * T), K * np.exp(-r * T)
    lower, upper = np.maximum(cp * (fs - fk), 0.0), np.where(cp == 1, fs, fk)
    valid = (price > lower) & (price < upper)
    sigma = np.full(price.shape, 0.3)
    done = np.zeros(price.shape, dtype=bool)
    for _ in range(30):
        diff = bsm_price(S, K, T, r, q, sigma, cp) - price
        done = np.abs(diff) < tol
        if done.all():
            break
        vega = greeks(S, K, T, r, q, sigma, cp)["vega"]
        step = np.divide(diff, vega, out=np.zeros_like(diff), where=vega > 1e-8)
        sigma = np.where(done, sigma, np.clip(sigma - step, 1e-4, 5.0))
    diff = bsm_price(S, K, T, r, q, sigma, cp) - price
    ok = (np.abs(diff) < tol) & (sigma > 1e-4) & (sigma < 5.0) & valid
    out = np.where(ok, sigma, np.nan)
    for i in np.flatnonzero((~ok & valid).ravel()):
        idx = np.unravel_index(i, price.shape)
        out[idx] = implied_vol(price[idx], S[idx], K[idx], T[idx], r[idx], q[idx], int(cp[idx]), tol=tol)
    return out
    # <<< SOLUTION
