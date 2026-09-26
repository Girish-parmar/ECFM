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
    raise NotImplementedError("✍️ Your turn: see the docstring")


def d1d2(S, K, T, r, q, sigma):
    """d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T),  d2 = d1 − σ√T."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def bsm_price(S, K, T, r, q, sigma, cp):
    """BSM price with continuous dividend yield q:  cp·(S e^{−qT} N(cp·d1) − K e^{−rT} N(cp·d2))."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def black76_price(F, K, T, r, sigma, cp):
    """Black-76 for options on futures: BSM with S = F and q = r."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def greeks(S, K, T, r, q, sigma, cp) -> dict[str, np.ndarray]:
    """First-order Greeks in RAW units: delta, gamma, vega (per 1.00 σ), theta (per year, = −∂V/∂T), rho (per 1.00 r).
      delta = cp e^{−qT} N(cp d1)            gamma = e^{−qT} n(d1) / (S σ √T)       vega = S e^{−qT} n(d1) √T
      theta = −S e^{−qT} n(d1) σ / (2√T) − cp r K e^{−rT} N(cp d2) + cp q S e^{−qT} N(cp d1)
      rho   = cp K T e^{−rT} N(cp d2)"""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def to_display(g: dict) -> dict:
    """Convert raw Greeks to display units: vega per vol point (÷100), theta per calendar day (÷365), rho per 1% (÷100).
    delta and gamma are unchanged. Return a NEW dict (do not modify g)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def parity_gap(call, put, S, K, T, r, q):
    """Put–call parity residual C − P − (S e^{−qT} − K e^{−rT}); ≈ 0 for European prices from one model."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def implied_forward(strikes, calls, puts, T, r, n_strikes: int = 3) -> float:
    """Forward implied by parity, robust to unknown (discrete) dividends: for the n_strikes strikes with the smallest
    |C − P| (closest to ATM), F_K = K + e^{rT}(C − P); return the median of those F_K."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def price_bounds(S, K, T, r, q, cp) -> tuple[float, float]:
    """No-arbitrage bounds of a European price: lower = max(cp (S e^{−qT} − K e^{−rT}), 0);
    upper = S e^{−qT} for a call, K e^{−rT} for a put."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def implied_vol(price, S, K, T, r, q, cp, lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-10,
                stats: dict | None = None) -> float:
    """Implied volatility. NaN if price is NOT strictly inside price_bounds. Otherwise Newton–Raphson on σ (start
    σ0 = clip(√(2|ln(S/K) + (r−q)T| / T), 0.05, 2.0), at most 20 steps, stop when |price error| < tol) and fall
    back to brentq on [lo, hi] (xtol=tol) if vega < 1e-8 or σ leaves (lo, hi) or Newton does not converge.
    If `stats` is a dict, set stats['method'] to 'newton' or 'brent' (the tests use it)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def implied_vol_chain(prices, S, K, T, r, q, cp, tol: float = 1e-10) -> np.ndarray:
    """Vectorized IV for a whole chain (arrays broadcast together). Run up to 30 Newton steps on ALL elements at once
    (start σ = 0.3); elements that did not converge (|error| >= tol, vega tiny, or σ outside (1e-4, 5)) are solved
    one by one with implied_vol. Check the no-arbitrage bounds FIRST (vectorized): a price not strictly inside them
    is NaN — otherwise Newton can "find" a σ for a zero price of a far OTM option."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
