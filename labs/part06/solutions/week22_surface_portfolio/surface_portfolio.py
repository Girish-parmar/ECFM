"""Week 22 (S7–S8) — Volatility surface (SVI, arbitrage checks, SABR) and portfolio Greeks, scenarios, delta hedging.

Log-moneyness k = ln(K/F), total implied variance w = σ²T. Dollar Greeks include the contract multiplier
(100 for US equity options, 50 for ES futures and ES futures options). Uses your week 21 pricing module.
Fill in every block marked "Your turn", then run:  python -m pytest week22_surface_portfolio
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from _loader import load

pr = load("week21_pricing_iv", "pricing")


# ---------------------------------------------------------------------------- S7 SVI
def svi_total_var(k, a, b, rho, m, s):
    """Raw SVI: w(k) = a + b[ρ(k − m) + √((k − m)² + s²)]."""
    # >>> SOLUTION
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))
    # <<< SOLUTION


def fit_svi(k, iv, T, weights=None) -> np.ndarray:
    """Fit raw SVI to one expiry's smile by bounded least squares on total variance (residual weights × (w_fit − w)),
    x0 = [min w, 0.1, −0.5, 0, 0.1], bounds a∈[−1,1], b∈[1e-6,5], ρ∈[−0.999,0.999], m∈[−1,1], s∈[1e-4,2].
    Raise ValueError if the fit gives negative minimum variance: a + b·s·√(1−ρ²) < 0. Return [a, b, ρ, m, s]."""
    # >>> SOLUTION
    w = np.asarray(iv) ** 2 * T
    wt = np.ones_like(w) if weights is None else np.asarray(weights)
    res = least_squares(lambda p: wt * (svi_total_var(k, *p) - w), x0=[w.min(), 0.1, -0.5, 0.0, 0.1],
                        bounds=([-1.0, 1e-6, -0.999, -1.0, 1e-4], [1.0, 5.0, 0.999, 1.0, 2.0]),
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    a, b, rho, m, s = res.x
    if a + b * s * np.sqrt(1 - rho ** 2) < 0:
        raise ValueError("SVI fit gives negative total variance")
    return res.x
    # <<< SOLUTION


def butterfly_g(k, a, b, rho, m, s):
    """Durrleman's density condition for a smile w(k); g(k) >= 0 everywhere means no butterfly arbitrage:
    g = (1 − k w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2, with w' and w'' the analytic SVI derivatives
    w' = b[ρ + (k−m)/√((k−m)²+s²)],  w'' = b s² / ((k−m)²+s²)^{3/2}."""
    # >>> SOLUTION
    x = k - m
    root = np.sqrt(x ** 2 + s ** 2)
    w = a + b * (rho * x + root)
    w1 = b * (rho + x / root)
    w2 = b * s ** 2 / root ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2
    # <<< SOLUTION


def calendar_violations(k_grid, params_by_T: dict[float, np.ndarray], tol: float = 1e-12) -> list[tuple]:
    """No calendar arbitrage: at every k, total variance must not DECREASE as T grows. For consecutive expiries
    T1 < T2 return (T1, T2, k) for every grid k with w(k; T2) < w(k; T1) − tol."""
    # >>> SOLUTION
    Ts = sorted(params_by_T)
    out = []
    for t1, t2 in zip(Ts, Ts[1:]):
        w1, w2 = svi_total_var(k_grid, *params_by_T[t1]), svi_total_var(k_grid, *params_by_T[t2])
        out += [(t1, t2, float(k)) for k in np.asarray(k_grid)[w2 < w1 - tol]]
    return out
    # <<< SOLUTION


def interp_iv(k: float, T: float, params_by_T: dict[float, np.ndarray]) -> float:
    """Implied vol at (k, T) by interpolating TOTAL VARIANCE linearly in T between the two fitted expiries around T
    (flat extrapolation of σ outside the fitted range): σ = √(w/T)."""
    # >>> SOLUTION
    Ts = sorted(params_by_T)
    if T <= Ts[0]:
        return float(np.sqrt(svi_total_var(k, *params_by_T[Ts[0]]) / Ts[0]))
    if T >= Ts[-1]:
        return float(np.sqrt(svi_total_var(k, *params_by_T[Ts[-1]]) / Ts[-1]))
    i = int(np.searchsorted(Ts, T))
    t1, t2 = Ts[i - 1], Ts[i]
    w1, w2 = svi_total_var(k, *params_by_T[t1]), svi_total_var(k, *params_by_T[t2])
    w = w1 + (w2 - w1) * (T - t1) / (t2 - t1)
    return float(np.sqrt(w / T))
    # <<< SOLUTION


def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) lognormal SABR implied vol. With z = (ν/α)(FK)^{(1−β)/2} ln(F/K),
    x(z) = ln[(√(1−2ρz+z²) + z − ρ)/(1−ρ)]:
    σ = α / {(FK)^{(1−β)/2} [1 + (1−β)²/24 ln²(F/K) + (1−β)⁴/1920 ln⁴(F/K)]} · (z/x(z))
        · {1 + [(1−β)²α²/(24(FK)^{1−β}) + ρβνα/(4(FK)^{(1−β)/2}) + (2−3ρ²)ν²/24] T}.
    At the money (|ln(F/K)| < 1e-12) use z/x(z) = 1."""
    # >>> SOLUTION
    F, K = np.asarray(F, dtype=float), np.asarray(K, dtype=float)
    lfk = np.log(F / K)
    fk = (F * K) ** ((1 - beta) / 2)
    z = nu / alpha * fk * lfk
    xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    ratio = np.where(np.abs(lfk) < 1e-12, 1.0, z / np.where(xz == 0, 1.0, xz))
    denom = fk * (1 + (1 - beta) ** 2 / 24 * lfk ** 2 + (1 - beta) ** 4 / 1920 * lfk ** 4)
    corr = 1 + ((1 - beta) ** 2 * alpha ** 2 / (24 * fk ** 2) + rho * beta * nu * alpha / (4 * fk)
                + (2 - 3 * rho ** 2) * nu ** 2 / 24) * T
    return alpha / denom * ratio * corr
    # <<< SOLUTION


# ------------------------------------------------------------------------ S8 portfolio
@dataclass
class Position:
    """kind: 'option' (BSM on a stock/ETF), 'fop' (Black-76 on a future), 'future', 'stock'.
    For options: K, T, sigma, cp; `underlying` is the spot or futures price. beta: to SPY, for beta-weighting."""
    symbol: str
    kind: str
    qty: float
    underlying: float
    multiplier: float = 1.0
    K: float = 0.0
    T: float = 0.0
    sigma: float = 0.0
    cp: int = 1
    r: float = 0.04
    q: float = 0.0
    beta: float = 1.0

    def value(self, u: float | None = None, sigma: float | None = None) -> float:
        """Market value (qty × multiplier × unit price) at underlying u and vol sigma (given).
        Futures have zero value (P&L only); use the change in value for scenarios."""
        u = self.underlying if u is None else u
        v = self.sigma if sigma is None else sigma
        if self.kind == "option":
            unit = pr.bsm_price(u, self.K, self.T, self.r, self.q, v, self.cp)
        elif self.kind == "fop":
            unit = pr.black76_price(u, self.K, self.T, self.r, v, self.cp)
        elif self.kind in ("stock", "future"):
            unit = u
        else:
            raise ValueError(self.kind)
        return float(self.qty * self.multiplier * unit)


def dollar_greeks(p: Position) -> dict[str, float]:
    """Dollar Greeks of one position (× qty × multiplier):
      dollar_delta   = Δ·U           (U = underlying; Δ = 1 for stock/future; Black-76 Δ uses S=F, q=r)
      dollar_gamma   = Γ·U²·0.01     (change in dollar delta for a 1% move)
      vega           = vega/100      ($ per vol point)          theta = theta/365 ($ per calendar day)
    Stock and futures have zero gamma, vega and theta."""
    # >>> SOLUTION
    m = p.qty * p.multiplier
    if p.kind in ("stock", "future"):
        return {"dollar_delta": m * p.underlying, "dollar_gamma": 0.0, "vega": 0.0, "theta": 0.0}
    q = p.r if p.kind == "fop" else p.q
    g = pr.greeks(p.underlying, p.K, p.T, p.r, q, p.sigma, p.cp)
    return {"dollar_delta": float(m * g["delta"] * p.underlying),
            "dollar_gamma": float(m * g["gamma"] * p.underlying ** 2 * 0.01),
            "vega": float(m * g["vega"] / 100), "theta": float(m * g["theta"] / 365)}
    # <<< SOLUTION


def book_report(positions: list[Position], spy_price: float) -> pd.DataFrame:
    """One row per position (index = symbol) with dollar_delta, dollar_gamma, vega, theta and beta_dollar_delta
    (= dollar_delta × beta), plus a 'TOTAL' row, and a column spy_shares = beta_dollar_delta / spy_price
    (the beta-weighted delta expressed in SPY shares)."""
    # >>> SOLUTION
    rows = {}
    for p in positions:
        g = dollar_greeks(p)
        g["beta_dollar_delta"] = g["dollar_delta"] * p.beta
        rows[p.symbol] = g
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.loc["TOTAL"] = df.sum()
    df["spy_shares"] = df["beta_dollar_delta"] / spy_price
    return df
    # <<< SOLUTION


def scenario_grid(positions: list[Position], spot_shocks, vol_shocks) -> pd.DataFrame:
    """FULL REVALUATION P&L of the book: rows = relative underlying shocks (e.g. −0.10), columns = absolute vol shocks
    (e.g. +0.05); each position's underlying moves by (1 + shock × beta) and its sigma by + vol shock (floored at 1e-4)."""
    # >>> SOLUTION
    base = sum(p.value() for p in positions)
    grid = pd.DataFrame(index=list(spot_shocks), columns=list(vol_shocks), dtype=float)
    for ds in spot_shocks:
        for dv in vol_shocks:
            grid.loc[ds, dv] = sum(p.value(p.underlying * (1 + ds * p.beta), max(p.sigma + dv, 1e-4))
                                   for p in positions) - base
    return grid
    # <<< SOLUTION


def taylor_pnl(positions: list[Position], spot_shock: float, vol_shock: float) -> float:
    """Delta–gamma–vega approximation of the same scenario, from dollar Greeks: for each position with move
    x = spot_shock × beta (a fraction): dollar_delta·x + ½·dollar_gamma·100·x² + vega·(vol_shock × 100)."""
    # >>> SOLUTION
    tot = 0.0
    for p in positions:
        g = dollar_greeks(p)
        x = spot_shock * p.beta
        tot += g["dollar_delta"] * x + 0.5 * g["dollar_gamma"] * 100 * x ** 2 + g["vega"] * vol_shock * 100
    return tot
    # <<< SOLUTION


def delta_hedge_pnl(S0=100., K=100., T=30 / 365, r=0.04, q=0.0, iv=0.20, rv=0.20, rebalances=30, n_paths=5000,
                    seed=0, cost=0.0) -> np.ndarray:
    """Sell 1 call at implied vol `iv` and delta-hedge `rebalances` times while the stock follows GBM with realized
    vol `rv`. Cash starts at premium − Δ₀S₀; each step: stock moves, cash earns e^{r dt} and the shares earn
    dividends Δ·S·(e^{q dt} − 1), then re-hedge to the new BSM delta (at expiry: 1 if S > K else 0).
    Trading costs: every share bought or sold (including the initial hedge) costs cost × S per share.
    Return the final P&L per path: cash + Δ·S − max(S − K, 0)."""
    # >>> SOLUTION
    rng = np.random.default_rng(seed)
    dt = T / rebalances
    S = np.full(n_paths, float(S0))
    delta = np.full(n_paths, pr.greeks(S0, K, T, r, q, iv, 1)["delta"])
    cash = pr.bsm_price(S0, K, T, r, q, iv, 1) - delta * S - cost * np.abs(delta) * S
    for i in range(1, rebalances + 1):
        S = S * np.exp((r - q - 0.5 * rv ** 2) * dt + rv * np.sqrt(dt) * rng.standard_normal(n_paths))
        cash = cash * np.exp(r * dt) + delta * S * (np.exp(q * dt) - 1)
        tau = T - i * dt
        new = pr.greeks(S, K, tau, r, q, iv, 1)["delta"] if tau > 1e-12 else (S > K).astype(float)
        cash -= (new - delta) * S + cost * np.abs(new - delta) * S
        delta = new
    return cash + delta * S - np.maximum(S - K, 0.0)
    # <<< SOLUTION
