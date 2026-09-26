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
    raise NotImplementedError("✍️ Your turn: see the docstring")


def fit_svi(k, iv, T, weights=None) -> np.ndarray:
    """Fit raw SVI to one expiry's smile by bounded least squares on total variance (residual weights × (w_fit − w)),
    x0 = [min w, 0.1, −0.5, 0, 0.1], bounds a∈[−1,1], b∈[1e-6,5], ρ∈[−0.999,0.999], m∈[−1,1], s∈[1e-4,2].
    Raise ValueError if the fit gives negative minimum variance: a + b·s·√(1−ρ²) < 0. Return [a, b, ρ, m, s]."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def butterfly_g(k, a, b, rho, m, s):
    """Durrleman's density condition for a smile w(k); g(k) >= 0 everywhere means no butterfly arbitrage:
    g = (1 − k w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2, with w' and w'' the analytic SVI derivatives
    w' = b[ρ + (k−m)/√((k−m)²+s²)],  w'' = b s² / ((k−m)²+s²)^{3/2}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def calendar_violations(k_grid, params_by_T: dict[float, np.ndarray], tol: float = 1e-12) -> list[tuple]:
    """No calendar arbitrage: at every k, total variance must not DECREASE as T grows. For consecutive expiries
    T1 < T2 return (T1, T2, k) for every grid k with w(k; T2) < w(k; T1) − tol."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def interp_iv(k: float, T: float, params_by_T: dict[float, np.ndarray]) -> float:
    """Implied vol at (k, T) by interpolating TOTAL VARIANCE linearly in T between the two fitted expiries around T
    (flat extrapolation of σ outside the fitted range): σ = √(w/T)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) lognormal SABR implied vol. With z = (ν/α)(FK)^{(1−β)/2} ln(F/K),
    x(z) = ln[(√(1−2ρz+z²) + z − ρ)/(1−ρ)]:
    σ = α / {(FK)^{(1−β)/2} [1 + (1−β)²/24 ln²(F/K) + (1−β)⁴/1920 ln⁴(F/K)]} · (z/x(z))
        · {1 + [(1−β)²α²/(24(FK)^{1−β}) + ρβνα/(4(FK)^{(1−β)/2}) + (2−3ρ²)ν²/24] T}.
    At the money (|ln(F/K)| < 1e-12) use z/x(z) = 1."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


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
    raise NotImplementedError("✍️ Your turn: see the docstring")


def book_report(positions: list[Position], spy_price: float) -> pd.DataFrame:
    """One row per position (index = symbol) with dollar_delta, dollar_gamma, vega, theta and beta_dollar_delta
    (= dollar_delta × beta), plus a 'TOTAL' row, and a column spy_shares = beta_dollar_delta / spy_price
    (the beta-weighted delta expressed in SPY shares)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def scenario_grid(positions: list[Position], spot_shocks, vol_shocks) -> pd.DataFrame:
    """FULL REVALUATION P&L of the book: rows = relative underlying shocks (e.g. −0.10), columns = absolute vol shocks
    (e.g. +0.05); each position's underlying moves by (1 + shock × beta) and its sigma by + vol shock (floored at 1e-4)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def taylor_pnl(positions: list[Position], spot_shock: float, vol_shock: float) -> float:
    """Delta–gamma–vega approximation of the same scenario, from dollar Greeks: for each position with move
    x = spot_shock × beta (a fraction): dollar_delta·x + ½·dollar_gamma·100·x² + vega·(vol_shock × 100)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def delta_hedge_pnl(S0=100., K=100., T=30 / 365, r=0.04, q=0.0, iv=0.20, rv=0.20, rebalances=30, n_paths=5000,
                    seed=0, cost=0.0) -> np.ndarray:
    """Sell 1 call at implied vol `iv` and delta-hedge `rebalances` times while the stock follows GBM with realized
    vol `rv`. Cash starts at premium − Δ₀S₀; each step: stock moves, cash earns e^{r dt} and the shares earn
    dividends Δ·S·(e^{q dt} − 1), then re-hedge to the new BSM delta (at expiry: 1 if S > K else 0).
    Trading costs: every share bought or sold (including the initial hedge) costs cost × S per share.
    Return the final P&L per path: cash + Δ·S − max(S − K, 0)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
