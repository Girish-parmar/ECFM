"""Week 30 (S21–S24) — Combining strategies and portfolio construction.

Diversification comes from LOW CORRELATION, not from the number of strategies; naive mean–variance maximizes
estimation error; judge every allocator OUT OF SAMPLE.
Fill in every block marked "Your turn", then run:  python -m pytest week30_portfolio
"""
from __future__ import annotations

from collections.abc import Callable

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def ledoit_wolf(returns) -> np.ndarray:
    """Ledoit–Wolf shrunk covariance (given; scikit-learn)."""
    return LedoitWolf().fit(np.asarray(returns)).covariance_


# ------------------------------------------------------------------------ S21 simple allocators
def inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    """w_i ∝ 1/σ_i (sample std, ddof=1), summing to 1."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def diversification_ratio(w, cov) -> float:
    """(Σ w_i σ_i) / √(w'Σw): 1 for a single asset or perfectly correlated assets, higher is more diversified."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ---------------------------------------------------------------------- S22 mean–variance
def min_variance(cov) -> np.ndarray:
    """Global minimum-variance weights Σ⁻¹1 / (1'Σ⁻¹1) (shorts allowed; use np.linalg.solve)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def min_variance_long_only(cov, max_weight: float = 1.0) -> np.ndarray:
    """cvxpy: minimize w'Σw subject to Σw = 1, 0 <= w <= max_weight (use cp.quad_form with cp.psd_wrap(Σ))."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def max_sharpe(mu, cov) -> np.ndarray:
    """Tangency portfolio Σ⁻¹μ normalized to sum to 1 (excess returns; shorts allowed)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------- S23 risk-based allocators
def risk_contributions(w, cov) -> np.ndarray:
    """Share of portfolio variance from each asset: w_i(Σw)_i / (w'Σw) (sums to 1)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def risk_parity(cov) -> np.ndarray:
    """Equal risk contribution, long-only, via the convex log-barrier form (Spinu 2013): minimize
    ½y'Cy − (1/n)Σ ln y (C = Σ / mean(diag Σ) for numerical stability) with L-BFGS-B, bounds y > 1e-9,
    analytic gradient Cy − 1/(n·y); return y / Σy."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def hrp(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (López de Prado 2016), exactly as in the lesson plan (S23):
    distance √(½(1 − ρ)) → single-linkage → leaf order; recursive bisection: split each cluster in two halves of the
    ORDER, cluster variance with inverse-variance weights inside the cluster, α = 1 − v_a/(v_a + v_b) to the left half.
    Return weights indexed like returns.columns."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def black_litterman(cov, w_mkt, delta: float = 2.5, tau: float = 0.05, P=None, Q=None, omega=None) -> np.ndarray:
    """Posterior expected returns. Equilibrium π = δΣw_mkt. Without views return π. With views (P: k × n, Q: k,
    omega: k × k, default diag(P τΣ P')):  μ = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def min_cvar(returns, alpha: float = 0.95, max_weight: float = 1.0) -> np.ndarray:
    """Rockafellar–Uryasev linear program for long-only minimum CVaR of historical scenarios R (T × n):
    minimize ζ + 1/((1−α)T)·Σ u_t  s.t. u_t >= −R_t·w − ζ, u >= 0, Σw = 1, 0 <= w <= max_weight."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ---------------------------------------------------------------- S24 walk-forward allocation
def walk_forward_allocation(returns: pd.DataFrame, allocator: Callable[[pd.DataFrame], np.ndarray], train: int = 252,
                            rebalance: int = 21) -> dict:
    """Every `rebalance` days from day `train` on, fit weights = allocator(the previous `train` days of returns) and
    hold them (no drift, for simplicity) until the next rebalance. Return {"returns": OOS portfolio return Series
    (from day train on), "turnover": mean Σ|Δw| per rebalance (the first allocation counts from zero weights),
    "weights": DataFrame of weights per rebalance date}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
