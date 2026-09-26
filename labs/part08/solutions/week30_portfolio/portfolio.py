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
    # >>> SOLUTION
    iv = 1 / returns.std(ddof=1)
    return iv / iv.sum()
    # <<< SOLUTION


def diversification_ratio(w, cov) -> float:
    """(Σ w_i σ_i) / √(w'Σw): 1 for a single asset or perfectly correlated assets, higher is more diversified."""
    # >>> SOLUTION
    w, cov = np.asarray(w, dtype=float), np.asarray(cov, dtype=float)
    return float(w @ np.sqrt(np.diag(cov)) / np.sqrt(w @ cov @ w))
    # <<< SOLUTION


# ---------------------------------------------------------------------- S22 mean–variance
def min_variance(cov) -> np.ndarray:
    """Global minimum-variance weights Σ⁻¹1 / (1'Σ⁻¹1) (shorts allowed; use np.linalg.solve)."""
    # >>> SOLUTION
    x = np.linalg.solve(np.asarray(cov, dtype=float), np.ones(len(cov)))
    return x / x.sum()
    # <<< SOLUTION


def min_variance_long_only(cov, max_weight: float = 1.0) -> np.ndarray:
    """cvxpy: minimize w'Σw subject to Σw = 1, 0 <= w <= max_weight (use cp.quad_form with cp.psd_wrap(Σ))."""
    # >>> SOLUTION
    n = len(cov)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(np.asarray(cov)))),
                      [cp.sum(w) == 1, w >= 0, w <= max_weight])
    prob.solve()
    return np.asarray(w.value).ravel()
    # <<< SOLUTION


def max_sharpe(mu, cov) -> np.ndarray:
    """Tangency portfolio Σ⁻¹μ normalized to sum to 1 (excess returns; shorts allowed)."""
    # >>> SOLUTION
    x = np.linalg.solve(np.asarray(cov, dtype=float), np.asarray(mu, dtype=float))
    return x / x.sum()
    # <<< SOLUTION


# --------------------------------------------------------------- S23 risk-based allocators
def risk_contributions(w, cov) -> np.ndarray:
    """Share of portfolio variance from each asset: w_i(Σw)_i / (w'Σw) (sums to 1)."""
    # >>> SOLUTION
    w, cov = np.asarray(w, dtype=float), np.asarray(cov, dtype=float)
    return w * (cov @ w) / (w @ cov @ w)
    # <<< SOLUTION


def risk_parity(cov) -> np.ndarray:
    """Equal risk contribution, long-only, via the convex log-barrier form (Spinu 2013): minimize
    ½y'Cy − (1/n)Σ ln y (C = Σ / mean(diag Σ) for numerical stability) with L-BFGS-B, bounds y > 1e-9,
    analytic gradient Cy − 1/(n·y); return y / Σy."""
    # >>> SOLUTION
    n = len(cov)
    c = np.asarray(cov, dtype=float) / np.mean(np.diag(cov))
    res = minimize(lambda y: 0.5 * y @ c @ y - np.log(y).sum() / n, np.full(n, 1.0),
                   jac=lambda y: c @ y - 1 / (n * y), bounds=[(1e-9, None)] * n, method="L-BFGS-B")
    return res.x / res.x.sum()
    # <<< SOLUTION


def hrp(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (López de Prado 2016), exactly as in the lesson plan (S23):
    distance √(½(1 − ρ)) → single-linkage → leaf order; recursive bisection: split each cluster in two halves of the
    ORDER, cluster variance with inverse-variance weights inside the cluster, α = 1 − v_a/(v_a + v_b) to the left half.
    Return weights indexed like returns.columns."""
    # >>> SOLUTION
    cov, corr = returns.cov(), returns.corr()
    dist = np.sqrt(0.5 * (1 - corr)).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    order = list(corr.index[leaves_list(linkage(squareform(dist, checks=False), "single"))])
    w = pd.Series(1.0, index=order)

    def cluster_var(items):
        c = cov.loc[items, items].to_numpy()
        ivp = 1 / np.diag(c)
        ivp /= ivp.sum()
        return ivp @ c @ ivp

    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for a, b in zip(clusters[::2], clusters[1::2]):
            va, vb = cluster_var(a), cluster_var(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
    return w.reindex(returns.columns)
    # <<< SOLUTION


def black_litterman(cov, w_mkt, delta: float = 2.5, tau: float = 0.05, P=None, Q=None, omega=None) -> np.ndarray:
    """Posterior expected returns. Equilibrium π = δΣw_mkt. Without views return π. With views (P: k × n, Q: k,
    omega: k × k, default diag(P τΣ P')):  μ = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]."""
    # >>> SOLUTION
    S = np.asarray(cov, dtype=float)
    pi = delta * S @ np.asarray(w_mkt, dtype=float)
    if P is None:
        return pi
    P, Q = np.atleast_2d(np.asarray(P, dtype=float)), np.asarray(Q, dtype=float)
    tS = tau * S
    om = np.diag(np.diag(P @ tS @ P.T)) if omega is None else np.asarray(omega, dtype=float)
    A = np.linalg.inv(tS) + P.T @ np.linalg.inv(om) @ P
    b = np.linalg.solve(tS, pi) + P.T @ np.linalg.solve(om, Q)
    return np.linalg.solve(A, b)
    # <<< SOLUTION


def min_cvar(returns, alpha: float = 0.95, max_weight: float = 1.0) -> np.ndarray:
    """Rockafellar–Uryasev linear program for long-only minimum CVaR of historical scenarios R (T × n):
    minimize ζ + 1/((1−α)T)·Σ u_t  s.t. u_t >= −R_t·w − ζ, u >= 0, Σw = 1, 0 <= w <= max_weight."""
    # >>> SOLUTION
    R = np.asarray(returns, dtype=float)
    T, n = R.shape
    w, z, u = cp.Variable(n), cp.Variable(), cp.Variable(T)
    prob = cp.Problem(cp.Minimize(z + cp.sum(u) / ((1 - alpha) * T)),
                      [u >= -R @ w - z, u >= 0, cp.sum(w) == 1, w >= 0, w <= max_weight])
    prob.solve()
    return np.asarray(w.value).ravel()
    # <<< SOLUTION


# ---------------------------------------------------------------- S24 walk-forward allocation
def walk_forward_allocation(returns: pd.DataFrame, allocator: Callable[[pd.DataFrame], np.ndarray], train: int = 252,
                            rebalance: int = 21) -> dict:
    """Every `rebalance` days from day `train` on, fit weights = allocator(the previous `train` days of returns) and
    hold them (no drift, for simplicity) until the next rebalance. Return {"returns": OOS portfolio return Series
    (from day train on), "turnover": mean Σ|Δw| per rebalance (the first allocation counts from zero weights),
    "weights": DataFrame of weights per rebalance date}."""
    # >>> SOLUTION
    X = returns.to_numpy()
    out = np.full(len(returns), np.nan)
    prev, turns, wrows, dates = np.zeros(X.shape[1]), [], [], []
    for start in range(train, len(returns), rebalance):
        w = np.asarray(allocator(returns.iloc[start - train:start]), dtype=float)
        turns.append(np.abs(w - prev).sum())
        prev = w
        end = min(start + rebalance, len(returns))
        out[start:end] = X[start:end] @ w
        wrows.append(w)
        dates.append(returns.index[start])
    return {"returns": pd.Series(out, index=returns.index).iloc[train:], "turnover": float(np.mean(turns)),
            "weights": pd.DataFrame(wrows, index=dates, columns=returns.columns)}
    # <<< SOLUTION
