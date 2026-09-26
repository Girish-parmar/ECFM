"""Shared, complete data generators for the Part 9 labs (nothing to fill in here).

Every generator has KNOWN true parameters, so the labs can check that estimators recover them. For graded research,
use a point-in-time universe of real prices (Part 8 S5) instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_ou(n: int, theta: float, mu: float = 0.0, sigma: float = 1.0, x0: float | None = None,
                seed: int = 0, dt: float = 1.0) -> np.ndarray:
    """Exact discretization of dX = θ(μ − X)dt + σdW (half-life = ln 2 / θ)."""
    rng = np.random.default_rng(seed)
    b = np.exp(-theta * dt)
    sd = sigma * np.sqrt((1 - b ** 2) / (2 * theta))
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    for t in range(1, n):
        x[t] = mu + b * (x[t - 1] - mu) + sd * rng.standard_normal()
    return x


def cointegrated_pair(n: int = 1500, beta: float = 1.5, half_life: float = 7.0, spread_sigma: float = 0.02,
                      seed: int = 0) -> pd.DataFrame:
    """Log prices y, x with y = 0.5 + β·x + OU spread (x a random walk). Columns y, x."""
    rng = np.random.default_rng(seed)
    x = 4.0 + np.cumsum(rng.normal(0, 0.01, n))
    s = simulate_ou(n, np.log(2) / half_life, 0.0, spread_sigma * np.sqrt(2 * np.log(2) / half_life), seed=seed + 1)
    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame({"y": 0.5 + beta * x + s, "x": x}, index=idx)


def drifting_beta_pair(n: int = 2000, beta0: float = 1.0, beta1: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """y = β_t·x + OU noise with β_t moving linearly from beta0 to beta1; columns y, x, beta (the truth)."""
    rng = np.random.default_rng(seed)
    x = 50 + np.cumsum(rng.normal(0, 0.5, n))
    beta = np.linspace(beta0, beta1, n)
    e = simulate_ou(n, np.log(2) / 5, 0.0, 0.5, seed=seed + 1)
    return pd.DataFrame({"y": beta * x + e, "x": x, "beta": beta}, index=pd.bdate_range("2016-01-04", periods=n))


def sector_universe(n_sectors: int = 4, per_sector: int = 8, n_days: int = 1000, pairs_per_sector: int = 1,
                    seed: int = 0) -> tuple[pd.DataFrame, pd.Series, list[tuple[str, str]]]:
    """Log prices of n_sectors × per_sector stocks (sector factor + market factor + idiosyncratic random walks).
    In each sector, `pairs_per_sector` pairs are made TRULY cointegrated (the second stock = first + OU spread).
    Returns (log prices, sector of each symbol, list of true pairs)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n_days)
    market = np.cumsum(rng.normal(0.0002, 0.008, n_days))
    cols, sectors, truth = {}, {}, []
    for s in range(n_sectors):
        sec = np.cumsum(rng.normal(0, 0.006, n_days))
        names = [f"S{s}_{i}" for i in range(per_sector)]
        for name in names:
            beta = rng.uniform(0.8, 1.2)
            cols[name] = 3.0 + beta * market + sec + np.cumsum(rng.normal(0, 0.012, n_days))
            sectors[name] = f"sector{s}"
        for p in range(pairs_per_sector):
            a, b = names[2 * p], names[2 * p + 1]
            hl = rng.uniform(4, 12)
            cols[b] = cols[a] + 0.1 + simulate_ou(n_days, np.log(2) / hl, 0.0, 0.02 * np.sqrt(2 * np.log(2) / hl),
                                                  seed=seed * 100 + s * 10 + p)
            truth.append((a, b))
    return pd.DataFrame(cols, index=idx), pd.Series(sectors), truth


def factor_universe(n_stocks: int = 60, n_days: int = 750, n_factors: int = 3, reversion_share: float = 0.5,
                    seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily returns = factor exposures × factor returns + idiosyncratic part. For a `reversion_share` of the stocks
    the idiosyncratic PRICE component has a fast OU part (half-life 3 days: tradeable residual reversion) plus noise; for
    the rest it is a pure random walk. Returns (returns DataFrame, boolean array: stock has reverting residual)."""
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 0.01, (n_days, n_factors))
    B = rng.normal(1.0, 0.3, (n_stocks, n_factors)) * np.array([1.0] + [0.5] * (n_factors - 1))
    rev = np.arange(n_stocks) < int(reversion_share * n_stocks)
    idio = np.empty((n_days, n_stocks))
    for i in range(n_stocks):
        if rev[i]:
            level = simulate_ou(n_days + 1, np.log(2) / 3, 0.0, 0.01 * np.sqrt(2 * np.log(2) / 3), seed=seed * 1000 + i)
            idio[:, i] = np.diff(level) + rng.normal(0, 0.008, n_days)       # reversion hidden in noise
        else:
            idio[:, i] = rng.normal(0, 0.01, n_days)
    R = F @ B.T + idio
    return pd.DataFrame(R, index=pd.bdate_range("2018-01-02", periods=n_days),
                        columns=[f"A{i:02d}" for i in range(n_stocks)]), rev


def breaking_pair(n: int = 1500, break_at: int = 900, beta: float = 1.2, half_life: float = 6.0, seed: int = 0
                  ) -> pd.DataFrame:
    """Log prices y, x cointegrated (OU spread, known half-life) until `break_at`; from there the spread becomes a
    random walk: the relationship DIES (a merger, a new business line…). Columns y, x."""
    rng = np.random.default_rng(seed)
    x = 4.0 + np.cumsum(rng.normal(0, 0.01, n))
    s = simulate_ou(n, np.log(2) / half_life, 0.0, 0.02 * np.sqrt(2 * np.log(2) / half_life), seed=seed + 1)
    s[break_at:] = s[break_at - 1] + np.cumsum(rng.normal(0, 0.008, n - break_at))
    return pd.DataFrame({"y": 0.3 + beta * x + s, "x": x}, index=pd.bdate_range("2016-01-04", periods=n))


def characteristics_panel(n_stocks: int = 100, n_months: int = 60, premium: float = 0.002, idio: float = 0.06,
                          seed: int = 0) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Monthly returns and two standardized characteristics (dates × symbols). Next month's return has a planted
    premium on `momentum` (return = market + premium · momentum[t−1] + idiosyncratic) and none on `size` (noise).
    Returns (returns, {"momentum": …, "size": …}); exposures at t explain returns at t+1."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    cols = [f"A{i:03d}" for i in range(n_stocks)]
    mom = rng.standard_normal((n_months, n_stocks))
    size = rng.standard_normal((n_months, n_stocks))
    market = rng.normal(0.008, 0.045, n_months)
    r = market[:, None] + rng.normal(0, idio, (n_months, n_stocks))
    r[1:] += premium * mom[:-1]
    return (pd.DataFrame(r, idx, cols),
            {"momentum": pd.DataFrame(mom, idx, cols), "size": pd.DataFrame(size, idx, cols)})
