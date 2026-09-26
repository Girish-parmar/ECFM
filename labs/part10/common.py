"""Shared, complete data generators for the Part 10 labs (nothing to fill in here).

Every generator has a KNOWN structure (a hidden regime, a known autocorrelation, a known volatility), so the labs can
check what a model is able to learn — and that it learns nothing on pure noise. For graded research, use real data
from your feature store instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tick_data(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Trades with UNEVEN activity: each day has its own intensity (quiet and busy days) and a U-shaped intraday
    profile. Every trade moves the log price by the same noise scale, so a fixed-TIME bar's variance follows the
    activity while a fixed-DOLLAR bar's does not. Columns price, size; index = trade timestamps."""
    rng = np.random.default_rng(seed)
    frames, logp = [], np.log(100.0)
    days = pd.bdate_range("2024-01-02", periods=n_days)
    for day in days:
        n = int(rng.lognormal(np.log(4000), 0.6))
        u = np.sort(rng.beta(0.6, 0.6, n))                           # U shape: busy open and close
        ts = day + pd.Timedelta(hours=9.5) + pd.to_timedelta(u * 6.5 * 3600, unit="s")
        steps = rng.standard_t(5, n) * 0.0004
        path = logp + np.cumsum(steps)
        logp = path[-1]
        size = np.round(rng.lognormal(np.log(100), 0.8, n)).clip(1)
        frames.append(pd.DataFrame({"price": np.round(np.exp(path), 4), "size": size}, index=ts))
    return pd.concat(frames)


def random_walk_prices(n: int = 2000, sigma: float = 0.01, seed: int = 0) -> pd.Series:
    """A log-price random walk (no signal at all), as a price Series on business days."""
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, sigma, n))), index=pd.bdate_range("2015-01-02", periods=n))


def regime_market(n: int = 2500, seed: int = 0, drift: float = 0.002, trend_vol: float = 0.007,
                  flip_every: float = 120) -> pd.DataFrame:
    """Daily bars with a HIDDEN two-state regime (persistent, ~100 days per spell).
    trend (regime 1): calm (vol 0.7%), a drift of ±0.2%/day whose sign flips about every 120 days → momentum works.
    chop  (regime 0): volatile (vol 1.5%), no drift, returns mean-revert day to day (AR(1) φ = −0.15) → momentum fails.
    Columns close, volume, regime (the TRUTH: never use it as a feature)."""
    rng = np.random.default_rng(seed)
    regime = np.empty(n, dtype=int)
    regime[0] = 1
    for t in range(1, n):
        regime[t] = regime[t - 1] if rng.random() < 0.99 else 1 - regime[t - 1]
    sign, r = 1.0, np.zeros(n)
    for t in range(1, n):
        if rng.random() < 1 / flip_every:
            sign = -sign
        if regime[t] == 1:
            r[t] = drift * sign + rng.normal(0, trend_vol)
        else:
            r[t] = -0.15 * r[t - 1] + rng.normal(0, 0.015)
    volume = rng.lognormal(np.log(1e6), 0.3, n) * np.where(regime == 1, 1.0, 1.6)
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(r)), "volume": volume.round(), "regime": regime},
                        index=pd.bdate_range("2012-01-02", periods=n))


def ar1(n: int = 3000, phi: float = 0.6, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """x[t] = φ x[t−1] + ε: the best possible correlation between x[t] and a forecast made at t−1 is φ."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def garch_returns(n: int = 3000, omega: float = 2e-6, alpha: float = 0.09, beta: float = 0.89, seed: int = 0
                  ) -> pd.DataFrame:
    """GARCH(1,1) daily returns with Student-t shocks. Columns ret, vol (the TRUE conditional volatility, known at
    the start of the day). Volatility clusters, so it is forecastable; the sign of the return is not."""
    rng = np.random.default_rng(seed)
    var = np.empty(n)
    r = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    shocks = rng.standard_t(6, n) / np.sqrt(6 / 4)
    for t in range(n):
        if t:
            var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        r[t] = np.sqrt(var[t]) * shocks[t]
    return pd.DataFrame({"ret": r, "vol": np.sqrt(var)}, index=pd.bdate_range("2010-01-04", periods=n))


def overlapping_dataset(n: int = 1500, horizon: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """NO signal, but OVERLAPPING labels: from a random walk, features = past 5/20/60-day returns and 20-day
    volatility at t, label = 1 if the return over the NEXT `horizon` days is positive, t1 = the date that label is
    resolved. Neighbouring rows share most of their label window, which is exactly what fools a shuffled k-fold.
    Returns (X, y, t1) aligned on the dates."""
    close = random_walk_prices(n + 80, seed=seed)
    r = np.log(close).diff()
    X = pd.DataFrame({"ret5": r.rolling(5).sum(), "ret20": r.rolling(20).sum(), "ret60": r.rolling(60).sum(),
                      "vol20": r.rolling(20).std()})
    fwd = np.log(close).shift(-horizon) - np.log(close)
    t1 = pd.Series(close.index[np.minimum(np.arange(len(close)) + horizon, len(close) - 1)], index=close.index)
    ok = X.notna().all(axis=1) & fwd.notna()
    return X[ok], (fwd[ok] > 0).astype(int), t1[ok]


def planted_features(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A classification set with KNOWN importances: y depends on `signal` only. `twin` is signal plus a little noise
    (a substitute), `noise_cont` is continuous noise (high cardinality, flatters MDI), `noise_bin` is binary noise.
    Rows are independent (no overlap)."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    X = pd.DataFrame({"signal": s, "twin": s + rng.normal(0, 0.3, n), "noise_cont": rng.normal(size=n),
                      "noise_bin": rng.integers(0, 2, n).astype(float)},
                     index=pd.bdate_range("2010-01-04", periods=n))
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-1.5 * s))).astype(int), index=X.index)
    return X, y
