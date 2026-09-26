"""Week 31 (S1–S4) — Mean reversion and pairs: OU processes, Bayesian estimation, cointegration, pair screening
with FDR, the pairs strategy and the Kalman hedge ratio.

Rule of the week: the FORMATION period (where you estimate and select) is strictly before the TRADING period.
Fill in every block marked "Your turn", then run:  python -m pytest week31_pairs
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ------------------------------------------------------------------------ S1 OU & Bayes
def fit_ou(x, dt: float = 1.0) -> dict:
    """Fit dX = θ(μ − X)dt + σdW through the exact AR(1) form X[t+1] = a + b·X[t] + e (np.polyfit degree 1):
    θ = −ln(b)/dt, μ = a/(1 − b), σ = std(residuals, ddof=2)·√(2θ/(1 − b²)), half_life = ln 2/θ.
    If b is not in (0, 1) the series is not mean-reverting: return θ = 0 and half_life = inf (μ, σ NaN)."""
    # >>> SOLUTION
    x = np.asarray(x, dtype=float)
    b, a = np.polyfit(x[:-1], x[1:], 1)
    if not 0 < b < 1:
        return {"theta": 0.0, "mu": np.nan, "sigma": np.nan, "half_life": np.inf}
    resid = x[1:] - (a + b * x[:-1])
    theta = -np.log(b) / dt
    return {"theta": theta, "mu": a / (1 - b), "sigma": resid.std(ddof=2) * np.sqrt(2 * theta / (1 - b ** 2)),
            "half_life": np.log(2) / theta}
    # <<< SOLUTION


def variance_ratio(x, q: int = 5) -> float:
    """Lo–MacKinlay variance ratio of the INCREMENTS of x: Var(q-step changes) / (q · Var(1-step changes)).
    ≈ 1 random walk, < 1 mean reversion, > 1 trending (ddof=1 variances)."""
    # >>> SOLUTION
    x = np.asarray(x, dtype=float)
    return float(np.var(x[q:] - x[:-q], ddof=1) / (q * np.var(np.diff(x), ddof=1)))
    # <<< SOLUTION


def normal_posterior_mean(returns, prior_mean: float = 0.0, prior_sd: float = 0.001) -> tuple[float, float]:
    """Conjugate normal–normal posterior of a MEAN return with the data variance treated as known (sample var, ddof=1):
    precision = 1/prior_sd² + n/s²; mean = (prior_mean/prior_sd² + n·x̄/s²) / precision. Return (mean, sd).
    The posterior SHRINKS a noisy sample mean toward the prior."""
    # >>> SOLUTION
    r = np.asarray(returns, dtype=float)
    n, s2 = r.size, r.var(ddof=1)
    prec = 1 / prior_sd ** 2 + n / s2
    return float((prior_mean / prior_sd ** 2 + n * r.mean() / s2) / prec), float(np.sqrt(1 / prec))
    # <<< SOLUTION


def half_life_ci(x, n_boot: int = 200, block: int = 50, seed: int = 0, alpha: float = 0.10) -> tuple[float, float]:
    """Moving-block bootstrap CI of the OU half-life. Resample blocks of `block` consecutive REGRESSION PAIRS
    (x[t], x[t+1]) (not increments: re-joining increments adds random-walk jumps and destroys the reversion), refit
    the AR(1) slope b on the resampled pairs (np.polyfit), half-life = ln 2 / (−ln b), inf if b is not in (0, 1).
    Return the (alpha/2, 1 − alpha/2) percentiles (method="inverted_cdf": no interpolation, so inf stays inf)."""
    # >>> SOLUTION
    x = np.asarray(x, dtype=float)
    lag, lead = x[:-1], x[1:]
    m = lag.size
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    hls = []
    for _ in range(n_boot):
        starts = rng.integers(0, m - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:m]
        b = np.polyfit(lag[idx], lead[idx], 1)[0]
        hls.append(np.log(2) / -np.log(b) if 0 < b < 1 else np.inf)
    lo, hi = np.percentile(hls, [100 * alpha / 2, 100 * (1 - alpha / 2)], method="inverted_cdf")   # inf-safe
    return float(lo), float(hi)
    # <<< SOLUTION


# --------------------------------------------------------------- S2 cointegration & screen
def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """OLS y = α + β·x (statsmodels), spread = y − β·x − α, p-value from statsmodels coint(y, x), and the spread's
    OU half-life. Return {"alpha", "beta", "pvalue", "spread" (Series), "half_life"}."""
    # >>> SOLUTION
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    alpha, beta = float(ols.params.iloc[0]), float(ols.params.iloc[1])
    spread = y - beta * x - alpha
    return {"alpha": alpha, "beta": beta, "pvalue": float(coint(y, x)[1]), "spread": spread,
            "half_life": fit_ou(spread.to_numpy())["half_life"]}
    # <<< SOLUTION


def johansen_rank(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> tuple[int, np.ndarray]:
    """Number of cointegrating relationships (trace statistic lr1 above its 95% critical value cvt[:, 1], counted
    from the top) and the eigenvector (evec[:, 0]) of the strongest one."""
    # >>> SOLUTION
    jo = coint_johansen(prices.to_numpy(), det_order=det_order, k_ar_diff=k_ar_diff)
    rank = 0
    for stat, cv in zip(jo.lr1, jo.cvt[:, 1]):
        if stat > cv:
            rank += 1
        else:
            break
    return rank, jo.evec[:, 0]
    # <<< SOLUTION


def bh_reject(pvalues, q: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg: boolean array, True for hypotheses rejected at FDR level q (largest k with
    p_(k) <= k·q/m; reject the k smallest)."""
    # >>> SOLUTION
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ok = p[order] <= np.arange(1, m + 1) * q / m
    out = np.zeros(m, dtype=bool)
    if ok.any():
        k = np.max(np.flatnonzero(ok))
        out[order[:k + 1]] = True
    return out
    # <<< SOLUTION


def screen_pairs(prices: pd.DataFrame, sectors: pd.Series | None = None, q: float = 0.05,
                 hl_range: tuple[float, float] = (2, 30)) -> pd.DataFrame:
    """Test every pair (a, b) with a < b in column order — only within the same sector if `sectors` is given
    (economic link first) — with engle_granger(prices[a], prices[b]). Columns: a, b, beta, pvalue, half_life,
    fdr_pass (bh_reject over ALL tested pairs), selected (fdr_pass and half-life inside hl_range). Sorted by pvalue."""
    # >>> SOLUTION
    rows = []
    for a, b in itertools.combinations(prices.columns, 2):
        if sectors is not None and sectors[a] != sectors[b]:
            continue
        eg = engle_granger(prices[a], prices[b])
        rows.append({"a": a, "b": b, "beta": eg["beta"], "pvalue": eg["pvalue"], "half_life": eg["half_life"]})
    df = pd.DataFrame(rows)
    df["fdr_pass"] = bh_reject(df["pvalue"].to_numpy(), q)
    df["selected"] = df["fdr_pass"] & df["half_life"].between(*hl_range)
    return df.sort_values("pvalue", ignore_index=True)
    # <<< SOLUTION


# ------------------------------------------------------------------------- S3 the strategy
def rolling_zscore(spread, window: int) -> np.ndarray:
    """(s − rolling mean)/rolling std (ddof=1) over the last `window` values including today; NaN in warm-up."""
    # >>> SOLUTION
    s = pd.Series(np.asarray(spread, dtype=float))
    r = s.rolling(window)
    return ((s - r.mean()) / r.std()).to_numpy()
    # <<< SOLUTION


def pairs_positions(z, entry: float = 2.0, exit_: float = 0.5, stop: float = 4.0, max_hold: int | None = None):
    """Spread position from the z-score (lesson plan S3): +1 long spread, −1 short spread.
    Flat → −sign(z) when entry < |z| < stop. In a trade → flat when |z| < exit_, |z| > stop, or held >= max_hold bars
    (bars counted after the entry bar). After a STOP or TIME STOP, no new entry until |z| has fallen below `entry`
    (do not re-enter a pair that just broke). NaN z: stay as you are. Decided at the close (filled next bar)."""
    # >>> SOLUTION
    z = np.asarray(z, dtype=float)
    pos, held, blocked = np.zeros(z.size), 0, False
    for t in range(z.size):
        p = pos[t - 1] if t else 0.0
        if np.isnan(z[t]):
            pos[t] = p
            continue
        if p == 0:
            blocked = blocked and abs(z[t]) >= entry
            if not blocked and entry < abs(z[t]) < stop:
                p, held = -np.sign(z[t]), 0
        else:
            held += 1
            if abs(z[t]) < exit_:
                p = 0.0
            elif abs(z[t]) > stop or (max_hold is not None and held >= max_hold):
                p, blocked = 0.0, True
        pos[t] = p
    return pos
    # <<< SOLUTION


def pair_pnl(pos, y, x, beta: float, cost_bps: float = 5.0, borrow_bps_year: float = 50.0) -> np.ndarray:
    """Daily P&L (as a return on the gross capital of 1 + |β|) of a spread position on LOG prices y, x held from the
    NEXT bar: leg returns are diff(y) and diff(x); spread return = pos[t−1]·(Δy − β·Δx) / (1 + |β|).
    Costs: |Δpos| · cost_bps/1e4 · 2 legs, charged on the bar the trade fills (the next bar). Borrow while in a
    position: short weight × borrow_bps_year/1e4/252, where the short weight is 1/(1+|β|) when short the spread
    (short y) and |β|/(1+|β|) when long it (short x). `beta` may also be an array aligned with y (a dynamic hedge):
    then beta[t] must be the hedge HELD over bar t, i.e. known at t−1 (pass it lagged)."""
    # >>> SOLUTION
    pos = np.asarray(pos, dtype=float)
    dy, dx = np.diff(np.asarray(y, float), prepend=np.nan), np.diff(np.asarray(x, float), prepend=np.nan)
    held = np.concatenate([[0.0], pos[:-1]])
    beta = np.asarray(beta, dtype=float)
    gross = 1 + np.abs(beta)
    ret = np.nan_to_num(held * (dy - beta * dx) / gross)
    trade_cost = np.abs(np.diff(pos, prepend=0.0)) * cost_bps / 1e4 * 2
    trade_cost = np.concatenate([[0.0], trade_cost[:-1]])                   # traded at the next bar
    short_w = np.where(held < 0, 1 / gross, np.where(held > 0, np.abs(beta) / gross, 0.0))
    borrow = short_w * borrow_bps_year / 1e4 / 252
    return ret - trade_cost - borrow
    # <<< SOLUTION


# ------------------------------------------------------------------------ S4 Kalman hedge
def kalman_hedge(y, x, delta: float = 1e-4, r: float = 1e-3):
    """Dynamic regression y_t = α_t + β_t·x_t + e_t with (α, β) a random walk (lesson plan S4): θ = 0, P = I,
    Vw = δ/(1−δ)·I. Each t: P += Vw; q_t = F P F' + r; e_t = y_t − F θ (forecast error BEFORE the update);
    K = P F'/q_t; θ += K e_t; P −= K F P. Return (beta, alpha, e, q) arrays (values after each update for α, β)."""
    # >>> SOLUTION
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = y.size
    theta, P = np.zeros(2), np.eye(2)
    Vw = delta / (1 - delta) * np.eye(2)
    beta, alpha, e, q = (np.empty(n) for _ in range(4))
    for t in range(n):
        F = np.array([1.0, x[t]])
        P = P + Vw
        q[t] = F @ P @ F + r
        e[t] = y[t] - F @ theta
        K = P @ F / q[t]
        theta = theta + K * e[t]
        P = P - np.outer(K, F) @ P
        alpha[t], beta[t] = theta
    return beta, alpha, e, q
    # <<< SOLUTION


def rolling_ols_beta(y, x, window: int) -> np.ndarray:
    """β of y on x (with intercept) over the last `window` bars INCLUDING today; NaN in warm-up."""
    # >>> SOLUTION
    y, x = pd.Series(np.asarray(y, float)), pd.Series(np.asarray(x, float))
    return (y.rolling(window).cov(x) / x.rolling(window).var()).to_numpy()
    # <<< SOLUTION
