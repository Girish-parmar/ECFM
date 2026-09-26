"""Helper functions for the Part 9 guided notebooks (MFAAT, Month 8: advanced statistical trading).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

All data is SYNTHETIC with KNOWN parameters (OU processes with a known half-life, truly cointegrated pairs among
unrelated stocks, a pair whose hedge ratio drifts, a pair whose relationship dies, a factor universe where some
residuals revert, a panel with a planted premium), so every estimator can be checked against the truth. The
definitions match the graded labs in labs/part09/.
"""
from __future__ import annotations

import itertools
import os
from decimal import Decimal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist
from scipy.stats import norm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P9_STRICT") == "1"      # tests: a failed check raises instead of continuing


# --------------------------------------------------------------------------------------
# Plot style and the exercise checker
# --------------------------------------------------------------------------------------
def use_course_style() -> None:
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=PALETTE), "figure.figsize": (9, 4.5),
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#52514e", "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "legend.frameon": False,
    })


def _numeric(x) -> bool:
    return isinstance(x, (int, float, np.number, np.ndarray)) and not isinstance(x, bool)


def _same(got, expected, rtol: float, atol: float) -> bool:
    if isinstance(expected, Decimal):
        return isinstance(got, Decimal) and got == expected            # exact, and a float is not money
    if isinstance(expected, (pd.DataFrame, pd.Series)):
        test = pd.testing.assert_frame_equal if isinstance(expected, pd.DataFrame) else pd.testing.assert_series_equal
        try:
            test(got, expected, check_exact=False, rtol=rtol, atol=atol, check_dtype=False, check_names=False,
                 check_freq=False)
            return True
        except (AssertionError, TypeError, AttributeError):
            return False
    if isinstance(expected, dict):
        return (isinstance(got, dict) and set(got) == set(expected)
                and all(_same(got[k], v, rtol, atol) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)) and expected and all(_numeric(v) and np.ndim(v) == 0 for v in expected):
        expected = np.asarray(expected, dtype=float)
    if _numeric(expected):
        g, e = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
        return g.shape == e.shape and np.allclose(g, e, rtol=rtol, atol=atol, equal_nan=True)
    if isinstance(expected, (list, tuple)):
        got = list(got)
        return len(got) == len(expected) and all(_same(a, b, rtol, atol) for a, b in zip(got, expected))
    return type(got) is type(expected) and got == expected if isinstance(expected, bool) else got == expected


def check(name: str, got, expected, rtol: float = 1e-6, atol: float = 1e-9):
    """Compare your answer with the reference: exact for Decimals, text, dates and objects (recursively inside
    lists and dicts); with a tolerance for floats and arrays. Returns your value if correct, else the reference
    (so the notebook keeps running)."""
    try:
        if got is Ellipsis or (isinstance(got, (tuple, list)) and any(g is Ellipsis for g in got)):
            raise ValueError("not done yet")
        ok = bool(_same(got, expected, rtol, atol))
    except Exception:  # noqa: BLE001 - any failure means "not correct yet"
        ok = False
    if ok:
        print(f"✅ {name}: correct")
        return got
    msg = f"❌ {name}: not matching the reference yet"
    if STRICT:
        raise AssertionError(msg)
    print(msg + " — continuing with the reference answer so the rest of the notebook runs.")
    return expected


def attempt(fn, *args, **kwargs):
    """Run fn; if it raises (for example because a blank `...` is still in it), return Ellipsis instead, so
    p.check reports "not done yet" and the notebook keeps going."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - an unfinished exercise may fail in any way
        return Ellipsis


def sharpe(r) -> float:
    """Annualized Sharpe ratio of daily returns (√252); NaN when the returns never vary (no trades)."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------------------
# Data generators (known truth)
# --------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------ S1 OU & Bayes
def fit_ou(x, dt: float = 1.0) -> dict:
    """Fit dX = θ(μ − X)dt + σdW through the exact AR(1) form X[t+1] = a + b·X[t] + e (np.polyfit degree 1):
    θ = −ln(b)/dt, μ = a/(1 − b), σ = std(residuals, ddof=2)·√(2θ/(1 − b²)), half_life = ln 2/θ.
    If b is not in (0, 1) the series is not mean-reverting: return θ = 0 and half_life = inf (μ, σ NaN)."""
    x = np.asarray(x, dtype=float)
    b, a = np.polyfit(x[:-1], x[1:], 1)
    if not 0 < b < 1:
        return {"theta": 0.0, "mu": np.nan, "sigma": np.nan, "half_life": np.inf}
    resid = x[1:] - (a + b * x[:-1])
    theta = -np.log(b) / dt
    return {"theta": theta, "mu": a / (1 - b), "sigma": resid.std(ddof=2) * np.sqrt(2 * theta / (1 - b ** 2)),
            "half_life": np.log(2) / theta}


def variance_ratio(x, q: int = 5) -> float:
    """Lo–MacKinlay variance ratio of the INCREMENTS of x: Var(q-step changes) / (q · Var(1-step changes)).
    ≈ 1 random walk, < 1 mean reversion, > 1 trending (ddof=1 variances)."""
    x = np.asarray(x, dtype=float)
    return float(np.var(x[q:] - x[:-q], ddof=1) / (q * np.var(np.diff(x), ddof=1)))


def normal_posterior_mean(returns, prior_mean: float = 0.0, prior_sd: float = 0.001) -> tuple[float, float]:
    """Conjugate normal–normal posterior of a MEAN return with the data variance treated as known (sample var, ddof=1):
    precision = 1/prior_sd² + n/s²; mean = (prior_mean/prior_sd² + n·x̄/s²) / precision. Return (mean, sd).
    The posterior SHRINKS a noisy sample mean toward the prior."""
    r = np.asarray(returns, dtype=float)
    n, s2 = r.size, r.var(ddof=1)
    prec = 1 / prior_sd ** 2 + n / s2
    return float((prior_mean / prior_sd ** 2 + n * r.mean() / s2) / prec), float(np.sqrt(1 / prec))


def half_life_ci(x, n_boot: int = 200, block: int = 50, seed: int = 0, alpha: float = 0.10) -> tuple[float, float]:
    """Moving-block bootstrap CI of the OU half-life. Resample blocks of `block` consecutive REGRESSION PAIRS
    (x[t], x[t+1]) (not increments: re-joining increments adds random-walk jumps and destroys the reversion), refit
    the AR(1) slope b on the resampled pairs (np.polyfit), half-life = ln 2 / (−ln b), inf if b is not in (0, 1).
    Return the (alpha/2, 1 − alpha/2) percentiles (method="inverted_cdf": no interpolation, so inf stays inf)."""
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


# --------------------------------------------------------------- S2 cointegration & screen
def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """OLS y = α + β·x (statsmodels), spread = y − β·x − α, p-value from statsmodels coint(y, x), and the spread's
    OU half-life. Return {"alpha", "beta", "pvalue", "spread" (Series), "half_life"}."""
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    alpha, beta = float(ols.params.iloc[0]), float(ols.params.iloc[1])
    spread = y - beta * x - alpha
    return {"alpha": alpha, "beta": beta, "pvalue": float(coint(y, x)[1]), "spread": spread,
            "half_life": fit_ou(spread.to_numpy())["half_life"]}


def johansen_rank(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> tuple[int, np.ndarray]:
    """Number of cointegrating relationships (trace statistic lr1 above its 95% critical value cvt[:, 1], counted
    from the top) and the eigenvector (evec[:, 0]) of the strongest one."""
    jo = coint_johansen(prices.to_numpy(), det_order=det_order, k_ar_diff=k_ar_diff)
    rank = 0
    for stat, cv in zip(jo.lr1, jo.cvt[:, 1]):
        if stat > cv:
            rank += 1
        else:
            break
    return rank, jo.evec[:, 0]


def bh_reject(pvalues, q: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg: boolean array, True for hypotheses rejected at FDR level q (largest k with
    p_(k) <= k·q/m; reject the k smallest)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ok = p[order] <= np.arange(1, m + 1) * q / m
    out = np.zeros(m, dtype=bool)
    if ok.any():
        k = np.max(np.flatnonzero(ok))
        out[order[:k + 1]] = True
    return out


def screen_pairs(prices: pd.DataFrame, sectors: pd.Series | None = None, q: float = 0.05,
                 hl_range: tuple[float, float] = (2, 30)) -> pd.DataFrame:
    """Test every pair (a, b) with a < b in column order — only within the same sector if `sectors` is given
    (economic link first) — with engle_granger(prices[a], prices[b]). Columns: a, b, beta, pvalue, half_life,
    fdr_pass (bh_reject over ALL tested pairs), selected (fdr_pass and half-life inside hl_range). Sorted by pvalue."""
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


# ------------------------------------------------------------------------- S3 the strategy
def rolling_zscore(spread, window: int) -> np.ndarray:
    """(s − rolling mean)/rolling std (ddof=1) over the last `window` values including today; NaN in warm-up."""
    s = pd.Series(np.asarray(spread, dtype=float))
    r = s.rolling(window)
    return ((s - r.mean()) / r.std()).to_numpy()


def pairs_positions(z, entry: float = 2.0, exit_: float = 0.5, stop: float = 4.0, max_hold: int | None = None):
    """Spread position from the z-score (lesson plan S3): +1 long spread, −1 short spread.
    Flat → −sign(z) when entry < |z| < stop. In a trade → flat when |z| < exit_, |z| > stop, or held >= max_hold bars
    (bars counted after the entry bar). After a STOP or TIME STOP, no new entry until |z| has fallen below `entry`
    (do not re-enter a pair that just broke). NaN z: stay as you are. Decided at the close (filled next bar)."""
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


def pair_pnl(pos, y, x, beta: float, cost_bps: float = 5.0, borrow_bps_year: float = 50.0) -> np.ndarray:
    """Daily P&L (as a return on the gross capital of 1 + |β|) of a spread position on LOG prices y, x held from the
    NEXT bar: leg returns are diff(y) and diff(x); spread return = pos[t−1]·(Δy − β·Δx) / (1 + |β|).
    Costs: |Δpos| · cost_bps/1e4 · 2 legs, charged on the bar the trade fills (the next bar). Borrow while in a
    position: short weight × borrow_bps_year/1e4/252, where the short weight is 1/(1+|β|) when short the spread
    (short y) and |β|/(1+|β|) when long it (short x). `beta` may also be an array aligned with y (a dynamic hedge):
    then beta[t] must be the hedge HELD over bar t, i.e. known at t−1 (pass it lagged)."""
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


# ------------------------------------------------------------------------ S4 Kalman hedge
def kalman_hedge(y, x, delta: float = 1e-4, r: float = 1e-3):
    """Dynamic regression y_t = α_t + β_t·x_t + e_t with (α, β) a random walk (lesson plan S4): θ = 0, P = I,
    Vw = δ/(1−δ)·I. Each t: P += Vw; q_t = F P F' + r; e_t = y_t − F θ (forecast error BEFORE the update);
    K = P F'/q_t; θ += K e_t; P −= K F P. Return (beta, alpha, e, q) arrays (values after each update for α, β)."""
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


def rolling_ols_beta(y, x, window: int) -> np.ndarray:
    """β of y on x (with intercept) over the last `window` bars INCLUDING today; NaN in warm-up."""
    y, x = pd.Series(np.asarray(y, float)), pd.Series(np.asarray(x, float))
    return (y.rolling(window).cov(x) / x.rolling(window).var()).to_numpy()


# --------------------------------------------------------------------- S5 PCA residual stat-arb
def s_scores(returns: pd.DataFrame, n_factors: int = 5, window: int = 60, max_half_life: float | None = None
             ) -> pd.Series:
    """Avellaneda–Lee s-scores from the LAST `window` rows (lesson plan S5): standardize the returns, factors =
    Z @ (top n_factors eigenvectors of the correlation matrix, largest eigenvalue first); regress each stock's RAW
    returns on [1, factors] (np.linalg.lstsq); X = cumulative residuals; fit X as AR(1) (b, a); skip the stock if b is
    not in (0, 1) or its half-life ln 2/(−ln b) exceeds max_half_life; s = (X[−1] − a/(1−b)) / σ_eq with
    σ_eq = √(var(AR residuals, ddof=2)/(1 − b²)). Return a Series of s for the kept stocks (columns order)."""
    R = returns.iloc[-window:]
    Z = ((R - R.mean()) / R.std()).to_numpy()
    _, vec = np.linalg.eigh(np.corrcoef(Z.T))
    F = Z @ vec[:, ::-1][:, :n_factors]
    A = np.c_[np.ones(window), F]
    coef, *_ = np.linalg.lstsq(A, R.to_numpy(), rcond=None)
    X = np.cumsum(R.to_numpy() - A @ coef, axis=0)                   # window × stocks
    lag, lead = X[:-1], X[1:]
    lm, ld = lag.mean(0), lead.mean(0)
    b = ((lag - lm) * (lead - ld)).sum(0) / ((lag - lm) ** 2).sum(0)
    a = ld - b * lm
    out = {}
    for j, col in enumerate(R.columns):
        if not 0 < b[j] < 1:
            continue
        if max_half_life is not None and np.log(2) / -np.log(b[j]) > max_half_life:
            continue
        e = lead[:, j] - (a[j] + b[j] * lag[:, j])
        sig_eq = np.sqrt(e.var(ddof=2) / (1 - b[j] ** 2))
        out[col] = (X[-1, j] - a[j] / (1 - b[j])) / sig_eq
    return pd.Series(out, dtype=float)


def s_score_positions(s: pd.Series, prev: pd.Series, open_: float = 1.25, close_long: float = 0.5,
                      close_short: float = 0.75) -> pd.Series:
    """Avellaneda–Lee rules per stock (index = prev.index): flat → +1 if s < −open_, −1 if s > open_; long → flat if
    s > −close_long; short → flat if s < close_short. A stock with no s-score today (not reverting) is closed."""
    out = pd.Series(0.0, index=prev.index)
    for name, p in prev.items():
        if name not in s.index:
            continue
        v = s[name]
        if p == 0:
            out[name] = 1.0 if v < -open_ else -1.0 if v > open_ else 0.0
        elif p > 0:
            out[name] = 0.0 if v > -close_long else 1.0
        else:
            out[name] = 0.0 if v < close_short else -1.0
    return out


def residual_backtest(returns: pd.DataFrame, n_factors: int = 3, window: int = 60, max_half_life: float = 10,
                      cost_bps: float = 5.0) -> dict:
    """Daily loop from t = window − 1: s-scores from the `window` rows ending at t (inclusive, decided at the close),
    positions by s_score_positions, DOLLAR-NEUTRAL weights (+0.5 split equally over the longs, −0.5 over the shorts;
    a side with no names gets 0), earned on row t+1, minus cost_bps/1e4 · Σ|Δweights|. Return {"returns": Series indexed by the earning dates,
    "positions": DataFrame of positions decided at each t}."""
    prev = pd.Series(0.0, index=returns.columns)
    w_prev = np.zeros(returns.shape[1])
    rets, rows, dates = [], [], []
    X = returns.to_numpy()
    for t in range(window - 1, len(returns) - 1):
        s = s_scores(returns.iloc[t - window + 1:t + 1], n_factors, window, max_half_life)
        prev = s_score_positions(s, prev)
        p = prev.to_numpy()
        n_long, n_short = (p > 0).sum(), (p < 0).sum()
        w = np.where(p > 0, 0.5 / max(n_long, 1), 0.0) - np.where(p < 0, 0.5 / max(n_short, 1), 0.0)
        rets.append(w @ X[t + 1] - cost_bps / 1e4 * np.abs(w - w_prev).sum())
        w_prev = w
        rows.append(prev.to_numpy())
        dates.append(returns.index[t])
    return {"returns": pd.Series(rets, index=returns.index[window:]),
            "positions": pd.DataFrame(rows, index=dates, columns=returns.columns)}


# ------------------------------------------------------------------ S6 factor models
def newey_west_se(x, lags: int) -> float:
    """HAC standard error of the MEAN of x: √[(γ0 + 2 Σ_{j=1..lags} (1 − j/(lags+1)) γ_j) / n], with
    γ_j = (1/n) Σ (x_t − x̄)(x_{t−j} − x̄) (divide by n, like statsmodels HAC without small-sample correction)."""
    x = np.asarray(x, dtype=float)
    n, d = x.size, x - x.mean()
    s = d @ d / n
    for j in range(1, lags + 1):
        s += 2 * (1 - j / (lags + 1)) * (d[j:] @ d[:-j]) / n
    return float(np.sqrt(s / n))


def fama_macbeth(returns: pd.DataFrame, exposures: dict[str, pd.DataFrame], nw_lags: int = 0) -> pd.DataFrame:
    """Lesson plan S6: for each consecutive pair of dates (t, t+1), regress returns at t+1 on the exposures at t across
    stocks (statsmodels OLS with a constant, rows with NaN dropped); premium = mean of the slopes. t_stat = mean /
    (std(ddof=1)/√n) when nw_lags == 0, else mean / newey_west_se(slopes, nw_lags). Index: const + exposure names."""
    lambdas = []
    for t, t1 in zip(returns.index[:-1], returns.index[1:]):
        X = pd.DataFrame({k: v.loc[t] for k, v in exposures.items()}).dropna()
        y = returns.loc[t1, X.index]
        lambdas.append(sm.OLS(y, sm.add_constant(X)).fit().params)
    L = pd.DataFrame(lambdas)
    if nw_lags == 0:
        se = L.std(ddof=1) / np.sqrt(len(L))
    else:
        se = L.apply(lambda c: newey_west_se(c.to_numpy(), nw_lags))
    return pd.DataFrame({"premium": L.mean(), "t_stat": L.mean() / se})


def neutralize(signal: pd.Series, sectors: pd.Series, beta: pd.Series) -> pd.Series:
    """Residual of a cross-sectional OLS of the signal on one dummy per sector (no extra constant) and beta: what is
    left has zero mean in every sector and zero covariance with beta."""
    D = pd.get_dummies(sectors.loc[signal.index], dtype=float)
    X = pd.concat([D, beta.loc[signal.index].rename("beta")], axis=1)
    return sm.OLS(signal, X).fit().resid


# ------------------------------------------------------ S7 breaks, stability, multiple testing
def chow_test(y, x, break_idx: int) -> tuple[float, float]:
    """Chow test for a break in y = α + βx at a KNOWN index (first row of the second segment), k = 2 parameters:
    F = ((SSR_pooled − SSR_1 − SSR_2)/k) / ((SSR_1 + SSR_2)/(n − 2k)); p from the F(k, n − 2k) distribution."""
    y, x = np.asarray(y, float), np.asarray(x, float)

    def ssr(yy, xx):
        return sm.OLS(yy, sm.add_constant(xx)).fit().ssr

    n, k = y.size, 2
    s1, s2 = ssr(y[:break_idx], x[:break_idx]), ssr(y[break_idx:], x[break_idx:])
    F = ((ssr(y, x) - s1 - s2) / k) / ((s1 + s2) / (n - 2 * k))
    return float(F), float(f_dist.sf(F, k, n - 2 * k))


def cusum_pvalue(y, x) -> float:
    """Break at an UNKNOWN date: OLS y on [1, x], then statsmodels breaks_cusumolsresid(residuals, ddof=2); its
    p-value."""
    resid = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit().resid
    return float(breaks_cusumolsresid(resid, ddof=2)[1])


def rolling_stability(y: pd.Series, x: pd.Series, window: int = 250, step: int = 21) -> pd.DataFrame:
    """Every `step` rows, starting with the first full window, run engle_granger on the last `window` rows.
    Index: the last date of each window. Columns: pvalue, half_life, beta, spread_vol (std of the spread, ddof=1)."""
    rows, idx = [], []
    for end in range(window, len(y) + 1, step):
        eg = engle_granger(y.iloc[end - window:end], x.iloc[end - window:end])
        rows.append({"pvalue": eg["pvalue"], "half_life": eg["half_life"], "beta": eg["beta"],
                     "spread_vol": float(eg["spread"].std(ddof=1))})
        idx.append(y.index[end - 1])
    return pd.DataFrame(rows, index=idx)


def retirement_date(stab: pd.DataFrame, max_p: float = 0.2, max_hl: float = 30, max_beta_drift: float = 0.25,
                    patience: int = 2):
    """Retire the pair at the first check where a rule has been broken on `patience` CONSECUTIVE checks. Broken:
    pvalue > max_p, or half_life > max_hl, or |β/β_first − 1| > max_beta_drift (β_first = the first row's β).
    Return the date, or None if the pair survives."""
    bad = ((stab["pvalue"] > max_p) | (stab["half_life"] > max_hl)
           | ((stab["beta"] / stab["beta"].iloc[0] - 1).abs() > max_beta_drift)).to_numpy()
    run = 0
    for date, b in zip(stab.index, bad):
        run = run + 1 if b else 0
        if run >= patience:
            return date
    return None


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano: start at a random index; at each step continue to the next index (wrapping around) with
    probability 1 − 1/mean_block, else jump to a new random index."""
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    jump = rng.random(n) < 1 / mean_block
    new = rng.integers(0, n, n)
    for t in range(1, n):
        idx[t] = new[t] if jump[t] else (idx[t - 1] + 1) % n
    return idx


def reality_check(perf, n_boot: int = 500, mean_block: float = 10, seed: int = 0) -> dict:
    """White's Reality Check for "is the BEST of K strategies better than zero?". perf: T × K excess returns.
    V = max_k √T·mean_k; each bootstrap (stationary_bootstrap_indices, same rows for all strategies):
    V* = max_k √T·(mean*_k − mean_k). Return {"best": column index of the best mean, "p_value": share of V* >= V,
    "naive_p": one-sided t-test p-value of the best strategy alone (as if it were the only one tried)}."""
    P = np.asarray(perf, dtype=float)
    T = P.shape[0]
    m = P.mean(axis=0)
    V = np.sqrt(T) * m.max()
    rng = np.random.default_rng(seed)
    Vs = np.array([np.sqrt(T) * (P[stationary_bootstrap_indices(T, mean_block, rng)].mean(axis=0) - m).max()
                   for _ in range(n_boot)])
    best = int(np.argmax(m))
    t = m[best] / (P[:, best].std(ddof=1) / np.sqrt(T))
    return {"best": best, "p_value": float((Vs >= V).mean()), "naive_p": float(norm.sf(t))}


# ----------------------------------------------------------- S8 ensembles and a book of pairs
def ensemble_positions(z, entries=(1.75, 2.0, 2.25), exits=(0.25, 0.5, 0.75), stop: float = 4.0,
                       max_hold: int | None = None) -> np.ndarray:
    """Average of pairs_positions over every (entry, exit) combination: a fractional position in [−1, 1] that is
    much less sensitive to any single parameter choice."""
    return np.mean([pairs_positions(z, e, x, stop, max_hold) for e in entries for x in exits], axis=0)


def pair_book_weights(pair_returns: pd.DataFrame, pairs: dict[str, tuple[str, str]], name_cap: float = 0.2,
                      max_iter: int = 1000) -> pd.Series:
    """Inverse-volatility weights over the pairs (columns of pair_returns, std ddof=1) summing to 1, with the load of
    every stock (sum of the weights of the pairs it appears in) <= name_cap: repeat {scale each pair by
    min(1, cap/load) over its two names; renormalize to 1} until the max load <= cap·(1 + 1e-9).
    Raise ValueError if that has not happened after max_iter rounds (infeasible cap)."""
    w = 1 / pair_returns.std(ddof=1)
    w = w / w.sum()
    for _ in range(max_iter):
        load_ = {}
        for p, (a, b) in pairs.items():
            load_[a] = load_.get(a, 0.0) + w[p]
            load_[b] = load_.get(b, 0.0) + w[p]
        if max(load_.values()) <= name_cap * (1 + 1e-9):
            return w
        f = pd.Series({p: min(1.0, name_cap / load_[a], name_cap / load_[b]) for p, (a, b) in pairs.items()})
        w = w * f
        w = w / w.sum()
    raise ValueError(f"name cap {name_cap} is infeasible for these pairs")


def book_exposures(pair_pos: pd.Series, pair_w: pd.Series, pairs: dict[str, tuple[str, str]], hedges: pd.Series,
                   betas: pd.Series, sectors: pd.Series) -> dict:
    """Stock-level view of a book of pairs. Pair p with position s (+1 long spread) and capital weight w, hedge h,
    names (a, b): a gets s·w/(1+|h|), b gets −s·w·h/(1+|h|) (a stock can collect weight from several pairs).
    Return {"weights": Series per stock (sorted index, zeros dropped), "gross": Σ|w|, "net": Σw,
    "net_beta": Σ w·beta, "sector_net": Series of Σw per sector (sorted)}."""
    w = {}
    for p, (a, b) in pairs.items():
        s, h = pair_pos[p] * pair_w[p], hedges[p]
        w[a] = w.get(a, 0.0) + s / (1 + abs(h))
        w[b] = w.get(b, 0.0) - s * h / (1 + abs(h))
    w = pd.Series(w).sort_index()
    w = w[w != 0]
    return {"weights": w, "gross": float(w.abs().sum()), "net": float(w.sum()),
            "net_beta": float((w * betas.loc[w.index]).sum()),
            "sector_net": w.groupby(sectors.loc[w.index]).sum().sort_index()}
