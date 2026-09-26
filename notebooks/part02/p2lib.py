"""Helper functions for the Part 2 guided notebooks (MFAAT, Month 2).

The notebooks give you the data loading and plotting code; you write the short formula cells
marked "✍️ Your turn". Each exercise ends with `p.check(...)`, which tells you whether your answer
matches the reference implementation in this file. If it does not, the notebook carries on with
the reference value so later cells still run.

Data: set the environment variable P2_DATA to the course price file (CSV or Parquet, wide format:
one row per date, one column per ticker). Without it, a SYNTHETIC market is generated: realistic
behaviour (fat tails, volatility clustering, leverage effect, a common market factor, one
cointegrated pair) but made-up numbers. Useful for practice; use the course data for graded work.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLE", "AAPL", "JPM", "XOM", "SMLCAP"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P2_STRICT") == "1"      # tests: a failed check raises instead of continuing


# --------------------------------------------------------------------------------------
# Plot style
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


# --------------------------------------------------------------------------------------
# Exercise checker
# --------------------------------------------------------------------------------------
def check(name: str, got, expected, rtol: float = 1e-6, atol: float = 1e-9):
    """Compare your answer with the reference. Returns your value if correct, else the reference."""
    try:
        if got is Ellipsis:
            raise ValueError("not done yet")
        if isinstance(expected, (pd.Series, pd.DataFrame)):
            got_arr = np.asarray(pd.DataFrame(got).reindex_like(pd.DataFrame(expected)), dtype=float)
        else:
            got_arr = np.asarray(got, dtype=float)
        ok = got_arr.shape == np.shape(expected) and np.allclose(
            got_arr, np.asarray(expected, dtype=float), rtol=rtol, atol=atol, equal_nan=True)
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


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _synthetic_market(start: str = "2008-01-02", end: str = "2025-06-30", seed: int = 42) -> pd.DataFrame:
    """SYNTHETIC daily closes with realistic stylized facts (made-up numbers)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    def gjr_t(n, omega, alpha, gamma, beta, nu, down_scale=1.0):
        z = stats.t.rvs(nu, size=n, random_state=rng)
        z = np.where(z < 0, z * down_scale, z / down_scale)          # down_scale > 1: heavier left tail
        z = (z - z.mean()) / z.std()
        r, h = np.empty(n), omega / (1 - alpha - gamma / 2 - beta)
        for t in range(n):
            r[t] = np.sqrt(h) * z[t]
            h = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * h
        return r

    stress = np.ones(n)                                    # crisis-like periods: higher volatility
    for a, b, m in [("2008-09-01", "2009-03-31", 2.2), ("2020-02-20", "2020-05-15", 2.8), ("2022-01-01", "2022-10-31", 1.4)]:
        stress[(dates >= a) & (dates <= b)] = m
    jumps = -np.abs(rng.normal(0.025, 0.01, n)) * (rng.random(n) < 0.004)   # rare crash days: negative skew
    mkt = gjr_t(n, 3.0e-6, 0.03, 0.12, 0.87, 5, down_scale=1.25) * stress + jumps + 0.0004
    spec = {  # beta to market, idiosyncratic vol scale, drift
        "SPY": (1.00, 0.10, 0.0000), "QQQ": (1.15, 0.45, 0.0001), "IWM": (1.20, 0.55, -0.0001),
        "TLT": (-0.25, 0.75, 0.0001), "GLD": (0.05, 0.85, 0.0001), "XLE": (1.05, 1.10, -0.0001),
        "AAPL": (1.20, 1.20, 0.0004), "JPM": (1.25, 1.00, 0.0001), "SMLCAP": (1.30, 2.20, -0.0002),
    }
    rets = {}
    for tk, (beta, idio, drift) in spec.items():
        rets[tk] = beta * mkt + idio * gjr_t(n, 1.2e-6, 0.05, 0.03, 0.90, 5) * np.sqrt(stress) + drift
    prices = pd.DataFrame({tk: 100 * np.exp(np.cumsum(r)) for tk, r in rets.items()}, index=dates)
    spread = np.zeros(n)                                   # XOM cointegrated with XLE (half-life ~ 10 days)
    for t in range(1, n):
        spread[t] = 0.933 * spread[t - 1] + rng.normal(0, 0.008)
    prices["XOM"] = np.exp(np.log(prices["XLE"]) * 0.9 + 0.4 + spread)
    prices.index.name = "date"
    return prices[TICKERS].round(4)


def load_prices(tickers: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    """Daily closing prices (dates × tickers) from P2_DATA, or the SYNTHETIC market if not set."""
    path = os.environ.get("P2_DATA")
    if path:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
    else:
        print("⚠️  Using the SYNTHETIC market (set P2_DATA to use the course data file).")
        df = _synthetic_market()
    df = df[tickers] if tickers else df
    return df.loc[start:] if start else df


def intraday_5min(ticker: str = "SPY", days: int = 250, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC 5-minute log returns (78 bars/day) whose daily variance clusters like real data."""
    rng = np.random.default_rng(seed + sum(map(ord, ticker)))
    daily_vol = np.exp(np.convolve(rng.normal(0, 0.35, days + 20), np.ones(20) / 20, mode="valid")[:days]) * 0.01
    idx, r = [], []
    for d, (day, vol) in enumerate(zip(pd.bdate_range("2024-01-02", periods=days), daily_vol)):
        u = np.linspace(-1, 1, 78)
        w = (1 + 1.5 * u**2); w /= w.sum()                  # more variance at open and close
        r.append(rng.standard_t(5, 78) / np.sqrt(5 / 3) * vol * np.sqrt(w))
        idx.append(pd.date_range(day + pd.Timedelta("14:30:00"), periods=78, freq="5min", tz="UTC"))
    return pd.DataFrame({"ret": np.concatenate(r)}, index=idx[0].append(idx[1:]))


# --------------------------------------------------------------------------------------
# Reference implementations (what your exercise answers are checked against)
# --------------------------------------------------------------------------------------
def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna(how="all")


def simple_returns(prices):
    return prices.pct_change(fill_method=None).dropna(how="all")


def portfolio_vol(w, cov, periods=252):
    return float(np.sqrt(np.asarray(w) @ np.asarray(cov) @ np.asarray(w) * periods))


def pca_explained(returns: pd.DataFrame, k=3):
    eigval, eigvec = np.linalg.eigh(np.corrcoef(returns.dropna().T.to_numpy()))
    eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
    return eigval[:k] / eigval.sum(), eigvec[:, :k]


def min_variance_weights(cov):
    x = np.linalg.solve(np.asarray(cov), np.ones(len(cov)))
    return x / x.sum()


def delta_gamma_pnl(delta, gamma, dS):
    return delta * dS + 0.5 * gamma * dS**2


def describe(r: pd.Series) -> dict:
    r = pd.Series(r).dropna().to_numpy()
    return {"mean": r.mean(), "std": r.std(ddof=1), "skew": stats.skew(r),
            "excess_kurtosis": stats.kurtosis(r), "jb_pvalue": stats.jarque_bera(r).pvalue}


def tail_count(r, k=4.0):
    r = pd.Series(r).dropna().to_numpy()
    z = (r - r.mean()) / r.std(ddof=1)
    return int((np.abs(z) > k).sum()), float(2 * stats.norm.sf(k) * len(r))


def autocorr(x: pd.Series, lag: int = 1) -> float:
    return float(pd.Series(x).dropna().autocorr(lag))


def tstat_mean(r):
    r = pd.Series(r).dropna().to_numpy()
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))


def years_needed(sharpe_annual, t_target=2.0):
    return (t_target / sharpe_annual) ** 2


def block_bootstrap_ci(r, stat=np.mean, block=20, n_boot=2000, alpha=0.05, seed=0):
    r = np.asarray(pd.Series(r).dropna()); n = len(r); rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    boots = [stat(r[np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n // block + 1)])[:n]])
             for _ in range(n_boot)]
    return np.quantile(boots, [alpha / 2, 1 - alpha / 2])


def variance_ratio(log_prices, q=5):
    lp = np.asarray(pd.Series(log_prices).dropna())
    r1 = np.diff(lp); rq = lp[q:] - lp[:-q]
    return rq.var(ddof=1) / (q * r1.var(ddof=1))


def half_life(x):
    x = np.asarray(pd.Series(x).dropna())
    b, a = np.polyfit(x[:-1], x[1:], 1)
    return np.log(2) / -np.log(b)


def ewma_var(r, lam=0.94):
    """EWMA variance forecast for each day, using returns up to the day before."""
    r = np.asarray(pd.Series(r).fillna(0.0)); var = np.empty(len(r)); var[0] = r[:20].var()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return var


def qlike(realized_var, forecast_var):
    ratio = np.asarray(realized_var) / np.asarray(forecast_var)
    return float(np.mean(ratio - np.log(ratio) - 1))


def hmm_forward_filter(model, x):
    """Filtered state probabilities P(state_t | data up to t) from a fitted hmmlearn GaussianHMM.

    Unlike model.predict (which uses the whole sample), each row only uses data up to that day.
    (The model parameters were still fitted on the full sample; refit on an expanding window for a
    fully honest backtest.)"""
    x = np.asarray(x).reshape(len(x), -1)
    k = model.n_components
    covs = model.covars_ if model.covariance_type == "full" else np.array([np.diag(np.atleast_1d(c)) for c in model.covars_])
    B = np.column_stack([stats.multivariate_normal(model.means_[j], covs[j]).pdf(x) for j in range(k)])
    alpha = np.zeros((len(x), k))
    a = model.startprob_ * B[0]; alpha[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (alpha[t - 1] @ model.transmat_) * B[t]
        alpha[t] = a / a.sum()
    return alpha


def gbm_paths(S0, mu, sigma, T, steps, n_paths, seed=0):
    dt = T / steps
    z = np.random.default_rng(seed).standard_normal((n_paths, steps))
    return S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1))


def max_drawdown(equity) -> float:
    eq = np.asarray(equity, dtype=float)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def sharpe(r, rf=0.0, periods=252):
    ex = np.asarray(pd.Series(r).dropna()) - rf / periods
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(periods))


def sortino(r, rf=0.0, periods=252):
    ex = np.asarray(pd.Series(r).dropna()) - rf / periods
    return float(ex.mean() / np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(periods))


def hist_var_cvar(r, alpha=0.99):
    r = np.asarray(pd.Series(r).dropna())
    var = -np.quantile(r, 1 - alpha)
    return float(var), float(-r[r <= -var].mean())


def metrics(r, rf=0.0, periods=252, benchmark=None) -> dict:
    r = pd.Series(r).dropna()
    eq = np.cumprod(1 + r.to_numpy())
    cagr = eq[-1] ** (periods / len(r)) - 1
    out = {"CAGR": cagr, "Volatility": r.std(ddof=1) * np.sqrt(periods), "Sharpe": sharpe(r, rf, periods),
           "Sortino": sortino(r, rf, periods), "Max drawdown": max_drawdown(eq),
           "Calmar": cagr / abs(max_drawdown(eq)), "Skew": stats.skew(r), "Excess kurtosis": stats.kurtosis(r),
           "VaR 99%": hist_var_cvar(r)[0], "CVaR 99%": hist_var_cvar(r)[1]}
    if benchmark is not None:
        b = pd.Series(benchmark).reindex(r.index)
        beta = np.cov(r, b)[0, 1] / b.var(ddof=1)
        active = r - b
        out.update({"Beta": beta, "Alpha (ann.)": (r.mean() - beta * b.mean()) * periods,
                    "Information ratio": active.mean() / active.std(ddof=1) * np.sqrt(periods)})
    return out


def bsm_call(S, K, T, r, q, vol) -> dict:
    """European call price, delta and gamma (Black–Scholes–Merton, continuous dividend yield)."""
    d1 = (np.log(S / K) + (r - q + vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    price = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return {"price": price, "delta": np.exp(-q * T) * stats.norm.cdf(d1),
            "gamma": np.exp(-q * T) * stats.norm.pdf(d1) / (S * vol * np.sqrt(T)),
            "vega": S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)}


def fdr_counts(pvals, alpha=0.05) -> tuple[int, int]:
    """(number significant at alpha, number still significant after Benjamini–Hochberg)."""
    pvals = np.asarray(pvals)
    return int((pvals < alpha).sum()), int((stats.false_discovery_control(pvals, method="bh") < alpha).sum())


def hedge_ratio(y, x) -> float:
    """OLS slope of y on x (Engle–Granger step 1)."""
    return float(np.polyfit(np.asarray(x), np.asarray(y), 1)[0])


def regime_vols(r, prob_stress, threshold=0.5, periods=252) -> tuple[float, float]:
    """Annualized volatility on calm days and on stressed days (prob_stress > threshold)."""
    r = np.asarray(r); stressed = np.asarray(prob_stress) > threshold
    return float(r[~stressed].std(ddof=1) * np.sqrt(periods)), float(r[stressed].std(ddof=1) * np.sqrt(periods))


def mc_drawdown_prob(paths, level=-0.25) -> float:
    """Share of simulated paths whose maximum drawdown is worse than `level`."""
    dd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
    return float((dd < level).mean())
