"""Week 32 (S5–S8) — Multi-asset stat-arb, factors and robustness: PCA residual s-scores, Fama–MacBeth, signal
neutralization, structural breaks and a pair retirement rule, White's Reality Check, parameter ensembles and a
portfolio of pairs with per-name caps.

Relationships die: every estimate here is re-made on a rolling window and checked before it is trusted.
Uses your week 31 engle_granger and pairs_positions.
Fill in every block marked "Your turn", then run:  python -m pytest week32_statarb
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist
from scipy.stats import norm
from statsmodels.stats.diagnostic import breaks_cusumolsresid

from _loader import load

pp = load("week31_pairs", "pairs")


# --------------------------------------------------------------------- S5 PCA residual stat-arb
def s_scores(returns: pd.DataFrame, n_factors: int = 5, window: int = 60, max_half_life: float | None = None
             ) -> pd.Series:
    """Avellaneda–Lee s-scores from the LAST `window` rows (lesson plan S5): standardize the returns, factors =
    Z @ (top n_factors eigenvectors of the correlation matrix, largest eigenvalue first); regress each stock's RAW
    returns on [1, factors] (np.linalg.lstsq); X = cumulative residuals; fit X as AR(1) (b, a); skip the stock if b is
    not in (0, 1) or its half-life ln 2/(−ln b) exceeds max_half_life; s = (X[−1] − a/(1−b)) / σ_eq with
    σ_eq = √(var(AR residuals, ddof=2)/(1 − b²)). Return a Series of s for the kept stocks (columns order)."""
    # >>> SOLUTION
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
    # <<< SOLUTION


def s_score_positions(s: pd.Series, prev: pd.Series, open_: float = 1.25, close_long: float = 0.5,
                      close_short: float = 0.75) -> pd.Series:
    """Avellaneda–Lee rules per stock (index = prev.index): flat → +1 if s < −open_, −1 if s > open_; long → flat if
    s > −close_long; short → flat if s < close_short. A stock with no s-score today (not reverting) is closed."""
    # >>> SOLUTION
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
    # <<< SOLUTION


def residual_backtest(returns: pd.DataFrame, n_factors: int = 3, window: int = 60, max_half_life: float = 10,
                      cost_bps: float = 5.0) -> dict:
    """Daily loop from t = window − 1: s-scores from the `window` rows ending at t (inclusive, decided at the close),
    positions by s_score_positions, DOLLAR-NEUTRAL weights (+0.5 split equally over the longs, −0.5 over the shorts;
    a side with no names gets 0), earned on row t+1, minus cost_bps/1e4 · Σ|Δweights|. Return {"returns": Series indexed by the earning dates,
    "positions": DataFrame of positions decided at each t}."""
    # >>> SOLUTION
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
    # <<< SOLUTION


# ------------------------------------------------------------------ S6 factor models
def newey_west_se(x, lags: int) -> float:
    """HAC standard error of the MEAN of x: √[(γ0 + 2 Σ_{j=1..lags} (1 − j/(lags+1)) γ_j) / n], with
    γ_j = (1/n) Σ (x_t − x̄)(x_{t−j} − x̄) (divide by n, like statsmodels HAC without small-sample correction)."""
    # >>> SOLUTION
    x = np.asarray(x, dtype=float)
    n, d = x.size, x - x.mean()
    s = d @ d / n
    for j in range(1, lags + 1):
        s += 2 * (1 - j / (lags + 1)) * (d[j:] @ d[:-j]) / n
    return float(np.sqrt(s / n))
    # <<< SOLUTION


def fama_macbeth(returns: pd.DataFrame, exposures: dict[str, pd.DataFrame], nw_lags: int = 0) -> pd.DataFrame:
    """Lesson plan S6: for each consecutive pair of dates (t, t+1), regress returns at t+1 on the exposures at t across
    stocks (statsmodels OLS with a constant, rows with NaN dropped); premium = mean of the slopes. t_stat = mean /
    (std(ddof=1)/√n) when nw_lags == 0, else mean / newey_west_se(slopes, nw_lags). Index: const + exposure names."""
    # >>> SOLUTION
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
    # <<< SOLUTION


def neutralize(signal: pd.Series, sectors: pd.Series, beta: pd.Series) -> pd.Series:
    """Residual of a cross-sectional OLS of the signal on one dummy per sector (no extra constant) and beta: what is
    left has zero mean in every sector and zero covariance with beta."""
    # >>> SOLUTION
    D = pd.get_dummies(sectors.loc[signal.index], dtype=float)
    X = pd.concat([D, beta.loc[signal.index].rename("beta")], axis=1)
    return sm.OLS(signal, X).fit().resid
    # <<< SOLUTION


# ------------------------------------------------------ S7 breaks, stability, multiple testing
def chow_test(y, x, break_idx: int) -> tuple[float, float]:
    """Chow test for a break in y = α + βx at a KNOWN index (first row of the second segment), k = 2 parameters:
    F = ((SSR_pooled − SSR_1 − SSR_2)/k) / ((SSR_1 + SSR_2)/(n − 2k)); p from the F(k, n − 2k) distribution."""
    # >>> SOLUTION
    y, x = np.asarray(y, float), np.asarray(x, float)

    def ssr(yy, xx):
        return sm.OLS(yy, sm.add_constant(xx)).fit().ssr

    n, k = y.size, 2
    s1, s2 = ssr(y[:break_idx], x[:break_idx]), ssr(y[break_idx:], x[break_idx:])
    F = ((ssr(y, x) - s1 - s2) / k) / ((s1 + s2) / (n - 2 * k))
    return float(F), float(f_dist.sf(F, k, n - 2 * k))
    # <<< SOLUTION


def cusum_pvalue(y, x) -> float:
    """Break at an UNKNOWN date: OLS y on [1, x], then statsmodels breaks_cusumolsresid(residuals, ddof=2); its
    p-value."""
    # >>> SOLUTION
    resid = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit().resid
    return float(breaks_cusumolsresid(resid, ddof=2)[1])
    # <<< SOLUTION


def rolling_stability(y: pd.Series, x: pd.Series, window: int = 250, step: int = 21) -> pd.DataFrame:
    """Every `step` rows, starting with the first full window, run pp.engle_granger on the last `window` rows.
    Index: the last date of each window. Columns: pvalue, half_life, beta, spread_vol (std of the spread, ddof=1)."""
    # >>> SOLUTION
    rows, idx = [], []
    for end in range(window, len(y) + 1, step):
        eg = pp.engle_granger(y.iloc[end - window:end], x.iloc[end - window:end])
        rows.append({"pvalue": eg["pvalue"], "half_life": eg["half_life"], "beta": eg["beta"],
                     "spread_vol": float(eg["spread"].std(ddof=1))})
        idx.append(y.index[end - 1])
    return pd.DataFrame(rows, index=idx)
    # <<< SOLUTION


def retirement_date(stab: pd.DataFrame, max_p: float = 0.2, max_hl: float = 30, max_beta_drift: float = 0.25,
                    patience: int = 2):
    """Retire the pair at the first check where a rule has been broken on `patience` CONSECUTIVE checks. Broken:
    pvalue > max_p, or half_life > max_hl, or |β/β_first − 1| > max_beta_drift (β_first = the first row's β).
    Return the date, or None if the pair survives."""
    # >>> SOLUTION
    bad = ((stab["pvalue"] > max_p) | (stab["half_life"] > max_hl)
           | ((stab["beta"] / stab["beta"].iloc[0] - 1).abs() > max_beta_drift)).to_numpy()
    run = 0
    for date, b in zip(stab.index, bad):
        run = run + 1 if b else 0
        if run >= patience:
            return date
    return None
    # <<< SOLUTION


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano: start at a random index; at each step continue to the next index (wrapping around) with
    probability 1 − 1/mean_block, else jump to a new random index."""
    # >>> SOLUTION
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    jump = rng.random(n) < 1 / mean_block
    new = rng.integers(0, n, n)
    for t in range(1, n):
        idx[t] = new[t] if jump[t] else (idx[t - 1] + 1) % n
    return idx
    # <<< SOLUTION


def reality_check(perf, n_boot: int = 500, mean_block: float = 10, seed: int = 0) -> dict:
    """White's Reality Check for "is the BEST of K strategies better than zero?". perf: T × K excess returns.
    V = max_k √T·mean_k; each bootstrap (stationary_bootstrap_indices, same rows for all strategies):
    V* = max_k √T·(mean*_k − mean_k). Return {"best": column index of the best mean, "p_value": share of V* >= V,
    "naive_p": one-sided t-test p-value of the best strategy alone (as if it were the only one tried)}."""
    # >>> SOLUTION
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
    # <<< SOLUTION


# ----------------------------------------------------------- S8 ensembles and a book of pairs
def ensemble_positions(z, entries=(1.75, 2.0, 2.25), exits=(0.25, 0.5, 0.75), stop: float = 4.0,
                       max_hold: int | None = None) -> np.ndarray:
    """Average of pp.pairs_positions over every (entry, exit) combination: a fractional position in [−1, 1] that is
    much less sensitive to any single parameter choice."""
    # >>> SOLUTION
    return np.mean([pp.pairs_positions(z, e, x, stop, max_hold) for e in entries for x in exits], axis=0)
    # <<< SOLUTION


def pair_book_weights(pair_returns: pd.DataFrame, pairs: dict[str, tuple[str, str]], name_cap: float = 0.2,
                      max_iter: int = 1000) -> pd.Series:
    """Inverse-volatility weights over the pairs (columns of pair_returns, std ddof=1) summing to 1, with the load of
    every stock (sum of the weights of the pairs it appears in) <= name_cap: repeat {scale each pair by
    min(1, cap/load) over its two names; renormalize to 1} until the max load <= cap·(1 + 1e-9).
    Raise ValueError if that has not happened after max_iter rounds (infeasible cap)."""
    # >>> SOLUTION
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
    # <<< SOLUTION


def book_exposures(pair_pos: pd.Series, pair_w: pd.Series, pairs: dict[str, tuple[str, str]], hedges: pd.Series,
                   betas: pd.Series, sectors: pd.Series) -> dict:
    """Stock-level view of a book of pairs. Pair p with position s (+1 long spread) and capital weight w, hedge h,
    names (a, b): a gets s·w/(1+|h|), b gets −s·w·h/(1+|h|) (a stock can collect weight from several pairs).
    Return {"weights": Series per stock (sorted index, zeros dropped), "gross": Σ|w|, "net": Σw,
    "net_beta": Σ w·beta, "sector_net": Series of Σw per sector (sorted)}."""
    # >>> SOLUTION
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
    # <<< SOLUTION
