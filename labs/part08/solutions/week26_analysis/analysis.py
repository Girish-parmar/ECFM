"""Week 26 (S5–S8) — Biases, performance and trade analytics, statistical significance, capacity.

Sharpe ratios are ESTIMATES with error. Per-period (e.g. daily) units are used inside the significance functions;
annualize only for display (× √252).
Fill in every block marked "Your turn", then run:  python -m pytest week26_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


# ------------------------------------------------------------------------- S5 survivorship
def members(membership: pd.DataFrame, date) -> list[str]:
    """Point-in-time universe: rows (symbol, start, end) where start <= date and (end is NaT or date < end).
    Sorted symbols. Using TODAY's members for the whole history is survivorship bias."""
    # >>> SOLUTION
    d = pd.Timestamp(date)
    m = membership[(membership["start"] <= d) & (membership["end"].isna() | (d < membership["end"]))]
    return sorted(m["symbol"])
    # <<< SOLUTION


# ------------------------------------------------------------------------ S6 performance
def max_drawdown(returns) -> tuple[float, int]:
    """(max drawdown <= 0 of the compounded equity, longest time under water in bars)."""
    # >>> SOLUTION
    eq = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1
    longest = run = 0
    for x in dd:
        run = run + 1 if x < 0 else 0
        longest = max(longest, run)
    return float(dd.min()), int(longest)
    # <<< SOLUTION


def tear_sheet(returns, periods: int = 252) -> dict[str, float]:
    """cagr, vol (annualized, ddof=1), sharpe (annualized), sortino (mean / downside deviation, where downside
    deviation = √mean(min(r, 0)²), annualized), max_dd, dd_duration, calmar (cagr / |max_dd|), skew, kurtosis
    (excess, scipy defaults), tail_ratio (95th percentile / |5th percentile|)."""
    # >>> SOLUTION
    r = np.asarray(returns, dtype=float)
    growth = np.prod(1 + r)
    cagr = growth ** (periods / r.size) - 1
    sd = r.std(ddof=1)
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    mdd, dur = max_drawdown(r)
    return {"cagr": cagr, "vol": sd * np.sqrt(periods), "sharpe": r.mean() / sd * np.sqrt(periods),
            "sortino": r.mean() / down * np.sqrt(periods), "max_dd": mdd, "dd_duration": dur,
            "calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "skew": float(skew(r)), "kurtosis": float(kurtosis(r)),
            "tail_ratio": float(np.percentile(r, 95) / abs(np.percentile(r, 5)))}
    # <<< SOLUTION


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """Compounded monthly returns as a year × month (1..12) table (NaN for missing months)."""
    # >>> SOLUTION
    m = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    return m.unstack().reindex(columns=range(1, 13))
    # <<< SOLUTION


def trades_from_positions(pos, open_, high, low) -> pd.DataFrame:
    """Trades of a position series decided at closes (pos[t] held from open[t+1]). A trade starts when the decided
    position changes from 0 to ±1 at t (entry = open[t+1]) and ends when it changes back to 0 (or flips) at u
    (exit = open[u+1]). Columns: entry_i (t+1), exit_i (u+1), direction, ret = direction·(exit/entry − 1),
    mae = worst adverse excursion (≤ 0) and mfe = best favourable excursion (≥ 0) relative to entry, using the
    highs/lows of bars entry_i..exit_i−1. A trade still open at the end is dropped."""
    # >>> SOLUTION
    p = np.sign(np.nan_to_num(np.asarray(pos, dtype=float)))
    rows, cur = [], None
    for t in range(1, p.size):
        if cur is not None and p[t] != cur[1] and t + 1 < p.size:
            i, d = cur
            e_px, x_px = open_[i], open_[t + 1]
            hi, lo = np.max(high[i:t + 1]), np.min(low[i:t + 1])
            fav = (hi / e_px - 1) if d > 0 else (1 - lo / e_px)
            adv = (lo / e_px - 1) if d > 0 else (1 - hi / e_px)
            rows.append({"entry_i": i, "exit_i": t + 1, "direction": int(d), "ret": d * (x_px / e_px - 1),
                         "mae": min(adv, 0.0), "mfe": max(fav, 0.0)})
            cur = None
        if cur is None and p[t] != 0 and p[t] != p[t - 1] and t + 1 < p.size:
            cur = (t + 1, p[t])
    return pd.DataFrame(rows, columns=["entry_i", "exit_i", "direction", "ret", "mae", "mfe"])
    # <<< SOLUTION


def trade_stats(trade_returns) -> dict[str, float]:
    """win_rate, payoff (mean win / |mean loss|), expectancy (mean return), profit_factor (Σ wins / |Σ losses|),
    max_consecutive_losses."""
    # >>> SOLUTION
    r = np.asarray(trade_returns, dtype=float)
    wins, losses = r[r > 0], r[r < 0]
    streak = best = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    return {"win_rate": float((r > 0).mean()), "payoff": float(wins.mean() / abs(losses.mean())),
            "expectancy": float(r.mean()), "profit_factor": float(wins.sum() / abs(losses.sum())),
            "max_consecutive_losses": best}
    # <<< SOLUTION


# ------------------------------------------------------------------ S7 statistical significance
def sharpe_se(sr: float, T: int) -> float:
    """Standard error of a per-period Sharpe, i.i.d. normal returns (Lo 2002): √((1 + SR²/2) / T)."""
    # >>> SOLUTION
    return float(np.sqrt((1 + 0.5 * sr ** 2) / T))
    # <<< SOLUTION


def psr(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (per-period units): Φ((SR − SR*)·√(T−1) / √(1 − γ3·SR + (γ4 − 1)/4·SR²)) with
    SR = mean/std(ddof=1), γ3 = skew, γ4 = kurtosis (NOT excess: fisher=False)."""
    # >>> SOLUTION
    r = np.asarray(returns, dtype=float)
    T = r.size
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))
    # <<< SOLUTION


def min_track_record(returns, sr_benchmark: float = 0.0, alpha: float = 0.05) -> float:
    """Minimum track record length (periods) for SR to beat sr_benchmark at confidence 1 − alpha:
    1 + (1 − γ3·SR + (γ4 − 1)/4·SR²) · (z_{1−α} / (SR − SR*))². inf if SR <= SR*."""
    # >>> SOLUTION
    r = np.asarray(returns, dtype=float)
    sr = r.mean() / r.std(ddof=1)
    if sr <= sr_benchmark:
        return float("inf")
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (norm.ppf(1 - alpha) / (sr - sr_benchmark)) ** 2)
    # <<< SOLUTION


def stationary_bootstrap_sharpes(returns, n_boot: int = 1000, mean_block: int = 20, seed: int = 0) -> np.ndarray:
    """Politis–Romano stationary bootstrap: each resampled series starts at a random index and continues
    consecutively (wrapping around), jumping to a new random index with probability 1/mean_block at every step.
    Return the n_boot per-period Sharpes (mean/std ddof=1) of the resampled series."""
    # >>> SOLUTION
    r = np.asarray(returns, dtype=float)
    T = r.size
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        idx[0] = rng.integers(T)
        jump = rng.random(T) < 1 / mean_block
        new = rng.integers(T, size=T)
        for t in range(1, T):
            idx[t] = new[t] if jump[t] else (idx[t - 1] + 1) % T
        s = r[idx]
        out[b] = s.mean() / s.std(ddof=1)
    return out
    # <<< SOLUTION


def drawdown_distribution(trade_returns, n: int = 2000, seed: int = 0) -> np.ndarray:
    """Monte Carlo trade reshuffling: n random orderings of the same trades; max drawdown of each (≤ 0)."""
    # >>> SOLUTION
    r = np.asarray(trade_returns, dtype=float)
    rng = np.random.default_rng(seed)
    return np.array([max_drawdown(rng.permutation(r))[0] for _ in range(n)])
    # <<< SOLUTION


# ----------------------------------------------------------------------------- S8 capacity
def net_sharpe_vs_capital(gross_returns, daily_turnover: float, adv_dollars: float, daily_vol: float,
                          capitals, k: float = 1.0, periods: int = 252) -> pd.Series:
    """For each capital C: traded dollars per day = daily_turnover × C; impact cost per day (as a return) =
    daily_turnover × k·σ_daily·√(traded / ADV) (square-root model); net returns = gross − cost.
    Return the annualized net Sharpe per capital (index = capitals)."""
    # >>> SOLUTION
    r = np.asarray(gross_returns, dtype=float)
    out = {}
    for C in capitals:
        traded = daily_turnover * C
        cost = daily_turnover * k * daily_vol * np.sqrt(traded / adv_dollars)
        net = r - cost
        out[C] = net.mean() / net.std(ddof=1) * np.sqrt(periods)
    return pd.Series(out)
    # <<< SOLUTION


def capacity(sharpe_by_capital: pd.Series, min_sharpe: float = 0.5) -> float:
    """Largest capital whose net Sharpe is still >= min_sharpe (0 if none)."""
    # >>> SOLUTION
    ok = sharpe_by_capital[sharpe_by_capital >= min_sharpe]
    return float(ok.index.max()) if len(ok) else 0.0
    # <<< SOLUTION


def implementation_shortfall_bps(decision_price: float, fill_price: float, side: int) -> float:
    """Cost of execution vs the decision price in bps, positive = worse: side × (fill − decision)/decision × 1e4."""
    # >>> SOLUTION
    return float(side * (fill_price - decision_price) / decision_price * 1e4)
    # <<< SOLUTION
