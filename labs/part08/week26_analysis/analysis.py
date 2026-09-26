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
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------ S6 performance
def max_drawdown(returns) -> tuple[float, int]:
    """(max drawdown <= 0 of the compounded equity, longest time under water in bars)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def tear_sheet(returns, periods: int = 252) -> dict[str, float]:
    """cagr, vol (annualized, ddof=1), sharpe (annualized), sortino (mean / downside deviation, where downside
    deviation = √mean(min(r, 0)²), annualized), max_dd, dd_duration, calmar (cagr / |max_dd|), skew, kurtosis
    (excess, scipy defaults), tail_ratio (95th percentile / |5th percentile|)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """Compounded monthly returns as a year × month (1..12) table (NaN for missing months)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def trades_from_positions(pos, open_, high, low) -> pd.DataFrame:
    """Trades of a position series decided at closes (pos[t] held from open[t+1]). A trade starts when the decided
    position changes from 0 to ±1 at t (entry = open[t+1]) and ends when it changes back to 0 (or flips) at u
    (exit = open[u+1]). Columns: entry_i (t+1), exit_i (u+1), direction, ret = direction·(exit/entry − 1),
    mae = worst adverse excursion (≤ 0) and mfe = best favourable excursion (≥ 0) relative to entry, using the
    highs/lows of bars entry_i..exit_i−1. A trade still open at the end is dropped."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def trade_stats(trade_returns) -> dict[str, float]:
    """win_rate, payoff (mean win / |mean loss|), expectancy (mean return), profit_factor (Σ wins / |Σ losses|),
    max_consecutive_losses."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------ S7 statistical significance
def sharpe_se(sr: float, T: int) -> float:
    """Standard error of a per-period Sharpe, i.i.d. normal returns (Lo 2002): √((1 + SR²/2) / T)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def psr(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (per-period units): Φ((SR − SR*)·√(T−1) / √(1 − γ3·SR + (γ4 − 1)/4·SR²)) with
    SR = mean/std(ddof=1), γ3 = skew, γ4 = kurtosis (NOT excess: fisher=False)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def min_track_record(returns, sr_benchmark: float = 0.0, alpha: float = 0.05) -> float:
    """Minimum track record length (periods) for SR to beat sr_benchmark at confidence 1 − alpha:
    1 + (1 − γ3·SR + (γ4 − 1)/4·SR²) · (z_{1−α} / (SR − SR*))². inf if SR <= SR*."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def stationary_bootstrap_sharpes(returns, n_boot: int = 1000, mean_block: int = 20, seed: int = 0) -> np.ndarray:
    """Politis–Romano stationary bootstrap: each resampled series starts at a random index and continues
    consecutively (wrapping around), jumping to a new random index with probability 1/mean_block at every step.
    Return the n_boot per-period Sharpes (mean/std ddof=1) of the resampled series."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def drawdown_distribution(trade_returns, n: int = 2000, seed: int = 0) -> np.ndarray:
    """Monte Carlo trade reshuffling: n random orderings of the same trades; max drawdown of each (≤ 0)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ----------------------------------------------------------------------------- S8 capacity
def net_sharpe_vs_capital(gross_returns, daily_turnover: float, adv_dollars: float, daily_vol: float,
                          capitals, k: float = 1.0, periods: int = 252) -> pd.Series:
    """For each capital C: traded dollars per day = daily_turnover × C; impact cost per day (as a return) =
    daily_turnover × k·σ_daily·√(traded / ADV) (square-root model); net returns = gross − cost.
    Return the annualized net Sharpe per capital (index = capitals)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def capacity(sharpe_by_capital: pd.Series, min_sharpe: float = 0.5) -> float:
    """Largest capital whose net Sharpe is still >= min_sharpe (0 if none)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def implementation_shortfall_bps(decision_price: float, fill_price: float, side: int) -> float:
    """Cost of execution vs the decision price in bps, positive = worse: side × (fill − decision)/decision × 1e4."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
