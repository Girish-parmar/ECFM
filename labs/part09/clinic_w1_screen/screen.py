"""Clinic W1 — Screen a sector universe, pick pairs with FDR control, backtest the best 5 out of sample (Part 9, S1–S4).

Formation (the first 750 days): screen within sectors, BH-FDR, half-life filter. Trading (the last 250 days): the
selected pairs with a fixed, a rolling and a Kalman hedge. Compare with a "naive" screen (raw p < 0.05, no FDR, no
half-life filter): its extra pairs are the false discoveries.
Run:  python screen.py         Test:  python -m pytest clinic_w1_screen    (needs week31_pairs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import sector_universe                                                   # noqa: E402

pp = load("week31_pairs", "pairs")
FORMATION = 750


def sharpe(r) -> float:
    """Annualized Sharpe of daily returns (given); 0 when there was no risk."""
    r = np.asarray(r, dtype=float)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0


def screens(prices: pd.DataFrame, sectors: pd.Series, formation: int = FORMATION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run pp.screen_pairs(within sectors) on the FORMATION rows only. Return (selected, naive_only): the rows with
    `selected`, and the rows with raw pvalue < 0.05 that were NOT selected (what a naive screen would add)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def trade_pair(y: pd.Series, x: pd.Series, hedge: str = "fixed", formation: int = FORMATION, cost_bps: float = 5.0,
               borrow_bps_year: float = 50.0, delta: float = 1e-7) -> dict:
    """Out-of-sample P&L of one pair over rows formation.. (all estimates use past data only).
    Formation: hl = pp.engle_granger(formation rows) half-life (capped to [2, 30]); z window = round(3·hl), at least
    10; time stop = round(3·hl) bars. Spreads, over ALL rows, then z = pp.rolling_zscore(spread, window):
      fixed   — spread = y − β x − α with the formation β, α (constant hedge);
      rolling — β_t = pp.rolling_ols_beta(y, x, formation) (NaN warm-up filled with the formation β), spread = y − β_t x;
      kalman  — pp.kalman_hedge(y, x, delta, r = variance of the formation spread); the spread is the forecast error
                e (y_t − α_{t−1} − β_{t−1}x_t: yesterday's hedge, no look-ahead). A large delta lets the hedge chase
                the spread and kills the signal (lesson plan, mistake 8).
    Positions = pp.pairs_positions(z, max_hold=time stop); P&L = pp.pair_pnl with the hedge LAGGED one bar (β_{t−1}).
    Return {"pnl": array over the trading rows, "sharpe", "total" (sum of the daily P&L), "trades" (number of
    entries in the trading rows)}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def oos_report(prices: pd.DataFrame, sectors: pd.Series, top: int = 5, formation: int = FORMATION) -> pd.DataFrame:
    """One row per (group, pair, hedge): group "selected" = the `top` selected pairs by p-value with each hedge
    (fixed, rolling, kalman); group "naive_only" = up to `top` naive-only pairs with the fixed hedge.
    Columns: group, pair ("a/b"), hedge, sharpe, total, trades."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    prices, sectors, truth = sector_universe(n_sectors=5, per_sector=8, n_days=1000, pairs_per_sector=1, seed=4)
    sel, naive = screens(prices, sectors)
    print(f"true pairs: {truth}")
    print("\nselected (formation):\n", sel.round(4).to_string())
    print("\nnaive-only (raw p < 0.05, rejected by FDR or half-life):\n", naive.round(4).to_string())
    rep = oos_report(prices, sectors)
    print("\nout of sample:\n", rep.round(3).to_string())
    print("\nmean OOS Sharpe:\n", rep.groupby(["group", "hedge"])["sharpe"].mean().round(2).to_string())
    for d in (1e-5, 1e-7):
        sr = [trade_pair(prices[a], prices[b], "kalman", delta=d)["sharpe"] for a, b in truth]
        print(f"Kalman delta={d:g}: mean OOS Sharpe on the true pairs {np.mean(sr):.2f}")
