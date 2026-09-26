"""Clinic W2 — A market-neutral book of pairs and its risk report (Part 9, S5–S8, milestone M5b).

Select ≥ 15 pairs in the formation period (FDR), weight them by inverse volatility of their FORMATION P&L with a
per-name cap, trade them out of sample and report what a risk manager asks: gross, net, net beta, sector nets, the
largest single-name load, VaR — and a crowded unwind in which every pair loses at the same time.
Run:  python book.py         Test:  python -m pytest clinic_w2_book    (needs week31_pairs and week32_statarb)
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
sa = load("week32_statarb", "statarb")
FORMATION = 750


def run_pair(y: pd.Series, x: pd.Series, formation: int = FORMATION) -> dict:
    """Fixed-hedge pair over ALL rows (given): formation β, α and half-life; z window and time stop = 3 half-lives."""
    eg = pp.engle_granger(y.iloc[:formation], x.iloc[:formation])
    hl = float(np.clip(eg["half_life"], 2, 30))
    z = pp.rolling_zscore(y - eg["beta"] * x - eg["alpha"], max(10, round(3 * hl)))
    pos = pp.pairs_positions(z, max_hold=round(3 * hl))
    return {"pos": pos, "pnl": pp.pair_pnl(pos, y, x, eg["beta"]), "beta": eg["beta"]}


def market_betas(prices: pd.DataFrame, formation: int = FORMATION) -> pd.Series:
    """β of each stock's daily log return on the equal-weighted average return, FORMATION rows only:
    cov(r_i, m) / var(m) (ddof=1 for both)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def build_book(prices: pd.DataFrame, sectors: pd.Series, formation: int = FORMATION, name_cap: float = 0.15) -> dict:
    """1. Select pairs: pp.screen_pairs(formation rows, sectors), rows with `selected`; name each pair "a/b".
    2. run_pair for each; weights = sa.pair_book_weights(FORMATION P&L of the pairs, pairs, name_cap).
    3. Trading rows (formation..): book return = Σ_p w_p · pnl_p; the position HELD on day t is pos[t−1].
    4. Daily exposures on the trading rows with sa.book_exposures(held positions, weights, pairs, hedges, betas,
       sectors), betas = market_betas(prices, formation).
    Return {"pairs": {name: (a, b)}, "weights": Series, "hedges": Series, "pnl": DataFrame (trading rows × pairs),
    "returns": Series (trading rows), "exposures": DataFrame (trading rows; columns gross, net, net_beta)}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def risk_report(book: dict, k: float = 3.0) -> dict:
    """sharpe (annualized, ddof=1), max_drawdown (of the cumulative SUM of returns, a negative number), n_pairs,
    avg_gross, max_abs_net_beta, max_name_load (largest Σ of weights of the pairs a stock appears in),
    var_99 (historical 1-day, positive = loss: −1st percentile of the returns), crowded_unwind (every pair loses k
    standard deviations of its OWN daily P&L on the same day: k · Σ w_p σ_p, positive = loss), and
    unwind_vs_var (crowded_unwind / var_99)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    prices, sectors, truth = sector_universe(n_sectors=6, per_sector=8, n_days=1000, pairs_per_sector=3, seed=7)
    book = build_book(prices, sectors)
    print(f"{len(book['pairs'])} pairs selected ({len(set(book['pairs'].values()) & set(truth))} of {len(truth)} "
          "true pairs)")
    print("\nweights:\n", book["weights"].round(3).to_string())
    print("\nrisk report:")
    for key, v in risk_report(book).items():
        print(f"  {key:>17}: {v:.4f}" if isinstance(v, float) else f"  {key:>17}: {v}")
