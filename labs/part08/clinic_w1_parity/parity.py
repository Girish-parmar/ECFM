"""Clinic W1 — Port three Part 7 strategies to the engine: parity report and cost sensitivity (Part 8, S1–S4).

The same strategy must agree between the first-look evaluator and the event-driven engine; every difference must be
explainable (fill timing, share rounding, weight drift, costs). Then: how fast does each edge disappear with costs?
Run:  python parity.py         Test:  python -m pytest clinic_w1_parity    (needs week25_engine and week26_analysis)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import quick_eval_pnl, regime_market, rsi2_signal, sma_cross_signal, tsmom_signal   # noqa: E402

en = load("week25_engine", "engine")
an = load("week26_analysis", "analysis")


def signals(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three Part 7 strategies (given): momentum, trend, mean reversion."""
    c = bars["close"].to_numpy()
    return {"tsmom": tsmom_signal(c, 120, 20), "sma_cross": sma_cross_signal(c, 20, 100), "rsi2": rsi2_signal(c)}


def engine_returns(bars: pd.DataFrame, decided, slippage_bps: float = 0.0, commission=None) -> np.ndarray:
    """Open-to-open returns of the engine's equity_open, aligned like quick_eval P&L (r[t] = eq_open[t+1]/eq_open[t]
    − 1, last = 0)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def parity_report(bars: pd.DataFrame, sigs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per strategy, no costs: sharpe_quick, sharpe_engine (an.tear_sheet), corr (of the two daily return series),
    total_quick, total_engine (compounded − 1). Index = strategy."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def cost_sensitivity(bars: pd.DataFrame, sigs: dict[str, np.ndarray], base_slippage_bps: float = 5.0,
                     multiples=(0, 1, 2, 3)) -> pd.DataFrame:
    """Engine Sharpe per strategy (rows) at slippage = multiple × base_slippage_bps plus the IB fixed commission
    (none at multiple 0). Columns = multiples."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    bars = regime_market(n_blocks=24, seed=5)
    sigs = signals(bars)
    print("Parity (no costs):\n", parity_report(bars, sigs).round(3), "\n")
    print("Engine Sharpe vs cost multiple (5 bps slippage + IB commission per unit):\n",
          cost_sensitivity(bars, sigs).round(2))
