"""Clinic W1 — The regime map: 7 strategies (one per linear group) × 4 market regimes (Part 7, S1–S4).

Where does each strategy work, and where does it fail? The synthetic market (common.regime_market) has KNOWN regimes,
so you can compare the map built from the true labels with one built from labels estimated from the data itself
(ADX for trend, realized vol vs its own history for volatility), which is all you have in real life.
Run:  python regime_map.py      Test:  python -m pytest clinic_w1_regime_map   (needs both week 23 labs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                    # noqa: E402
from common import adx, regime_market                       # noqa: E402

fw = load("week23_framework_momentum", "framework")
ln = load("week23_linear_groups", "linear")
REGIMES = ["trend/high vol", "trend/low vol", "range/high vol", "range/low vol"]


def seven_strategies(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    """Decided positions of one strategy per group (given parameters; NaN → 0):
      momentum        fw.tsmom(close, lookback=60, vol_n=20, target_vol=0.10)
      range-bound     ln.bollinger_fade(high, low, close)
      mean reversion  ln.rsi2_reversion(close, 10, 70, trend_n=200, use_trend=False)
      either-way      fw.donchian_breakout(high, low, close, 20, 10)
      volatility      ln.vol_target_overlay(ones, daily close-to-close returns, target=0.10, n=20)  (buy & hold, vol-targeted)
      mathematical    sign of the one-bar change of ln.kalman_level(close, q=1e-4, r=1e-2)  (0 on bar 0)
      statistical     ln.turn_of_month(index, 1, 3) as 0/1 long
    Keys exactly: momentum, range-bound, mean reversion, either-way, volatility, mathematical, statistical."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def estimate_regimes(bars: pd.DataFrame, adx_n: int = 14, adx_trend: float = 25.0, vol_n: int = 20) -> pd.DataFrame:
    """Causal regime estimates: trend = ADX(adx_n) > adx_trend; high_vol = rolling vol_n-day realized vol (std of log
    returns) above its EXPANDING median up to that day (both known at the close). Columns trend, high_vol (0/1)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def regime_names(labels: pd.DataFrame) -> pd.Series:
    """'trend/high vol', 'trend/low vol', 'range/high vol', 'range/low vol' from the 0/1 columns trend, high_vol."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def regime_table(bars: pd.DataFrame, signals: dict[str, np.ndarray], labels: pd.DataFrame,
                 cost_bps: float = 2.0) -> pd.DataFrame:
    """Annualized Sharpe of each strategy's quick_eval P&L restricted to the bars of each regime (the regime of the bar
    on which the P&L is EARNED). Rows = strategies (dict order), columns = REGIMES; NaN if a regime has < 20 bars or
    no variance."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    bars = regime_market(n_blocks=40, seed=7)
    sig = seven_strategies(bars)
    print("True regimes:\n", regime_table(bars, sig, bars[["trend", "high_vol"]]).round(2), "\n")
    est = estimate_regimes(bars)
    print("Estimated regimes (what you would have known):\n", regime_table(bars, sig, est).round(2))
    print("\nlabel accuracy:", {c: round(float((est[c] == bars[c]).mean()), 3) for c in ("trend", "high_vol")})
