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
    # >>> SOLUTION
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))  # noqa: E741
    rets = np.zeros_like(c)
    rets[1:] = c[1:] / c[:-1] - 1
    level = ln.kalman_level(c, 1e-4, 1e-2)
    kal = np.zeros_like(c)
    kal[1:] = np.sign(np.diff(level))
    sig = {
        "momentum": fw.tsmom(c, lookback=60, vol_n=20, target_vol=0.10),
        "range-bound": ln.bollinger_fade(h, l, c),
        "mean reversion": ln.rsi2_reversion(c, 10, 70, trend_n=200, use_trend=False),
        "either-way": fw.donchian_breakout(h, l, c, 20, 10),
        "volatility": ln.vol_target_overlay(np.ones_like(c), rets, target=0.10, n=20),
        "mathematical": kal,
        "statistical": ln.turn_of_month(bars.index, 1, 3).astype(float),
    }
    return {k: np.nan_to_num(np.asarray(v, dtype=float)) for k, v in sig.items()}
    # <<< SOLUTION


def estimate_regimes(bars: pd.DataFrame, adx_n: int = 14, adx_trend: float = 25.0, vol_n: int = 20) -> pd.DataFrame:
    """Causal regime estimates: trend = ADX(adx_n) > adx_trend; high_vol = rolling vol_n-day realized vol (std of log
    returns) above its EXPANDING median up to that day (both known at the close). Columns trend, high_vol (0/1)."""
    # >>> SOLUTION
    c = bars["close"]
    a = adx(bars["high"].to_numpy(), bars["low"].to_numpy(), c.to_numpy(), adx_n)
    rv = np.log(c).diff().rolling(vol_n).std()
    return pd.DataFrame({"trend": (np.nan_to_num(a) > adx_trend).astype(int),
                         "high_vol": (rv > rv.expanding().median()).astype(int)}, index=bars.index)
    # <<< SOLUTION


def regime_names(labels: pd.DataFrame) -> pd.Series:
    """'trend/high vol', 'trend/low vol', 'range/high vol', 'range/low vol' from the 0/1 columns trend, high_vol."""
    # >>> SOLUTION
    t = np.where(labels["trend"] == 1, "trend", "range")
    v = np.where(labels["high_vol"] == 1, "high vol", "low vol")
    return pd.Series([f"{a}/{b}" for a, b in zip(t, v)], index=labels.index)
    # <<< SOLUTION


def regime_table(bars: pd.DataFrame, signals: dict[str, np.ndarray], labels: pd.DataFrame,
                 cost_bps: float = 2.0) -> pd.DataFrame:
    """Annualized Sharpe of each strategy's quick_eval P&L restricted to the bars of each regime (the regime of the bar
    on which the P&L is EARNED). Rows = strategies (dict order), columns = REGIMES; NaN if a regime has < 20 bars or
    no variance."""
    # >>> SOLUTION
    names = regime_names(labels).to_numpy()
    out = pd.DataFrame(index=list(signals), columns=REGIMES, dtype=float)
    for s, pos in signals.items():
        pnl = fw.quick_eval(pos, bars["open"].to_numpy(), cost_bps)["pnl"]
        for reg in REGIMES:
            p = pnl[names == reg]
            out.loc[s, reg] = p.mean() / p.std() * np.sqrt(252) if p.size >= 20 and p.std() > 0 else np.nan
    return out
    # <<< SOLUTION


if __name__ == "__main__":
    bars = regime_market(n_blocks=40, seed=7)
    sig = seven_strategies(bars)
    print("True regimes:\n", regime_table(bars, sig, bars[["trend", "high_vol"]]).round(2), "\n")
    est = estimate_regimes(bars)
    print("Estimated regimes (what you would have known):\n", regime_table(bars, sig, est).round(2))
    print("\nlabel accuracy:", {c: round(float((est[c] == bars[c]).mean()), 3) for c in ("trend", "high_vol")})
