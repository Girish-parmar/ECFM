"""Clinic W6 — Portfolio defense: choose the platform's default allocator with OUT-OF-SAMPLE evidence (Part 8, S21–S24).

Six allocators, one walk-forward protocol (1-year training window, monthly rebalance), the same five strategy return
streams. Report OOS Sharpe, volatility, max drawdown and turnover, then recommend by a rule written down in advance.
Run:  python allocators.py      Test:  python -m pytest clinic_w6_allocators   (needs week26_analysis, week30_portfolio)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                          # noqa: E402
from common import strategy_returns               # noqa: E402

an = load("week26_analysis", "analysis")
pf = load("week30_portfolio", "portfolio")


def allocators() -> dict:
    """name -> allocator(train_returns DataFrame) -> weights array (given)."""
    return {
        "1/N": lambda x: np.ones(x.shape[1]) / x.shape[1],
        "inverse vol": lambda x: pf.inverse_vol_weights(x).to_numpy(),
        "min variance (LW, long-only)": lambda x: pf.min_variance_long_only(pf.ledoit_wolf(x), max_weight=0.5),
        "risk parity": lambda x: pf.risk_parity(x.cov().to_numpy()),
        "HRP": lambda x: pf.hrp(x).to_numpy(),
        "min CVaR": lambda x: pf.min_cvar(x.to_numpy(), 0.95, max_weight=0.5),
    }


def compare(returns: pd.DataFrame, train: int = 252, rebalance: int = 21) -> pd.DataFrame:
    """Walk-forward every allocator (pf.walk_forward_allocation). Rows = allocators (dict order); columns sharpe,
    vol, max_dd (an.tear_sheet of the OOS returns) and turnover."""
    # >>> SOLUTION
    rows = {}
    for name, fn in allocators().items():
        res = pf.walk_forward_allocation(returns, fn, train, rebalance)
        ts = an.tear_sheet(res["returns"].to_numpy())
        rows[name] = {"sharpe": ts["sharpe"], "vol": ts["vol"], "max_dd": ts["max_dd"], "turnover": res["turnover"]}
    return pd.DataFrame.from_dict(rows, orient="index")
    # <<< SOLUTION


def recommend(table: pd.DataFrame, max_turnover: float = 0.5, tolerance: float = 0.1) -> str:
    """The rule, fixed IN ADVANCE: among allocators with turnover <= max_turnover, take those whose Sharpe is within
    `tolerance` of the best; of these choose the one with the LOWEST turnover (simplicity breaks ties). Falls back to
    '1/N' if nothing qualifies."""
    # >>> SOLUTION
    ok = table[table["turnover"] <= max_turnover]
    if ok.empty:
        return "1/N"
    near = ok[ok["sharpe"] >= ok["sharpe"].max() - tolerance]
    return str(near["turnover"].idxmin())
    # <<< SOLUTION


if __name__ == "__main__":
    r = strategy_returns(n_days=2520, seed=0)
    tab = compare(r)
    print(tab.round(3))
    print("\nRecommended default allocator:", recommend(tab))
