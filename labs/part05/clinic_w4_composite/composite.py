"""Clinic W4 — Daily sentiment composite and stress-test report (Part 5, S13–S15).

Components (VIX percentile, term-structure ratio, breadth, COT, news index...) arrive with different scales. Each is
z-scored against ITS OWN PAST ONLY, sign-aligned so that high = calm/bullish, averaged into one composite, labelled
with a hysteresis regime, and related to forward returns by quintile. The stress report replays positions through
historical one-day index moves and hypothetical shocks.
Run:  python composite.py      Test:  python -m pytest clinic_w4_composite     (needs week20_sentiment_tail)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                    # noqa: E402

st = load("week20_sentiment_tail", "sentiment_tail")

# S&P 500 one-day close-to-close moves on well-known stress days (approximate; check against your own data).
HISTORICAL_SHOCKS = {
    "1987-10-19 Black Monday": -0.2047,
    "2008-10-15 Lehman aftermath": -0.0903,
    "2010-05-06 Flash crash (close)": -0.0324,
    "2015-08-24 China devaluation": -0.0394,
    "2018-02-05 Volmageddon": -0.0410,
    "2020-03-16 COVID": -0.1198,
    "2022-09-13 CPI / rates shock": -0.0432,
}
HYPOTHETICAL_SHOCKS = {"Equity gap -10%": -0.10, "Equity gap -20%": -0.20, "Rally +5%": 0.05}


def rolling_z(x: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """z[t] = (x[t] − mean) / std over the trailing window ENDING AT t (rolling(window, min_periods)), sample std.
    NaN where std is 0 or there are fewer than min_periods observations. Uses no future data."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def composite(components: pd.DataFrame, signs: dict[str, int], window: int = 252, min_periods: int = 60) -> pd.Series:
    """Composite = row mean of sign × rolling_z(component) over the components that are available that day
    (skip NaN); NaN when none is available. `signs` gives +1 if a high value is bullish/calm, −1 if it is fearful
    (e.g. VIX percentile −1, breadth +1)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def quintile_table(score: pd.Series, fwd_returns: pd.Series) -> pd.DataFrame:
    """Rows where both are available; bucket the score into quintiles 1..5 (pd.qcut, labels 1..5).
    Return a DataFrame indexed by quintile with columns count, mean_fwd, hit_rate (share of fwd > 0)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def stress_report(positions: dict[str, tuple[float, float]], scenarios: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per scenario (default: HISTORICAL_SHOCKS then HYPOTHETICAL_SHOCKS) with the shock and the P&L of each
    position plus TOTAL (use st.scenario_pnl). Columns: shock, <symbols...>, TOTAL. Sorted by TOTAL ascending (worst
    first)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def demo_components(n: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic components and SPY-like forward 20-day returns with a weak built-in link (given, for the demo)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n, tz="UTC")
    latent = pd.Series(rng.normal(size=n)).rolling(20, min_periods=1).mean().to_numpy()
    comp = pd.DataFrame({
        "vix_pct": 0.5 - 0.3 * latent + rng.normal(0, 0.2, n),
        "breadth": 0.5 + 0.3 * latent + rng.normal(0, 0.2, n),
        "cot_index": np.where(rng.random(n) < 0.8, np.nan, 50 + 20 * latent + rng.normal(0, 10, n)),
    }, index=idx)
    comp["cot_index"] = comp["cot_index"].ffill()
    fwd = pd.Series(0.004 + 0.04 * latent + rng.normal(0, 0.03, n), index=idx)
    return comp, fwd


if __name__ == "__main__":
    comp, fwd = demo_components()
    score = composite(comp, {"vix_pct": -1, "breadth": 1, "cot_index": 1})
    print(quintile_table(score, fwd).round(4))
    reg = st.regime_with_hysteresis(score.to_numpy(), 1.0, 0.5)
    print("regime days:", dict(zip(*np.unique(reg, return_counts=True))))
    rep = stress_report({"SPY": (100_000, 1.0), "QQQ": (50_000, 1.2), "TLT": (40_000, -0.3)})
    print(rep.round({"shock": 4, "SPY": 0, "QQQ": 0, "TLT": 0, "TOTAL": 0}))
