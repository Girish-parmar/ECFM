"""Clinic W2 — Edge study of candlestick patterns across a universe with Benjamini–Hochberg FDR (Part 5, S4–S5).

The skill being taught is the TESTING METHOD, not belief in patterns: on the synthetic universe no pattern has a real
edge, so an honest study finds (almost) nothing, while a pattern that secretly peeks at the future looks spectacular.
Run on your own Part 4 bars for the graded report (≥ 50 symbols, daily bars).
Run:  python edge_study.py        Test:  python -m pytest clinic_w2_edge_study   (needs week17_core and week18_groups)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                  # noqa: E402
from common import arrays, synthetic_universe             # noqa: E402

core = load("week17_core", "core")
gr = load("week18_groups", "groups")

PATTERNS = {
    "engulfing": core.engulfing,
    "hammer": core.hammer,
    "star": core.morning_evening_star,
}


def lookahead_cheat(o, h, l, c):  # noqa: E741
    """A 'pattern' with a hidden bug: it uses the open two bars AHEAD (given, for the demonstration)."""
    out = np.zeros(c.shape, dtype=np.int8)
    out[:-2] = np.where(o[2:] > o[1:-1] * 1.005, 1, 0)
    return out


def edge_table(universe: dict[str, pd.DataFrame], patterns: dict, horizons=(5,), n_perm: int = 1000,
               alpha: float = 0.10) -> pd.DataFrame:
    """One row per (pattern, side, symbol, horizon) that has at least one event, where side 'bull' tests signal == +1
    and 'bear' tests signal == −1. Columns: pattern, side, symbol, horizon, n, edge, p_value, hit_rate (from
    gr.pattern_edge with seed=0), then q_value = gr.bh_adjust over ALL rows and discovery = q_value <= alpha.
    For a bear row the edge is expected to be NEGATIVE; keep the sign as computed."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per (pattern, side): tests (rows), events (sum n), mean_edge (event-weighted mean of edge), share_p05 (share
    of rows with raw p < 0.05) and discoveries (sum). Sorted by pattern, side."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    uni = synthetic_universe(20, 1000)
    t = edge_table(uni, {**PATTERNS, "lookahead_cheat": lookahead_cheat}, horizons=(1, 5))
    print(summary(t).to_string(index=False))
    print(f"\n{len(t)} tests, {int(t['discovery'].sum())} discoveries at FDR 10%")
