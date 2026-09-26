"""Clinic W2 — The M3a risk report (Part 6, S7–S8).

From the recorded chain: fit an SVI smile per expiry on OUT-OF-THE-MONEY options (using the rate and forward implied by
put–call parity from clinic W1), check butterfly and calendar arbitrage, read ATM IV and the 25-delta risk reversal.
Then report a sample book: dollar Greeks, beta-weighted delta and a full-revaluation spot × vol grid, as Markdown.
Run:  python risk_report.py     Test:  python -m pytest clinic_w2_risk_report   (needs week21, week22 and clinic W1)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                          # noqa: E402
from common import load_chain                     # noqa: E402

pr = load("week21_pricing_iv", "pricing")
sp = load("week22_surface_portfolio", "surface_portfolio")
cm = load("clinic_w1_chain_vs_broker", "compare")


def otm_smile(g: pd.DataFrame, S: float, r: float, q: float, F: float) -> tuple[np.ndarray, np.ndarray]:
    """One expiry: keep OTM rows (calls K >= F, puts K < F), compute IV from mid with pr.implied_vol_chain, drop NaN.
    Return (k = ln(K/F), iv) sorted by k."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def iv_at_delta(params, T: float, F: float, r: float, target: float, cp: int) -> float:
    """IV on the fitted smile at the strike whose Black-76 delta (on the forward: e^{−rT}·cp·N(cp·d1)) is closest to
    `target`: search a grid of 2001 k values in [−0.6, 0.6], each with its own smile IV."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def surface_snapshot(c: pd.DataFrame, S: float, inputs: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Per expiry (index): T, F, a, b, rho, m, s (sp.fit_svi on the OTM smile), rmse (of the fitted IV vs the market
    IV), atm_iv (σ at k = 0), rr25 (25Δ call IV − 25Δ put IV, via iv_at_delta) and butterfly_ok (sp.butterfly_g >= 0 on
    k ∈ [−1, 1]). Also return the calendar violations from sp.calendar_violations on k ∈ [−0.5, 0.5] (51 points)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def sample_book(snap: pd.DataFrame, S: float) -> list:
    """The clinic's sample book (given): SPY options priced off the fitted surface, ES future and ES put, TSLA stock."""
    first = snap.iloc[0]
    iv_at = lambda K: float(np.sqrt(sp.svi_total_var(np.log(K / first["F"]), *first[["a", "b", "rho", "m", "s"]])  # noqa: E731
                                    / first["T"]))
    T = float(first["T"])
    return [
        sp.Position("SPY 30d C510", "option", -20, S, 100, K=510, T=T, sigma=iv_at(510), cp=1, q=0.013),
        sp.Position("SPY 30d P470", "option", -20, S, 100, K=470, T=T, sigma=iv_at(470), cp=-1, q=0.013),
        sp.Position("SPY", "stock", 300, S, 1),
        sp.Position("ESM6", "future", -1, 5020.0, 50),
        sp.Position("ES 60d P4800", "fop", 4, 5020.0, 50, K=4800, T=60 / 365, sigma=0.21, cp=-1),
        sp.Position("TSLA", "stock", 200, 250.0, 1, beta=2.0),
    ]


def to_markdown_table(df: pd.DataFrame, fmt: str = "{:,.0f}") -> str:
    """DataFrame -> Markdown table with the index as the first column (given; avoids the tabulate dependency)."""
    head = "| | " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join(fmt.format(v) if isinstance(v, (int, float, np.floating)) and
                                               not isinstance(v, (bool, np.bool_)) else str(v) for v in row) + " |"
            for i, row in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep, *body])


def risk_report(book: list, spy_price: float, snap: pd.DataFrame, violations: list,
                spot_shocks=(-0.2, -0.1, -0.05, 0.0, 0.05, 0.1), vol_shocks=(-0.05, 0.0, 0.05, 0.15)) -> str:
    """Markdown report with these sections, in order:
    '# M3a risk report' · '## Dollar Greeks' (sp.book_report) · '## Spot × vol scenarios (full revaluation)'
    (sp.scenario_grid; index = spot shock, columns = vol shock) · '## Worst scenario' (one line: shock pair and P&L)
    · '## Volatility surface' (snapshot columns T, atm_iv, rr25, rmse, butterfly_ok, 4 decimals) ·
    '## Arbitrage checks' ('No calendar arbitrage.' or the number of violations). Use to_markdown_table."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def build(asof: date | None = None) -> str:
    """Whole clinic (given)."""
    chain, meta = load_chain()
    asof = asof or date.fromisoformat(meta["asof"])
    S = float(meta["spot"])
    c = cm.prepare(chain, asof)
    inputs = cm.implied_inputs(c, S)
    snap, viol = surface_snapshot(c, S, inputs)
    return risk_report(sample_book(snap, S), S, snap, viol)


if __name__ == "__main__":
    print(build())
