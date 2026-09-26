"""Clinic W2 — Option playbook (Part 7, S5–S8).

Pick three structures from the recorded chain (one for direction, one for a range, one hedge), build them with your
week 24 builder at the chain's own implied vols, analyze them (payoff, breakevens, POP, Greeks, spot × vol grid) and
prepare combo orders priced at the net mid (IB BAG and Alpaca multi-leg). Paper-trade them via the Part 4 OMS.
Run:  python playbook.py        Test:  python -m pytest clinic_w2_option_playbook   (needs week24_option_builder)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                        # noqa: E402
from common import bsm_greeks, implied_vol, load_chain          # noqa: E402

ob = load("week24_option_builder", "builder")


def prepare(chain: pd.DataFrame, S: float, asof: date, r: float, q: float) -> pd.DataFrame:
    """Keep two-sided quotes (bid > 0), add mid, T = days/365 and iv = implied_vol(mid) (common.implied_vol);
    drop rows whose IV is NaN. Return a copy."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def smile(c: pd.DataFrame, expiry: date, S: float) -> callable:
    """IV as a function of strike for one expiry, read from OUT-OF-THE-MONEY options (puts below S, calls at/above S),
    linearly interpolated (np.interp, flat beyond the ends)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def pick_strike(c: pd.DataFrame, expiry: date, cp: int, target_delta: float, S: float, r: float, q: float) -> float:
    """Listed strike of that expiry and type whose BSM delta at its own IV is closest to target_delta."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def build_playbook(c: pd.DataFrame, S: float, r: float, q: float) -> dict:
    """Three structures (expiries sorted ascending: e0, e1, e2):
      'direction'  ob.vertical(+1, long ≈ 0.50Δ call, short ≈ 0.30Δ call) on e1           (bull call spread)
      'range'      iron condor on e0: shorts at the ≈ 0.16Δ put and call, wings 20 points further out
      'hedge'      ob.collar on e2 for 100 shares: long ≈ −0.25Δ put, short ≈ 0.25Δ call
    Each leg's IV comes from smile(). Return {name: OptionStrategy}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def leg_quotes(c: pd.DataFrame, strategy, expiry_by_T: dict[float, date]) -> pd.DataFrame:
    """Bid/ask of each OPTION leg (given): rows in leg order with columns qty, bid, ask, strike, cp, expiry."""
    rows = []
    for L in strategy.legs:
        if L.cp == 0:
            continue
        e = expiry_by_T[L.T]
        row = c[(c["expiry"] == e) & (c["strike"] == L.K) & (c["cp"] == L.cp)].iloc[0]
        rows.append({"qty": L.qty, "bid": row["bid"], "ask": row["ask"], "strike": L.K, "cp": L.cp, "expiry": e})
    return pd.DataFrame(rows)


def occ(root: str, expiry: date, cp: int, strike: float) -> str:
    return f"{root}{expiry:%y%m%d}{'C' if cp == 1 else 'P'}{int(round(strike * 1000)):08d}"


def scenario_grid(strategy, S: float, r: float, q: float, spot_shocks=(-0.1, -0.05, 0.0, 0.05, 0.1),
                  vol_shocks=(-0.05, 0.0, 0.05)) -> pd.DataFrame:
    """P&L today (dt = 0) for relative spot shocks (rows) × absolute IV shifts (columns): value(S(1+ds), dvol) − value(S)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def playbook_report(c: pd.DataFrame, book: dict, S: float, r: float, q: float) -> tuple[str, dict]:
    """Markdown with one '## <name>: <strategy.name>' section per structure containing: the legs, a line
    'cost / breakevens / max profit / max loss / POP', a Greeks line and the scenario grid; plus the combo orders:
    {name: {"combo": ob.combo_prices(...), "ib": ob.to_ib_combo(strategy, conIds 1001.., "SPY"),
            "alpaca": ob.to_alpaca_mleg(strategy, OCC symbols, limit = combo mid)}}. The collar's stock leg is traded
    separately (its combo covers the option legs only)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def build() -> tuple[str, dict, dict]:
    """Whole clinic (given)."""
    chain, meta = load_chain()
    S, r, q = meta["spot"], meta["r"], meta["q"]
    c = prepare(chain, S, date.fromisoformat(meta["asof"]), r, q)
    book = build_playbook(c, S, r, q)
    md, orders = playbook_report(c, book, S, r, q)
    return md, orders, book


if __name__ == "__main__":
    print(build()[0])
