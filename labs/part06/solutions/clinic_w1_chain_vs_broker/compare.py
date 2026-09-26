"""Clinic W1 — Our IV and Greeks vs the broker's model Greeks (Part 6, S2–S4).

"Our Greeks differ from IB's" is usually a difference in INPUTS (price used, rate, dividends, time, model), not a bug.
The recorded chain (data/spy_chain.csv, synthetic) carries IB-style modelGreeks in display units. First compare with
guessed inputs (r = 4%, no dividend); then recover the discount rate and forward from put–call parity and compare
again: the differences should (almost) vanish.
Run:  python compare.py        Test:  python -m pytest clinic_w1_chain_vs_broker   (needs week21 labs)
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
fc = load("week21_futures_chains", "futures_chains")


def prepare(chain: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Add mid = (bid+ask)/2, T = calendar days to expiry / 365, and IB Greeks in LIBRARY units:
    ib_vega_raw = ib_vega × 100, ib_theta_raw = ib_theta × 365. Then drop illiquid rows with
    fc.liquidity_filter(max_spread_pct=0.5). Return the filtered copy."""
    # >>> SOLUTION
    c = chain.copy()
    c["mid"] = (c["bid"] + c["ask"]) / 2
    c["T"] = [(e - asof).days / 365 for e in c["expiry"]]
    c["ib_vega_raw"], c["ib_theta_raw"] = c["ib_vega"] * 100, c["ib_theta"] * 365
    return fc.liquidity_filter(c, max_spread_pct=0.5)
    # <<< SOLUTION


def add_our_greeks(c: pd.DataFrame, S: float, r, q) -> pd.DataFrame:
    """our_iv from the mid (pr.implied_vol_chain), then our_delta, our_gamma and our_vega_raw, our_theta_raw with
    pr.greeks at our_iv. r and q may be scalars or per-row arrays. Return a copy."""
    # >>> SOLUTION
    c = c.copy()
    c["our_iv"] = pr.implied_vol_chain(c["mid"].to_numpy(), S, c["strike"].to_numpy(), c["T"].to_numpy(), r, q,
                                       c["cp"].to_numpy())
    g = pr.greeks(S, c["strike"].to_numpy(), c["T"].to_numpy(), r, q, c["our_iv"].to_numpy(), c["cp"].to_numpy())
    c["our_delta"], c["our_gamma"] = g["delta"], g["gamma"]
    c["our_vega_raw"], c["our_theta_raw"] = g["vega"], g["theta"]
    return c
    # <<< SOLUTION


def difference_table(c: pd.DataFrame, forwards: pd.Series | None = None) -> pd.DataFrame:
    """Per expiry: max absolute difference (ours − IB) of iv, delta, vega_raw and theta_raw, as columns
    d_iv, d_delta, d_vega, d_theta (index = expiry). If `forwards` (expiry → F) is given, use only OUT-OF-THE-MONEY
    rows (calls with K >= F, puts with K < F): deep ITM options have almost no time value, so their IV is
    ill-conditioned and practitioners never read the smile from them."""
    # >>> SOLUTION
    if forwards is not None:
        F = c["expiry"].map(forwards)
        c = c[((c["cp"] == 1) & (c["strike"] >= F)) | ((c["cp"] == -1) & (c["strike"] < F))]
    d = pd.DataFrame({"expiry": c["expiry"], "d_iv": (c["our_iv"] - c["ib_iv"]).abs(),
                      "d_delta": (c["our_delta"] - c["ib_delta"]).abs(),
                      "d_vega": (c["our_vega_raw"] - c["ib_vega_raw"]).abs(),
                      "d_theta": (c["our_theta_raw"] - c["ib_theta_raw"]).abs()})
    return d.groupby("expiry").max()
    # <<< SOLUTION


def implied_rate_forward(strikes, calls, puts, T: float, n_strikes: int = 6) -> tuple[float, float]:
    """Parity says C − P = D·(F − K) with D = e^{−rT}. Take the n_strikes strikes with the smallest |C − P| (nearest
    the money, tightest quotes) and fit C − P = a − D·K by least squares (np.polyfit degree 1). Return
    (r = −ln(D)/T, F = a/D)."""
    # >>> SOLUTION
    K, C, P = (np.asarray(x, dtype=float) for x in (strikes, calls, puts))
    idx = np.argsort(np.abs(C - P))[:n_strikes]
    slope, a = np.polyfit(K[idx], (C - P)[idx], 1)
    D = -slope
    return float(-np.log(D) / T), float(a / D)
    # <<< SOLUTION


def implied_inputs(c: pd.DataFrame, S: float) -> pd.DataFrame:
    """Per expiry: pair calls and puts on the same strike (pivot mid by strike × cp), run implied_rate_forward and
    derive the dividend yield q = r − ln(F/S)/T. Return a DataFrame indexed by expiry with columns T, r, F, q."""
    # >>> SOLUTION
    rows = {}
    for exp, g in c.groupby("expiry"):
        piv = g.pivot_table(index="strike", columns="cp", values="mid").dropna()
        T = float(g["T"].iloc[0])
        r, F = implied_rate_forward(piv.index.to_numpy(), piv[1].to_numpy(), piv[-1].to_numpy(), T)
        rows[exp] = {"T": T, "r": r, "F": F, "q": r - np.log(F / S) / T}
    return pd.DataFrame.from_dict(rows, orient="index")
    # <<< SOLUTION


def compare(guess_r: float = 0.04, guess_q: float = 0.0):
    """Full clinic run (given): differences with guessed inputs, implied inputs, differences with implied inputs
    (all rows), and the same on out-of-the-money rows only."""
    chain, meta = load_chain()
    asof, S = date.fromisoformat(meta["asof"]), float(meta["spot"])
    c = prepare(chain, asof)
    before = difference_table(add_our_greeks(c, S, guess_r, guess_q))
    inputs = implied_inputs(c, S)
    r_row, q_row = c["expiry"].map(inputs["r"]).to_numpy(), c["expiry"].map(inputs["q"]).to_numpy()
    ours = add_our_greeks(c, S, r_row, q_row)
    return before, inputs, difference_table(ours), difference_table(ours, inputs["F"])


if __name__ == "__main__":
    before, inputs, after, after_otm = compare()
    print("Guessed inputs (r = 4%, q = 0):\n", before.round(5), "\n")
    print("Implied from put-call parity:\n", inputs.round(5), "\n")
    print("With implied inputs, all rows:\n", after.round(6), "\n")
    print("With implied inputs, out-of-the-money rows:\n", after_otm.round(6))
