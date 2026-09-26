"""Instructors only: write golden/vollib.npz (needs `pip install vollib`) and the recorded chain in data/.

The chain is SYNTHETIC: prices come from a known SVI surface with true inputs (r, q) that the learner does not know;
the ib_* columns imitate IB modelGreeks (vega per vol point, theta per day) computed with the broker's inputs.
Run from labs/part06:   python tools/make_data.py
"""
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["P6_SOLUTIONS"] = "1"
from _loader import load                     # noqa: E402
from common import DATA, GOLDEN              # noqa: E402

pr = load("week21_pricing_iv", "pricing")
sp = load("week22_surface_portfolio", "surface_portfolio")


def golden() -> None:
    from vollib.black import black
    from vollib.black_scholes_merton import black_scholes_merton as bsm
    from vollib.black_scholes_merton.greeks import analytical as ga
    from vollib.black_scholes_merton.implied_volatility import implied_volatility
    rows = []
    for K in np.arange(60, 161, 10.0):
        for T in (0.02, 0.25, 1.0, 2.0):
            for r in (0.0, 0.04):
                for q in (0.0, 0.02):
                    for sig in (0.1, 0.3, 0.8):
                        for cp in (1, -1):
                            f = "c" if cp == 1 else "p"
                            args = (f, 100.0, K, T, r, sig, q)
                            p = bsm(*args)
                            try:
                                iv = implied_volatility(p, 100.0, K, T, r, q, f) if p > 1e-6 else np.nan
                            except Exception:
                                iv = np.nan
                            rows.append([K, T, r, q, sig, cp, p, ga.delta(*args), ga.gamma(*args), ga.vega(*args),
                                         ga.theta(*args), ga.rho(*args), iv, black(f, 100.0, K, T, r, sig)])
    a = np.array(rows, dtype=float)
    names = ["K", "T", "r", "q", "sigma", "cp", "price", "delta", "gamma", "vega", "theta", "rho", "iv", "black76"]
    GOLDEN.mkdir(exist_ok=True)
    np.savez_compressed(GOLDEN / "vollib.npz", **{n: a[:, i] for i, n in enumerate(names)})
    from importlib.metadata import version
    (GOLDEN / "VERSION").write_text(f"vollib {version('vollib')}; S = 100; display units "
                                    "(vega per vol point, theta per day, rho per 1%)\n")


def chain() -> None:
    rng = np.random.default_rng(11)
    S, r_true, q_true = 500.0, 0.045, 0.013
    asof = date(2026, 3, 2)
    surface = {30: (0.0012, 0.012, -0.70, 0.02, 0.08), 58: (0.0025, 0.020, -0.65, 0.02, 0.10),
               93: (0.0045, 0.028, -0.60, 0.03, 0.12)}             # ATM ≈ 17–18%, put skew
    by_T = {d / 365: np.array(v) for d, v in surface.items()}
    assert not sp.calendar_violations(np.linspace(-0.3, 0.2, 51), by_T), "surface has calendar arbitrage"
    rows = []
    for dte, params in surface.items():
        exp = asof + timedelta(days=dte)
        T = dte / 365
        F = S * np.exp((r_true - q_true) * T)
        for K in np.arange(420.0, 581.0, 10.0):
            iv = float(np.sqrt(sp.svi_total_var(np.log(K / F), *params) / T))
            for cp in (1, -1):
                mid = float(pr.bsm_price(S, K, T, r_true, q_true, iv, cp))
                half = max(0.01, 0.004 * mid) + 0.02 * abs(np.log(K / S)) * 10
                bid, ask = round(max(mid - half, 0.0), 2), round(mid + half, 2)
                mid_q = (bid + ask) / 2
                ib_iv = pr.implied_vol(mid_q, S, K, T, r_true, q_true, cp)
                g = pr.greeks(S, K, T, r_true, q_true, ib_iv, cp)
                rows.append({"expiry": exp, "strike": K, "cp": cp, "bid": bid, "ask": ask,
                             "oi": int(rng.integers(50, 20000)), "ib_iv": ib_iv, "ib_delta": g["delta"],
                             "ib_gamma": g["gamma"], "ib_vega": g["vega"] / 100, "ib_theta": g["theta"] / 365})
    df = pd.DataFrame(rows)
    stale = rng.choice(len(df), 4, replace=False)
    df.loc[stale, "bid"] = 0.0                           # a few one-sided quotes to filter out
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "spy_chain.csv", index=False, float_format="%.10g")
    (DATA / "spy_chain_meta.json").write_text(json.dumps({"spot": S, "asof": str(asof), "rate_guess": 0.04,
                                                          "note": "synthetic snapshot; ib_* in IB display units"}))


if __name__ == "__main__":
    golden()
    chain()
    print("wrote", sorted(p.name for p in GOLDEN.iterdir()), sorted(p.name for p in DATA.iterdir()))
