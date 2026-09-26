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
    # >>> SOLUTION
    c = chain[chain["bid"] > 0].copy()
    c["mid"] = (c["bid"] + c["ask"]) / 2
    c["T"] = [(e - asof).days / 365 for e in c["expiry"]]
    c["iv"] = [implied_vol(m, S, k, t, r, q, int(cp)) for m, k, t, cp in zip(c["mid"], c["strike"], c["T"], c["cp"])]
    return c.dropna(subset=["iv"])
    # <<< SOLUTION


def smile(c: pd.DataFrame, expiry: date, S: float) -> callable:
    """IV as a function of strike for one expiry, read from OUT-OF-THE-MONEY options (puts below S, calls at/above S),
    linearly interpolated (np.interp, flat beyond the ends)."""
    # >>> SOLUTION
    g = c[c["expiry"] == expiry]
    otm = g[((g["cp"] == -1) & (g["strike"] < S)) | ((g["cp"] == 1) & (g["strike"] >= S))].sort_values("strike")
    ks, ivs = otm["strike"].to_numpy(), otm["iv"].to_numpy()
    return lambda K: float(np.interp(K, ks, ivs))
    # <<< SOLUTION


def pick_strike(c: pd.DataFrame, expiry: date, cp: int, target_delta: float, S: float, r: float, q: float) -> float:
    """Listed strike of that expiry and type whose BSM delta at its own IV is closest to target_delta."""
    # >>> SOLUTION
    g = c[(c["expiry"] == expiry) & (c["cp"] == cp)]
    d = [bsm_greeks(S, k, t, r, q, v, cp)["delta"] for k, t, v in zip(g["strike"], g["T"], g["iv"])]
    return float(g["strike"].to_numpy()[int(np.argmin(np.abs(np.array(d) - target_delta)))])
    # <<< SOLUTION


def build_playbook(c: pd.DataFrame, S: float, r: float, q: float) -> dict:
    """Three structures (expiries sorted ascending: e0, e1, e2):
      'direction'  ob.vertical(+1, long ≈ 0.50Δ call, short ≈ 0.30Δ call) on e1           (bull call spread)
      'range'      iron condor on e0: shorts at the ≈ 0.16Δ put and call, wings 20 points further out
      'hedge'      ob.collar on e2 for 100 shares: long ≈ −0.25Δ put, short ≈ 0.25Δ call
    Each leg's IV comes from smile(). Return {name: OptionStrategy}."""
    # >>> SOLUTION
    e0, e1, e2 = sorted(c["expiry"].unique())
    T = {e: (c.loc[c["expiry"] == e, "T"].iloc[0]) for e in (e0, e1, e2)}
    s0, s1, s2 = smile(c, e0, S), smile(c, e1, S), smile(c, e2, S)
    kl, ks = pick_strike(c, e1, 1, 0.50, S, r, q), pick_strike(c, e1, 1, 0.30, S, r, q)
    kp, kc = pick_strike(c, e0, -1, -0.16, S, r, q), pick_strike(c, e0, 1, 0.16, S, r, q)
    condor = (ob.OptionStrategy("iron condor").add(-1, kp - 20, T[e0], 1, s0(kp - 20)).add(-1, kp, T[e0], -1, s0(kp))
              .add(1, kc, T[e0], -1, s0(kc)).add(1, kc + 20, T[e0], 1, s0(kc + 20)))
    hp, hc = pick_strike(c, e2, -1, -0.25, S, r, q), pick_strike(c, e2, 1, 0.25, S, r, q)
    return {"direction": ob.vertical(1, kl, ks, T[e1], s1), "range": condor,
            "hedge": ob.collar(hp, hc, T[e2], s2)}
    # <<< SOLUTION


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
    # >>> SOLUTION
    base = float(strategy.value(S, r, q))
    return pd.DataFrame([[float(strategy.value(S * (1 + ds), r, q, dvol=dv)) - base for dv in vol_shocks]
                         for ds in spot_shocks], index=list(spot_shocks), columns=list(vol_shocks))
    # <<< SOLUTION


def playbook_report(c: pd.DataFrame, book: dict, S: float, r: float, q: float) -> tuple[str, dict]:
    """Markdown with one '## <name>: <strategy.name>' section per structure containing: the legs, a line
    'cost / breakevens / max profit / max loss / POP', a Greeks line and the scenario grid; plus the combo orders:
    {name: {"combo": ob.combo_prices(...), "ib": ob.to_ib_combo(strategy, conIds 1001.., "SPY"),
            "alpaca": ob.to_alpaca_mleg(strategy, OCC symbols, limit = combo mid)}}. The collar's stock leg is traded
    separately (its combo covers the option legs only)."""
    # >>> SOLUTION
    expiry_by_T = {t: e for e, t in c.groupby("expiry")["T"].first().items()}
    lines, orders, conid = ["# Option playbook", f"Spot {S:.2f}, r {r:.3%}, q {q:.3%}"], {}, 1001
    for name, st in book.items():
        a, g = st.analyze(S, r, q), st.greeks(S, r, q)
        quotes = leg_quotes(c, st, expiry_by_T)
        prices = ob.combo_prices(quotes["qty"], quotes["bid"], quotes["ask"])
        opt_only = ob.OptionStrategy(st.name, [L for L in st.legs if L.cp != 0], st.multiplier)
        ib = ob.to_ib_combo(opt_only, list(range(conid, conid + len(quotes))), "SPY")
        conid += len(quotes)
        alp = ob.to_alpaca_mleg(st, [occ("SPY", e, cp, k) for e, cp, k in zip(quotes["expiry"], quotes["cp"],
                                                                                quotes["strike"])], prices["mid"])
        orders[name] = {"combo": prices, "ib": ib, "alpaca": alp}
        legs = ", ".join(f"{L.qty:+d} {'stock' if L.cp == 0 else ('C' if L.cp == 1 else 'P') + f'{L.K:g}'}"
                         for L in st.legs)
        lines += [f"## {name}: {st.name}", f"Legs: {legs}",
                  f"cost {a['cost']:,.0f} / breakevens {[float(x) for x in a['breakevens']]} / max profit {a['max_profit']:,.0f} / "
                  f"max loss {a['max_loss']:,.0f} / POP {a['pop']:.0%}",
                  "Greeks: " + ", ".join(f"{k} {v:,.1f}" for k, v in g.items()),
                  f"Combo net mid {prices['mid']:.2f}, natural {prices['natural']:.2f} (legging costs "
                  f"{prices['leg_cost']:.2f} per share)",
                  scenario_grid(st, S, r, q).round(0).to_string()]
    return "\n\n".join(lines) + "\n", orders
    # <<< SOLUTION


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
