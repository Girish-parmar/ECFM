"""Build the Part 6 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part06:  python tools/build_notebooks.py
Cells are ("md", text), ("code", code) or ("ex", starter_code, solution_code).
Edit the content here and rebuild, so starter and solution notebooks never drift apart.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
KERNEL = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}

SETUP = """import sys
from pathlib import Path
for d in (Path.cwd(), Path.cwd().parent):       # p6lib.py is in notebooks/part06/
    sys.path.insert(0, str(d))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p6lib as p

p.use_course_style()"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 6 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_06_FUTURES_OPTIONS_ENGINEERING.md) · graded labs in [`labs/part06/`](../../labs/part06/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.
All data is synthetic with known parameters, so every estimate can be compared with the truth. Units: T in years, σ as a decimal, vega per 1.00 σ, theta per year, `cp = +1` call / `−1` put.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
NB["01_futures_continuous"] = [
    header("01", "Futures: fair value, rolls and continuous series", "S1 (Futures library)",
           "1. Price a future from spot and carry, and read the carry back from two contract months.\n"
           "2. Stitch contracts into a continuous series three ways, and see what each preserves.\n"
           "3. Keep orders on the real contract month, never on an adjusted price."),
    ("code", SETUP),
    ("md", "## 1. A futures strip\n\n"
           "Quarterly index futures (March, June, September, December; expiry on the third Friday), each priced as spot × e^{carry·T} with "
           "a known carry of **3%** a year, a market in contango. Each contract trades for about six months before it expires."),
    ("code", """prices, spot, expiries, rolls = p.futures_panel()
ax = prices.plot(figsize=(11, 4), lw=1, legend=False)
spot.plot(ax=ax, color="black", lw=0.8, label="spot")
ax.set_title("Each contract converges to spot at its expiry"); plt.show()
print("roll dates (8 business days before each expiry):", [d.date().isoformat() for d in rolls[:4]], "…")"""),
    ("md", "## 2. Fair value and implied carry\n\n"
           "Cost of carry: `F = S·e^{(r−q)T}`. Turn it around and two contract months on the same day tell you the carry the market is "
           "pricing: `(r − q) = ln(F_far / F_near) / (T_far − T_near)`."),
    ("md", YOUR_TURN),
    ("ex", """def fair_value(S, r, q, T):
    return ...                                    # ✍️

def implied_carry(f_near, f_far, t_near, t_far):
    return ...                                    # ✍️

day = pd.Timestamp("2024-04-15")
near, far = prices.loc[day].dropna().index[:2]
t = [(e - day.date()).days / 365 for e in expiries if f"ES{p.MONTH_CODES[e.month - 1]}{str(e.year)[-1]}" in (near, far)]
mine = [fair_value(5000.0, 0.05, 0.015, 0.25), implied_carry(prices.at[day, near], prices.at[day, far], t[0], t[1])]
mine = p.check("fair value and carry", mine, [p.fair_value(5000.0, 0.05, 0.015, 0.25),
                                             p.implied_carry(prices.at[day, near], prices.at[day, far], t[0], t[1])])
print(f"F = {mine[0]:.2f};  carry implied by {near}/{far} on {day.date()}: {mine[1]:.2%} (true 3.00%)")""",
     """def fair_value(S, r, q, T):
    return S * np.exp((r - q) * T)

def implied_carry(f_near, f_far, t_near, t_far):
    return np.log(f_far / f_near) / (t_far - t_near)

day = pd.Timestamp("2024-04-15")
near, far = prices.loc[day].dropna().index[:2]
t = [(e - day.date()).days / 365 for e in expiries if f"ES{p.MONTH_CODES[e.month - 1]}{str(e.year)[-1]}" in (near, far)]
mine = [fair_value(5000.0, 0.05, 0.015, 0.25), implied_carry(prices.at[day, near], prices.at[day, far], t[0], t[1])]
mine = p.check("fair value and carry", mine, [p.fair_value(5000.0, 0.05, 0.015, 0.25),
                                             p.implied_carry(prices.at[day, near], prices.at[day, far], t[0], t[1])])
print(f"F = {mine[0]:.2f};  carry implied by {near}/{far} on {day.date()}: {mine[1]:.2%} (true 3.00%)")"""),
    ("md", "## 3. One series from many contracts\n\n"
           "Indicators need one long series. Holding the front contract and switching at each roll date (the **unadjusted** series) "
           "creates a jump at every roll: the next contract trades higher in contango. **Back-adjusting** removes it by shifting all history "
           "*before* each roll:\n"
           "* **difference:** add `new − old` (both on the roll date) to every earlier price;\n"
           "* **ratio:** multiply every earlier price by `new / old`.\n\n"
           "The latest contract's prices never change. `rolls[i]` is the day the strategy switches from column `i` to column `i + 1`."),
    ("md", YOUR_TURN),
    ("ex", """def back_adjust_difference(prices, rolls):
    out = p.continuous(prices, rolls, "none")     # the unadjusted series
    cols = list(prices.columns)
    for i, d in enumerate(rolls):
        gap = ...                                 # ✍️ new contract minus old contract, on the roll date
        out[out.index < d] += gap
    return out

mine = p.attempt(back_adjust_difference, prices, rolls)
mine = p.check("difference back-adjustment", mine, p.continuous(prices, rolls, "difference"))""",
     """def back_adjust_difference(prices, rolls):
    out = p.continuous(prices, rolls, "none")     # the unadjusted series
    cols = list(prices.columns)
    for i, d in enumerate(rolls):
        gap = prices.at[d, cols[i + 1]] - prices.at[d, cols[i]]
        out[out.index < d] += gap
    return out

mine = p.attempt(back_adjust_difference, prices, rolls)
mine = p.check("difference back-adjustment", mine, p.continuous(prices, rolls, "difference"))"""),
    ("code", """series = {m: p.continuous(prices, rolls, m) for m in ("none", "difference", "ratio")}
fig, ax = plt.subplots(figsize=(11, 4))
for m, s in series.items():
    ax.plot(s.index, s, lw=1, label=m)
for d in rolls:
    ax.axvline(d, color="#e6e5e0", lw=1)
ax.set_title("Unadjusted vs back-adjusted continuous series (grey lines: rolls)"); ax.legend(); plt.show()

roll_days = pd.DatetimeIndex(rolls)
rows = {}
for m, s in series.items():
    pts = s.diff()
    rows[m] = {"mean change on roll days": pts.loc[roll_days].mean(),
               "mean change on other days": pts.drop(roll_days).mean(),
               "total change over the sample": s.iloc[-1] - s.iloc[0],
               "first price": s.iloc[0]}
display(pd.DataFrame(rows).T.round(2))
gaps = [prices.at[d, prices.columns[i + 1]] - prices.at[d, prices.columns[i]] for i, d in enumerate(rolls)]
print(f"contango gaps at the {len(rolls)} rolls: {np.round(gaps, 2)} → {sum(gaps):.0f} points in total")"""),
    ("md", "Unadjusted, every roll day adds the contango gap to the series: about +30 points on top of the day's real move. Nobody earned it, "
           "yet a return, momentum or breakout calculation counts it, and over the sample the unadjusted series overstates the P&L of "
           "actually holding the front contract by the sum of the gaps. "
           "Difference adjustment keeps **point changes** exactly (the P&L of one contract), ratio adjustment keeps **percentage returns**. "
           "Both change the level of old prices, so an adjusted price is never a price you can trade.\n\n"
           "## 4. Orders go to the real contract\n\n"
           "Signals come from the adjusted series; orders must name the contract month actually held on that day. With `rolls` sorted, the "
           "held column index is the number of roll dates on or before the day."),
    ("md", YOUR_TURN),
    ("ex", """def held_contract(day, prices, rolls):
    i = ...                                       # ✍️ how many roll dates are on or before `day`
    return prices.columns[i]

days = [pd.Timestamp(d) for d in ("2024-01-10", "2024-03-05", "2024-03-06", "2024-12-31", "2025-11-28")]
mine = [p.attempt(held_contract, d, prices, rolls) for d in days]
expected = [prices.columns[int(np.searchsorted(pd.DatetimeIndex(rolls), d, side="right"))] for d in days]
mine = p.check("held_contract", mine, expected)
list(zip([d.date().isoformat() for d in days], mine))""",
     """def held_contract(day, prices, rolls):
    i = sum(day >= r for r in rolls)
    return prices.columns[i]

days = [pd.Timestamp(d) for d in ("2024-01-10", "2024-03-05", "2024-03-06", "2024-12-31", "2025-11-28")]
mine = [p.attempt(held_contract, d, prices, rolls) for d in days]
expected = [prices.columns[int(np.searchsorted(pd.DatetimeIndex(rolls), d, side="right"))] for d in days]
mine = p.check("held_contract", mine, expected)
list(zip([d.date().isoformat() for d in days], mine))"""),
    ("md", "## Wrap-up\n\n"
           "* Fair value from carry, and carry from the curve.\n"
           "* Signals on a back-adjusted series (difference for point P&L, ratio for returns); orders on the actual contract month.\n"
           "* Graded version: `labs/part06/week21_futures_chains` (roll rules including volume crossover, all three continuous methods, tested for what each preserves)."),
]

# ---------------------------------------------------------------------------------------------- 02
NB["02_option_chain"] = [
    header("02", "Option chains and strike selection", "S2 (Option chain & strike library)",
           "1. Build OCC option symbols.\n"
           "2. Filter a chain to quotes you could actually trade.\n"
           "3. Pick the at-the-money strike on the **forward**, not on spot.\n"
           "4. Find the strikes one expected move away."),
    ("code", SETUP + "\nfrom datetime import date"),
    ("md", "## 1. A chain\n\n"
           "One expiry, 30 days out, spot 600, rate 4.5%, dividend yield 1.3%. Prices come from a known smile (`iv_true`) with bid/ask spreads "
           "that widen in the wings, zero bids far out, and open interest concentrated near the money."),
    ("code", """S, T, r, q = 600.0, 30 / 365, 0.045, 0.013
chain = p.synthetic_chain(S, T, r, q)
calls, puts = chain[chain.cp == 1], chain[chain.cp == -1]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(calls.strike, calls.mid, label="calls"); axes[0].plot(puts.strike, puts.mid, label="puts"); axes[0].set_title("Mid prices"); axes[0].legend()
axes[1].bar(calls.strike, calls.oi, width=4); axes[1].set_title("Open interest (calls)")
plt.tight_layout(); plt.show()
chain.head()"""),
    ("md", "## 2. OCC symbols\n\n"
           "Every listed US option has a 21-character-style OCC symbol: **root + YYMMDD + C/P + strike × 1000 as 8 digits**, "
           "e.g. `AAPL261218C00200000` is the AAPL 18 Dec 2026 200 call. (Brokers may pad the root to six characters; we don't.)"),
    ("md", YOUR_TURN),
    ("ex", """def occ_symbol(root, expiry, cp, strike):
    return ...                                    # ✍️ f-string: {expiry:%y%m%d} and {int(round(strike * 1000)):08d} help

cases = [("AAPL", date(2026, 12, 18), "C", 200.0), ("SPY", date(2025, 3, 21), "P", 572.5), ("QQQ", date(2025, 1, 17), "C", 0.5)]
mine = [occ_symbol(*c) for c in cases]
mine = p.check("occ_symbol", mine, [p.occ_symbol(*c) for c in cases])
[(m, p.parse_occ(m)) for m in mine]""",
     """def occ_symbol(root, expiry, cp, strike):
    return f"{root}{expiry:%y%m%d}{cp}{int(round(strike * 1000)):08d}"

cases = [("AAPL", date(2026, 12, 18), "C", 200.0), ("SPY", date(2025, 3, 21), "P", 572.5), ("QQQ", date(2025, 1, 17), "C", 0.5)]
mine = [occ_symbol(*c) for c in cases]
mine = p.check("occ_symbol", mine, [p.occ_symbol(*c) for c in cases])
[(m, p.parse_occ(m)) for m in mine]"""),
    ("md", "## 3. Quotes you can trade\n\n"
           "Keep a row only if the **bid is above zero**, the ask is at least the bid, the relative spread `(ask − bid)/mid` is at most "
           "`max_spread_pct`, and open interest is at least `min_oi`. Everything downstream (implied vols, the smile, strike selection) "
           "should use the filtered chain."),
    ("md", YOUR_TURN),
    ("ex", """def liquidity_filter(chain, max_spread_pct=0.10, min_oi=100):
    spread = (chain["ask"] - chain["bid"]) / chain["mid"]
    keep = ...                                    # ✍️ the four conditions, combined with &
    return chain[keep]

mine = p.attempt(liquidity_filter, chain)
mine = p.check("liquidity_filter", mine, p.liquidity_filter(chain))
print(f"kept {len(mine)} of {len(chain)} quotes; strikes kept: {mine.strike.min():.0f}–{mine.strike.max():.0f}")""",
     """def liquidity_filter(chain, max_spread_pct=0.10, min_oi=100):
    spread = (chain["ask"] - chain["bid"]) / chain["mid"]
    keep = (chain["bid"] > 0) & (chain["ask"] >= chain["bid"]) & (spread <= max_spread_pct) & (chain["oi"] >= min_oi)
    return chain[keep]

mine = p.attempt(liquidity_filter, chain)
mine = p.check("liquidity_filter", mine, p.liquidity_filter(chain))
print(f"kept {len(mine)} of {len(chain)} quotes; strikes kept: {mine.strike.min():.0f}–{mine.strike.max():.0f}")"""),
    ("md", "## 4. At the money means at the forward\n\n"
           "With a positive carry (r > q) the forward is above spot, and the strike where calls and puts are worth the same is near the "
           "**forward**. Choose the listed strike nearest to it (ties go to the lower strike)."),
    ("md", YOUR_TURN),
    ("ex", """def atm_strike(strikes, forward):
    k = np.sort(np.asarray(strikes, dtype=float))
    return ...                                    # ✍️ the strike with the smallest |k − forward| (argmin picks the first, i.e. lower, on a tie)

F = p.fair_value(S, r, q, T)
strikes = calls.strike.to_numpy()
cases = [(strikes, F), (strikes, S), (strikes, 602.5), (np.arange(90, 111, 1.0), 100.5)]
mine = [atm_strike(k, f) for k, f in cases]
mine = p.check("atm_strike", mine, [p.atm_strike(k, f) for k, f in cases])
print(f"30 days: forward {F:.2f} → ATM strike {mine[0]:.0f}; spot {S:.0f} → {mine[1]:.0f} (the same here)")
F1y = p.fair_value(S, r, q, 1.0)
print(f"1 year:  forward {F1y:.2f} → ATM strike {p.atm_strike(np.arange(400, 801, 5.0), F1y):.0f}; spot → 600: a whole 20 points off")""",
     """def atm_strike(strikes, forward):
    k = np.sort(np.asarray(strikes, dtype=float))
    return float(k[np.argmin(np.abs(k - forward))])

F = p.fair_value(S, r, q, T)
strikes = calls.strike.to_numpy()
cases = [(strikes, F), (strikes, S), (strikes, 602.5), (np.arange(90, 111, 1.0), 100.5)]
mine = [atm_strike(k, f) for k, f in cases]
mine = p.check("atm_strike", mine, [p.atm_strike(k, f) for k, f in cases])
print(f"30 days: forward {F:.2f} → ATM strike {mine[0]:.0f}; spot {S:.0f} → {mine[1]:.0f} (the same here)")
F1y = p.fair_value(S, r, q, 1.0)
print(f"1 year:  forward {F1y:.2f} → ATM strike {p.atm_strike(np.arange(400, 801, 5.0), F1y):.0f}; spot → 600: a whole 20 points off")"""),
    ("code", """c_at = calls.set_index("strike").model_mid; p_at = puts.set_index("strike").model_mid
for k in (600.0, 605.0):
    print(f"K = {k:.0f}: call − put = {c_at[k] - p_at[k]:+.3f}")
print("the strike where call − put changes sign is the forward: that's where 'at the money' is")"""),
    ("md", "## 5. One expected move\n\n"
           "A common strike rule: sell the strikes one **expected move** away, `S·(1 ± σ√T)` with the ATM implied vol."),
    ("code", """atm_iv = float(calls.set_index("strike").iv_true[p.atm_strike(strikes, F)])
lo, hi = p.expected_move_strikes(strikes, S, atm_iv, T)
print(f"ATM IV {atm_iv:.1%} → expected move ±{S * atm_iv * np.sqrt(T):.1f} → strikes {lo:.0f} / {hi:.0f}")"""),
    ("md", "## Wrap-up\n\n"
           "* OCC symbols identify contracts across brokers.\n"
           "* Filter before you compute anything; stale and one-sided quotes make holes in the smile.\n"
           "* ATM and moneyness are measured against the forward.\n"
           "* Graded version: `labs/part06/week21_futures_chains` (IB and Alpaca chains into one schema, expiry selection, strikes by delta)."),
]

# ---------------------------------------------------------------------------------------------- 03
NB["03_pricing_and_greeks"] = [
    header("03", "BSM, Black-76 and first-order Greeks", "S3 (BSM, Black-76 & first-order Greeks)",
           "1. Write the Black–Scholes–Merton price with a dividend yield.\n"
           "2. Write delta and vega, and check them against the curves.\n"
           "3. Convert raw Greeks to the units a broker shows.\n"
           "4. Price an option on a future with Black-76."),
    ("code", SETUP),
    ("md", "## 1. The BSM price\n\n"
           "With a continuous dividend yield `q`: `V = cp·(S e^{−qT} N(cp·d1) − K e^{−rT} N(cp·d2))`, where `cp = +1` for a call, `−1` for a put, "
           "and `p.d1d2` gives `d1 = [ln(S/K) + (r − q + σ²/2)T]/(σ√T)`, `d2 = d1 − σ√T`. `p.N` is the normal CDF. Arrays should work too."),
    ("md", YOUR_TURN),
    ("ex", """def bsm_price(S, K, T, r, q, sigma, cp):
    d1, d2 = p.d1d2(S, K, T, r, q, sigma)
    return ...                                    # ✍️

K = np.array([540.0, 600.0, 660.0])
mine = [bsm_price(600.0, K, 30 / 365, 0.045, 0.013, 0.18, 1), bsm_price(600.0, K, 30 / 365, 0.045, 0.013, 0.18, -1)]
mine = p.check("bsm_price", mine, [p.bsm_price(600.0, K, 30 / 365, 0.045, 0.013, 0.18, cp) for cp in (1, -1)])
pd.DataFrame({"strike": K, "call": mine[0], "put": mine[1]}).round(4)""",
     """def bsm_price(S, K, T, r, q, sigma, cp):
    d1, d2 = p.d1d2(S, K, T, r, q, sigma)
    return cp * (S * np.exp(-q * T) * p.N(cp * d1) - K * np.exp(-r * T) * p.N(cp * d2))

K = np.array([540.0, 600.0, 660.0])
mine = [bsm_price(600.0, K, 30 / 365, 0.045, 0.013, 0.18, 1), bsm_price(600.0, K, 30 / 365, 0.045, 0.013, 0.18, -1)]
mine = p.check("bsm_price", mine, [p.bsm_price(600.0, K, 30 / 365, 0.045, 0.013, 0.18, cp) for cp in (1, -1)])
pd.DataFrame({"strike": K, "call": mine[0], "put": mine[1]}).round(4)"""),
    ("code", """gap = p.parity_gap(mine[0], mine[1], 600.0, K, 30 / 365, 0.045, 0.013)
print("put–call parity residual C − P − (S e^{−qT} − K e^{−rT}):", gap)"""),
    ("md", "## 2. Delta and vega\n\n"
           "* `delta = cp · e^{−qT} · N(cp·d1)`\n"
           "* `vega = S · e^{−qT} · n(d1) · √T` (per 1.00 of σ, the same for calls and puts; `p.n` is the normal density)"),
    ("md", YOUR_TURN),
    ("ex", """def delta_vega(S, K, T, r, q, sigma, cp):
    d1, _ = p.d1d2(S, K, T, r, q, sigma)
    delta = ...                                   # ✍️
    vega = ...                                    # ✍️
    return delta, vega

grid = np.linspace(480, 720, 7)
mine = [delta_vega(grid, 600.0, 30 / 365, 0.045, 0.013, 0.18, cp) for cp in (1, -1)]
ref = [tuple(p.greeks(grid, 600.0, 30 / 365, 0.045, 0.013, 0.18, cp)[g] for g in ("delta", "vega")) for cp in (1, -1)]
mine = p.check("delta and vega", mine, ref)
pd.DataFrame({"S": grid, "call Δ": mine[0][0], "put Δ": mine[1][0], "vega": mine[0][1]}).round(4)""",
     """def delta_vega(S, K, T, r, q, sigma, cp):
    d1, _ = p.d1d2(S, K, T, r, q, sigma)
    delta = cp * np.exp(-q * T) * p.N(cp * d1)
    vega = S * np.exp(-q * T) * p.n(d1) * np.sqrt(T)
    return delta, vega

grid = np.linspace(480, 720, 7)
mine = [delta_vega(grid, 600.0, 30 / 365, 0.045, 0.013, 0.18, cp) for cp in (1, -1)]
ref = [tuple(p.greeks(grid, 600.0, 30 / 365, 0.045, 0.013, 0.18, cp)[g] for g in ("delta", "vega")) for cp in (1, -1)]
mine = p.check("delta and vega", mine, ref)
pd.DataFrame({"S": grid, "call Δ": mine[0][0], "put Δ": mine[1][0], "vega": mine[0][1]}).round(4)"""),
    ("code", """Sg = np.linspace(450, 750, 300)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for dte, col in [(90, p.PALETTE[0]), (30, p.PALETTE[1]), (5, p.PALETTE[7])]:
    g = p.greeks(Sg, 600.0, dte / 365, 0.045, 0.013, 0.18, 1)
    for ax, name in zip(axes, ("delta", "gamma", "vega", "theta")):
        ax.plot(Sg, g[name], color=col, label=f"{dte} DTE")
for ax, name in zip(axes, ("delta", "gamma", "vega", "theta (per year)")):
    ax.set_title(name); ax.axvline(600, color="#e6e5e0", lw=1)
axes[0].legend(); plt.tight_layout(); plt.show()"""),
    ("md", "## 3. Units: the most common \"my Greeks are wrong\" bug\n\n"
           "The library's raw units are chosen for maths; brokers and py_vollib show **vega per vol point** (÷100), **theta per calendar day** (÷365) "
           "and **rho per 1%** (÷100). Delta and gamma don't change. Return a **new** dict; don't modify the input."),
    ("md", YOUR_TURN),
    ("ex", """def to_display(g):
    out = dict(g)
    out["vega"], out["theta"], out["rho"] = ...   # ✍️ the three conversions
    return out

raw = p.greeks(600.0, 600.0, 30 / 365, 0.045, 0.013, 0.18, 1)
mine = p.attempt(to_display, raw)
mine = p.check("to_display", mine, p.to_display(raw))
pd.DataFrame({"raw": {k: float(v) for k, v in raw.items()}, "display": {k: float(v) for k, v in mine.items()}}).round(4)""",
     """def to_display(g):
    out = dict(g)
    out["vega"], out["theta"], out["rho"] = g["vega"] / 100, g["theta"] / 365, g["rho"] / 100
    return out

raw = p.greeks(600.0, 600.0, 30 / 365, 0.045, 0.013, 0.18, 1)
mine = p.attempt(to_display, raw)
mine = p.check("to_display", mine, p.to_display(raw))
pd.DataFrame({"raw": {k: float(v) for k, v in raw.items()}, "display": {k: float(v) for k, v in mine.items()}}).round(4)"""),
    ("md", "Read it as: this call gains about 0.68 if implied vol rises one point, and loses about 0.23 a day with nothing else changing. "
           "Compare a raw vega of 68 with a broker's 0.68 and you'd think your model was 100× off.\n\n"
           "## 4. Black-76: options on futures\n\n"
           "An option on a future has no carry of its own to model: use BSM with `S = F` and `q = r`. Check it against pricing the same option "
           "off spot with the carry that produced `F`."),
    ("code", """S, r, q, T, sig = 600.0, 0.045, 0.013, 30 / 365, 0.18
F = p.fair_value(S, r, q, T)
print(f"Black-76 on F = {F:.3f}:     {p.black76_price(F, 610.0, T, r, sig, 1):.6f}")
print(f"BSM on spot with q = {q:.3f}: {p.bsm_price(S, 610.0, T, r, q, sig, 1):.6f}   (same option, same answer)")"""),
    ("md", "## Wrap-up\n\n"
           "* One pricer, `cp = ±1`, arrays in and out; Black-76 is BSM with `S = F, q = r`.\n"
           "* Keep raw units inside the library; convert once, at the display edge.\n"
           "* Graded version: `labs/part06/week21_pricing_iv` (prices and all first-order Greeks against py_vollib golden values)."),
]

# ---------------------------------------------------------------------------------------------- 04
NB["04_implied_vol"] = [
    header("04", "Implied volatility and the implied forward", "S4 (Implied volatility)",
           "1. Solve for implied volatility with Newton's method.\n"
           "2. See where Newton fails, and fall back to a bracketing solver.\n"
           "3. Recover the rate and the forward from put–call parity.\n"
           "4. Build the smile from out-of-the-money options with the right inputs."),
    ("code", SETUP),
    ("md", "## 1. Newton's method\n\n"
           "Implied vol is the σ that makes the model price equal the market price. Newton: `σ ← σ − (model(σ) − price) / vega(σ)`. "
           "Start at `sigma0`, stop when `|model − price| < tol`, and give up (NaN) if vega is tiny, σ leaves `(1e-4, 5)`, or `steps` run out."),
    ("md", YOUR_TURN),
    ("ex", """def newton_iv(price, S, K, T, r, q, cp, sigma0=0.3, steps=20, tol=1e-10):
    sigma = sigma0
    for _ in range(steps):
        diff = p.bsm_price(S, K, T, r, q, sigma, cp) - price
        if abs(diff) < tol:
            return float(sigma)
        vega = p.greeks(S, K, T, r, q, sigma, cp)["vega"]
        if vega < 1e-8:
            return np.nan
        sigma = ...                               # ✍️ the Newton step
        if not 1e-4 < sigma < 5:
            return np.nan
    return np.nan

T, r, q = 30 / 365, 0.045, 0.013
cases = [(p.bsm_price(600.0, K, T, r, q, 0.22, cp), 600.0, K, T, r, q, cp) for K, cp in [(600.0, 1), (560.0, -1), (650.0, 1)]]
mine = [p.attempt(newton_iv, *c) for c in cases]
mine = p.check("newton_iv", mine, [p.newton_iv(*c) for c in cases])
mine""",
     """def newton_iv(price, S, K, T, r, q, cp, sigma0=0.3, steps=20, tol=1e-10):
    sigma = sigma0
    for _ in range(steps):
        diff = p.bsm_price(S, K, T, r, q, sigma, cp) - price
        if abs(diff) < tol:
            return float(sigma)
        vega = p.greeks(S, K, T, r, q, sigma, cp)["vega"]
        if vega < 1e-8:
            return np.nan
        sigma = sigma - diff / vega
        if not 1e-4 < sigma < 5:
            return np.nan
    return np.nan

T, r, q = 30 / 365, 0.045, 0.013
cases = [(p.bsm_price(600.0, K, T, r, q, 0.22, cp), 600.0, K, T, r, q, cp) for K, cp in [(600.0, 1), (560.0, -1), (650.0, 1)]]
mine = [p.attempt(newton_iv, *c) for c in cases]
mine = p.check("newton_iv", mine, [p.newton_iv(*c) for c in cases])
mine"""),
    ("md", "## 2. Where Newton breaks\n\n"
           "Far out of the money and close to expiry, vega at the starting guess is almost zero and the first step shoots off. "
           "The fix is a **bracketing** method (Brent) that can't leave the interval, used when Newton fails. And before any of it, "
           "a price outside the no-arbitrage bounds has **no** implied vol: return NaN instead of a made-up number."),
    ("code", """rows = []
for K, dte, true in [(700.0, 7, 0.35), (760.0, 30, 0.45), (520.0, 3, 0.60), (600.0, 30, 0.18)]:
    Tk = dte / 365
    price = p.bsm_price(600.0, K, Tk, r, q, true, 1 if K >= 600 else -1)
    cp = 1 if K >= 600 else -1
    rows.append({"strike": K, "DTE": dte, "price": round(price, 6), "true σ": true,
                 "Newton from 0.3": p.newton_iv(price, 600.0, K, Tk, r, q, cp),
                 "Newton + Brent": p.implied_vol(price, 600.0, K, Tk, r, q, cp)})
display(pd.DataFrame(rows))
print("a price below intrinsic has no IV:", p.implied_vol(1.0, 600.0, 580.0, T, r, q, 1))"""),
    ("md", "## 3. Rate and forward from put–call parity\n\n"
           "Your IVs are only as good as `r`, `q` and the forward you feed in, and dividends are guesses. Parity gives them for free: "
           "`C − P = DF·F − DF·K`, a straight line in `K`. Fit `C − P = a + b·K` (`np.polyfit(K, C − P, 1)` returns `b, a`); then "
           "`DF = −b`, `r = −ln(DF)/T`, `F = a/DF`. Use mids of liquid near-the-money strikes."),
    ("code", """S = 600.0
chain = p.synthetic_chain(S, T, r, q)
liq = p.liquidity_filter(chain)
both = liq.pivot_table(index="strike", columns="cp", values="mid").dropna()
near = both[(both.index > 570) & (both.index < 630)]
near.head()"""),
    ("md", YOUR_TURN),
    ("ex", """def implied_rate_forward(strikes, calls, puts, T):
    b, a = np.polyfit(np.asarray(strikes, float), np.asarray(calls, float) - np.asarray(puts, float), 1)
    df_ = ...                                     # ✍️ the discount factor
    return ..., ...                               # ✍️ (r, F)

mine = p.attempt(implied_rate_forward, near.index, near[1], near[-1], T)
mine = p.check("implied_rate_forward", mine, p.implied_rate_forward(near.index, near[1], near[-1], T))
print(f"implied r = {mine[0]:.3%} (true 4.500%), implied F = {mine[1]:.2f} (true {p.fair_value(S, r, q, T):.2f})")""",
     """def implied_rate_forward(strikes, calls, puts, T):
    b, a = np.polyfit(np.asarray(strikes, float), np.asarray(calls, float) - np.asarray(puts, float), 1)
    df_ = -b
    return float(-np.log(df_) / T), float(a / df_)

mine = p.attempt(implied_rate_forward, near.index, near[1], near[-1], T)
mine = p.check("implied_rate_forward", mine, p.implied_rate_forward(near.index, near[1], near[-1], T))
print(f"implied r = {mine[0]:.3%} (true 4.500%), implied F = {mine[1]:.2f} (true {p.fair_value(S, r, q, T):.2f})")"""),
    ("md", "## 4. The smile, done right\n\n"
           "Build the smile from **out-of-the-money** options: puts below the forward, calls at or above it. In-the-money options are mostly "
           "intrinsic value, so their IV is ill-conditioned (a penny moves it a lot). Price them off the forward with Black-76. "
           "Compare with a common shortcut: guess `r = 4%`, ignore the dividend, use spot."),
    ("code", """r_imp, F_imp = mine
otm = liq[((liq.cp == -1) & (liq.strike < F_imp)) | ((liq.cp == 1) & (liq.strike >= F_imp))]
good = [p.implied_vol(m, F_imp, k, T, r_imp, r_imp, cp) for m, k, cp in zip(otm["mid"], otm.strike, otm.cp)]
guess = [p.implied_vol(m, S, k, T, 0.04, 0.0, cp) for m, k, cp in zip(liq["mid"], liq.strike, liq.cp)]
fig, ax = plt.subplots()
ax.plot(otm.strike, otm.iv_true * 100, color="black", lw=1, label="true smile")
ax.plot(otm.strike, np.array(good) * 100, "o", ms=4, label="OTM mids, implied r and F")
ax.plot(liq.strike[liq.cp == 1], np.array(guess)[liq.cp.to_numpy() == 1] * 100, "x", ms=4, label="calls, guessed r, spot, q = 0")
ax.plot(liq.strike[liq.cp == -1], np.array(guess)[liq.cp.to_numpy() == -1] * 100, "+", ms=5, label="puts, guessed r, spot, q = 0")
ax.set(xlabel="strike", ylabel="IV, %", title="Same quotes, different inputs"); ax.legend(); plt.show()
err = np.nanmax(np.abs(np.array(good) - otm.iv_true.to_numpy())) * 100
print(f"largest error with implied inputs: {err:.2f} vol points (from bid/ask rounding of the mids)")"""),
    ("md", "With guessed inputs, calls and puts at the same strike disagree, and neither matches the truth: the gap is the wrong forward. "
           "With parity-implied inputs the OTM smile lands on the true one.\n\n"
           "## Wrap-up\n\n"
           "* Newton for speed, Brent as the fallback, NaN outside the bounds.\n"
           "* Take `r` and the forward from parity, per expiry; read the smile from OTM options.\n"
           "* Graded versions: `labs/part06/week21_pricing_iv` (IV with fallback statistics, vectorized chain IV) and Clinic W1 (our IV vs the broker's)."),
]

# ---------------------------------------------------------------------------------------------- 05
NB["05_second_order_greeks"] = [
    header("05", "Second-order Greeks and finite differences", "S5 (Second-order Greeks)",
           "1. Compute gamma by bump-and-revalue, and pick the bump size.\n"
           "2. Write vanna and volga in closed form, and verify them by finite differences.\n"
           "3. Watch gamma, theta and charm explode as expiry approaches."),
    ("code", SETUP),
    ("code", """base = dict(S=600.0, K=600.0, T=30 / 365, r=0.045, q=0.013, sigma=0.18, cp=1)
price = p.bsm_price"""),
    ("md", "## 1. Gamma by bump-and-revalue\n\n"
           "Any Greek of any pricer can be estimated by re-pricing with bumped inputs. For gamma, the central second difference: "
           "`Γ ≈ (V(S + h) − 2V(S) + V(S − h)) / h²`. `args` is a dict of the pricer's keyword arguments."),
    ("md", YOUR_TURN),
    ("ex", """def fd_gamma(pricer, args, h):
    up, dn = dict(args, S=args["S"] + h), dict(args, S=args["S"] - h)
    return ...                                    # ✍️

hs = [5.0, 1.0, 0.1]
mine = [fd_gamma(price, base, h) for h in hs]
mine = p.check("fd_gamma", mine, [p.bump(price, base, "S", h, order=2) for h in hs])
exact = p.greeks(**base)["gamma"]
pd.DataFrame({"h": hs, "finite difference": mine, "error vs closed form": np.array(mine) - exact})""",
     """def fd_gamma(pricer, args, h):
    up, dn = dict(args, S=args["S"] + h), dict(args, S=args["S"] - h)
    return (pricer(**up) - 2 * pricer(**args) + pricer(**dn)) / h ** 2

hs = [5.0, 1.0, 0.1]
mine = [fd_gamma(price, base, h) for h in hs]
mine = p.check("fd_gamma", mine, [p.bump(price, base, "S", h, order=2) for h in hs])
exact = p.greeks(**base)["gamma"]
pd.DataFrame({"h": hs, "finite difference": mine, "error vs closed form": np.array(mine) - exact})"""),
    ("md", "Smaller isn't always better. The truncation error shrinks like `h²`, but the rounding error of subtracting nearly equal prices "
           "grows like `ε/h²`. Sweep `h` and the error traces a **U**:"),
    ("code", """hh = np.logspace(-5, 1.5, 60)
err = [abs(p.bump(price, base, "S", h, order=2) - exact) for h in hh]
fig, ax = plt.subplots()
ax.loglog(hh, err)
ax.set(xlabel="bump h (price units)", ylabel="|FD gamma − exact|", title="Truncation error on the right, rounding error on the left"); plt.show()
h_rel = 1e-3 * base["S"]
print(f"best h here ≈ {hh[int(np.argmin(err))]:.3g}; a bump of 0.1% of spot (h = {h_rel:.1f}) gives an error of "
      f"{abs(p.bump(price, base, 'S', h_rel, order=2) - exact):.1e}, with gamma itself {exact:.1e}")"""),
    ("md", "## 2. Vanna and volga\n\n"
           "* **vanna** = ∂Δ/∂σ = ∂vega/∂S = `−e^{−qT} n(d1) d2 / σ`: how delta moves when vol moves (skew risk)\n"
           "* **volga** (vomma) = ∂vega/∂σ = `vega · d1 · d2 / σ`: the convexity of the price in vol (wing risk)\n\n"
           "`vega` is `p.greeks(...)[\"vega\"]` (raw, per 1.00 σ)."),
    ("md", YOUR_TURN),
    ("ex", """def vanna_volga(S, K, T, r, q, sigma, cp):
    d1, d2 = p.d1d2(S, K, T, r, q, sigma)
    vega = p.greeks(S, K, T, r, q, sigma, cp)["vega"]
    vanna = ...                                   # ✍️
    volga = ...                                   # ✍️
    return vanna, volga

Ks = np.array([540.0, 600.0, 660.0])
args = dict(base, K=Ks)
mine = list(vanna_volga(**args))
ref = p.second_order(**args)
mine = p.check("vanna and volga", mine, [ref["vanna"], ref["volga"]])
fd = [p.bump(lambda **a: p.greeks(**a)["delta"], args, "sigma", 1e-4), p.bump(lambda **a: p.greeks(**a)["vega"], args, "sigma", 1e-4)]
pd.DataFrame({"K": Ks, "vanna": mine[0], "vanna (FD)": fd[0], "volga": mine[1], "volga (FD)": fd[1]}).round(4)""",
     """def vanna_volga(S, K, T, r, q, sigma, cp):
    d1, d2 = p.d1d2(S, K, T, r, q, sigma)
    vega = p.greeks(S, K, T, r, q, sigma, cp)["vega"]
    vanna = -np.exp(-q * T) * p.n(d1) * d2 / sigma
    volga = vega * d1 * d2 / sigma
    return vanna, volga

Ks = np.array([540.0, 600.0, 660.0])
args = dict(base, K=Ks)
mine = list(vanna_volga(**args))
ref = p.second_order(**args)
mine = p.check("vanna and volga", mine, [ref["vanna"], ref["volga"]])
fd = [p.bump(lambda **a: p.greeks(**a)["delta"], args, "sigma", 1e-4), p.bump(lambda **a: p.greeks(**a)["vega"], args, "sigma", 1e-4)]
pd.DataFrame({"K": Ks, "vanna": mine[0], "vanna (FD)": fd[0], "volga": mine[1], "volga (FD)": fd[1]}).round(4)"""),
    ("md", "Vanna has opposite signs in the two wings: a downside put's delta moves one way when vol rises, an upside call's the other. "
           "Volga is near zero at the money and largest in the wings, which is why wing options carry a vol-of-vol premium.\n\n"
           "## 3. The last week\n\n"
           "An ATM option's gamma and theta both grow like `1/√T`. **Charm** (how delta drifts as time passes, per year) makes a "
           "delta-hedged position drift over a weekend even if the stock doesn't move."),
    ("code", """dte = np.arange(60, 0, -1)
rows = []
for d in dte:
    g = p.to_display(p.greeks(600.0, 600.0, d / 365, 0.045, 0.013, 0.18, 1))
    g2 = p.second_order(605.0, 600.0, d / 365, 0.045, 0.013, 0.18, 1)
    rows.append({"DTE": d, "ATM gamma": g["gamma"], "ATM theta/day": g["theta"], "charm/day (K=600, S=605)": g2["charm"] / 365})
t = pd.DataFrame(rows).set_index("DTE")
t.plot(subplots=True, figsize=(10, 6), sharex=True, title="Second-order effects near expiry"); plt.gca().invert_xaxis(); plt.show()
print(t.loc[[30, 7, 1]].round(4))"""),
    ("md", "## Wrap-up\n\n"
           "* Closed forms where you have them; bump-and-revalue (with a sensible `h`) for everything else, and to test the closed forms.\n"
           "* Near expiry the second-order Greeks dominate; hedge more often, or hold less gamma.\n"
           "* Graded version: `labs/part06/week22_greeks_numerics` (eight second-order Greeks verified by finite differences)."),
]

# ---------------------------------------------------------------------------------------------- 06
NB["06_numerical_methods"] = [
    header("06", "Trees, Monte Carlo and Greeks from simulation", "S6 (Numerical pricing methods)",
           "1. Write the backward step of a binomial tree, and price an American put.\n"
           "2. See the tree converge to BSM for a European option.\n"
           "3. Cut Monte Carlo error with antithetic variates.\n"
           "4. Get a usable Monte Carlo delta with common random numbers."),
    ("code", SETUP),
    ("md", "## 1. The CRR binomial tree\n\n"
           "Split `T` into `steps`; each step the price goes up by `u = e^{σ√dt}` or down by `d = 1/u`, with risk-neutral probability "
           "`p = (e^{(r−q)dt} − d)/(u − d)`. Start from the payoffs at expiry and step back: each node is the discounted expected value of "
           "its two children, and for an **American** option the larger of that and exercising now. At step `i` the node prices are "
           "`S·u^j·d^(i−j)` for `j = 0 … i`."),
    ("md", YOUR_TURN),
    ("ex", """def crr_price(S, K, T, r, q, sigma, cp, steps=500, american=True):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    pu = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    v = np.maximum(cp * (S * u ** j * d ** (steps - j) - K), 0.0)       # payoffs at expiry
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        v = ...                                   # ✍️ discounted expectation of the two children: v[1:] is up, v[:-1] is down
        if american:
            v = np.maximum(v, cp * (S * u ** j * d ** (i - j) - K))
    return float(v[0])

cases = [(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, -1, 500, True), (100.0, 100.0, 1.0, 0.05, 0.0, 0.2, -1, 500, False),
         (100.0, 110.0, 0.5, 0.05, 0.02, 0.3, 1, 300, True)]
mine = [p.attempt(crr_price, *c) for c in cases]
mine = p.check("crr_price", mine, [p.crr_price(*c) for c in cases])
print(f"American put {mine[0]:.4f}, European put {mine[1]:.4f} (BSM {p.bsm_price(100, 100, 1, 0.05, 0, 0.2, -1):.4f}) → "
      f"early-exercise premium {mine[0] - mine[1]:.4f}")""",
     """def crr_price(S, K, T, r, q, sigma, cp, steps=500, american=True):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    pu = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    v = np.maximum(cp * (S * u ** j * d ** (steps - j) - K), 0.0)       # payoffs at expiry
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        v = disc * (pu * v[1:] + (1 - pu) * v[:-1])
        if american:
            v = np.maximum(v, cp * (S * u ** j * d ** (i - j) - K))
    return float(v[0])

cases = [(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, -1, 500, True), (100.0, 100.0, 1.0, 0.05, 0.0, 0.2, -1, 500, False),
         (100.0, 110.0, 0.5, 0.05, 0.02, 0.3, 1, 300, True)]
mine = [p.attempt(crr_price, *c) for c in cases]
mine = p.check("crr_price", mine, [p.crr_price(*c) for c in cases])
print(f"American put {mine[0]:.4f}, European put {mine[1]:.4f} (BSM {p.bsm_price(100, 100, 1, 0.05, 0, 0.2, -1):.4f}) → "
      f"early-exercise premium {mine[0] - mine[1]:.4f}")"""),
    ("code", """steps = np.arange(10, 301, 1)
err = [p.crr_price(100, 100, 1, 0.05, 0, 0.2, 1, s, american=False) - p.bsm_price(100, 100, 1, 0.05, 0, 0.2, 1) for s in steps]
Ks = np.linspace(70, 130, 25)
prem = [p.crr_price(100, k, 1, 0.05, 0, 0.2, -1, 300) - p.bsm_price(100, k, 1, 0.05, 0, 0.2, -1) for k in Ks]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(steps, err, lw=1); axes[0].axhline(0, color="black", lw=0.6)
axes[0].set(xlabel="steps", ylabel="tree − BSM", title="European call: converges, oscillating odd/even")
axes[1].plot(Ks, prem); axes[1].set(xlabel="strike (spot 100)", ylabel="American − European", title="Early-exercise premium of a put")
plt.tight_layout(); plt.show()"""),
    ("md", "The premium grows with moneyness: a deep in-the-money put is worth exercising early to earn interest on the strike. "
           "That is why BSM implied vols of American puts come out too high (common mistake #5).\n\n"
           "## 2. Monte Carlo and antithetic variates\n\n"
           "Simulate terminal prices `S·exp((r − q − σ²/2)T + σ√T·z)`, average the discounted payoffs, report the standard error. "
           "**Antithetic variates:** for every `z` also use `−z`. Average each pair **first**: the pairs, not the individual paths, are the "
           "independent samples. Return `(price, standard error)`."),
    ("md", YOUR_TURN),
    ("ex", """def mc_antithetic(S, K, T, r, q, sigma, cp, n_paths=100_000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    drift, vol = (r - q - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T)
    pay_up = np.maximum(cp * (S * np.exp(drift + vol * z) - K), 0)
    pay_dn = ...                                  # ✍️ the same with −z
    x = np.exp(-r * T) * 0.5 * (pay_up + pay_dn)  # one discounted sample per pair
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))

mine = p.attempt(mc_antithetic, 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1)
mine = p.check("mc_antithetic", mine, p.mc_european(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1))
plain = p.mc_european(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1, antithetic=False)
print(f"BSM {p.bsm_price(100, 100, 1, 0.05, 0, 0.2, 1):.4f}")
print(f"plain MC       {plain[0]:.4f} ± {plain[1]:.4f}")
print(f"antithetic MC  {mine[0]:.4f} ± {mine[1]:.4f}   (same 100,000 payoff evaluations, half the random draws)")""",
     """def mc_antithetic(S, K, T, r, q, sigma, cp, n_paths=100_000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    drift, vol = (r - q - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T)
    pay_up = np.maximum(cp * (S * np.exp(drift + vol * z) - K), 0)
    pay_dn = np.maximum(cp * (S * np.exp(drift - vol * z) - K), 0)
    x = np.exp(-r * T) * 0.5 * (pay_up + pay_dn)  # one discounted sample per pair
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))

mine = p.attempt(mc_antithetic, 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1)
mine = p.check("mc_antithetic", mine, p.mc_european(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1))
plain = p.mc_european(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1, antithetic=False)
print(f"BSM {p.bsm_price(100, 100, 1, 0.05, 0, 0.2, 1):.4f}")
print(f"plain MC       {plain[0]:.4f} ± {plain[1]:.4f}")
print(f"antithetic MC  {mine[0]:.4f} ± {mine[1]:.4f}   (same 100,000 payoff evaluations, half the random draws)")"""),
    ("md", "## 3. Greeks from simulation: common random numbers\n\n"
           "A Monte Carlo delta by bump-and-revalue subtracts two noisy prices. With **different** random numbers for `S + h` and `S − h`, "
           "the noise of each price (a standard error of about 0.1 with 20,000 paths) is divided by `2h = 1` and swamps the answer. With the **same** random numbers the noise "
           "cancels."),
    ("code", """exact = p.greeks(100, 100, 1, 0.05, 0, 0.2, 1)["delta"]
same = [p.mc_delta(100, 100, 1, 0.05, 0, 0.2, 1, common=True, seed=s) for s in range(40)]
diff = [p.mc_delta(100, 100, 1, 0.05, 0, 0.2, 1, common=False, seed=s) for s in range(40)]
fig, ax = plt.subplots()
ax.hist(diff, bins=20, alpha=0.7, label=f"independent draws (sd {np.std(diff):.3f})")
ax.hist(same, bins=20, alpha=0.7, label=f"common random numbers (sd {np.std(same):.4f})")
ax.axvline(exact, color="black", lw=1, label=f"exact {exact:.4f}")
ax.set(xlabel="MC delta, 40 runs", title="Same work, very different Greeks"); ax.legend(); plt.show()"""),
    ("md", "## Wrap-up\n\n"
           "* Trees for early exercise; check them against BSM on European options.\n"
           "* Monte Carlo: always report the standard error; antithetics are nearly free.\n"
           "* Bumped MC Greeks need common random numbers (or pathwise / automatic differentiation).\n"
           "* Graded version: `labs/part06/week22_greeks_numerics` (also Crank–Nicolson with Rannacher start-up and second-order convergence)."),
]

# ---------------------------------------------------------------------------------------------- 07
NB["07_vol_surface"] = [
    header("07", "The volatility surface: SVI, arbitrage checks and SABR", "S7 (Volatility surface)",
           "1. Write the raw SVI smile and fit it to market IVs.\n"
           "2. Check a smile for butterfly arbitrage and a surface for calendar arbitrage.\n"
           "3. Interpolate between expiries in total variance.\n"
           "4. See how SABR's parameters shape a smile."),
    ("code", SETUP),
    ("md", "## 1. Raw SVI\n\n"
           "Gatheral's raw SVI describes one expiry's **total implied variance** `w = σ²·T` as a function of log-moneyness `k = ln(K/F)`: "
           "`w(k) = a + b·(ρ(k − m) + √((k − m)² + s²))`. `a` sets the level, `b` the slope of the wings, `ρ` the skew, `m` shifts it, `s` "
           "rounds the bottom."),
    ("md", YOUR_TURN),
    ("ex", """def svi_total_var(k, a, b, rho, m, s):
    return ...                                    # ✍️

k = np.linspace(-0.25, 0.15, 9)
mine = svi_total_var(k, *p.TRUE_SVI)
mine = p.check("svi_total_var", mine, p.svi_total_var(k, *p.TRUE_SVI))
np.sqrt(np.asarray(mine) / (30 / 365)).round(4)       # as implied vols""",
     """def svi_total_var(k, a, b, rho, m, s):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))

k = np.linspace(-0.25, 0.15, 9)
mine = svi_total_var(k, *p.TRUE_SVI)
mine = p.check("svi_total_var", mine, p.svi_total_var(k, *p.TRUE_SVI))
np.sqrt(np.asarray(mine) / (30 / 365)).round(4)       # as implied vols"""),
    ("md", "## 2. Fit it to a chain\n\n"
           "Take the liquid out-of-the-money quotes, compute IVs off the parity-implied forward (notebook 04), and fit SVI by bounded least "
           "squares on total variance. The chain was generated from `p.TRUE_SVI`, so we can see whether the fit recovers it."),
    ("code", """S, T, r, q = 600.0, 30 / 365, 0.045, 0.013
chain = p.liquidity_filter(p.synthetic_chain(S, T, r, q))
both = chain.pivot_table(index="strike", columns="cp", values="mid").dropna()
near = both[(both.index > 570) & (both.index < 630)]
r_imp, F = p.implied_rate_forward(near.index, near[1], near[-1], T)
otm = chain[((chain.cp == -1) & (chain.strike < F)) | ((chain.cp == 1) & (chain.strike >= F))]
iv = np.array([p.implied_vol(m, F, K, T, r_imp, r_imp, cp) for m, K, cp in zip(otm["mid"], otm.strike, otm.cp)])
kk = np.log(otm.strike.to_numpy() / F)
fit = p.fit_svi(kk, iv, T)
display(pd.DataFrame({"true": p.TRUE_SVI, "fitted": fit}, index=["a", "b", "rho", "m", "s"]).round(4))
grid = np.linspace(kk.min(), kk.max(), 200)
plt.plot(kk, iv * 100, "o", ms=4, label="market IV (OTM mids)")
plt.plot(grid, np.sqrt(p.svi_total_var(grid, *fit) / T) * 100, label="SVI fit")
plt.xlabel("k = ln(K/F)"); plt.ylabel("IV, %"); plt.title("A 30-day equity smile"); plt.legend(); plt.show()"""),
    ("md", "## 3. Butterfly arbitrage\n\n"
           "A smile implies a risk-neutral density; if the density is negative somewhere, a butterfly spread there has a negative price: "
           "free money, or more likely a broken fit. Durrleman's condition `g(k) >= 0` checks it (`p.butterfly_g`). An unconstrained fit "
           "with wings that are too steep fails:"),
    ("code", """bad = (0.002, 0.2, -0.8, 0.0, 0.01)
k = np.linspace(-0.4, 0.3, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for prm, name in [(p.TRUE_SVI, "fitted-style smile"), (bad, "too-steep wings")]:
    axes[0].plot(k, np.sqrt(p.svi_total_var(k, *prm) / T) * 100, label=name)
    axes[1].plot(k, p.butterfly_g(k, *prm), label=name)
axes[1].axhline(0, color="black", lw=0.8); axes[1].set_ylim(-3, 3)
axes[0].set_title("IV, %"); axes[1].set_title("g(k): must stay >= 0"); axes[1].legend(); plt.tight_layout(); plt.show()
print(f"bad smile: g < 0 for k in [{k[p.butterfly_g(k, *bad) < 0].min():.2f}, {k[p.butterfly_g(k, *bad) < 0].max():.2f}]")"""),
    ("md", "## 4. Calendar arbitrage and interpolation\n\n"
           "Across expiries, total variance at a fixed `k` must **not decrease** with `T` (a longer option can't be worth less). "
           "Return every `(T_short, T_long, k)` on the grid where `w` falls from one expiry to the next (`w_long < w_short`)."),
    ("code", """surface = {30 / 365: p.TRUE_SVI,
           60 / 365: (0.0030, 0.050, -0.65, 0.03, 0.10),
           90 / 365: (0.0044, 0.030, -0.60, 0.02, 0.10)}        # the 90-day smile holds too little total variance
k_grid = np.round(np.linspace(-0.3, 0.2, 11), 3)"""),
    ("md", YOUR_TURN),
    ("ex", """def calendar_violations(k_grid, params_by_T):
    Ts = sorted(params_by_T)
    out = []
    for t1, t2 in zip(Ts, Ts[1:]):
        w1 = p.svi_total_var(k_grid, *params_by_T[t1])
        w2 = p.svi_total_var(k_grid, *params_by_T[t2])
        out += ...                                # ✍️ [(t1, t2, float(k)) for each k where w2 < w1 − 1e-12]
    return out

mine = p.attempt(calendar_violations, k_grid, surface)
mine = p.check("calendar_violations", mine, p.calendar_violations(k_grid, surface))
[(round(a * 365), round(b * 365), k) for a, b, k in mine]""",
     """def calendar_violations(k_grid, params_by_T):
    Ts = sorted(params_by_T)
    out = []
    for t1, t2 in zip(Ts, Ts[1:]):
        w1 = p.svi_total_var(k_grid, *params_by_T[t1])
        w2 = p.svi_total_var(k_grid, *params_by_T[t2])
        out += [(t1, t2, float(k)) for k, a, b in zip(k_grid, w1, w2) if b < a - 1e-12]
    return out

mine = p.attempt(calendar_violations, k_grid, surface)
mine = p.check("calendar_violations", mine, p.calendar_violations(k_grid, surface))
[(round(a * 365), round(b * 365), k) for a, b, k in mine]"""),
    ("md", "Between listed expiries, interpolate **total variance** linearly in `T` at fixed `k`, then convert back: `σ = √(w/T)`. "
           "Interpolating the vols themselves can create calendar arbitrage even when both expiries are clean."),
    ("md", YOUR_TURN),
    ("ex", """def interp_iv(k, T, T1, params1, T2, params2):
    w1, w2 = p.svi_total_var(k, *params1), p.svi_total_var(k, *params2)
    w = ...                                       # ✍️ linear in T between (T1, w1) and (T2, w2)
    return np.sqrt(w / T)

T1, T2 = 30 / 365, 60 / 365
cases = [(0.0, 45 / 365), (-0.1, 40 / 365), (0.05, 55 / 365)]
mine = [p.attempt(interp_iv, k, t, T1, surface[T1], T2, surface[T2]) for k, t in cases]
expected = [np.sqrt((p.svi_total_var(k, *surface[T1]) + (p.svi_total_var(k, *surface[T2]) - p.svi_total_var(k, *surface[T1])) * (t - T1) / (T2 - T1)) / t)
            for k, t in cases]
mine = p.check("interp_iv", mine, expected)
[f"{x:.2%}" for x in mine]""",
     """def interp_iv(k, T, T1, params1, T2, params2):
    w1, w2 = p.svi_total_var(k, *params1), p.svi_total_var(k, *params2)
    w = w1 + (w2 - w1) * (T - T1) / (T2 - T1)
    return np.sqrt(w / T)

T1, T2 = 30 / 365, 60 / 365
cases = [(0.0, 45 / 365), (-0.1, 40 / 365), (0.05, 55 / 365)]
mine = [p.attempt(interp_iv, k, t, T1, surface[T1], T2, surface[T2]) for k, t in cases]
expected = [np.sqrt((p.svi_total_var(k, *surface[T1]) + (p.svi_total_var(k, *surface[T2]) - p.svi_total_var(k, *surface[T1])) * (t - T1) / (T2 - T1)) / t)
            for k, t in cases]
mine = p.check("interp_iv", mine, expected)
[f"{x:.2%}" for x in mine]"""),
    ("md", "## 5. SABR: a model with meaning in its parameters\n\n"
           "SABR models the forward and its volatility as two correlated processes. Its four parameters read like a trader's view: `α` the "
           "level, `β` the backbone (how vol moves with the forward), `ρ` the spot–vol correlation (skew), `ν` the vol of vol (curvature). "
           "Hagan's formula gives the implied vol directly (`p.sabr_vol`)."),
    ("code", """Fq, Ks = 100.0, np.linspace(70, 130, 121)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for rho_ in (-0.6, -0.3, 0.0, 0.3):
    axes[0].plot(Ks, p.sabr_vol(Fq, Ks, 1.0, 0.2, 1.0, rho_, 0.5) * 100, label=f"ρ = {rho_}")
for nu in (0.1, 0.4, 0.8):
    axes[1].plot(Ks, p.sabr_vol(Fq, Ks, 1.0, 0.2, 1.0, -0.3, nu) * 100, label=f"ν = {nu}")
axes[0].set_title("ρ tilts the smile (skew)"); axes[1].set_title("ν bends it (curvature)")
for ax in axes:
    ax.set_xlabel("strike (F = 100)"); ax.legend()
axes[0].set_ylabel("IV, %"); plt.tight_layout(); plt.show()"""),
    ("md", "## Wrap-up\n\n"
           "* Fit smiles in total variance with bounds, then **check** them: butterfly within an expiry, calendar across expiries.\n"
           "* Interpolate in total variance, not in vol.\n"
           "* Graded version: `labs/part06/week22_surface_portfolio` (SVI recovery, Durrleman and calendar checks, SABR) and the Clinic W2 surface snapshot."),
]

# ---------------------------------------------------------------------------------------------- 08
NB["08_portfolio_and_hedging"] = [
    header("08", "Portfolio Greeks, scenarios and delta hedging", "S8 (Portfolio Greeks, scenario risk & delta hedging)",
           "1. Turn per-option Greeks into dollar Greeks for a book, with multipliers.\n"
           "2. Compare the delta–gamma–vega approximation with full revaluation, and see it fail for a crash.\n"
           "3. Size the stock hedge that flattens a book's delta.\n"
           "4. Simulate delta hedging: where the P&L comes from, and what hedging costs."),
    ("code", SETUP),
    ("md", "## 1. A small book\n\n"
           "SPY-like underlying at 600, 30-day options with a multiplier of **100** (from contract details, never hard-coded in real code)."),
    ("code", """S = 600.0
book = [p.Position("short 20 × 560 puts", "option", -20, K=560.0, T=30 / 365, iv=0.24, cp=-1),
        p.Position("long 10 × 620 calls", "option", 10, K=620.0, T=30 / 365, iv=0.15, cp=1),
        p.Position("short 5 × 600 calls", "option", -5, K=600.0, T=30 / 365, iv=0.18, cp=1),
        p.Position("long 300 shares", "stock", 300, multiplier=1.0)]
pd.DataFrame([vars(b) for b in book])"""),
    ("md", "## 2. Dollar Greeks\n\n"
           "Per-contract Greeks are per one share of underlying. For a position multiply by `qty × multiplier`:\n"
           "* **$delta** = Δ · qty · mult · S: the dollar exposure, the P&L of a 100% move if it stayed linear;\n"
           "* **vega per point** = (vega/100) · qty · mult: P&L if implied vol rises one point.\n\n"
           "`p.to_display(p.greeks(...))` gives vega per point. (The reference also reports $gamma for a 1% move and theta per day.)"),
    ("md", YOUR_TURN),
    ("ex", """def dollar_delta_vega(pos, S, r=0.045, q=0.013):
    if pos.kind == "stock":
        return pos.qty * pos.multiplier * S, 0.0
    g = p.to_display(p.greeks(S, pos.K, pos.T, r, q, pos.iv, pos.cp))
    m = pos.qty * pos.multiplier
    return ..., ...                               # ✍️ ($delta, vega per point)

mine = [p.attempt(dollar_delta_vega, b, S) for b in book]
ref = [(p.dollar_greeks(b, S)["$delta"], p.dollar_greeks(b, S)["vega/pt"]) for b in book]
mine = p.check("dollar delta and vega", mine, ref)
report = pd.DataFrame([p.dollar_greeks(b, S) for b in book], index=[b.name for b in book])
report.loc["book"] = report.sum()
report.round(0)""",
     """def dollar_delta_vega(pos, S, r=0.045, q=0.013):
    if pos.kind == "stock":
        return pos.qty * pos.multiplier * S, 0.0
    g = p.to_display(p.greeks(S, pos.K, pos.T, r, q, pos.iv, pos.cp))
    m = pos.qty * pos.multiplier
    return float(g["delta"] * m * S), float(g["vega"] * m)

mine = [p.attempt(dollar_delta_vega, b, S) for b in book]
ref = [(p.dollar_greeks(b, S)["$delta"], p.dollar_greeks(b, S)["vega/pt"]) for b in book]
mine = p.check("dollar delta and vega", mine, ref)
report = pd.DataFrame([p.dollar_greeks(b, S) for b in book], index=[b.name for b in book])
report.loc["book"] = report.sum()
report.round(0)"""),
    ("md", "## 3. Taylor vs full revaluation\n\n"
           "The quick risk estimate is a Taylor expansion: `ΔV ≈ Δ·ΔS + ½Γ·ΔS² + vega·Δσ`, summed over positions with quantities and "
           "multipliers (raw units: `ΔS = S·dS` in price, `Δσ` as a decimal). A stock position contributes only `qty·mult·ΔS`."),
    ("md", YOUR_TURN),
    ("ex", """def taylor_pnl(book, S, dS, dvol, r=0.045, q=0.013):
    total = 0.0
    for pos in book:
        m = pos.qty * pos.multiplier
        x = S * dS
        if pos.kind == "stock":
            total += m * x
            continue
        g = p.greeks(S, pos.K, pos.T, r, q, pos.iv, pos.cp)
        total += ...                              # ✍️ m × (delta·x + ½·gamma·x² + vega·dvol)
    return float(total)

shocks = [(0.01, 0.0), (-0.05, 0.03), (-0.20, 0.25), (0.10, -0.05)]
mine = [p.attempt(taylor_pnl, book, S, a, b) for a, b in shocks]
mine = p.check("taylor_pnl", mine, [p.taylor_pnl(book, S, a, b) for a, b in shocks])
pd.DataFrame({"spot move": [a for a, _ in shocks], "vol move": [b for _, b in shocks], "Taylor": mine,
              "full revaluation": [p.full_reval(book, S, a, b) for a, b in shocks]}).round(0)""",
     """def taylor_pnl(book, S, dS, dvol, r=0.045, q=0.013):
    total = 0.0
    for pos in book:
        m = pos.qty * pos.multiplier
        x = S * dS
        if pos.kind == "stock":
            total += m * x
            continue
        g = p.greeks(S, pos.K, pos.T, r, q, pos.iv, pos.cp)
        total += m * (g["delta"] * x + 0.5 * g["gamma"] * x ** 2 + g["vega"] * dvol)
    return float(total)

shocks = [(0.01, 0.0), (-0.05, 0.03), (-0.20, 0.25), (0.10, -0.05)]
mine = [p.attempt(taylor_pnl, book, S, a, b) for a, b in shocks]
mine = p.check("taylor_pnl", mine, [p.taylor_pnl(book, S, a, b) for a, b in shocks])
pd.DataFrame({"spot move": [a for a, _ in shocks], "vol move": [b for _, b in shocks], "Taylor": mine,
              "full revaluation": [p.full_reval(book, S, a, b) for a, b in shocks]}).round({"Taylor": 0, "full revaluation": 0})"""),
    ("code", """spot = np.array([-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
vol = np.array([-0.05, 0.0, 0.05, 0.10, 0.25])
full = pd.DataFrame([[p.full_reval(book, S, a, b) for b in vol] for a in spot], index=spot, columns=vol)
tay = pd.DataFrame([[p.taylor_pnl(book, S, a, b) for b in vol] for a in spot], index=spot, columns=vol)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (name, t) in zip(axes, [("full revaluation P&L", full), ("Taylor error (Taylor − full)", tay - full)]):
    im = ax.imshow(t.to_numpy() / 1000, cmap="RdBu", aspect="auto", vmin=-np.abs(t.to_numpy()).max() / 1000, vmax=np.abs(t.to_numpy()).max() / 1000)
    ax.set_xticks(range(len(vol)), [f"{v:+.0%}" for v in vol]); ax.set_yticks(range(len(spot)), [f"{s:+.0%}" for s in spot])
    ax.set(xlabel="vol shock (points / 100)", ylabel="spot shock", title=name + ", $k"); plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.show()
worst = full.stack().idxmin()
print(f"worst scenario {worst}: full revaluation {full.stack().min():,.0f}, Taylor {tay.loc[worst]:,.0f}")"""),
    ("md", "The approximation is fine for small moves and badly wrong for the crash: the short puts' gamma grows as they go into the money, "
           "which a second-order expansion can't see. Risk limits use the full-revaluation grid.\n\n"
           "## 4. The delta hedge\n\n"
           "Flatten the book's delta with shares: the book's delta in **shares** is `Σ Δ·qty·mult` (a share has Δ = 1, multiplier 1). "
           "Trade the opposite, rounded to whole shares."),
    ("md", YOUR_TURN),
    ("ex", """def hedge_shares(book, S, r=0.045, q=0.013):
    delta_shares = 0.0
    for pos in book:
        d = 1.0 if pos.kind == "stock" else p.greeks(S, pos.K, pos.T, r, q, pos.iv, pos.cp)["delta"]
        delta_shares += ...                       # ✍️
    return -int(round(delta_shares))

mine = p.attempt(hedge_shares, book, S)
expected = -int(round(sum(p.dollar_greeks(b, S)["$delta"] for b in book) / S))
mine = p.check("hedge_shares", mine, expected)
hedged = book + [p.Position("hedge", "stock", mine, multiplier=1.0)]
print(f"trade {mine:+d} shares; book $delta after hedge: {sum(p.dollar_greeks(b, S)['$delta'] for b in hedged):,.0f}")""",
     """def hedge_shares(book, S, r=0.045, q=0.013):
    delta_shares = 0.0
    for pos in book:
        d = 1.0 if pos.kind == "stock" else p.greeks(S, pos.K, pos.T, r, q, pos.iv, pos.cp)["delta"]
        delta_shares += d * pos.qty * pos.multiplier
    return -int(round(delta_shares))

mine = p.attempt(hedge_shares, book, S)
expected = -int(round(sum(p.dollar_greeks(b, S)["$delta"] for b in book) / S))
mine = p.check("hedge_shares", mine, expected)
hedged = book + [p.Position("hedge", "stock", mine, multiplier=1.0)]
print(f"trade {mine:+d} shares; book $delta after hedge: {sum(p.dollar_greeks(b, S)['$delta'] for b in hedged):,.0f}")"""),
    ("md", "## 5. What a delta-hedged option earns\n\n"
           "Buy an ATM option at an implied vol of 20% and delta-hedge it daily until expiry. What you earn depends on the vol the stock "
           "then **realizes**: roughly `Σ ½Γ·S²·(realized² − implied²)·dt`. Hedging less often adds noise; hedging costs eat the edge."),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for rv in (0.15, 0.20, 0.25):
    pnl = p.delta_hedge_pnl(iv=0.20, rv=rv, steps=30)
    axes[0].hist(pnl, bins=40, alpha=0.6, label=f"realized {rv:.0%}: mean {pnl.mean():+.2f}")
axes[0].set(title="Long option at 20% IV, hedged daily", xlabel="P&L per option (spot 100)"); axes[0].legend()
rows = []
for steps in (5, 30, 120):
    for cost in (0.0, 5.0):
        pnl = p.delta_hedge_pnl(iv=0.20, rv=0.20, steps=steps, cost_bps=cost)
        rows.append({"hedges": steps, "cost (bp)": cost, "mean": pnl.mean(), "sd": pnl.std()})
t = pd.DataFrame(rows)
for cost, g in t.groupby("cost (bp)"):
    axes[1].plot(g.hedges, g["sd"], "o-", label=f"sd of P&L, cost {cost:.0f} bp")
axes[1].set(xscale="log", xlabel="hedges until expiry", title="Hedging more often: less noise, more cost"); axes[1].legend()
plt.tight_layout(); plt.show()
t.round(3)"""),
    ("md", "## Wrap-up\n\n"
           "* Greeks × quantity × multiplier; report in dollars and per vol point.\n"
           "* Taylor for intuition, full revaluation for limits.\n"
           "* A delta-hedged option is a bet on realized vs implied volatility; hedge frequency trades noise against cost.\n"
           "* Graded versions: `labs/part06/week22_surface_portfolio` (beta-weighted book report, scenario grid, hedging simulator) and the Clinic W2 risk report."),
]


def build():
    (ROOT / "solutions").mkdir(exist_ok=True)
    for name, cells in NB.items():
        for kind in ("starter", "solution"):
            n = nbf.v4.new_notebook()
            n.metadata.update(KERNEL)
            out = []
            if kind == "solution":
                out.append(nbf.v4.new_markdown_cell("> **INSTRUCTOR SOLUTIONS** — do not share with learners before the session."))
            for c in cells:
                if c[0] == "md":
                    out.append(nbf.v4.new_markdown_cell(c[1]))
                elif c[0] == "code":
                    out.append(nbf.v4.new_code_cell(c[1]))
                else:
                    cell = nbf.v4.new_code_cell(c[1] if kind == "starter" else c[2])
                    cell.metadata["tags"] = ["exercise"]
                    out.append(cell)
            n.cells = out
            path = ROOT / (f"{name}.ipynb" if kind == "starter" else f"solutions/{name}_solution.ipynb")
            nbf.write(n, path)
    print(f"built {len(NB)} starter + {len(NB)} solution notebooks")


if __name__ == "__main__":
    build()
