"""Build the Part 8 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part08:  python tools/build_notebooks.py
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
for d in (Path.cwd(), Path.cwd().parent):       # p8lib.py is in notebooks/part08/
    sys.path.insert(0, str(d))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p8lib as p

p.use_course_style()"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 8 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_08_BACKTESTING_RISK_PORTFOLIO.md) · graded labs in [`labs/part08/`](../../labs/part08/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.
All data is synthetic with a known truth (known regimes, known Sharpe ratios, pure noise), so every statistic can be checked against reality and every discovery against luck.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
NB["01_research_log_and_engine"] = [
    header("01", "The research log and the event-driven engine", "S1 (Backtesting philosophy & taxonomy) · S2 (Event-driven architecture)",
           "1. Give every backtest configuration a stable fingerprint, and log every run.\n"
           "2. Write the ledger that turns fills into cash, positions and equity (with futures multipliers).\n"
           "3. Order same-timestamp events deterministically.\n"
           "4. Run the event-driven engine on a strategy from Part 7."),
    ("code", SETUP + "\nimport hashlib, json"),
    ("md", "## 1. Every run leaves a trace\n\n"
           "A backtest is an experiment, and the most dangerous number in research is the one you don't record: how many things you tried. "
           "The Deflated Sharpe Ratio (notebook 07) needs **all** the trials, including the failures. So every run is logged with a "
           "fingerprint of its configuration: the first 12 hex characters of the SHA-256 of `json.dumps(config, sort_keys=True)`. "
           "Sorting the keys makes the hash independent of the order you wrote them in."),
    ("md", YOUR_TURN),
    ("ex", """def config_hash(config):
    return ...                                    # ✍️ sha256 of the sorted JSON, first 12 hex characters

configs = [{"strategy": "sma_cross", "params": {"fast": 20, "slow": 100}, "data_version": "v1"},
           {"data_version": "v1", "params": {"slow": 100, "fast": 20}, "strategy": "sma_cross"},     # same, reordered
           {"strategy": "sma_cross", "params": {"fast": 20, "slow": 101}, "data_version": "v1"}]
mine = [p.attempt(config_hash, c) for c in configs]
mine = p.check("config_hash", mine, [p.config_hash(c) for c in configs])
mine""",
     """def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]

configs = [{"strategy": "sma_cross", "params": {"fast": 20, "slow": 100}, "data_version": "v1"},
           {"data_version": "v1", "params": {"slow": 100, "fast": 20}, "strategy": "sma_cross"},     # same, reordered
           {"strategy": "sma_cross", "params": {"fast": 20, "slow": 101}, "data_version": "v1"}]
mine = [p.attempt(config_hash, c) for c in configs]
mine = p.check("config_hash", mine, [p.config_hash(c) for c in configs])
mine"""),
    ("code", """bars = p.regime_market()
o, c = bars.open.to_numpy(), bars.close.to_numpy()
log = p.ResearchLog()                                      # sqlite in memory; a file in real research
for fast in (5, 10, 20, 40):
    for slow in (50, 100, 150, 200, 250):
        pnl = p.first_look_pnl(p.sma_cross_signal(c, fast, slow), o)
        log.record("sma_cross", {"fast": fast, "slow": slow}, "regime_market:v1", p.sharpe(pnl), p.max_drawdown(pnl)[0], len(pnl))
trials = log.trials("sma_cross")
print(f"{len(trials)} trials logged; best Sharpe {trials.sharpe.max():.2f}, median {trials.sharpe.median():.2f}")
trials.sort_values("sharpe", ascending=False).head()"""),
    ("md", "The best of 20 tries is what gets reported; the other 19 are what make it believable, or not.\n\n"
           "## 2. The ledger\n\n"
           "The engine's portfolio keeps cash, positions, last prices and **contract multipliers** (ES futures: 50 dollars per point). "
           "A fill of `qty` at `price` costs `qty × price × multiplier + fee` in cash (a sale, with negative `qty`, brings cash in); fees "
           "accumulate in `self.fees`. Equity is cash plus every position marked at its last price, times its multiplier."),
    ("md", YOUR_TURN),
    ("ex", """class MyPortfolio(p.Portfolio):
    def on_fill(self, sym, qty, price, fee=0.0):
        m = self.mult.get(sym, 1.0)
        ...                                       # ✍️ update cash, fees and the position

    @property
    def equity(self):
        return ...                                # ✍️ cash + Σ qty × last price × multiplier

def replay(cls):
    pf = cls(1_000_000, multipliers={"ES": 50})
    out = []
    for kind, *args in [("fill", "SPY", 100, 500.0, 1.0), ("fill", "ES", 2, 5000.0, 4.5), ("mark", "SPY", 505.0),
                        ("mark", "ES", 5010.0), ("fill", "SPY", -50, 506.0, 1.0), ("mark", "ES", 4990.0)]:
        pf.on_fill(*args) if kind == "fill" else pf.mark(*args)
        if kind == "fill":
            pf.mark(args[0], args[2])
        out.append((round(pf.cash, 2), round(pf.equity, 2) if pf.equity is not Ellipsis else Ellipsis))
    return out

mine = p.attempt(replay, MyPortfolio)
mine = p.check("ledger", mine, replay(p.Portfolio))
pd.DataFrame(mine, columns=["cash", "equity"])""",
     """class MyPortfolio(p.Portfolio):
    def on_fill(self, sym, qty, price, fee=0.0):
        m = self.mult.get(sym, 1.0)
        self.cash -= qty * price * m + fee
        self.fees += fee
        self.pos[sym] = self.pos.get(sym, 0.0) + qty

    @property
    def equity(self):
        return self.cash + sum(q * self.last.get(s, 0.0) * self.mult.get(s, 1.0) for s, q in self.pos.items())

def replay(cls):
    pf = cls(1_000_000, multipliers={"ES": 50})
    out = []
    for kind, *args in [("fill", "SPY", 100, 500.0, 1.0), ("fill", "ES", 2, 5000.0, 4.5), ("mark", "SPY", 505.0),
                        ("mark", "ES", 5010.0), ("fill", "SPY", -50, 506.0, 1.0), ("mark", "ES", 4990.0)]:
        pf.on_fill(*args) if kind == "fill" else pf.mark(*args)
        if kind == "fill":
            pf.mark(args[0], args[2])
        out.append((round(pf.cash, 2), round(pf.equity, 2) if pf.equity is not Ellipsis else Ellipsis))
    return out

mine = p.attempt(replay, MyPortfolio)
mine = p.check("ledger", mine, replay(p.Portfolio))
pd.DataFrame(mine, columns=["cash", "equity"])"""),
    ("md", "Two ES contracts cost $500,000 of *notional* here only because this toy ledger books futures like stock. A real futures ledger "
           "books margin and daily variation instead; the equity arithmetic (P&L = Δprice × qty × 50) is the same.\n\n"
           "## 3. Deterministic event order\n\n"
           "The engine is a priority queue of events. Events with the **same timestamp** must be processed in a fixed order, or two runs "
           "of the same backtest can differ: fills first (priority 0), then bars (1), then timers (2), and within a tie, the order they were "
           "added (`seq`). Write the sort key."),
    ("md", YOUR_TURN),
    ("ex", """PRIORITY = {"fill": 0, "bar": 1, "timer": 2}
events = [{"ts": 2, "kind": "timer", "seq": 0}, {"ts": 1, "kind": "bar", "seq": 1}, {"ts": 2, "kind": "bar", "seq": 2},
          {"ts": 2, "kind": "fill", "seq": 3}, {"ts": 1, "kind": "timer", "seq": 4}, {"ts": 2, "kind": "fill", "seq": 5}]

def event_key(e):
    return ...                                    # ✍️ a tuple: timestamp, then priority, then sequence number

mine = p.attempt(lambda: [(e["ts"], e["kind"], e["seq"]) for e in sorted(events, key=event_key)])
q = p.EventQueue()
for e in events:
    q.push(e["ts"], PRIORITY[e["kind"]], e["kind"], e["seq"])
ref = []
while q:
    ev = q.pop()
    ref.append((ev.ts, ev.kind, ev.data))
mine = p.check("event order", mine, ref)
mine""",
     """PRIORITY = {"fill": 0, "bar": 1, "timer": 2}
events = [{"ts": 2, "kind": "timer", "seq": 0}, {"ts": 1, "kind": "bar", "seq": 1}, {"ts": 2, "kind": "bar", "seq": 2},
          {"ts": 2, "kind": "fill", "seq": 3}, {"ts": 1, "kind": "timer", "seq": 4}, {"ts": 2, "kind": "fill", "seq": 5}]

def event_key(e):
    return (e["ts"], PRIORITY[e["kind"]], e["seq"])

mine = p.attempt(lambda: [(e["ts"], e["kind"], e["seq"]) for e in sorted(events, key=event_key)])
q = p.EventQueue()
for e in events:
    q.push(e["ts"], PRIORITY[e["kind"]], e["kind"], e["seq"])
ref = []
while q:
    ev = q.pop()
    ref.append((ev.ts, ev.kind, ev.data))
mine = p.check("event order", mine, ref)
mine"""),
    ("md", "## 4. The engine\n\n"
           "`p.run_backtest` puts it together for one symbol: at each close the strategy's target weight becomes a share order; the order "
           "fills at the **next open** (with slippage and commission if given); positions are marked at the open and the close. "
           "Same strategy, same bars as Part 7."),
    ("code", """sig = p.sma_cross_signal(c, 20, 100)
res = p.run_backtest(bars, sig, slippage_bps=2.0, commission=p.ib_fixed_commission)
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(res.index, res.equity / 1e6); axes[0].set_ylabel("equity, $m")
axes[1].plot(res.index, res.position); axes[1].set_ylabel("shares")
axes[0].set_title("SMA 20/100 through the event-driven engine"); plt.show()
print(f"final equity ${res.equity.iloc[-1]:,.0f}; commissions paid ${res.fees.iloc[-1]:,.0f}")"""),
    ("md", "## Wrap-up\n\n"
           "* Log every run, with a configuration hash and a data version.\n"
           "* The ledger handles multipliers and fees; equity is cash plus marked positions.\n"
           "* Same-timestamp events have a fixed order, so every run replays identically.\n"
           "* Graded version: `labs/part08/week25_engine` (DuckDB research log, event queue, ledger, fills, the engine)."),
]

# ---------------------------------------------------------------------------------------------- 02
NB["02_fills_costs_parity"] = [
    header("02", "Fills, costs and parity", "S3 (Fill & cost models) · S4 (Framework comparison & parity testing)",
           "1. Fill limit orders only when the price trades through them.\n"
           "2. Fill stop orders at the gap when the market opens beyond the stop.\n"
           "3. Estimate market impact with the square-root law.\n"
           "4. Prove the engine and the vectorized first look agree, then see what costs do to each strategy."),
    ("code", SETUP),
    ("md", "## 1. Limit orders: touching is not filling\n\n"
           "A resting limit order joins a queue. If the price only **touches** your level, the orders ahead of you may absorb all the volume. "
           "Conservative rule: a **buy** limit fills only if the bar's **low is below** the limit (at `min(open, limit)`, since a gap down "
           "fills at the open); a **sell** limit only if the **high is above** it (at `max(open, limit)`). Otherwise `None`."),
    ("md", YOUR_TURN),
    ("ex", """def limit_fill_price(qty, limit, bar):
    if qty > 0:
        return ...                                # ✍️ buy
    return ...                                    # ✍️ sell

bars_ = [{"open": 101.0, "high": 102.0, "low": 99.5, "close": 100.5}, {"open": 101.0, "high": 102.0, "low": 100.0, "close": 100.5},
         {"open": 98.0, "high": 99.0, "low": 97.0, "close": 98.5}, {"open": 99.0, "high": 101.0, "low": 98.0, "close": 100.5},
         {"open": 102.0, "high": 103.0, "low": 101.5, "close": 102.5}]
cases = [(100, 100.0, b) for b in bars_] + [(-100, 101.0, b) for b in bars_]
mine = [p.attempt(limit_fill_price, *cs) for cs in cases]
mine = p.check("limit_fill_price", mine, [p.limit_fill_price(*cs) for cs in cases])
pd.DataFrame({"side": ["buy 100 lmt"] * 5 + ["sell 101 lmt"] * 5, "low": [b["low"] for b in bars_] * 2,
              "high": [b["high"] for b in bars_] * 2, "fill": mine})""",
     """def limit_fill_price(qty, limit, bar):
    if qty > 0:
        return min(bar["open"], limit) if bar["low"] < limit else None
    return max(bar["open"], limit) if bar["high"] > limit else None

bars_ = [{"open": 101.0, "high": 102.0, "low": 99.5, "close": 100.5}, {"open": 101.0, "high": 102.0, "low": 100.0, "close": 100.5},
         {"open": 98.0, "high": 99.0, "low": 97.0, "close": 98.5}, {"open": 99.0, "high": 101.0, "low": 98.0, "close": 100.5},
         {"open": 102.0, "high": 103.0, "low": 101.5, "close": 102.5}]
cases = [(100, 100.0, b) for b in bars_] + [(-100, 101.0, b) for b in bars_]
mine = [p.attempt(limit_fill_price, *cs) for cs in cases]
mine = p.check("limit_fill_price", mine, [p.limit_fill_price(*cs) for cs in cases])
pd.DataFrame({"side": ["buy 100 lmt"] * 5 + ["sell 101 lmt"] * 5, "low": [b["low"] for b in bars_] * 2,
              "high": [b["high"] for b in bars_] * 2, "fill": mine})"""),
    ("md", "Why it matters: every day, bid 1% below yesterday's close and sell at today's close. Require the price to trade through the limit "
           "by more and more ticks, and watch the average P&L per fill. The fills you lose are the ones where the price touched your level "
           "and bounced: the best ones. That's **adverse selection**, and it's why limit-order strategies look better in backtests than live."),
    ("code", """bars = p.regime_market()
o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
rows = []
for k in (0, 1, 3, 10):
    fills = []
    for t in range(1, len(c)):
        lim = round(c[t - 1] * 0.99, 2)
        if round(l[t], 2) <= lim - k * 0.01:
            fills.append(c[t] / min(o[t], lim) - 1)
    rows.append({"require the low this far through (ticks)": k, "fills": len(fills), "mean P&L per fill (bp)": np.mean(fills) * 1e4})
pd.DataFrame(rows).round(1)"""),
    ("md", "## 2. Stop orders and gaps\n\n"
           "A stop becomes a market order once triggered. If the market **opens beyond** the stop, you get the open, not your stop: a sell "
           "stop triggers if the low reaches it (`low <= stop`) and fills at `min(open, stop)`; a buy stop if `high >= stop` at `max(open, stop)`."),
    ("md", YOUR_TURN),
    ("ex", """def stop_fill_price(qty, stop, bar):
    if qty > 0:
        return ...                                # ✍️ buy stop
    return ...                                    # ✍️ sell stop

cases = [(-100, 95.0, {"open": 99.0, "high": 100.0, "low": 94.0, "close": 96.0}),    # triggered during the day
         (-100, 95.0, {"open": 90.0, "high": 92.0, "low": 88.0, "close": 91.0}),     # gap down through the stop
         (-100, 95.0, {"open": 99.0, "high": 100.0, "low": 96.0, "close": 97.0}),    # not triggered
         (100, 105.0, {"open": 108.0, "high": 110.0, "low": 107.0, "close": 109.0})] # gap up through a buy stop
mine = [p.attempt(stop_fill_price, *cs) for cs in cases]
mine = p.check("stop_fill_price", mine, [p.stop_fill_price(*cs) for cs in cases])
mine""",
     """def stop_fill_price(qty, stop, bar):
    if qty > 0:
        return max(bar["open"], stop) if bar["high"] >= stop else None
    return min(bar["open"], stop) if bar["low"] <= stop else None

cases = [(-100, 95.0, {"open": 99.0, "high": 100.0, "low": 94.0, "close": 96.0}),    # triggered during the day
         (-100, 95.0, {"open": 90.0, "high": 92.0, "low": 88.0, "close": 91.0}),     # gap down through the stop
         (-100, 95.0, {"open": 99.0, "high": 100.0, "low": 96.0, "close": 97.0}),    # not triggered
         (100, 105.0, {"open": 108.0, "high": 110.0, "low": 107.0, "close": 109.0})] # gap up through a buy stop
mine = [p.attempt(stop_fill_price, *cs) for cs in cases]
mine = p.check("stop_fill_price", mine, [p.stop_fill_price(*cs) for cs in cases])
mine"""),
    ("md", "## 3. Market impact: the square-root law\n\n"
           "Big orders move the price. The empirical rule: impact ≈ `k · σ_daily · √(Q / ADV)`, in basis points `1e4 · k · σ · √(|Q| / ADV)`, "
           "where `Q` is the order size and `ADV` the average daily volume, in the same units."),
    ("md", YOUR_TURN),
    ("ex", """def sqrt_impact_bps(order_shares, adv_shares, daily_vol, k=1.0):
    return ...                                    # ✍️

sizes = [1_000, 10_000, 100_000, 1_000_000]
mine = [p.attempt(sqrt_impact_bps, q, 5_000_000, 0.015) for q in sizes]
mine = p.check("sqrt_impact_bps", mine, [p.sqrt_impact_bps(q, 5_000_000, 0.015) for q in sizes])
pd.DataFrame({"order (shares)": sizes, "% of ADV": [q / 5e6 * 100 for q in sizes], "impact (bp)": mine,
              "commission (bp) at $100": [p.ib_fixed_commission(q, 100.0) / (q * 100.0) * 1e4 for q in sizes]}).round(2)""",
     """def sqrt_impact_bps(order_shares, adv_shares, daily_vol, k=1.0):
    return float(1e4 * k * daily_vol * np.sqrt(abs(order_shares) / adv_shares))

sizes = [1_000, 10_000, 100_000, 1_000_000]
mine = [p.attempt(sqrt_impact_bps, q, 5_000_000, 0.015) for q in sizes]
mine = p.check("sqrt_impact_bps", mine, [p.sqrt_impact_bps(q, 5_000_000, 0.015) for q in sizes])
pd.DataFrame({"order (shares)": sizes, "% of ADV": [q / 5e6 * 100 for q in sizes], "impact (bp)": mine,
              "commission (bp) at $100": [p.ib_fixed_commission(q, 100.0) / (q * 100.0) * 1e4 for q in sizes]}).round(2)"""),
    ("md", "Commissions are a rounding error for size; impact grows with the square root and becomes *the* cost. Capacity (notebook 04) "
           "follows from this.\n\n"
           "## 4. Parity: two implementations, one answer\n\n"
           "The vectorized first look (Part 7) and the event-driven engine must agree when costs are off. A difference means one of them is "
           "wrong, and you want to know which before trusting either. Then turn costs on and see who survives."),
    ("code", """sigs = {"SMA 20/100": p.sma_cross_signal(c, 20, 100), "TSMOM 120": p.tsmom_signal(c),
        "daily reversal": np.r_[0.0, np.where(c[1:] < c[:-1], 1.0, 0.0)]}
rows = {}
for name, s in sigs.items():
    fast = p.first_look_pnl(s, o, cost_bps=0)
    eng = p.engine_returns(bars, s)
    rows[name] = {"corr": np.corrcoef(fast, eng)[0, 1], "max |diff| (bp)": np.abs(fast - eng).max() * 1e4,
                  **{f"Sharpe @ {bps} bp": p.sharpe(p.engine_returns(bars, s, slippage_bps=bps)) for bps in (0, 5, 10)}}
pd.DataFrame(rows).T.round(3)"""),
    ("md", "The remaining difference comes from whole shares and from the engine compounding its equity; it is tiny and explained, which is "
           "what a parity test is for. Costs barely move the slow strategies; the fast one, already losing before costs, sinks from about "
           "−0.5 to −1.3 at 10 bp: a high-turnover idea has to clear a much higher bar.\n\n"
           "## Wrap-up\n\n"
           "* Limit fills need trade-through; stop fills pay the gap; impact grows like √(size / ADV).\n"
           "* Parity-test every engine against an independent implementation, then stress costs (2× is part of the robustness scorecard).\n"
           "* Graded versions: `labs/part08/week25_engine` and Clinic W1 (parity and cost sensitivity)."),
]

# ---------------------------------------------------------------------------------------------- 03
NB["03_biases_and_tearsheet"] = [
    header("03", "Survivorship bias and the tear sheet", "S5 (Backtesting biases & pitfalls) · S6 (Performance analysis)",
           "1. Build a point-in-time universe and measure what survivorship bias adds.\n"
           "2. Compute the maximum drawdown and the longest time under water.\n"
           "3. Add Sortino and Calmar to a tear sheet, and read five strategies side by side."),
    ("code", SETUP),
    ("md", "## 1. Survivorship bias\n\n"
           "Sixty stocks over ten years; some go bust and are **delisted** (their price series ends), a few list later. If you backtest on "
           "the stocks that exist *today*, you have silently removed every loser from the past. A point-in-time universe on date `d` holds "
           "the symbols with `start <= d` and (`end` is missing or `d < end`)."),
    ("code", """closes, membership = p.stock_universe()
membership[membership.end.notna() | (membership.start > closes.index[0])]"""),
    ("md", YOUR_TURN),
    ("ex", """def members(membership, date):
    d = pd.Timestamp(date)
    m = ...                                       # ✍️ the rows alive on d (listed, and not yet delisted)
    return sorted(m["symbol"])

dates = ["2014-01-02", "2017-06-30", "2020-03-02", "2023-12-29"]
res = [p.attempt(members, membership, d) for d in dates]
mine = [len(x) for x in res] if all(x is not Ellipsis for x in res) else Ellipsis
mine = p.check("point-in-time universe sizes", mine, [len(p.members(membership, d)) for d in dates])
dict(zip(dates, mine))""",
     """def members(membership, date):
    d = pd.Timestamp(date)
    m = membership[(membership["start"] <= d) & (membership["end"].isna() | (d < membership["end"]))]
    return sorted(m["symbol"])

dates = ["2014-01-02", "2017-06-30", "2020-03-02", "2023-12-29"]
res = [p.attempt(members, membership, d) for d in dates]
mine = [len(x) for x in res] if all(x is not Ellipsis for x in res) else Ellipsis
mine = p.check("point-in-time universe sizes", mine, [len(p.members(membership, d)) for d in dates])
dict(zip(dates, mine))"""),
    ("code", """rets = closes.pct_change(fill_method=None)
survivors = [s for s in closes.columns if s in p.members(membership, closes.index[-1])]
biased = rets[survivors].mean(axis=1).fillna(0)                               # today's members, all history
pit = rets.apply(lambda row: row[[s for s in p.members(membership, row.name) if pd.notna(row.get(s))]].mean(), axis=1).fillna(0)
for name, r in [("survivors only (biased)", biased), ("point-in-time universe", pit)]:
    print(f"{name:26s} CAGR {p.tear_sheet(r)['cagr']:+.2%}  Sharpe {p.tear_sheet(r)['sharpe']:.2f}")"""),
    ("md", "## 2. Drawdown and time under water\n\n"
           "Compound the returns into equity, compare each point with the running peak (starting from 1.0), and report the **worst** "
           "drawdown (a negative number) and the **longest run** of consecutive bars below a previous peak."),
    ("md", YOUR_TURN),
    ("ex", """def max_drawdown(returns):
    eq = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1
    longest = run = 0
    for x in dd:
        ...                                       # ✍️ count consecutive bars under water, keep the longest run
    return float(dd.min()), int(longest)

strats = p.strategy_returns()
mine = [p.attempt(max_drawdown, strats[s]) for s in strats]
mine = p.check("max_drawdown", mine, [p.max_drawdown(strats[s]) for s in strats])
pd.DataFrame(mine, index=strats.columns, columns=["max drawdown", "longest under water (days)"])""",
     """def max_drawdown(returns):
    eq = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1
    longest = run = 0
    for x in dd:
        run = run + 1 if x < 0 else 0
        longest = max(longest, run)
    return float(dd.min()), int(longest)

strats = p.strategy_returns()
mine = [p.attempt(max_drawdown, strats[s]) for s in strats]
mine = p.check("max_drawdown", mine, [p.max_drawdown(strats[s]) for s in strats])
pd.DataFrame(mine, index=strats.columns, columns=["max drawdown", "longest under water (days)"])"""),
    ("md", "Five strategies with **known** annual Sharpe ratios (0.8, 0.6, 0.5, 0.4, 0.3). Even the best one spends over a year under "
           "water. Tell your future self before going live.\n\n"
           "## 3. Sortino and Calmar\n\n"
           "* **Sortino** = mean / downside deviation × √252, where downside deviation = `√mean(min(r, 0)²)` (only losses count as risk);\n"
           "* **Calmar** = CAGR / |max drawdown|."),
    ("md", YOUR_TURN),
    ("ex", """def sortino_calmar(returns):
    r = np.asarray(returns, dtype=float)
    cagr = np.prod(1 + r) ** (252 / r.size) - 1
    mdd = p.max_drawdown(r)[0]
    sortino = ...                                 # ✍️
    calmar = ...                                  # ✍️
    return float(sortino), float(calmar)

mine = [p.attempt(sortino_calmar, strats[s]) for s in strats]
mine = p.check("sortino and calmar", mine, [(p.tear_sheet(strats[s])["sortino"], p.tear_sheet(strats[s])["calmar"]) for s in strats])
pd.DataFrame({s: p.tear_sheet(strats[s]) for s in strats}).T.round(3)""",
     """def sortino_calmar(returns):
    r = np.asarray(returns, dtype=float)
    cagr = np.prod(1 + r) ** (252 / r.size) - 1
    mdd = p.max_drawdown(r)[0]
    sortino = r.mean() / np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    calmar = cagr / abs(mdd)
    return float(sortino), float(calmar)

mine = [p.attempt(sortino_calmar, strats[s]) for s in strats]
mine = p.check("sortino and calmar", mine, [(p.tear_sheet(strats[s])["sortino"], p.tear_sheet(strats[s])["calmar"]) for s in strats])
pd.DataFrame({s: p.tear_sheet(strats[s]) for s in strats}).T.round(3)"""),
    ("code", """m = p.monthly_table(strats["S1"]) * 100
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(m.to_numpy(), cmap="RdBu", vmin=-6, vmax=6, aspect="auto")
ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]); ax.set_yticks(range(len(m)), m.index)
ax.set_title("S1 monthly returns, % (true Sharpe 0.8)"); plt.colorbar(im); plt.show()
print(f"negative months: {(m < 0).to_numpy().sum()} of {m.notna().to_numpy().sum()}")"""),
    ("md", "Compare each estimated Sharpe in the tear sheet with the true value it was generated with: ten years of data still misses "
           "by several tenths. Notebook 04 turns that into a standard error.\n\n"
           "## Wrap-up\n\n"
           "* Point-in-time universes, always; delisted names stay in the history.\n"
           "* A tear sheet shows return, risk, drawdown depth and length, and tail shape, not just a Sharpe.\n"
           "* Graded version: `labs/part08/week26_analysis` (plus trade statistics with MAE/MFE)."),
]

# ---------------------------------------------------------------------------------------------- 04
NB["04_significance_and_capacity"] = [
    header("04", "Is the Sharpe real? Significance and capacity", "S7 (Statistical significance) · S8 (Capacity & backtest-vs-paper reconciliation)",
           "1. Put an error bar on a Sharpe ratio, analytically and by bootstrap.\n"
           "2. Compute the Probabilistic Sharpe Ratio and the minimum track record length.\n"
           "3. Find the capital at which impact costs kill a strategy."),
    ("code", SETUP),
    ("code", """strats = p.strategy_returns()
true_sr = dict(zip(strats.columns, (0.8, 0.6, 0.5, 0.4, 0.3)))
pd.DataFrame({"true Sharpe": true_sr, "estimated (10 years)": {s: p.sharpe(strats[s]) for s in strats}}).round(2)"""),
    ("md", "## 1. The Sharpe ratio's error bar\n\n"
           "With i.i.d. returns the standard error of a per-period Sharpe `SR` over `T` periods is `√((1 + SR²/2) / T)` (Lo, 2002). In annual "
           "units multiply by √252. For ten years of a 0.8 Sharpe strategy that is about ±0.32: a 95% interval from roughly 0.2 to 1.4."),
    ("code", """r = strats["S1"].to_numpy()
sr_d = r.mean() / r.std(ddof=1)
se = p.sharpe_se(sr_d, len(r)) * np.sqrt(252)
boot = p.stationary_bootstrap_sharpes(r, n_boot=400) * np.sqrt(252)
print(f"S1: Sharpe {sr_d * np.sqrt(252):.2f} ± {se:.2f} (analytic);  bootstrap 95% interval {np.percentile(boot, 2.5):.2f} to {np.percentile(boot, 97.5):.2f}")
plt.hist(boot, bins=30); plt.axvline(0.8, color="black", lw=1, label="true 0.8"); plt.legend(); plt.title("Stationary bootstrap of S1's Sharpe"); plt.show()"""),
    ("md", "## 2. The Probabilistic Sharpe Ratio\n\n"
           "PSR is the probability that the true Sharpe exceeds a benchmark `SR*`, correcting for the sample length **and** for skew and fat "
           "tails: `Φ((SR − SR*)·√(T − 1) / √(1 − γ3·SR + (γ4 − 1)/4·SR²))`, with per-period `SR` (ddof=1), `γ3` the skewness and `γ4` the "
           "**non-excess** kurtosis (`scipy.stats.kurtosis(r, fisher=False)`, 3 for a normal)."),
    ("md", YOUR_TURN),
    ("ex", """from scipy.stats import norm, skew, kurtosis

def psr(returns, sr_benchmark=0.0):
    r = np.asarray(returns, dtype=float)
    T = r.size
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return ...                                    # ✍️ Φ(...)

mine = [p.attempt(psr, strats[s]) for s in strats]
mine = p.check("psr", mine, [p.psr(strats[s]) for s in strats])
pd.Series(mine, index=strats.columns, name="P(true Sharpe > 0)").round(4)""",
     """from scipy.stats import norm, skew, kurtosis

def psr(returns, sr_benchmark=0.0):
    r = np.asarray(returns, dtype=float)
    T = r.size
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))

mine = [p.attempt(psr, strats[s]) for s in strats]
mine = p.check("psr", mine, [p.psr(strats[s]) for s in strats])
pd.Series(mine, index=strats.columns, name="P(true Sharpe > 0)").round(4)"""),
    ("md", "Turn it around: how long must a track record be before a Sharpe this size is significant at 95%? "
           "`MinTRL = 1 + (1 − γ3·SR + (γ4 − 1)/4·SR²) · (z_{0.95} / (SR − SR*))²` periods (infinite if `SR <= SR*`)."),
    ("md", YOUR_TURN),
    ("ex", """def min_track_record(returns, sr_benchmark=0.0, alpha=0.05):
    r = np.asarray(returns, dtype=float)
    sr = r.mean() / r.std(ddof=1)
    if sr <= sr_benchmark:
        return float("inf")
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return ...                                    # ✍️ in periods (days)

mine = [p.attempt(min_track_record, strats[s]) for s in strats]
mine = p.check("min_track_record", mine, [p.min_track_record(strats[s]) for s in strats])
pd.Series(np.array(mine) / 252, index=strats.columns, name="years needed at 95%").round(1)""",
     """def min_track_record(returns, sr_benchmark=0.0, alpha=0.05):
    r = np.asarray(returns, dtype=float)
    sr = r.mean() / r.std(ddof=1)
    if sr <= sr_benchmark:
        return float("inf")
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (norm.ppf(1 - alpha) / (sr - sr_benchmark)) ** 2)

mine = [p.attempt(min_track_record, strats[s]) for s in strats]
mine = p.check("min_track_record", mine, [p.min_track_record(strats[s]) for s in strats])
pd.Series(np.array(mine) / 252, index=strats.columns, name="years needed at 95%").round(1)"""),
    ("md", "## 3. Capacity\n\n"
           "A strategy's edge is fixed; its costs grow with size. With turnover `τ` (fraction of capital traded per day) and the square-root "
           "law, the daily cost is `τ · k · σ_daily · √(τ · C / ADV)` for capital `C`. `p.net_sharpe_vs_capital` gives the net Sharpe for a list "
           "of capitals. **Capacity** is the largest capital whose net Sharpe is still at least `min_sharpe` (0 if none)."),
    ("md", YOUR_TURN),
    ("ex", """def capacity(sharpe_by_capital, min_sharpe=0.5):
    ok = ...                                      # ✍️ the entries at or above min_sharpe
    return float(ok.index.max()) if len(ok) else 0.0

caps = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]
curves = {name: p.net_sharpe_vs_capital(strats["S1"], turnover, adv_dollars=5e8, daily_vol=0.015, capitals=caps)
          for name, turnover in [("slow (5% a day)", 0.05), ("fast (100% a day)", 1.0)]}
mine = [p.attempt(capacity, cv) for cv in curves.values()]
mine = p.check("capacity", mine, [p.capacity(cv) for cv in curves.values()])
fig, ax = plt.subplots()
for (name, cv), cap in zip(curves.items(), mine):
    ax.semilogx(cv.index, cv.values, "o-", label=f"{name}: capacity ${cap:,.0f}")
ax.axhline(0.5, color=p.PALETTE[7], ls="--", lw=1); ax.set(xlabel="capital, $", ylabel="net Sharpe"); ax.legend(); plt.show()""",
     """def capacity(sharpe_by_capital, min_sharpe=0.5):
    ok = sharpe_by_capital[sharpe_by_capital >= min_sharpe]
    return float(ok.index.max()) if len(ok) else 0.0

caps = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]
curves = {name: p.net_sharpe_vs_capital(strats["S1"], turnover, adv_dollars=5e8, daily_vol=0.015, capitals=caps)
          for name, turnover in [("slow (5% a day)", 0.05), ("fast (100% a day)", 1.0)]}
mine = [p.attempt(capacity, cv) for cv in curves.values()]
mine = p.check("capacity", mine, [p.capacity(cv) for cv in curves.values()])
fig, ax = plt.subplots()
for (name, cv), cap in zip(curves.items(), mine):
    ax.semilogx(cv.index, cv.values, "o-", label=f"{name}: capacity ${cap:,.0f}")
ax.axhline(0.5, color=p.PALETTE[7], ls="--", lw=1); ax.set(xlabel="capital, $", ylabel="net Sharpe"); ax.legend(); plt.show()"""),
    ("md", "Same gross returns, same market: turnover alone decides how much money the idea can take.\n\n"
           "## Wrap-up\n\n"
           "* Report every Sharpe with an interval; PSR corrects for sample length, skew and tails.\n"
           "* Low-Sharpe strategies need decades to prove themselves: combine evidence, don't wait for significance alone.\n"
           "* Capacity comes from turnover and impact, not from the idea.\n"
           "* Graded version: `labs/part08/week26_analysis` (trade reshuffling, implementation shortfall)."),
]

# ---------------------------------------------------------------------------------------------- 05
NB["05_search_and_stability"] = [
    header("05", "Parameter search, stability and trade-offs", "S9 (Search spaces, objectives & stability) · S10 (Bayesian, genetic & multi-objective optimization)",
           "1. Run a grid search and log every trial.\n"
           "2. Score parameters by the plateau around them, not by their peak.\n"
           "3. Find the Pareto front between Sharpe and turnover.\n"
           "4. Compare grid and random search on the same budget."),
    ("code", SETUP),
    ("md", "## 1. The objective\n\n"
           "Time-series momentum (Part 7) with two parameters: the momentum `lookback` and the volatility window `vol_n`. Optimize on the "
           "**first half** of the known-regime market only; keep the second half for later."),
    ("code", """bars = p.regime_market()
o, c = bars.open.to_numpy(), bars.close.to_numpy()
half = len(c) // 2

def pnl(lookback, vol_n):
    return p.first_look_pnl(p.tsmom_signal(c, lookback, vol_n), o)

def in_sample_sharpe(lookback, vol_n):
    return p.sharpe(pnl(lookback, vol_n)[:half])

grid = {"lookback": [20, 40, 60, 90, 120, 160, 200, 250], "vol_n": [10, 20, 40, 60, 90, 120]}"""),
    ("md", "## 2. Grid search\n\n"
           "Evaluate the objective on **every combination** (`itertools.product` over the grid's values, in the grid's key order) and return a "
           "DataFrame with one column per parameter plus `score`, one row per combination in evaluation order."),
    ("md", YOUR_TURN),
    ("ex", """import itertools

def grid_search(objective, grid):
    keys = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        ...                                       # ✍️ append params plus {"score": objective(**params)}
    return pd.DataFrame(rows)

mine = p.attempt(grid_search, in_sample_sharpe, grid)
mine = p.check("grid_search", mine, p.grid_search(in_sample_sharpe, grid))
mine.pivot(index="vol_n", columns="lookback", values="score").round(2)""",
     """import itertools

def grid_search(objective, grid):
    keys = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        rows.append(params | {"score": objective(**params)})
    return pd.DataFrame(rows)

mine = p.attempt(grid_search, in_sample_sharpe, grid)
mine = p.check("grid_search", mine, p.grid_search(in_sample_sharpe, grid))
mine.pivot(index="vol_n", columns="lookback", values="score").round(2)"""),
    ("md", "48 trials. The best in-sample score is the most optimistic number in the table: it is the maximum of 48 noisy estimates.\n\n"
           "## 3. Peaks and plateaus\n\n"
           "A parameter set on an isolated spike is probably fitted to noise; one in the middle of a broad high region survives small "
           "changes. Replace each cell of the `(vol_n × lookback)` table by the **mean of its neighbourhood**: the cells within `radius` grid "
           "steps in both directions, clipped at the edges (`np.nanmean`)."),
    ("md", YOUR_TURN),
    ("ex", """def plateau_scores(results, x, y, radius=1):
    tab = results.pivot(index=y, columns=x, values="score").sort_index().sort_index(axis=1)
    a = tab.to_numpy()
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = ...                       # ✍️ mean of the (2r+1)×(2r+1) window around (i, j), clipped
    return pd.DataFrame(out, index=tab.index, columns=tab.columns)

res = p.grid_search(in_sample_sharpe, grid)
mine = p.attempt(plateau_scores, res, "lookback", "vol_n")
mine = p.check("plateau_scores", mine, p.plateau_scores(res, "lookback", "vol_n"))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (t, tab) in zip(axes, [("raw in-sample Sharpe", res.pivot(index="vol_n", columns="lookback", values="score")), ("plateau score", mine)]):
    im = ax.imshow(tab.to_numpy(), cmap="viridis", aspect="auto"); plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(tab.columns)), tab.columns); ax.set_yticks(range(len(tab.index)), tab.index)
    ax.set(xlabel="lookback", ylabel="vol_n", title=t)
plt.tight_layout(); plt.show()""",
     """def plateau_scores(results, x, y, radius=1):
    tab = results.pivot(index=y, columns=x, values="score").sort_index().sort_index(axis=1)
    a = tab.to_numpy()
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = np.nanmean(a[max(0, i - radius):i + radius + 1, max(0, j - radius):j + radius + 1])
    return pd.DataFrame(out, index=tab.index, columns=tab.columns)

res = p.grid_search(in_sample_sharpe, grid)
mine = p.attempt(plateau_scores, res, "lookback", "vol_n")
mine = p.check("plateau_scores", mine, p.plateau_scores(res, "lookback", "vol_n"))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (t, tab) in zip(axes, [("raw in-sample Sharpe", res.pivot(index="vol_n", columns="lookback", values="score")), ("plateau score", mine)]):
    im = ax.imshow(tab.to_numpy(), cmap="viridis", aspect="auto"); plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(tab.columns)), tab.columns); ax.set_yticks(range(len(tab.index)), tab.index)
    ax.set(xlabel="lookback", ylabel="vol_n", title=t)
plt.tight_layout(); plt.show()"""),
    ("code", """pl = p.plateau_scores(res, "lookback", "vol_n")
peak = res.loc[res.score.idxmax()]
i, j = np.unravel_index(np.nanargmax(pl.to_numpy()), pl.shape)
choices = {"peak": (int(peak.lookback), int(peak.vol_n)), "plateau": (int(pl.columns[j]), int(pl.index[i]))}
oos = {(r.lookback, r.vol_n): p.sharpe(pnl(int(r.lookback), int(r.vol_n))[half:]) for r in res.itertuples()}
for k, (lb, vn) in choices.items():
    print(f"{k:8s} lookback {lb:3d}, vol_n {vn:3d}: in-sample {in_sample_sharpe(lb, vn):.2f}, out-of-sample {oos[(lb, vn)]:.2f}")
print(f"correlation of in-sample and out-of-sample Sharpe across the 48 trials: {np.corrcoef(res.score, list(oos.values()))[0, 1]:.2f}")"""),
    ("md", "Here the best lookback (160) is a whole **ridge**, so the peak and the plateau choice agree and both hold up out of sample. That "
           "is the good case. When the peak is an isolated spike, the plateau score moves you to safer ground; it costs nothing when it "
           "isn't needed.\n\n"
           "## 4. More than one objective\n\n"
           "Sharpe isn't the only thing that matters: turnover costs money and capacity (notebook 04). A configuration is on the **Pareto "
           "front** if no other configuration is at least as good on both criteria and strictly better on one. Return the non-dominated rows "
           "sorted by turnover."),
    ("code", """rows = []
for lb in grid["lookback"]:
    for vn in grid["vol_n"]:
        sig = p.tsmom_signal(c, lb, vn)
        rows.append({"lookback": lb, "vol_n": vn, "sharpe": p.sharpe(p.first_look_pnl(sig, o)),
                     "turnover": np.abs(np.diff(sig)).sum() / len(sig) * 252})
trials = pd.DataFrame(rows)"""),
    ("md", YOUR_TURN),
    ("ex", """def pareto_front(df, maximize, minimize):
    a, b = df[maximize].to_numpy(), df[minimize].to_numpy()
    keep = []
    for i in range(len(df)):
        dominated = ...                           # ✍️ does any row have a >= a[i] and b <= b[i], and be strictly better on one?
        if not dominated:
            keep.append(i)
    return df.iloc[keep].sort_values(minimize)

mine = p.attempt(pareto_front, trials, "sharpe", "turnover")
mine = p.check("pareto_front", mine, p.pareto_front(trials, "sharpe", "turnover"))
plt.scatter(trials.turnover, trials.sharpe, s=15, color="#b5b4ad", label="all 48 trials")
plt.plot(mine.turnover, mine.sharpe, "o-", color=p.PALETTE[1], label="Pareto front")
plt.xlabel("turnover (× capital per year)"); plt.ylabel("Sharpe"); plt.legend(); plt.show()
mine.round(3)""",
     """def pareto_front(df, maximize, minimize):
    a, b = df[maximize].to_numpy(), df[minimize].to_numpy()
    keep = []
    for i in range(len(df)):
        dominated = np.any((a >= a[i]) & (b <= b[i]) & ((a > a[i]) | (b < b[i])))
        if not dominated:
            keep.append(i)
    return df.iloc[keep].sort_values(minimize)

mine = p.attempt(pareto_front, trials, "sharpe", "turnover")
mine = p.check("pareto_front", mine, p.pareto_front(trials, "sharpe", "turnover"))
plt.scatter(trials.turnover, trials.sharpe, s=15, color="#b5b4ad", label="all 48 trials")
plt.plot(mine.turnover, mine.sharpe, "o-", color=p.PALETTE[1], label="Pareto front")
plt.xlabel("turnover (× capital per year)"); plt.ylabel("Sharpe"); plt.legend(); plt.show()
mine.round(3)"""),
    ("md", "## 5. Grid vs random search\n\n"
           "With the same budget of 48 trials, random search tries 48 *different* values of each parameter instead of 8 and 6. When only one "
           "or two parameters matter, that finds better regions (Bergstra & Bengio, 2012). Bayesian optimizers (Optuna, used in the lab) go "
           "further by learning where to look next. All of them make the best score **more** optimistic, which is why every trial is logged."),
    ("code", """space = {"lookback": list(range(20, 251, 10)), "vol_n": list(range(10, 121, 5))}
rnd = p.random_search(in_sample_sharpe, space, n=48, seed=0)
print(f"grid:   best in-sample {res.score.max():.2f} from {res.lookback.nunique()}×{res.vol_n.nunique()} distinct values")
print(f"random: best in-sample {rnd.score.max():.2f} from {rnd.lookback.nunique()}×{rnd.vol_n.nunique()} distinct values")"""),
    ("md", "## Wrap-up\n\n"
           "* Declare the search space, log every trial, and score stability (plateaus), not peaks.\n"
           "* Trade-offs are a front, not a single number.\n"
           "* Graded version: `labs/part08/week27_optimization` (grid, random and Optuna search, plateau scores)."),
]

# ---------------------------------------------------------------------------------------------- 06
NB["06_walk_forward_and_cv"] = [
    header("06", "Walk-forward analysis and purged cross-validation", "S11 (Walk-forward analysis) · S12 (Purged k-fold & CPCV)",
           "1. Generate rolling walk-forward windows, and measure walk-forward efficiency.\n"
           "2. See a model that knows only the date score R² = 0.89 on pure noise with shuffled k-fold.\n"
           "3. Purge and embargo the folds, and the leak disappears.\n"
           "4. Count the backtest paths combinatorial purged CV produces."),
    ("code", SETUP),
    ("md", "## 1. Walk-forward windows\n\n"
           "Optimize on `train` bars, trade the next `test` bars, move forward by `test`, repeat. Rolling windows keep `train` bars; anchored "
           "ones start at 0. Yield `(train_idx, test_idx)` as integer arrays, and stop when a full test window no longer fits."),
    ("md", YOUR_TURN),
    ("ex", """def walk_forward(n, train, test, anchored=False):
    start = 0
    while start + train + test <= n:
        tr0 = 0 if anchored else start
        yield ...                                 # ✍️ (train indices, test indices) as np.arange
        start += test

cases = [(20, 8, 4, False), (20, 8, 4, True), (3000, 750, 250, False)]
mine = [p.attempt(lambda *a: [(tr.tolist(), te.tolist()) for tr, te in walk_forward(*a)], *cs) for cs in cases]
mine = p.check("walk_forward", mine, [[(tr.tolist(), te.tolist()) for tr, te in p.walk_forward(*cs)] for cs in cases])
for tr, te in mine[0]:
    print(f"train {tr[0]:2d}–{tr[-1]:2d}   test {te[0]:2d}–{te[-1]:2d}")""",
     """def walk_forward(n, train, test, anchored=False):
    start = 0
    while start + train + test <= n:
        tr0 = 0 if anchored else start
        yield np.arange(tr0, start + train), np.arange(start + train, start + train + test)
        start += test

cases = [(20, 8, 4, False), (20, 8, 4, True), (3000, 750, 250, False)]
mine = [p.attempt(lambda *a: [(tr.tolist(), te.tolist()) for tr, te in walk_forward(*a)], *cs) for cs in cases]
mine = p.check("walk_forward", mine, [[(tr.tolist(), te.tolist()) for tr, te in p.walk_forward(*cs)] for cs in cases])
for tr, te in mine[0]:
    print(f"train {tr[0]:2d}–{tr[-1]:2d}   test {te[0]:2d}–{te[-1]:2d}")"""),
    ("md", "In each window, pick the TSMOM parameters with the best in-sample Sharpe, then record what they earn out of sample. "
           "**Walk-forward efficiency** (WFE) = mean OOS Sharpe / mean IS Sharpe: near 1 means the in-sample choice carried over; near 0 "
           "means it didn't."),
    ("code", """bars = p.regime_market()
o, c = bars.open.to_numpy(), bars.close.to_numpy()
wf = p.walk_forward_optimize(lambda lookback: p.first_look_pnl(p.tsmom_signal(c, lookback), o),
                             {"lookback": [40, 80, 120, 180, 250]}, len(c), 750, 250)
display(pd.DataFrame({"chosen lookback": [d["lookback"] for d in wf["params"]], "in-sample Sharpe": wf["is_sharpe"],
                      "out-of-sample Sharpe": wf["oos_sharpe"]}).round(2).T)
print(f"stitched out-of-sample Sharpe {p.sharpe(wf['oos']):.2f}; walk-forward efficiency {wf['wfe']:.2f}")"""),
    ("md", "## 2. Why ordinary k-fold lies on financial data\n\n"
           "A typical label is a **forward return** over several days, so neighbouring labels overlap. Here the returns are pure noise "
           "(nothing is predictable) and each label is the sum of the next 20 daily returns. The \"model\" is a 1-nearest-neighbour whose "
           "only feature is the **date**: it predicts each test label with the label of the closest training day. It can only succeed by "
           "leakage."),
    ("code", """r, y = p.overlapping_labels(horizon=20)
for name, splits in [("shuffled k-fold (sklearn KFold(shuffle=True))", p.shuffled_kfold(len(y))),
                     ("contiguous k-fold, no purging", p.contiguous_kfold(len(y)))]:
    print(f"{name:46s} out-of-sample R² = {p.nearest_neighbour_r2(y, splits):+.2f}")"""),
    ("md", "Shuffled k-fold gives a model with **no information** an R² of about 0.9: every test day has a training neighbour whose label "
           "shares 19 of its 20 days. Contiguous folds leak only at their edges.\n\n"
           "## 3. Purging and embargo\n\n"
           "With contiguous test folds, **purge** every training sample whose label window `[i, i + label_horizon]` reaches into the fold "
           "(keep `i` only if `i + label_horizon < fold start`), and **embargo** the `ceil(embargo × n)` samples right after the fold "
           "(keep `i` only if `i > fold end + emb`). Yield `(train_idx, test_idx)`."),
    ("md", YOUR_TURN),
    ("ex", """def purged_kfold(n, k=5, label_horizon=5, embargo=0.01):
    emb = int(np.ceil(embargo * n))
    idx = np.arange(n)
    for test in np.array_split(idx, k):
        lo, hi = test[0], test[-1]
        keep = ...                                # ✍️ boolean mask of the training samples to keep
        yield idx[keep], test

mine = p.attempt(lambda: [(tr.tolist(), te.tolist()) for tr, te in purged_kfold(100, 4, 5, 0.03)])
mine = p.check("purged_kfold", mine, [(tr.tolist(), te.tolist()) for tr, te in p.purged_kfold(100, 4, 5, 0.03)])
for tr, te in (mine[1:3]):
    gap = [i for i in range(100) if i not in tr and i not in te]
    print(f"test {te[0]}–{te[-1]}: removed from training {gap}")
print(f"purged 5-fold, horizon 20:  out-of-sample R² = {p.nearest_neighbour_r2(y, p.purged_kfold(len(y), 5, 20, 0.01)):+.2f}")""",
     """def purged_kfold(n, k=5, label_horizon=5, embargo=0.01):
    emb = int(np.ceil(embargo * n))
    idx = np.arange(n)
    for test in np.array_split(idx, k):
        lo, hi = test[0], test[-1]
        keep = (idx + label_horizon < lo) | (idx > hi + emb)
        yield idx[keep], test

mine = p.attempt(lambda: [(tr.tolist(), te.tolist()) for tr, te in purged_kfold(100, 4, 5, 0.03)])
mine = p.check("purged_kfold", mine, [(tr.tolist(), te.tolist()) for tr, te in p.purged_kfold(100, 4, 5, 0.03)])
for tr, te in (mine[1:3]):
    gap = [i for i in range(100) if i not in tr and i not in te]
    print(f"test {te[0]}–{te[-1]}: removed from training {gap}")
print(f"purged 5-fold, horizon 20:  out-of-sample R² = {p.nearest_neighbour_r2(y, p.purged_kfold(len(y), 5, 20, 0.01)):+.2f}")"""),
    ("md", "The leak is gone: the date-only model is now (correctly) worse than predicting the mean.\n\n"
           "## 4. Combinatorial purged CV\n\n"
           "Walk-forward gives **one** out-of-sample path, so one lucky or unlucky period decides everything. CPCV splits the data into `N` "
           "groups and tests on every combination of `k` of them (purged as above). Each group is tested `C(N−1, k−1)` times, so the "
           "out-of-sample results assemble into `C(N, k)·k / N` complete backtest paths."),
    ("md", YOUR_TURN),
    ("ex", """from math import comb

def cpcv_n_paths(n_groups, k_test):
    return ...                                    # ✍️ an int

cases = [(6, 2), (10, 2), (10, 3), (12, 4)]
mine = [p.attempt(cpcv_n_paths, *cs) for cs in cases]
mine = p.check("cpcv_n_paths", mine, [p.cpcv_n_paths(*cs) for cs in cases])
pd.DataFrame({"groups N": [a for a, _ in cases], "test groups k": [b for _, b in cases], "splits C(N,k)": [comb(a, b) for a, b in cases],
              "backtest paths": mine})""",
     """from math import comb

def cpcv_n_paths(n_groups, k_test):
    return comb(n_groups, k_test) * k_test // n_groups

cases = [(6, 2), (10, 2), (10, 3), (12, 4)]
mine = [p.attempt(cpcv_n_paths, *cs) for cs in cases]
mine = p.check("cpcv_n_paths", mine, [p.cpcv_n_paths(*cs) for cs in cases])
pd.DataFrame({"groups N": [a for a, _ in cases], "test groups k": [b for _, b in cases], "splits C(N,k)": [comb(a, b) for a, b in cases],
              "backtest paths": mine})"""),
    ("code", """splits = p.cpcv_splits(600, n_groups=6, k_test=2, label_horizon=10, embargo=0.02)
grid_img = np.zeros((len(splits), 600))
for row, (train, combo, test) in enumerate(splits):
    grid_img[row, train] = 1; grid_img[row, test] = 2
plt.figure(figsize=(10, 4)); plt.imshow(grid_img, aspect="auto", cmap="Greys", interpolation="nearest")
plt.xlabel("sample"); plt.ylabel("split"); plt.title("CPCV (N=6, k=2): black = test, grey = train, white = purged or embargoed"); plt.show()"""),
    ("md", "## Wrap-up\n\n"
           "* Walk-forward mimics how the strategy would really have been re-fitted; report OOS results and the WFE.\n"
           "* Never shuffle time series; purge overlapping labels and embargo after each test fold.\n"
           "* CPCV turns one OOS path into many, so you see a distribution instead of one draw.\n"
           "* Graded version: `labs/part08/week27_optimization` (walk-forward optimizer, purged k-fold, CPCV splits)."),
]

# ---------------------------------------------------------------------------------------------- 07
NB["07_deflated_sharpe_and_pbo"] = [
    header("07", "The Deflated Sharpe Ratio and the probability of backtest overfitting", "S13 (The Deflated Sharpe Ratio) · S14 (Probability of backtest overfitting, CSCV)",
           "1. Compute how high the best Sharpe of N worthless strategies is expected to be.\n"
           "2. Deflate a Sharpe ratio for the number of trials behind it.\n"
           "3. Measure the probability that the in-sample winner is a loser out of sample (PBO)."),
    ("code", SETUP),
    ("md", "## 1. The best of 200 worthless strategies\n\n"
           "Two hundred strategies of pure noise: every true Sharpe is exactly zero. Six years of daily data each. Pick the best."),
    ("code", """noise = p.noise_strategies(n_days=1500, n=200)
srs = np.array([p.sharpe(noise[:, i]) for i in range(noise.shape[1])])
best = int(np.argmax(srs))
print(f"best annual Sharpe {srs[best]:.2f} (true Sharpe: 0); {np.mean(srs > 0.5):.0%} of the 200 look better than 0.5")
plt.hist(srs, bins=30); plt.axvline(srs[best], color=p.PALETTE[7]); plt.title("Sharpe ratios of 200 noise strategies"); plt.show()"""),
    ("md", "How high should the best of `N` unskilled trials be? With `V` the variance of the trial Sharpes (per period, ddof=1) and "
           "`γ ≈ 0.5772` the Euler–Mascheroni constant (`p.EULER`):\n\n"
           "`E[max SR] ≈ √V · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))`"),
    ("md", YOUR_TURN),
    ("ex", """from scipy.stats import norm

def expected_max_sharpe(trial_srs):
    s = np.asarray(trial_srs, dtype=float)
    n, v = s.size, s.var(ddof=1)
    return ...                                    # ✍️

per_period = srs / np.sqrt(252)
cases = [per_period, per_period[:20], np.r_[per_period, -per_period]]     # 200, the first 20, and 400 (with mirror images)
mine = [p.attempt(expected_max_sharpe, s) for s in cases]
mine = p.check("expected_max_sharpe", mine, [p.expected_max_sharpe(s) for s in cases])
print(f"expected best annual Sharpe of pure noise: N=200 → {mine[0] * np.sqrt(252):.2f}; N=20 → {mine[1] * np.sqrt(252):.2f}; N=400 → {mine[2] * np.sqrt(252):.2f}")""",
     """from scipy.stats import norm

def expected_max_sharpe(trial_srs):
    s = np.asarray(trial_srs, dtype=float)
    n, v = s.size, s.var(ddof=1)
    return float(np.sqrt(v) * ((1 - p.EULER) * norm.ppf(1 - 1 / n) + p.EULER * norm.ppf(1 - 1 / (n * np.e))))

per_period = srs / np.sqrt(252)
cases = [per_period, per_period[:20], np.r_[per_period, -per_period]]     # 200, the first 20, and 400 (with mirror images)
mine = [p.attempt(expected_max_sharpe, s) for s in cases]
mine = p.check("expected_max_sharpe", mine, [p.expected_max_sharpe(s) for s in cases])
print(f"expected best annual Sharpe of pure noise: N=200 → {mine[0] * np.sqrt(252):.2f}; N=20 → {mine[1] * np.sqrt(252):.2f}; N=400 → {mine[2] * np.sqrt(252):.2f}")"""),
    ("md", "## 2. Deflating the winner\n\n"
           "The **Deflated Sharpe Ratio** is the PSR of the chosen strategy (notebook 04) with the benchmark raised from 0 to that expected "
           "maximum `SR0`. It answers: *given how many things I tried, how likely is this Sharpe to be real?* Return `(DSR, SR0)`; the usual "
           "bar is DSR >= 0.95."),
    ("md", YOUR_TURN),
    ("ex", """def deflated_sharpe(returns, trial_srs):
    sr0 = p.expected_max_sharpe(trial_srs)
    return ..., sr0                               # ✍️ the PSR against sr0

skilled = p.noise_strategies(n_days=1500, n=200, seed=7, skilled=1, skill_sharpe=1.5)[:, 0]
cases = [(noise[:, best], per_period), (skilled, per_period), (noise[:, best], per_period[:5])]
mine = [p.attempt(deflated_sharpe, *cs) for cs in cases]
mine = p.check("deflated_sharpe", mine, [p.deflated_sharpe(*cs) for cs in cases])
pd.DataFrame({"strategy": ["best of 200 noise", "a real Sharpe-1.5 strategy", "best of 200 noise, but claiming 5 trials"],
              "annual Sharpe": [p.sharpe(x) for x, _ in cases], "PSR vs 0": [p.psr(x) for x, _ in cases],
              "DSR": [d for d, _ in mine]}).round(3)""",
     """def deflated_sharpe(returns, trial_srs):
    sr0 = p.expected_max_sharpe(trial_srs)
    return p.psr(returns, sr0), sr0

skilled = p.noise_strategies(n_days=1500, n=200, seed=7, skilled=1, skill_sharpe=1.5)[:, 0]
cases = [(noise[:, best], per_period), (skilled, per_period), (noise[:, best], per_period[:5])]
mine = [p.attempt(deflated_sharpe, *cs) for cs in cases]
mine = p.check("deflated_sharpe", mine, [p.deflated_sharpe(*cs) for cs in cases])
pd.DataFrame({"strategy": ["best of 200 noise", "a real Sharpe-1.5 strategy", "best of 200 noise, but claiming 5 trials"],
              "annual Sharpe": [p.sharpe(x) for x, _ in cases], "PSR vs 0": [p.psr(x) for x, _ in cases],
              "DSR": [d for d, _ in mine]}).round(3)"""),
    ("md", "The ordinary PSR says the noise winner is very likely real. The DSR, which knows about the 199 siblings, says it isn't. Under-report "
           "the trials (the third row) and the DSR is fooled too: that's why the research log (notebook 01) records every run.\n\n"
           "The second row is the uncomfortable one: a strategy with a **true** Sharpe of 1.5 happened to realize only 0.75 over these six "
           "years, and as one of 200 tries it can't be told apart from luck either. Deflation is strict by design; the answer is more "
           "evidence (longer history, other markets, paper trading), not fewer recorded trials.\n\n"
           "## 3. The probability of backtest overfitting (CSCV)\n\n"
           "Split the history into `S` blocks. For every way of choosing half the blocks as in-sample: pick the configuration with the best "
           "in-sample Sharpe, and find its **relative rank** out of sample, `w = (rank among the N OOS Sharpes, 1 = worst) / (N + 1)`, and its "
           "logit `ln(w / (1 − w))`. PBO is the share of logits `<= 0`: how often the in-sample winner lands at or below the OOS median. "
           "Write the rank-and-logit step."),
    ("md", YOUR_TURN),
    ("code", "import itertools"),
    ("ex", """def pbo_cscv(perf, S=10):
    T, N = perf.shape
    blocks = np.array_split(np.arange(T), S)
    sr = lambda x: x.mean(0) / x.std(0, ddof=1)
    logits = []
    for is_blocks in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in is_blocks])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in is_blocks])
        best = int(np.argmax(sr(perf[is_idx])))
        rank = sr(perf[oos_idx]).argsort().argsort()[best] + 1        # 1 = worst out of sample
        logits.append(...)                                            # ✍️ the logit of the relative rank rank / (N + 1)
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits

configs = noise[:, :50]
mine = p.attempt(pbo_cscv, configs)
mine = p.check("pbo_cscv", mine, p.pbo_cscv(configs, S=10))
print(f"PBO of 50 noise configurations: {mine[0]:.2f}")""",
     """def pbo_cscv(perf, S=10):
    T, N = perf.shape
    blocks = np.array_split(np.arange(T), S)
    sr = lambda x: x.mean(0) / x.std(0, ddof=1)
    logits = []
    for is_blocks in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in is_blocks])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in is_blocks])
        best = int(np.argmax(sr(perf[is_idx])))
        rank = sr(perf[oos_idx]).argsort().argsort()[best] + 1        # 1 = worst out of sample
        logits.append(np.log(rank / (N + 1) / (1 - rank / (N + 1))))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits

configs = noise[:, :50]
mine = p.attempt(pbo_cscv, configs)
mine = p.check("pbo_cscv", mine, p.pbo_cscv(configs, S=10))
print(f"PBO of 50 noise configurations: {mine[0]:.2f}")"""),
    ("md", "One sample of noise gives a noisy PBO. Repeat it over several independent noise sets, and compare with sets where five of the "
           "50 configurations have a real edge:"),
    ("code", """rows = []
for seed in range(8):
    rows.append({"seed": seed, "50 noise configs": p.pbo_cscv(p.noise_strategies(1500, 50, seed=seed), S=10)[0],
                 "5 skilled + 45 noise": p.pbo_cscv(p.noise_strategies(1500, 50, seed=seed, skilled=5, skill_sharpe=1.5), S=10)[0]})
t = pd.DataFrame(rows).set_index("seed")
display(t.round(2).T)
print(f"mean PBO: noise {t.iloc[:, 0].mean():.2f}, with real skill {t.iloc[:, 1].mean():.2f}")"""),
    ("md", "Noise averages around 0.4–0.5 (the in-sample winner is roughly a coin flip out of sample), and any single estimate can land far from it. Real skill "
           "pushes PBO toward 0. Use PBO alongside DSR, never alone.\n\n"
           "## Wrap-up\n\n"
           "* The more you try, the higher the best result by luck: deflate for the number of trials, and log them all.\n"
           "* PBO asks whether in-sample ranking survives out of sample.\n"
           "* Graded version: `labs/part08/week28_overfitting` and Clinic W4 (the validation dossier)."),
]

# ---------------------------------------------------------------------------------------------- 08
NB["08_robustness_and_gate"] = [
    header("08", "Robustness tests and the validation gate", "S15 (Robustness tests) · S16 (The validation gate & M4 release)",
           "1. Test whether the result survives without its best days.\n"
           "2. Perturb every parameter and check the Sharpe holds.\n"
           "3. Combine everything into a pass/fail validation gate, and see a decent strategy fail it."),
    ("code", SETUP),
    ("md", "## 1. The candidate\n\n"
           "TSMOM on the known-regime market, lookback 160 and vol window 60 (the plateau from notebook 05). `run(params, delay, cost_mult)` "
           "returns its daily P&L, optionally with the signal delayed and the costs multiplied: every robustness test is a variation of it."),
    ("code", """bars = p.regime_market()
o, c = bars.open.to_numpy(), bars.close.to_numpy()

def run(params, delay=0, cost_mult=1.0):
    sig = p.tsmom_signal(c, params["lookback"], params["vol_n"])
    if delay:
        sig = np.r_[np.zeros(delay), sig[:-delay]]
    return p.first_look_pnl(sig, o, cost_bps=2.0 * cost_mult)

params = {"lookback": 160, "vol_n": 60}
base = run(params)
print(f"base Sharpe {p.sharpe(base):.2f}")"""),
    ("md", "## 2. Without the best days\n\n"
           "If a handful of days make the whole result, it is a lottery ticket. Remove the `k` largest daily P&Ls and recompute the Sharpe "
           "(`np.argsort` gives the order; `np.delete` removes by index)."),
    ("md", YOUR_TURN),
    ("ex", """def sharpe_without_best(pnl, k=5):
    pnl = np.asarray(pnl, dtype=float)
    rest = ...                                    # ✍️ pnl with its k largest values removed
    return p.sharpe(rest)

ks = [1, 5, 20, 50]
mine = [p.attempt(sharpe_without_best, base, k) for k in ks]
mine = p.check("sharpe_without_best", mine, [p.sharpe(np.delete(base, np.argsort(base)[-k:])) for k in ks])
pd.Series(mine, index=[f"without best {k}" for k in ks]).round(2)""",
     """def sharpe_without_best(pnl, k=5):
    pnl = np.asarray(pnl, dtype=float)
    rest = np.delete(pnl, np.argsort(pnl)[-k:])
    return p.sharpe(rest)

ks = [1, 5, 20, 50]
mine = [p.attempt(sharpe_without_best, base, k) for k in ks]
mine = p.check("sharpe_without_best", mine, [p.sharpe(np.delete(base, np.argsort(base)[-k:])) for k in ks])
pd.Series(mine, index=[f"without best {k}" for k in ks]).round(2)"""),
    ("md", "## 3. Parameter perturbation\n\n"
           "For every **integer** parameter, rerun with it scaled by `1 − perturb` and `1 + perturb` (rounded to an int), keeping the others. "
           "Return `(test name, Sharpe)` pairs named like `'lookback -20%'`, `'lookback +20%'`, in the dict's order, minus before plus."),
    ("md", YOUR_TURN),
    ("ex", """def perturbation_tests(run, params, perturb=0.2):
    out = []
    for k, v in params.items():
        for sgn, lab in ((-1, "-"), (1, "+")):
            p2 = ...                              # ✍️ params with k replaced by int(round(v × (1 ± perturb)))
            out.append((f"{k} {lab}{int(perturb * 100)}%", p.sharpe(run(p2))))
    return out

mine = p.attempt(perturbation_tests, run, params)
card = p.robustness_scorecard(run, params)
mine = p.check("perturbation_tests", mine, [(t, v) for t, v, _ in card.itertuples(index=False)][:4])
card.round(2)""",
     """def perturbation_tests(run, params, perturb=0.2):
    out = []
    for k, v in params.items():
        for sgn, lab in ((-1, "-"), (1, "+")):
            p2 = dict(params, **{k: int(round(v * (1 + sgn * perturb)))})
            out.append((f"{k} {lab}{int(perturb * 100)}%", p.sharpe(run(p2))))
    return out

mine = p.attempt(perturbation_tests, run, params)
card = p.robustness_scorecard(run, params)
mine = p.check("perturbation_tests", mine, [(t, v) for t, v, _ in card.itertuples(index=False)][:4])
card.round(2)"""),
    ("md", "The full scorecard adds a one-bar delay, doubled costs, and each half of the history on its own. TSMOM passes all nine.\n\n"
           "## 4. The validation gate\n\n"
           "A strategy is promoted to paper trading only if it passes **every** check (`p.GATE` holds the thresholds):\n\n"
           "| Check | Pass if |\n|---|---|\n| out-of-sample Sharpe (walk-forward) | >= 0.5 |\n| Deflated Sharpe of the OOS P&L, given all trials | >= 0.95 |\n"
           "| PBO of the whole search | <= 0.2 |\n| share of robustness tests passed | >= 0.8 |\n| OOS max drawdown | >= −25% |\n\n"
           "Build the evidence: the 48-trial grid from notebook 05 (for the trial Sharpes and PBO), a walk-forward OOS series, and the scorecard."),
    ("code", """grid = [dict(lookback=lb, vol_n=vn) for lb in (20, 40, 60, 90, 120, 160, 200, 250) for vn in (10, 20, 40, 60, 90, 120)]
perf = np.column_stack([run(g) for g in grid])
trial_srs = perf.mean(0) / perf.std(0, ddof=1)
pbo, _ = p.pbo_cscv(perf, S=10)
wf = p.walk_forward_optimize(lambda lookback, vol_n: run(dict(lookback=lookback, vol_n=vol_n)),
                             {"lookback": [40, 90, 160, 250], "vol_n": [20, 60]}, len(c), 750, 250)
print(f"{len(grid)} trials, PBO {pbo:.2f}, walk-forward OOS Sharpe {p.sharpe(wf['oos']):.2f}")"""),
    ("md", YOUR_TURN + "\n\nWrite the gate: compute each check's value and whether it passes, and `passed = all of them`. Return "
           "`{\"checks\": {name: (value, passed)}, \"passed\": bool}` with the names `oos_sharpe`, `dsr`, `pbo`, `robustness`, `max_dd`."),
    ("ex", """def validation_gate(oos_pnl, trial_srs, pbo, scorecard, c=p.GATE):
    r = np.asarray(oos_pnl, dtype=float)
    oos_sr = p.sharpe(r)
    dsr, _ = p.deflated_sharpe(r, trial_srs)
    rob = float(scorecard["passed"].mean())
    mdd = p.max_drawdown(r)[0]
    checks = {"oos_sharpe": (oos_sr, oos_sr >= c["min_oos_sharpe"]),
              "dsr": ...,                         # ✍️
              "pbo": ...,                         # ✍️
              "robustness": (rob, rob >= c["min_robustness"]),
              "max_dd": (mdd, mdd >= c["max_drawdown"])}
    return {"checks": checks, "passed": all(ok for _, ok in checks.values())}

mine = p.attempt(validation_gate, wf["oos"], trial_srs, pbo, card)
mine = p.check("validation_gate", mine, p.validation_gate(wf["oos"], trial_srs, pbo, card))
display(pd.DataFrame(mine["checks"], index=["value", "passed"]).T)
print("VERDICT:", "PASS" if mine["passed"] else "FAIL")""",
     """def validation_gate(oos_pnl, trial_srs, pbo, scorecard, c=p.GATE):
    r = np.asarray(oos_pnl, dtype=float)
    oos_sr = p.sharpe(r)
    dsr, _ = p.deflated_sharpe(r, trial_srs)
    rob = float(scorecard["passed"].mean())
    mdd = p.max_drawdown(r)[0]
    checks = {"oos_sharpe": (oos_sr, oos_sr >= c["min_oos_sharpe"]),
              "dsr": (dsr, dsr >= c["min_dsr"]),
              "pbo": (pbo, pbo <= c["max_pbo"]),
              "robustness": (rob, rob >= c["min_robustness"]),
              "max_dd": (mdd, mdd >= c["max_drawdown"])}
    return {"checks": checks, "passed": all(ok for _, ok in checks.values())}

mine = p.attempt(validation_gate, wf["oos"], trial_srs, pbo, card)
mine = p.check("validation_gate", mine, p.validation_gate(wf["oos"], trial_srs, pbo, card))
display(pd.DataFrame(mine["checks"], index=["value", "passed"]).T)
print("VERDICT:", "PASS" if mine["passed"] else "FAIL")"""),
    ("md", "A strategy with a real mechanism (it earns in trends by construction), a 0.9 out-of-sample Sharpe and a perfect robustness "
           "scorecard still **fails**: six years of out-of-sample data after 48 trials isn't enough evidence (DSR just under 0.95), and the "
           "search ranking is not stable enough (PBO above 0.2). The right response is more evidence (more markets, more history, paper "
           "trading), not loosening the gate until it passes.\n\n"
           "## Wrap-up\n\n"
           "* Robustness: without the best days, parameters ±20%, a bar of delay, doubled costs, each half alone.\n"
           "* The gate is a pre-registered, all-must-pass rule; write it down before you look at the results.\n"
           "* Graded versions: `labs/part08/week28_overfitting` (scorecard, gate, dossier) and Clinic W4 (validating three strategies)."),
]

# ---------------------------------------------------------------------------------------------- 09
NB["09_risk_engine_and_var"] = [
    header("09", "The risk engine, VaR and stress tests", "S17 (Risk framework & the risk engine) · S18 (Market-risk measures & stress testing)",
           "1. Write a risk rule and chain rules into a risk engine.\n"
           "2. Compute historical VaR and CVaR, and compare them with the normal approximation.\n"
           "3. Backtest VaR through crises, and see correlations change when it matters."),
    ("code", SETUP),
    ("md", "## 1. Pre-trade risk rules\n\n"
           "Every order passes the risk engine, in the backtest as well as live. A rule looks at the order and a context and approves or "
           "rejects with a reason. Write **single-name concentration**: the position after the order, `|current $ position + qty × price|`, "
           "divided by equity, must be at most `max_weight`. Reason text: `f\"{symbol} weight {w:.1%} > {max_weight:.1%}\"`."),
    ("code", """ctx = {"equity": 1_000_000, "gross": 900_000, "positions": {"SPY": 150_000, "TLT": 100_000},
       "day_pnl": -12_000, "start_equity": 1_012_000}
orders = [{"symbol": "SPY", "qty": 100, "price": 500.0}, {"symbol": "SPY", "qty": 200, "price": 500.0},
          {"symbol": "TLT", "qty": -1500, "price": 90.0}, {"symbol": "GLD", "qty": 1000, "price": 220.0},
          {"symbol": "QQQ", "qty": 3000, "price": 440.0}]"""),
    ("md", YOUR_TURN),
    ("ex", """class MaxPositionWeight(p.RiskRule):
    def __init__(self, max_weight):
        self.max_weight = max_weight

    def check(self, order, ctx):
        after = ...                               # ✍️ the $ position in this symbol after the order
        w = abs(after) / ctx["equity"]
        return p.RiskDecision(w <= self.max_weight, f"{order['symbol']} weight {w:.1%} > {self.max_weight:.1%}")

mine = [p.attempt(MaxPositionWeight(0.2).check, o_, ctx) for o_ in orders]
mine = p.check("MaxPositionWeight", mine, [p.MaxPositionWeight(0.2).check(o_, ctx) for o_ in orders])
[(o_["symbol"], o_["qty"], d.approved, d.reason) for o_, d in zip(orders, mine)]""",
     """class MaxPositionWeight(p.RiskRule):
    def __init__(self, max_weight):
        self.max_weight = max_weight

    def check(self, order, ctx):
        after = ctx["positions"].get(order["symbol"], 0.0) + order["qty"] * order["price"]
        w = abs(after) / ctx["equity"]
        return p.RiskDecision(w <= self.max_weight, f"{order['symbol']} weight {w:.1%} > {self.max_weight:.1%}")

mine = [p.attempt(MaxPositionWeight(0.2).check, o_, ctx) for o_ in orders]
mine = p.check("MaxPositionWeight", mine, [p.MaxPositionWeight(0.2).check(o_, ctx) for o_ in orders])
[(o_["symbol"], o_["qty"], d.approved, d.reason) for o_, d in zip(orders, mine)]"""),
    ("md", "## 2. The engine: a chain of rules\n\n"
           "**Chain of Responsibility:** ask each rule in turn; the first rejection stops the chain, and the reason is prefixed with the "
           "rule's class name (`\"MaxNotional: notional 1,320,000 > 250,000\"`). Log every rejection as `(symbol, reason)`. "
           "If every rule approves, approve."),
    ("md", YOUR_TURN),
    ("ex", """class RiskEngine:
    def __init__(self, rules):
        self.rules, self.log = rules, []

    def check(self, order, ctx):
        for rule in self.rules:
            d = rule.check(order, ctx)
            if not d.approved:
                ...                               # ✍️ build the prefixed reason, log it, and return a rejection
        return p.RiskDecision(True)

rules = [p.MaxNotional(250_000), p.MaxGrossExposure(1.5), p.MaxPositionWeight(0.2), p.DailyLossLimit(0.02)]
mine = p.attempt(lambda: [(d.approved, d.reason) for d in (RiskEngine(rules).check(o_, ctx) for o_ in orders)])
ref_engine = p.RiskEngine(rules)
mine = p.check("RiskEngine", mine, [(d.approved, d.reason) for d in (ref_engine.check(o_, ctx) for o_ in orders)])
pd.DataFrame(mine, index=[f"{o_['symbol']} {o_['qty']:+}" for o_ in orders], columns=["approved", "reason"])""",
     """class RiskEngine:
    def __init__(self, rules):
        self.rules, self.log = rules, []

    def check(self, order, ctx):
        for rule in self.rules:
            d = rule.check(order, ctx)
            if not d.approved:
                reason = f"{type(rule).__name__}: {d.reason}"
                self.log.append((order["symbol"], reason))
                return p.RiskDecision(False, reason)
        return p.RiskDecision(True)

rules = [p.MaxNotional(250_000), p.MaxGrossExposure(1.5), p.MaxPositionWeight(0.2), p.DailyLossLimit(0.02)]
mine = p.attempt(lambda: [(d.approved, d.reason) for d in (RiskEngine(rules).check(o_, ctx) for o_ in orders)])
ref_engine = p.RiskEngine(rules)
mine = p.check("RiskEngine", mine, [(d.approved, d.reason) for d in (ref_engine.check(o_, ctx) for o_ in orders)])
pd.DataFrame(mine, index=[f"{o_['symbol']} {o_['qty']:+}" for o_ in orders], columns=["approved", "reason"])"""),
    ("md", "## 3. VaR and CVaR\n\n"
           "A 60/40-style portfolio of six synthetic assets, ten years of daily returns, with three crisis episodes in which equity "
           "correlations jump and bonds rally.\n\n"
           "Historical **VaR** at level `α` is the `α`-quantile of the losses (`−returns`); **CVaR** (expected shortfall) is the mean of the "
           "losses at or beyond the VaR. Both as positive numbers."),
    ("code", """A = p.asset_returns()
w = np.array([0.35, 0.25, 0.25, 0.05, 0.05, 0.05])
port = pd.Series(A.to_numpy() @ w, index=A.index)
A.describe().loc[["mean", "std"]].T.assign(mean=lambda d: d["mean"] * 252, std=lambda d: d["std"] * np.sqrt(252)).round(3)"""),
    ("md", YOUR_TURN),
    ("ex", """def hist_var_cvar(returns, alpha=0.99):
    losses = -np.asarray(returns, dtype=float)
    var = ...                                     # ✍️
    cvar = ...                                    # ✍️
    return float(var), float(cvar)

mine = [p.attempt(hist_var_cvar, port, a) for a in (0.95, 0.99, 0.999)]
mine = p.check("hist_var_cvar", mine, [p.hist_var_cvar(port, a) for a in (0.95, 0.99, 0.999)])
cov = A.cov().to_numpy()
pd.DataFrame({"historical VaR": [v for v, _ in mine], "historical CVaR": [c_ for _, c_ in mine],
              "normal VaR": [p.parametric_var(w, cov, a) for a in (0.95, 0.99, 0.999)]}, index=["95%", "99%", "99.9%"]).mul(100).round(2)""",
     """def hist_var_cvar(returns, alpha=0.99):
    losses = -np.asarray(returns, dtype=float)
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)

mine = [p.attempt(hist_var_cvar, port, a) for a in (0.95, 0.99, 0.999)]
mine = p.check("hist_var_cvar", mine, [p.hist_var_cvar(port, a) for a in (0.95, 0.99, 0.999)])
cov = A.cov().to_numpy()
pd.DataFrame({"historical VaR": [v for v, _ in mine], "historical CVaR": [c_ for _, c_ in mine],
              "normal VaR": [p.parametric_var(w, cov, a) for a in (0.95, 0.99, 0.999)]}, index=["95%", "99%", "99.9%"]).mul(100).round(2)"""),
    ("md", "The normal approximation is even a little conservative at 95%, but falls behind further out: at 99.9% the historical VaR is "
           "almost twice the normal one. Fat tails and crises live there. **Component VaR** splits "
           "the (normal) VaR by asset; it adds up to the total:"),
    ("code", """comp = p.component_var(w, cov)
pd.Series(comp / comp.sum(), index=A.columns, name="share of VaR").round(3).to_frame().assign(weight=w)"""),
    ("md", "60% of the capital in equities carries over 90% of the risk.\n\n"
           "## 4. VaR through crises\n\n"
           "Estimate a 99% VaR each day from the previous 500 days and count the days it is breached (about 20 expected in 2,000 days). "
           "Then look at what the correlations did during the crises."),
    ("code", """x = port.to_numpy()
hits = {"normal": [], "historical": []}
for t in range(500, len(x)):
    win = x[t - 500:t]
    hits["normal"].append(-x[t] > 2.326 * win.std(ddof=1))
    hits["historical"].append(-x[t] > np.quantile(-win, 0.99))
print({k: int(np.sum(v)) for k, v in hits.items()}, f"breaches; expected about {0.01 * (len(x) - 500):.0f}")
crisis = np.zeros(len(A), dtype=bool)
for s in (600, 1700, 2300):
    crisis[s:s + 60] = True
pairs = [("EQ_US", "EQ_INTL"), ("EQ_US", "CMDTY"), ("EQ_US", "BOND_10Y")]
pd.DataFrame({"calm": [A[~crisis][a].corr(A[~crisis][b]) for a, b in pairs], "crisis": [A[crisis][a].corr(A[crisis][b]) for a, b in pairs]},
             index=[f"{a} / {b}" for a, b in pairs]).round(2)"""),
    ("md", "Diversification shrinks exactly when it is needed: equity markets move together in a crisis. Stress tests therefore use "
           "**crisis** correlations and scenario shocks, not the calm-period covariance.\n\n"
           "## Wrap-up\n\n"
           "* One risk engine, rules chained, every rejection logged, in backtest and live.\n"
           "* Historical VaR/CVaR, backtested; normal VaR only as a quick first look.\n"
           "* Stress correlations and scenarios on top of VaR.\n"
           "* Graded version: `labs/part08/week29_risk_sizing` (six rules including sector and ADV limits, VaR measures)."),
]

# ---------------------------------------------------------------------------------------------- 10
NB["10_position_sizing_and_kelly"] = [
    header("10", "Position sizing and the Kelly criterion", "S19 (Position sizing I: fixed risk & volatility) · S20 (Position sizing II: Kelly & estimation error)",
           "1. Size a trade so that its stop loses a fixed fraction of equity.\n"
           "2. Simulate the risk of ruin for different bet sizes.\n"
           "3. Compute the Kelly fraction, and see why full Kelly with an estimated edge is dangerous."),
    ("code", SETUP),
    ("md", "## 1. Fixed-fractional sizing\n\n"
           "Decide how much of equity one trade may lose (say 1%), and let the stop distance set the size: "
           "`units = floor(equity × risk_frac / (|entry − stop| × multiplier))`."),
    ("md", YOUR_TURN),
    ("ex", """def risk_per_trade_size(equity, risk_frac, entry, stop, multiplier=1.0):
    return ...                                    # ✍️ an int (floored)

cases = [(100_000, 0.01, 50.0, 48.0), (100_000, 0.01, 50.0, 45.0), (250_000, 0.005, 5000.0, 4980.0, 50.0), (250_000, 0.005, 5000.0, 4980.0, 5.0)]
mine = [p.attempt(risk_per_trade_size, *cs) for cs in cases]
mine = p.check("risk_per_trade_size", mine, [p.risk_per_trade_size(*cs) for cs in cases])
pd.DataFrame({"case": ["stock, stop 2 below", "stock, stop 5 below", "ES, stop 20 points", "MES, stop 20 points"], "units": mine})""",
     """def risk_per_trade_size(equity, risk_frac, entry, stop, multiplier=1.0):
    return int(equity * risk_frac / (abs(entry - stop) * multiplier))

cases = [(100_000, 0.01, 50.0, 48.0), (100_000, 0.01, 50.0, 45.0), (250_000, 0.005, 5000.0, 4980.0, 50.0), (250_000, 0.005, 5000.0, 4980.0, 5.0)]
mine = [p.attempt(risk_per_trade_size, *cs) for cs in cases]
mine = p.check("risk_per_trade_size", mine, [p.risk_per_trade_size(*cs) for cs in cases])
pd.DataFrame({"case": ["stock, stop 2 below", "stock, stop 5 below", "ES, stop 20 points", "MES, stop 20 points"], "units": mine})"""),
    ("md", "A wider stop means a smaller position, so every trade risks the same. With ES a $1,250 risk budget can't even buy one contract; "
           "micros can size it properly.\n\n"
           "## 2. Risk of ruin\n\n"
           "A system that wins 40% of the time, winning 2R and losing 1R: a positive expectancy of 0.2R per trade. Simulate `n_sims` sequences "
           "of `n_trades` results (R-multiples drawn with replacement), equity `×= 1 + risk_frac·R`, and report the share of paths whose equity "
           "**ever** falls to `ruin` (0.5 = a 50% drawdown from the start)."),
    ("md", YOUR_TURN),
    ("ex", """def risk_of_ruin(r_multiples, risk_frac, ruin=0.5, n_trades=500, n_sims=2000, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.choice(np.asarray(r_multiples, dtype=float), size=(n_sims, n_trades))
    eq = ...                                      # ✍️ the equity path of every simulation (cumulative product)
    return float(np.mean(eq.min(axis=1) <= ruin))

R = np.array([-1.0] * 60 + [2.0] * 40)
fracs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
mine = [p.attempt(risk_of_ruin, R, f) for f in fracs]
mine = p.check("risk_of_ruin", mine, [p.risk_of_ruin(R, f) for f in fracs])
pd.Series(mine, index=[f"risk {f:.1%} per trade" for f in fracs], name="P(ever down 50%)").round(3)""",
     """def risk_of_ruin(r_multiples, risk_frac, ruin=0.5, n_trades=500, n_sims=2000, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.choice(np.asarray(r_multiples, dtype=float), size=(n_sims, n_trades))
    eq = np.cumprod(1 + risk_frac * R, axis=1)
    return float(np.mean(eq.min(axis=1) <= ruin))

R = np.array([-1.0] * 60 + [2.0] * 40)
fracs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
mine = [p.attempt(risk_of_ruin, R, f) for f in fracs]
mine = p.check("risk_of_ruin", mine, [p.risk_of_ruin(R, f) for f in fracs])
pd.Series(mine, index=[f"risk {f:.1%} per trade" for f in fracs], name="P(ever down 50%)").round(3)"""),
    ("md", "Same edge, same trades: at 2% per trade the account is essentially safe, at 10% it halves in almost every other path.\n\n"
           "## 3. Kelly\n\n"
           "The **Kelly** fraction maximizes long-run growth:\n"
           "* discrete bets (win probability `p`, win/loss ratio `b`): `f* = p − (1 − p)/b`;\n"
           "* continuous returns (mean `μ`, volatility `σ`, rate `r`): leverage `(μ − r)/σ²`."),
    ("md", YOUR_TURN),
    ("ex", """def kelly_discrete(p_win, win_loss_ratio):
    return ...                                    # ✍️

def kelly_continuous(mu, sigma, r=0.0):
    return ...                                    # ✍️

mine = [kelly_discrete(0.4, 2.0), kelly_discrete(0.55, 1.0), kelly_continuous(0.08, 0.16), kelly_continuous(0.08, 0.16, 0.03)]
mine = p.check("kelly", mine, [p.kelly_discrete(0.4, 2.0), p.kelly_discrete(0.55, 1.0), p.kelly_continuous(0.08, 0.16), p.kelly_continuous(0.08, 0.16, 0.03)])
dict(zip(["40% × 2R", "55% even money", "equity 8%/16%", "equity, 3% rate"], np.round(mine, 3)))""",
     """def kelly_discrete(p_win, win_loss_ratio):
    return p_win - (1 - p_win) / win_loss_ratio

def kelly_continuous(mu, sigma, r=0.0):
    return (mu - r) / sigma ** 2

mine = [kelly_discrete(0.4, 2.0), kelly_discrete(0.55, 1.0), kelly_continuous(0.08, 0.16), kelly_continuous(0.08, 0.16, 0.03)]
mine = p.check("kelly", mine, [p.kelly_discrete(0.4, 2.0), p.kelly_discrete(0.55, 1.0), p.kelly_continuous(0.08, 0.16), p.kelly_continuous(0.08, 0.16, 0.03)])
dict(zip(["40% × 2R", "55% even money", "equity 8%/16%", "equity, 3% rate"], np.round(mine, 3)))"""),
    ("md", "The 40%/2R system's Kelly bet is exactly the 10% that gave a 46% chance of halving the account above: Kelly maximizes growth, "
           "not comfort. And an equity index with an 8% edge and 16% vol has a Kelly leverage above 3.\n\n"
           "Worse, `μ` is **estimated**. Estimate it from five years of data, lever at a fraction of the estimated Kelly, and run twenty years "
           "of the true process:"),
    ("code", """sim = p.kelly_growth_simulation(n_paths=500)
sim.index = [f"{f:g}× Kelly" for f in sim.index]
sim.round(3)"""),
    ("md", "Half Kelly grows the median investor the most here, and with far smaller drawdowns; full Kelly on an estimated edge loses money "
           "for about a third of investors; double Kelly ruins most of them. Use fractional Kelly as a **cap**, and volatility targeting "
           "for the day-to-day size.\n\n"
           "## Wrap-up\n\n"
           "* Size by risk (stop distance, ATR, volatility), not by conviction.\n"
           "* Simulate ruin before choosing the fraction.\n"
           "* Kelly with an estimated edge overbets; half Kelly or less.\n"
           "* Graded version: `labs/part08/week29_risk_sizing` (volatility targeting, Turtle units, multi-asset Kelly)."),
]

# ---------------------------------------------------------------------------------------------- 11
NB["11_combining_and_mean_variance"] = [
    header("11", "Combining strategies and mean–variance optimization", "S21 (Combining strategies) · S22 (Mean–variance & covariance estimation)",
           "1. Weight strategies by inverse volatility, and measure diversification.\n"
           "2. Write the minimum-variance portfolio.\n"
           "3. Shrink the covariance matrix, and see why sample-based mean–variance falls apart out of sample."),
    ("code", SETUP),
    ("md", "## 1. Five strategies\n\n"
           "The five synthetic strategies from notebook 03 (true Sharpes 0.8 to 0.3, vols 8% to 20%, pairwise correlation 0.2)."),
    ("code", """S = p.strategy_returns()
display(pd.DataFrame({"vol": S.std() * np.sqrt(252), "Sharpe": S.apply(p.sharpe)}).round(3).T)
S.corr().round(2)"""),
    ("md", YOUR_TURN + "\n\n**Inverse volatility:** `w_i ∝ 1/σ_i` (sample std, ddof=1), normalized to sum to 1. The simplest way to stop the "
           "most volatile strategy from dominating the book."),
    ("ex", """def inverse_vol_weights(returns):
    iv = ...                                      # ✍️ 1 / std of each column
    return iv / iv.sum()

mine = p.attempt(inverse_vol_weights, S)
mine = p.check("inverse_vol_weights", mine, p.inverse_vol_weights(S))
cov = S.cov()
print(f"diversification ratio: 1/N {p.diversification_ratio(np.full(5, 0.2), cov):.2f}, inverse vol {p.diversification_ratio(mine, cov):.2f}")
mine.round(3)""",
     """def inverse_vol_weights(returns):
    iv = 1 / returns.std(ddof=1)
    return iv / iv.sum()

mine = p.attempt(inverse_vol_weights, S)
mine = p.check("inverse_vol_weights", mine, p.inverse_vol_weights(S))
cov = S.cov()
print(f"diversification ratio: 1/N {p.diversification_ratio(np.full(5, 0.2), cov):.2f}, inverse vol {p.diversification_ratio(mine, cov):.2f}")
mine.round(3)"""),
    ("md", "The diversification ratio `Σ w_i σ_i / σ_p` is 1 for a single strategy; above 1.6 here, because correlations are low.\n\n"
           "## 2. Minimum variance\n\n"
           "The global minimum-variance portfolio (shorts allowed) is `Σ⁻¹1 / (1'Σ⁻¹1)`: solve `Σx = 1` with `np.linalg.solve`, then "
           "normalize. It needs only the covariance, no expected returns."),
    ("md", YOUR_TURN),
    ("ex", """def min_variance(cov):
    cov = np.asarray(cov, dtype=float)
    x = ...                                       # ✍️ solve Σ x = 1
    return x / x.sum()

mine = p.attempt(min_variance, cov)
mine = p.check("min_variance", mine, p.min_variance(cov.to_numpy()))
pd.Series(mine, index=S.columns).round(3)""",
     """def min_variance(cov):
    cov = np.asarray(cov, dtype=float)
    x = np.linalg.solve(cov, np.ones(len(cov)))
    return x / x.sum()

mine = p.attempt(min_variance, cov)
mine = p.check("min_variance", mine, p.min_variance(cov.to_numpy()))
pd.Series(mine, index=S.columns).round(3)"""),
    ("md", "## 3. Why sample mean–variance fails\n\n"
           "The maximum-Sharpe (tangency) portfolio `Σ⁻¹μ` needs **expected returns**, and sample means are extremely noisy (notebook 04). "
           "Fit it on each half of the history:"),
    ("code", """h = len(S) // 2
pd.DataFrame({"first half": p.max_sharpe(S.iloc[:h].mean(), S.iloc[:h].cov()), "second half": p.max_sharpe(S.iloc[h:].mean(), S.iloc[h:].cov()),
              "true Sharpes": [0.8, 0.6, 0.5, 0.4, 0.3]}, index=S.columns).round(2)"""),
    ("md", "Weights flip sign between halves: the optimizer is maximizing estimation error. The covariance is better estimated than the "
           "means, but with many assets and short windows it too is noisy. **Ledoit–Wolf shrinkage** pulls the sample covariance `S` toward "
           "a scaled identity `μI` (`μ` = the average variance), with an intensity `δ` estimated from the data (`p.shrinkage_intensity`): "
           "`Σ_shrunk = δ·μ·I + (1 − δ)·S`, where `S` is the sample covariance with divisor `T` (demeaned `X'X / T`)."),
    ("md", YOUR_TURN),
    ("ex", """def ledoit_wolf(returns):
    X = np.asarray(returns, dtype=float)
    X = X - X.mean(0)
    S_ = X.T @ X / X.shape[0]
    mu = np.trace(S_) / S_.shape[0]
    d = p.shrinkage_intensity(returns)
    return ...                                    # ✍️

few = S.iloc[:60]                                 # three months of data: the regime where shrinkage matters
mine = [p.attempt(ledoit_wolf, S), p.attempt(ledoit_wolf, few)]
mine = p.check("ledoit_wolf", mine, [p.ledoit_wolf(S), p.ledoit_wolf(few)])
print(f"shrinkage intensity: 10 years of data {p.shrinkage_intensity(S):.3f}; 60 days {p.shrinkage_intensity(few):.3f}")""",
     """def ledoit_wolf(returns):
    X = np.asarray(returns, dtype=float)
    X = X - X.mean(0)
    S_ = X.T @ X / X.shape[0]
    mu = np.trace(S_) / S_.shape[0]
    d = p.shrinkage_intensity(returns)
    return d * mu * np.eye(S_.shape[0]) + (1 - d) * S_

few = S.iloc[:60]                                 # three months of data: the regime where shrinkage matters
mine = [p.attempt(ledoit_wolf, S), p.attempt(ledoit_wolf, few)]
mine = p.check("ledoit_wolf", mine, [p.ledoit_wolf(S), p.ledoit_wolf(few)])
print(f"shrinkage intensity: 10 years of data {p.shrinkage_intensity(S):.3f}; 60 days {p.shrinkage_intensity(few):.3f}")"""),
    ("md", "With ten years the data speaks for itself (little shrinkage); with 60 days the estimator leans heavily on the prior. Now the "
           "honest comparison: re-fit every month on the previous year and trade the next month (**walk-forward allocation**)."),
    ("code", """alloc = {"1/N": lambda X: np.full(X.shape[1], 1 / X.shape[1]),
         "inverse vol": lambda X: p.inverse_vol_weights(X).to_numpy(),
         "min variance (sample)": lambda X: p.min_variance(X.cov().to_numpy()),
         "min variance (shrunk)": lambda X: p.min_variance(p.ledoit_wolf(X)),
         "max Sharpe (sample)": lambda X: p.max_sharpe(X.mean().to_numpy(), X.cov().to_numpy())}
rows = {}
for name, f in alloc.items():
    res = p.walk_forward_allocation(S, f)
    r = res["returns"]
    rows[name] = {"OOS Sharpe": p.sharpe(r), "OOS vol": r.std() * np.sqrt(252), "turnover per rebalance": res["turnover"]}
pd.DataFrame(rows).T.round(3)"""),
    ("md", "Sample max-Sharpe explodes (a vol in the thousands of percent and huge turnover: the weights swing wildly from month to month). "
           "Simple 1/N and inverse vol are the ones to beat, as DeMiguel, Garlappi & Uppal (2009) found on real data.\n\n"
           "## Wrap-up\n\n"
           "* Start with 1/N and inverse vol; anything fancier must beat them **out of sample**, after turnover.\n"
           "* Never feed sample means to an optimizer; shrink covariances.\n"
           "* Graded version: `labs/part08/week30_portfolio` (long-only minimum variance with cvxpy)."),
]

# ---------------------------------------------------------------------------------------------- 12
NB["12_risk_based_allocation"] = [
    header("12", "Risk parity, HRP, Black–Litterman and CVaR", "S23 (Risk parity, HRP, Black–Litterman & CVaR) · S24 (Integration & M5a release)",
           "1. Compute risk contributions, and build a portfolio where they are equal.\n"
           "2. Write the recursive bisection step of Hierarchical Risk Parity.\n"
           "3. Blend equilibrium returns with a view using Black–Litterman.\n"
           "4. Compare seven allocators walk-forward, through three crises."),
    ("code", SETUP),
    ("md", "## 1. Risk contributions\n\n"
           "Capital weights hide where the risk is. The share of portfolio variance coming from asset `i` is `w_i·(Σw)_i / (w'Σw)`; the shares "
           "sum to 1. Six assets from notebook 09."),
    ("code", """A = p.asset_returns()
cov = A.cov().to_numpy() * 252"""),
    ("md", YOUR_TURN),
    ("ex", """def risk_contributions(w, cov):
    w = np.asarray(w, dtype=float)
    return ...                                    # ✍️

w6040 = np.array([0.35, 0.25, 0.25, 0.05, 0.05, 0.05])
w_rp = p.risk_parity(cov)
mine = [p.attempt(risk_contributions, w6040, cov), p.attempt(risk_contributions, w_rp, cov)]
mine = p.check("risk_contributions", mine, [p.risk_contributions(w6040, cov), p.risk_contributions(w_rp, cov)])
pd.DataFrame({"60/40 weights": w6040, "60/40 risk": mine[0], "risk-parity weights": w_rp, "risk-parity risk": mine[1]}, index=A.columns).round(3)""",
     """def risk_contributions(w, cov):
    w = np.asarray(w, dtype=float)
    return w * (cov @ w) / (w @ cov @ w)

w6040 = np.array([0.35, 0.25, 0.25, 0.05, 0.05, 0.05])
w_rp = p.risk_parity(cov)
mine = [p.attempt(risk_contributions, w6040, cov), p.attempt(risk_contributions, w_rp, cov)]
mine = p.check("risk_contributions", mine, [p.risk_contributions(w6040, cov), p.risk_contributions(w_rp, cov)])
pd.DataFrame({"60/40 weights": w6040, "60/40 risk": mine[0], "risk-parity weights": w_rp, "risk-parity risk": mine[1]}, index=A.columns).round(3)"""),
    ("md", "Risk parity equalizes the contributions by putting most of the capital into the least volatile asset; to reach a useful return "
           "it is usually levered (or volatility-targeted, notebook 10).\n\n"
           "## 2. Hierarchical Risk Parity\n\n"
           "HRP (López de Prado, 2016) avoids inverting the covariance matrix. Cluster the assets by correlation and order them so similar "
           "assets sit together (`p.hrp_order`); then **bisect** that order recursively: at each split, the variance of each half's "
           "inverse-variance portfolio (`p.cluster_variance`) decides how to share the weight, `α = 1 − v_a / (v_a + v_b)` to the left half."),
    ("code", """print("quasi-diagonal order:", p.hrp_order(A))"""),
    ("md", YOUR_TURN),
    ("ex", """def hrp(returns):
    cov_ = returns.cov()
    order = p.hrp_order(returns)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for a, b in zip(clusters[::2], clusters[1::2]):
            va, vb = p.cluster_variance(cov_, a), p.cluster_variance(cov_, b)
            alpha = ...                           # ✍️ the left half's share
            w[a] *= alpha
            w[b] *= 1 - alpha
    return w.reindex(returns.columns)

mine = p.attempt(hrp, A)
mine = p.check("hrp", mine, p.hrp(A))
pd.DataFrame({"HRP": mine, "risk parity": w_rp, "inverse vol": p.inverse_vol_weights(A)}, index=A.columns).round(3)""",
     """def hrp(returns):
    cov_ = returns.cov()
    order = p.hrp_order(returns)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for a, b in zip(clusters[::2], clusters[1::2]):
            va, vb = p.cluster_variance(cov_, a), p.cluster_variance(cov_, b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
    return w.reindex(returns.columns)

mine = p.attempt(hrp, A)
mine = p.check("hrp", mine, p.hrp(A))
pd.DataFrame({"HRP": mine, "risk parity": w_rp, "inverse vol": p.inverse_vol_weights(A)}, index=A.columns).round(3)"""),
    ("md", "## 3. Black–Litterman\n\n"
           "Instead of noisy sample means, start from the returns the market portfolio **implies**, `π = δ·Σ·w_mkt`, and tilt them by your "
           "views. A view is a row of `P` (which assets) with an expected value in `Q`, and uncertainty `Ω` (default `diag(P·τΣ·P')`). The "
           "posterior is `μ = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]`. Use `np.linalg.inv` and `np.linalg.solve`."),
    ("md", YOUR_TURN),
    ("ex", """def black_litterman(cov, w_mkt, P, Q, delta=2.5, tau=0.05):
    S_ = np.asarray(cov, dtype=float)
    pi = delta * S_ @ np.asarray(w_mkt, dtype=float)
    P, Q = np.atleast_2d(np.asarray(P, dtype=float)), np.asarray(Q, dtype=float)
    tS = tau * S_
    om = np.diag(np.diag(P @ tS @ P.T))
    A_ = ...                                      # ✍️ (τΣ)⁻¹ + P'Ω⁻¹P
    b = ...                                       # ✍️ (τΣ)⁻¹π + P'Ω⁻¹Q
    return np.linalg.solve(A_, b)

P = [[-1, 1, 0, 0, 0, 0]]                         # view: international equities beat US equities…
Q = [0.02]                                        # …by 2% a year
mine = p.attempt(black_litterman, cov, w6040, P, Q)
mine = p.check("black_litterman", mine, p.black_litterman(cov, w6040, P=P, Q=Q))
pd.DataFrame({"equilibrium π": p.black_litterman(cov, w6040), "with the view": mine}, index=A.columns).mul(100).round(2)""",
     """def black_litterman(cov, w_mkt, P, Q, delta=2.5, tau=0.05):
    S_ = np.asarray(cov, dtype=float)
    pi = delta * S_ @ np.asarray(w_mkt, dtype=float)
    P, Q = np.atleast_2d(np.asarray(P, dtype=float)), np.asarray(Q, dtype=float)
    tS = tau * S_
    om = np.diag(np.diag(P @ tS @ P.T))
    A_ = np.linalg.inv(tS) + P.T @ np.linalg.inv(om) @ P
    b = np.linalg.solve(tS, pi) + P.T @ np.linalg.solve(om, Q)
    return np.linalg.solve(A_, b)

P = [[-1, 1, 0, 0, 0, 0]]                         # view: international equities beat US equities…
Q = [0.02]                                        # …by 2% a year
mine = p.attempt(black_litterman, cov, w6040, P, Q)
mine = p.check("black_litterman", mine, p.black_litterman(cov, w6040, P=P, Q=Q))
pd.DataFrame({"equilibrium π": p.black_litterman(cov, w6040), "with the view": mine}, index=A.columns).mul(100).round(2)"""),
    ("md", "The view moves the two equity markets apart, and moves the correlated assets (commodities) along with them, by an amount "
           "set by the view's confidence. The result is a sane input for an optimizer, unlike sample means.\n\n"
           "## 4. Minimum CVaR, and the comparison\n\n"
           "`p.min_cvar` solves the Rockafellar–Uryasev linear program: the long-only weights that minimize the historical 95% CVaR. Now "
           "compare seven allocators **walk-forward**: refit monthly on the previous year, hold the next month, through all three crises."),
    ("code", """alloc = {"1/N": lambda X: np.full(X.shape[1], 1 / X.shape[1]),
         "inverse vol": lambda X: p.inverse_vol_weights(X).to_numpy(),
         "min variance (sample)": lambda X: p.min_variance(X.cov().to_numpy()),
         "min variance (shrunk)": lambda X: p.min_variance(p.ledoit_wolf(X)),
         "risk parity": lambda X: p.risk_parity(X.cov().to_numpy()),
         "HRP": lambda X: p.hrp(X).to_numpy(),
         "min CVaR 95%": lambda X: p.min_cvar(X, 0.95)}
rows, curves = {}, {}
for name, f in alloc.items():
    res = p.walk_forward_allocation(A, f)
    r = res["returns"]
    curves[name] = (1 + r).cumprod()
    rows[name] = {"OOS return": r.mean() * 252, "OOS vol": r.std() * np.sqrt(252), "Sharpe": p.sharpe(r),
                  "max drawdown": p.max_drawdown(r)[0], "turnover": res["turnover"]}
table = pd.DataFrame(rows).T
display(table.round(3))
fig, ax = plt.subplots(figsize=(11, 4))
for name, eq in curves.items():
    ax.plot(eq.index, eq, lw=1.2, label=name)
for s in (600, 1700, 2300):
    ax.axvspan(A.index[s], A.index[s + 59], color=p.PALETTE[7], alpha=0.1)
ax.set_title("Walk-forward allocators (shaded: crises)"); ax.legend(ncol=2, fontsize=8); plt.show()"""),
    ("md", "Read the table in two columns. By Sharpe and drawdown the risk-based allocators win, because they load up on the steadiest "
           "assets; by return 1/N wins. A fair comparison scales every portfolio to the **same volatility** first (notebook 10), and "
           "counts turnover as cost. Whichever wins must win walk-forward, not in-sample (common mistake #12).\n\n"
           "## Wrap-up\n\n"
           "* Look at risk contributions, not capital weights.\n"
           "* HRP and risk parity avoid inverting noisy matrices; Black–Litterman gives sane expected returns; CVaR optimizes the tail.\n"
           "* The allocator itself is a parameter: choose it walk-forward.\n"
           "* Graded versions: `labs/part08/week30_portfolio` and Clinic W6 (allocator comparison and recommendation)."),
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
