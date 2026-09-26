"""Build the Part 7 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part07:  python tools/build_notebooks.py
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
for d in (Path.cwd(), Path.cwd().parent):       # p7lib.py is in notebooks/part07/
    sys.path.insert(0, str(d))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p7lib as p

p.use_course_style()"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 7 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_07_STRATEGY_LIBRARY.md) · graded labs in [`labs/part07/`](../../labs/part07/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.
All data is synthetic, built from regimes you know, and every strategy here is a **hypothesis** with a first-look evaluation: the honest backtest comes in Part 8.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
NB["01_framework_quick_eval"] = [
    header("01", "The strategy framework and the first-look evaluator", "S1 (Strategy framework, spec template & first-look evaluator)",
           "1. Write a strategy as a registered class that only emits intents.\n"
           "2. Write the first-look evaluator with the next-bar fill rule.\n"
           "3. See what a same-bar fill does to a worthless signal.\n"
           "4. Write the mandatory next-bar execution test and catch a strategy that peeks."),
    ("code", SETUP),
    ("md", "## 1. A market with known regimes\n\n"
           "Twenty-four half-year blocks, each either **trending** (a steady drift up or down) or **range-bound** (pulled back to where it "
           "started), and either **calm** (10% vol) or **volatile** (28% vol). The labels are in the data, so later we can ask which strategy "
           "works where."),
    ("code", """bars = p.regime_market()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(bars.index, bars.close, lw=0.8, color="black")
for i in range(0, len(bars), 125):
    blk = bars.iloc[i:i + 125]
    ax.axvspan(blk.index[0], blk.index[-1], alpha=0.12, color=p.PALETTE[2] if blk.trend.iloc[0] else p.PALETTE[1], lw=0)
ax.set_title("Green: trending blocks · orange: range-bound blocks"); plt.show()
bars.head(3)"""),
    ("md", "## 2. One class, three runtimes\n\n"
           "A strategy never calls a broker. It sees the bars **up to now** through `self.ctx.history(n)`, and says what it wants with "
           "`self.target(weight, reason)`: an *intent*. The research runner, the backtester (Part 8) and the live OMS (Part 4) all drive the "
           "same class. `@p.register(\"name\")` adds it to the registry, and `params` declares defaults with their allowed ranges "
           "(`name: (default, min, max)`) so configs can be validated and optimizers know the search space."),
    ("md", YOUR_TURN + "\n\nWrite `on_bar`: once there are at least `slow` bars of history, target weight **1** when the fast SMA of the closes "
           "is above the slow SMA, else **0** (use `p.sma(closes, n)[-1]`). Emit nothing during the warm-up."),
    ("ex", """p.REGISTRY.pop("sma_cross", None)                # so the cell can be re-run

@p.register("sma_cross")
class SmaCross(p.Strategy):
    params = {"fast": (20, 5, 100), "slow": (100, 20, 300)}

    def on_bar(self):
        closes = self.ctx.history(self.p["slow"])["close"].to_numpy()
        if len(closes) < self.p["slow"]:
            return
        ...                                       # ✍️ target 1 or 0 with a reason

decided = p.attempt(p.run, SmaCross, bars)
fast, slow = p.sma(bars.close, 20), p.sma(bars.close, 100)
ref_dec = np.where(np.nan_to_num(fast) > np.nan_to_num(slow, nan=np.inf), 1.0, 0.0)
mine = p.check("sma_cross decisions", decided, ref_dec)
print(f"in the market {mine.mean():.0%} of bars")""",
     """p.REGISTRY.pop("sma_cross", None)                # so the cell can be re-run

@p.register("sma_cross")
class SmaCross(p.Strategy):
    params = {"fast": (20, 5, 100), "slow": (100, 20, 300)}

    def on_bar(self):
        closes = self.ctx.history(self.p["slow"])["close"].to_numpy()
        if len(closes) < self.p["slow"]:
            return
        up = p.sma(closes, self.p["fast"])[-1] > p.sma(closes, self.p["slow"])[-1]
        self.target(1.0 if up else 0.0, "fast above slow" if up else "fast below slow")

decided = p.attempt(p.run, SmaCross, bars)
fast, slow = p.sma(bars.close, 20), p.sma(bars.close, 100)
ref_dec = np.where(np.nan_to_num(fast) > np.nan_to_num(slow, nan=np.inf), 1.0, 0.0)
mine = p.check("sma_cross decisions", decided, ref_dec)
print(f"in the market {mine.mean():.0%} of bars")"""),
    ("code", """try:
    p.run(p.REGISTRY["sma_cross"], bars, fast=500)
except ValueError as e:
    print("config validation:", e)"""),
    ("md", "## 3. The first-look evaluator\n\n"
           "A decision made at bar `t`'s **close** can be filled at bar `t+1`'s **open** at the earliest. So the position held from open `t` "
           "to open `t+1` is `decided[t−1]` (flat on the first bar), and each change of position pays `cost_bps` on the turnover. "
           "Return the daily P&L array `pos · ret − turnover · cost_bps / 10,000`."),
    ("md", YOUR_TURN),
    ("ex", """def first_look_pnl(decided, open_, cost_bps=2.0):
    decided = np.nan_to_num(np.asarray(decided, dtype=float))
    pos = np.zeros_like(decided)
    pos[1:] = ...                                 # ✍️ the decision of the previous bar
    ret = np.zeros_like(open_)
    ret[:-1] = open_[1:] / open_[:-1] - 1         # open-to-open
    turnover = np.abs(np.diff(pos, prepend=0.0))
    return ...                                    # ✍️ the P&L after costs

o = bars.open.to_numpy()
mine = p.attempt(first_look_pnl, ref_dec, o)
mine = p.check("first-look P&L", mine, p.quick_eval(ref_dec, o)["pnl"])
p.summary(p.quick_eval(ref_dec, o))""",
     """def first_look_pnl(decided, open_, cost_bps=2.0):
    decided = np.nan_to_num(np.asarray(decided, dtype=float))
    pos = np.zeros_like(decided)
    pos[1:] = decided[:-1]
    ret = np.zeros_like(open_)
    ret[:-1] = open_[1:] / open_[:-1] - 1         # open-to-open
    turnover = np.abs(np.diff(pos, prepend=0.0))
    return pos * ret - turnover * cost_bps / 1e4

o = bars.open.to_numpy()
mine = p.attempt(first_look_pnl, ref_dec, o)
mine = p.check("first-look P&L", mine, p.quick_eval(ref_dec, o)["pnl"])
p.summary(p.quick_eval(ref_dec, o))"""),
    ("md", "## 4. Same-bar fills are fantasy\n\n"
           "A worthless signal: \"today closed higher than yesterday, so be long\". Evaluate it honestly (fill at the next open), and then with "
           "the most common backtest bug, filling at the open **of the bar that produced the signal**: the position is taken before the close "
           "that decided it."),
    ("code", """c = bars.close.to_numpy()
today_up = np.r_[False, c[1:] > c[:-1]].astype(float)
honest = p.quick_eval(today_up, o)
peeking = p.quick_eval(np.r_[today_up[1:], 0.0], o)     # shifting the decision one bar earlier = filling on its own bar
print(f"next-bar fill: Sharpe {honest['sharpe']:+.2f}")
print(f"same-bar fill: Sharpe {peeking['sharpe']:+.2f}  ← a result like this is a bug until proven otherwise")"""),
    ("md", "## 5. The mandatory next-bar execution test\n\n"
           "The runner only hands a strategy `history()`, but nothing stops a careless strategy from reading `self.ctx.bars` directly. "
           "The test that catches any such leak: run the strategy on the data **cut at bar t**, and check that its decisions up to `t` are "
           "identical to the ones it made on the full data. A causal strategy can't tell the difference; a peeking one can."),
    ("code", """p.REGISTRY.pop("peeker", None)

@p.register("peeker")
class Peeker(p.Strategy):
    \"\"\"Looks one bar ahead through ctx.bars: it would never pass review.\"\"\"
    def on_bar(self):
        b, t = self.ctx.bars, self.ctx.t
        if t + 1 < len(b):
            self.target(1.0 if b.close.iloc[t + 1] > b.close.iloc[t] else 0.0, "knows tomorrow")"""),
    ("md", YOUR_TURN),
    ("ex", """def next_bar_test(strategy_cls, bars, cuts=(500, 1200, 2000)):
    full = p.run(strategy_cls, bars)
    for t in cuts:
        cut = p.run(strategy_cls, bars.iloc[: t + 1])
        if ...:                                   # ✍️ the decisions up to t differ from the full-data run
            return False
    return True

mine = [p.attempt(next_bar_test, p.REGISTRY["sma_cross"], bars), p.attempt(next_bar_test, p.REGISTRY["peeker"], bars)]
mine = p.check("next_bar_test", mine, [True, False])
print(f"sma_cross passes: {mine[0]};  peeker passes: {mine[1]}  (its first-look Sharpe: {p.quick_eval(p.run(Peeker, bars), o)['sharpe']:.1f})")""",
     """def next_bar_test(strategy_cls, bars, cuts=(500, 1200, 2000)):
    full = p.run(strategy_cls, bars)
    for t in cuts:
        cut = p.run(strategy_cls, bars.iloc[: t + 1])
        if not np.array_equal(cut, full[: t + 1]):
            return False
    return True

mine = [p.attempt(next_bar_test, p.REGISTRY["sma_cross"], bars), p.attempt(next_bar_test, p.REGISTRY["peeker"], bars)]
mine = p.check("next_bar_test", mine, [True, False])
print(f"sma_cross passes: {mine[0]};  peeker passes: {mine[1]}  (its first-look Sharpe: {p.quick_eval(p.run(Peeker, bars), o)['sharpe']:.1f})")"""),
    ("md", "## Wrap-up\n\n"
           "* Strategies emit intents; the risk engine and OMS decide what is sent.\n"
           "* Declared parameter ranges make configs checkable and optimization honest.\n"
           "* Next-bar fills in the evaluator, and a truncation test on every strategy (it is a pass/fail criterion of milestone M3b).\n"
           "* The first-look evaluator is deliberately crude; Part 8 replaces it.\n"
           "* Graded version: `labs/part07/week23_framework_momentum` (base class, registry, YAML configs, runner, `quick_eval`, the next-bar test)."),
]

# ---------------------------------------------------------------------------------------------- 02
NB["02_momentum"] = [
    header("02", "Directional momentum", "S2 (Directional momentum group)",
           "1. Write time-series momentum with volatility targeting.\n"
           "2. Write the exits of a Donchian breakout (a small state machine).\n"
           "3. Build 12-1 cross-sectional momentum.\n"
           "4. See where momentum works, and where it bleeds, using the known regimes."),
    ("code", SETUP),
    ("code", """bars = p.regime_market()
o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))"""),
    ("md", "## 1. Time-series momentum (TSMOM)\n\n"
           "Be long if the asset is up over the last year, short if down, and size the position so its risk is constant: "
           "`sign(C[t]/C[t−lookback] − 1) × target_vol / realized_vol`, clipped to `±max_lev`. Realized vol is the rolling standard deviation "
           "of daily log returns over `vol_n` days, annualized. Without the vol scaling, momentum's drawdowns come in the volatile periods."),
    ("md", YOUR_TURN),
    ("ex", """def tsmom(close, lookback=252, vol_n=60, target_vol=0.10, max_lev=2.0):
    past = np.full(close.shape, np.nan)
    past[lookback:] = close[lookback:] / close[:-lookback] - 1
    r = np.full(close.shape, np.nan)
    r[1:] = np.diff(np.log(close))
    vol = pd.Series(r).rolling(vol_n).std().to_numpy() * np.sqrt(252)
    return ...                                    # ✍️ direction × target / vol, clipped to ±max_lev

mine = p.attempt(tsmom, c)
mine = p.check("tsmom", mine, p.tsmom(c))
p.summary(p.quick_eval(mine, o))""",
     """def tsmom(close, lookback=252, vol_n=60, target_vol=0.10, max_lev=2.0):
    past = np.full(close.shape, np.nan)
    past[lookback:] = close[lookback:] / close[:-lookback] - 1
    r = np.full(close.shape, np.nan)
    r[1:] = np.diff(np.log(close))
    vol = pd.Series(r).rolling(vol_n).std().to_numpy() * np.sqrt(252)
    return np.clip(np.sign(past) * target_vol / vol, -max_lev, max_lev)

mine = p.attempt(tsmom, c)
mine = p.check("tsmom", mine, p.tsmom(c))
p.summary(p.quick_eval(mine, o))"""),
    ("md", "## 2. The Donchian breakout\n\n"
           "The Turtle rule, long and short. Channels use the **prior** bars only (today's bar can't be part of the level it breaks): "
           "entries on a break of the prior `entry_n`-bar high or low, exits on a break of the prior `exit_n`-bar low (for a long) or high "
           "(for a short). The flat branch is written; write the two exits."),
    ("md", YOUR_TURN),
    ("ex", """def donchian(high, low, close, entry_n=20, exit_n=10):
    pos = np.zeros(close.size)
    for t in range(max(entry_n, exit_n), close.size):
        prev = pos[t - 1]
        if prev == 0:
            if close[t] > high[t - entry_n:t].max():
                pos[t] = 1
            elif close[t] < low[t - entry_n:t].min():
                pos[t] = -1
        elif prev == 1:
            pos[t] = ...                          # ✍️ 0 if close < the prior exit_n lows' minimum, else stay long
        else:
            pos[t] = ...                          # ✍️ 0 if close > the prior exit_n highs' maximum, else stay short
    return pos

mine = p.attempt(donchian, h, l, c)
mine = p.check("donchian", mine, p.donchian_breakout(h, l, c))
print(f"long {np.mean(np.asarray(mine) == 1):.0%}, short {np.mean(np.asarray(mine) == -1):.0%}, flat {np.mean(np.asarray(mine) == 0):.0%} of bars")""",
     """def donchian(high, low, close, entry_n=20, exit_n=10):
    pos = np.zeros(close.size)
    for t in range(max(entry_n, exit_n), close.size):
        prev = pos[t - 1]
        if prev == 0:
            if close[t] > high[t - entry_n:t].max():
                pos[t] = 1
            elif close[t] < low[t - entry_n:t].min():
                pos[t] = -1
        elif prev == 1:
            pos[t] = 0 if close[t] < low[t - exit_n:t].min() else 1
        else:
            pos[t] = 0 if close[t] > high[t - exit_n:t].max() else -1
    return pos

mine = p.attempt(donchian, h, l, c)
mine = p.check("donchian", mine, p.donchian_breakout(h, l, c))
print(f"long {np.mean(np.asarray(mine) == 1):.0%}, short {np.mean(np.asarray(mine) == -1):.0%}, flat {np.mean(np.asarray(mine) == 0):.0%} of bars")"""),
    ("md", "## 3. Where momentum works\n\n"
           "Sharpe ratio of each strategy's first-look P&L **inside** each of the four known regimes. This table, not the overall Sharpe, "
           "is what the strategy spec's \"works when / fails when\" section is made of."),
    ("code", """signals = {"TSMOM": p.tsmom(c), "Donchian 20/10": p.donchian_breakout(h, l, c), "buy & hold": np.ones(len(c))}
table = p.regime_table(bars, signals)
display(table.round(2))
eq = {k: np.cumprod(1 + p.quick_eval(v, o)["pnl"]) for k, v in signals.items()}
fig, ax = plt.subplots(figsize=(11, 3.8))
for k, v in eq.items():
    ax.plot(bars.index, v, label=k)
ax.set_yscale("log"); ax.set_title("First-look equity (not a backtest)"); ax.legend(); plt.show()"""),
    ("md", "## 4. Cross-sectional momentum (12-1)\n\n"
           "Across a universe, buy the recent **winners**. Momentum is measured from 12 months ago to 1 month ago, skipping the latest month "
           "(which tends to reverse): `mom = closes.shift(1) / closes.shift(12) − 1` on month-end prices. Rank across assets each month "
           "(`rank(axis=1, pct=True)`) and equal-weight the names ranked strictly above `1 − top_q`."),
    ("code", """uni = p.momentum_universe()
uni.iloc[:, :6].plot(figsize=(10, 3.5), legend=False, logy=True, title="12 assets with persistent (but slowly changing) drifts"); plt.show()"""),
    ("md", YOUR_TURN),
    ("ex", """def xs_momentum(closes, top_q=0.25):
    mom = ...                                     # ✍️ 12-1 momentum
    rank = mom.rank(axis=1, pct=True)
    w = (rank > 1 - top_q + 1e-12).astype(float)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)

mine = p.attempt(xs_momentum, uni)
mine = p.check("cross-sectional momentum", mine, p.cross_sectional_momentum(uni))
fwd = uni.pct_change().shift(-1)                  # next month's return: weights set at month end t earn month t+1
top, avg = (mine * fwd).sum(axis=1)[13:-1], fwd.mean(axis=1)[13:-1]
sr = lambda x: x.mean() / x.std() * np.sqrt(12)
print(f"winners: Sharpe {sr(top):.2f};  equal weight: {sr(avg):.2f};  winners minus average: {sr(top - avg):.2f}")""",
     """def xs_momentum(closes, top_q=0.25):
    mom = closes.shift(1) / closes.shift(12) - 1
    rank = mom.rank(axis=1, pct=True)
    w = (rank > 1 - top_q + 1e-12).astype(float)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)

mine = p.attempt(xs_momentum, uni)
mine = p.check("cross-sectional momentum", mine, p.cross_sectional_momentum(uni))
fwd = uni.pct_change().shift(-1)                  # next month's return: weights set at month end t earn month t+1
top, avg = (mine * fwd).sum(axis=1)[13:-1], fwd.mean(axis=1)[13:-1]
sr = lambda x: x.mean() / x.std() * np.sqrt(12)
print(f"winners: Sharpe {sr(top):.2f};  equal weight: {sr(avg):.2f};  winners minus average: {sr(top - avg):.2f}")"""),
    ("md", "In this universe expected returns are persistent by construction, which is exactly the assumption momentum bets on. "
           "In real markets the premium is smaller, decays after publication, and crashes when the market rebounds sharply "
           "(Daniel & Moskowitz, 2016).\n\n"
           "## Wrap-up\n\n"
           "* Momentum earns in trends and bleeds in ranges: say so in the spec, and size by volatility.\n"
           "* Breakout channels use prior bars only; momentum signals skip the most recent month.\n"
           "* Graded version: `labs/part07/week23_framework_momentum` (TSMOM, Donchian vectorized and live-style, 12-1, dual momentum)."),
]

# ---------------------------------------------------------------------------------------------- 03
NB["03_mean_reversion"] = [
    header("03", "Mean reversion and range-bound strategies", "S3 (Mean reversion & range-bound groups)",
           "1. Compute internal bar strength with the flat-bar guard.\n"
           "2. Give a z-score reversion trade a time stop.\n"
           "3. Put a trend filter on a short-term RSI(2) entry.\n"
           "4. See mean reversion earn in ranges and lose in trends."),
    ("code", SETUP),
    ("code", """bars = p.regime_market()
o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))"""),
    ("md", "## 1. Internal bar strength\n\n"
           "`IBS = (C − L) / (H − L)`: 0 when the bar closed on its low, 1 on its high. A low IBS is a classic short-term buy signal in index ETFs. "
           "On a flat bar (H = L) return **0.5**, not a division by zero."),
    ("md", YOUR_TURN),
    ("ex", """def ibs(high, low, close):
    rng_ = high - low
    return ...                                    # ✍️ np.divide(..., out=np.full(close.shape, 0.5), where=rng_ > 0)

h2, l2, c2 = np.append(h, 50.0), np.append(l, 50.0), np.append(c, 50.0)       # one flat bar at the end
mine = p.attempt(ibs, h2, l2, c2)
mine = p.check("ibs", mine, p.ibs(h2, l2, c2))
np.round(mine[-4:], 3)""",
     """def ibs(high, low, close):
    rng_ = high - low
    return np.divide(close - low, rng_, out=np.full(close.shape, 0.5), where=rng_ > 0)

h2, l2, c2 = np.append(h, 50.0), np.append(l, 50.0), np.append(c, 50.0)       # one flat bar at the end
mine = p.attempt(ibs, h2, l2, c2)
mine = p.check("ibs", mine, p.ibs(h2, l2, c2))
np.round(mine[-4:], 3)"""),
    ("md", "## 2. Z-score reversion with a time stop\n\n"
           "`z = (C − SMA(n)) / rolling std`. Flat → long when `z < −entry`, short when `z > entry`. In a trade → exit when the price has "
           "come back (`|z| < exit_`) **or** after `max_hold` bars. The time stop is the admission that a move that doesn't revert is a trend. "
           "`held` counts the bars since entry (1 on the entry bar). Write the in-trade branch."),
    ("md", YOUR_TURN),
    ("ex", """def zscore_reversion(close, n=20, entry=2.0, exit_=0.5, max_hold=10):
    z = p.rolling_z(close, n)
    pos, held = np.zeros(z.size), 0
    for t in range(1, z.size):
        if not np.isfinite(z[t]):
            continue
        if pos[t - 1] == 0:
            pos[t] = 1.0 if z[t] < -entry else (-1.0 if z[t] > entry else 0.0)
            held = 1 if pos[t] else 0
        else:
            held += 1
            pos[t] = ...                          # ✍️ 0 if reverted or held too long, else keep pos[t − 1]
    return pos

mine = p.attempt(zscore_reversion, c)
mine = p.check("zscore_reversion", mine, p.zscore_reversion(c))
print(f"in a trade {np.mean(np.asarray(mine) != 0):.0%} of bars")""",
     """def zscore_reversion(close, n=20, entry=2.0, exit_=0.5, max_hold=10):
    z = p.rolling_z(close, n)
    pos, held = np.zeros(z.size), 0
    for t in range(1, z.size):
        if not np.isfinite(z[t]):
            continue
        if pos[t - 1] == 0:
            pos[t] = 1.0 if z[t] < -entry else (-1.0 if z[t] > entry else 0.0)
            held = 1 if pos[t] else 0
        else:
            held += 1
            pos[t] = 0.0 if (abs(z[t]) < exit_ or held > max_hold) else pos[t - 1]
    return pos

mine = p.attempt(zscore_reversion, c)
mine = p.check("zscore_reversion", mine, p.zscore_reversion(c))
print(f"in a trade {np.mean(np.asarray(mine) != 0):.0%} of bars")"""),
    ("md", "## 3. RSI(2) with a trend filter\n\n"
           "Connors' rule: buy a sharp short-term dip (`RSI(2) < entry`), but **only while the long-term trend is up** "
           "(`close > SMA(trend_n)`), and sell when `RSI(2) > exit_`. Write the entry condition."),
    ("md", YOUR_TURN),
    ("ex", """def rsi2(close, entry=10, exit_=70, trend_n=200, use_trend=True):
    r, m = p.rsi(close, 2), p.sma(close, trend_n)
    pos = np.zeros(close.size)
    for t in range(1, close.size):
        if pos[t - 1] == 0:
            ok = ...                              # ✍️ RSI(2) below entry, and (no filter, or close above its SMA)
            pos[t] = 1.0 if ok else 0.0
        else:
            pos[t] = 0.0 if r[t] > exit_ else 1.0
    return pos

mine = [p.attempt(rsi2, c, use_trend=True), p.attempt(rsi2, c, use_trend=False)]
mine = p.check("rsi2", mine, [p.rsi2_reversion(c, use_trend=True), p.rsi2_reversion(c, use_trend=False)])
print(f"trades taken: {int(np.sum(np.diff(mine[0]) > 0))} with the filter, {int(np.sum(np.diff(mine[1]) > 0))} without")""",
     """def rsi2(close, entry=10, exit_=70, trend_n=200, use_trend=True):
    r, m = p.rsi(close, 2), p.sma(close, trend_n)
    pos = np.zeros(close.size)
    for t in range(1, close.size):
        if pos[t - 1] == 0:
            ok = r[t] < entry and (not use_trend or close[t] > m[t])
            pos[t] = 1.0 if ok else 0.0
        else:
            pos[t] = 0.0 if r[t] > exit_ else 1.0
    return pos

mine = [p.attempt(rsi2, c, use_trend=True), p.attempt(rsi2, c, use_trend=False)]
mine = p.check("rsi2", mine, [p.rsi2_reversion(c, use_trend=True), p.rsi2_reversion(c, use_trend=False)])
print(f"trades taken: {int(np.sum(np.diff(mine[0]) > 0))} with the filter, {int(np.sum(np.diff(mine[1]) > 0))} without")"""),
    ("md", "## 4. Where mean reversion works\n\n"
           "Same regime table as for momentum. The time stop and the trend filter don't make mean reversion work in a trend; they limit "
           "how much it loses there."),
    ("code", """signals = {"z-score (time stop 10)": p.zscore_reversion(c), "z-score (no time stop)": p.zscore_reversion(c, max_hold=10_000),
           "RSI(2) + trend filter": p.rsi2_reversion(c), "RSI(2), no filter": p.rsi2_reversion(c, use_trend=False),
           "IBS < 0.2, one day": (p.ibs(h, l, c) < 0.2).astype(float)}
display(p.regime_table(bars, signals).round(2))
worst = {k: pd.Series(p.quick_eval(v, o)["pnl"]).rolling(20).sum().min() for k, v in signals.items()}
print("worst 20-day P&L:", {k: f"{v:.1%}" for k, v in worst.items()})"""),
    ("md", "## Wrap-up\n\n"
           "* Mean reversion is the mirror image of momentum: it earns in ranges and pays in trends.\n"
           "* Always add a regime filter and a maximum holding period, and state them in the spec.\n"
           "* Graded version: `labs/part07/week23_linear_groups` (RSI(2), IBS, z-score, Bollinger fade gated by ADX, opening-range breakout)."),
]

# ---------------------------------------------------------------------------------------------- 04
NB["04_vol_math_stat"] = [
    header("04", "Volatility overlays, the Hurst exponent and pairs", "S4 (Either-way, volatility, mathematical & statistical groups)",
           "1. Put a volatility-target overlay on any strategy.\n"
           "2. Measure trending vs mean-reverting behaviour with the Hurst exponent.\n"
           "3. Build a pairs trade whose hedge ratio never sees the future.\n"
           "4. Watch a pair break."),
    ("code", SETUP),
    ("code", """bars = p.regime_market()
o, c = bars.open.to_numpy(), bars.close.to_numpy()"""),
    ("md", "## 1. The volatility-target overlay\n\n"
           "Scale any position so its risk is roughly constant: multiply by `target / realized vol`, where realized vol is the annualized "
           "rolling standard deviation (`ddof=1`) of the returns **known at each close** over the last `n` bars. Clip the scale to "
           "`[0, max_lev]`, and use 0 where the vol is not known yet (`np.nan_to_num`)."),
    ("md", YOUR_TURN),
    ("ex", """def vol_target_overlay(pos, returns, target=0.10, n=20, max_lev=2.0):
    vol = pd.Series(returns).rolling(n).std().to_numpy() * np.sqrt(252)
    scale = ...                                   # ✍️ target / vol, 0 where vol is 0, clipped to [0, max_lev]
    return np.asarray(pos, dtype=float) * np.nan_to_num(scale)

ret = np.r_[0.0, c[1:] / c[:-1] - 1]              # close-to-close return known at each close
mine = p.attempt(vol_target_overlay, np.ones(len(c)), ret)
mine = p.check("vol_target_overlay", mine, p.vol_target_overlay(np.ones(len(c)), ret))
for name, pos in [("buy & hold", np.ones(len(c))), ("buy & hold, vol-targeted to 10%", mine)]:
    s = p.summary(p.quick_eval(pos, o))
    print(f"{name:32s} Sharpe {s['sharpe']:+.2f}  max drawdown {s['max_dd']:.0%}")""",
     """def vol_target_overlay(pos, returns, target=0.10, n=20, max_lev=2.0):
    vol = pd.Series(returns).rolling(n).std().to_numpy() * np.sqrt(252)
    scale = np.clip(np.divide(target, vol, out=np.zeros_like(vol), where=vol > 0), 0, max_lev)
    return np.asarray(pos, dtype=float) * np.nan_to_num(scale)

ret = np.r_[0.0, c[1:] / c[:-1] - 1]              # close-to-close return known at each close
mine = p.attempt(vol_target_overlay, np.ones(len(c)), ret)
mine = p.check("vol_target_overlay", mine, p.vol_target_overlay(np.ones(len(c)), ret))
for name, pos in [("buy & hold", np.ones(len(c))), ("buy & hold, vol-targeted to 10%", mine)]:
    s = p.summary(p.quick_eval(pos, o))
    print(f"{name:32s} Sharpe {s['sharpe']:+.2f}  max drawdown {s['max_dd']:.0%}")"""),
    ("md", "## 2. The Hurst exponent\n\n"
           "How does the spread of price changes grow with the horizon? For a random walk, `std(x[t+lag] − x[t]) ∝ lag^0.5`. Faster growth "
           "(`H > 0.5`) means moves persist (trending); slower (`H < 0.5`) means they undo themselves (mean reverting). Estimate `H` as the "
           "slope of `log std` on `log lag` (`np.polyfit(..., 1)[0]`). Pass **log** prices."),
    ("md", YOUR_TURN),
    ("ex", """def hurst(x, lags=range(2, 64)):
    lags = np.asarray(list(lags))
    tau = ...                                     # ✍️ the std of x[lag:] − x[:-lag] for each lag
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])

logc = np.log(c)
blocks = {f"block {i // 125:2d} ({'trend' if bars.trend.iloc[i] else 'range'})": logc[i:i + 125] for i in range(0, 750, 125)}
mine = {k: p.attempt(hurst, v) for k, v in blocks.items()}
mine = p.check("hurst", mine, {k: p.hurst_exponent(v) for k, v in blocks.items()})
pd.Series(mine).round(2)""",
     """def hurst(x, lags=range(2, 64)):
    lags = np.asarray(list(lags))
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])

logc = np.log(c)
blocks = {f"block {i // 125:2d} ({'trend' if bars.trend.iloc[i] else 'range'})": logc[i:i + 125] for i in range(0, 750, 125)}
mine = {k: p.attempt(hurst, v) for k, v in blocks.items()}
mine = p.check("hurst", mine, {k: p.hurst_exponent(v) for k, v in blocks.items()})
pd.Series(mine).round(2)"""),
    ("md", "Single blocks are noisy. Average over all 24 blocks by their **true** regime (with lags up to 19 bars, short enough for "
           "125-bar blocks), and compare with a pure random walk:"),
    ("code", """by_regime = {"range-bound": [], "trending": []}
for i in range(0, len(logc), 125):
    by_regime["trending" if bars.trend.iloc[i] else "range-bound"].append(p.hurst_exponent(logc[i:i + 125], range(2, 20)))
rw = np.cumsum(np.random.default_rng(0).normal(0, 0.01, 3000))
display(pd.DataFrame({k: {"mean H": np.mean(v), "sd across blocks": np.std(v), "blocks": len(v)} for k, v in by_regime.items()}).T.round(2))
print(f"random walk: H = {p.hurst_exponent(rw, range(2, 20)):.2f}")"""),
    ("md", "Range-bound blocks sit clearly below 0.5 on average. The trending blocks sit near 0.5, not above it: their trend is a steady "
           "**drift** plus independent noise, and the Hurst exponent measures whether *changes* persist, which a drift doesn't create "
           "(the standard deviation removes it). And block-to-block scatter is large. A regime label estimated from the data is a noisy "
           "guess at the truth; the lab's Clinic W1 compares strategies under true and estimated labels.\n\n"
           "## 3. A pairs trade without look-ahead\n\n"
           "Two log prices with `y ≈ β·x + spread`, where the spread mean-reverts. At bar `t`, estimate `β` by regression on the **previous** "
           "`lookback` bars only, then compare today's spread with that window. Write the window: it must end at `t − 1`."),
    ("code", """y, x = p.cointegrated_pair()
plt.figure(figsize=(10, 3.2)); plt.plot(y, label="log y"); plt.plot(x, label="log x"); plt.legend(); plt.title("A cointegrated pair (true β = 1.4)"); plt.show()"""),
    ("md", YOUR_TURN),
    ("ex", """def pairs(y, x, lookback=60, entry=2.0, exit_=0.5):
    pos, betas = np.zeros(y.size), np.full(y.size, np.nan)
    for t in range(lookback, y.size):
        ys, xs = ...                              # ✍️ the previous `lookback` bars of y and x (t itself excluded)
        beta = np.polyfit(xs, ys, 1)[0]
        betas[t] = beta
        spread = ys - beta * xs
        sd = spread.std()
        z = (y[t] - beta * x[t] - spread.mean()) / sd if sd > 0 else 0.0
        prev = pos[t - 1]
        if prev == 0:
            pos[t] = 1.0 if z < -entry else (-1.0 if z > entry else 0.0)
        else:
            pos[t] = 0.0 if abs(z) < exit_ else prev
    return pos, betas

res = p.attempt(pairs, y, x)
mine = list(res) if res is not Ellipsis else Ellipsis
mine = p.check("pairs", mine, list(p.pairs_positions(y, x)))
pnl = p.pairs_pnl(mine[0], y, x, mine[1])
print(f"spread trade: Sharpe {pnl.mean() / pnl.std() * np.sqrt(252):.2f}, mean estimated β {np.nanmean(mine[1]):.2f}")""",
     """def pairs(y, x, lookback=60, entry=2.0, exit_=0.5):
    pos, betas = np.zeros(y.size), np.full(y.size, np.nan)
    for t in range(lookback, y.size):
        ys, xs = y[t - lookback:t], x[t - lookback:t]
        beta = np.polyfit(xs, ys, 1)[0]
        betas[t] = beta
        spread = ys - beta * xs
        sd = spread.std()
        z = (y[t] - beta * x[t] - spread.mean()) / sd if sd > 0 else 0.0
        prev = pos[t - 1]
        if prev == 0:
            pos[t] = 1.0 if z < -entry else (-1.0 if z > entry else 0.0)
        else:
            pos[t] = 0.0 if abs(z) < exit_ else prev
    return pos, betas

res = p.attempt(pairs, y, x)
mine = list(res) if res is not Ellipsis else Ellipsis
mine = p.check("pairs", mine, list(p.pairs_positions(y, x)))
pnl = p.pairs_pnl(mine[0], y, x, mine[1])
print(f"spread trade: Sharpe {pnl.mean() / pnl.std() * np.sqrt(252):.2f}, mean estimated β {np.nanmean(mine[1]):.2f}")"""),
    ("md", "## 4. When the pair breaks\n\n"
           "Same pair, but at bar 900 the relationship changes (β drops by 40%). The rolling hedge ratio adapts, slowly, and the spread "
           "\"reverts\" to a level that no longer exists. That is why pairs books need a break test and a retirement rule (Part 9)."),
    ("code", """y2, x2 = p.cointegrated_pair(break_at=900)
pos2, b2 = p.pairs_positions(y2, x2)
pnl2 = p.pairs_pnl(pos2, y2, x2, b2)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(b2); axes[0].axvline(900, color=p.PALETTE[7], ls="--"); axes[0].set_title("Rolling β (true 1.4, then 0.84)")
axes[1].plot(np.cumsum(pnl2)); axes[1].axvline(900, color=p.PALETTE[7], ls="--"); axes[1].set_title("Cumulative spread P&L (log units)")
plt.tight_layout(); plt.show()
print(f"P&L before the break {pnl2[:900].sum():+.3f}, after {pnl2[900:].sum():+.3f}")"""),
    ("md", "## Wrap-up\n\n"
           "* Volatility targeting is an overlay for any strategy: it evens out risk, and here it cuts the drawdown.\n"
           "* Hurst and similar statistics label regimes, noisily.\n"
           "* Every estimated input (β, mean, std) uses only past bars; relationships break, so monitor them.\n"
           "* Graded version: `labs/part07/week23_linear_groups` (VIX-regime overlay, Kalman level, turn-of-month, overnight vs intraday too)."),
]

# ---------------------------------------------------------------------------------------------- 05
NB["05_option_builder"] = [
    header("05", "The option strategy builder", "S5 (Option strategy builder)",
           "1. Value a multi-leg strategy today, later, and at expiry.\n"
           "2. Find the breakevens of its expiry P&L.\n"
           "3. Decide whether a structure is defined-risk.\n"
           "4. See why a high probability of profit is not an edge."),
    ("code", SETUP),
    ("md", "## 1. Legs and strategies\n\n"
           "A strategy is a list of legs `(cp, K, T, qty, iv)` with `cp = +1` call, `−1` put, `0` for the underlying, and a multiplier of 100. "
           "Here is a 30-day iron condor on a stock at 600, shorts at about 16 delta, wings 10 points further out, each strike priced on a "
           "simple equity smile (`p.skew_iv`)."),
    ("code", """S, T = 600.0, 30 / 365
ivf = p.skew_iv(S)
strikes = np.arange(450, 751, 5.0)
ic = p.iron_condor(S, T, ivf, strikes)
pd.DataFrame([vars(L) for L in ic.legs]).round(4)"""),
    ("md", "## 2. The value of a strategy\n\n"
           "Mark each leg after `dt` years have passed and with every IV shifted by `dvol`: an option leg with time left (`tau = T − dt > 1e-9`) "
           "at `p.bsm_price(S, K, tau, r, q, iv + dvol, cp)`, an expired one at its intrinsic value `max(cp·(S − K), 0)`, the underlying at "
           "`S`. Sum `qty × value` and multiply by the multiplier. `S` may be an array."),
    ("md", YOUR_TURN),
    ("ex", """def strategy_value(strat, S, r=0.04, q=0.0, dt=0.0, dvol=0.0):
    S = np.asarray(S, dtype=float)
    v = np.zeros_like(S)
    for L in strat.legs:
        if L.cp == 0:
            v = v + L.qty * S
            continue
        tau = L.T - dt
        leg = ...                                 # ✍️ BSM with the time left, or intrinsic value once expired
        v = v + L.qty * leg
    return v * strat.multiplier

grid = np.linspace(540, 660, 7)
cases = [dict(dt=0.0), dict(dt=15 / 365), dict(dt=T), dict(dt=0.0, dvol=0.05)]
mine = [p.attempt(strategy_value, ic, grid, **k) for k in cases]
mine = p.check("strategy_value", mine, [ic.value(grid, **k) for k in cases])
pd.DataFrame(np.round(mine, 0), index=["today", "in 15 days", "at expiry", "today, IV +5"], columns=grid)""",
     """def strategy_value(strat, S, r=0.04, q=0.0, dt=0.0, dvol=0.0):
    S = np.asarray(S, dtype=float)
    v = np.zeros_like(S)
    for L in strat.legs:
        if L.cp == 0:
            v = v + L.qty * S
            continue
        tau = L.T - dt
        leg = p.bsm_price(S, L.K, tau, r, q, L.iv + dvol, L.cp) if tau > 1e-9 else np.maximum(L.cp * (S - L.K), 0.0)
        v = v + L.qty * leg
    return v * strat.multiplier

grid = np.linspace(540, 660, 7)
cases = [dict(dt=0.0), dict(dt=15 / 365), dict(dt=T), dict(dt=0.0, dvol=0.05)]
mine = [p.attempt(strategy_value, ic, grid, **k) for k in cases]
mine = p.check("strategy_value", mine, [ic.value(grid, **k) for k in cases])
pd.DataFrame(np.round(mine, 0), index=["today", "in 15 days", "at expiry", "today, IV +5"], columns=grid)"""),
    ("md", "The condor is a **credit** (negative value today: we receive money). It gains as time passes and loses if IV rises."),
    ("code", """g = np.linspace(520, 680, 400)
cost = float(ic.value(S))
fig, ax = plt.subplots()
for dt, lab in [(0.0, "today"), (15 / 365, "in 15 days"), (T, "at expiry")]:
    ax.plot(g, ic.value(g, dt=dt) - cost, label=lab)
ax.axhline(0, color="black", lw=0.6); ax.axvline(S, color="#e6e5e0")
ax.set(xlabel="stock price", ylabel="P&L, $", title="Iron condor P&L"); ax.legend(); plt.show()
ic.greeks(S)"""),
    ("md", "## 3. Breakevens\n\n"
           "On a fine price grid, the breakevens are where the expiry P&L changes sign: take `grid[1:]` where `sign(pnl[1:]) != sign(pnl[:-1])`, "
           "rounded to 2 decimals."),
    ("md", YOUR_TURN),
    ("ex", """def breakevens(grid, pnl):
    sign = np.sign(pnl)
    return ...                                    # ✍️

gg = np.linspace(0.5 * S, 1.5 * S, 2001)
pnl = ic.payoff_at_first_expiry(gg) - ic.value(S)
mine = p.attempt(breakevens, gg, pnl)
mine = p.check("breakevens", mine, ic.analyze(S)["breakevens"])
a = ic.analyze(S)
print(f"breakevens {mine}; credit ${-a['cost']:.0f}; max profit ${a['max_profit']:.0f}; max loss ${a['max_loss']:.0f}; POP {a['pop']:.0%}")""",
     """def breakevens(grid, pnl):
    sign = np.sign(pnl)
    return grid[1:][sign[1:] != sign[:-1]].round(2)

gg = np.linspace(0.5 * S, 1.5 * S, 2001)
pnl = ic.payoff_at_first_expiry(gg) - ic.value(S)
mine = p.attempt(breakevens, gg, pnl)
mine = p.check("breakevens", mine, ic.analyze(S)["breakevens"])
a = ic.analyze(S)
print(f"breakevens {mine}; credit ${-a['cost']:.0f}; max profit ${a['max_profit']:.0f}; max loss ${a['max_loss']:.0f}; POP {a['pop']:.0%}")"""),
    ("md", "## 4. A high POP is not an edge\n\n"
           "The condor wins about 70% of the time, but its maximum loss is nearly four times its maximum profit. Simulate many expiries "
           "under the same one-volatility lognormal the POP assumes and look at the average: a 71% win rate and still a **losing** trade on "
           "average, because each loss is worth three wins. Win rate tells you how the P&L is *shaped*, not whether it's positive. Size "
           "short-premium trades by their max loss, not their POP."),
    ("code", """rng = np.random.default_rng(0)
sig = np.mean([L.iv for L in ic.legs])
ST = S * np.exp((0.04 - 0.5 * sig ** 2) * T + sig * np.sqrt(T) * rng.standard_normal(200_000))
sim = ic.payoff_at_first_expiry(ST) - ic.value(S) * np.exp(0.04 * T)
print(f"win rate {np.mean(sim > 0):.0%}, average P&L ${sim.mean():+.1f}, average win ${sim[sim > 0].mean():.0f}, average loss ${sim[sim <= 0].mean():.0f}")"""),
    ("md", "## 5. Defined risk\n\n"
           "The library's templates are **defined-risk** by default: every short option is covered. Rule: net call quantity plus shares must "
           "be `>= 0` (short calls covered by long calls or stock), and net put quantity `>= 0` (short puts covered by long puts)."),
    ("md", YOUR_TURN),
    ("ex", """def is_defined_risk(strat):
    calls = sum(L.qty for L in strat.legs if L.cp == 1)
    puts = sum(L.qty for L in strat.legs if L.cp == -1)
    shares = sum(L.qty for L in strat.legs if L.cp == 0)
    return ...                                    # ✍️

book = {"iron condor": ic,
        "short strangle": p.OptionStrategy("short strangle").add(-1, 560, T, -1, ivf(560)).add(1, 640, T, -1, ivf(640)),
        "covered call": p.OptionStrategy("covered call").add(0, 0, 0, 1).add(1, 630, T, -1, ivf(630)),
        "put ratio spread": p.OptionStrategy("1×2 put ratio").add(-1, 590, T, 1, ivf(590)).add(-1, 570, T, -2, ivf(570)),
        "bull call spread": p.vertical(1, 600, 620, T, ivf)}
mine = {k: p.attempt(is_defined_risk, v) for k, v in book.items()}
mine = p.check("is_defined_risk", mine, {k: v.is_defined_risk() for k, v in book.items()})
mine""",
     """def is_defined_risk(strat):
    calls = sum(L.qty for L in strat.legs if L.cp == 1)
    puts = sum(L.qty for L in strat.legs if L.cp == -1)
    shares = sum(L.qty for L in strat.legs if L.cp == 0)
    return calls + shares >= 0 and puts >= 0

book = {"iron condor": ic,
        "short strangle": p.OptionStrategy("short strangle").add(-1, 560, T, -1, ivf(560)).add(1, 640, T, -1, ivf(640)),
        "covered call": p.OptionStrategy("covered call").add(0, 0, 0, 1).add(1, 630, T, -1, ivf(630)),
        "put ratio spread": p.OptionStrategy("1×2 put ratio").add(-1, 590, T, 1, ivf(590)).add(-1, 570, T, -2, ivf(570)),
        "bull call spread": p.vertical(1, 600, 620, T, ivf)}
mine = {k: p.attempt(is_defined_risk, v) for k, v in book.items()}
mine = p.check("is_defined_risk", mine, {k: v.is_defined_risk() for k, v in book.items()})
mine"""),
    ("md", "## Wrap-up\n\n"
           "* One builder values any structure today, later, at expiry and under vol shifts.\n"
           "* Report breakevens, max profit, max loss and POP, and never size by POP.\n"
           "* Templates default to defined risk.\n"
           "* Graded version: `labs/part07/week24_option_builder` (the builder, templates, IB `BAG` and Alpaca multi-leg orders)."),
]

# ---------------------------------------------------------------------------------------------- 06
NB["06_directional_options"] = [
    header("06", "Directional option strategies and the structure selector", "S6 (Directional & mean-reversion option strategies)",
           "1. Pick strikes by delta.\n"
           "2. Compute IV rank and IV percentile, and see how they differ.\n"
           "3. Price a combo at its mid and at its natural price.\n"
           "4. Compare three ways to be bullish, and map a view to a structure."),
    ("code", SETUP),
    ("code", """S, T = 600.0, 30 / 365
ivf = p.skew_iv(S)
strikes = np.arange(450, 751, 5.0)"""),
    ("md", "## 1. Strikes by delta\n\n"
           "Traders ask for \"the 30-delta call\" or \"the 16-delta put\", not a strike. Price each listed strike with its own IV (`ivf(K)`), "
           "compute its delta with `p.bsm_greeks(S, K, T, r, q, iv, cp)[\"delta\"]`, and return the strike whose delta is closest to the target "
           "(puts have negative deltas)."),
    ("md", YOUR_TURN),
    ("ex", """def strike_for_delta(S, T, r, q, iv_fn, cp, target, strikes):
    K = np.asarray(strikes, dtype=float)
    d = ...                                       # ✍️ the delta of every strike, each with its own IV
    return float(K[np.argmin(np.abs(d - target))])

targets = [(1, 0.50), (1, 0.30), (1, 0.16), (-1, -0.30), (-1, -0.16)]
mine = [p.attempt(strike_for_delta, S, T, 0.04, 0.0, ivf, cp, tg, strikes) for cp, tg in targets]
mine = p.check("strike_for_delta", mine, [p.strike_for_delta(S, T, 0.04, 0.0, ivf, cp, tg, strikes) for cp, tg in targets])
dict(zip(["50Δ call", "30Δ call", "16Δ call", "30Δ put", "16Δ put"], mine))""",
     """def strike_for_delta(S, T, r, q, iv_fn, cp, target, strikes):
    K = np.asarray(strikes, dtype=float)
    d = np.array([p.bsm_greeks(S, k, T, r, q, iv_fn(k), cp)["delta"] for k in K])
    return float(K[np.argmin(np.abs(d - target))])

targets = [(1, 0.50), (1, 0.30), (1, 0.16), (-1, -0.30), (-1, -0.16)]
mine = [p.attempt(strike_for_delta, S, T, 0.04, 0.0, ivf, cp, tg, strikes) for cp, tg in targets]
mine = p.check("strike_for_delta", mine, [p.strike_for_delta(S, T, 0.04, 0.0, ivf, cp, tg, strikes) for cp, tg in targets])
dict(zip(["50Δ call", "30Δ call", "16Δ call", "30Δ put", "16Δ put"], mine))"""),
    ("md", "The 16-delta put is further from spot than the 16-delta call: the skew makes puts expensive, so their deltas are larger at the "
           "same distance.\n\n"
           "## 2. Is implied vol high?\n\n"
           "Over the last `window` values (today included):\n"
           "* **IV rank** = `(today − min) / (max − min) × 100` (50 if flat): where today sits in the range;\n"
           "* **IV percentile** = share (%) of the previous `window − 1` values strictly **below** today: how often it was lower."),
    ("md", YOUR_TURN),
    ("ex", """def iv_rank(hist, window=252):
    x = np.asarray(hist, dtype=float)[-window:]
    return ...                                    # ✍️

def iv_percentile(hist, window=252):
    x = np.asarray(hist, dtype=float)[-window:]
    return ...                                    # ✍️

hist = p.iv_history()
days = [300, 420, 599]
mine = [(p.attempt(iv_rank, hist[: d + 1]), p.attempt(iv_percentile, hist[: d + 1])) for d in days]
mine = p.check("iv rank and percentile", mine, [(p.iv_rank(hist[: d + 1]), p.iv_percentile(hist[: d + 1])) for d in days])
fig, ax = plt.subplots(figsize=(10, 3.2)); ax.plot(hist * 100); [ax.axvline(d, color=p.PALETTE[7], ls="--", lw=1) for d in days]
ax.set_title("30-day implied vol, %"); plt.show()
pd.DataFrame(mine, index=[f"day {d}" for d in days], columns=["IV rank", "IV percentile"]).round(1)""",
     """def iv_rank(hist, window=252):
    x = np.asarray(hist, dtype=float)[-window:]
    lo, hi = x.min(), x.max()
    return 50.0 if hi == lo else float((x[-1] - lo) / (hi - lo) * 100)

def iv_percentile(hist, window=252):
    x = np.asarray(hist, dtype=float)[-window:]
    return float(np.mean(x[:-1] < x[-1]) * 100)

hist = p.iv_history()
days = [300, 420, 599]
mine = [(p.attempt(iv_rank, hist[: d + 1]), p.attempt(iv_percentile, hist[: d + 1])) for d in days]
mine = p.check("iv rank and percentile", mine, [(p.iv_rank(hist[: d + 1]), p.iv_percentile(hist[: d + 1])) for d in days])
fig, ax = plt.subplots(figsize=(10, 3.2)); ax.plot(hist * 100); [ax.axvline(d, color=p.PALETTE[7], ls="--", lw=1) for d in days]
ax.set_title("30-day implied vol, %"); plt.show()
pd.DataFrame(mine, index=[f"day {d}" for d in days], columns=["IV rank", "IV percentile"]).round(1)"""),
    ("md", "One spike in the window drags the rank down for a year; the percentile ignores how big the spike was. Say which one a rule uses.\n\n"
           "## 3. Combo prices: mid vs natural\n\n"
           "A multi-leg order is quoted per share as a net price (positive = you pay a debit). The **mid** is `Σ qty·(bid + ask)/2`. The "
           "**natural** price is what crossing every leg costs: buy at the ask, sell at the bid, `Σ (qty·ask if qty > 0 else qty·bid)`. "
           "`leg_cost = natural − mid` is what legging in would throw away."),
    ("md", YOUR_TURN),
    ("ex", """def combo_prices(qtys, bids, asks):
    q, b, a = (np.asarray(v, dtype=float) for v in (qtys, bids, asks))
    mid = float(np.sum(q * (b + a) / 2))
    natural = ...                                 # ✍️
    return {"mid": mid, "natural": natural, "leg_cost": natural - mid}

condor = dict(qtys=[1, -1, -1, 1], bids=[2.10, 4.05, 3.60, 1.45], asks=[2.25, 4.25, 3.80, 1.60])
spread = dict(qtys=[1, -1], bids=[13.90, 5.30], asks=[14.30, 5.55])
mine = [p.attempt(combo_prices, **condor), p.attempt(combo_prices, **spread)]
mine = p.check("combo_prices", mine, [p.combo_prices(**condor), p.combo_prices(**spread)])
pd.DataFrame(mine, index=["iron condor (credit)", "bull call spread (debit)"]).round(3)""",
     """def combo_prices(qtys, bids, asks):
    q, b, a = (np.asarray(v, dtype=float) for v in (qtys, bids, asks))
    mid = float(np.sum(q * (b + a) / 2))
    natural = float(np.sum(np.where(q > 0, q * a, q * b)))
    return {"mid": mid, "natural": natural, "leg_cost": natural - mid}

condor = dict(qtys=[1, -1, -1, 1], bids=[2.10, 4.05, 3.60, 1.45], asks=[2.25, 4.25, 3.80, 1.60])
spread = dict(qtys=[1, -1], bids=[13.90, 5.30], asks=[14.30, 5.55])
mine = [p.attempt(combo_prices, **condor), p.attempt(combo_prices, **spread)]
mine = p.check("combo_prices", mine, [p.combo_prices(**condor), p.combo_prices(**spread)])
pd.DataFrame(mine, index=["iron condor (credit)", "bull call spread (debit)"]).round(3)"""),
    ("md", "The condor's credit at the mid is $4.15 per share; crossing all four legs gives up $0.35 of it, about 8% of the premium, on every "
           "entry and again on every exit. Send one combo order at (or near) the net mid.\n\n"
           "## 4. Three ways to be bullish\n\n"
           "100 shares, one 600 call, or a 600/620 bull call spread: same direction, very different shapes, costs and Greeks."),
    ("code", """ways = {"100 shares": p.OptionStrategy("stock").add(0, 0, 0, 1),
        "long 600 call": p.OptionStrategy("call").add(1, 600, T, 1, ivf(600)),
        "600/620 bull call spread": p.vertical(1, 600, 620, T, ivf)}
g = np.linspace(540, 660, 300)
fig, ax = plt.subplots()
rows = {}
for k, st in ways.items():
    ax.plot(g, st.value(g, dt=T) - st.value(S), label=k)
    has_options = any(L.cp != 0 for L in st.legs)
    a = st.analyze(S) if has_options else {"cost": float(st.value(S)), "max_loss": -float(st.value(S))}   # stock can go to 0
    gr = st.greeks(S)
    rows[k] = {"cost $": a["cost"], "max loss $": a["max_loss"], "delta (shares)": gr["delta"], "vega $/pt": gr["vega"], "theta $/day": gr["theta"]}
ax.set_ylim(-3000, 3000); ax.axhline(0, color="black", lw=0.6); ax.set(xlabel="price at expiry", ylabel="P&L, $"); ax.legend(); plt.show()
pd.DataFrame(rows).T.round(1)"""),
    ("md", "## 5. From a view to a structure\n\n"
           "`p.select_structure` writes down the classic rules as code: with a directional view, buy options when IV is low, use debit "
           "spreads in the middle, sell credit spreads when IV is rich; with no view, sell a condor when IV is high, a calendar in contango, "
           "else no trade. It is a **hypothesis** to test in Part 8, not a law."),
    ("code", """grid = pd.DataFrame({ivr: {d: p.select_structure(d, ivr, term_slope=0.01) for d in (1, 0, -1)} for ivr in (10, 45, 85)})
grid.index = ["bullish", "no view", "bearish"]; grid.columns = [f"IV rank {c}" for c in grid.columns]
grid"""),
    ("md", "## Wrap-up\n\n"
           "* Strikes by delta, on a smile; IV rank vs percentile, stated explicitly.\n"
           "* Always send multi-leg trades as one combo at the net mid; model the leg cost in first looks.\n"
           "* Graded version: `labs/part07/week24_option_builder` and the Clinic W2 option playbook."),
]

# ---------------------------------------------------------------------------------------------- 07
NB["07_range_event_vol"] = [
    header("07", "Range, event and volatility option strategies", "S7 (Range-bound, either-way & volatility option strategies)",
           "1. Read the market's expected move from the straddle.\n"
           "2. Compare implied and realized moves over past events.\n"
           "3. Separate a research-only volatility risk premium from a tradable signal.\n"
           "4. Look at the worst month of a premium seller before its win rate."),
    ("code", SETUP),
    ("md", "## 1. The implied move\n\n"
           "The ATM straddle's price, as a fraction of spot, is the market's **expected absolute move** to expiry. A quick rule: "
           "`straddle ≈ √(2/π)·S·σ·√T ≈ 0.8·S·σ·√T`."),
    ("md", YOUR_TURN),
    ("ex", """def implied_move(straddle_price, spot):
    return ...                                    # ✍️ as a fraction of spot

def straddle_rule(S, sigma, T):
    return ...                                    # ✍️ use √(2/π) exactly

S, sig, T = 600.0, 0.25, 7 / 365
exact = float(p.bsm_price(S, S, T, 0.0, 0.0, sig, 1) + p.bsm_price(S, S, T, 0.0, 0.0, sig, -1))
mine = [p.attempt(implied_move, exact, S), p.attempt(straddle_rule, S, sig, T)]
mine = p.check("implied move and rule of thumb", mine, [p.implied_move(exact, S), p.straddle_rule_of_thumb(S, sig, T)])
print(f"7-day ATM straddle ${exact:.2f} → implied move ±{mine[0]:.2%}; rule of thumb ${mine[1]:.2f}")""",
     """def implied_move(straddle_price, spot):
    return straddle_price / spot

def straddle_rule(S, sigma, T):
    return float(np.sqrt(2 / np.pi) * S * sigma * np.sqrt(T))

S, sig, T = 600.0, 0.25, 7 / 365
exact = float(p.bsm_price(S, S, T, 0.0, 0.0, sig, 1) + p.bsm_price(S, S, T, 0.0, 0.0, sig, -1))
mine = [p.attempt(implied_move, exact, S), p.attempt(straddle_rule, S, sig, T)]
mine = p.check("implied move and rule of thumb", mine, [p.implied_move(exact, S), p.straddle_rule_of_thumb(S, sig, T)])
print(f"7-day ATM straddle ${exact:.2f} → implied move ±{mine[0]:.2%}; rule of thumb ${mine[1]:.2f}")"""),
    ("md", "## 2. An event study\n\n"
           "Forty past earnings events: the implied move from the straddle bought the day before, and the realized absolute move. "
           "Return `n`, the mean implied and realized moves, `edge = mean(realized − implied)` (a long straddle's rough edge per unit of spot) "
           "and the share of events where realized beat implied."),
    ("md", YOUR_TURN),
    ("ex", """def event_study(implied, realized):
    implied, realized = np.asarray(implied, float), np.asarray(realized, float)
    return {"n": int(implied.size), "mean_implied": float(implied.mean()), "mean_realized": float(realized.mean()),
            "edge": ...,                          # ✍️
            "share_realized_above": ...}          # ✍️

ev = p.earnings_events()
mine = p.attempt(event_study, ev.implied, ev.realized)
ref = {"n": 40, "mean_implied": float(ev.implied.mean()), "mean_realized": float(ev.realized.mean()),
       "edge": float((ev.realized - ev.implied).mean()), "share_realized_above": float((ev.realized > ev.implied).mean())}
mine = p.check("event_study", mine, ref)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(ev.implied * 100, ev.realized * 100); ax.plot([0, 20], [0, 20], color="black", lw=0.8)
ax.set(xlabel="implied move, %", ylabel="realized move, %", title="Above the line: the straddle buyer won"); plt.show()
{k: round(v, 4) for k, v in mine.items()}""",
     """def event_study(implied, realized):
    implied, realized = np.asarray(implied, float), np.asarray(realized, float)
    return {"n": int(implied.size), "mean_implied": float(implied.mean()), "mean_realized": float(realized.mean()),
            "edge": float(np.mean(realized - implied)),
            "share_realized_above": float(np.mean(realized > implied))}

ev = p.earnings_events()
mine = p.attempt(event_study, ev.implied, ev.realized)
ref = {"n": 40, "mean_implied": float(ev.implied.mean()), "mean_realized": float(ev.realized.mean()),
       "edge": float((ev.realized - ev.implied).mean()), "share_realized_above": float((ev.realized > ev.implied).mean())}
mine = p.check("event_study", mine, ref)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(ev.implied * 100, ev.realized * 100); ax.plot([0, 20], [0, 20], color="black", lw=0.8)
ax.set(xlabel="implied move, %", ylabel="realized move, %", title="Above the line: the straddle buyer won"); plt.show()
{k: round(v, 4) for k, v in mine.items()}"""),
    ("md", "Here the straddle seller usually wins, and occasionally loses a lot (the fat tail). Forty events is a small sample: an edge "
           "estimate this noisy needs Part 8's statistics before it becomes a strategy.\n\n"
           "## 3. The volatility risk premium: research vs trading\n\n"
           "Implied vol is usually above the vol that then realizes: the **variance risk premium** that option sellers collect. Measuring it "
           "needs the **future** realized vol (`p.vrp_research`, fine for research). A trading signal can only use **trailing** realized vol, "
           "known at today's close. Write the tradable version: `iv − p.realized_vol(close, window)`."),
    ("code", """close, iv = p.vol_market()
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(iv.index, iv * 100, label="30-day implied"); ax.plot(close.index, p.realized_vol(close) * 100, label="21-day realized (trailing)")
ax.legend(); ax.set_title("Implied vol sits above realized, most of the time"); plt.show()"""),
    ("md", YOUR_TURN),
    ("ex", """def vrp_signal(iv_30d, close, window=21):
    return ...                                    # ✍️ implied today minus TRAILING realized vol

mine = p.attempt(vrp_signal, iv, close)
mine = p.check("vrp_signal", mine, p.vrp_signal(iv, close))

cut = close.index[900]
research_full, research_cut = p.vrp_research(iv, close), p.vrp_research(iv[:cut], close[:cut])
signal_full, signal_cut = p.vrp_signal(iv, close), p.vrp_signal(iv[:cut], close[:cut])
same = lambda a, b: np.allclose(a.loc[:cut].dropna(), b.dropna().reindex(a.loc[:cut].dropna().index), equal_nan=True)
print("truncation test (values up to the cut unchanged when the future is removed):")
print("  research VRP:", same(research_full, research_cut), "  ← uses the future: research only")
print("  tradable VRP:", same(signal_full, signal_cut))""",
     """def vrp_signal(iv_30d, close, window=21):
    return iv_30d - p.realized_vol(close, window)

mine = p.attempt(vrp_signal, iv, close)
mine = p.check("vrp_signal", mine, p.vrp_signal(iv, close))

cut = close.index[900]
research_full, research_cut = p.vrp_research(iv, close), p.vrp_research(iv[:cut], close[:cut])
signal_full, signal_cut = p.vrp_signal(iv, close), p.vrp_signal(iv[:cut], close[:cut])
same = lambda a, b: np.allclose(a.loc[:cut].dropna(), b.dropna().reindex(a.loc[:cut].dropna().index), equal_nan=True)
print("truncation test (values up to the cut unchanged when the future is removed):")
print("  research VRP:", same(research_full, research_cut), "  ← uses the future: research only")
print("  tradable VRP:", same(signal_full, signal_cut))"""),
    ("md", "## 4. The premium seller's worst month\n\n"
           "Sell a 30-day ATM straddle at the start of every month at the implied vol, hold to expiry. Look at the whole distribution, "
           "worst month first, before the win rate."),
    ("code", """S_, V_ = close.to_numpy(), iv.to_numpy()
pnl = []
for t in range(0, len(S_) - 21, 21):
    prem = p.bsm_price(S_[t], S_[t], 21 / 252, 0.0, 0.0, V_[t], 1) + p.bsm_price(S_[t], S_[t], 21 / 252, 0.0, 0.0, V_[t], -1)
    pnl.append((prem - abs(S_[t + 21] - S_[t])) / S_[t])
pnl = np.array(pnl)
fig, ax = plt.subplots()
ax.hist(pnl * 100, bins=30); ax.axvline(0, color="black", lw=0.8)
ax.set(xlabel="P&L per month, % of spot", title="Short ATM straddle, held to expiry"); plt.show()
print(f"win rate {np.mean(pnl > 0):.0%}; mean {pnl.mean():+.2%}; best month {pnl.max():+.2%}; worst month {pnl.min():+.2%}")"""),
    ("md", "## Wrap-up\n\n"
           "* The straddle prices the expected move; compare it with what events actually delivered.\n"
           "* Research features may use the future; trading signals may not. Truncation tests keep them apart.\n"
           "* Premium selling has a positive average and a brutal left tail: show the worst week first, size by it.\n"
           "* Graded version: `labs/part07/week24_vol_hedging`."),
]

# ---------------------------------------------------------------------------------------------- 08
NB["08_hedging"] = [
    header("08", "Hedging a portfolio", "S8 (Hedging strategies & M3b release)",
           "1. Size an index-futures hedge with beta.\n"
           "2. Find the call strike that makes a collar cost nothing.\n"
           "3. Compare no hedge, puts, a collar and futures through a crash.\n"
           "4. Judge a hedge by the portfolio's drawdown and its cost, not by its own P&L."),
    ("code", SETUP),
    ("md", "## 1. A futures hedge\n\n"
           "To hedge a portfolio with index futures, sell `round(hedge_ratio × β × value / (futures price × multiplier))` contracts. "
           "β is the portfolio's sensitivity to the index; the multiplier is 50 for ES, 5 for MES."),
    ("md", YOUR_TURN),
    ("ex", """def futures_hedge_contracts(value, beta, fut_price, multiplier, hedge_ratio=1.0):
    return ...                                    # ✍️ an int

cases = [(2_000_000, 1.2, 5000.0, 50, 1.0), (2_000_000, 1.2, 5000.0, 5, 0.5), (350_000, 0.8, 5000.0, 5, 1.0)]
mine = [p.attempt(futures_hedge_contracts, *c) for c in cases]
mine = p.check("futures_hedge_contracts", mine, [p.futures_hedge_contracts(*c) for c in cases])
dict(zip(["$2m, β 1.2, full, ES", "$2m, β 1.2, half, MES", "$350k, β 0.8, full, MES"], mine))""",
     """def futures_hedge_contracts(value, beta, fut_price, multiplier, hedge_ratio=1.0):
    return int(round(hedge_ratio * beta * value / (fut_price * multiplier)))

cases = [(2_000_000, 1.2, 5000.0, 50, 1.0), (2_000_000, 1.2, 5000.0, 5, 0.5), (350_000, 0.8, 5000.0, 5, 1.0)]
mine = [p.attempt(futures_hedge_contracts, *c) for c in cases]
mine = p.check("futures_hedge_contracts", mine, [p.futures_hedge_contracts(*c) for c in cases])
dict(zip(["$2m, β 1.2, full, ES", "$2m, β 1.2, half, MES", "$350k, β 0.8, full, MES"], mine))"""),
    ("md", "A small account can't hedge precisely with ES (one contract is $250k of index); micro contracts make the rounding error tolerable.\n\n"
           "## 2. A zero-cost collar\n\n"
           "Buy a protective put at `k_put` and pay for it by selling a call: find the call strike **above spot** whose premium equals the put's. "
           "Price each strike with its own IV and solve with `brentq` on `[S, 3S]`."),
    ("md", YOUR_TURN),
    ("ex", """from scipy.optimize import brentq

def zero_cost_call_strike(S, T, k_put, iv_fn, r=0.04, q=0.0):
    put = p.bsm_price(S, k_put, T, r, q, iv_fn(k_put), -1)
    f = ...                                       # ✍️ a function of k: the call premium at k minus the put premium
    return float(brentq(f, S, 3 * S, xtol=1e-8))

S = 600.0
cases = [(S, 0.25, 0.90 * S, p.skew_iv(S)), (S, 0.25, 0.95 * S, p.skew_iv(S)), (S, 0.25, 0.90 * S, lambda k: 0.18)]
mine = [p.attempt(zero_cost_call_strike, *c) for c in cases]
mine = p.check("zero_cost_call_strike", mine, [p.zero_cost_call_strike(*c) for c in cases])
pd.DataFrame({"put strike": [c[2] for c in cases], "smile": ["skewed", "skewed", "flat 18%"], "zero-cost call strike": mine}).round(2)""",
     """from scipy.optimize import brentq

def zero_cost_call_strike(S, T, k_put, iv_fn, r=0.04, q=0.0):
    put = p.bsm_price(S, k_put, T, r, q, iv_fn(k_put), -1)
    f = lambda k: p.bsm_price(S, k, T, r, q, iv_fn(k), 1) - put
    return float(brentq(f, S, 3 * S, xtol=1e-8))

S = 600.0
cases = [(S, 0.25, 0.90 * S, p.skew_iv(S)), (S, 0.25, 0.95 * S, p.skew_iv(S)), (S, 0.25, 0.90 * S, lambda k: 0.18)]
mine = [p.attempt(zero_cost_call_strike, *c) for c in cases]
mine = p.check("zero_cost_call_strike", mine, [p.zero_cost_call_strike(*c) for c in cases])
pd.DataFrame({"put strike": [c[2] for c in cases], "smile": ["skewed", "skewed", "flat 18%"], "zero-cost call strike": mine}).round(2)"""),
    ("md", "With a skewed smile the put is expensive and the call cheap, so the call you must sell is much closer to spot: the skew is the "
           "price of protection.\n\n"
           "## 3. Through a crash\n\n"
           "One million in the index: two calm years, a 30% crash in three weeks, then a recovery. Four choices: no hedge; rolling 10% "
           "out-of-the-money 3-month puts every month; the same puts with zero-cost calls (a collar); and shorting half the position in futures."),
    ("code", """close, iv = p.crash_path()
curves = {m: p.hedge_study(close, iv, m) for m in ("none", "puts", "collar", "futures")}
fig, ax = plt.subplots(figsize=(11, 4))
for m, eq in curves.items():
    ax.plot(eq.index, eq / 1e6, label=m)
ax.set(ylabel="equity, $m", title="Hedges through a crash"); ax.legend(); plt.show()"""),
    ("md", "Judge each hedge by the **portfolio**: its total return, its maximum drawdown, and the difference in total return versus no "
           "hedge (`cost_vs_none`, negative when the hedge cost more than it gave back). Compute the maximum drawdown of an equity curve: "
           "the most negative `equity / running maximum − 1`."),
    ("md", YOUR_TURN),
    ("ex", """def hedge_report(curves):
    rows = {}
    for m, eq in curves.items():
        rows[m] = {"total_return": eq.iloc[-1] / eq.iloc[0] - 1,
                   "max_drawdown": ...}           # ✍️
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["cost_vs_none"] = df["total_return"] - df.loc["none", "total_return"]
    return df

mine = p.attempt(hedge_report, curves)
mine = p.check("hedge_report (through the crash)", mine, p.hedge_report(curves))
calm = {m: eq.iloc[:500] for m, eq in curves.items()}
display(mine.round(3))
print("the calm years only (the insurance premium, with nothing to show for it):")
p.hedge_report(calm).round(3)""",
     """def hedge_report(curves):
    rows = {}
    for m, eq in curves.items():
        rows[m] = {"total_return": eq.iloc[-1] / eq.iloc[0] - 1,
                   "max_drawdown": (eq / eq.cummax() - 1).min()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["cost_vs_none"] = df["total_return"] - df.loc["none", "total_return"]
    return df

mine = p.attempt(hedge_report, curves)
mine = p.check("hedge_report (through the crash)", mine, p.hedge_report(curves))
calm = {m: eq.iloc[:500] for m, eq in curves.items()}
display(mine.round(3))
print("the calm years only (the insurance premium, with nothing to show for it):")
p.hedge_report(calm).round(3)"""),
    ("md", "In the calm years the puts and the futures cost money (the collar roughly breaks even, having given up upside a calm market "
           "didn't deliver), and a hedge judged by its own P&L gets cancelled just before it is needed "
           "(common mistake #12). Through the crash each one cuts the drawdown, at different prices: puts keep the upside, the collar gives "
           "it up to finance the puts, and futures remove return in both directions.\n\n"
           "## Wrap-up\n\n"
           "* Size futures hedges with β and the right contract size.\n"
           "* Collars trade upside for protection at a price set by the skew.\n"
           "* Evaluate hedges at the portfolio level: drawdown reduction per unit of cost over a full cycle.\n"
           "* Graded version: `labs/part07/week24_vol_hedging` (futures sizing, zero-cost collar, the hedging study)."),
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
