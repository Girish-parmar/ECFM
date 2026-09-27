"""Build the Part 10 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part10:  python tools/build_notebooks.py
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
for d in (Path.cwd(), Path.cwd().parent):       # p10lib.py is in notebooks/part10/
    sys.path.insert(0, str(d))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p10lib as p

p.use_course_style()"""

SETUP_DL = SETUP + """
import torch
import p10dl as dl"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 10 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_10_MACHINE_LEARNING.md) · graded labs in [`labs/part10/`](../../labs/part10/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.
All data is synthetic with a known structure (a hidden regime, a known autocorrelation, planted importances, pure noise), so you can see what a model can learn, and that it learns nothing from noise.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
_DIR_CHECK = """
rw = p.random_walk_prices(2500)                         # pure noise
mk = p.regime_market()                                  # a hidden trend/chop regime
mine = [p.attempt(direction_dataset, rw), p.attempt(direction_dataset, mk.close)]
mine = p.check("direction_dataset", mine, [p.direction_dataset(rw), p.direction_dataset(mk.close)])
X, y, fwd = mine[0]
X.head(3)"""

_DB_CHECK = """
ticks = p.tick_data()                                   # 30 days of trades with quiet and busy days
tbars = p.time_bars(ticks, "30min")
threshold = (ticks["price"] * ticks["size"]).sum() / len(tbars)      # the same number of bars
dbars = p.check("dollar_bars", p.attempt(dollar_bars, ticks, threshold), p.dollar_bars(ticks, threshold))
pd.DataFrame({"time bars (30 min)": p.bar_stats(tbars["close"]), "dollar bars": p.bar_stats(dbars["close"])}).T.round(3)"""

_CUSUM_HEAD = """def cusum_events(close, h):
    events, s_pos, s_neg = [], 0.0, 0.0
    for ts, x in np.log(close).diff().dropna().items():
"""
_CUSUM_TAIL = """        if s_pos > h:
            s_pos = 0.0
            events.append(ts)
        elif s_neg < -h:
            s_neg = 0.0
            events.append(ts)
    return pd.DatetimeIndex(events)

cases = [(mk.close, 0.015), (mk.close, 0.04)]
mine = [p.attempt(cusum_events, *c) for c in cases]
mine = p.check("cusum_events", mine, [p.cusum_events(*c) for c in cases])
print(f"{len(mk)} days → {len(mine[0])} events with h = 1.5%, {len(mine[1])} with h = 4%")
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(mk["close"].iloc[:500], lw=1)
ev = mine[1][mine[1] <= mk.index[499]]
ax.scatter(ev, mk["close"].loc[ev], color="#eb6834", s=18, zorder=3, label="CUSUM event (h = 4%)")
ax.legend(); ax.set_title("events fire when the price has moved enough, not every day"); plt.show()"""

NB["01_baseline_bars"] = [
    header("01", "A baseline first, then better bars and events", "S1–S2 (Framing financial ML · Data structures & event sampling)",
           "1. Build the simplest ML dataset (lagged returns → tomorrow's direction) and a walk-forward baseline.\n"
           "2. See why a model must beat the base rate, not 50%.\n"
           "3. Build dollar bars and compare their statistics with time bars.\n"
           "4. Sample events with a CUSUM filter instead of using every day."),
    ("code", SETUP),
    ("md", "## 1. The baseline\n\n"
           "Before any fancy model, set the bar: the last 5 log returns as features, tomorrow's direction as the label. Features at `t` "
           "are `lag0 = r_t, lag1 = r_{t−1}, …` (`r.shift(k)`); the target is 1 if `r_{t+1} > 0`; also keep `fwd = r_{t+1}` for the "
           "P&L. Drop rows with any NaN (the start, and the last row which has no tomorrow)."),
    ("md", YOUR_TURN),
    ("ex", """def direction_dataset(close, n_lags=5):
    r = np.log(close).diff()
    X = pd.DataFrame({f"lag{k}": ... for k in range(n_lags)})       # ✍️
    fwd = ...                                                        # ✍️ tomorrow's return, aligned to today
    ok = X.notna().all(axis=1) & fwd.notna()
    return X[ok], (fwd[ok] > 0).astype(int), fwd[ok]
""" + _DIR_CHECK,
     """def direction_dataset(close, n_lags=5):
    r = np.log(close).diff()
    X = pd.DataFrame({f"lag{k}": r.shift(k) for k in range(n_lags)})
    fwd = r.shift(-1)
    ok = X.notna().all(axis=1) & fwd.notna()
    return X[ok], (fwd[ok] > 0).astype(int), fwd[ok]
""" + _DIR_CHECK),
    ("md", "Now a walk-forward logistic regression: every 21 days, refit on everything before and trade the next 21 days "
           "(+1 if `p(up) > 0.5`, else −1), paying 5 bp per unit of position change (`p.walk_forward_logit`)."),
    ("code", """rows = {}
for name, close in (("random walk", rw), ("regime market", mk["close"])):
    wf = p.walk_forward_logit(*p.direction_dataset(close))
    rows[name] = {"accuracy": wf["accuracy"], "share of up days": wf["always_up"], "net Sharpe": p.sharpe(wf["returns"])}
pd.DataFrame(rows).T.round(3)"""),
    ("md", "On the random walk, the accuracy is exactly one minus the share of up days: the model learned to say \"down\" every day, the "
           "base rate of its training data. Its positive Sharpe is the luck of a random walk that happened to fall. On the regime "
           "market, the accuracy is *below* \"always up\". A 51% accuracy means nothing until it beats the base rate, a simple rule, "
           "and costs, out of sample. That is the rule for all of Part 10.\n\n"
           "## 2. Bars that follow activity\n\n"
           "Time bars sample a quiet lunch hour as often as a busy open, so their returns mix small and large variances: fat tails. "
           "**Dollar bars** close each time another fixed amount of money has traded: bar id = cumulative `price × size` `//` "
           "threshold. Group the ticks by bar id; each bar is labeled with the time of its **last** trade (a bar is known only when "
           "it closes)."),
    ("md", YOUR_TURN),
    ("ex", """def dollar_bars(ticks, threshold):
    dv = (ticks["price"] * ticks["size"]).cumsum()
    bar_id = ...                                                     # ✍️ integer bar id per trade
    g = ticks.groupby(bar_id)
    bars = g["price"].agg(open="first", high="max", low="min", close="last")
    bars["volume"] = g["size"].sum()
    bars.index = pd.DatetimeIndex(ticks.index.to_series().groupby(bar_id).last().to_numpy())
    return bars
""" + _DB_CHECK,
     """def dollar_bars(ticks, threshold):
    dv = (ticks["price"] * ticks["size"]).cumsum()
    bar_id = (dv // threshold).astype(int).to_numpy()
    g = ticks.groupby(bar_id)
    bars = g["price"].agg(open="first", high="max", low="min", close="last")
    bars["volume"] = g["size"].sum()
    bars.index = pd.DatetimeIndex(ticks.index.to_series().groupby(bar_id).last().to_numpy())
    return bars
""" + _DB_CHECK),
    ("code", """per_day = pd.DataFrame({"time bars": tbars.groupby(tbars.index.date).size(),
                        "dollar bars": dbars.groupby(dbars.index.date).size()})
per_day.plot(kind="bar", figsize=(11, 3), width=0.8, title="bars per day: a fixed clock vs the market's activity")
plt.xticks([]); plt.show()"""),
    ("md", "Same number of bars, very different samples: the dollar bars' returns are much closer to normal (a Jarque–Bera "
           "statistic near 1 instead of 16), because each bar carries about the same amount of trading. Busy days get more bars, "
           "quiet days fewer.\n\n"
           "## 3. Sample events, not days\n\n"
           "Most days carry no new information. The symmetric **CUSUM filter** accumulates log returns `x` in two running sums, "
           "`s+ = max(0, s+ + x)` and `s− = min(0, s− + x)`, and records an event (and resets that sum) when `s+ > h` or `s− < −h`. "
           "Labels are then built only at events."),
    ("md", YOUR_TURN),
    ("ex", _CUSUM_HEAD + """        s_pos, s_neg = ..., ...                                      # ✍️ update both sums
""" + _CUSUM_TAIL,
     _CUSUM_HEAD + """        s_pos, s_neg = max(0.0, s_pos + x), min(0.0, s_neg + x)
""" + _CUSUM_TAIL),
    ("md", "## Wrap-up\n\n"
           "* Baseline first: every model must beat the base rate and a simple rule, after costs, out of sample.\n"
           "* Dollar (or volume) bars sample by activity and give better-behaved returns.\n"
           "* CUSUM events keep the informative moments; fewer, less overlapping labels.\n"
           "* Graded version: `labs/part10/week33_features` (`walk_forward_logit`, `dollar_bars`, `cusum_events`)."),
]

# ---------------------------------------------------------------------------------------------- 02
_FFD_HEAD = """def ffd_weights(d, thresh=1e-4):
    w, k = [1.0], 1
    while True:
"""
_FFD_TAIL = """        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])                                         # oldest first

cases = [(0.4,), (0.9,), (1.0,), (0.3, 1e-3)]
mine = [p.attempt(ffd_weights, *c) for c in cases]
mine = p.check("ffd_weights", mine, [p.ffd_weights(*c) for c in cases])
print(f"d = 0.4: {len(mine[0])} weights, the newest four {np.round(mine[0][-4:], 3)};  d = 1: {mine[2]} (a plain difference)")"""

_ROLL_CHECK = """
rng = np.random.default_rng(1)
mid = pd.Series(50 + np.cumsum(rng.normal(0, 0.05, 3000)))           # the efficient price
trades = mid + 0.10 / 2 * rng.choice([-1, 1], 3000)                  # trades at the bid or the ask: spread 0.10
cases = [(trades, 250), (mid, 250), (trades, 20)]
mine = [p.attempt(roll_spread, *c) for c in cases]
mine = p.check("roll_spread", mine, [p.roll_spread(*c) for c in cases])
print(f"true spread 0.10 → Roll estimate {mine[0].mean():.4f} on trade prices, {mine[1].mean():.4f} on the mid (no bounce)")"""

_FS_HEAD = """class MyFeatureStore(p.FeatureStore):
    def get(self, symbols, features, as_of):
"""
_FS_TAIL = """        last = known.sort_values("ts").groupby(["symbol", "feature"])["value"].last()
        out = pd.DataFrame(np.nan, index=symbols, columns=features)
        for (s, f), v in last.items():
            if s in out.index and f in out.columns:
                out.loc[s, f] = v
        return out

quarters = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"])
eps = pd.Series([1.00, 1.10, 1.20, 1.30], index=quarters)
stores = []
for cls in (MyFeatureStore, p.FeatureStore):
    fs = cls()
    fs.put("AAA", "eps", eps, delay=pd.Timedelta(days=45))                   # published 45 days after quarter end
    fs.put("AAA", "eps", pd.Series([1.25], index=quarters[2:3]), delay=pd.Timedelta(days=100))   # a later revision
    fs.put("AAA", "price", pd.Series([50.0, 52.0], index=pd.to_datetime(["2023-11-01", "2024-01-02"])))
    stores.append(fs)
dates = ["2023-04-30", "2023-08-01", "2023-11-20", "2024-01-15", "2024-02-20"]
mine = {d: p.attempt(stores[0].get, ["AAA", "BBB"], ["eps", "price"], d) for d in dates}
mine = p.check("FeatureStore.get", mine, {d: stores[1].get(["AAA", "BBB"], ["eps", "price"], d) for d in dates})
pd.DataFrame({d: mine[d].loc["AAA"] for d in dates}).T.rename_axis("as of")"""

NB["02_fracdiff_feature_store"] = [
    header("02", "Fractional differentiation, microstructure features and a point-in-time store",
           "S3–S4 (Features I: fractional differentiation & microstructure · Features II & the feature store)",
           "1. Compute fractional-differencing weights and find the smallest d that makes a price series stationary.\n"
           "2. Recover a known bid–ask spread from trade prices with the Roll estimator.\n"
           "3. Query a point-in-time feature store that handles publication delays and revisions.\n"
           "4. Catch a look-ahead feature with the truncation test."),
    ("code", SETUP),
    ("md", "## 1. Stationary, but with memory\n\n"
           "Prices remember everything (non-stationary); returns (`d = 1`) remember nothing. **Fractional differencing** with `0 < d < 1` "
           "keeps some memory and can still be stationary. The weights: `w_0 = 1`, `w_k = −w_{k−1}·(d − k + 1)/k`, stopping at the "
           "first `|w_k| < thresh` (a fixed-width window). The function returns them **oldest first**, so that `w @ x[t−L+1..t]` is "
           "the transformed value at `t`."),
    ("md", YOUR_TURN),
    ("ex", _FFD_HEAD + """        w_k = ...                                                    # ✍️
""" + _FFD_TAIL,
     _FFD_HEAD + """        w_k = -w[-1] * (d - k + 1) / k
""" + _FFD_TAIL),
    ("md", "`p.min_ffd` tries `d = 0, 0.1, …, 1` on the log price of the regime market and reports the ADF p-value of the "
           "transformed series and its correlation with the original:"),
    ("code", """mk = p.regime_market()
d_star, table = p.min_ffd(np.log(mk["close"]))
display(table.round(3).T)
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(p.frac_diff(np.log(mk["close"]), d_star), lw=0.8, label=f"d = {d_star}")
ax.plot(np.log(mk["close"]).diff(), lw=0.5, alpha=0.6, label="d = 1 (returns)")
ax.legend(); ax.set_title("fractionally differenced log price vs returns"); plt.show()"""),
    ("md", "The smallest stationary `d` (ADF p < 0.05) is well below 1, and that series still has a correlation of about 0.97 with "
           "the price level. Returns keep almost none (0.05): they throw away the memory a model could use.\n\n"
           "## 2. The Roll spread\n\n"
           "Trades bounce between the bid and the ask, which makes consecutive price changes negatively correlated. Roll (1984): "
           "`spread = 2·√(−cov(Δp_t, Δp_{t−1}))` over a rolling window, 0 where the covariance is positive. Use "
           "`dp.rolling(window).cov(dp.shift(1))`."),
    ("md", YOUR_TURN),
    ("ex", """def roll_spread(close, window=20):
    dp = close.diff()
    c = ...                                                          # ✍️ rolling autocovariance of price changes
    return 2 * np.sqrt((-c).clip(lower=0))
""" + _ROLL_CHECK,
     """def roll_spread(close, window=20):
    dp = close.diff()
    c = dp.rolling(window).cov(dp.shift(1))
    return 2 * np.sqrt((-c).clip(lower=0))
""" + _ROLL_CHECK),
    ("md", "The estimator recovers the planted spread from trade prices alone. On the mid price there is no bounce, and what is "
           "left is estimation noise.\n\n"
           "## 3. What was known, and when\n\n"
           "A quarterly EPS number is published about 45 days after the quarter ends, and may be revised later. A backtest that "
           "uses it on the quarter-end date has looked into the future. The store keeps every value with `ts` (what it describes) "
           "and `available_at` (when it became known). `get(symbols, features, as_of)` must use only rows with "
           "`available_at <= as_of`, and among those take the latest `ts`."),
    ("md", YOUR_TURN),
    ("ex", _FS_HEAD + """        known = ...                                                  # ✍️ the rows already available at as_of
""" + _FS_TAIL,
     _FS_HEAD + """        known = self.rows[self.rows["available_at"] <= pd.Timestamp(as_of)]
""" + _FS_TAIL),
    ("md", "On 1 August the Q2 number (quarter end 30 June) is not out yet, so the store still returns Q1. Now look at 20 November: "
           "Q3's 1.20 was published on 14 November, yet the store answers 1.10 (Q2). The revision to 1.25 *replaced* the original "
           "row, so the store forgot the 1.20 that everyone could see until the revision arrived in January. This simple store keeps "
           "one vintage per date; a real point-in-time store keeps every vintage (a bitemporal table), or backtests silently use "
           "numbers that did not exist yet, or lose the ones that did.\n\n"
           "## 4. The truncation test\n\n"
           "A feature has no look-ahead if its value at `t` is the same whether it is computed on the whole history or on the data "
           "up to `t` only (`p.truncation_test`). A **centered** rolling mean fails; a trailing mean and fractional differencing pass:"),
    ("code", """lp = np.log(mk["close"])
points = list(range(300, 2400, 150))
features = {"trailing 20-day mean": lambda s: s.rolling(20).mean(),
            "centered 20-day mean": lambda s: s.rolling(20, center=True).mean(),
            "frac_diff d = 0.4": lambda s: p.frac_diff(s, 0.4)}
pd.Series({k: len(p.truncation_test(f, lp, points)) for k, f in features.items()}, name=f"failures out of {len(points)}").to_frame()"""),
    ("md", "## Wrap-up\n\n"
           "* Fractional differencing: stationarity with most of the memory; choose the smallest `d` that passes ADF.\n"
           "* Microstructure features (Roll spread, Amihud illiquidity) recover real trading frictions.\n"
           "* Store every value with the time it became known, and keep every revision; run the truncation test on every feature.\n"
           "* Graded version: `labs/part10/week33_features` (`frac_diff`, `min_ffd`, `roll_spread`, `amihud`, `FeatureStore`, "
           "`truncation_test`)."),
]

# ---------------------------------------------------------------------------------------------- 03
_TB_HEAD = """def triple_barrier(close, events, vol, pt=1.0, sl=1.0, max_hold=10):
    out, idx = [], close.index
    for t0 in events:
        i0 = idx.get_loc(t0)
        if i0 + 1 >= len(idx) or np.isnan(vol.loc[t0]):
            continue
        path = close.iloc[i0 + 1: i0 + 1 + max_hold] / close.iloc[i0] - 1
        up, dn = pt * vol.loc[t0], -sl * vol.loc[t0]
        hit_up, hit_dn = path[path >= up].index.min(), path[path <= dn].index.min()
"""
_TB_TAIL = """        out.append((t0, t1, path.loc[t1], label))
    return pd.DataFrame(out, columns=["t0", "t1", "ret", "label"]).set_index("t0")

mk = p.regime_market()
close = mk["close"]
events = p.cusum_events(close, 0.015)
events = events[events >= close.index[100]]
vol = p.daily_vol(close)
cases = [(close, events, vol), (close, events[:200], vol, 2.0, 1.0, 5)]
mine = [p.attempt(triple_barrier, *c) for c in cases]
mine = p.check("triple_barrier", mine, [p.triple_barrier(*c) for c in cases])
tb = mine[0]
print(f"{len(tb)} labels: {tb['label'].value_counts().to_dict()}  (+1 upper, −1 lower, 0 time)")"""

_UNIQ_CHECK = """
t1 = tb["t1"]
mine = p.check("avg_uniqueness", p.attempt(avg_uniqueness, t1, close.index), p.avg_uniqueness(t1, close.index))
every_day = p.triple_barrier(close, close.index[100:-11], vol)
print(f"CUSUM events: mean uniqueness {mine.mean():.2f};  a label on every day: "
      f"{p.avg_uniqueness(every_day['t1'], close.index).mean():.2f} ({len(every_day)} heavily overlapping labels)")"""

_BET_CHECK = """
probs = np.array([0.3, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99])
mine = p.check("bet_size", p.attempt(bet_size, probs), p.bet_size(probs))
pd.Series(mine, index=probs, name="size").round(3).to_frame().T"""

NB["03_labels_meta"] = [
    header("03", "Triple-barrier labels, uniqueness and meta-labels", "S5–S6 (Triple-barrier labels & sample weights · Meta-labeling & bet sizing)",
           "1. Label events with profit-taking, stop-loss and time barriers scaled by volatility.\n"
           "2. Measure how much labels overlap, and weight them by uniqueness.\n"
           "3. Turn a primary rule into meta-labels: \"should I take this trade?\"\n"
           "4. Size bets from a probability."),
    ("code", SETUP),
    ("md", "## 1. Three barriers\n\n"
           "At each event `t0`, follow the price over the next `max_hold` bars: `path = close / close[t0] − 1`. The upper barrier is "
           "`pt × vol[t0]`, the lower `−sl × vol[t0]` (volatility known at `t0`). The label ends at `t1`, the **first** bar that "
           "touches either barrier, or the last bar of the path if neither is touched. Label `+1` upper, `−1` lower, `0` time. "
           "`hit_up` and `hit_dn` are the first touching dates (NaT if none): take the earliest one that exists."),
    ("md", YOUR_TURN),
    ("ex", _TB_HEAD + """        t1 = ...                                                     # ✍️ the earliest touch, else the last bar
        label = ...                                                  # ✍️ +1, −1 or 0
""" + _TB_TAIL,
     _TB_HEAD + """        t1 = min([t for t in (hit_up, hit_dn) if pd.notna(t)], default=path.index[-1])
        label = 1 if t1 == hit_up else -1 if t1 == hit_dn else 0
""" + _TB_TAIL),
    ("code", """t0 = tb.index[40]
i0 = close.index.get_loc(t0)
seg = close.iloc[i0: i0 + 12] / close.loc[t0] - 1
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(seg.index, seg, marker="o", lw=1)
ax.axhline(vol.loc[t0], color="#1baf7a", ls="--", label="profit-taking"); ax.axhline(-vol.loc[t0], color="#e34948", ls="--", label="stop-loss")
ax.axvline(close.index[i0 + 10], color="#8a8984", ls=":", label="time barrier"); ax.axvline(tb.loc[t0, "t1"], color="black", lw=1, label="t1")
ax.legend(fontsize=8); ax.set_title(f"one event: label {tb.loc[t0, 'label']:+d}"); plt.show()"""),
    ("md", "## 2. Overlapping labels are not independent\n\n"
           "Two labels that span the same days share information. For each bar, count how many labels are active "
           "(`p.num_concurrent`). A label's **average uniqueness** is the mean of `1 / concurrency` over its own bars `[t0, t1]`: "
           "1 if it overlaps nothing, 0.5 if it always shares its days with one other label. Use it as a sample weight."),
    ("md", YOUR_TURN),
    ("ex", """def avg_uniqueness(t1, index):
    c = p.num_concurrent(t1, index)
    return pd.Series({t0: ... for t0, end in t1.items()})           # ✍️
""" + _UNIQ_CHECK,
     """def avg_uniqueness(t1, index):
    c = p.num_concurrent(t1, index)
    return pd.Series({t0: float((1 / c.loc[t0:end]).mean()) for t0, end in t1.items()})
""" + _UNIQ_CHECK),
    ("md", "Labeling every day makes each label mostly a copy of its neighbours; CUSUM events are much more independent. A standard "
           "bootstrap of overlapping labels draws redundant samples; the **sequential bootstrap** favours labels that add new days:"),
    ("code", """sub = tb["t1"].iloc[:60]
idx = close.loc[sub.index[0]:sub.max()].index

def mean_uniqueness(draws):                           # uniqueness of the drawn sample, repeats included
    spans = [(sub.index[j], sub.iloc[j]) for j in draws]
    c = pd.Series(0, index=idx)
    for s, e in spans:
        c.loc[s:e] += 1
    return np.mean([(1 / c.loc[s:e]).mean() for s, e in spans])

seq = p.sequential_bootstrap(sub, idx, seed=0)
std = list(np.random.default_rng(0).integers(0, len(sub), len(sub)))
print(f"mean uniqueness of the drawn sample: sequential bootstrap {mean_uniqueness(seq):.2f}, standard bootstrap {mean_uniqueness(std):.2f}")"""),
    ("md", "## 3. Meta-labeling\n\n"
           "Keep a simple **primary** rule for the side (here 20-day momentum) and ask ML a narrower question: *given this signal, "
           "will the trade make money?* The meta-label is 1 when `side × barrier return > 0`. In the regime market, momentum should "
           "work in the trend regime and fail in chop. Does it?"),
    ("code", """side = p.momentum_side(close).loc[tb.index]
meta = p.meta_labels(side, tb["ret"])
regime = mk["regime"].loc[tb.index].map({0: "chop (hidden)", 1: "trend (hidden)"})
meta.groupby(regime).agg(["mean", "size"]).rename(columns={"mean": "win rate", "size": "events"}).round(3)"""),
    ("md", "Barely above 50% in trend and below it in chop: a meta-model that could tell the regimes apart would skip the chop "
           "trades. It does not have to be right often, only to size down when it is unsure. Turn a probability `p` into a size "
           "with `z = (p − 1/n)/√(p(1 − p))` (`n = 2` classes) and `size = clip(2Φ(z) − 1, 0, 1)` (`norm.cdf` from scipy)."),
    ("md", YOUR_TURN),
    ("ex", """from scipy.stats import norm

def bet_size(prob, n_classes=2):
    prob = np.asarray(prob, dtype=float)
    z = ...                                                          # ✍️
    return ...                                                       # ✍️ clipped to [0, 1]
""" + _BET_CHECK,
     """from scipy.stats import norm

def bet_size(prob, n_classes=2):
    prob = np.asarray(prob, dtype=float)
    z = (prob - 1 / n_classes) / np.sqrt(prob * (1 - prob))
    return np.clip(2 * norm.cdf(z) - 1, 0, 1)
""" + _BET_CHECK),
    ("md", "Below 50% the size is zero (skip the trade); it grows slowly at first, so a barely-confident model trades small. "
           "Notebook 07 runs the full walk-forward meta-labeled strategy.\n\n"
           "## Wrap-up\n\n"
           "* Label with barriers scaled by volatility, and record when each label **ends** (`t1`): purging needs it.\n"
           "* Overlapping labels are not independent: weight by uniqueness, bootstrap sequentially.\n"
           "* Meta-labeling: a simple rule picks the side, ML decides whether and how much.\n"
           "* Graded version: `labs/part10/week34_labels` (`triple_barrier`, `avg_uniqueness`, `sequential_bootstrap`, "
           "`meta_labels`, `bet_size`)."),
]

# ---------------------------------------------------------------------------------------------- 04
_REL_CHECK = """
cases = [(prob_raw, y_test), (prob_cal, y_test), ([0.05, 0.15, 0.35, 0.95, 1.0], [0, 1, 0, 1, 1])]
mine = [p.attempt(reliability_table, *c) for c in cases]
mine = p.check("reliability_table", mine, [p.reliability_table(*c) for c in cases])
print(f"base rate {y_test.mean():.3f}; mean predicted probability: raw {prob_raw.mean():.3f}, calibrated {prob_cal.mean():.3f};  "
      f"AUC raw {roc_auc_score(y_test, prob_raw):.3f}, calibrated {roc_auc_score(y_test, prob_cal):.3f}")
display(pd.concat({"raw (balanced weights)": mine[0], "calibrated (isotonic)": mine[1]}, axis=1).round(3))"""

_HMMF_HEAD = """def hmm_filter(x, params):
    x = np.asarray(x, dtype=float)
    B = p._normal_pdf(x, params["mu"], params["var"]) + 1e-300      # likelihood of each observation in each state
    out = np.empty_like(B)
    a = params["pi"] * B[0]
    out[0] = a / a.sum()
    for t in range(1, len(x)):
"""
_HMMF_TAIL = """    return out

mk = p.regime_market()
r = np.log(mk["close"]).diff().dropna()
params = p.hmm_fit(r.to_numpy())
mine = p.check("hmm_filter", p.attempt(hmm_filter, r.to_numpy(), params), p.hmm_filter(r.to_numpy(), params))
print("state volatilities:", np.round(np.sqrt(params["var"]), 4), " stay probabilities:", np.round(np.diag(params["A"]), 3))"""

NB["04_models_regimes"] = [
    header("04", "Supervised models, calibration and regimes", "S7–S8 (Supervised models · Unsupervised learning & market regimes)",
           "1. Compare a logistic regression, a random forest and gradient boosting with the best possible model.\n"
           "2. See how class weights break probabilities, and fix them with calibration.\n"
           "3. Find hidden regimes with a Gaussian mixture and a hidden Markov model.\n"
           "4. Tell a smoothed regime label from a real-time one."),
    ("code", SETUP),
    ("md", "## 1. Three models against the truth\n\n"
           "`p.planted_features()` has one real `signal` with a known probability `P(y = 1) = 1/(1 + e^{−1.5·signal})`, a `twin` "
           "(signal + noise) and two noise features. Knowing the true probability gives the best possible AUC, so we can see how "
           "close each model gets."),
    ("code", """from sklearn.metrics import roc_auc_score

X, y = p.planted_features()
true_p = 1 / (1 + np.exp(-1.5 * X["signal"]))
tr, te = slice(0, 1400), slice(1400, 2000)
rows = {"best possible (true probability)": {"AUC": roc_auc_score(y.iloc[te], true_p.iloc[te]), "mean |p − true p|": 0.0}}
for kind in ("logit", "rf", "gbm"):
    prob = p.make_model(kind).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    rows[kind] = {"AUC": roc_auc_score(y.iloc[te], prob), "mean |p − true p|": np.abs(prob - true_p.iloc[te]).mean()}
pd.DataFrame(rows).T.round(3)"""),
    ("md", "The logistic regression is almost perfect: the truth *is* logistic. The forest and the boosted trees are close in AUC "
           "but their probabilities are further from the truth. More flexible is not better when the signal is simple.\n\n"
           "## 2. Probabilities you can size with\n\n"
           "A crash-type label is rare (about 10% here). A common trick for imbalanced labels, `class_weight=\"balanced\"`, makes the "
           "model predict far too often: its \"probabilities\" average about 0.35. Bet sizing on them would be badly wrong. "
           "**Calibrate** on a separate, later sample (`p.calibrate`, isotonic). Then check with a **reliability table**: equal-width "
           "probability bins on [0, 1] (`bins = 5` → bin = `min(int(p·5), 4)`), and per non-empty bin the mean prediction, the "
           "observed frequency and the count."),
    ("code", """from sklearn.linear_model import LogisticRegression

Xr, _ = p.planted_features(4000, seed=3)
true_rare = 1 / (1 + np.exp(-(1.5 * Xr["signal"] - 3)))
y_rare = pd.Series((np.random.default_rng(5).random(len(Xr)) < true_rare).astype(int), index=Xr.index)
model = LogisticRegression(class_weight="balanced").fit(Xr.iloc[:2000], y_rare.iloc[:2000])
calibrated = p.calibrate(model, Xr.iloc[2000:3000], y_rare.iloc[2000:3000])
y_test = y_rare.iloc[3000:]
prob_raw = model.predict_proba(Xr.iloc[3000:])[:, 1]
prob_cal = calibrated.predict_proba(Xr.iloc[3000:])[:, 1]"""),
    ("md", YOUR_TURN),
    ("ex", """def reliability_table(prob, y, bins=5):
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=float)
    b = ...                                                          # ✍️ bin number of each prediction
    g = pd.DataFrame({"b": b, "p": prob, "y": y}).groupby("b")
    return pd.DataFrame({"mean_pred": g["p"].mean(), "frac_pos": g["y"].mean(), "count": g.size()})
""" + _REL_CHECK,
     """def reliability_table(prob, y, bins=5):
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=float)
    b = np.minimum((prob * bins).astype(int), bins - 1)
    g = pd.DataFrame({"b": b, "p": prob, "y": y}).groupby("b")
    return pd.DataFrame({"mean_pred": g["p"].mean(), "frac_pos": g["y"].mean(), "count": g.size()})
""" + _REL_CHECK),
    ("md", "Raw, a predicted 0.9 happens less than half the time. Calibrated, the mean prediction matches the base rate and each "
           "well-populated bin's prediction is close to what happens (the top bins hold only a few dozen predictions, so they are "
           "noisy). Calibration barely changes the ranking (the AUC); it makes the numbers usable for sizing.\n\n"
           "## 3. Regimes\n\n"
           "The regime market hides a calm trending state and a volatile choppy one. A **Gaussian mixture** on rolling volatility and "
           "efficiency clusters days without any notion of time. A **hidden Markov model** adds persistence: today's state depends on "
           "yesterday's. `p.hmm_fit` fits a 2-state Gaussian HMM on the returns (Baum–Welch). Its **forward filter** gives "
           "`P(state_t | returns up to t)`, a real-time signal: from yesterday's probabilities, predict with the transition matrix "
           "and weight by today's likelihood, `a = (out[t−1] @ A) × B[t]`, then normalize."),
    ("md", YOUR_TURN),
    ("ex", _HMMF_HEAD + """        a = ...                                                      # ✍️ predict, then weight by B[t]
        out[t] = ...                                                 # ✍️ normalize
""" + _HMMF_TAIL,
     _HMMF_HEAD + """        a = (out[t - 1] @ params["A"]) * B[t]
        out[t] = a / a.sum()
""" + _HMMF_TAIL),
    ("code", """rr = np.log(mk["close"]).diff()
feats = pd.DataFrame({"vol20": rr.rolling(20).std(), "er20": rr.rolling(20).sum().abs() / rr.abs().rolling(20).sum()})
calm_truth = (mk["regime"] == 1).astype(int)                      # the hidden trend regime is the calm one
calm_state = int(np.argmin(params["var"]))
labels = {"GMM (vol20, er20)": (p.gmm_regimes(feats) == 0).astype(int),
          "HMM, Viterbi (whole sample)": (p.hmm_regimes(rr) == 0).astype(int),
          "HMM, forward filter (real time)": pd.Series((mine[:, calm_state] > 0.5).astype(int), index=r.index)}
pd.DataFrame({k: {"agreement with the hidden regime": (v == calm_truth.loc[v.index]).mean(),
                  "regime switches": int((v.diff() != 0).sum())} for k, v in labels.items()}
             | {"truth": {"agreement with the hidden regime": 1.0, "regime switches": int((mk["regime"].diff() != 0).sum())}}).T.round(3)"""),
    ("md", "The GMM gets most days right but flickers; the HMM's Viterbi path is both more accurate and about as persistent as "
           "the truth. But Viterbi uses the **whole** sample, future included: it describes the past, it cannot be traded. The "
           "real-time filter is still good, with more switches. Does the regime matter for a strategy?"),
    ("code", """momentum = p.momentum_side(mk["close"]).shift(1) * rr                 # yesterday's signal, today's return
p.regime_performance(momentum, labels["HMM, forward filter (real time)"].map({1: "calm", 0: "volatile"})).round(3)"""),
    ("md", "Momentum earns in the calm state and loses in the volatile one, as the hidden regimes were built to do. Here a real-time "
           "regime label is a useful **feature** (or a filter for the meta-model).\n\n"
           "## Wrap-up\n\n"
           "* Start with the simplest model that fits the signal; flexibility costs variance.\n"
           "* Class weights and many models give scores, not probabilities: calibrate on later data before sizing.\n"
           "* Regime labels from smoothers (Viterbi) look great and use the future; use filtered probabilities in trading.\n"
           "* Graded version: `labs/part10/week34_labels` (`make_model`, `calibrate`, `reliability_table`, `gmm_regimes`, "
           "`hmm_regimes` with hmmlearn)."),
]

# ---------------------------------------------------------------------------------------------- 05
_PKF_HEAD = """class MyPurgedKFold(p.PurgedKFold):
    def split(self, X, y=None, groups=None):
        t0 = self.t1.index
        n, emb = len(t0), int(np.ceil(self.embargo * len(t0)))
        for test in np.array_split(np.arange(n), self.n_splits):
            test_start, test_end = t0[test[0]], self.t1.iloc[test].max()
            emb_end = t0[min(test[-1] + emb, n - 1)]
"""
_PKF_TAIL = """            yield train, test

X, y, t1 = p.overlapping_dataset()                       # NO signal; labels span the next 20 days
toy_t1 = pd.Series(pd.bdate_range("2024-01-01", periods=12)[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11]],
                   index=pd.bdate_range("2024-01-01", periods=12))
cases = [(t1, X), (toy_t1, toy_t1), (t1, X, 3, 0.05)]
mine = [p.attempt(lambda t, data, k=5, e=0.01: list(MyPurgedKFold(t, k, e).split(data)), *c) for c in cases]
mine = p.check("PurgedKFold.split", mine, [list(p.PurgedKFold(c[0], *c[2:]).split(c[1])) for c in cases])
train, test = mine[0][2]
print(f"fold 3 of 5: {len(test)} test rows, {len(train)} training rows, {len(X) - len(test) - len(train)} purged or embargoed")"""

_MDA_HEAD = """from sklearn.base import clone
from sklearn.metrics import log_loss

def mda_importance(model, X, y, cv, n_repeats=3, seed=0, groups=None):
    rng = np.random.default_rng(seed)
    groups = groups or {c: [c] for c in X.columns}
    drops = {g: [] for g in groups}
    for tr, te in cv.split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        base = log_loss(y.iloc[te], m.predict_proba(X.iloc[te]), labels=m.classes_)
        for g, cols in groups.items():
            for _ in range(n_repeats):
                Xs = X.iloc[te].copy()
                perm = rng.permutation(len(Xs))
"""
_MDA_TAIL = """    return pd.Series({g: float(np.mean(v)) for g, v in drops.items()}).sort_values(ascending=False)

Xp, yp = p.planted_features()                            # y depends on `signal` only; `twin` = signal + noise
cv_p = p.PurgedKFold(pd.Series(Xp.index, index=Xp.index), 5, 0.0)   # independent rows: plain contiguous folds
rf = p.make_model("rf")
clusters = p.cluster_features(Xp)
mine = {"MDA": p.attempt(mda_importance, rf, Xp, yp, cv_p), "clustered MDA": p.attempt(mda_importance, rf, Xp, yp, cv_p, 3, 0, clusters)}
mine = p.check("mda_importance", mine, {"MDA": p.mda_importance(rf, Xp, yp, cv_p), "clustered MDA": p.mda_importance(rf, Xp, yp, cv_p, 3, 0, clusters)})
print("clusters:", clusters)"""

NB["05_purged_cv_importance"] = [
    header("05", "Purged cross-validation and feature importance", "S9–S10 (Cross-validation for financial ML · Feature importance)",
           "1. Watch a shuffled k-fold find skill in pure noise.\n"
           "2. Write the purging rule that removes overlapping training labels.\n"
           "3. Compare MDI, MDA and single-feature importance on features with known importance.\n"
           "4. Handle substitute features with clustered MDA."),
    ("code", SETUP),
    ("md", "## 1. Skill in noise\n\n"
           "`p.overlapping_dataset()` is a random walk: features are past returns and volatility, the label is whether the **next 20 "
           "days** go up. There is nothing to learn. But neighbouring rows share 19 of their 20 label days, so a shuffled k-fold puts "
           "near-copies of each test row in the training set.\n\n"
           "**Purged k-fold** keeps contiguous test folds and drops every training label that overlaps the test period. A training "
           "row (label from `t0` to `t1`) survives only if it **ends before** the test fold starts (`t1 < test_start`) or **starts "
           "after** both the last test label ends and the embargo (`t0 > max(test_end, emb_end)`). Return the surviving positions "
           "(`np.where(...)[0]`)."),
    ("md", YOUR_TURN),
    ("ex", _PKF_HEAD + """            train = ...                                              # ✍️
""" + _PKF_TAIL,
     _PKF_HEAD + """            train = np.where((self.t1.to_numpy() < test_start) | (t0 > max(test_end, emb_end)))[0]
""" + _PKF_TAIL),
    ("code", """from sklearn.model_selection import KFold

rf = p.make_model("rf")
schemes = {"shuffled k-fold": p.ShuffledKFold(5), "contiguous k-fold, no purging": KFold(5), "purged k-fold": p.PurgedKFold(t1, 5, 0.01)}
pd.Series({k: p.cv_scores(rf, X, y, cv)["auc"].mean() for k, cv in schemes.items()}, name="random forest AUC on pure noise").round(3).to_frame()"""),
    ("md", "The shuffled k-fold reports an AUC near 0.7 on a random walk: pure leakage through overlapping labels. Contiguous folds "
           "leak only at their edges; purging removes even that. Every CV number in a financial ML report must come from purged "
           "folds (or a walk-forward).\n\n"
           "## 2. Which features matter?\n\n"
           "`p.planted_features()` knows the answer: `signal` drives the label, `twin` is a noisy copy of it, `noise_cont` and "
           "`noise_bin` are noise. **MDI** (the forest's built-in importance) is computed in sample and favours continuous features. "
           "**MDA** is out of sample: shuffle a feature in the test fold and measure how much the log loss gets worse. With "
           "`groups`, the columns of a group are shuffled **together** with one permutation (`Xs[cols].to_numpy()[perm]`)."),
    ("md", YOUR_TURN),
    ("ex", _MDA_HEAD + """                Xs[cols] = ...                                   # ✍️ the group's columns, rows permuted together
                drops[g].append(...)                             # ✍️ the log-loss increase over base
""" + _MDA_TAIL,
     _MDA_HEAD + """                Xs[cols] = Xs[cols].to_numpy()[perm]
                drops[g].append(log_loss(y.iloc[te], m.predict_proba(Xs), labels=m.classes_) - base)
""" + _MDA_TAIL),
    ("code", """mdi = p.mdi_importance(p.make_model("rf").fit(Xp, yp), Xp.columns)
sfi = p.sfi_importance(p.make_model("logit"), Xp, yp, cv_p)
pd.DataFrame({"MDI (in sample)": mdi, "MDA (log-loss increase)": mine["MDA"], "SFI (AUC alone)": sfi}).round(3)"""),
    ("md", "MDI hands some importance to continuous noise, just because a tree can split on it. MDA gives noise about zero, and "
           "splits the signal's importance between `signal` and its `twin`: shuffling one hurts little because the other is still "
           "there (a substitution effect). Single-feature importance sees each feature alone, so `twin` looks almost as good as "
           "`signal`. **Clustered MDA** shuffles correlated features together: the `signal` cluster gets the full importance "
           "(compare the clustered value with the two single MDA values)."),
    ("code", """mine["clustered MDA"].round(3).to_frame("clustered MDA")"""),
    ("md", "## Wrap-up\n\n"
           "* Overlapping labels + shuffled folds = skill in noise. Purge and embargo, always.\n"
           "* Trust out-of-sample importance (MDA, SFI); MDI flatters continuous and noisy features.\n"
           "* Correlated features share importance; cluster them before you drop \"unimportant\" ones.\n"
           "* Graded version: `labs/part10/week35_validation` (`PurgedKFold`, `cv_scores`, `mda_importance`, `sfi_importance`, "
           "`cluster_features`)."),
]

# ---------------------------------------------------------------------------------------------- 06
_SHUF_CHECK = """
mk = p.regime_market()
ds = p.meta_dataset(mk)                                  # momentum meta-labels at CUSUM events (notebook 03)
Xm, ym = ds[p.FEATURES], ds["y"]
cv = p.PurgedKFold(ds["t1"], 5, 0.01)
rf = p.make_model("rf")
mine = [p.attempt(shuffled_labels_test, rf, Xm, ym, cv, s) for s in range(3)]
mine = p.check("shuffled_labels_test", mine, [p.shuffled_labels_test(rf, Xm, ym, cv, seed=s) for s in range(3)])
real = p.cv_scores(rf, Xm, ym, cv)["auc"].mean()
print(f"real labels: purged AUC {real:.3f};  3 shuffles: {[round(m['auc'], 3) for m in mine]}")"""

_OVL_CHECK = """
from sklearn.model_selection import KFold

t1 = ds["t1"]
splits = {"purged fold 1": next(iter(cv.split(Xm))), "plain k-fold, fold 3": list(KFold(5).split(Xm))[2],
          "shuffled k-fold, fold 1": next(iter(p.ShuffledKFold(5).split(Xm)))}
mine = {k: p.attempt(overlap_test, tr, te, t1) for k, (tr, te) in splits.items()}
mine = p.check("overlap_test", mine, {k: p.overlap_test(tr, te, t1) for k, (tr, te) in splits.items()})
pd.DataFrame(mine).T"""

_ENS_CHECK = """
n = len(ds)
cut = int(0.6 * n)
known = ds.iloc[:cut][ds["t1"].iloc[:cut] < ds.index[cut]]          # labels resolved before the test period
test = ds.iloc[cut:]
make = lambda s: p.make_model("rf", s)                                # noqa: E731
mine = p.check("seed_ensemble", p.attempt(seed_ensemble, make, known[p.FEATURES], known["y"], test[p.FEATURES]),
               p.seed_ensemble(make, known[p.FEATURES], known["y"], test[p.FEATURES]))
single = [roc_auc_score(test["y"], make(s).fit(known[p.FEATURES], known["y"]).predict_proba(test[p.FEATURES])[:, 1]) for s in range(5)]
print(f"test AUC of the 5 single seeds: {np.round(single, 3)};  the 5-seed ensemble: {roc_auc_score(test['y'], mine):.3f}")"""

NB["06_audit_tuning_registry"] = [
    header("06", "The leakage audit, honest tuning and a model registry", "S11–S12 (Leakage & overfitting audit · Hyperparameters, ensembles & registry)",
           "1. Run the audit: shuffled labels, a canary, a time shift and an overlap check.\n"
           "2. Tune hyperparameters while logging every trial.\n"
           "3. Average seeds instead of picking the lucky one.\n"
           "4. Move a model through registry stages that refuse shortcuts."),
    ("code", SETUP + "\nfrom sklearn.metrics import roc_auc_score"),
    ("md", "## 1. The audit\n\n"
           "Four tests, run on every model before anyone looks at its Sharpe ratio. First: **shuffle the labels** and refit under the "
           "same CV. A pipeline that still finds skill is leaking. Permute `y` with `np.random.default_rng(seed).permutation`, keep "
           "the index, and return `{\"auc\": mean CV AUC, \"passed\": |auc − 0.5| < tol}`."),
    ("md", YOUR_TURN),
    ("ex", """def shuffled_labels_test(model, X, y, cv, seed=0, tol=0.05):
    ys = ...                                                         # ✍️ y with its values permuted, same index
    auc = float(p.cv_scores(model, X, ys, cv)["auc"].mean())
    return {"auc": auc, "passed": abs(auc - 0.5) < tol}
""" + _SHUF_CHECK,
     """def shuffled_labels_test(model, X, y, cv, seed=0, tol=0.05):
    ys = pd.Series(np.random.default_rng(seed).permutation(y.to_numpy()), index=y.index)
    auc = float(p.cv_scores(model, X, ys, cv)["auc"].mean())
    return {"auc": auc, "passed": abs(auc - 0.5) < tol}
""" + _SHUF_CHECK),
    ("md", "With about 800 events, one shuffle is noisy: some land below 0.45 and \"fail\" the fixed tolerance, and cross-validated "
           "AUCs under the null tend to sit slightly **below** 0.5 (each training fold's class mix mirrors its test fold's). Read it "
           "as a permutation test: the real model's AUC should sit above the spread of the shuffles, as it does here, if only "
           "just. Second: every training label must end before the test span starts or start after it ends. Count the training "
           "rows whose `[t0, t1]` overlaps `[first test t0, last test t1]`."),
    ("md", YOUR_TURN),
    ("ex", """def overlap_test(train_idx, test_idx, t1):
    t0 = t1.index
    lo, hi = t0[test_idx].min(), t1.iloc[test_idx].max()
    tr_t0, tr_t1 = t0[train_idx], t1.iloc[train_idx].to_numpy()
    n = ...                                                          # ✍️ training labels that overlap [lo, hi]
    return {"n_overlap": n, "passed": n == 0}
""" + _OVL_CHECK,
     """def overlap_test(train_idx, test_idx, t1):
    t0 = t1.index
    lo, hi = t0[test_idx].min(), t1.iloc[test_idx].max()
    tr_t0, tr_t1 = t0[train_idx], t1.iloc[train_idx].to_numpy()
    n = int(((tr_t1 >= lo) & (tr_t0 <= hi)).sum())
    return {"n_overlap": n, "passed": n == 0}
""" + _OVL_CHECK),
    ("md", "The last two tests are given. A **canary** (a feature that *is* the label plus noise) must come out as the most important "
           "feature, or the importance machinery is broken. A **time shift** (features one bar later) must not improve the AUC, "
           "or the features contain information from the future."),
    ("code", """print("canary:", p.canary_test(rf, Xm, ym, cv))
print("time shift:", {k: round(v, 3) if isinstance(v, float) else v for k, v in p.time_shift_test(rf, Xm, ym, cv).items()})"""),
    ("md", "## 2. Tune, and count every trial\n\n"
           "`p.random_search` tries 10 random settings of a gradient-boosting model (the labs use Optuna) and **logs every trial**: "
           "the number of trials goes into the Deflated Sharpe Ratio (Part 8)."),
    ("code", """best, trials = p.random_search(Xm, ym, cv, n_trials=10)
print("best:", {k: round(v, 4) for k, v in best.items()}, f" (ln 2 = {np.log(2):.4f}: the log loss of always saying 50%)")
trials.round(4)"""),
    ("md", "The winning trial's log loss is essentially ln 2, a coin flip: the search picked the model that learned the least (tiny "
           "learning rate, big leaves), because every model that learned more mostly learned noise. That is a result: there is "
           "little to find in these features, and 10 more trials now go into the research log.\n\n"
           "## 3. Seeds are trials too\n\n"
           "A random forest's AUC changes with its seed. Reporting the best seed is selection bias. Average the predicted "
           "probabilities of several seeds instead (`make(seed)` builds the model)."),
    ("md", YOUR_TURN),
    ("ex", """def seed_ensemble(make, X_train, y_train, X_test, seeds=(0, 1, 2, 3, 4)):
    return ...                                                       # ✍️ mean of predict_proba[:, 1] over the seeds
""" + _ENS_CHECK,
     """def seed_ensemble(make, X_train, y_train, X_test, seeds=(0, 1, 2, 3, 4)):
    return np.mean([make(s).fit(X_train, y_train).predict_proba(X_test)[:, 1] for s in seeds], axis=0)
""" + _ENS_CHECK),
    ("md", "## 4. A registry that says no\n\n"
           "`p.ModelRegistry` is a small file-backed stand-in for MLflow. Versions start in `research` and move **one stage at a "
           "time** (`research → shadow → paper → live`), only if the audit passed. Promoting to live demotes the previous live "
           "version to paper."),
    ("code", """import tempfile
from pathlib import Path

reg = p.ModelRegistry(Path(tempfile.mkdtemp()) / "registry.json")
v1 = reg.register("meta_momentum", {"audit_passed": True, "cv_auc": round(float(real), 3), "trials": 10 + 5})
v2 = reg.register("meta_momentum", {"audit_passed": False, "cv_auc": 0.69, "note": "shuffled k-fold"})
for stage in ("shadow", "paper", "live"):
    reg.promote("meta_momentum", v1, stage)
for attempt, (version, stage) in {"skip a stage": (v2, "paper"), "failed audit": (v2, "shadow")}.items():
    try:
        reg.promote("meta_momentum", version, stage)
    except ValueError as err:
        print(f"{attempt}: refused ({err})")
print("live:", reg.get("meta_momentum", "live"))"""),
    ("md", "## Wrap-up\n\n"
           "* Audit before you believe: shuffled labels (as a permutation test), a canary, a time shift, no overlap.\n"
           "* Log every trial and every seed; the best of many is biased upward.\n"
           "* Promotion is a process with gates, not a decision in a notebook.\n"
           "* Graded version: `labs/part10/week35_validation` (the audit, `optuna_search`, `seed_ensemble`, `ModelRegistry`) and "
           "Clinic W3 (the ML research report)."),
]

# ---------------------------------------------------------------------------------------------- 07
_WFM_HEAD = """def walk_forward_meta(ds, kind="rf", start=1 / 3, retrain_every=100, cost_bps=5.0):
    n, cost, rows = len(ds), cost_bps / 1e4, []
    for s in range(int(start * n), n, retrain_every):
"""
_WFM_TAIL = """        m = p.fit_weighted(p.make_model(kind), known[p.FEATURES], known["y"], known["w"])
        block = ds.iloc[s:s + retrain_every]
        prob = m.predict_proba(block[p.FEATURES])[:, 1]
        size = p.bet_size(prob)
        rows.append(pd.DataFrame({"prob": prob, "size": size, "y": block["y"], "primary": block["ret"] - 2 * cost,
                                  "meta": size * block["ret"] - 2 * cost * size}, index=block.index))
    return pd.concat(rows)

mk = p.regime_market(6000, seed=1)                       # the labs' market: 24 years, hidden trend/chop regimes
ds = p.meta_dataset(mk)
mine = p.check("walk_forward_meta", p.attempt(walk_forward_meta, ds, "logit"), p.walk_forward_meta(ds, "logit"))
print(f"{len(ds)} events; {len(mine)} traded out of sample, retrained every 100 events")"""

_HAR_CHECK = """
g = p.garch_returns()                                    # GARCH volatility: clusters, so it is forecastable
cases = [g["ret"].abs(), g["ret"].abs().iloc[:60]]
mine = [p.attempt(har_features, c) for c in cases]
mine = p.check("har_features", mine, [p.har_features(c) for c in cases])
har = mine[0]
fc = p.vol_forecasts(har, split=int(0.6 * len(har)))
true_vol = g["vol"].shift(-1).loc[fc.index]              # the TRUE conditional volatility of tomorrow
pd.DataFrame({"correlation with the true volatility": {c: np.corrcoef(fc[c], true_vol)[0, 1] for c in ("naive", "har", "gbm")},
              "correlation with tomorrow's |return|": {c: np.corrcoef(fc[c], fc["target"])[0, 1] for c in ("naive", "har", "gbm")}}).round(3)"""

_RISK_CHECK = """
cw = p.crash_warning(g["ret"], split=1500)
cases = [(g["ret"], cw["prob"]), (g["ret"], cw["prob"], 0.7, 0.0)]
mine = [p.attempt(risk_off_returns, *c) for c in cases]
mine = p.check("risk_off_returns", mine, [p.risk_off_returns(*c) for c in cases])
hold = g["ret"].loc[mine[0].index]
print(f"crash base rate {cw['label'].mean():.3f}; PR-AUC {p.pr_auc(cw['prob'], cw['label']):.3f} "
      f"({p.pr_auc(cw['prob'], cw['label']) / cw['label'].mean():.1f}× the base rate); warnings on {(cw['prob'] > 0.5).mean():.0%} of days")
pd.DataFrame({"buy and hold": {"max drawdown": p.max_drawdown(hold), "Sharpe": p.sharpe(hold)},
              "halve exposure after a warning": {"max drawdown": p.max_drawdown(mine[0]), "Sharpe": p.sharpe(mine[0])}}).round(3)"""

NB["07_ml_strategies"] = [
    header("07", "ML strategies: meta-labeled momentum, volatility and crash warnings",
           "S13–S14 (ML strategy I: direction with meta-labeling · ML strategy II: ranking, regimes, volatility & rare events)",
           "1. Run a walk-forward meta-labeled momentum strategy with purged retraining.\n"
           "2. Forecast volatility with HAR and with gradient boosting.\n"
           "3. Build a crash warning, judge it by PR-AUC, and use it only to cut risk."),
    ("code", SETUP + "\nfrom sklearn.metrics import roc_auc_score"),
    ("md", "## 1. Meta-labeled momentum, walk-forward\n\n"
           "Every 100 events, retrain the meta-model and trade the next block. The model may only learn from events whose label was "
           "**already resolved**: their `t1` is strictly before the current event's `t0`. Anything else trains on outcomes that "
           "haven't happened yet (a subtle leak that a simple `ds.iloc[:s]` makes)."),
    ("md", YOUR_TURN),
    ("ex", _WFM_HEAD + """        known = ...                                                  # ✍️ events before s whose label ended before ds.index[s]
""" + _WFM_TAIL,
     _WFM_HEAD + """        known = ds.iloc[:s][ds["t1"].iloc[:s] < ds.index[s]]
""" + _WFM_TAIL),
    ("code", """oos = {kind: (mine if kind == "logit" else p.walk_forward_meta(ds, kind)) for kind in ("logit", "rf")}
years = (mine.index[-1] - mine.index[0]).days / 365.25
table = pd.concat({kind: p.meta_summary(o, years) for kind, o in oos.items()})
display(table.round(3))
regime = mk["regime"].loc[oos["rf"].index].map({0: "chop", 1: "trend"})
print("random forest: mean bet size by hidden regime", oos["rf"]["size"].groupby(regime).mean().round(3).to_dict(),
      f"| AUC {roc_auc_score(oos['rf']['y'], oos['rf']['prob']):.3f}")"""),
    ("md", "The primary momentum rule loses after costs. The meta-model's AUC is only about 0.54, yet it turns the strategy "
           "positive: it bets several times more in the trend regime than in chop, and skips many losing trades. Meta-labeling "
           "rarely needs a strong model; it needs one that knows when to stay small. (Two model families tried: log both for the "
           "Deflated Sharpe Ratio.)\n\n"
           "## 2. Volatility: the simple model wins\n\n"
           "HAR (Corsi 2009) forecasts tomorrow's volatility proxy `v_{t+1}` (here `|return|`) from `d = v_t`, `w` = the mean of the "
           "last 5 values and `m` = the mean of the last 22, all ending at `t`. Build the table with `target = v.shift(-1)` and drop "
           "the NaN rows."),
    ("md", YOUR_TURN),
    ("ex", """def har_features(absret):
    return pd.DataFrame({"d": absret, "w": ..., "m": ...,                 # ✍️ w and m
                         "target": absret.shift(-1)}).dropna()
""" + _HAR_CHECK,
     """def har_features(absret):
    return pd.DataFrame({"d": absret, "w": absret.rolling(5).mean(), "m": absret.rolling(22).mean(),
                         "target": absret.shift(-1)}).dropna()
""" + _HAR_CHECK),
    ("md", "A linear HAR tracks the true GARCH volatility almost perfectly; gradient boosting on the same three inputs is much "
           "worse, because the noisy target (a single day's `|return|`) invites it to fit noise. Neither can predict tomorrow's "
           "`|return|` well: that is mostly luck. Judge volatility models against the latent truth when you have it, and against "
           "HAR always.\n\n"
           "## 3. Rare events: warn, don't bet\n\n"
           "`p.crash_warning` estimates the probability of a 4% fall within 5 days from recent volatility (a balanced logistic "
           "regression, trained only on resolved labels). Crashes happen on about 4% of days, so compare its **PR-AUC** with 4%, "
           "not with 0.5. Use the warning to **cut risk**: exposure 1, or `low` on the day *after* a warning "
           "(`prob.shift(1) > threshold`). Never short on it."),
    ("md", YOUR_TURN),
    ("ex", """def risk_off_returns(ret, prob, threshold=0.5, low=0.5):
    expo = ...                                                       # ✍️ exposure Series, decided the day before
    return ret.loc[expo.index] * expo
""" + _RISK_CHECK,
     """def risk_off_returns(ret, prob, threshold=0.5, low=0.5):
    expo = pd.Series(np.where(prob > threshold, low, 1.0), index=prob.index).shift(1).dropna()
    return ret.loc[expo.index] * expo
""" + _RISK_CHECK),
    ("md", "About twice the base rate is a modest warning, but it is enough to shave the worst drawdown and even lift the Sharpe "
           "ratio a little, because crashes cluster in high-volatility periods. The same model used as a short signal would be "
           "wrong most of the time.\n\n"
           "## Wrap-up\n\n"
           "* Retrain on resolved labels only; walk forward; count every model family you try.\n"
           "* Meta-labeling earns by sizing, not by predicting direction.\n"
           "* Volatility: HAR first. Rare events: PR-AUC against the base rate, used for risk cuts.\n"
           "* Graded version: `labs/part10/week36_strategies` (`walk_forward_meta`, `meta_summary`, `vol_forecasts`, "
           "`crash_warning`, `rank_ic`)."),
]

# ---------------------------------------------------------------------------------------------- 08
_PSI_HEAD = """def psi(expected, actual, bins=10):
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
"""
_PSI_TAIL = """
rng = np.random.default_rng(0)
ref, same, shifted = rng.normal(size=2000), rng.normal(size=2000), rng.normal(0.5, 1, 2000)
cases = [(ref, same), (ref, shifted), (ref, rng.normal(size=60))]
mine = [p.attempt(psi, *c) for c in cases]
mine = p.check("psi", mine, [p.psi(*c) for c in cases])
small = [p.psi(ref, rng.normal(size=60)) for _ in range(200)]
print(f"same distribution: PSI {mine[0]:.3f};  shifted by 0.5σ: {mine[1]:.3f};  "
      f"60 values, no drift at all: mean {np.mean(small):.3f}, above 0.25 in {np.mean(np.array(small) > 0.25):.0%} of 200 tries")"""

_SR_HEAD = """from collections import deque

class MyStreamingRolling:
    def __init__(self, window):
        self.window, self.buf = window, deque()
        self.s = self.s2 = 0.0

    def update(self, x):
        self.buf.append(x)
        self.s += x
        self.s2 += x * x
        if len(self.buf) > self.window:
"""
_SR_TAIL = """        if len(self.buf) < self.window:
            return np.nan, np.nan
        n = self.window
        mean = self.s / n
        return mean, float(np.sqrt(max((self.s2 - n * mean * mean) / (n - 1), 0.0)))

values = np.log(p.regime_market()["close"]).diff().dropna().to_numpy()[:400]
live = MyStreamingRolling(20)
mine = p.check("StreamingRolling", p.attempt(lambda: np.array([live.update(v) for v in values])),
               pd.Series(values).rolling(20).agg(["mean", "std"]).to_numpy())
print(f"live (O(1) per tick) vs pandas (batch): max difference {np.nanmax(np.abs(mine - pd.Series(values).rolling(20).agg(['mean', 'std']).to_numpy())):.1e}")"""

NB["08_serving_drift"] = [
    header("08", "Serving, drift monitoring and shadow mode", "S15–S16 (Deployment, drift & retraining · ML project defense)",
           "1. Measure distribution drift with the Population Stability Index, and see how noisy it is on small windows.\n"
           "2. Monitor a feature over time and find when a known shift is detected.\n"
           "3. Compute a live feature one tick at a time, identical to its batch version.\n"
           "4. Compare a champion and a challenger in shadow mode."),
    ("code", SETUP),
    ("md", "## 1. The Population Stability Index\n\n"
           "Cut the training distribution into 10 quantile bins (outer edges ±∞), compute the share of training values `e` and of "
           "live values `a` in each bin (+1e-6 against empty bins), and sum `(a − e)·ln(a/e)`. Rules of thumb: below 0.1 stable, "
           "0.1–0.25 watch, above 0.25 drift."),
    ("md", YOUR_TURN),
    ("ex", _PSI_HEAD + """    return ...                                                       # ✍️
""" + _PSI_TAIL,
     _PSI_HEAD + """    return float(np.sum((a - e) * np.log(a / e)))
""" + _PSI_TAIL),
    ("md", "A half-sigma shift crosses the drift line; so does pure sampling noise on a small window. PSI on 60 values averages "
           "well above its large-sample value and \"drifts\" in a sizeable share of tries. Use windows large enough for the "
           "number of bins, or a test with a p-value (KS).\n\n"
           "## 2. A monitor on a known shift\n\n"
           "Two features, trained on 1,000 stationary days. The live period has 600 days; from live day 300 on, feature `a` shifts "
           "by 0.75σ. `p.drift_table` computes the PSI of each feature per block of live days (and a KS p-value for `a`); "
           "`p.first_alarm` returns the first block with any PSI above 0.25."),
    ("code", """rng = np.random.default_rng(3)
train = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
live = pd.DataFrame({"a": np.r_[rng.normal(size=300), rng.normal(0.75, 1, 300)], "b": rng.normal(size=600)},
                    index=pd.RangeIndex(1, 601, name="live day"))
for window in (30, 100):
    table = p.drift_table(train, live, window)
    alarms = (table[["a", "b"]] > 0.25).any(axis=1)
    print(f"blocks of {window} days: first alarm on live day {p.first_alarm(table)};  "
          f"false alarms before day 300: {int(alarms[table.index <= 300].sum())} of {int((table.index <= 300).sum())} blocks")
p.drift_table(train, live, 100).round(3)"""),
    ("md", "Small blocks raise alarms before anything has changed; larger blocks are quiet until the shift, then fire. Pick the "
           "window from the false-alarm rate you can live with, and confirm with a second statistic before retraining.\n\n"
           "## 3. Train/serve parity\n\n"
           "In research, features are computed on whole columns (`rolling(20)`); live, one tick arrives at a time. The live version "
           "must give **the same numbers**. Keep running sums `s` and `s2` of the last `window` values: when the buffer is too "
           "long, remove the oldest value from both sums."),
    ("md", YOUR_TURN),
    ("ex", _SR_HEAD + """            old = ...                                                # ✍️ drop the oldest value
            self.s -= ...                                            # ✍️
            self.s2 -= ...                                           # ✍️
""" + _SR_TAIL,
     _SR_HEAD + """            old = self.buf.popleft()
            self.s -= old
            self.s2 -= old * old
""" + _SR_TAIL),
    ("md", "Equal to machine precision. (Running sums can lose precision over millions of ticks; recompute them from the buffer "
           "now and then.) The parity test belongs in CI: the same input through both paths, the same output.\n\n"
           "## 4. Shadow mode\n\n"
           "A challenger runs next to the live champion on the same data; its positions are logged, only the champion trades. "
           "Promote only with enough days **and** a clear margin (`p.shadow_compare`: at least 60 days and a Sharpe margin of 0.3)."),
    ("code", """mk = p.regime_market()
r = np.log(mk["close"]).diff()
champion, challenger = p.momentum_side(mk["close"], 20), p.momentum_side(mk["close"], 60)
rows = {f"{n} days": p.shadow_compare(r.iloc[:n], champion.iloc[:n], challenger.iloc[:n]) for n in (40, 400, 1500)}
pd.DataFrame(rows).T"""),
    ("md", "After 40 days nothing can be decided (the 60-day signal has barely started). After 400 days the challenger is ahead, "
           "but by less than the margin: no promotion. After 1,500 days it clears the margin, though both are weak strategies: "
           "shadow mode picks the better of two, it does not make either good.\n\n"
           "## S16 · The project defense\n\n"
           "The ML project defense (milestone M6) is a report, not a notebook. It must show, in this order: the baseline; "
           "the data (bars, events, point-in-time features, truncation test); the labels and weights; purged CV results with the "
           "audit; every trial counted for the DSR; walk-forward results after costs against the primary rule; the monitoring "
           "plan (drift, parity, shadow period). Clinic W3 in the labs builds it.\n\n"
           "## Wrap-up\n\n"
           "* PSI needs enough data per bin; small windows cry wolf.\n"
           "* Live features must equal research features: test the parity.\n"
           "* Shadow mode first, promotion by rule.\n"
           "* Graded version: `labs/part10/week36_strategies` (`psi`, `drift_table`, `StreamingRolling`, `shadow_compare`)."),
]

# ---------------------------------------------------------------------------------------------- 09
_WDS_HEAD = """from torch.utils.data import Dataset

class MyWindowDataset(Dataset):
    def __init__(self, X, y, lookback, normalize=True):
        X = np.asarray(X, dtype=np.float32)
        self.X = X.reshape(len(X), -1)                                # (time, features)
        self.y, self.L, self.normalize = np.asarray(y, dtype=np.float32), lookback, normalize

    def __len__(self):
        return len(self.X) - self.L + 1

    def __getitem__(self, i):
"""
_WDS_TAIL = """
x = p.ar1(3000, 0.6)                                     # x[t] = 0.6·x[t−1] + noise: the best forecast correlation is 0.6
L = 20
def items(ds):                                           # three windows and their targets, as NumPy arrays
    return [tuple(t.numpy() for t in ds[i]) for i in (0, 7, 2000)]

mine = [p.attempt(items, MyWindowDataset(x[:-1], x[1:], L, norm)) for norm in (True, False)]
mine = p.check("WindowDataset", mine, [items(dl.WindowDataset(x[:-1], x[1:], L, norm)) for norm in (True, False)])
ref_ds = dl.WindowDataset(x[:-1], x[1:], L)
splits = dl.time_splits(len(ref_ds), L)
print(f"{len(ref_ds)} windows → train {len(splits[0])}, validation {len(splits[1])}, test {len(splits[2])}, "
      f"with gaps of {L} windows so no bar is shared")"""

_CONF_CHECK = """
cases = [(np.arange(1.0, 20.0), np.array([0.0, 10.0]), 0.1), (-np.arange(1.0, 10.0), np.zeros(1), 0.1), (res_val, pred_test, 0.1)]
mine = [p.attempt(conformal_interval, *c) for c in cases]
mine = p.check("conformal_interval", mine, [dl.conformal_interval(*c) for c in cases])
lo, hi = mine[2]
print(f"90% split-conformal interval: ± {(hi - lo)[0] / 2:.3f};  test coverage {dl.coverage(lo, hi, y_test):.3f}")"""

NB["09_pytorch_sequences"] = [
    header("09", "PyTorch for time series: windows, sequence models and conformal intervals",
           "S17–S18 (PyTorch for financial time series · Sequence models & uncertainty)",
           "1. Build a window dataset whose targets never leak into the inputs.\n"
           "2. Train an MLP, an LSTM and a causal TCN, and compare them with a one-line baseline.\n"
           "3. See what normalizing each window does to a level signal.\n"
           "4. Wrap any model's forecasts in a split-conformal interval, and watch it fail when volatility changes."),
    ("code", SETUP_DL),
    ("md", "## 1. Windows\n\n"
           "Item `i` of a window dataset is the input window `X[i .. i+L−1]` and the target `y[i+L−1]`, where `y` is already the "
           "**future** value aligned to the window's last row (here `y[t] = x[t+1]`). With `normalize=True`, each window is scaled "
           "with its **own** column means and stds (`(w − w.mean(0)) / (w.std(0) + 1e-8)`): no information from other windows, "
           "but also no information about the level. Return two float32 tensors (`torch.from_numpy(w)`, `torch.tensor(...)`)."),
    ("md", YOUR_TURN),
    ("ex", _WDS_HEAD + """        w = self.X[...]                                              # ✍️ the L rows ending at i+L−1
        if self.normalize:
            w = ...                                                  # ✍️
        return torch.from_numpy(w), torch.tensor(self.y[...])        # ✍️ the target of the window's last row
""" + _WDS_TAIL,
     _WDS_HEAD + """        w = self.X[i: i + self.L]
        if self.normalize:
            w = (w - w.mean(0)) / (w.std(0) + 1e-8)
        return torch.from_numpy(w), torch.tensor(self.y[i + self.L - 1])
""" + _WDS_TAIL),
    ("md", "## 2. Three networks against \"tomorrow ∝ today\"\n\n"
           "`dl.train` is a standard loop: AdamW, gradient clipping, early stopping on the validation loss, best weights restored. "
           "Train each model on the same windows, with and without per-window normalization, and compare the test correlation with "
           "the naive forecast `x[t]` (for an AR(1) with φ = 0.6, the best possible correlation is about 0.6). About 20 seconds."),
    ("code", """rows = {}
for norm in (True, False):
    ds = dl.WindowDataset(x[:-1], x[1:], L, normalize=norm)
    tr, va, te = dl.loaders(ds, splits)
    for name, make in (("MLP", lambda: dl.MLP(1, L, 16)), ("LSTM", lambda: dl.LSTMRegressor(1, 16)), ("TCN", lambda: dl.TCNRegressor(1, 8, 3))):
        dl.set_seed(0)
        model = make()
        dl.train(model, tr, va, epochs=30, patience=4)
        pred, y_test = dl.predict(model, te)
        rows[(name, "per-window normalized" if norm else "raw windows")] = np.corrcoef(pred, y_test)[0, 1]
naive = x[:-1][splits[2] + L - 1]
rows[("naive: tomorrow ∝ today", "")] = np.corrcoef(naive, y_test)[0, 1]
pd.Series(rows, name="test correlation with the target").round(3).to_frame()"""),
    ("md", "Nothing beats the naive rule, and per-window normalization makes every network worse: it erases the level of `x[t]`, "
           "which *is* the signal. The networks learn the AR(1), slowly and with more variance than a one-line rule. That is the "
           "normal outcome on simple structure; deep learning must earn its place against baselines like this.\n\n"
           "## 3. Split-conformal intervals\n\n"
           "Any point forecast can get an interval with a coverage guarantee, if the future looks like the past. On a calibration "
           "set (here the validation windows), take the absolute residuals `r`; with `n` of them, `q` is the `k`-th smallest where "
           "`k = ceil((n + 1)(1 − α))` (at most `n`); the interval is `pred ± q`. `np.quantile(r, min(1, k/n), "
           "method=\"inverted_cdf\")` gives exactly that."),
    ("code", """ds = dl.WindowDataset(x[:-1], x[1:], L, normalize=False)
tr, va, te = dl.loaders(ds, splits)
dl.set_seed(0)
lstm = dl.LSTMRegressor(1, 16)
dl.train(lstm, tr, va, epochs=30, patience=4)
pred_val, y_val = dl.predict(lstm, va)
pred_test, y_test = dl.predict(lstm, te)
res_val = y_val - pred_val"""),
    ("md", YOUR_TURN),
    ("ex", """def conformal_interval(cal_residuals, pred, alpha=0.1):
    r = np.abs(np.asarray(cal_residuals, dtype=float))
    n = len(r)
    q = ...                                                          # ✍️
    pred = np.asarray(pred, dtype=float)
    return pred - q, pred + q
""" + _CONF_CHECK,
     """def conformal_interval(cal_residuals, pred, alpha=0.1):
    r = np.abs(np.asarray(cal_residuals, dtype=float))
    n = len(r)
    q = np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="inverted_cdf")
    pred = np.asarray(pred, dtype=float)
    return pred - q, pred + q
""" + _CONF_CHECK),
    ("md", "About 90%, as promised. The promise needs **exchangeability**: calibration and test data from the same distribution. "
           "Double the noise in the test period and the same recipe fails:"),
    ("code", """rng = np.random.default_rng(9)
noise = rng.normal(0, 1, 3000)
cut = int(0.85 * 3000)
noise[cut:] *= 2                                         # volatility doubles where the test period starts
x2 = np.zeros(3000)
for t in range(1, 3000):
    x2[t] = 0.6 * x2[t - 1] + noise[t]
ds2 = dl.WindowDataset(x2[:-1], x2[1:], L, normalize=False)
tr2, va2, te2 = dl.loaders(ds2, splits)
dl.set_seed(0)
mlp = dl.MLP(1, L, 16)
dl.train(mlp, tr2, va2, epochs=30, patience=4)
pv, yv = dl.predict(mlp, va2)
pt, yt = dl.predict(mlp, te2)
lo2, hi2 = dl.conformal_interval(yv - pv, pt, 0.1)
print(f"calibrated on the calm period, tested after the volatility doubles: coverage {dl.coverage(lo2, hi2, yt):.2f} instead of 0.90")"""),
    ("md", "## Wrap-up\n\n"
           "* Windows end at `t`, targets are after `t`; splits need gaps of one look-back.\n"
           "* Every network gets the same windows as a naive baseline, and usually loses to it on simple structure.\n"
           "* Don't normalize away the signal: scale with **training** statistics when the level matters.\n"
           "* Conformal intervals are cheap and honest, until the regime changes: recalibrate on recent data.\n"
           "* Graded version: `labs/part10/week37_deep` (`WindowDataset`, `time_splits`, `train`, MLP/LSTM/TCN, "
           "`conformal_interval`) and Clinic W5 (LSTM vs LightGBM vs HAR)."),
]

# ---------------------------------------------------------------------------------------------- 10
_CC_HEAD = """from torch import nn

class MyCausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x):                                            # x: (batch, channels, time)
"""
_CC_TAIL = """
dl.set_seed(0)
x_in = torch.randn(2, 3, 30)
outs = []
for kernel, dilation in ((3, 1), (3, 4), (2, 8)):
    mine_conv, ref_conv = MyCausalConv1d(3, 4, kernel, dilation), dl.CausalConv1d(3, 4, kernel, dilation)
    ref_conv.load_state_dict(mine_conv.state_dict())                  # the same weights
    with torch.no_grad():
        got = p.attempt(lambda m=mine_conv: m(x_in).numpy())
        outs.append((got, ref_conv(x_in).numpy()))
p.check("CausalConv1d", [o[0] for o in outs], [o[1] for o in outs])

future = x_in.clone()
future[:, :, 20:] += 5.0                                 # change only the inputs from t = 20 on
with torch.no_grad():
    causal = (ref_conv(future) - ref_conv(x_in)).abs()[:, :, :20].max().item()
    centered_conv = nn.Conv1d(3, 4, 3, padding=1)
    centered = (centered_conv(future) - centered_conv(x_in)).abs()[:, :, 19].max().item()
print(f"outputs before t = 20 change by {causal:.1e} with causal padding, and by {centered:.2f} at t = 19 with centered padding")"""

_AE_CHECK = """
mine = p.check("anomaly_scores", p.attempt(anomaly_scores, ae, F.to_numpy(), mu, sd), dl.anomaly_scores(ae, F.to_numpy(), mu, sd))
score = pd.Series(mine, index=F.index)
from sklearn.metrics import roc_auc_score
print("median anomaly score by hidden regime:", score.groupby(regime).median().round(3).to_dict(),
      f"| AUC for spotting chop days: {roc_auc_score((regime == 'chop').astype(int), score):.2f}")"""

_PIN_CHECK = """
pred, target = torch.tensor([0.0, 1.0, 2.0, 3.0]), torch.tensor([1.0, 1.0, 0.0, 5.0])
mine = [p.attempt(lambda q=q: pinball_loss(pred, target, q).item()) for q in (0.1, 0.5, 0.9)]
mine = p.check("pinball_loss", mine, [dl.pinball_loss(pred, target, q).item() for q in (0.1, 0.5, 0.9)])
print("pinball losses for q = 0.1, 0.5, 0.9:", np.round(mine, 3))"""

NB["10_dl_strategies"] = [
    header("10", "Deep-learning building blocks: causality, anomalies, uncertainty and the honest comparison",
           "S19–S20 (Deep-learning strategies & advanced analysis · Optimization & the honest comparison)",
           "1. Write a causal convolution and prove it cannot see the future.\n"
           "2. Score market days with an autoencoder trained on calm days.\n"
           "3. Estimate uncertainty with MC dropout and forecast quantiles with the pinball loss.\n"
           "4. Report seed variance, not the best seed."),
    ("code", SETUP_DL),
    ("md", "## 1. Causal convolutions\n\n"
           "A convolution over time with symmetric padding mixes `t−1`, `t` and `t+1`: the output at `t` sees the future. A **causal** "
           "convolution pads only on the **left**, by `(kernel − 1)·dilation` zeros, so the output at `t` depends on inputs up to "
           "`t` and the length is unchanged. `nn.functional.pad(x, (left, right))` pads the last dimension."),
    ("md", YOUR_TURN),
    ("ex", _CC_HEAD + """        return self.conv(...)                                        # ✍️ pad on the left only
""" + _CC_TAIL,
     _CC_HEAD + """        return self.conv(nn.functional.pad(x, (self.pad, 0)))
""" + _CC_TAIL),
    ("md", "Exactly zero: the causal output cannot react to anything after `t`. Put this test in your test suite for every "
           "sequence model; a centered convolution fails it immediately.\n\n"
           "## 2. An autoencoder as an anomaly detector\n\n"
           "Train a small autoencoder (6 → 16 → 3 → 16 → 6) to reconstruct **calm** days' features from the regime market. On days "
           "unlike its training data it reconstructs poorly: the mean squared reconstruction error of the standardized row "
           "(standardized with the **training** mean and std) is the anomaly score."),
    ("code", """mk = p.regime_market()
r = np.log(mk["close"]).diff()
F = pd.DataFrame({"vol5": r.rolling(5).std(), "vol20": r.rolling(20).std(), "ret5": r.rolling(5).sum(),
                  "ret20": r.rolling(20).sum(), "absr": r.abs(), "er20": r.rolling(20).sum().abs() / r.abs().rolling(20).sum()}).dropna()
regime = mk["regime"].loc[F.index].map({0: "chop", 1: "trend"})
calm_days = F[regime == "trend"].iloc[:600]
ae, mu, sd = dl.fit_autoencoder(calm_days.to_numpy())"""),
    ("md", YOUR_TURN),
    ("ex", """def anomaly_scores(model, X, mu, sd):
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return ...                                                   # ✍️ mean squared error per row, as a NumPy array
""" + _AE_CHECK,
     """def anomaly_scores(model, X, mu, sd):
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return ((model(Z) - Z) ** 2).mean(1).numpy()
""" + _AE_CHECK),
    ("md", "Trained on calm days only, the autoencoder flags choppy days without ever seeing a label. In the platform this is a "
           "monitoring signal (\"today doesn't look like the data the models were trained on\"), not a trading signal.\n\n"
           "## 3. Uncertainty and quantiles\n\n"
           "**MC dropout** keeps dropout on at prediction time and runs many passes: the spread of the outputs is a rough "
           "uncertainty. It grows on inputs unlike the training data:"),
    ("code", """x = p.ar1(3000, 0.6)
L = 20
ds = dl.WindowDataset(x[:-1], x[1:], L, normalize=False)
splits = dl.time_splits(len(ds), L)
tr, va, te = dl.loaders(ds, splits)
dl.set_seed(0)
mlp = dl.MLP(1, L, 32, dropout=0.2)
dl.train(mlp, tr, va, epochs=30, patience=4)
windows = torch.stack([ds[i][0] for i in splits[2]])
_, std_normal = dl.mc_dropout(mlp, windows, 50)
_, std_extreme = dl.mc_dropout(mlp, windows[:50] * 4, 50)
print(f"MC-dropout std: {std_normal.mean():.2f} on test windows, {std_extreme.mean():.2f} on windows 4× larger than anything seen")"""),
    ("md", "To forecast a **quantile** instead of the mean, train with the pinball loss: with error `e = y − pred`, the loss is the "
           "mean of `max(q·e, (q − 1)·e)` (`torch.maximum`). It is minimized when a share `q` of the targets lies below the "
           "prediction."),
    ("md", YOUR_TURN),
    ("ex", """def pinball_loss(pred, y, q):
    e = y - pred
    return ...                                                       # ✍️
""" + _PIN_CHECK,
     """def pinball_loss(pred, y, q):
    e = y - pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))
""" + _PIN_CHECK),
    ("code", """rows = {}
for q in (0.1, 0.5, 0.9):
    dl.set_seed(0)
    model = dl.MLP(1, L, 16)
    dl.train(model, tr, va, epochs=30, patience=4, loss_fn=lambda a, b, q=q: dl.pinball_loss(a, b, q))
    pred, y_test = dl.predict(model, te)
    rows[f"q = {q}"] = np.mean(y_test <= pred)
pd.Series(rows, name="share of test targets below the forecast").round(3).to_frame()"""),
    ("md", "## 4. The honest comparison\n\n"
           "A deep model's result depends on its seed. Report the spread over seeds, next to the baseline, on the same test windows:"),
    ("code", """cors = []
for seed in range(3):
    dl.set_seed(seed)
    lstm = dl.LSTMRegressor(1, 16)
    dl.train(lstm, tr, va, epochs=30, patience=4)
    pred, y_test = dl.predict(lstm, te)
    cors.append(np.corrcoef(pred, y_test)[0, 1])
naive = np.corrcoef(x[:-1][splits[2] + L - 1], y_test)[0, 1]
print(f"LSTM over 3 seeds: {np.round(cors, 3)} (mean {np.mean(cors):.3f});  naive rule: {naive:.3f}")"""),
    ("md", "Three seeds, three slightly different results, all at or just below the one-line rule. The honest report says exactly "
           "that; the dishonest one shows the best seed on a different test window. Tuning (learning rate, window, hidden size) "
           "multiplies the trials: count them all.\n\n"
           "## Wrap-up\n\n"
           "* Test causality directly: perturb the future, the past outputs must not move.\n"
           "* Autoencoders make good \"this looks unusual\" monitors.\n"
           "* MC dropout and quantile losses give uncertainty; conformal intervals give coverage.\n"
           "* Seeds, baselines and trial counts belong in every deep-learning result.\n"
           "* Graded version: `labs/part10/week37_deep` (`CausalConv1d`, `TCNRegressor`, `pinball_loss`, `mc_dropout`, "
           "`AutoEncoder`, `anomaly_scores`)."),
]

# ---------------------------------------------------------------------------------------------- 11
_VI_HEAD = """def value_iteration(P, R, gamma=0.9, tol=1e-10):
    V = np.zeros(P.shape[1])
    while True:
"""
_VI_TAIL = """        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new

# states: 0 = calm, 1 = volatile; actions: 0 = invest, 1 = cash (the action doesn't change the market)
P = np.array([[[0.9, 0.1], [0.2, 0.8]], [[0.9, 0.1], [0.2, 0.8]]])
R = np.array([[1.0, 0.0], [-2.0, 0.0]])                  # investing earns 1 in calm, loses 2 in volatile
cases = [(P, R, 0.9), (P, R, 0.0), (np.array([np.eye(2), [[0, 1], [1, 0]]]), np.array([[0.0, 0.0], [0.0, 1.0]]), 0.9)]
mine = [p.attempt(value_iteration, *c) for c in cases]
mine = p.check("value_iteration", mine, [p.value_iteration(*c) for c in cases])
V, policy = mine[0]
print(f"values: calm {V[0]:.2f}, volatile {V[1]:.2f};  policy: calm → {['invest', 'cash'][policy[0]]}, volatile → {['invest', 'cash'][policy[1]]}")"""

_ENV_HEAD = """class MyTradingEnv(p.TradingEnv):
    def step(self, action):
        new_pos = float(action) - 1.0                                # action 0/1/2 → position −1/0/+1
"""
_ENV_TAIL = """        self.pos, self.t = new_pos, self.t + 1
        terminated = self.t >= self.end
        obs = self._obs() if not terminated else np.zeros(self.L + 1)
        return obs, float(reward), terminated, False, {}

returns = p.ar1(4000, 0.3, 0.01, seed=0)                 # AR(1) returns with φ = 0.3: momentum at lag 1
train, test = returns[:3000], returns[3000:]
actions = np.random.default_rng(0).integers(0, 3, 300)

def rewards(env):
    env.reset()
    return [env.step(a)[1] for a in actions]

mine = p.check("TradingEnv.step", p.attempt(rewards, MyTradingEnv(test, cost=0.0005)), rewards(p.TradingEnv(test, cost=0.0005)))
print(f"300 random actions: total reward {np.sum(mine):.4f}")"""

_DSR_CHECK = """
steady = np.r_[np.full(60, 0.001) + np.random.default_rng(1).normal(0, 0.002, 60), -0.03, np.full(20, 0.001)]
cases = [(steady,), (test[:500], 0.05)]
mine = [p.attempt(differential_sharpe, *c) for c in cases]
mine = p.check("differential_sharpe", mine, [p.differential_sharpe(*c) for c in cases])
print(f"day 60 (a −3% shock after steady gains): D = {mine[0][60]:.1f};  a typical steady day: D = {np.median(mine[0][30:60]):.2f}")"""

NB["11_rl_foundations_env"] = [
    header("11", "RL foundations and a trading environment", "S21–S22 (RL foundations · Trading environment design)",
           "1. Solve a small Markov decision process with value iteration.\n"
           "2. See tabular Q-learning overfit one price path and generalize from many.\n"
           "3. Write the reward of a trading environment with the right timing and costs.\n"
           "4. Compute the differential Sharpe ratio, a reward that cares about risk."),
    ("code", SETUP),
    ("md", "## 1. Value iteration\n\n"
           "With known transitions `P[a, s, s']` and rewards `R[s, a]`, the values solve `V(s) = max_a [R(s, a) + γ Σ P(s'|s, a) V(s')]`. "
           "Iterate: `Q = R + γ · einsum(\"ast,t->sa\", P, V)`, `V = max over actions`, until `V` stops changing. A two-state market: "
           "investing earns 1 in the calm state and loses 2 in the volatile one; states persist."),
    ("md", YOUR_TURN),
    ("ex", _VI_HEAD + """        Q = ...                                                      # ✍️ (states × actions)
""" + _VI_TAIL,
     _VI_HEAD + """        Q = R + gamma * np.einsum("ast,t->sa", P, V)
""" + _VI_TAIL),
    ("md", "Invest when calm, sit in cash when volatile: obvious here, but the same equation solves problems whose answer isn't "
           "obvious. In trading we rarely know `P` and `R`, so we **learn** from experience instead.\n\n"
           "## 2. Q-learning: one path or many?\n\n"
           "Tabular Q-learning on a mean-reverting OU path: the state is the price level in 17 buckets, the actions are short, flat "
           "or long, the reward is the next price change (`p.q_learning_paths`). Train once on a single 100-step path for 200 "
           "epochs, once on 200 different paths for one epoch, and test both on 300 new paths:"),
    ("code", """EDGES = np.linspace(-4, 4, 17)

def ou_path(n, seed, theta=0.2):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] * (1 - theta) + rng.normal()
    return x

one, many = [ou_path(100, 0)], [ou_path(100, s) for s in range(10, 210)]
new = [ou_path(100, s) for s in range(1000, 1300)]
q_one = p.q_learning_paths(one, EDGES, epochs=200)
q_many = p.q_learning_paths(many, EDGES, epochs=1)
pd.DataFrame({"one path, 200 epochs": {"in sample": p.greedy_reward(q_one, one, EDGES), "new paths": p.greedy_reward(q_one, new, EDGES)},
              "200 paths, 1 epoch": {"in sample": p.greedy_reward(q_many, many, EDGES), "new paths": p.greedy_reward(q_many, new, EDGES)}}).round(3)"""),
    ("md", "Trained on one history, the agent memorizes it: a great in-sample reward that falls apart on new paths. Trained on many "
           "paths it earns less in sample and keeps it out of sample. A backtest is one path; RL needs many (simulated or "
           "bootstrapped) paths.\n\n"
           "## 3. The environment\n\n"
           "A Gymnasium-style environment (`reset()`, `step(action)`) over daily returns `r`. The observation at step `t` contains "
           "returns **up to** `t−1`; the action sets the new position (−1, 0, +1), which then earns `r[t]`, the return **after** the "
           "decision. Trading costs `cost × |position change|`. Write the reward."),
    ("md", YOUR_TURN),
    ("ex", _ENV_HEAD + """        reward = ...                                                 # ✍️ uses self.r[self.t], self.cost and self.pos
""" + _ENV_TAIL,
     _ENV_HEAD + """        reward = new_pos * self.r[self.t] - self.cost * abs(new_pos - self.pos)
""" + _ENV_TAIL),
    ("md", "Before any agent, run simple rules through the environment on the test period (observations scaled with the "
           "**training** volatility):"),
    ("code", """rules = {"flat": lambda o: 1, "long only": lambda o: 2, "momentum": lambda o: 2 if o[-2] > 0 else 0,
         "reversal": lambda o: 0 if o[-2] > 0 else 2}
env = p.TradingEnv(test, cost=0.0002, scale=train.std())
pd.Series({k: p.run_policy(env, f) for k, f in rules.items()}, name="total test reward (2 bp costs)").round(3).to_frame()"""),
    ("md", "The data has momentum at lag 1, and the one-line momentum rule harvests it. Any agent will be judged against this "
           "number (notebook 12).\n\n"
           "## 4. A reward that cares about risk\n\n"
           "Summing P&L rewards risk-taking. Moody & Saffell's **differential Sharpe ratio** is the marginal contribution of today's "
           "return `R_t` to an exponentially weighted Sharpe ratio. Track `A` (mean) and `B` (second moment), both starting at 0: "
           "`ΔA = R_t − A`, `ΔB = R_t² − B`, then `D_t = (B·ΔA − ½·A·ΔB) / (B − A²)^{3/2}` using the **previous** `A` and `B` "
           "(`D_t = 0` while `B − A² <= 0`), and finally `A += η·ΔA`, `B += η·ΔB`."),
    ("md", YOUR_TURN),
    ("ex", """def differential_sharpe(returns, eta=0.01):
    A = B = 0.0
    out = []
    for R in np.asarray(returns, dtype=float):
        dA, dB = R - A, R * R - B
        den = B - A * A
        out.append(...)                                              # ✍️ D_t (0 while den <= 0)
        A, B = ..., ...                                              # ✍️ update the moving moments
    return np.array(out)
""" + _DSR_CHECK,
     """def differential_sharpe(returns, eta=0.01):
    A = B = 0.0
    out = []
    for R in np.asarray(returns, dtype=float):
        dA, dB = R - A, R * R - B
        den = B - A * A
        out.append((B * dA - 0.5 * A * dB) / den ** 1.5 if den > 0 else 0.0)
        A, B = A + eta * dA, B + eta * dB
    return np.array(out)
""" + _DSR_CHECK),
    ("md", "A loss after a calm run is punished far harder than a gain is rewarded: the reward sees the risk. Use it (or a P&L "
           "reward with a variance penalty) when the agent's position size matters.\n\n"
           "## Wrap-up\n\n"
           "* Value iteration when the model is known; learning when it isn't.\n"
           "* One history is one sample: train on many paths.\n"
           "* The reward at `t` uses the return **after** the decision, net of costs; scale observations with training statistics.\n"
           "* Graded version: `labs/part10/week38_rl` (`value_iteration`, `q_learning_paths`, `TradingEnv` with Gymnasium, "
           "`differential_sharpe`)."),
]

# ---------------------------------------------------------------------------------------------- 12
_QL_HEAD = """def q_learning_env(env, n_states, state_fn=p.sign_state, episodes=200, alpha=0.05, gamma=0.9, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, 3))
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        s, done = state_fn(obs), False
        while not done:
            a = int(rng.integers(3)) if rng.random() < eps else int(Q[s].argmax())
            obs, r, done, _, _ = env.step(a)
            s2 = state_fn(obs)
"""
_QL_TAIL = """            s = s2
    return Q

returns = p.ar1(4000, 0.3, 0.01, seed=0)
train, test = returns[:3000], returns[3000:]
train_env = p.TradingEnv(train, cost=0.0002, episode_len=250)          # random 250-day episodes
cases = [dict(episodes=40, seed=0), dict(episodes=40, seed=1, gamma=0.0)]
mine = [p.attempt(q_learning_env, train_env, 6, **c) for c in cases]
mine = p.check("q_learning_env", mine, [p.q_learning_env(train_env, 6, **c) for c in cases])
print("states: (last return > 0) × 3 + position + 1;  Q after 40 episodes:")
pd.DataFrame(mine[0], columns=["short", "flat", "long"], index=[f"{'up' if s >= 3 else 'down'} day, position {s % 3 - 1:+d}" for s in range(6)]).round(4)"""

_AC_HEAD = """def almgren_chriss_schedule(X, N, sigma, eta, gamma, lam):
    eta_t = eta - gamma / 2
"""
_AC_MID = """    if kappa < 1e-12:
        return p.twap_schedule(X, N)
    j = np.arange(N + 1)
"""
_AC_TAIL = """    return -np.diff(holdings)

X_SH, N, SIGMA, ETA, GAMMA = 1000, 20, 0.3, 0.01, 0.001
cases = [(X_SH, N, SIGMA, ETA, GAMMA, 1e-3), (X_SH, N, SIGMA, ETA, GAMMA, 1e-2), (500, 10, 0.5, 0.02, 0.0, 1e-4)]"""

NB["12_rl_agents_execution"] = [
    header("12", "RL agents under scrutiny, and optimal execution", "S23–S24 (RL strategies: trading, execution & hedging · RL optimization & M7a)",
           "1. Write the Q-learning update and train an agent in the trading environment.\n"
           "2. Judge it across seeds, regimes and costs, against simple rules.\n"
           "3. Compute the Almgren–Chriss execution schedule and its cost–risk trade-off.\n"
           "4. Check the schedule in a simulated execution environment."),
    ("code", SETUP),
    ("md", "## 1. A Q-learning trader\n\n"
           "Six states (was the last return up or down? × current position), three actions. After each step, move `Q[s, a]` toward "
           "the target `r + γ · max Q[s']`, or just `r` when the episode has ended: `Q[s, a] += α·(target − Q[s, a])`."),
    ("md", YOUR_TURN),
    ("ex", _QL_HEAD + """            target = ...                                             # ✍️
            Q[s, a] += ...                                           # ✍️
""" + _QL_TAIL,
     _QL_HEAD + """            target = r if done else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
""" + _QL_TAIL),
    ("md", "## 2. The defense: seeds, regimes, costs\n\n"
           "An RL result is only credible across **seeds** (the agent's randomness), **regimes** (momentum φ = +0.3, noise φ = 0, "
           "reversal φ = −0.3) and **costs**, next to the simple rules. Five agents per regime, 300 episodes each (about 10 seconds):"),
    ("code", """def greedy(Q):
    return lambda o: int(Q[p.sign_state(o)].argmax())

rules = {"long only": lambda o: 2, "momentum": lambda o: 2 if o[-2] > 0 else 0, "reversal": lambda o: 0 if o[-2] > 0 else 2}
rows = {}
for phi in (0.3, 0.0, -0.3):
    r = p.ar1(4000, phi, 0.01, seed=0)
    tr, te = r[:3000], r[3000:]
    test_env = p.TradingEnv(te, cost=0.0002, scale=tr.std())
    agents = [p.run_policy(test_env, greedy(p.q_learning_env(p.TradingEnv(tr, cost=0.0002, episode_len=250), 6, episodes=300, seed=s)))
              for s in range(5)]
    rows[f"φ = {phi:+.1f}"] = {"agent min": min(agents), "agent mean": np.mean(agents), "agent max": max(agents)} | {
        k: p.run_policy(test_env, f) for k, f in rules.items()}
pd.DataFrame(rows).T.round(2)"""),
    ("md", "With momentum, the agents' results range from a loss to the momentum rule's result, never above it: at best the agent "
           "rediscovers the one-line rule. On noise it learns to stay out; with reversal it learns reversal. That is what a working "
           "agent looks like on data with known structure, and why its seed spread must be reported.\n\n"
           "Now train with **free** trades and test at a realistic 30 bp:"),
    ("code", """r = p.ar1(4000, 0.3, 0.01, seed=0)
tr, te = r[:3000], r[3000:]
costly_test = p.TradingEnv(te, cost=0.003, scale=tr.std())
rows = {}
for s in range(3):
    free = p.q_learning_env(p.TradingEnv(tr, cost=0.0, episode_len=250), 6, episodes=300, seed=s)
    aware = p.q_learning_env(p.TradingEnv(tr, cost=0.003, episode_len=250), 6, episodes=300, seed=s)
    rows[f"seed {s}"] = {"trained with free trades": p.run_policy(costly_test, greedy(free)),
                         "trained with 30 bp costs": p.run_policy(costly_test, greedy(aware))}
print(f"momentum rule at 30 bp: {p.run_policy(costly_test, rules['momentum']):.2f}")
pd.DataFrame(rows).T.round(3)"""),
    ("md", "Agents trained without costs trade every day and lose at real costs. Agents trained with costs learn that the edge "
           "doesn't pay for them and stay flat, which is the right answer: even the momentum rule loses at 30 bp.\n\n"
           "## 3. Optimal execution\n\n"
           "Selling `X` shares over `N` periods: fast selling costs market impact, slow selling risks the price moving. Almgren & "
           "Chriss (2000) minimize `E[cost] + λ·Var[cost]`. With `η̃ = η − γ/2` and `κ` from `cosh κ = 1 + λσ²/(2η̃)`, the holdings "
           "after `j` trades are `x_j = X·sinh(κ(N − j))/sinh(κN)`; the trades are the differences. As `λ → 0` it becomes TWAP."),
    ("md", YOUR_TURN),
    ("ex", _AC_HEAD + """    kappa = ...                                                      # ✍️ np.arccosh(...)
""" + _AC_MID + """    holdings = ...                                                   # ✍️ x_j for j = 0..N
""" + _AC_TAIL + """
mine = [p.attempt(almgren_chriss_schedule, *c) for c in cases]
mine = p.check("almgren_chriss_schedule", mine, [p.almgren_chriss_schedule(*c) for c in cases])
ac = mine[0]""",
     _AC_HEAD + """    kappa = np.arccosh(1 + lam * sigma ** 2 / (2 * eta_t))
""" + _AC_MID + """    holdings = X * np.sinh(kappa * (N - j)) / np.sinh(kappa * N)
""" + _AC_TAIL + """
mine = [p.attempt(almgren_chriss_schedule, *c) for c in cases]
mine = p.check("almgren_chriss_schedule", mine, [p.almgren_chriss_schedule(*c) for c in cases])
ac = mine[0]"""),
    ("code", """twap = p.twap_schedule(X_SH, N)
(e_tw, v_tw), (e_ac, v_ac) = p.cost_moments(twap, SIGMA, ETA, GAMMA), p.cost_moments(ac, SIGMA, ETA, GAMMA)
print(f"TWAP: expected cost {e_tw:.0f}, std {np.sqrt(v_tw):.0f};  Almgren–Chriss (λ = 1e-3): expected cost {e_ac:.0f}, std {np.sqrt(v_ac):.0f}")
print(f"→ {e_ac / e_tw - 1:+.1%} expected cost for {v_ac / v_tw - 1:+.1%} variance")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].bar(np.arange(N) - 0.2, twap, 0.4, label="TWAP"); axes[0].bar(np.arange(N) + 0.2, ac, 0.4, label="Almgren–Chriss")
axes[0].legend(); axes[0].set_title("shares sold per period")
lams = np.geomspace(1e-6, 1e-1, 30)
frontier = np.array([p.cost_moments(p.almgren_chriss_schedule(X_SH, N, SIGMA, ETA, GAMMA, lam), SIGMA, ETA, GAMMA) for lam in lams])
axes[1].plot(np.sqrt(frontier[:, 1]), frontier[:, 0], marker=".")
axes[1].scatter([np.sqrt(v_tw)], [e_tw], color="#eb6834", zorder=3, label="TWAP")
axes[1].set_xlabel("std of cost"); axes[1].set_ylabel("expected cost"); axes[1].legend(); axes[1].set_title("the efficient frontier of schedules")
plt.tight_layout(); plt.show()"""),
    ("md", "Almgren–Chriss front-loads the selling: it pays a little more expected cost to cut the variance by about a third. The "
           "frontier is the benchmark any RL execution agent must beat on its own objective, and a hard one to beat, because it "
           "is optimal for this model.\n\n"
           "## 4. The execution environment\n\n"
           "`p.ExecutionEnv` simulates the same model with random prices (action = the fraction of the remaining shares to sell "
           "now; total reward = −implementation shortfall). Run each schedule on 500 paths:"),
    ("code", """env = p.ExecutionEnv(X_SH, N, SIGMA, ETA, GAMMA)
sims = {"TWAP": p.simulate_execution(env, p.schedule_policy(twap)), "Almgren–Chriss": p.simulate_execution(env, p.schedule_policy(ac)),
        "wait, dump at the end": p.simulate_execution(env, lambda obs, k: 0.0, 200),
        "dump everything now": p.simulate_execution(env, lambda obs, k: 1.0, 200)}
pd.DataFrame({k: {"mean shortfall": v.mean(), "std": v.std()} for k, v in sims.items()}).T.round(0)"""),
    ("md", "The simulation agrees with the formulas within sampling error, and the naive extremes cost ten times more. An RL "
           "execution agent would be trained in this environment, and it passes only if it lands on (or beyond) the frontier "
           "above, across seeds.\n\n"
           "## S24 · Tuning RL and the M7a release\n\n"
           "Tuning an RL agent (learning rate, γ, ε schedule, state design, reward) is a search like any other: every configuration "
           "is a trial for the Deflated Sharpe Ratio, and each must be judged across seeds and regimes, as above. The milestone "
           "M7a release requires: the environment's timing and cost tests, the rule baselines, the seed/regime/cost table, and a "
           "clear statement of whether the agent beats the best rule. Clinic W6 in the labs builds that defense (optionally with "
           "PPO from Stable-Baselines3).\n\n"
           "## Wrap-up\n\n"
           "* An RL agent is credible only across seeds, regimes and costs, next to simple rules.\n"
           "* Train with realistic costs; free trades teach over-trading.\n"
           "* In execution, Almgren–Chriss defines the frontier an agent must reach.\n"
           "* Graded version: `labs/part10/week38_rl` (`q_learning_env`, `almgren_chriss_schedule`, `cost_moments`, "
           "`ExecutionEnv`) and Clinic W6 (the RL defense)."),
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
