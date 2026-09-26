"""Build the Part 5 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part05:  python tools/build_notebooks.py
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
for d in (Path.cwd(), Path.cwd().parent):       # p5lib.py is in notebooks/part05/
    sys.path.insert(0, str(d))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p5lib as p

p.use_course_style()"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 5 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_05_ANALYTICS_LIBRARY.md) · graded labs in [`labs/part05/`](../../labs/part05/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.
All data is synthetic with a known structure, so you always know which effects are real and which are luck.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
NB["01_core_indicators"] = [
    header("01", "Core indicators, done right", "S1 (Library architecture & helpers) · S2 (Core indicators)",
           "1. See how a registry turns scattered functions into a documented, testable library.\n"
           "2. Write an EMA with TA-Lib's seeding, and see why pandas gives different early values.\n"
           "3. Write Wilder's RSI smoothing and the True Range behind ATR.\n"
           "4. Run the library contract (length, warm-up, no look-ahead) and catch a leaky indicator."),
    ("code", SETUP),
    ("md", "## 1. A registry, not a folder of functions\n\n"
           "Every indicator in the library is registered with a decorator that records its **group** and its **look-back** "
           "(how many leading values are NaN). The catalog, the docs and the tests are all generated from this registry."),
    ("code", """@p.indicator("wma", "trend", lambda n=10: n - 1)
def wma(x, n=10):
    \"\"\"Linearly weighted moving average. Look-back n−1.\"\"\"
    w = np.arange(1, n + 1)
    return pd.Series(x).rolling(n).apply(lambda v: (v * w).sum() / w.sum(), raw=True).to_numpy()

pd.DataFrame([{"name": k, "group": v["group"], "look-back (defaults)": v["lookback"](), "doc": v["doc"]}
              for k, v in p.REGISTRY.items()])"""),
    ("md", "## 2. The EMA and its seed\n\n"
           "`EMA[t] = α·x[t] + (1 − α)·EMA[t−1]` with `α = 2/(n+1)`. The recursion needs a starting value, and **the seed is where "
           "libraries disagree**. TA-Lib (and our library) starts at bar `n−1` with the **simple average of the first `n` values**, "
           "and reports NaN before that (look-back `n−1`)."),
    ("md", YOUR_TURN),
    ("ex", """def ema(x, n=20):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.size < n:
        return out
    a = 2.0 / (n + 1)
    out[n - 1] = ...                              # ✍️ the seed: the mean of the first n values
    for i in range(n, x.size):
        out[i] = ...                              # ✍️ the EMA recursion
    return out

df = p.synthetic_ohlcv(1500, seed=7)
o, h, l, c, v = p.arrays(df)
mine = p.attempt(ema, c, 20)
mine = p.check("ema", mine, p.ema(c, 20))
mine[15:25].round(4)""",
     """def ema(x, n=20):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.size < n:
        return out
    a = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

df = p.synthetic_ohlcv(1500, seed=7)
o, h, l, c, v = p.arrays(df)
mine = p.attempt(ema, c, 20)
mine = p.check("ema", mine, p.ema(c, 20))
mine[15:25].round(4)"""),
    ("md", "`pandas.ewm(span=n, adjust=False)` seeds with the **first value** instead and has no NaN warm-up. The two agree eventually, "
           "because the seed's weight decays as `(1 − α)^t`, but not at the start. This is the cause of most \"my RSI differs from TradingView\" questions."),
    ("code", """gap = np.abs(p.ema(c, 20) - p.ema_pandas(c, 20))
fig, ax = plt.subplots()
ax.semilogy(np.arange(len(c)), gap)
ax.axhline(1e-8, color=p.PALETTE[7], ls="--", lw=1, label="golden-test tolerance 1e-8")
ax.set(xlim=(0, 300), xlabel="bar", ylabel="|TA-Lib seed − pandas seed|", title="EMA(20): the seeding gap decays, slowly")
ax.legend(); plt.show()
first_ok = int(np.argmax(gap < 1e-8))
print(f"the two conventions differ by more than 1e-8 until bar {first_ok}")"""),
    ("md", "## 3. Wilder's RSI\n\n"
           "RSI smooths average gains and losses with **Wilder's** method, which is an EMA with `α = 1/n`: "
           "`avg = (avg·(n−1) + new) / n`. The averages are seeded with the plain mean of the first `n` changes, so the first RSI is at bar `n`. "
           "`gain` and `loss` below hold the price changes (`np.diff`), so the change *into* bar `i` is at position `i − 1`."),
    ("md", YOUR_TURN),
    ("ex", """def rsi(close, n=14):
    out = np.full(close.shape, np.nan)
    d = np.diff(close)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[:n].mean(), loss[:n].mean()
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(n + 1, close.size):
        ag = ...                                  # ✍️ Wilder-smooth the average gain with gain[i - 1]
        al = ...                                  # ✍️ and the average loss with loss[i - 1]
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out

mine = p.attempt(rsi, c, 14)
mine = p.check("rsi", mine, p.rsi(c, 14))
mine[12:20].round(3)""",
     """def rsi(close, n=14):
    out = np.full(close.shape, np.nan)
    d = np.diff(close)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[:n].mean(), loss[:n].mean()
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(n + 1, close.size):
        ag = (ag * (n - 1) + gain[i - 1]) / n
        al = (al * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out

mine = p.attempt(rsi, c, 14)
mine = p.check("rsi", mine, p.rsi(c, 14))
mine[12:20].round(3)"""),
    ("md", "## 4. True Range and ATR\n\n"
           "The True Range includes the overnight gap: `max(H − L, |H − C₋₁|, |L − C₋₁|)`. Bar 0 has no previous close, so its TR is NaN. "
           "ATR is the Wilder average of TR (look-back `n`)."),
    ("md", YOUR_TURN),
    ("ex", """def true_range(high, low, close):
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = ...                                      # ✍️ the largest of the three distances, bar by bar
    tr[0] = np.nan
    return tr

mine = p.attempt(true_range, h, l, c)
mine = p.check("true_range", mine, p.true_range(h, l, c))
gap_share = np.nanmean(p.true_range(h, l, c) > (h - l) + 1e-12)
print(f"on {gap_share:.0%} of bars the gap makes the true range larger than the bar's own range")""",
     """def true_range(high, low, close):
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = np.nan
    return tr

mine = p.attempt(true_range, h, l, c)
mine = p.check("true_range", mine, p.true_range(h, l, c))
gap_share = np.nanmean(p.true_range(h, l, c) > (h - l) + 1e-12)
print(f"on {gap_share:.0%} of bars the gap makes the true range larger than the bar's own range")"""),
    ("code", """fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axes[0].plot(df.index, c, lw=1, label="close"); axes[0].plot(df.index, p.ema(c, 50), label="EMA 50"); axes[0].legend()
axes[1].plot(df.index, p.rsi(c, 14), lw=1, color=p.PALETTE[1]); axes[1].axhline(70, ls="--", lw=1); axes[1].axhline(30, ls="--", lw=1)
axes[1].set_ylabel("RSI 14")
axes[2].plot(df.index, p.atr(h, l, c, 14) / c * 100, lw=1, color=p.PALETTE[2]); axes[2].set_ylabel("ATR 14, % of price")
axes[0].set_title("Core indicators on synthetic bars (note the volatility clusters in ATR)")
plt.tight_layout(); plt.show()"""),
    ("md", "## 5. The library contract\n\n"
           "Every indicator must keep the **same length** as its input, be NaN **exactly** during its look-back, and have **no look-ahead**: "
           "values up to bar `t` must not change when the data after `t` is deleted. `p.audit` checks all three. "
           "Here it runs over the registry, and over a centered moving average that looks perfectly reasonable on a chart."),
    ("code", """rows = {name: p.audit(lambda x, f=spec["fn"]: f(x), c, spec["lookback"]()) for name, spec in p.REGISTRY.items() if name != "atr"}
rows["centered MA (leaky)"] = p.audit(p.leaky_smooth, c, 2)
pd.DataFrame(rows).T.replace({True: "✔", False: "✘"})"""),
    ("md", "A centered average uses `n//2` future bars: every backtest that uses it is fiction. The truncation test catches it in one line.\n\n"
           "## Wrap-up\n\n"
           "* Register every indicator with its look-back; generate the catalog and the tests from the registry.\n"
           "* Know your seeding convention and document it; golden tests compare from bar 0.\n"
           "* Run the contract on everything: length, warm-up, truncation.\n"
           "* Graded version: `labs/part05/week17_core` (10 indicators matched to TA-Lib at 1e-8 with golden files)."),
]

# ---------------------------------------------------------------------------------------------- 02
NB["02_streaming_indicators"] = [
    header("02", "Streaming indicators", "S3 (Streaming, incremental indicators)",
           "1. Turn a vectorized SMA and EMA into O(1) streaming updates.\n"
           "2. Keep a rolling maximum with a monotonic deque.\n"
           "3. Prove streaming equals vectorized on every prefix, with a property test.\n"
           "4. See the cost of recomputing, and the bug of updating on every tick."),
    ("code", SETUP + "\nfrom collections import deque\nimport time"),
    ("md", "## 1. Why streaming\n\n"
           "Live, a strategy gets one new bar at a time. Recomputing an indicator over the whole history on every bar costs O(n) per bar, "
           "O(n²) per day. A **streaming** indicator keeps a little state and updates in O(1). The contract: its output equals the vectorized "
           "version on **every prefix** of the data."),
    ("code", """df = p.synthetic_ohlcv(3000, seed=1)
o, h, l, c, v = p.arrays(df)

t0 = time.perf_counter()
recomputed = [p.ema(c[: i + 1], 20)[-1] for i in range(len(c))]
t_re = time.perf_counter() - t0
t0 = time.perf_counter()
streamed = p.stream(p.StreamingEMA(20), c)
t_st = time.perf_counter() - t0
print(f"recompute every bar: {t_re:.2f} s   streaming: {t_st * 1000:.1f} ms   ({t_re / t_st:,.0f}× faster)")
print("same values:", np.allclose(recomputed, streamed, equal_nan=True))"""),
    ("md", "## 2. A streaming SMA\n\nKeep the last `n` values in a `deque` and a running total: add the new value, and once there are more than `n`, "
           "remove the oldest from both. Return `None` until the window is full."),
    ("md", YOUR_TURN),
    ("ex", """class MyStreamingSMA:
    def __init__(self, n):
        self.n, self.buf, self.total = n, deque(), 0.0

    def update(self, x):
        self.buf.append(x)
        self.total += x
        if len(self.buf) > self.n:
            ...                                   # ✍️ remove the oldest value from buf and from the total
        return self.total / self.n if len(self.buf) == self.n else None

mine = p.stream(MyStreamingSMA(20), c)
mine = p.check("streaming SMA", mine, p.sma(c, 20))
mine[17:23].round(4)""",
     """class MyStreamingSMA:
    def __init__(self, n):
        self.n, self.buf, self.total = n, deque(), 0.0

    def update(self, x):
        self.buf.append(x)
        self.total += x
        if len(self.buf) > self.n:
            self.total -= self.buf.popleft()
        return self.total / self.n if len(self.buf) == self.n else None

mine = p.stream(MyStreamingSMA(20), c)
mine = p.check("streaming SMA", mine, p.sma(c, 20))
mine[17:23].round(4)"""),
    ("md", "## 3. A streaming EMA\n\nCollect the first `n` values to compute the SMA seed; after that, apply the recursion to the stored value. "
           "Return `None` until the seed exists (so the output matches `p.ema`, NaN for the first `n − 1` bars)."),
    ("md", YOUR_TURN),
    ("ex", """class MyStreamingEMA:
    def __init__(self, n):
        self.n, self.alpha, self._seed, self.value = n, 2.0 / (n + 1), [], None

    def update(self, x):
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) == self.n:
                self.value = ...                  # ✍️ the SMA seed
        else:
            self.value = ...                      # ✍️ the recursion
        return self.value

mine = p.attempt(p.stream, MyStreamingEMA(20), c)
mine = p.check("streaming EMA", mine, p.ema(c, 20))""",
     """class MyStreamingEMA:
    def __init__(self, n):
        self.n, self.alpha, self._seed, self.value = n, 2.0 / (n + 1), [], None

    def update(self, x):
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) == self.n:
                self.value = sum(self._seed) / self.n
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

mine = p.attempt(p.stream, MyStreamingEMA(20), c)
mine = p.check("streaming EMA", mine, p.ema(c, 20))"""),
    ("md", "## 4. Rolling maximum in O(1): the monotonic deque\n\n"
           "Donchian channels and the Stochastic need the max of the last `n` highs. Rescanning the window is O(n) per bar. "
           "Instead keep a deque of `(index, value)` with **decreasing values**:\n"
           "* before appending `x`, pop from the back every value `<= x` (they can never be the max while `x` is in the window);\n"
           "* after appending, pop the front if its index has left the window (`index <= i − n`);\n"
           "* the front is the max."),
    ("md", YOUR_TURN),
    ("ex", """class MyRollingMax:
    def __init__(self, n):
        self.n, self.i, self.q = n, -1, deque()

    def update(self, x):
        self.i += 1
        while ...:                                # ✍️ the back of the queue holds a value <= x
            self.q.pop()
        self.q.append((self.i, x))
        if ...:                                   # ✍️ the front's index has left the window
            self.q.popleft()
        return self.q[0][1] if self.i >= self.n - 1 else None

mine = p.attempt(p.stream, MyRollingMax(20), h)
mine = p.check("rolling max", mine, pd.Series(h).rolling(20).max().to_numpy())""",
     """class MyRollingMax:
    def __init__(self, n):
        self.n, self.i, self.q = n, -1, deque()

    def update(self, x):
        self.i += 1
        while self.q and self.q[-1][1] <= x:
            self.q.pop()
        self.q.append((self.i, x))
        if self.q[0][0] <= self.i - self.n:
            self.q.popleft()
        return self.q[0][1] if self.i >= self.n - 1 else None

mine = p.attempt(p.stream, MyRollingMax(20), h)
mine = p.check("rolling max", mine, pd.Series(h).rolling(20).max().to_numpy())"""),
    ("md", "## 5. Prove it for every input: a property test\n\n"
           "Examples check the data you picked. Hypothesis generates hundreds of arrays, including nasty ones (constant runs, huge jumps), "
           "and checks the streaming rolling standard deviation (Welford-style add/remove updates) against pandas on every prefix.\n\n"
           "The first time we ran this with a fixed tolerance of 1e-6, Hypothesis found a failure within seconds, and shrank it to the "
           "simplest case: one huge value followed by a flat run."),
    ("code", """spike = np.array([3045.0] + [2.0] * 24)
got = p.stream(p.StreamingStd(20), spike)[19:23]
want = pd.Series(spike).rolling(20).std(ddof=0).to_numpy()[19:23]
print("streaming:", got)
print("pandas:   ", want)"""),
    ("md", "When the spike leaves the window, the running sum of squares should drop back to exactly 0, but subtracting a number of about "
           "9·10⁶ leaves rounding error behind, and the square root magnifies it near zero variance: 1e-5 on data of size 3·10³, a relative "
           "error of 3·10⁻⁹. That is floating point, not a logic bug. The test must say so explicitly: **the tolerance scales with the data**. "
           "(Production code also clamps a tiny negative variance to 0, or re-syncs from the buffer every `n` bars.)"),
    ("code", """from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

@settings(max_examples=200, deadline=None)
@given(arrays(np.float64, st.integers(25, 300), elements=st.floats(1, 1e4)))
def test_streaming_std_matches_pandas(x):
    got = p.stream(p.StreamingStd(20), x)
    want = pd.Series(x).rolling(20).std(ddof=0).to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6 * np.abs(x).max(), equal_nan=True)

test_streaming_std_matches_pandas()
print("✔ 200 random arrays: streaming std == pandas rolling std on every prefix, within a scale-aware tolerance")"""),
    ("md", "## 6. `update` on the close, `peek` in between\n\n"
           "A live chart wants the indicator value of the **forming** bar on every tick. Calling `update` on each tick feeds the same bar "
           "into the state several times. `peek(price)` computes the value without changing the state; `update` runs once, on the bar close."),
    ("code", """rng = np.random.default_rng(0)
closes = c[:300]
wrong, right = p.StreamingEMA(20), p.StreamingEMA(20)
for px in closes:
    ticks = px * (1 + rng.normal(0, 0.002, 4))              # four intrabar ticks, then the close
    for t in ticks:
        wrong.update(t)                                     # the bug: every tick mutates the state
        right.peek(t)                                       # display only
    wrong.update(px)
    right.update(px)
ref = p.ema(closes, 20)[-1]
print(f"vectorized EMA of the closes: {ref:.4f}")
print(f"update on the close only:     {right.value:.4f}")
print(f"update on every tick:         {wrong.value:.4f}  ← a different indicator (an EMA of ticks), not the one you backtested")"""),
    ("md", "## Wrap-up\n\n"
           "* Streaming indicators: running sums, recursions, monotonic deques, Welford updates, all O(1) per bar.\n"
           "* The equivalence test on every prefix is what lets you trust live values to match the backtest.\n"
           "* `update` once per closed bar; `peek` for intrabar displays.\n"
           "* Graded version: `labs/part05/week17_core` (`StreamingEMA`, `StreamingRSI`, `RollingMax`, `StreamingBollinger` with Hypothesis tests)."),
]

# ---------------------------------------------------------------------------------------------- 03
NB["03_candles_and_edge"] = [
    header("03", "Candlesticks, and does the pattern work?", "S4 (Candlestick patterns) · S5 (Edge testing)",
           "1. Compute candle anatomy safely, flat bars included.\n"
           "2. Write the bullish engulfing rule and see why a hammer needs trend context.\n"
           "3. Measure forward returns the honest way: entry at the next open.\n"
           "4. Test 600 patterns at once and control false discoveries with Benjamini–Hochberg."),
    ("code", SETUP + "\nfrom scipy import stats"),
    ("md", "## 1. Candle anatomy\n\n"
           "Every candlestick rule is built from five numbers per bar: the **body** `|C − O|`, the **range** `H − L`, the **upper shadow** "
           "`H − max(O, C)`, the **lower shadow** `min(O, C) − L`, and **body %** `body / range`. A flat bar (H = L, common in thin markets and "
           "pre-open data) has range 0: body % must be 0 there, not `inf` or NaN. `np.divide(a, b, out=np.zeros_like(a), where=b > 0)` does that."),
    ("md", YOUR_TURN),
    ("ex", """def anatomy(o, h, l, c):
    body, rng_ = np.abs(c - o), h - l
    return {"body": body, "range": rng_,
            "upper": ...,                         # ✍️ H − max(O, C)
            "lower": ...,                         # ✍️ min(O, C) − L
            "body_pct": ...}                      # ✍️ body / range, 0 where the range is 0

df = p.synthetic_ohlcv(1500, seed=7)
o, h, l, c, v = p.arrays(df)
o2, h2, l2, c2 = (np.append(a, 50.0) for a in (o, h, l, c))       # one flat bar at the end
mine = anatomy(o2, h2, l2, c2)
mine = p.check("anatomy", mine, p.anatomy(o2, h2, l2, c2))
pd.DataFrame(mine).tail(3)""",
     """def anatomy(o, h, l, c):
    body, rng_ = np.abs(c - o), h - l
    return {"body": body, "range": rng_,
            "upper": h - np.maximum(o, c),
            "lower": np.minimum(o, c) - l,
            "body_pct": np.divide(body, rng_, out=np.zeros_like(body), where=rng_ > 0)}

df = p.synthetic_ohlcv(1500, seed=7)
o, h, l, c, v = p.arrays(df)
o2, h2, l2, c2 = (np.append(a, 50.0) for a in (o, h, l, c))       # one flat bar at the end
mine = anatomy(o2, h2, l2, c2)
mine = p.check("anatomy", mine, p.anatomy(o2, h2, l2, c2))
pd.DataFrame(mine).tail(3)"""),
    ("md", "## 2. Bullish engulfing\n\n"
           "Bar `t−1` is bearish (`C < O`), bar `t` is bullish, and bar `t`'s body **covers** the previous body: `O_t <= C_{t−1}` and "
           "`C_t >= O_{t−1}`, with a strictly larger body. Output convention: a boolean per bar, `False` at bar 0."),
    ("md", YOUR_TURN),
    ("ex", """def bullish_engulfing(o, h, l, c):
    out = np.zeros(c.shape, dtype=bool)
    po, pc = o[:-1], c[:-1]                       # previous bar
    co, cc = o[1:], c[1:]                         # current bar
    out[1:] = (pc < po) & (cc > co) & ...         # ✍️ covers the previous body, and is larger
    return out

mine = p.attempt(bullish_engulfing, o, h, l, c)
mine = p.check("bullish_engulfing", mine, p.bullish_engulfing(o, h, l, c))
print(f"{int(np.sum(mine))} signals in {len(c)} bars")""",
     """def bullish_engulfing(o, h, l, c):
    out = np.zeros(c.shape, dtype=bool)
    po, pc = o[:-1], c[:-1]                       # previous bar
    co, cc = o[1:], c[1:]                         # current bar
    out[1:] = (pc < po) & (cc > co) & (co <= pc) & (cc >= po) & ((cc - co) > (po - pc))
    return out

mine = p.attempt(bullish_engulfing, o, h, l, c)
mine = p.check("bullish_engulfing", mine, p.bullish_engulfing(o, h, l, c))
print(f"{int(np.sum(mine))} signals in {len(c)} bars")"""),
    ("md", "**Context matters.** A hammer (long lower shadow, tiny upper shadow) is a reversal pattern only **after a decline**. "
           "The same shape after a rise is called a *hanging man* and means something else. Two hand-made sequences:"),
    ("code", """def seq(trend):
    base = 100 + trend * np.arange(12)
    o_ = base + 0.1; c_ = base - 0.1 if trend < 0 else base + 0.3
    h_, l_ = np.maximum(o_, c_) + 0.2, np.minimum(o_, c_) - 0.2
    o_[-1], c_[-1] = base[-1] + 0.05, base[-1] + 0.25; h_[-1] = c_[-1] + 0.01; l_[-1] = o_[-1] - 1.5   # the hammer shape
    return o_, h_, l_, c_

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, (name, trend) in zip(axes, [("after a decline", -0.5), ("after a rise", 0.5)]):
    o_, h_, l_, c_ = seq(trend)
    for i in range(12):
        col = p.PALETTE[2] if c_[i] >= o_[i] else p.PALETTE[7]
        ax.vlines(i, l_[i], h_[i], color=col, lw=1); ax.vlines(i, min(o_[i], c_[i]), max(o_[i], c_[i]), color=col, lw=6)
    flagged = p.hammer(o_, h_, l_, c_)[-1]
    ax.set(title=f"{name}: hammer → {flagged}", xticks=[])
plt.tight_layout(); plt.show()"""),
    ("md", "## 3. Forward returns, honestly\n\n"
           "A pattern is known only when bar `t` **closes**. The earliest fill is the **next bar's open**. The return of a trade held "
           "`h` bars is `O[t+1+h] / O[t+1] − 1`; where the exit is beyond the data, NaN."),
    ("md", YOUR_TURN),
    ("ex", """def forward_returns(open_, horizon):
    n = open_.size
    out = np.full(open_.shape, np.nan)
    out[: n - horizon - 1] = ...                  # ✍️ O[t+1+h] / O[t+1] − 1 for every t that has an exit
    return out

mine = p.attempt(forward_returns, o, 5)
mine = p.check("forward_returns", mine, p.forward_returns(o, 5))
mine[-8:]""",
     """def forward_returns(open_, horizon):
    n = open_.size
    out = np.full(open_.shape, np.nan)
    out[: n - horizon - 1] = open_[1 + horizon:] / open_[1:n - horizon] - 1
    return out

mine = p.attempt(forward_returns, o, 5)
mine = p.check("forward_returns", mine, p.forward_returns(o, 5))
mine[-8:]"""),
    ("code", """fwd = p.forward_returns(o, 5)
eng = p.bullish_engulfing(o, h, l, c)
mean_after, pval = p.permutation_pvalue(eng, fwd, n_perm=2000)
print(f"engulfing: {eng.sum()} signals, mean 5-bar return after {mean_after:+.3%} vs {np.nanmean(fwd):+.3%} unconditional")
print(f"permutation p-value {pval:.3f}: on this synthetic symbol, with no planted edge, nothing to see")"""),
    ("md", "## 4. 600 tests, and the ones that look significant\n\n"
           "Now the trap. Take 20 symbols and 30 patterns that are pure coin flips (each bar signals with probability 5%). Add one pattern "
           "that **cheats**: it signals when the next two opens rise, which it can't know at the close. Test each (pattern, symbol) pair with a "
           "one-sided t-test of the forward returns after the signal against the rest. At 5%, about 30 of the 600 honest tests pass by luck."),
    ("code", """uni = p.universe(20, 1000)
rng = np.random.default_rng(42)
rows = []
for sym, d in uni.items():
    o_, h_, l_, c_, _ = p.arrays(d)
    f = p.forward_returns(o_, 5)
    patterns = {f"coin{j:02d}": rng.random(len(c_)) < 0.05 for j in range(30)}
    patterns["cheat"] = np.r_[o_[2:] > o_[1:-1] * 1.004, [False, False]]
    for name, sig in patterns.items():
        ok = ~np.isnan(f)
        a, b = f[sig & ok], f[~sig & ok]
        rows.append({"pattern": name, "symbol": sym, "n": len(a), "mean": a.mean(),
                     "p": stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue})
tests = pd.DataFrame(rows)
honest = tests["pattern"] != "cheat"
print(f"honest tests with p < 0.05: {(tests.loc[honest, 'p'] < 0.05).sum()} of {honest.sum()}")"""),
    ("md", "**Benjamini–Hochberg** controls the *false discovery rate*: among what you call discoveries, the expected share of flukes stays "
           "below `α`. Sort the p-values; the adjusted value of the `i`-th smallest is `p_(i) · m / i`; then make them non-decreasing from the top "
           "(a running minimum from the largest down) and cap at 1. Return them in the **input order**."),
    ("md", YOUR_TURN),
    ("ex", """def bh_adjust(p_values):
    pv = np.asarray(p_values, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    q = ...                                       # ✍️ the sorted p-values × m / rank, rank = 1..m
    q = np.minimum.accumulate(q[::-1])[::-1]      # non-decreasing from the top
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out

mine = p.attempt(bh_adjust, tests["p"].to_numpy())
mine = p.check("bh_adjust", mine, p.bh_adjust(tests["p"].to_numpy()))
tests["q"] = mine
print(f"discoveries at FDR 5%: honest {(tests.loc[honest, 'q'] < 0.05).sum()}, cheat {(tests.loc[~honest, 'q'] < 0.05).sum()} of 20")""",
     """def bh_adjust(p_values):
    pv = np.asarray(p_values, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    q = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]      # non-decreasing from the top
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out

mine = p.attempt(bh_adjust, tests["p"].to_numpy())
mine = p.check("bh_adjust", mine, p.bh_adjust(tests["p"].to_numpy()))
tests["q"] = mine
print(f"discoveries at FDR 5%: honest {(tests.loc[honest, 'q'] < 0.05).sum()}, cheat {(tests.loc[~honest, 'q'] < 0.05).sum()} of 20")"""),
    ("code", """fig, ax = plt.subplots()
ax.hist(tests.loc[honest, "p"], bins=20, alpha=0.8, label="30 coin-flip patterns × 20 symbols")
ax.hist(tests.loc[~honest, "p"], bins=20, alpha=0.8, label="the cheating pattern")
ax.axvline(0.05, color=p.PALETTE[7], ls="--", lw=1)
ax.set(xlabel="p-value", ylabel="tests", title="Honest p-values are uniform; a leak piles up at 0"); ax.legend(); plt.show()"""),
    ("md", "BH removes the lucky coin flips and keeps the cheat. That's the other lesson: a pattern that survives every statistical test "
           "can still be a bug. **A result that looks too good is a look-ahead until proven otherwise.**\n\n"
           "## Wrap-up\n\n"
           "* Normalize candle rules and guard the flat bar; add trend context.\n"
           "* Enter at the next open; compare with the unconditional distribution.\n"
           "* Correct for the number of tests you ran (BH), then confirm out of sample.\n"
           "* Graded versions: `labs/part05/week17_core` (patterns), `week18_groups` (`pattern_edge`, `bh_adjust`), Clinic W2 edge study."),
]

# ---------------------------------------------------------------------------------------------- 04
NB["04_volatility_and_trend"] = [
    header("04", "Volatility estimators and adaptive trend", "S6 (Direction group) · S7 (Volatility & momentum groups)",
           "1. Estimate volatility from the high–low range (Parkinson, Garman–Klass).\n"
           "2. Measure how much more precise range estimators are, on bars with a known σ.\n"
           "3. Write Kaufman's efficiency ratio, the engine of the adaptive moving average.\n"
           "4. See KAMA follow a trend and sit still in chop."),
    ("code", SETUP),
    ("md", "## 1. Bars with a known volatility\n\n"
           "To test an estimator you need the right answer. `p.brownian_bars` simulates each day as 390 one-minute steps of a random walk with a "
           "daily σ of **1%**, and records the open, high, low and close. No drift, no overnight gap: the textbook world these estimators assume."),
    ("code", """bars = p.brownian_bars(500, sigma=0.01, seed=0)
o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
print(f"close-to-close estimate over 500 days: {p.close_to_close_vol(c):.4%} (true 1.0000%)")
bars.head()"""),
    ("md", "## 2. Range estimators\n\n"
           "The close-to-close estimator uses one number per day. The range uses the path inside the day:\n\n"
           "* **Parkinson:** `σ² = mean(ln(H/L)²) / (4 ln 2)`\n"
           "* **Garman–Klass:** `σ² = mean(½ ln(H/L)² − (2 ln 2 − 1) ln(C/O)²)`\n\n"
           "Return the daily σ (the square root)."),
    ("md", YOUR_TURN),
    ("ex", """def parkinson_vol(h, l):
    return ...                                    # ✍️

def garman_klass_vol(o, h, l, c):
    return ...                                    # ✍️

mine = [parkinson_vol(h, l), garman_klass_vol(o, h, l, c)]
mine = p.check("range estimators", mine, [p.parkinson_vol(h, l), p.garman_klass_vol(o, h, l, c)])
[f"{x:.4%}" for x in mine]""",
     """def parkinson_vol(h, l):
    return float(np.sqrt(np.mean(np.log(h / l) ** 2) / (4 * np.log(2))))

def garman_klass_vol(o, h, l, c):
    return float(np.sqrt(np.mean(0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2)))

mine = [parkinson_vol(h, l), garman_klass_vol(o, h, l, c)]
mine = p.check("range estimators", mine, [p.parkinson_vol(h, l), p.garman_klass_vol(o, h, l, c)])
[f"{x:.4%}" for x in mine]"""),
    ("md", "Both come out slightly **below** 1%: we only see 390 prices a day, so the observed high and low miss the true extremes between them. "
           "Real data has the same bias, plus gaps and drift (Yang–Zhang, in the lab, handles those). The prize is **precision**: "
           "over short windows the range estimators wobble much less."),
    ("code", """est = {"close-to-close": [], "Parkinson": [], "Garman–Klass": []}
for s in range(300):
    b = p.brownian_bars(20, sigma=0.01, seed=100 + s)
    o_, h_, l_, c_ = (b[k].to_numpy() for k in ("open", "high", "low", "close"))
    est["close-to-close"].append(np.sqrt(np.mean(np.log(c_ / o_) ** 2)))   # one day's return is open→close here
    est["Parkinson"].append(p.parkinson_vol(h_, l_))
    est["Garman–Klass"].append(p.garman_klass_vol(o_, h_, l_, c_))
fig, ax = plt.subplots()
for k, vals in est.items():
    ax.hist(np.array(vals) * 100, bins=30, alpha=0.6, label=k)
ax.axvline(1.0, color="black", lw=1)
ax.set(xlabel="estimated daily σ, % (20-day windows)", ylabel="windows", title="Same data, different precision"); ax.legend(); plt.show()
sd = {k: np.std(v) for k, v in est.items()}
for k in ("Parkinson", "Garman–Klass"):
    print(f"{k}: {sd['close-to-close'] ** 2 / sd[k] ** 2:.1f}× as efficient as close-to-close (variance ratio)")"""),
    ("md", "## 3. The efficiency ratio\n\n"
           "Kaufman's **efficiency ratio** asks how much of the path was progress: the net move over `n` bars divided by the sum of the "
           "absolute bar-to-bar moves over the same bars. 1 is a straight line; near 0 is chop. "
           "`ER[t] = |C[t] − C[t−n]| / Σ|ΔC|` over those `n` changes (0 when the sum is 0), NaN for `t < n`."),
    ("code", """x = p.trend_then_chop()
plt.plot(x); plt.title("200 bars of trend, then 200 bars of chop"); plt.show()"""),
    ("md", YOUR_TURN + "\n\nHint: `pd.Series(np.abs(np.diff(close))).rolling(n).sum()` gives the path length ending at each change; "
           "the one that matches `change[0]` (bars 0…n) is at position `n − 1`."),
    ("ex", """def efficiency_ratio(close, n=10):
    out = np.full(close.shape, np.nan)
    change = np.abs(close[n:] - close[:-n])                           # net move, aligned with bars n, n+1, …
    path = ...                                                        # ✍️ the path length over the same n changes
    out[n:] = np.divide(change, path, out=np.zeros_like(change), where=path > 0)
    return out

mine = p.attempt(efficiency_ratio, x, 10)
mine = p.check("efficiency_ratio", mine, p.efficiency_ratio(x, 10))
print(f"mean ER: trend {np.nanmean(mine[:200]):.2f}, chop {np.nanmean(mine[200:]):.2f}")""",
     """def efficiency_ratio(close, n=10):
    out = np.full(close.shape, np.nan)
    change = np.abs(close[n:] - close[:-n])                           # net move, aligned with bars n, n+1, …
    path = pd.Series(np.abs(np.diff(close))).rolling(n).sum().to_numpy()[n - 1:]
    out[n:] = np.divide(change, path, out=np.zeros_like(change), where=path > 0)
    return out

mine = p.attempt(efficiency_ratio, x, 10)
mine = p.check("efficiency_ratio", mine, p.efficiency_ratio(x, 10))
print(f"mean ER: trend {np.nanmean(mine[:200]):.2f}, chop {np.nanmean(mine[200:]):.2f}")"""),
    ("md", "## 4. KAMA: an average that speeds up in trends\n\n"
           "KAMA uses the ER to set its smoothing constant each bar: between a fast EMA (2) in a clean trend and a slow EMA (30) in chop. "
           "Count how often each line changes direction in the chop half: every change is a potential whipsaw trade."),
    ("code", """k, e = p.kama(x, 10), p.ema(x, 10)
fig, ax = plt.subplots()
ax.plot(x, lw=0.8, color="#b5b4ad", label="price"); ax.plot(e, label="EMA 10"); ax.plot(k, label="KAMA 10")
ax.axvline(200, color="black", lw=0.8); ax.set_title("KAMA follows the trend and goes flat in the chop"); ax.legend(); plt.show()
flips = {name: int(np.sum(np.diff(np.sign(np.diff(series[200:]))) != 0)) for name, series in [("EMA 10", e), ("KAMA 10", k)]}
print("direction changes in the chop half:", flips)"""),
    ("md", "## Wrap-up\n\n"
           "* Range-based volatility is several times more efficient than close-to-close: shorter windows, same precision.\n"
           "* Test estimators on data with a known answer before trusting them on real data.\n"
           "* Adaptive averages trade some lag in trends for fewer direction changes (whipsaws) in chop.\n"
           "* Graded version: `labs/part05/week18_groups` (five estimators including Yang–Zhang, KAMA and ADX against TA-Lib, SuperTrend)."),
]

# ---------------------------------------------------------------------------------------------- 05
NB["05_vwap_and_timeframes"] = [
    header("05", "VWAP, volume profile and higher timeframes", "S8 (Price-volume-time group & multi-timeframe)",
           "1. Compute session VWAP that resets every day, and an anchored VWAP.\n"
           "2. Find the point of control of a volume profile.\n"
           "3. Put an hourly value on 5-minute bars without looking ahead.\n"
           "4. Watch the default `resample` leak the future into a backtest."),
    ("code", SETUP),
    ("md", "## 1. Intraday bars\n\n"
           "Five days of 5-minute bars, each **stamped at its close** (09:35 … 16:00 New York), with the usual U-shaped volume."),
    ("code", """bars = p.intraday_bars(days=5)
display(bars.head(3))
bars["volume"].groupby(bars.index.time).mean().plot(title="Average volume by time of day (5-minute bars)"); plt.show()"""),
    ("md", "## 2. Session VWAP\n\n"
           "`VWAP = Σ(price · volume) / Σ(volume)` from the session's first bar, **resetting each day**. "
           "Group by `bars.index.date` and take cumulative sums within each group."),
    ("md", YOUR_TURN),
    ("ex", """def session_vwap(df):
    day = df.index.date
    pv = ...                                      # ✍️ cumulative price × volume within each day
    vol = ...                                     # ✍️ cumulative volume within each day
    return pv / vol

mine = p.attempt(session_vwap, bars)
mine = p.check("session_vwap", mine, p.session_vwap(bars))
mine.iloc[76:80]""",
     """def session_vwap(df):
    day = df.index.date
    pv = (df["close"] * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    return pv / vol

mine = p.attempt(session_vwap, bars)
mine = p.check("session_vwap", mine, p.session_vwap(bars))
mine.iloc[76:80]"""),
    ("code", """low_bar = int(np.argmin(bars["close"].to_numpy()[:200]))
avwap = p.anchored_vwap(bars["close"], bars["volume"], low_bar)
x = np.arange(len(bars))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(x, bars["close"], lw=1, color="#8a8984", label="close")
ax.plot(x, p.session_vwap(bars), label="session VWAP (resets daily)")
ax.plot(x, avwap, label=f"VWAP anchored at bar {low_bar} (the low)")
for d in range(1, 5):
    ax.axvline(d * 78, color="#e6e5e0", lw=1)
ax.set(xlabel="bar", title="Session vs anchored VWAP"); ax.legend(); plt.show()"""),
    ("md", "## 3. Volume profile and the point of control\n\n"
           "A volume profile sums volume by **price** instead of by time. Its busiest price bin is the **point of control** (POC), a level "
           "traders watch as support or resistance. Use `np.histogram(price, bins=bins, weights=volume)` and return the centre of the fullest bin."),
    ("md", YOUR_TURN),
    ("ex", """def point_of_control(price, volume, bins=40):
    hist, edges = np.histogram(price, bins=bins, weights=volume)
    i = ...                                       # ✍️ the index of the bin with the most volume
    return float((edges[i] + edges[i + 1]) / 2)

mine = p.attempt(point_of_control, bars["close"], bars["volume"])
mine = p.check("point_of_control", mine, p.point_of_control(bars["close"], bars["volume"]))
mine""",
     """def point_of_control(price, volume, bins=40):
    hist, edges = np.histogram(price, bins=bins, weights=volume)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)

mine = p.attempt(point_of_control, bars["close"], bars["volume"])
mine = p.check("point_of_control", mine, p.point_of_control(bars["close"], bars["volume"]))
mine"""),
    ("code", """hist, edges = np.histogram(bars["close"], bins=40, weights=bars["volume"])
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh((edges[:-1] + edges[1:]) / 2, hist, height=np.diff(edges) * 0.9)
ax.axhline(mine, color=p.PALETTE[7], ls="--", label=f"POC {mine:.2f}")
ax.set(xlabel="volume", ylabel="price", title="Volume profile, 5 days"); ax.legend(); plt.show()"""),
    ("md", "## 4. Higher timeframes without look-ahead\n\n"
           "A 5-minute strategy often uses an hourly indicator. The hourly bar that covers 10:00–11:00 is **known only at 11:00**. "
           "With close-stamped bars: resample with `closed=\"right\", label=\"right\"` so each hourly bar is stamped when it completes, take "
           "`.last()`, drop the empty hours, and forward-fill onto the 5-minute index."),
    ("md", YOUR_TURN),
    ("ex", """def align_higher_tf(close, rule="1h"):
    h = ...                                       # ✍️ completed higher-timeframe closes, stamped at completion
    return h.reindex(close.index, method="ffill")

mine = p.attempt(align_higher_tf, bars["close"])
mine = p.check("align_higher_tf", mine, p.align_higher_tf(bars["close"]))
pd.DataFrame({"close": bars["close"], "hourly (honest)": mine, "hourly (default resample)": p.align_higher_tf_leaky(bars["close"])}).iloc[4:16].round(2)""",
     """def align_higher_tf(close, rule="1h"):
    h = close.resample(rule, closed="right", label="right").last().dropna()
    return h.reindex(close.index, method="ffill")

mine = p.attempt(align_higher_tf, bars["close"])
mine = p.check("align_higher_tf", mine, p.align_higher_tf(bars["close"]))
pd.DataFrame({"close": bars["close"], "hourly (honest)": mine, "hourly (default resample)": p.align_higher_tf_leaky(bars["close"])}).iloc[4:16].round(2)"""),
    ("md", "Look at the default column: before 10:00 it already shows the **09:55** close, and from 10:00 on it shows the close of **10:55**, "
           "up to fifty-five minutes early. The honest column only changes when an hour completes. Any rule that compares price to that value is trading on the future. A toy rule makes it obvious: "
           "\"buy for one bar when the 5-minute close is below the hourly close\"."),
    ("code", """big = p.intraday_bars(days=120, seed=11)
nxt = big["close"].shift(-1) / big["close"] - 1
for name, htf in [("honest", p.align_higher_tf(big["close"])), ("default resample", p.align_higher_tf_leaky(big["close"]))]:
    sig = big["close"] < htf
    print(f"{name:17s}: mean next-bar return when signalled {nxt[sig].mean() * 1e4:+.2f} bp, otherwise {nxt[~sig].mean() * 1e4:+.2f} bp")"""),
    ("md", "## Wrap-up\n\n"
           "* VWAP resets each session; anchored VWAP starts wherever the story starts.\n"
           "* Stamp bars at their close and align higher timeframes on **completed** bars only.\n"
           "* A basis-point edge from a one-bar rule on random data is a leak, not an edge.\n"
           "* Graded version: `labs/part05/week18_groups` (`session_vwap`, `anchored_vwap`, `volume_profile` with value area, `align_higher_tf`)."),
]

# ---------------------------------------------------------------------------------------------- 06
NB["06_levels_and_swings"] = [
    header("06", "Pivot levels, swings and the confirmation trap", "S9 (Levels & support/resistance) · S10 (Pivots & swings) · S11 (Chart patterns)",
           "1. Compute floor pivot levels for today from yesterday's bar.\n"
           "2. Find swing highs and lows (fractals) and when each becomes known.\n"
           "3. See the biggest look-ahead bug in chart patterns: using the swing bar instead of the confirmation bar.\n"
           "4. Prove a pattern detector is honest with a truncation test."),
    ("code", SETUP),
    ("code", """df = p.synthetic_ohlcv(1500, seed=7)
o, h, l, c, v = p.arrays(df)"""),
    ("md", "## 1. Floor pivots\n\n"
           "Floor traders compute today's levels from **yesterday's** high, low and close: `P = (H + L + C)/3`, `R1 = 2P − L`, `S1 = 2P − H`, "
           "`R2 = P + (H − L)`, `S2 = P − (H − L)`. Shifting by one bar is what keeps them honest; bar 0 has no yesterday, so it is NaN."),
    ("md", YOUR_TURN),
    ("ex", """def classic_pivots(high, low, close):
    h, l, c = (np.concatenate([[np.nan], a[:-1]]) for a in (high, low, close))   # yesterday's values
    pp = ...                                      # ✍️ the pivot
    return {"P": pp, "R1": ..., "S1": ...,        # ✍️ R1 and S1
            "R2": pp + (h - l), "S2": pp - (h - l)}

mine = p.attempt(classic_pivots, h, l, c)
mine = p.check("classic_pivots", mine, p.classic_pivots(h, l, c))
pd.DataFrame(mine).iloc[:4].round(3)""",
     """def classic_pivots(high, low, close):
    h, l, c = (np.concatenate([[np.nan], a[:-1]]) for a in (high, low, close))   # yesterday's values
    pp = (h + l + c) / 3
    return {"P": pp, "R1": 2 * pp - l, "S1": 2 * pp - h,
            "R2": pp + (h - l), "S2": pp - (h - l)}

mine = p.attempt(classic_pivots, h, l, c)
mine = p.check("classic_pivots", mine, p.classic_pivots(h, l, c))
pd.DataFrame(mine).iloc[:4].round(3)"""),
    ("code", """lv = p.classic_pivots(h, l, c)
touch_r1 = np.nanmean(h[1:] >= lv["R1"][1:]); touch_s1 = np.nanmean(l[1:] <= lv["S1"][1:])
print(f"the day's high reaches R1 on {touch_r1:.0%} of days, the low reaches S1 on {touch_s1:.0%}")
print("on a random walk these are just distances of about one range; whether they act as levels is an empirical question")"""),
    ("md", "## 2. Swing points (fractals) and when you know them\n\n"
           "Bar `i` is a **swing high** if its high is *strictly* above the highs of the `k` bars on each side; a **swing low** likewise with lows. "
           "You can only know this after the `k` bars to the right have closed, so each swing is recorded with `known_from = i + k`. "
           "Append `p.Pivot(idx, known_from, price, kind)` with kind `-1` for a low and `+1` for a high (a low before a high at the same bar)."),
    ("md", YOUR_TURN),
    ("ex", """def fractals(high, low, k=2):
    out = []
    for i in range(k, len(high) - k):
        nb = np.r_[i - k:i, i + 1:i + k + 1]      # the k bars on each side
        if ...:                                   # ✍️ low[i] strictly below all the neighbours' lows
            out.append(p.Pivot(i, i + k, float(low[i]), -1))
        if ...:                                   # ✍️ high[i] strictly above all the neighbours' highs
            out.append(p.Pivot(i, i + k, float(high[i]), +1))
    return out

mine = fractals(h, l, 2)
mine = p.check("fractals", mine, p.fractals(h, l, 2))
print(f"{len(mine)} swings; the first three: {mine[:3]}")""",
     """def fractals(high, low, k=2):
    out = []
    for i in range(k, len(high) - k):
        nb = np.r_[i - k:i, i + 1:i + k + 1]      # the k bars on each side
        if (low[i] < low[nb]).all():
            out.append(p.Pivot(i, i + k, float(low[i]), -1))
        if (high[i] > high[nb]).all():
            out.append(p.Pivot(i, i + k, float(high[i]), +1))
    return out

mine = fractals(h, l, 2)
mine = p.check("fractals", mine, p.fractals(h, l, 2))
print(f"{len(mine)} swings; the first three: {mine[:3]}")"""),
    ("md", "## 3. The confirmation trap\n\n"
           "A ZigZag swing needs price to reverse by `pct` before it is confirmed, which can take many bars. Draw the swings on a chart and "
           "they look like perfect turning points, because the chart shows them at the **swing bar**. A backtest that buys *there* is using "
           "the reversal that confirmed it."),
    ("code", """zz = p.zigzag(c, 0.05)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(c, lw=0.8, color="#8a8984")
for pv in zz[:14]:
    col = p.PALETTE[2] if pv.kind == -1 else p.PALETTE[7]
    ax.plot(pv.idx, pv.price, "o", color=col)
    ax.annotate("", xy=(pv.known_from, c[pv.known_from]), xytext=(pv.idx, pv.price), arrowprops={"arrowstyle": "->", "color": col, "lw": 0.8})
ax.set(xlim=(0, zz[13].known_from + 20), title="ZigZag 5%: the swing (dot) and the bar it becomes known (arrow tip)"); plt.show()
lag = np.array([pv.known_from - pv.idx for pv in zz])
print(f"{len(zz)} swings; confirmation arrives {np.median(lag):.0f} bars after the swing (median)")"""),
    ("code", """rows = []
for name, piv in [("ZigZag 5%", zz), ("fractal k=2", p.fractals(h, l, 2)), ("fractal k=5", p.fractals(h, l, 5))]:
    for when, flag in [("swing bar (bug)", False), ("confirmation bar", True)]:
        r = p.swing_low_trades(piv, o, horizon=10, use_known_from=flag)
        rows.append({"swings": name, "buy after": when, "trades": len(r), "mean 10-bar return": f"{r.mean():+.2%}"})
pd.DataFrame(rows)"""),
    ("md", "Buying the swing low \"at\" the swing bar looks like an edge in every detector, largest for ZigZag. Waiting for confirmation, "
           "as a live system must, the edge is gone.\n\n"
           "## 4. The truncation test\n\n"
           "The general cure: run the detector on the data **up to bar t** and check that every swing it reports as known by `t` is also in the "
           "full-data result. A detector that uses the future will report different swings when the future is cut off."),
    ("code", """def truncation_ok(detector, x, cuts):
    full = detector(x)
    for t in cuts:
        known_full = [pv for pv in full if pv.known_from <= t]
        known_cut = [pv for pv in detector(x[: t + 1]) if pv.known_from <= t]
        if known_full != known_cut:
            return False
    return True

def zigzag_with_last_swing(x, pct=0.05):
    \"\"\"A tempting variant: also report the latest extreme as a swing, 'known' at its own bar.\"\"\"
    out = p.zigzag(x, pct)
    tail = x[out[-1].known_from:] if out else x
    j = int(np.argmin(tail)) + (out[-1].known_from if out else 0)
    return out + [p.Pivot(j, j, float(x[j]), -1)]

cuts = range(200, 1500, 50)
print("zigzag:                ", truncation_ok(lambda x: p.zigzag(x, 0.05), c, cuts))
print("fractals k=2:          ", truncation_ok(lambda x: p.fractals(x, x, 2), c, cuts))
print("zigzag + 'latest low': ", truncation_ok(zigzag_with_last_swing, c, cuts))"""),
    ("md", "## Wrap-up\n\n"
           "* Levels from yesterday's bar are known today; swings are known only at confirmation.\n"
           "* Store `known_from` with every swing and trade from it, never from the swing bar.\n"
           "* Truncation tests catch the leak whatever the detector does inside.\n"
           "* Graded versions: `labs/part05/week19_patterns` (pivots, ZigZag, fractals, double tops, head & shoulders) and the Clinic W3 scanner audit."),
]

# ---------------------------------------------------------------------------------------------- 07
NB["07_actions_and_conditions"] = [
    header("07", "Actions and entry conditions", "S12 (Useful actions & entry conditions)",
           "1. Write a crossover with an explicit tie rule and NaN handling.\n"
           "2. Count bars since an event.\n"
           "3. Build `within(n)`, and compose conditions like a strategy spec.\n"
           "4. See how many signals each extra condition removes."),
    ("code", SETUP),
    ("code", """df = p.synthetic_ohlcv(1500, seed=21)
o, h, l, c, v = p.arrays(df)"""),
    ("md", "## 1. Crossover, precisely\n\n"
           "\"Fast crosses above slow\" needs a rule for ties and gaps. Ours: at bar `t`, `a[t] > b[t]` **and** `a[t−1] <= b[t−1]`. "
           "So a touch followed by a cross counts once, at the cross. Any NaN among the four values means no signal, and bar 0 is never a cross."),
    ("md", YOUR_TURN),
    ("ex", """def crossover(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(a.shape, dtype=bool)
    with np.errstate(invalid="ignore"):           # comparisons with NaN are False, which is what we want
        out[1:] = ...                             # ✍️ above now, at or below on the previous bar
    return out

cases = [([1, 2, 2, 3, 1], [2, 2, 2, 2, 2]),              # touch, then cross at bar 3
         ([1, 3, 1, 3, 1], [2, 2, 2, 2, 2]),              # two crosses
         ([np.nan, 3, 1, 3, 3], [2, 2, np.nan, 2, 2]),    # NaNs block the signal
         (list(p.ema(c, 10)), list(p.ema(c, 30)))]        # the real thing
mine = [p.attempt(crossover, a, b) for a, b in cases]
mine = p.check("crossover", mine, [p.crossover(a, b) for a, b in cases])
[m[:5].tolist() for m in mine[:3]] + [f"EMA 10 × 30: {int(mine[3].sum())} crosses"]""",
     """def crossover(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(a.shape, dtype=bool)
    with np.errstate(invalid="ignore"):           # comparisons with NaN are False, which is what we want
        out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out

cases = [([1, 2, 2, 3, 1], [2, 2, 2, 2, 2]),              # touch, then cross at bar 3
         ([1, 3, 1, 3, 1], [2, 2, 2, 2, 2]),              # two crosses
         ([np.nan, 3, 1, 3, 3], [2, 2, np.nan, 2, 2]),    # NaNs block the signal
         (list(p.ema(c, 10)), list(p.ema(c, 30)))]        # the real thing
mine = [p.attempt(crossover, a, b) for a, b in cases]
mine = p.check("crossover", mine, [p.crossover(a, b) for a, b in cases])
[m[:5].tolist() for m in mine[:3]] + [f"EMA 10 × 30: {int(mine[3].sum())} crosses"]"""),
    ("md", "## 2. Bars since\n\n"
           "`bars_since(cond)` is 0 on a bar where `cond` is True, 1 on the next bar, and so on; NaN before the first True. "
           "It turns \"the golden cross happened recently\" into a number you can threshold."),
    ("md", YOUR_TURN),
    ("ex", """def bars_since(cond):
    out = np.full(len(cond), np.nan)
    last = None
    for i, v in enumerate(cond):
        ...                                       # ✍️ remember the bar of the last True; once there is one, out[i] = i − that bar
    return out

sample = [False, True, False, False, True, False, False, False]
cross = p.crossover(p.ema(c, 10), p.ema(c, 30))
mine = [bars_since(sample), bars_since(cross)]
mine = p.check("bars_since", mine, [p.bars_since(sample), p.bars_since(cross)])
mine[0]""",
     """def bars_since(cond):
    out = np.full(len(cond), np.nan)
    last = None
    for i, v in enumerate(cond):
        if v:
            last = i
        if last is not None:
            out[i] = i - last
    return out

sample = [False, True, False, False, True, False, False, False]
cross = p.crossover(p.ema(c, 10), p.ema(c, 30))
mine = [bars_since(sample), bars_since(cross)]
mine = p.check("bars_since", mine, [p.bars_since(sample), p.bars_since(cross)])
mine[0]"""),
    ("md", "## 3. Conditions that remember: `within(n)`\n\n"
           "Entry rules rarely need two events on the *same* bar: \"RSI crossed above 30 **within the last 3 bars** and price is above its "
           "200-bar average\". `within(cond, n)` is True at `t` if `cond` was True on any of the bars `t−n+1 … t`. "
           "A rolling maximum of the booleans does it: `pd.Series(cond).rolling(n, min_periods=1).max()`, back to a boolean array."),
    ("md", YOUR_TURN),
    ("ex", """def within(cond, n):
    return ...                                    # ✍️

mine = p.attempt(within, sample, 2)
mine = p.check("within", mine, p.within(sample, 2))
mine""",
     """def within(cond, n):
    return pd.Series(np.asarray(cond, bool)).rolling(n, min_periods=1).max().to_numpy().astype(bool)

mine = p.attempt(within, sample, 2)
mine = p.check("within", mine, p.within(sample, 2))
mine"""),
    ("md", "## 4. Composing an entry rule\n\n"
           "`p.Condition` wraps a boolean array and supports `&`, `|`, `~`, `.within(n)` and `.confirm(n)` (True for `n` bars in a row). "
           "A strategy's entry rule then reads like its spec. Watch how each condition thins the signals."),
    ("code", """r = p.rsi(c, 14)
rsi_up = p.Condition(p.crossover(r, np.full_like(r, 30.0)), "rsi_x_30")
uptrend = p.Condition(c > p.sma(c, 200), "above_sma200")
narrow = p.Condition(p.nr_n(h, l, 7), "nr7")
inside = p.Condition(p.inside_bar(h, l), "inside")

steps = [rsi_up, rsi_up.within(3), rsi_up.within(3) & uptrend.confirm(5), rsi_up.within(3) & uptrend.confirm(5) & (narrow | inside)]
for s in steps:
    print(s)
entry = steps[2].values
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(c, lw=0.8, color="#8a8984", label="close"); ax.plot(p.sma(c, 200), label="SMA 200")
ax.plot(np.flatnonzero(entry), c[entry], "^", color=p.PALETTE[2], label="entry bars")
ax.set_title("RSI crossed above 30 within 3 bars, in a confirmed uptrend"); ax.legend(); plt.show()"""),
    ("md", "Note that `within(3)` turns one event into up to three True bars. Before backtesting, decide whether you want an **entry "
           "event** (use `crossover` on the combined condition) or a **state** you hold while it is True.\n\n"
           "## Wrap-up\n\n"
           "* Every action states its tie and NaN rules; test them with tiny hand-made arrays.\n"
           "* `bars_since`, `within`, `confirm` turn events into states you can combine.\n"
           "* Graded version: `labs/part05/week19_patterns` (crossovers, gaps, inside bars, NR-n, new highs, `bars_since`, the `Condition` class)."),
]

# ---------------------------------------------------------------------------------------------- 08
NB["08_sentiment_and_tails"] = [
    header("08", "Sentiment, news and tail risk", "S13 (Market sentiment) · S14 (News & NLP sentiment) · S15 (Black swans & tail events)",
           "1. Join weekly sentiment data at the time it was **published**, not the date it describes.\n"
           "2. Score headlines with a lexicon that understands \"not\".\n"
           "3. Turn scored news into a decaying index.\n"
           "4. Estimate a tail index and see normal VaR fail an exceedance test."),
    ("code", SETUP),
    ("md", "## 1. Point-in-time: COT is measured Tuesday, published Friday\n\n"
           "The CFTC's Commitments of Traders report describes positions on **Tuesday** and is released on **Friday at 15:30 New York**. "
           "A backtest that joins it on the Tuesday date knows it three days early. Each daily bar (stamped at the 16:00 close, in UTC) must "
           "get the latest release whose `available_at` is at or before the bar: `pd.merge_asof(..., left_on=\"ts\", right_on=\"available_at\", "
           "direction=\"backward\")`."),
    ("code", """cot = p.cot_releases()
daily = p.daily_closes_utc()
cot.head(3)"""),
    ("md", YOUR_TURN),
    ("ex", """def pit_join(bars, releases, value_cols):
    r = releases.sort_values("available_at")[["available_at", *value_cols]]
    return ...                                    # ✍️ merge_asof, each bar gets the last release published at or before it

mine = p.attempt(pit_join, daily, cot, ["net_long"])
mine = p.check("pit_join", mine, p.pit_join(daily, cot, ["net_long"]))
mine.iloc[2:8]""",
     """def pit_join(bars, releases, value_cols):
    r = releases.sort_values("available_at")[["available_at", *value_cols]]
    return pd.merge_asof(bars.sort_values("ts"), r, left_on="ts", right_on="available_at", direction="backward")

mine = p.attempt(pit_join, daily, cot, ["net_long"])
mine = p.check("pit_join", mine, p.pit_join(daily, cot, ["net_long"]))
mine.iloc[2:8]"""),
    ("code", """leaky = pd.merge_asof(daily, cot[["report_date", "net_long"]].rename(columns={"net_long": "leaky"}),
                      left_on="ts", right_on="report_date", direction="backward")
honest = p.pit_join(daily, cot, ["net_long"])
early = (leaky["leaky"] != honest["net_long"]) & honest["net_long"].notna()
print(f"joined on the report date, {early.mean():.0%} of bars see a number that was not public yet (Tuesday to Friday, every week)")"""),
    ("md", "## 2. Headlines and negation\n\n"
           "A lexicon scorer counts positive and negative words. Without negation, *\"did not beat estimates\"* scores as good news. "
           "Our rule: split the headline into **clauses** (at punctuation and \"but\"), and flip a word's sign when one of the previous "
           "`window` tokens **in the same clause** is a negator (`p.NEGATORS`). The score is the mean of the signs, 0 with no sentiment words."),
    ("md", YOUR_TURN),
    ("ex", """def lexicon_score(text, positive=p.POSITIVE, negative=p.NEGATIVE, window=2):
    signs = []
    for clause in p.CLAUSE.split(text.lower()):
        toks = p.TOKEN.findall(clause)
        for i, t in enumerate(toks):
            s = 1 if t in positive else (-1 if t in negative else 0)
            if s == 0:
                continue
            if ...:                               # ✍️ one of the previous `window` tokens of this clause is a negator
                s = -s
            signs.append(s)
    return float(sum(signs) / len(signs)) if signs else 0.0

mine = [lexicon_score(t) for t in p.HEADLINES]
mine = p.check("lexicon_score", mine, [p.lexicon_score(t) for t in p.HEADLINES])
naive = [p.lexicon_score(t, window=0) for t in p.HEADLINES]
pd.DataFrame({"headline": p.HEADLINES, "no negation": naive, "with negation": mine})""",
     """def lexicon_score(text, positive=p.POSITIVE, negative=p.NEGATIVE, window=2):
    signs = []
    for clause in p.CLAUSE.split(text.lower()):
        toks = p.TOKEN.findall(clause)
        for i, t in enumerate(toks):
            s = 1 if t in positive else (-1 if t in negative else 0)
            if s == 0:
                continue
            if any(w in p.NEGATORS for w in toks[max(0, i - window):i]):
                s = -s
            signs.append(s)
    return float(sum(signs) / len(signs)) if signs else 0.0

mine = [lexicon_score(t) for t in p.HEADLINES]
mine = p.check("lexicon_score", mine, [p.lexicon_score(t) for t in p.HEADLINES])
naive = [p.lexicon_score(t, window=0) for t in p.HEADLINES]
pd.DataFrame({"headline": p.HEADLINES, "no negation": naive, "with negation": mine})"""),
    ("md", "## 3. A decaying news index\n\n"
           "Scored headlines arrive at random times. A usable feature is a **decay-weighted sum**: each item contributes "
           "`score · exp(−age/τ)` once it is published, and nothing before. With τ = 24 hours, yesterday's news counts about a third."),
    ("code", """rng = np.random.default_rng(4)
times = pd.Series(pd.to_datetime("2025-03-03 13:00", utc=True) + pd.to_timedelta(np.sort(rng.uniform(0, 10 * 24, 40)), unit="h"))
scores = rng.choice([-1.0, -0.5, 0.5, 1.0], size=40, p=[0.3, 0.2, 0.2, 0.3])
grid = pd.date_range("2025-03-03 12:00", periods=11 * 24, freq="1h", tz="UTC")
idx = p.decay_index(times, scores, grid, tau_hours=24)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(grid, idx, label="news index (τ = 24 h)")
ax.vlines(times, 0, scores, color=[p.PALETTE[2] if s > 0 else p.PALETTE[7] for s in scores], lw=1.5, label="scored headlines")
ax.axhline(0, color="black", lw=0.6); ax.legend(); ax.set_title("Each headline starts a decaying contribution when it is published"); plt.show()"""),
    ("md", "## 4. Tails: how fat, and what it does to VaR\n\n"
           "Returns have fat tails: big losses are far more common than a normal distribution allows. The **Hill estimator** measures the "
           "tail index ξ from the `k` largest losses (sorted descending, `x[0]` the biggest): `ξ = mean(ln(x[i] / x[k]))` for `i = 0 … k−1`. "
           "For Student-t returns with 3 degrees of freedom the true ξ is 1/3."),
    ("code", """r = p.fat_tailed_returns(5000, df_=3.0, seed=9)
losses = -r
print(f"daily σ {r.std():.2%}, worst day {losses.max():.2%} = {losses.max() / r.std():.1f} σ")"""),
    ("md", YOUR_TURN),
    ("ex", """def hill_estimator(losses, k):
    x = np.sort(np.asarray(losses, float))[::-1]  # largest first
    return ...                                    # ✍️

ks = [50, 100, 200, 400]
mine = [hill_estimator(losses, k) for k in ks]
mine = p.check("hill_estimator", mine, [p.hill_estimator(losses, k) for k in ks])
dict(zip(ks, np.round(mine, 3)))""",
     """def hill_estimator(losses, k):
    x = np.sort(np.asarray(losses, float))[::-1]  # largest first
    return float(np.mean(np.log(x[:k] / x[k])))

ks = [50, 100, 200, 400]
mine = [hill_estimator(losses, k) for k in ks]
mine = p.check("hill_estimator", mine, [p.hill_estimator(losses, k) for k in ks])
dict(zip(ks, np.round(mine, 3)))"""),
    ("md", "The estimate depends on `k` (too few points: noisy; too many: the body of the distribution creeps in). Now the test that "
           "matters: fit VaR on the first half, count how often the second half's losses exceed it, and ask the **Kupiec** test whether "
           "that count is consistent with the promised rate."),
    ("code", """fit, test = losses[:2500], losses[2500:]
rows = []
for q in (0.99, 0.995, 0.999):
    z = {0.99: 2.326, 0.995: 2.576, 0.999: 3.090}[q]
    models = {"normal": z * fit.std(), "historical": np.quantile(fit, q), "Hill (EVT)": p.hill_var(fit, 100, q)}
    for name, var in models.items():
        x = int((test > var).sum())
        rows.append({"level": q, "model": name, "VaR": f"{var:.2%}", "expected": round(2500 * (1 - q), 1),
                     "exceedances": x, "Kupiec p": round(p.kupiec_pvalue(2500, x, 1 - q), 4)})
pd.DataFrame(rows)"""),
    ("md", "Normal VaR looks fine at 99% and falls apart further out: at 99.9% the losses beyond it come several times more often than promised, "
           "and Kupiec rejects it. The tail-aware estimates stay close. That is the whole case for EVT in risk limits.\n\n"
           "## Wrap-up\n\n"
           "* Join every external dataset on **when it was known** (`available_at`), never on the date it describes.\n"
           "* Negation and clauses matter even for simple lexicons; keep sentiment features point-in-time too.\n"
           "* Use tail-aware VaR/ES and backtest exceedances.\n"
           "* Graded version: `labs/part05/week20_sentiment_tail` (VIX regimes, breadth, COT join, lexicon, news de-duplication, Hill and GPD VaR/ES, Kupiec) and the Clinic W4 composite."),
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
