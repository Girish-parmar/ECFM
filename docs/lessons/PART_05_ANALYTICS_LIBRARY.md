# Part 5 — Analytics Library: Patterns, Indicators & Sentiment: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 2, **Month 5** (program weeks 17–20) |
| Format | 16 sessions × 90 min (4 per week) + 4 lab clinics × 120 min + self-study (~6 hrs/week) |
| Total effort | ~24 hrs live + 8 hrs clinic + 24 hrs self-study ≈ **56 hours** |
| Data | Cached Parquet bars from Part 4 (IB + Alpaca), FRED, CBOE indices via IB, CFTC COT, Alpaca news |
| Platform milestone | **M2 Analytics Library**: `quantforge/lib/` helpers, candlestick & chart patterns, 6 indicator function groups, sentiment, actions |
| Covers original items | "Advance with Python" 6–11, 36–38 · Platform roadmap 8–17 |

---

## 1. Learning Objectives

By the end of Part 5 the learner will be able to:

1. **Design** a function library with one consistent API, metadata (look-back, outputs, group) and a registry, following the Open/Closed principle.
2. **Implement** indicators two ways, **vectorized** (research/backtest) and **streaming** (live, O(1) per update), and prove they give identical results.
3. **Validate** every indicator against a reference (TA-Lib) with golden tests, and explain every difference caused by seeding and warm-up conventions.
4. **Detect** candlestick and chart patterns algorithmically, and handle the **confirmation lag** so no pattern uses future bars.
5. **Measure** whether a pattern or indicator signal has a statistically significant edge, with correction for multiple testing.
6. **Build** sentiment indicators from market data (VIX term structure, breadth, put/call, COT) and from text (lexicon, FinBERT, LLM), joined **point-in-time**.
7. **Quantify** tail risk with extreme value theory and build a stress-scenario library of historical crashes.
8. **Ship** the library as tested, documented, benchmarked code (≥ 90% coverage).

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| 2.2 Statistics (hypothesis tests, bootstrap, multiple testing) | S5, S11, S14 |
| 2.3 Time-series econometrics (volatility clustering, stationarity) | S7, S15 |
| 2.4 Risk metrics (VaR, CVaR) | S15 |
| 3.2 Advanced Python (`numba`, profiling, generators) | S1–S3, S16 |
| 3.4 LLD (Specification, Registry, Template Method) | S1, S12 |
| 3.5 Scientific stack (NumPy, pandas, Polars) | All sessions |
| Part 4 `DataHandler` and canonical bar schema | All labs |
| Platform **M1** | Streaming indicators hook into the live bar stream |

---

## 3. Weekly Overview

| Week | Theme | Sessions | Clinic Lab | Platform Output |
|---|---|---|---|---|
| **W1** | Library design, core indicators, candlesticks | S1 Library architecture · S2 Core indicators · S3 Streaming indicators · S4 Candlestick patterns | Golden tests vs TA-Lib for 10 indicators | `lib/helpers.py`, `lib/registry.py`, `lib/indicators/core.py`, `lib/patterns/candles.py` |
| **W2** | Statistical edge & advanced indicator groups | S5 Pattern edge testing · S6 Direction group · S7 Volatility & momentum groups · S8 Price-volume-time group | Edge study of 20 candlestick patterns with FDR control | `lib/indicators/{direction,volatility,momentum,price_volume_time}.py`, `lib/stats/edge.py` |
| **W3** | Levels, chart patterns, actions | S9 Levels group · S10 Pivots & trendlines · S11 Chart patterns · S12 Actions & entry conditions | No-look-ahead chart-pattern scanner on 100 symbols | `lib/indicators/levels.py`, `lib/patterns/charts.py`, `lib/actions.py`, `lib/entry.py` |
| **W4** | Sentiment, news, tail events, release | S13 Market sentiment · S14 News & NLP sentiment · S15 Black swans & tail risk · S16 Integration & release | Daily sentiment composite + stress-test report | `lib/indicators/sentiment.py`, `research/news/`, `lib/tail.py`, **M2 release** |

---

## 4. Library Catalog (target for M2)

| Group (roadmap item) | File | Minimum functions |
|---|---|---|
| Helpers (8) | `lib/helpers.py` | `rolling_window`, `shift`, `pct_change`, `log_returns`, `zscore`, `rank_pct`, `resample_ohlcv`, `align_higher_tf`, `session_mask`, `fill_policy` |
| Candlestick patterns (9) | `lib/patterns/candles.py` | anatomy features + 30 patterns (doji family, hammer/hanging man, inverted hammer/shooting star, engulfing, harami, piercing/dark cloud, morning/evening star, three soldiers/crows, tweezer, marubozu, spinning top, kicker, three inside/outside, abandoned baby…) |
| Chart patterns (10) | `lib/patterns/charts.py` | `zigzag`, `fractals`, `find_swings`, `trendline`, `head_shoulders`, `double_top_bottom`, `triangle`, `flag_pennant`, `wedge`, `rectangle`, `cup_handle` |
| Momentum (11) | `lib/indicators/momentum.py` | RSI, Stochastic, Stoch RSI, MACD, ROC, CCI, Williams %R, TSI, Fisher Transform, Connors RSI, Ehlers Super Smoother / Roofing filter |
| Volatility (12) | `lib/indicators/volatility.py` | TR, ATR, Bollinger (+%B, width), Keltner, Donchian, squeeze, Choppiness, HV close-to-close, Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang |
| Price-volume-time (13) | `lib/indicators/price_volume_time.py` | VWAP (session & anchored, ±σ bands), OBV, A/D, CMF, MFI, relative volume by time of day, volume profile (POC, VAH, VAL), TPO profile |
| Direction (14) | `lib/indicators/direction.py` | SMA, EMA, WMA, DEMA, TEMA, Hull, KAMA, ADX/DMI, Aroon, Parabolic SAR, SuperTrend, Ichimoku, linear-regression slope |
| Sentiment (15) | `lib/indicators/sentiment.py` | VIX percentile, VIX/VIX3M ratio, VVIX, put/call ratio, advance/decline line, McClellan oscillator, % above N-DMA, new highs–lows, COT net positioning index, AAII bull–bear spread, news sentiment index |
| Spread, range, S/R (16) | `lib/indicators/levels.py` | pivot points (classic, Fibonacci, Camarilla, Woodie), swing S/R clustering, Fibonacci levels, ADR, opening range, pair spread z-score, bid–ask spread stats |
| Actions (17) | `lib/actions.py` | `crossover`, `crossunder`, `cross_level`, `gap_up/down`, `gap_filled`, `inside_bar`, `outside_bar`, `nr_n`, `new_high/low`, `consecutive`, `bars_since`, `rising/falling`, `touch` |
| Entry conditions (18, foundation) | `lib/entry.py` | `Condition` with `&`, `|`, `~` composition (Specification pattern) |

**API conventions (non-negotiable):**
- Core functions take and return **NumPy arrays** (`float64`), so they are fast and library-agnostic; thin wrappers adapt pandas/Polars.
- Output length always equals input length; warm-up positions are `NaN`, **never** dropped or back-filled.
- Value at index *t* uses only data up to and including bar *t*. Anything that needs future bars (pivots) returns an explicit **confirmation index**.
- Every function is registered with its group, parameters, outputs and `lookback(params)`.

---

## 5. Session-by-Session Plan

> Each session: **15 min recap/theory → 45 min live coding → 20 min guided lab → 10 min wrap-up and homework.**
> Notebooks live in `notebooks/part05/`; promoted code lives in `quantforge/lib/`.

### Week 1 — Library Design, Core Indicators, Candlesticks

#### S1 · Library Architecture & Helpers (item 8)

| Block | Content |
|---|---|
| Theory | Why a library and not notebook snippets: one tested truth for backtest and live. **Layering:** `numba`/NumPy kernels → typed Python API → pandas/Polars adapters → registry. **Metadata:** look-back (warm-up bars), outputs, group, parameter ranges (used later by the optimizer in Part 8). NaN policy and length invariance. Float precision and why exact equality is never tested. Documentation standard (NumPy docstrings with formula, reference, look-back). |
| Live coding | Registry decorator (below), helpers module, `pytest` golden-test harness that loads TA-Lib outputs. |
| Lab | Register `sma`, `log_returns`, `zscore`; list the registry by group; generate a Markdown catalog automatically. |
| Homework | Write `align_higher_tf` (implemented in S8) test cases first, before the function exists (test-driven). |

```python
# quantforge/lib/registry.py
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    group: str                        # momentum | volatility | direction | pvt | sentiment | levels
    outputs: tuple[str, ...]
    lookback: Callable[..., int]      # warm-up bars as a function of params
    params: dict[str, tuple]          # name -> (default, min, max) for the optimizer

REGISTRY: dict[str, tuple[IndicatorSpec, Callable]] = {}

def indicator(name, group, lookback, params, outputs=("value",)):
    def deco(fn):
        REGISTRY[name] = (IndicatorSpec(name, group, outputs, lookback, params), fn)
        return fn
    return deco
```

#### S2 · Core Indicators, Done Right (5.2)

| Block | Content |
|---|---|
| Theory | SMA, EMA (`α = 2/(n+1)`), WMA, RSI (Wilder smoothing = EMA with `α = 1/n`), MACD, Bollinger Bands (population σ), True Range and ATR (Wilder), Stochastic, OBV, session VWAP. **Seeding conventions** cause most "my RSI is different" bugs: TA-Lib seeds EMA with the SMA of the first *n* values; `pandas.ewm(adjust=False)` seeds with the first value; `adjust=True` uses a weighted average. Look-back of each (EMA *n*−1, RSI *n*, ATR *n*). |
| Live coding | `numba` kernels for SMA, EMA, RSI, ATR (below); compare against TA-Lib and against `pandas.ewm` to show the seeding gap closing with time. |
| Lab | Golden tests for 10 core indicators on 5 symbols; tolerance `1e-8`. |
| Homework | MACD, Bollinger, Stochastic with golden tests; write down each seeding decision in the docstring. |

```python
import numpy as np
from numba import njit

@njit(cache=True)
def ema(x, n):
    """EMA seeded with the SMA of the first n values (TA-Lib convention). Look-back: n-1."""
    out = np.full(x.shape[0], np.nan)
    if x.shape[0] < n:
        return out
    alpha = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, x.shape[0]):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out

@njit(cache=True)
def rsi(close, n=14):
    """Wilder RSI. Look-back: n."""
    out = np.full(close.shape[0], np.nan)
    if close.shape[0] <= n:
        return out
    d = np.diff(close)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag, al = gain[:n].mean(), loss[:n].mean()
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(n + 1, close.shape[0]):
        ag = (ag * (n - 1) + gain[i - 1]) / n
        al = (al * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out

@njit(cache=True)
def atr(high, low, close, n=14):
    """Wilder ATR. Look-back: n."""
    out = np.full(high.shape[0], np.nan)
    if high.shape[0] <= n:
        return out
    tr = np.empty(high.shape[0])
    tr[0] = np.nan
    for i in range(1, high.shape[0]):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    out[n] = tr[1:n + 1].mean()
    for i in range(n + 1, high.shape[0]):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out
```

> Verified: these kernels match `talib.EMA`, `talib.RSI` and `talib.ATR` to within 1e-13 on 3,000 synthetic bars.

#### S3 · Streaming (Incremental) Indicators

| Block | Content |
|---|---|
| Theory | Live trading cannot recompute 5,000 bars on every tick. **Streaming indicators** hold state and update in O(1): running sums (SMA), recursive filters (EMA, Wilder), monotonic deques (rolling max/min for Donchian and Stochastic in amortized O(1)), Welford's algorithm (rolling variance). Handling the forming bar: `update(bar)` on bar close vs `peek(price)` for intrabar values without mutating state. **The equivalence contract:** streaming output == vectorized output for every prefix of the data. |
| Live coding | `StreamingEMA`, `StreamingRSI`, `StreamingATR`, `RollingMax` (deque); property-based equivalence test with `hypothesis`. |
| Lab | Plug streaming indicators into the Part 4 live bar stream (Alpaca crypto works 24/7); log values each bar. |
| Homework | Streaming Bollinger with Welford's algorithm; equivalence test must pass. |

```python
class StreamingEMA:
    def __init__(self, n: int):
        self.n, self.alpha = n, 2.0 / (n + 1)
        self._seed: list[float] = []
        self.value: float | None = None

    def update(self, x: float) -> float | None:
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) == self.n:
                self.value = sum(self._seed) / self.n
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

# tests/lib/test_streaming.py
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

@given(arrays(np.float64, st.integers(30, 500), elements=st.floats(1, 1e4)))
def test_streaming_ema_matches_vectorized(x):
    s = StreamingEMA(20)
    stream = np.array([np.nan if (v := s.update(xi)) is None else v for xi in x])
    np.testing.assert_allclose(stream, ema(x, 20), rtol=1e-9, equal_nan=True)
```

#### S4 · Candlestick Patterns (5.1, item 9)

| Block | Content |
|---|---|
| Theory | **Anatomy features** (body, upper/lower shadow, range, body %, gaps between bodies), all normalized by ATR or recent average range so thresholds work across price levels. Rule definitions for 30+ patterns; single-, two- and three-bar patterns. **Context matters:** a hammer is only a hammer after a decline (trend filter). Output convention: `+1` bullish, `-1` bearish, `0` none (TA-Lib uses ±100). Parameterize thresholds (e.g. doji body ≤ 10% of range) so they can be optimized and tested. |
| Live coding | Anatomy features + engulfing, hammer, doji, morning star; compare with TA-Lib `CDL*` functions. |
| Lab | Implement 10 patterns; for each, count signals on 20 symbols and compare with TA-Lib; explain disagreements by definition. |
| Homework | 20 more patterns, each with a docstring stating the exact rule and a hand-made unit test (a synthetic OHLC sequence that must and must not trigger). |

```python
def candle_anatomy(o, h, l, c):
    body = np.abs(c - o)
    rng = h - l
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    body_pct = np.divide(body, rng, out=np.zeros_like(body), where=rng > 0)
    return body, rng, upper, lower, body_pct

def bullish_engulfing(o, h, l, c):
    out = np.zeros(c.shape, dtype=bool)
    po, pc = o[:-1], c[:-1]
    out[1:] = ((pc < po) & (c[1:] > o[1:])            # bearish then bullish
               & (o[1:] <= pc) & (c[1:] >= po)        # body engulfs previous body
               & ((c[1:] - o[1:]) > (po - pc)))
    return out

def hammer(o, h, l, c, trend_n=10, min_lower=2.0, max_upper=0.1):
    body, rng, upper, lower, _ = candle_anatomy(o, h, l, c)
    shape = (lower >= min_lower * np.maximum(body, 1e-12)) & (upper <= max_upper * rng)
    downtrend = c < sma(c, trend_n)                  # context filter
    return shape & np.roll(downtrend, 1) & (np.arange(c.size) > 0)
```

> Verified: `bullish_engulfing` above produces the same signals as the bullish half of `talib.CDLENGULFING` on the verification dataset.

**Clinic W1:** golden-test suite for 10 core indicators and 10 candlestick patterns, run in CI; coverage report.

---

### Week 2 — Statistical Edge & Advanced Indicator Groups

#### S5 · Does the Pattern Actually Work? Edge Testing

| Block | Content |
|---|---|
| Theory | **Event study** of forward returns after a signal. Entry at the **next bar's open** (the signal is known only at the close). Holding horizons 1, 5, 10 bars. Compare with the unconditional distribution: permutation test, bootstrap confidence interval, hit rate, payoff ratio. Overlapping horizons create autocorrelation, so use block bootstrap or non-overlapping samples. **Multiple testing:** 30 patterns × 3 horizons × 20 symbols = 1,800 tests, so ~90 "significant" at 5% by luck alone; control with Benjamini–Hochberg FDR. Out-of-sample confirmation. |
| Live coding | `pattern_edge()` (below) + FDR across all patterns. |
| Lab | Edge table for 20 patterns on 50 liquid US stocks, daily bars, 2010–2025; which survive FDR at 10%? |
| Homework | Repeat on 5-minute bars; write one page on why results differ between timeframes. |

```python
from scipy.stats import false_discovery_control

def pattern_edge(signal, open_, horizon=5, n_perm=5000, seed=0):
    """Enter at the NEXT bar's open, hold `horizon` bars; permutation test vs all bars."""
    rng = np.random.default_rng(seed)
    entry = np.roll(open_, -1)
    exit_ = np.roll(open_, -(1 + horizon))
    fwd = np.log(exit_ / entry)
    fwd[-(1 + horizon):] = np.nan                    # no future data at the end
    valid = ~np.isnan(fwd)
    sig, fwd = signal[valid], fwd[valid]
    k = int(sig.sum())
    observed = fwd[sig].mean() - fwd.mean()
    null = np.array([fwd[rng.choice(fwd.size, k, replace=False)].mean()
                     for _ in range(n_perm)]) - fwd.mean()
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return {"n": k, "edge": observed, "p_value": p, "hit_rate": float((fwd[sig] > 0).mean())}

# results: list of dicts from pattern_edge over all (pattern, symbol, horizon)
adj = false_discovery_control([r["p_value"] for r in results], method="bh")
```

#### S6 · Direction Group (item 14)

| Block | Content |
|---|---|
| Theory | Lag vs smoothness trade-off. DEMA/TEMA, Hull MA, **KAMA** (efficiency ratio), linear-regression slope, ADX/DMI (trend strength vs direction), Aroon, Parabolic SAR, **SuperTrend** (path-dependent: needs a loop, not vectorizable), **Ichimoku** and its look-ahead trap (Chikou span is plotted 26 bars back, so using it as a feature leaks the future; Senkou spans are shifted forward and must be aligned carefully). |
| Live coding | KAMA, ADX and SuperTrend with `numba` (below). |
| Lab | Plot all trend filters on SPY 2020–2025; measure lag at turning points and number of whipsaws. |
| Homework | Ichimoku implemented with a `safe=True` mode that returns only values known at bar *t*. |

```python
@njit(cache=True)
def supertrend(high, low, close, n=10, mult=3.0):
    """Returns (line, direction) with direction +1 up / -1 down. Path-dependent."""
    a = atr(high, low, close, n)
    hl2 = (high + low) / 2.0
    up_b, lo_b = hl2 + mult * a, hl2 - mult * a
    fu, fl = up_b.copy(), lo_b.copy()
    line = np.full(close.shape[0], np.nan)
    direction = np.zeros(close.shape[0], dtype=np.int8)
    for i in range(n + 1, close.shape[0]):
        fu[i] = up_b[i] if (up_b[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lo_b[i] if (lo_b[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if close[i] > fu[i - 1]:
            direction[i] = 1
        elif close[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1] if direction[i - 1] != 0 else 1
        line[i] = fl[i] if direction[i] == 1 else fu[i]
    return line, direction
```

#### S7 · Volatility & Momentum Groups (items 11, 12)

| Block | Content |
|---|---|
| Theory | **Range-based volatility estimators** are 5–8× more efficient than close-to-close (table below). Bollinger %B and bandwidth, Keltner, Donchian, **squeeze** (Bollinger inside Keltner), Choppiness index. **Momentum:** ROC, CCI, Williams %R, TSI, Stoch RSI, Connors RSI, Fisher Transform; Ehlers **Super Smoother** and **Roofing filter** (removing high-frequency noise and trend before measuring cycles). Bounded vs unbounded oscillators and how to normalize them (percentile rank, z-score) for ML in Part 10. |
| Live coding | All five volatility estimators; squeeze indicator; Super Smoother. |
| Lab | Compare the estimators as forecasts of next-week realized volatility (mean squared error, per estimator). |
| Homework | Momentum group complete with golden tests where TA-Lib has a reference (ROC, CCI, WILLR, STOCHRSI). |

| Estimator | Per-bar variance term | Notes |
|---|---|---|
| Close-to-close | `ln(Cₜ/Cₜ₋₁)²` | Baseline |
| Parkinson | `ln(H/L)² / (4 ln 2)` | Uses the range; biased low with gaps |
| Garman–Klass | `½ ln(H/L)² − (2 ln 2 − 1) ln(C/O)²` | Adds open/close; assumes no drift |
| Rogers–Satchell | `ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)` | Robust to drift |
| Yang–Zhang | `σ²_overnight + k·σ²_open-to-close + (1−k)·σ²_RS`, `k = 0.34 / (1.34 + (n+1)/(n−1))` | Handles overnight gaps and drift |

Annualize with √(bars per year): 252 for daily; 252 × 78 for 5-minute US equity bars during regular hours.

#### S8 · Price-Volume-Time Group (item 13) & Multi-Timeframe

| Block | Content |
|---|---|
| Theory | Session VWAP (resets each session) and **anchored VWAP** (from an event: earnings, swing low); VWAP σ-bands. OBV, Accumulation/Distribution, Chaikin Money Flow, MFI. **Relative volume by time of day** (volume vs the average for that same 5-minute slot). **Volume profile:** point of control (POC), value area high/low (70% of volume). TPO/market profile concept. **Multi-timeframe without look-ahead:** a higher-timeframe bar is known only after it closes; shift by one higher-timeframe bar, then forward-fill onto the lower timeframe. US sessions start at 09:30, so offset hourly bins by 30 minutes. |
| Live coding | `volume_profile()` and `align_higher_tf()` (below). |
| Lab | Show the look-ahead bug: a strategy using unshifted hourly EMA on 5-minute bars vs the correct version; compare backtest equity curves (the bug looks "profitable"). |
| Homework | Anchored VWAP from the most recent confirmed swing low (uses S10 pivots). |

```python
def volume_profile(price, volume, bins=50, value_area=0.70):
    hist, edges = np.histogram(price, bins=bins, weights=volume)
    poc = int(np.argmax(hist))
    lo = hi = poc
    total, acc = hist.sum(), hist[poc]
    while acc < value_area * total:                  # grow toward the heavier neighbour
        left = hist[lo - 1] if lo > 0 else -1.0
        right = hist[hi + 1] if hi < len(hist) - 1 else -1.0
        if right >= left:
            hi += 1; acc += hist[hi]
        else:
            lo -= 1; acc += hist[lo]
    mid = (edges[:-1] + edges[1:]) / 2
    return {"poc": mid[poc], "val": edges[lo], "vah": edges[hi + 1]}

def align_higher_tf(df_ltf: pd.DataFrame, fn, rule="1h", offset="30min") -> pd.Series:
    """Compute fn on completed higher-TF bars and map to lower-TF bars with NO look-ahead.
    Bars are labeled by OPEN time (canonical schema)."""
    htf_close = (df_ltf["close"]
                 .resample(rule, label="left", closed="left", offset=offset)
                 .last().dropna())
    htf_val = pd.Series(fn(htf_close.to_numpy()), index=htf_close.index)
    return htf_val.shift(1).reindex(df_ltf.index, method="ffill")   # known only after the HTF bar closes
```

**Clinic W2:** edge study of 20 candlestick patterns across 50 symbols with BH-FDR; presentation of which (if any) survive, and on which timeframe.

---

### Week 3 — Levels, Chart Patterns, Actions

#### S9 · Spread, Range, Support & Resistance Group (item 16)

| Block | Content |
|---|---|
| Theory | Floor pivot points (classic, Fibonacci, Camarilla, Woodie) from the previous session's H/L/C. **Data-driven S/R:** cluster confirmed swing prices with kernel density estimation or DBSCAN; strength = number of touches × volume at level × recency. Fibonacci retracements (as levels to test statistically, not as belief). Range statistics: average daily range (ADR), opening range (first 15/30 min), range expansion/contraction. **Spread measures:** quoted and effective bid–ask spread (from Part 4 quotes), pair spread and its z-score (a preview of Part 9). |
| Live coding | Pivot points; KDE-based S/R levels from swing highs/lows; opening range. |
| Lab | Test whether price reacts at KDE levels more than at random levels (placebo test). |
| Homework | S/R level object with `touches`, `last_touch`, `strength` and a `nearest(price)` helper. |

#### S10 · Pivots, Swings & Trendlines (5.5, item 10 part 1)

| Block | Content |
|---|---|
| Theory | Swing detection methods: **ZigZag** (% or ATR reversal threshold), **Williams fractals** (k bars on each side), `scipy.signal.find_peaks` with prominence. **Confirmation lag:** a swing high at bar *t* is only known when price has reversed enough, at a later bar (the confirmation index); signals must use the confirmation index, never the pivot index. Trendlines: fit lines through the last 2–3 swing points; RANSAC for robustness; breakout = close beyond line ± ATR buffer. |
| Live coding | `zigzag()` with confirmation index (below); trendline fitting. |
| Lab | Visualize pivot index vs confirmation index on AAPL; count how many bars of lag each method has. |
| Homework | `fractals(high, low, k)` with confirmation index = pivot index + k. |

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Pivot:
    idx: int            # bar where the swing extreme happened
    confirm_idx: int    # first bar where the swing is KNOWN; signals must use this
    price: float
    kind: int           # +1 swing high, -1 swing low

def zigzag(high, low, pct):
    pivots, trend, hi_i, lo_i = [], 0, 0, 0
    for i in range(1, len(high)):
        if trend >= 0 and high[i] >= high[hi_i]:
            hi_i = i
        if trend <= 0 and low[i] <= low[lo_i]:
            lo_i = i
        if trend >= 0 and low[i] <= high[hi_i] * (1 - pct):
            pivots.append(Pivot(hi_i, i, float(high[hi_i]), +1))
            trend, lo_i = -1, i
        elif trend <= 0 and high[i] >= low[lo_i] * (1 + pct):
            pivots.append(Pivot(lo_i, i, float(low[lo_i]), -1))
            trend, hi_i = +1, i
    return pivots
```

#### S11 · Chart Pattern Identifier (item 10 part 2)

| Block | Content |
|---|---|
| Theory | Patterns as **rules over a sequence of pivots**: head & shoulders (5 pivots: L-H-L-HH-L-H with shoulders within tolerance and a neckline), double top/bottom, ascending/descending/symmetric triangles (converging trendline slopes), flags and pennants (sharp pole + tight consolidation), wedges, rectangles, cup & handle. Tolerances in ATR units. **Quality score** (symmetry, volume behaviour, fit error). Pattern completion = neckline/boundary break, evaluated at the confirmation bar. Harmonic patterns (Gartley, Bat, Butterfly, Crab): Fibonacci-ratio rules on XABCD pivots; overview only. |
| Live coding | Double top/bottom and head & shoulders detectors over the pivot list. |
| Lab | **Scanner:** run all detectors on 100 symbols, daily bars; output a table (symbol, pattern, confirm date, quality, breakout level) and an event study of post-breakout returns. |
| Homework | Triangle and flag detectors; unit tests with synthetic price paths that contain exactly one pattern. |

```python
def double_top(pivots, tol_atr, atr_at, min_sep=5):
    """Two swing highs within tol_atr*ATR of each other, separated by a swing low.
    Signal fires when price closes below the middle low (neckline), checked by the caller."""
    found = []
    for a, b, c in zip(pivots, pivots[1:], pivots[2:]):
        if (a.kind, b.kind, c.kind) != (+1, -1, +1):
            continue
        if c.idx - a.idx < min_sep:
            continue
        if abs(a.price - c.price) <= tol_atr * atr_at[c.idx]:
            found.append({"first": a, "neckline": b.price, "second": c,
                          "known_from": c.confirm_idx})
    return found
```

#### S12 · Useful Actions & Entry Conditions (5.6, items 17, 18)

| Block | Content |
|---|---|
| Theory | Precise definitions matter: `crossover(a, b)` = `a[t] > b[t] and a[t-1] <= b[t-1]` (a tie counts as "not above"). Crossing a scalar level; gaps (full vs partial, gap fill); inside/outside bars; NR4/NR7; new N-bar high/low; `bars_since`; `consecutive`; rising/falling. **Specification pattern:** conditions as composable objects with `&`, `|`, `~`, readable names (for the trade journal) and a look-back. This is the foundation of the Part 7 strategy library and the Part 12 strategy creator. |
| Live coding | Actions module (below) + `Condition` class. |
| Lab | Express 5 textbook entry rules as composed conditions; check each against a hand-written loop version. |
| Homework | Add `within(n)` ("condition was true within the last n bars") and `confirm(n)` ("true for n consecutive bars") combinators. |

```python
def crossover(a, b):
    a = np.asarray(a, dtype=float)
    b = np.broadcast_to(np.asarray(b, dtype=float), a.shape)
    out = np.zeros(a.shape, dtype=bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])   # NaN comparisons are False
    return out

def bars_since(cond):
    out, last = np.full(len(cond), np.nan), -1
    for i, c in enumerate(cond):
        if c:
            last = i
        if last >= 0:
            out[i] = i - last
    return out

class Condition:
    def __init__(self, fn, name: str):
        self.fn, self.name = fn, name
    def __call__(self, ctx) -> np.ndarray:
        return self.fn(ctx)
    def __and__(self, other): return Condition(lambda x: self(x) & other(x), f"({self.name} AND {other.name})")
    def __or__(self, other):  return Condition(lambda x: self(x) | other(x), f"({self.name} OR {other.name})")
    def __invert__(self):     return Condition(lambda x: ~self(x), f"NOT {self.name}")

golden_cross = Condition(lambda d: crossover(ema(d["close"], 50), ema(d["close"], 200)), "EMA50 x EMA200")
not_overbought = ~Condition(lambda d: rsi(d["close"], 14) > 70, "RSI>70")
entry = golden_cross & not_overbought
```

**Clinic W3:** chart-pattern scanner on 100 symbols with a **no-look-ahead audit**: re-run the scanner on data truncated at each signal's `known_from` bar and confirm the same signal appears.

---

### Week 4 — Sentiment, News, Tail Events, Release

#### S13 · Market Sentiment Indicators (5.4, item 15)

| Block | Content |
|---|---|
| Theory | **Volatility-based:** VIX level and 1-year percentile, VIX/VIX3M ratio (> 1 = inverted term structure = stress), VVIX, SKEW index. **Options:** CBOE put/call ratios (equity vs index). **Breadth:** advance/decline line, McClellan oscillator, % of stocks above 50/200-DMA, new highs − new lows (compute from your own universe to avoid paid feeds). **Positioning:** CFTC Commitments of Traders (COT) net positioning index, short interest, AAII bull–bear spread. **Point-in-time rule:** each source has a publication lag. COT data is as of Tuesday and released on Friday; AAII is released weekly; short interest is published twice a month with a delay. Join with `merge_asof` on the **availability** timestamp, never the reference date. Sources: IB `Index("VIX", "CBOE")` / `Index("VIX3M", "CBOE")`, FRED (`VIXCLS`), CFTC public reporting API. |
| Live coding | VIX term-structure ratio from IB; breadth from a 500-stock universe; COT join (below). |
| Lab | Build a daily sentiment composite (z-scored average of 5 components); study SPY forward 20-day returns by composite quintile. |
| Homework | Add a regime label (risk-on / neutral / risk-off) with hysteresis so it does not flip daily. |

```python
# Point-in-time join: COT is as of Tuesday, available Friday 15:30 ET
cot["available_at"] = (cot["report_date"] + pd.Timedelta(days=3)).dt.tz_localize("America/New_York") \
                      + pd.Timedelta(hours=15, minutes=30)
cot["available_at"] = cot["available_at"].dt.tz_convert("UTC")
bars = pd.merge_asof(bars.sort_values("ts"), cot.sort_values("available_at"),
                     left_on="ts", right_on="available_at", direction="backward")
```

#### S14 · News Analysis & NLP Sentiment (5.7, 5.8)

| Block | Content |
|---|---|
| Theory | **Sources:** Alpaca news API (Benzinga, historical + WebSocket), IB news (requires provider subscriptions), SEC EDGAR filings (8-K for events). **Pipeline:** ingest → deduplicate (same story, many sources) → entity-to-ticker mapping → score → aggregate → align to bars. **Scoring methods, compared:** Loughran–McDonald finance lexicon (fast, transparent), **FinBERT** (`ProsusAI/finbert`, local, free), LLM with a JSON output schema (most nuanced, slowest and costs money; cache results). **Aggregation:** decay-weighted sentiment index `Sₜ = Σ sᵢ·exp(−(t − tᵢ)/τ)` over news published up to *t*. **Event study:** abnormal returns vs a market model around news, split by sentiment. Timing: news published after the close affects the next open. |
| Live coding | Alpaca news download → FinBERT scoring → decay index (below). |
| Lab | Score 12 months of news for 20 stocks with lexicon and FinBERT; compare how well each predicts next-day open-to-close returns. |
| Homework | Score the same headlines with an LLM using a fixed JSON schema; measure agreement (Cohen's κ) with FinBERT. |

```python
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from transformers import pipeline

news = NewsClient(key, secret).get_news(NewsRequest(symbols="AAPL", start=start, end=end, limit=50))
items = news.data["news"]
headlines = [n.headline for n in items]

finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
scores = []
for probs in finbert(headlines, truncation=True):
    p = {d["label"]: d["score"] for d in probs}
    scores.append(p["positive"] - p["negative"])        # score in [-1, 1]

def decay_index(event_ts: np.ndarray, event_score: np.ndarray, bar_ts: np.ndarray, tau_hours=24.0):
    """Sentiment at each bar from news published at or before that bar (no look-ahead)."""
    out = np.zeros(bar_ts.size)
    for j, t in enumerate(bar_ts):
        m = event_ts <= t
        age_h = (t - event_ts[m]) / np.timedelta64(1, "h")
        out[j] = np.sum(event_score[m] * np.exp(-age_h / tau_hours))
    return out
```

#### S15 · Black Swans & Tail Events (5.9)

| Block | Content |
|---|---|
| Theory | Why normal-distribution risk models fail: fat tails, volatility clustering, correlation going to 1 in crises. **Extreme value theory:** Hill tail-index estimator; peaks-over-threshold with the generalized Pareto distribution (GPD); EVT VaR and Expected Shortfall. **Crash library** for stress tests: Oct 1987, 2008 (Lehman), May 6 2010 flash crash, Aug 24 2015, Feb 5 2018 ("Volmageddon", short-vol blow-up), Mar 2020 (COVID), 2022 rates shock. **Early-warning indicators** (not predictors): VIX term-structure inversion, high-yield credit spreads (FRED `BAMLH0A0HYM2`), breadth collapse, SKEW. Tail hedging overview (details in Parts 6–7). |
| Live coding | EVT VaR/ES (below); replay a strategy's positions through the crash library. |
| Lab | Compare normal VaR, historical VaR and EVT VaR at 99% and 99.5% for SPY, QQQ, BTC; backtest exceedances (Kupiec test). |
| Homework | Stress report: for a sample portfolio, P&L under each historical crash and under 3 hypothetical shocks (−10% equity gap, vol ×3, correlations → 0.9). |

```python
from scipy.stats import genpareto

def evt_var_es(returns, q=0.99, threshold_q=0.95):
    """Peaks-over-threshold EVT on losses. Valid for 0 < xi < 1."""
    losses = -np.asarray(returns)
    u = np.quantile(losses, threshold_q)
    exc = losses[losses > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    n, nu = losses.size, exc.size
    var = u + beta / xi * ((n / nu * (1 - q)) ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi)
    return {"xi": xi, "VaR": var, "ES": es}

def hill(returns, k=100):
    """Hill estimator of the tail index xi on the k largest losses."""
    x = np.sort(-np.asarray(returns))[::-1]
    return np.mean(np.log(x[:k])) - np.log(x[k])
```

> Verified: on 20,000 Student-t (3 d.f.) returns, both estimators recover ξ ≈ 1/3 and EVT VaR(99%) is within 2% of the empirical quantile.

#### S16 · Integration, Performance & M2 Release

| Block | Content |
|---|---|
| Theory | Release checklist: registry complete, docstrings with formula and look-back, golden tests, streaming equivalence, property tests (length invariance, NaN warm-up, shift-invariance where expected: e.g. RSI unchanged when all prices are scaled ×10), look-ahead audit, benchmarks. **Performance:** pure Python vs NumPy vs `numba` vs Polars; `numba` caching and first-call compile cost; benchmark with `pytest-benchmark`. Auto-generated docs (MkDocs + `mkdocstrings`) from the registry. |
| Live coding | Generic property tests that run over every registered indicator (below); benchmark suite. |
| Lab | Fix every failing property test across the library. |
| Homework | Final assessment (Section 8). |

```python
# tests/lib/test_all_indicators.py — runs for every registered single-input indicator
import pytest
from quantforge.lib.registry import REGISTRY

CLOSE_ONLY = [name for name, (spec, fn) in REGISTRY.items() if spec.group in {"momentum", "direction"}]

@pytest.mark.parametrize("name", CLOSE_ONLY)
def test_length_and_warmup(name, close):
    spec, fn = REGISTRY[name]
    out = fn(close)
    assert out.shape == close.shape
    lb = spec.lookback(**{k: v[0] for k, v in spec.params.items()})
    assert np.isnan(out[:lb]).all() and not np.isnan(out[lb:]).any()

@pytest.mark.parametrize("name", CLOSE_ONLY)
def test_no_lookahead(name, close):
    """Value at t must not change when future bars are removed."""
    _, fn = REGISTRY[name]
    full, cut = fn(close), fn(close[:-50])
    np.testing.assert_allclose(full[:-50], cut, equal_nan=True)
```

**Clinic W4:** assessment dry run and code review of each learner's M2 pull request.

---

## 6. Notebook Map (`notebooks/part05/`)

| Notebook | Session | Promoted to |
|---|---|---|
| `01_library_registry.ipynb` | S1 | `lib/registry.py`, `lib/helpers.py` |
| `02_core_indicators_golden.ipynb` | S2 | `lib/indicators/core.py`, `tests/lib/golden/` |
| `03_streaming_indicators.ipynb` | S3 | `lib/streaming.py` |
| `04_candlestick_patterns.ipynb` | S4 | `lib/patterns/candles.py` |
| `05_pattern_edge_fdr.ipynb` | S5 | `lib/stats/edge.py` |
| `06_direction_group.ipynb` | S6 | `lib/indicators/direction.py` |
| `07_volatility_momentum.ipynb` | S7 | `lib/indicators/volatility.py`, `lib/indicators/momentum.py` |
| `08_pvt_multitimeframe.ipynb` | S8 | `lib/indicators/price_volume_time.py`, `lib/helpers.py` |
| `09_levels_sr.ipynb` | S9 | `lib/indicators/levels.py` |
| `10_pivots_trendlines.ipynb` | S10 | `lib/patterns/charts.py` |
| `11_chart_pattern_scanner.ipynb` | S11 | `lib/patterns/charts.py`, `screener/patterns.py` |
| `12_actions_conditions.ipynb` | S12 | `lib/actions.py`, `lib/entry.py` |
| `13_market_sentiment.ipynb` | S13 | `lib/indicators/sentiment.py`, `data/sources/{fred,cftc}.py` |
| `14_news_nlp.ipynb` | S14 | `research/news/` |
| `15_tail_risk_evt.ipynb` | S15 | `lib/tail.py`, `research/rare_events/scenarios.py` |
| `16_release_benchmarks.ipynb` | S16 | `tests/lib/`, `docs/` |

---

## 7. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Different EMA/RSI seeding than the reference | Values differ at the start, converge later | Document the convention; golden tests from bar 0 |
| 2 | Dropping warm-up rows | Arrays misaligned with prices, off-by-*n* signals | Length-invariance property test |
| 3 | Using pivot index instead of confirmation index | Chart patterns look extremely profitable | Truncation (no-look-ahead) test; always use `known_from` |
| 4 | Unshifted higher-timeframe values | Backtest too good on intraday data | `align_higher_tf` with shift; the S8 lab |
| 5 | Ichimoku Chikou span as a feature | Future leakage | `safe=True` mode; truncation test |
| 6 | Joining sentiment on reference date, not publication time | Leakage of COT/AAII/news | `merge_asof` on `available_at` |
| 7 | Signals acted on at the same bar's close | Unrealistic fills | Enter at next bar's open in edge tests and backtests |
| 8 | Testing 1,800 hypotheses and reporting the best | False discoveries | BH-FDR; out-of-sample confirmation |
| 9 | Pattern thresholds in absolute price units | Pattern works only on one price level | Normalize by ATR or average range |
| 10 | Streaming indicator drifts from vectorized | Live signals differ from backtest | Equivalence property test on every prefix |
| 11 | Mutating indicator state on intrabar ticks | Double-counted updates | Separate `update(bar)` from `peek(price)` |
| 12 | Division by zero on flat bars (range = 0) | `inf`/`NaN` propagation | `np.divide(..., where=...)`; flat-bar unit tests |
| 13 | Mixing data feeds (IEX vs consolidated volume) in volume indicators | Unstable volume signals | Keep `source` on bars; one feed per strategy |
| 14 | Normal VaR for tail risk | Loss far beyond VaR in crises | EVT VaR/ES; exceedance backtests |

---

## 8. Assessment — Platform Milestone M2

**Task:** ship `quantforge.lib` v0.2 and a research report.

1. **Library:** ≥ 70 registered functions across all groups in Section 4, with docstrings (formula, look-back, reference).
2. **Correctness:** golden tests vs TA-Lib for every function TA-Lib also provides (tolerance 1e-8); streaming ≡ vectorized for ≥ 10 streaming indicators; generic property tests (length, warm-up, no look-ahead) pass for every function.
3. **Patterns:** 30 candlestick patterns with synthetic-sequence unit tests; ≥ 5 chart-pattern detectors using confirmation indices, passing the truncation audit.
4. **Research report (≤ 8 pages):** edge study of candlestick patterns and one chart pattern on ≥ 50 symbols with BH-FDR; daily sentiment composite and its relation to forward returns; EVT tail-risk comparison.
5. **Engineering:** ≥ 90% coverage on `lib/`, `mypy --strict`, CI green, benchmark table, auto-generated docs.

| Criterion | Points |
|---|---|
| Indicator correctness (golden + property tests) | 20 |
| Streaming implementations and equivalence | 10 |
| Candlestick & chart patterns (rules, tests, no look-ahead) | 15 |
| Sentiment (market + news), point-in-time correct | 15 |
| Statistical rigor of the edge study (FDR, out-of-sample) | 15 |
| Tail-risk analysis (EVT, stress scenarios) | 10 |
| Code quality, docs, coverage, benchmarks | 10 |
| Report clarity and honest conclusions | 5 |
| **Total** | **100** |

Pass mark: 70, **and** the no-look-ahead property test must pass for every function (mandatory).

---

## 9. Further Reading

| Type | Reference |
|---|---|
| Book | Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research. (RSI, ATR, ADX, Parabolic SAR) |
| Book | Nison, S. (2001). *Japanese Candlestick Charting Techniques* (2nd ed.). New York Institute of Finance. |
| Book | Bulkowski, T. (2021). *Encyclopedia of Chart Patterns* (3rd ed.). Wiley. |
| Book | Ehlers, J. F. (2013). *Cycle Analytics for Traders*. Wiley. |
| Book | Aronson, D. (2006). *Evidence-Based Technical Analysis*. Wiley. (data-mining bias, statistical testing of rules) |
| Book | Sinclair, E. (2013). *Volatility Trading* (2nd ed.). Wiley. (volatility estimators) |
| Paper | Lo, A., Mamaysky, H. & Wang, J. (2000). "Foundations of Technical Analysis." *Journal of Finance*, 55(4). (algorithmic pattern recognition) |
| Paper | Yang, D. & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices." *Journal of Business*, 73(3). |
| Paper | Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability?" *Journal of Finance*, 66(1). (finance sentiment lexicon) |
| Paper | Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." arXiv:1908.10063. |
| Paper | Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate." *JRSS B*, 57(1). |
| Book | McNeil, A., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management* (rev. ed.). Princeton. (EVT) |
| Book | Taleb, N. N. (2007). *The Black Swan*. Random House. |
| Docs | TA-Lib function reference; `numba` performance guide; Alpaca News API; CFTC COT public reporting |

---

## 10. Instructor Notes

- Regenerate TA-Lib golden files whenever the TA-Lib version is bumped; pin it in `uv.lock`.
- Stress, repeatedly, that most classic patterns show **no significant edge** after costs and FDR. The skill being taught is the testing method, not a belief in any pattern.
- Keep a "look-ahead hall of shame": one bugged backtest per week (unshifted HTF, pivot index, Chikou, reference-date joins) and ask learners to find the bug.
- FinBERT runs on CPU for a few thousand headlines; for larger news sets, give learners GPU credits or pre-scored datasets.
- LLM scoring costs money per call; require caching by headline hash and a fixed prompt version for reproducibility.
