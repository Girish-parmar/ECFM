"""Helper functions for the Part 5 guided notebooks (MFAAT, Month 5: the analytics library — indicators, patterns,
sentiment and tail risk).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

All data here is SYNTHETIC (generated with fixed seeds), so every notebook runs offline and every "edge" you find is
one we planted, or luck. The graded, test-driven versions (with TA-Lib golden files) are in labs/part05/.

Library conventions used throughout (lesson plan S1): NumPy arrays in and out, output length = input length,
NaN only during the declared look-back, and no look-ahead.
"""
from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P5_STRICT") == "1"      # tests: a failed check raises instead of continuing


# --------------------------------------------------------------------------------------
# Plot style and the exercise checker
# --------------------------------------------------------------------------------------
def use_course_style() -> None:
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=PALETTE), "figure.figsize": (9, 4.5),
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#52514e", "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "legend.frameon": False,
    })


def _numeric(x) -> bool:
    return isinstance(x, (int, float, np.number, np.ndarray)) and not isinstance(x, bool)


def _same(got, expected, rtol: float, atol: float) -> bool:
    if isinstance(expected, Decimal):
        return isinstance(got, Decimal) and got == expected            # exact, and a float is not money
    if isinstance(expected, (pd.DataFrame, pd.Series)):
        test = pd.testing.assert_frame_equal if isinstance(expected, pd.DataFrame) else pd.testing.assert_series_equal
        try:
            test(got, expected, check_exact=False, rtol=rtol, atol=atol, check_dtype=False, check_names=False,
                 check_freq=False)
            return True
        except (AssertionError, TypeError, AttributeError):
            return False
    if isinstance(expected, dict):
        return (isinstance(got, dict) and set(got) == set(expected)
                and all(_same(got[k], v, rtol, atol) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)) and expected and all(_numeric(v) and np.ndim(v) == 0 for v in expected):
        expected = np.asarray(expected, dtype=float)
    if _numeric(expected):
        g, e = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
        return g.shape == e.shape and np.allclose(g, e, rtol=rtol, atol=atol, equal_nan=True)
    if isinstance(expected, (list, tuple)):
        got = list(got)
        return len(got) == len(expected) and all(_same(a, b, rtol, atol) for a, b in zip(got, expected))
    return type(got) is type(expected) and got == expected if isinstance(expected, bool) else got == expected


def check(name: str, got, expected, rtol: float = 1e-6, atol: float = 1e-9):
    """Compare your answer with the reference: exact for Decimals, text, dates and objects (recursively inside
    lists and dicts); with a tolerance for floats and arrays. Returns your value if correct, else the reference
    (so the notebook keeps running)."""
    try:
        if got is Ellipsis or (isinstance(got, (tuple, list)) and any(g is Ellipsis for g in got)):
            raise ValueError("not done yet")
        ok = bool(_same(got, expected, rtol, atol))
    except Exception:  # noqa: BLE001 - any failure means "not correct yet"
        ok = False
    if ok:
        print(f"✅ {name}: correct")
        return got
    msg = f"❌ {name}: not matching the reference yet"
    if STRICT:
        raise AssertionError(msg)
    print(msg + " — continuing with the reference answer so the rest of the notebook runs.")
    return expected


def attempt(fn, *args, **kwargs):
    """Run fn; if it raises (for example because a blank `...` is still in it), return Ellipsis instead, so
    p.check reports "not done yet" and the notebook keeps going."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - an unfinished exercise may fail in any way
        return Ellipsis


# --------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------
def synthetic_ohlcv(n: int = 1500, seed: int = 0, start: str = "2020-01-02", price0: float = 100.0,
                    drift: float = 0.0002) -> pd.DataFrame:
    """Daily bars with volatility clustering (GARCH-like), Student-t shocks and overnight gaps."""
    rng = np.random.default_rng(seed)
    var, lr = 0.0001, np.empty(n)
    for t in range(n):
        z = rng.standard_t(5) / np.sqrt(5 / 3)
        lr[t] = drift + np.sqrt(var) * z
        var = 0.000002 + 0.08 * lr[t] ** 2 + 0.9 * var
    close = price0 * np.exp(np.cumsum(lr))
    prev = np.concatenate([[price0], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.3, n) * np.abs(lr))
    span = np.abs(lr) + rng.gamma(2.0, 0.004, n)
    high = np.maximum(open_, close) * np.exp(rng.uniform(0.1, 0.6, n) * span)
    low = np.minimum(open_, close) * np.exp(-rng.uniform(0.1, 0.6, n) * span)
    volume = np.round(1e6 * np.exp(rng.normal(0, 0.3, n)) * (1 + 40 * np.abs(lr)))
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def universe(n_symbols: int = 20, n: int = 1000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Independent symbols: no pattern has a real edge on them."""
    return {f"S{i:02d}": synthetic_ohlcv(n, seed=seed * 1000 + i) for i in range(n_symbols)}


def arrays(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return tuple(df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close", "volume"))


# --------------------------------------------------------------------------------------
# 01 · core indicators and the registry
# --------------------------------------------------------------------------------------
REGISTRY: dict[str, dict] = {}


def indicator(name: str, group: str, lookback):
    """Decorator: register an indicator with its group and a look-back function of its parameters."""
    def wrap(fn):
        REGISTRY[name] = {"fn": fn, "group": group, "lookback": lookback, "doc": (fn.__doc__ or "").strip().split("\n")[0]}
        return fn
    return wrap


@indicator("sma", "trend", lambda n=20: n - 1)
def sma(x: np.ndarray, n: int = 20) -> np.ndarray:
    """Simple moving average. Look-back n−1."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.size >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


@indicator("ema", "trend", lambda n=20: n - 1)
def ema(x: np.ndarray, n: int = 20) -> np.ndarray:
    """EMA, α = 2/(n+1), seeded with the SMA of the first n values (TA-Lib). Look-back n−1."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.size < n:
        return out
    a = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def ema_pandas(x: np.ndarray, n: int = 20) -> np.ndarray:
    """pandas' ewm(span=n, adjust=False): seeded with the FIRST value, no warm-up NaN."""
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


@indicator("rsi", "momentum", lambda n=14: n)
def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI: average gain/loss seeded with the mean of the first n changes, then
    avg = (avg·(n−1) + new)/n. RSI = 100 − 100/(1 + avg_gain/avg_loss), 100 when avg_loss is 0. Look-back n."""
    close = np.asarray(close, dtype=float)
    out = np.full(close.shape, np.nan)
    if close.size <= n:
        return out
    d = np.diff(close)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[:n].mean(), loss[:n].mean()
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(n + 1, close.size):
        ag = (ag * (n - 1) + gain[i - 1]) / n
        al = (al * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def true_range(high, low, close) -> np.ndarray:
    """max(H−L, |H−C₋₁|, |L−C₋₁|); NaN at bar 0 (no previous close)."""
    high, low, close = (np.asarray(a, dtype=float) for a in (high, low, close))
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = np.nan
    return tr


@indicator("atr", "volatility", lambda n=14: n)
def atr(high, low, close, n: int = 14) -> np.ndarray:
    """Wilder ATR: the mean of TR[1..n] at bar n, then (ATR·(n−1) + TR)/n. Look-back n."""
    tr = true_range(high, low, close)
    out = np.full(tr.shape, np.nan)
    if tr.size <= n:
        return out
    out[n] = tr[1:n + 1].mean()
    for i in range(n + 1, tr.size):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


@indicator("bbands_width", "volatility", lambda n=20: n - 1)
def bbands_width(close, n: int = 20, k: float = 2.0) -> np.ndarray:
    """(upper − lower) / middle of Bollinger Bands with the POPULATION σ. Look-back n−1."""
    s = pd.Series(np.asarray(close, dtype=float))
    mid, sd = s.rolling(n).mean(), s.rolling(n).std(ddof=0)
    return (2 * k * sd / mid).to_numpy()


def leaky_smooth(x, n: int = 5) -> np.ndarray:
    """A centered moving average: looks fine on a chart, uses n//2 FUTURE values. For the audit demo."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(n, center=True).mean().to_numpy()


def audit(fn, x: np.ndarray, lookback: int) -> dict[str, bool]:
    """The library contract: same length, NaN exactly during the look-back, and no look-ahead (values up to t
    do not change when the data after t is removed)."""
    y = fn(x)
    nan = np.isnan(y)
    ok_len = y.shape == x.shape
    ok_warm = bool(nan[:lookback].all() and not nan[lookback:].any())
    cut = int(len(x) * 0.6)
    ok_la = bool(np.allclose(fn(x[:cut]), y[:cut], equal_nan=True))
    return {"length": ok_len, "warm-up": ok_warm, "no look-ahead": ok_la}


# --------------------------------------------------------------------------------------
# 02 · streaming indicators
# --------------------------------------------------------------------------------------
class StreamingEMA:
    """O(1) EMA; equals ema(x, n) on every prefix. None until n values are in."""

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
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def peek(self, x: float) -> float | None:
        """The value IF the bar closed at x now, without changing state (for intrabar displays)."""
        if self.value is None:
            return None
        return self.alpha * x + (1 - self.alpha) * self.value


class RollingMax:
    """Max of the last n values in amortized O(1) with a monotonic deque of (index, value), values decreasing.
    None until n values are in."""

    def __init__(self, n: int):
        self.n, self.i, self.q = n, -1, deque()

    def update(self, x: float) -> float | None:
        self.i += 1
        while self.q and self.q[-1][1] <= x:
            self.q.pop()                                   # can never be the max again
        self.q.append((self.i, x))
        if self.q[0][0] <= self.i - self.n:
            self.q.popleft()                               # fell out of the window
        return self.q[0][1] if self.i >= self.n - 1 else None


class StreamingStd:
    """Rolling POPULATION std over n values with Welford-style add/remove updates."""

    def __init__(self, n: int):
        self.n, self.buf, self.mean, self.m2 = n, deque(), 0.0, 0.0

    def update(self, x: float) -> float | None:
        self.buf.append(x)
        if len(self.buf) <= self.n:
            k = len(self.buf)
            d = x - self.mean
            self.mean += d / k
            self.m2 += d * (x - self.mean)
        else:
            old = self.buf.popleft()
            new_mean = self.mean + (x - old) / self.n
            self.m2 += (x - old) * (x - new_mean + old - self.mean)
            self.mean = new_mean
        return float(np.sqrt(max(self.m2, 0.0) / self.n)) if len(self.buf) == self.n else None


def stream(obj, x) -> np.ndarray:
    """Feed x one value at a time; None → NaN."""
    return np.array([np.nan if (v := obj.update(float(xi))) is None else v for xi in x])


# --------------------------------------------------------------------------------------
# 03 · candlesticks and edge testing
# --------------------------------------------------------------------------------------
def anatomy(o, h, l, c) -> dict[str, np.ndarray]:  # noqa: E741
    """body |C−O|, range H−L, upper shadow H−max(O,C), lower shadow min(O,C)−L, body_pct = body/range (0 on a
    flat bar)."""
    body, rng_ = np.abs(c - o), h - l
    return {"body": body, "range": rng_, "upper": h - np.maximum(o, c), "lower": np.minimum(o, c) - l,
            "body_pct": np.divide(body, rng_, out=np.zeros_like(body), where=rng_ > 0)}


def bullish_engulfing(o, h, l, c) -> np.ndarray:  # noqa: E741
    """Bar t−1 bearish, bar t bullish, and t's body covers t−1's body (O_t <= C_{t−1}, C_t >= O_{t−1}) and is larger."""
    out = np.zeros(c.shape, dtype=bool)
    po, pc = o[:-1], c[:-1]
    out[1:] = (pc < po) & (c[1:] > o[1:]) & (o[1:] <= pc) & (c[1:] >= po) & ((c[1:] - o[1:]) > (po - pc))
    return out


def hammer(o, h, l, c, trend_n: int = 10, min_lower: float = 2.0, max_upper: float = 0.1) -> np.ndarray:  # noqa: E741
    """Long lower shadow (>= min_lower × body), tiny upper shadow (<= max_upper × range), AFTER a decline
    (previous close below its SMA(trend_n))."""
    a = anatomy(o, h, l, c)
    shape = (a["lower"] >= min_lower * np.maximum(a["body"], 1e-12)) & (a["upper"] <= max_upper * a["range"])
    down = c < sma(c, trend_n)
    prev_down = np.concatenate([[False], down[:-1]])
    return shape & prev_down


def forward_returns(open_: np.ndarray, horizon: int) -> np.ndarray:
    """Return of a trade entered at the NEXT bar's open and exited `horizon` bars later at the open:
    r[t] = O[t+1+h]/O[t+1] − 1. NaN where the exit is beyond the data."""
    o = np.asarray(open_, dtype=float)
    out = np.full(o.shape, np.nan)
    if o.size > horizon + 1:
        out[: o.size - horizon - 1] = o[1 + horizon:] / o[1:o.size - horizon] - 1
    return out


def permutation_pvalue(signal: np.ndarray, fwd: np.ndarray, n_perm: int = 2000, seed: int = 0) -> tuple[float, float]:
    """(mean forward return after the signal, one-sided p-value): the share of random bar sets of the same size whose
    mean is at least as high, with the +1 correction."""
    ok = ~np.isnan(fwd)
    sig = signal & ok
    k = int(sig.sum())
    if k == 0:
        return np.nan, 1.0
    obs = float(fwd[sig].mean())
    pool = fwd[ok]
    rng = np.random.default_rng(seed)
    perm = np.array([pool[rng.choice(pool.size, k, replace=False)].mean() for _ in range(n_perm)])
    return obs, float((1 + (perm >= obs).sum()) / (1 + n_perm))


def bh_adjust(p_values) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


# --------------------------------------------------------------------------------------
# 04 · volatility and momentum
# --------------------------------------------------------------------------------------
def brownian_bars(n: int = 500, sigma: float = 0.01, steps: int = 390, seed: int = 0) -> pd.DataFrame:
    """Daily OHLC from a driftless intraday random walk in log price with KNOWN daily σ; each day opens at the
    previous close (no overnight gap)."""
    rng = np.random.default_rng(seed)
    lp = np.concatenate([[0.0], np.cumsum(rng.normal(0, sigma / np.sqrt(steps), n * steps))])
    days = np.stack([lp[d * steps:(d + 1) * steps + 1] for d in range(n)])
    px = 100 * np.exp(days)
    return pd.DataFrame({"open": px[:, 0], "high": px.max(axis=1), "low": px.min(axis=1), "close": px[:, -1]})


def close_to_close_vol(c) -> float:
    r = np.diff(np.log(c))
    return float(np.sqrt(np.mean(r ** 2)))


def parkinson_vol(h, l) -> float:  # noqa: E741
    """sqrt(mean(ln(H/L)²) / (4 ln 2)): daily σ from the range."""
    return float(np.sqrt(np.mean(np.log(h / l) ** 2) / (4 * np.log(2))))


def garman_klass_vol(o, h, l, c) -> float:  # noqa: E741
    """sqrt(mean(½ ln(H/L)² − (2 ln 2 − 1) ln(C/O)²))."""
    return float(np.sqrt(np.mean(0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2)))


def efficiency_ratio(close, n: int = 10) -> np.ndarray:
    """ER[t] = |C[t] − C[t−n]| / Σ|ΔC| over the same n changes (0 if that sum is 0); NaN for t < n."""
    c = np.asarray(close, dtype=float)
    out = np.full(c.shape, np.nan)
    if c.size <= n:
        return out
    change = np.abs(c[n:] - c[:-n])
    vol = pd.Series(np.abs(np.diff(c))).rolling(n).sum().to_numpy()[n - 1:]
    out[n:] = np.divide(change, vol, out=np.zeros_like(change), where=vol > 0)
    return out


def kama(close, n: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Kaufman adaptive MA: sc = (ER·(2/(fast+1) − 2/(slow+1)) + 2/(slow+1))², starts from C[n−1]."""
    c = np.asarray(close, dtype=float)
    er = efficiency_ratio(c, n)
    fs, ss = 2 / (fast + 1), 2 / (slow + 1)
    out = np.full(c.shape, np.nan)
    k = c[n - 1]
    for t in range(n, c.size):
        sc = (er[t] * (fs - ss) + ss) ** 2
        k = k + sc * (c[t] - k)
        out[t] = k
    return out


def trend_then_chop(n: int = 400, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trend = np.r_[np.linspace(0, 8, n // 2), np.full(n - n // 2, 8.0)]
    return 100 + trend + rng.normal(0, 0.1, n)


# --------------------------------------------------------------------------------------
# 05 · price-volume-time and multi-timeframe
# --------------------------------------------------------------------------------------
def intraday_bars(days: int = 5, seed: int = 2) -> pd.DataFrame:
    """5-minute bars stamped at their CLOSE time (09:35 … 16:00 New York), with U-shaped volume."""
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range("2025-03-03", periods=days):
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=35), d + pd.Timedelta(hours=16), freq="5min",
                            tz="America/New_York")
        r = rng.normal(0.0001, 0.0015, len(idx))
        close = px * np.exp(np.cumsum(r))
        u = np.linspace(-1, 1, len(idx))
        vol = np.round(rng.gamma(4, 1, len(idx)) * (1 + 2.5 * u ** 2) * 10_000)
        frames.append(pd.DataFrame({"close": close, "volume": vol}, index=idx))
        px = close[-1] * np.exp(rng.normal(0, 0.004))           # overnight move
    return pd.concat(frames)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Σ(price·volume)/Σ(volume) from the session start, resetting each day (price = close here)."""
    day = df.index.date
    pv = (df["close"] * df["volume"]).groupby(day).cumsum()
    return pv / df["volume"].groupby(day).cumsum()


def anchored_vwap(price, volume, anchor: int) -> np.ndarray:
    price, volume = np.asarray(price, float), np.asarray(volume, float)
    out = np.full(price.shape, np.nan)
    out[anchor:] = np.cumsum(price[anchor:] * volume[anchor:]) / np.cumsum(volume[anchor:])
    return out


def point_of_control(price, volume, bins: int = 40) -> float:
    """Centre of the price bin holding the most volume (np.histogram with `bins` equal bins)."""
    hist, edges = np.histogram(price, bins=bins, weights=volume)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def align_higher_tf(close: pd.Series, rule: str = "1h") -> pd.Series:
    """Higher-timeframe close for each lower-timeframe bar, using only COMPLETED higher bars. Bars are stamped at their
    close: resample with closed='right', label='right' (each higher bar is stamped when it completes), drop empty
    ones, then forward-fill onto the lower-timeframe index."""
    h = close.resample(rule, closed="right", label="right").last().dropna()
    return h.reindex(close.index, method="ffill")


def align_higher_tf_leaky(close: pd.Series, rule: str = "1h") -> pd.Series:
    """The common bug: default resample labels each hour at its START, so 10:05 already sees the 11:00 close."""
    h = close.resample(rule).last().dropna()
    return h.reindex(close.index, method="ffill")


# --------------------------------------------------------------------------------------
# 06 · levels, pivots and swings
# --------------------------------------------------------------------------------------
def classic_pivots(high, low, close) -> dict[str, np.ndarray]:
    """Floor pivots for bar t from bar t−1's H, L, C: P=(H+L+C)/3, R1=2P−L, S1=2P−H, R2=P+(H−L), S2=P−(H−L).
    NaN at bar 0."""
    h, l, c = (np.concatenate([[np.nan], np.asarray(a, float)[:-1]]) for a in (high, low, close))  # noqa: E741
    pp = (h + l + c) / 3
    return {"P": pp, "R1": 2 * pp - l, "S1": 2 * pp - h, "R2": pp + (h - l), "S2": pp - (h - l)}


@dataclass(frozen=True)
class Pivot:
    idx: int          # bar of the swing
    known_from: int   # first bar at which the swing is confirmed (idx + k)
    price: float
    kind: int         # +1 swing high, −1 swing low


def fractals(high, low, k: int = 2) -> list[Pivot]:
    """Swing high at i if high[i] is STRICTLY above the k highs on each side (swing low: strictly below the k lows).
    Confirmed only k bars later: known_from = i + k. Ordered by idx, a low before a high at the same bar."""
    h, l = np.asarray(high, float), np.asarray(low, float)  # noqa: E741
    out = []
    for i in range(k, len(h) - k):
        nb = np.r_[i - k:i, i + 1:i + k + 1]
        if (l[i] < l[nb]).all():
            out.append(Pivot(i, i + k, float(l[i]), -1))
        if (h[i] > h[nb]).all():
            out.append(Pivot(i, i + k, float(h[i]), +1))
    return out


def zigzag(close, pct: float = 0.05) -> list[Pivot]:
    """Alternating swings of at least pct. A swing is confirmed (known_from) at the bar where price has reversed
    pct from it."""
    c = np.asarray(close, float)
    out, direction, ext = [], 0, 0
    for t in range(1, c.size):
        if direction == 0:
            if abs(c[t] / c[0] - 1) >= pct:
                direction, ext = (1 if c[t] > c[0] else -1), t
        elif direction == 1:
            if c[t] > c[ext]:
                ext = t
            elif c[t] <= c[ext] * (1 - pct):
                out.append(Pivot(ext, t, float(c[ext]), +1))
                direction, ext = -1, t
        else:
            if c[t] < c[ext]:
                ext = t
            elif c[t] >= c[ext] * (1 + pct):
                out.append(Pivot(ext, t, float(c[ext]), -1))
                direction, ext = 1, t
    return out


def swing_low_trades(pivots: list[Pivot], open_: np.ndarray, horizon: int = 10, use_known_from: bool = True) -> np.ndarray:
    """Buy at the next open after each swing low (after it is confirmed, or — the bug — right after the swing bar
    itself) and hold `horizon` bars. Returns the trade returns."""
    fwd = forward_returns(open_, horizon)
    bars = [(p.known_from if use_known_from else p.idx) for p in pivots if p.kind == -1]
    r = np.array([fwd[b] for b in bars if b < len(fwd)])
    return r[~np.isnan(r)]


# --------------------------------------------------------------------------------------
# 07 · actions and entry conditions
# --------------------------------------------------------------------------------------
def crossover(a, b) -> np.ndarray:
    """True at t when a[t] > b[t] and a[t−1] <= b[t−1] (a touch then a cross counts once). False where any of the four
    values is NaN, and at t = 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(a.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def bars_since(cond) -> np.ndarray:
    """Bars since cond was last True (0 on a True bar); NaN before the first True."""
    cond = np.asarray(cond, bool)
    out = np.full(cond.shape, np.nan)
    last = None
    for i, v in enumerate(cond):
        if v:
            last = i
        if last is not None:
            out[i] = i - last
    return out


def inside_bar(high, low) -> np.ndarray:
    h, l = np.asarray(high, float), np.asarray(low, float)  # noqa: E741
    out = np.zeros(h.shape, dtype=bool)
    out[1:] = (h[1:] < h[:-1]) & (l[1:] > l[:-1])
    return out


def nr_n(high, low, n: int = 7) -> np.ndarray:
    """Narrowest range of the last n bars (strictly narrower than the previous n−1)."""
    rng_ = np.asarray(high, float) - np.asarray(low, float)
    prev_min = pd.Series(rng_).shift(1).rolling(n - 1).min().to_numpy()
    with np.errstate(invalid="ignore"):
        return rng_ < prev_min


def within(cond, n: int) -> np.ndarray:
    """True at t if cond was True on any of the bars t−n+1 … t."""
    return pd.Series(np.asarray(cond, bool)).rolling(n, min_periods=1).max().to_numpy().astype(bool)


def confirm(cond, n: int) -> np.ndarray:
    """True at t if cond was True on all of the bars t−n+1 … t."""
    return pd.Series(np.asarray(cond, float)).rolling(n).min().fillna(0).to_numpy().astype(bool)


class Condition:
    """A boolean series you can combine: a & b, a | b, ~a, a.within(n), a.confirm(n)."""

    def __init__(self, values, name: str = "cond"):
        self.values, self.name = np.asarray(values, bool), name

    def __and__(self, o):
        return Condition(self.values & o.values, f"({self.name} & {o.name})")

    def __or__(self, o):
        return Condition(self.values | o.values, f"({self.name} | {o.name})")

    def __invert__(self):
        return Condition(~self.values, f"~{self.name}")

    def within(self, n: int):
        return Condition(within(self.values, n), f"{self.name}.within({n})")

    def confirm(self, n: int):
        return Condition(confirm(self.values, n), f"{self.name}.confirm({n})")

    def __repr__(self):
        return f"Condition({self.name}: {int(self.values.sum())} bars True)"


# --------------------------------------------------------------------------------------
# 08 · sentiment, news and tail risk
# --------------------------------------------------------------------------------------
def cot_releases(start: str = "2023-01-03", weeks: int = 104, seed: int = 5) -> pd.DataFrame:
    """Weekly COT-style positioning: measured on Tuesday (report_date), published Friday 15:30 New York
    (available_at, in UTC)."""
    rng = np.random.default_rng(seed)
    tue = pd.date_range(start, periods=weeks, freq="W-TUE")
    fri = (tue + pd.Timedelta(days=3)).tz_localize("America/New_York") + pd.Timedelta(hours=15, minutes=30)
    net = np.cumsum(rng.normal(0, 8000, weeks)).round()
    return pd.DataFrame({"report_date": tue.tz_localize("UTC"), "available_at": fri.tz_convert("UTC"), "net_long": net})


def daily_closes_utc(start: str = "2023-01-03", days: int = 520) -> pd.DataFrame:
    """Daily bars stamped at the 16:00 New York close, in UTC."""
    d = pd.bdate_range(start, periods=days).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    return pd.DataFrame({"ts": d.tz_convert("UTC")})


def pit_join(bars: pd.DataFrame, releases: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """For each bar, the latest release already PUBLISHED at the bar's time (merge_asof on available_at, backward)."""
    r = releases.sort_values("available_at")[["available_at", *value_cols]]
    return pd.merge_asof(bars.sort_values("ts"), r, left_on="ts", right_on="available_at", direction="backward")


NEGATORS = {"not", "no", "never", "without", "hardly", "isn't", "wasn't", "don't", "doesn't", "didn't", "won't"}
POSITIVE = {"beat", "beats", "strong", "growth", "record", "upgrade", "raises", "surge", "profit", "gains"}
NEGATIVE = {"miss", "misses", "weak", "loss", "downgrade", "cuts", "lawsuit", "plunge", "decline", "fraud"}
TOKEN = re.compile(r"[a-z']+")
CLAUSE = re.compile(r"[,.;:!?]|\bbut\b")

HEADLINES = [
    "Acme beats estimates, raises guidance",
    "Acme did not beat estimates",
    "Acme sees no decline in demand",
    "Profit growth strong but lawsuit looms",
    "Analyst downgrade after weak quarter",
    "Not a record quarter, but no loss either",
    "Shares plunge as fraud probe widens",
    "Company announces new CFO",
]


def lexicon_score(text: str, positive=POSITIVE, negative=NEGATIVE, window: int = 2) -> float:
    """Split the lowercased text into clauses (CLAUSE), tokenize each (TOKEN). A positive word is +1, a negative
    −1, flipped when one of the previous `window` tokens OF THE SAME CLAUSE is a negator. Score = mean of the signs
    (0.0 with no sentiment word)."""
    signs = []
    for clause in CLAUSE.split(text.lower()):
        toks = TOKEN.findall(clause)
        for i, t in enumerate(toks):
            s = 1 if t in positive else (-1 if t in negative else 0)
            if s == 0:
                continue
            if any(w in NEGATORS for w in toks[max(0, i - window):i]):
                s = -s
            signs.append(s)
    return float(sum(signs) / len(signs)) if signs else 0.0


def decay_index(event_ts: pd.Series, score: np.ndarray, grid: pd.DatetimeIndex, tau_hours: float = 24.0) -> np.ndarray:
    """At each grid time, Σ score·exp(−age/τ) over events already published (age >= 0)."""
    et = pd.DatetimeIndex(event_ts).as_unit("ns").asi8
    g = pd.DatetimeIndex(grid).as_unit("ns").asi8
    age_h = (g[:, None] - et[None, :]) / 3.6e12
    w = np.where(age_h >= 0, np.exp(-np.clip(age_h, 0, None) / tau_hours), 0.0)
    return w @ np.asarray(score, float)


def fat_tailed_returns(n: int = 2500, df_: float = 3.0, scale: float = 0.01, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_t(df_, n) * scale / np.sqrt(df_ / (df_ - 2))


def hill_estimator(losses, k: int) -> float:
    """Tail index ξ from the k largest losses: mean of ln(X_(i) / X_(k+1)) for i = 1..k (X sorted descending)."""
    x = np.sort(np.asarray(losses, float))[::-1]
    return float(np.mean(np.log(x[:k] / x[k])))


def hill_var(losses, k: int, q: float) -> float:
    """Weissman quantile: VaR_q = X_(k+1) · (k / (n·(1−q)))^ξ."""
    x = np.sort(np.asarray(losses, float))[::-1]
    return float(x[k] * (k / (x.size * (1 - q))) ** hill_estimator(losses, k))


def kupiec_pvalue(n: int, exceedances: int, p: float) -> float:
    """Kupiec proportion-of-failures test. With x exceedances in n days and p̂ = x/n:
    LR = −2 [ln L(p) − ln L(p̂)], ln L(q) = (n−x)·ln(1−q) + x·ln(q); p-value = P(χ²(1) > LR)."""
    from scipy.special import xlogy
    from scipy.stats import chi2

    x, phat = exceedances, exceedances / n

    def ll(q):
        return xlogy(n - x, 1 - q) + xlogy(x, q)
    return float(chi2.sf(-2 * (ll(p) - ll(phat)), 1))
