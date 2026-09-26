"""Week 17 — Library design, core indicators, streaming indicators, candlesticks (Part 5, S1–S4).

API conventions (from the lesson plan, non-negotiable):
  * inputs and outputs are float64 NumPy arrays; output length == input length;
  * warm-up positions are NaN, never dropped or back-filled;
  * the value at t uses only bars 0..t.
Golden tests compare your indicators with TA-Lib outputs stored in golden/core.npz (tolerance 1e-8), so you do not
need TA-Lib installed. Fill in every block marked "Your turn", then run:  python -m pytest week17_core
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------------ S1 registry
@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    group: str                          # direction | momentum | volatility | pvt | candles | ...
    outputs: tuple[str, ...]
    lookback: Callable[..., int]        # warm-up bars as a function of the parameters
    params: dict[str, tuple]            # name -> (default, min, max), used by the Part 8 optimizer


REGISTRY: dict[str, tuple[IndicatorSpec, Callable]] = {}


def indicator(name: str, group: str, lookback: Callable[..., int], params: dict[str, tuple],
              outputs: tuple[str, ...] = ("value",)):
    """Decorator: store (IndicatorSpec, fn) in REGISTRY under `name` and return fn unchanged.
    A name that is already registered raises ValueError (two functions must never share a name)."""
    def deco(fn):
        pass  # ✍️ Your turn: see the docstring
        return fn
    return deco


def default_lookback(name: str) -> int:
    """Look-back of a registered indicator with its default parameters."""
    spec, _ = REGISTRY[name]
    return spec.lookback(**{k: v[0] for k, v in spec.params.items()})


def catalog_markdown() -> str:
    """Markdown table of the registry, sorted by group then name, with this exact header:
    | Name | Group | Outputs | Params | Look-back |
    |---|---|---|---|---|
    Rows: | sma | direction | value | n=20 | 19 |   (params as 'k=default' joined by ', ', outputs joined by ', ')."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------- S1 helpers
def log_returns(x: np.ndarray) -> np.ndarray:
    """r[t] = ln(x[t] / x[t-1]); r[0] = NaN."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def rank_pct(x: np.ndarray, n: int) -> np.ndarray:
    """Percentile rank of x[t] inside its trailing window x[t-n+1..t]: the fraction of the n values that are
    <= x[t] (so the window maximum scores 1.0). NaN for the first n-1 bars. Used to normalize oscillators."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# -------------------------------------------------------------------- S2 core indicators
@indicator("sma", "direction", lambda n: n - 1, {"n": (20, 2, 200)})
def sma(x: np.ndarray, n: int = 20) -> np.ndarray:
    """Simple moving average. Look-back n-1."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def _ema_from(x: np.ndarray, n: int, start: int) -> np.ndarray:
    """EMA whose SMA seed covers x[start : start+n]; NaN before start+n-1 (given helper, used by ema and macd)."""
    out = np.full(x.shape, np.nan)
    s = start + n - 1
    if s >= x.size:
        return out
    alpha = 2.0 / (n + 1)
    out[s] = x[start:start + n].mean()
    for i in range(s + 1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


@indicator("ema", "direction", lambda n: n - 1, {"n": (20, 2, 200)})
def ema(x: np.ndarray, n: int = 20) -> np.ndarray:
    """EMA with alpha = 2/(n+1), SEEDED WITH THE SMA OF THE FIRST n VALUES (TA-Lib convention;
    pandas ewm(adjust=False) seeds with the first value instead). Look-back n-1."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("rsi", "momentum", lambda n: n, {"n": (14, 2, 100)})
def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI. Average gain/loss seeded with the simple mean of the first n changes, then
    avg = (avg × (n-1) + new) / n. RSI = 100 when the average loss is 0. Look-back n."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """TR[t] = max(H-L, |H-C[t-1]|, |L-C[t-1]|); TR[0] = NaN (no previous close)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("atr", "volatility", lambda n: n, {"n": (14, 2, 100)})
def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ATR: first value at bar n = mean(TR[1..n]); then (prev × (n-1) + TR) / n. Look-back n."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("macd", "momentum", lambda fast, slow, signal: slow + signal - 2,
           {"fast": (12, 2, 50), "slow": (26, 5, 100), "signal": (9, 2, 50)},
           outputs=("macd", "signal", "hist"))
def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """(macd, signal, hist) exactly as TA-Lib computes them:
      * the FAST EMA is seeded over bars slow-fast .. slow-1 (so fast and slow EMAs both start at bar slow-1);
      * macd = fast EMA − slow EMA from bar slow-1;
      * signal = EMA(signal) of the macd line, seeded over bars slow-1 .. slow+signal-2;  hist = macd − signal;
      * ALL THREE outputs are NaN before bar slow+signal-2 (the look-back)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("bbands", "volatility", lambda n, k: n - 1, {"n": (20, 2, 200), "k": (2.0, 0.5, 4.0)},
           outputs=("upper", "middle", "lower"))
def bbands(close: np.ndarray, n: int = 20, k: float = 2.0):
    """Bollinger Bands (upper, middle, lower): middle = SMA(n), bands = middle ± k × POPULATION std (ddof=0)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def _rolling(x: np.ndarray, n: int, fn) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if x.size >= n:
        out[n - 1:] = fn(np.lib.stride_tricks.sliding_window_view(x, n), axis=1)
    return out


@indicator("stoch", "momentum", lambda k, smooth_k, d: k + smooth_k + d - 3,
           {"k": (14, 2, 100), "smooth_k": (3, 1, 10), "d": (3, 1, 10)}, outputs=("k", "d"))
def stoch(high, low, close, k: int = 14, smooth_k: int = 3, d: int = 3):
    """Slow stochastic as TA-Lib STOCH with SMA smoothing:
    fast %K = 100 × (C − lowest low_k) / (highest high_k − lowest low_k) (0 when the range is 0);
    slow %K = SMA(smooth_k) of fast %K; %D = SMA(d) of slow %K. BOTH outputs NaN before k+smooth_k+d-3."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("willr", "momentum", lambda n: n - 1, {"n": (14, 2, 100)})
def willr(high, low, close, n: int = 14) -> np.ndarray:
    """Williams %R = −100 × (highest high_n − C) / (highest high_n − lowest low_n); 0 when the range is 0."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("obv", "pvt", lambda: 0, {})
def obv(close, volume) -> np.ndarray:
    """On-balance volume, TA-Lib convention: OBV[0] = volume[0]; then add the volume on an up close, subtract it
    on a down close, unchanged on an equal close. Look-back 0."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ----------------------------------------------------------------- S3 streaming indicators
class StreamingEMA:
    """O(1) EMA identical to ema(x, n) on every prefix. update(x) consumes one bar and returns the value
    (None during warm-up). peek(x) returns what update(x) WOULD return, without changing any state
    (for the still-forming bar)."""

    def __init__(self, n: int):
        self.n, self.alpha = n, 2.0 / (n + 1)
        self._seed: list[float] = []
        self.value: float | None = None

    def update(self, x: float) -> float | None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def peek(self, x: float) -> float | None:
        raise NotImplementedError("✍️ Your turn: see the docstring")


class StreamingRSI:
    """O(1) Wilder RSI identical to rsi(close, n) on every prefix; update(close) returns None during warm-up."""

    def __init__(self, n: int = 14):
        self.n = n
        self.prev: float | None = None
        self._gains: list[float] = []
        self._losses: list[float] = []
        self.ag: float | None = None
        self.al: float | None = None

    def update(self, close: float) -> float | None:
        raise NotImplementedError("✍️ Your turn: see the docstring")


class RollingMax:
    """Maximum of the last n values in amortized O(1) with a monotonic deque of (index, value), values
    decreasing from front to back. update(x) returns the max once n values were seen, else None."""

    def __init__(self, n: int):
        self.n, self.i = n, -1
        self.dq: deque[tuple[int, float]] = deque()

    def update(self, x: float) -> float | None:
        raise NotImplementedError("✍️ Your turn: see the docstring")


class StreamingBollinger:
    """O(1) Bollinger Bands over the last n closes using a ROLLING WELFORD update of (mean, M2), where
    M2 = sum of squared deviations from the mean. While the window fills, use the classic Welford step:
        d = x − mean;  mean += d / count;  M2 += d × (x − mean)
    Once full, replace the oldest value `old` by x in one step:
        new_mean = mean + (x − old) / n;  M2 += (x − old) × (x − new_mean + old − mean);  mean = new_mean
    Population variance = max(M2, 0) / n. (A running sum of squares cancels catastrophically after a large value
    leaves the window; Welford avoids most of that.) update(x) returns (upper, middle, lower), or None in warm-up."""

    def __init__(self, n: int = 20, k: float = 2.0):
        self.n, self.k = n, k
        self.window: deque[float] = deque()
        self.mean = self.m2 = 0.0

    def update(self, x: float):
        raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------- S4 candlesticks
def anatomy(o, h, l, c) -> dict[str, np.ndarray]:  # noqa: E741
    """Candle features: body = |c-o|, range = h-l, upper = h - max(o,c), lower = min(o,c) - l,
    body_pct = body / range (0 when range == 0)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("doji", "candles", lambda max_body_pct: 0, {"max_body_pct": (0.1, 0.01, 0.3)})
def doji(o, h, l, c, max_body_pct: float = 0.1) -> np.ndarray:  # noqa: E741
    """+1 where body <= max_body_pct × range and range > 0 (neutral pattern, reported as +1), else 0. int8."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("engulfing", "candles", lambda: 1, {})
def engulfing(o, h, l, c) -> np.ndarray:  # noqa: E741
    """+1 bullish: previous candle black (c<o), current white, and the current body engulfs the previous one:
    (c >= prev_o and o < prev_c) or (c > prev_o and o <= prev_c). −1 bearish: the mirror image. 0 otherwise.
    Bar 0 is always 0. int8. (Matches the sign of TA-Lib CDLENGULFING.)"""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("hammer", "candles", lambda trend_n, min_lower, max_upper: trend_n,
           {"trend_n": (10, 3, 50), "min_lower": (2.0, 1.0, 4.0), "max_upper": (0.1, 0.0, 0.3)})
def hammer(o, h, l, c, trend_n: int = 10, min_lower: float = 2.0, max_upper: float = 0.1) -> np.ndarray:  # noqa: E741
    """+1 on a hammer AFTER A DECLINE: lower shadow >= min_lower × body, upper shadow <= max_upper × range,
    body > 0, and the PREVIOUS close below SMA(trend_n) of close (context known before this bar). int8."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@indicator("morning_evening_star", "candles", lambda: 2, {})
def morning_evening_star(o, h, l, c, star_pct: float = 0.3) -> np.ndarray:  # noqa: E741
    """Three-bar reversal, signal on the THIRD bar.
    +1 morning star: bar1 long black (body_pct >= 0.6), bar2 small body (body_pct <= star_pct) with its body entirely
       below bar1's close, bar3 white closing above the midpoint of bar1's body.
    −1 evening star: the mirror (bar1 long white, bar2 small with body above bar1's close, bar3 black closing below
       bar1's midpoint). int8; the first two bars are 0."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
