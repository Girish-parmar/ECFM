"""Helper functions for the Part 7 guided notebooks (MFAAT, Month 6: the strategy library).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

All data is SYNTHETIC. The market is built from blocks whose regime is KNOWN (trending or range-bound, calm or
volatile), so you can see which strategy works where. No strategy here is presented as profitable: each is a hypothesis
with a first-look evaluation, and the honest backtest comes in Part 8. The graded versions are in labs/part07/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P7_STRICT") == "1"      # tests: a failed check raises instead of continuing


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
# Indicators and pricing from earlier parts (complete, so this part stands alone)
# --------------------------------------------------------------------------------------
def sma(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.size >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rsi(close, n=14):
    close = np.asarray(close, dtype=float)
    out = np.full(close.shape, np.nan)
    if close.size <= n:
        return out
    d = np.diff(close)
    g, lo = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = g[:n].mean(), lo[:n].mean()
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, close.size):
        ag, al = (ag * (n - 1) + g[i - 1]) / n, (al * (n - 1) + lo[i - 1]) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def bsm_price(S, K, T, r, q, sigma, cp):
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    return cp * (S * np.exp(-q * T) * norm.cdf(cp * d1) - K * np.exp(-r * T) * norm.cdf(cp * d2))


def bsm_greeks(S, K, T, r, q, sigma, cp) -> dict:
    """Raw units: delta, gamma, vega per 1.00 σ, theta per year."""
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    dq, dr, pdf = np.exp(-q * T), np.exp(-r * T), norm.pdf(d1)
    return {"delta": cp * dq * norm.cdf(cp * d1), "gamma": dq * pdf / (S * sigma * sq), "vega": S * dq * pdf * sq,
            "theta": -S * dq * pdf * sigma / (2 * sq) - cp * r * K * dr * norm.cdf(cp * d2)
            + cp * q * S * dq * norm.cdf(cp * d1)}


# --------------------------------------------------------------------------------------
# Synthetic markets with known regimes
# --------------------------------------------------------------------------------------
def regime_market(n_blocks: int = 24, block: int = 125, seed: int = 0) -> pd.DataFrame:
    """Daily bars from blocks with a KNOWN regime: trend (1 = trending up or down, 0 = range-bound and mean
    reverting) and high_vol (1 = 28% annual vol, 0 = 10%)."""
    rng = np.random.default_rng(seed)
    closes, opens, tl, vl = [], [], [], []
    px = 100.0
    for b in range(n_blocks):
        trend = b % 2 == 0 if rng.random() < 0.8 else b % 2 == 1
        high = rng.random() < 0.5
        sig = (0.28 if high else 0.10) / np.sqrt(252)
        drift = rng.choice([-1, 1]) * 0.15 * sig if trend else 0.0
        anchor = px
        for _ in range(block):
            o = px * np.exp(rng.normal(0, 0.25 * sig))
            lr = drift + rng.normal(0, sig) if trend else -0.08 * np.log(o / anchor) + rng.normal(0, sig)
            px = o * np.exp(lr)
            opens.append(o)
            closes.append(px)
            tl.append(int(trend))
            vl.append(int(high))
    o, c = np.array(opens), np.array(closes)
    span = np.abs(np.log(c / o)) + rng.gamma(2.0, 0.003, c.size)
    h = np.maximum(o, c) * np.exp(rng.uniform(0.1, 0.6, c.size) * span)
    lo = np.minimum(o, c) * np.exp(-rng.uniform(0.1, 0.6, c.size) * span)
    idx = pd.bdate_range("2012-01-02", periods=c.size)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "trend": tl, "high_vol": vl}, index=idx)


def regime_table(bars: pd.DataFrame, signals: dict[str, np.ndarray], cost_bps: float = 2.0) -> pd.DataFrame:
    """Annualized Sharpe of each strategy's first-look P&L inside each known regime."""
    names = {(0, 0): "range · calm", (0, 1): "range · volatile", (1, 0): "trend · calm", (1, 1): "trend · volatile"}
    lab = [names[(t, v)] for t, v in zip(bars["trend"], bars["high_vol"])]
    out = {}
    for s, sig in signals.items():
        pnl = pd.Series(quick_eval(sig, bars["open"].to_numpy(), cost_bps)["pnl"], index=bars.index)
        out[s] = pnl.groupby(lab).apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan)
    return pd.DataFrame(out).T[list(names.values())]


# --------------------------------------------------------------------------------------
# 01 · the framework and the first-look evaluator
# --------------------------------------------------------------------------------------
@dataclass
class Intent:
    t: int
    symbol: str
    weight: float
    strategy: str
    reason: str


@dataclass
class StrategyContext:
    """What a strategy may touch: past bars only, the current bar index, and an intent sink."""
    bars: pd.DataFrame
    symbol: str = "X"
    t: int = -1
    intents: list = field(default_factory=list)

    def history(self, n: int | None = None) -> pd.DataFrame:
        start = 0 if n is None else max(0, self.t + 1 - n)
        return self.bars.iloc[start:self.t + 1]


class Strategy:
    """Template Method: the runner calls on_start, on_bar for every bar, on_stop. Subclasses write on_bar and emit
    intents with self.target(weight, reason); they never talk to a broker."""
    name = "base"
    params: dict = {}

    def __init__(self, ctx: StrategyContext, **overrides):
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        self.p = {k: v[0] for k, v in self.params.items()}
        for k, v in overrides.items():
            lo, hi = self.params[k][1], self.params[k][2]
            if not lo <= v <= hi:
                raise ValueError(f"{self.name}: {k}={v} outside [{lo}, {hi}]")
            self.p[k] = v
        self.ctx = ctx

    def on_start(self):
        pass

    def on_bar(self):
        raise NotImplementedError

    def on_stop(self):
        pass

    def target(self, weight: float, reason: str):
        self.ctx.intents.append(Intent(self.ctx.t, self.ctx.symbol, float(weight), self.name, reason))


REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        if name in REGISTRY:
            raise ValueError(f"strategy {name!r} already registered")
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def run(strategy_cls, bars: pd.DataFrame, symbol: str = "X", **params) -> np.ndarray:
    """decided[t] = weight in force after bar t's decisions (the last intent of the bar wins; none keeps the previous
    weight; start at 0)."""
    ctx = StrategyContext(bars, symbol)
    s = strategy_cls(ctx, **params)
    s.on_start()
    decided, w = np.zeros(len(bars)), 0.0
    for t in range(len(bars)):
        ctx.t = t
        before = len(ctx.intents)
        s.on_bar()
        if len(ctx.intents) > before:
            w = ctx.intents[-1].weight
        decided[t] = w
    s.on_stop()
    return decided


def quick_eval(decided, open_, cost_bps: float = 2.0, periods_per_year: int = 252) -> dict:
    """decided[t] is filled at the OPEN of bar t+1: pos[t] = decided[t−1] (pos[0] = 0); ret[t] = open[t+1]/open[t] − 1
    (0 on the last bar); turnover[t] = |pos[t] − pos[t−1]|; pnl = pos·ret − turnover·cost_bps/1e4."""
    decided = np.nan_to_num(np.asarray(decided, dtype=float))
    open_ = np.asarray(open_, dtype=float)
    pos = np.zeros_like(decided)
    pos[1:] = decided[:-1]
    ret = np.zeros_like(open_)
    ret[:-1] = open_[1:] / open_[:-1] - 1
    turnover = np.abs(np.diff(pos, prepend=0.0))
    pnl = pos * ret - turnover * cost_bps / 1e4
    equity = np.cumprod(1 + pnl)
    dd = equity / np.maximum.accumulate(equity) - 1
    sd = pnl.std()
    return {"cagr": equity[-1] ** (periods_per_year / len(pnl)) - 1,
            "sharpe": pnl.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan,
            "max_dd": dd.min(), "exposure": float(np.mean(pos != 0)),
            "turnover": turnover.sum() / len(pnl) * periods_per_year, "pnl": pnl}


def summary(res: dict) -> dict:
    return {k: round(float(v), 3) for k, v in res.items() if k != "pnl"}


# --------------------------------------------------------------------------------------
# 02 · momentum
# --------------------------------------------------------------------------------------
def tsmom(close, lookback: int = 252, vol_n: int = 60, target_vol: float = 0.10, max_lev: float = 2.0) -> np.ndarray:
    """sign(close[t]/close[t−lookback] − 1) × target_vol / realized vol (rolling std of daily log returns, ddof=1,
    ×√252), clipped to ±max_lev; NaN in warm-up."""
    close = np.asarray(close, dtype=float)
    past = np.full(close.shape, np.nan)
    past[lookback:] = close[lookback:] / close[:-lookback] - 1
    r = np.full(close.shape, np.nan)
    r[1:] = np.diff(np.log(close))
    vol = pd.Series(r).rolling(vol_n).std().to_numpy() * np.sqrt(252)
    return np.clip(np.sign(past) * target_vol / vol, -max_lev, max_lev)


def donchian_breakout(high, low, close, entry_n: int = 20, exit_n: int = 10) -> np.ndarray:
    """Flat → +1 if close > max of the PRIOR entry_n highs, −1 if close < min of the prior entry_n lows.
    Long → 0 if close < min of the prior exit_n lows. Short → 0 if close > max of the prior exit_n highs."""
    h, lo, c = (np.asarray(a, dtype=float) for a in (high, low, close))
    pos = np.zeros(c.size)
    for t in range(max(entry_n, exit_n), c.size):
        prev = pos[t - 1]
        if prev == 0:
            if c[t] > h[t - entry_n:t].max():
                pos[t] = 1
            elif c[t] < lo[t - entry_n:t].min():
                pos[t] = -1
        elif prev == 1:
            pos[t] = 0 if c[t] < lo[t - exit_n:t].min() else 1
        else:
            pos[t] = 0 if c[t] > h[t - exit_n:t].max() else -1
    return pos


def momentum_universe(n_assets: int = 12, months: int = 180, seed: int = 4) -> pd.DataFrame:
    """Month-end prices of assets whose expected return is PERSISTENT (a slowly changing drift per asset), plus noise:
    the world in which 12-1 momentum works."""
    rng = np.random.default_rng(seed)
    mu = np.zeros((months, n_assets))
    mu[0] = rng.normal(0, 0.008, n_assets)
    for t in range(1, months):
        mu[t] = 0.98 * mu[t - 1] + rng.normal(0, 0.002, n_assets)
    r = mu + rng.normal(0, 0.04, (months, n_assets))
    idx = pd.date_range("2010-01-31", periods=months, freq="ME")
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"A{i:02d}" for i in range(n_assets)])


def cross_sectional_momentum(closes: pd.DataFrame, top_q: float = 0.25) -> pd.DataFrame:
    """12-1 momentum = closes.shift(1)/closes.shift(12) − 1; equal-weight names ranked strictly above 1 − top_q."""
    mom = closes.shift(1) / closes.shift(12) - 1
    rank = mom.rank(axis=1, pct=True)
    w = (rank > 1 - top_q + 1e-12).astype(float)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)


# --------------------------------------------------------------------------------------
# 03 · mean reversion and range-bound
# --------------------------------------------------------------------------------------
def rsi2_reversion(close, entry: float = 10, exit_: float = 70, trend_n: int = 200, use_trend: bool = True,
                   max_hold: int | None = None) -> np.ndarray:
    """Long-only: flat → long when RSI(2) < entry (and close > SMA(trend_n) if use_trend); long → flat when RSI(2) >
    exit_ or after max_hold bars in the trade."""
    close = np.asarray(close, dtype=float)
    r, m = rsi(close, 2), sma(close, trend_n)
    pos, held = np.zeros(close.size), 0
    for t in range(1, close.size):
        if pos[t - 1] == 0:
            ok = r[t] < entry and (not use_trend or close[t] > m[t])
            pos[t], held = (1.0, 1) if ok else (0.0, 0)
        else:
            held += 1
            done = r[t] > exit_ or (max_hold is not None and held > max_hold)
            pos[t] = 0.0 if done else 1.0
    return pos


def ibs(high, low, close) -> np.ndarray:
    """(C − L)/(H − L); 0.5 on a flat bar."""
    h, lo, c = (np.asarray(a, dtype=float) for a in (high, low, close))
    rng_ = h - lo
    return np.divide(c - lo, rng_, out=np.full(c.shape, 0.5), where=rng_ > 0)


def rolling_z(close, n: int = 20) -> np.ndarray:
    c = np.asarray(close, dtype=float)
    sd = np.full(c.shape, np.nan)
    if c.size >= n:
        sd[n - 1:] = np.lib.stride_tricks.sliding_window_view(c, n).std(axis=1)
    return (c - sma(c, n)) / sd


def zscore_reversion(close, n: int = 20, entry: float = 2.0, exit_: float = 0.5, max_hold: int = 10) -> np.ndarray:
    """Flat → +1 if z < −entry, −1 if z > entry. In a trade → 0 when |z| < exit_ or after max_hold bars (time stop)."""
    z = rolling_z(close, n)
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


# --------------------------------------------------------------------------------------
# 04 · either-way, volatility, mathematical and statistical groups
# --------------------------------------------------------------------------------------
def vol_target_overlay(pos, returns, target: float = 0.10, n: int = 20, max_lev: float = 2.0) -> np.ndarray:
    """pos × clip(target / annualized rolling std of `returns` (ddof=1, over bars t−n+1..t), 0, max_lev); 0 where the vol
    is not known yet."""
    vol = pd.Series(np.asarray(returns, dtype=float)).rolling(n).std().to_numpy() * np.sqrt(252)
    scale = np.clip(np.divide(target, vol, out=np.zeros_like(vol), where=vol > 0), 0, max_lev)
    return np.asarray(pos, dtype=float) * np.nan_to_num(scale)


def hurst_exponent(x, lags=range(2, 64)) -> float:
    """Slope of log std(x[t+lag] − x[t]) on log lag: ≈ 0.5 random walk, > 0.5 trending, < 0.5 mean reverting."""
    x = np.asarray(x, dtype=float)
    lags = np.asarray(list(lags))
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def cointegrated_pair(n: int = 1500, beta: float = 1.4, seed: int = 8, break_at: int | None = None):
    """Log prices y, x with y = β·x + an OU spread; optionally the relationship breaks (β jumps) at `break_at`."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.012, n)) + 4.0
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.95 * s[t - 1] + rng.normal(0, 0.006)
    b = np.full(n, beta)
    if break_at is not None:
        b[break_at:] = beta * 0.6
    return b * x + s + 0.5, x


def pairs_positions(y, x, lookback: int = 60, entry: float = 2.0, exit_: float = 0.5):
    """At each t >= lookback: OLS β on the PREVIOUS lookback bars, spread and z of today's spread against that window.
    Flat → +1 (long y, short β·x) if z < −entry, −1 if z > entry; in a trade → 0 when |z| < exit_."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
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


def pairs_pnl(pos, y, x, betas) -> np.ndarray:
    """Daily P&L of holding pos[t−1] units of the spread (log returns: Δy − β·Δx with the β chosen when entering)."""
    dy, dx = np.diff(y, prepend=y[0]), np.diff(x, prepend=x[0])
    p_ = np.concatenate([[0.0], pos[:-1]])
    b_ = np.concatenate([[np.nan], betas[:-1]])
    return np.nan_to_num(p_ * (dy - b_ * dx))


def turn_of_month(index: pd.DatetimeIndex, last_days: int = 1, first_days: int = 3) -> np.ndarray:
    """1 on the last `last_days` and first `first_days` trading days of each month."""
    s = pd.Series(1, index=index)
    month = index.to_period("M")
    pos_in = s.groupby(month).cumcount().to_numpy()
    from_end = s.groupby(month).cumcount(ascending=False).to_numpy()
    return ((pos_in < first_days) | (from_end < last_days)).astype(float)


# --------------------------------------------------------------------------------------
# 05–07 · the option strategy builder and templates
# --------------------------------------------------------------------------------------
@dataclass
class Leg:
    cp: int          # +1 call, −1 put, 0 the underlying
    K: float
    T: float
    qty: int
    iv: float = 0.2


@dataclass
class OptionStrategy:
    name: str
    legs: list = field(default_factory=list)
    multiplier: int = 100

    def add(self, cp, K, T, qty, iv=0.2):
        self.legs.append(Leg(cp, K, T, qty, iv))
        return self

    @property
    def first_expiry(self) -> float:
        return min(L.T for L in self.legs if L.cp != 0)

    def value(self, S, r=0.04, q=0.0, dt=0.0, dvol=0.0):
        """Dollar value after dt years and a vol shift dvol: option legs at BSM (intrinsic once expired), the
        underlying at qty·S, all × multiplier. Works for scalar or array S."""
        S = np.asarray(S, dtype=float)
        v = np.zeros_like(S)
        for L in self.legs:
            if L.cp == 0:
                v = v + L.qty * S
                continue
            tau = L.T - dt
            leg = bsm_price(S, L.K, tau, r, q, L.iv + dvol, L.cp) if tau > 1e-9 else np.maximum(L.cp * (S - L.K), 0.0)
            v = v + L.qty * leg
        return v * self.multiplier

    def payoff_at_first_expiry(self, S, r=0.04, q=0.0):
        return self.value(S, r, q, dt=self.first_expiry)

    def greeks(self, S, r=0.04, q=0.0) -> dict:
        """Net delta and gamma in shares, vega per vol point, theta per day, all × multiplier."""
        tot = dict.fromkeys(("delta", "gamma", "vega", "theta"), 0.0)
        for L in self.legs:
            if L.cp == 0:
                tot["delta"] += L.qty
                continue
            g = bsm_greeks(S, L.K, L.T, r, q, L.iv, L.cp)
            tot["delta"] += L.qty * g["delta"]
            tot["gamma"] += L.qty * g["gamma"]
            tot["vega"] += L.qty * g["vega"] / 100
            tot["theta"] += L.qty * g["theta"] / 365
        return {k: float(v * self.multiplier) for k, v in tot.items()}

    def analyze(self, S, r=0.04, q=0.0, sigma=None, grid=None) -> dict:
        """cost (debit > 0), P&L at the first expiry on a grid, breakevens, max profit/loss, POP (lognormal)."""
        grid = np.linspace(0.5 * S, 1.5 * S, 2001) if grid is None else np.asarray(grid, dtype=float)
        cost = float(self.value(S, r, q))
        pnl = self.payoff_at_first_expiry(grid, r, q) - cost
        sign = np.sign(pnl)
        T0 = self.first_expiry
        sigma = sigma or float(np.mean([L.iv for L in self.legs if L.cp != 0]))
        z = (np.log(grid / S) - (r - q - 0.5 * sigma ** 2) * T0) / (sigma * np.sqrt(T0))
        dens = norm.pdf(z) / (grid * sigma * np.sqrt(T0))
        return {"cost": cost, "breakevens": grid[1:][sign[1:] != sign[:-1]].round(2),
                "max_profit": float(pnl.max()), "max_loss": float(pnl.min()),
                "pop": float(np.trapezoid(dens * (pnl > 0), grid))}

    def is_defined_risk(self) -> bool:
        calls = sum(L.qty for L in self.legs if L.cp == 1)
        puts = sum(L.qty for L in self.legs if L.cp == -1)
        shares = sum(L.qty for L in self.legs if L.cp == 0)
        return calls + shares >= 0 and puts >= 0


def skew_iv(S: float, atm: float = 0.18, slope: float = -0.8, curve: float = 1.5):
    """A simple equity smile: iv(K) = atm + slope·ln(K/S) + curve·ln(K/S)² (floored at 5%)."""
    return lambda K: float(max(0.05, atm + slope * np.log(K / S) + curve * np.log(K / S) ** 2))


def strike_for_delta(S, T, r, q, iv_fn, cp, target, strikes) -> float:
    K = np.asarray(strikes, dtype=float)
    d = np.array([bsm_greeks(S, k, T, r, q, iv_fn(k), cp)["delta"] for k in K])
    return float(K[np.argmin(np.abs(d - target))])


def iron_condor(S, T, iv_fn, strikes, short_delta=0.16, wing=10.0, r=0.04, q=0.0, qty=1) -> OptionStrategy:
    kp = strike_for_delta(S, T, r, q, iv_fn, -1, -short_delta, strikes)
    kc = strike_for_delta(S, T, r, q, iv_fn, 1, short_delta, strikes)
    return (OptionStrategy("iron condor").add(-1, kp - wing, T, qty, iv_fn(kp - wing)).add(-1, kp, T, -qty, iv_fn(kp))
            .add(1, kc, T, -qty, iv_fn(kc)).add(1, kc + wing, T, qty, iv_fn(kc + wing)))


def vertical(cp, k_long, k_short, T, iv_fn, qty=1) -> OptionStrategy:
    if cp == 1:
        name = "bull call spread" if k_long < k_short else "bear call spread"
    else:
        name = "bear put spread" if k_long > k_short else "bull put spread"
    return OptionStrategy(name).add(cp, k_long, T, qty, iv_fn(k_long)).add(cp, k_short, T, -qty, iv_fn(k_short))


def iv_rank(iv_history, window: int = 252) -> float:
    """(today − min)/(max − min)·100 over the last `window` values; 50 if flat."""
    x = np.asarray(iv_history, dtype=float)[-window:]
    lo, hi = x.min(), x.max()
    return 50.0 if hi == lo else float((x[-1] - lo) / (hi - lo) * 100)


def iv_percentile(iv_history, window: int = 252) -> float:
    """Share (%) of the previous window−1 values strictly below today's."""
    x = np.asarray(iv_history, dtype=float)[-window:]
    return float(np.mean(x[:-1] < x[-1]) * 100)


def iv_history(n: int = 600, seed: int = 6) -> np.ndarray:
    """A 30-day implied vol series: mean reverting around 18% with occasional spikes."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.18
    for t in range(1, n):
        shock = rng.normal(0, 0.008) + (rng.random() < 0.01) * rng.uniform(0.08, 0.2)
        x[t] = max(0.09, x[t - 1] + 0.04 * (0.18 - x[t - 1]) + shock)
    return x


def select_structure(direction: int, ivr: float, term_slope: float = 0.0, event: bool = False,
                     implied_move: float | None = None, expected_move: float | None = None) -> str:
    """event with implied < expected move → long_straddle; direction ±1 by IV rank tiers (<30 buy options, 30–70
    debit spread, >=70 credit spread); direction 0: ivr >= 50 iron_condor, else contango → calendar, else no_trade."""
    if event and implied_move is not None and expected_move is not None and implied_move < expected_move:
        return "long_straddle"
    if direction != 0:
        tiers = (("long_call", "bull_call_spread", "bull_put_spread") if direction > 0
                 else ("long_put", "bear_put_spread", "bear_call_spread"))
        return tiers[0] if ivr < 30 else (tiers[1] if ivr < 70 else tiers[2])
    if ivr >= 50:
        return "iron_condor"
    return "calendar" if term_slope > 0 else "no_trade"


def combo_prices(qtys, bids, asks) -> dict:
    """Per-share combo prices (debit > 0): mid, natural (buy at ask, sell at bid), leg_cost = natural − mid."""
    q, b, a = (np.asarray(x, dtype=float) for x in (qtys, bids, asks))
    mid = float(np.sum(q * (b + a) / 2))
    natural = float(np.sum(np.where(q > 0, q * a, q * b)))
    return {"mid": mid, "natural": natural, "leg_cost": natural - mid}


def implied_move(straddle_price: float, spot: float) -> float:
    return straddle_price / spot


def straddle_rule_of_thumb(S, sigma, T) -> float:
    """ATM straddle ≈ √(2/π)·S·σ·√T ≈ 0.8·S·σ·√T."""
    return float(np.sqrt(2 / np.pi) * S * sigma * np.sqrt(T))


def earnings_events(n: int = 40, seed: int = 12) -> pd.DataFrame:
    """Past earnings events: the implied move (from the straddle before the event) and the realized absolute move.
    Realized moves are fat-tailed and on average a little SMALLER than implied (the event premium)."""
    rng = np.random.default_rng(seed)
    implied = rng.uniform(0.03, 0.08, n)
    realized = np.abs(rng.standard_t(3, n)) * implied * 0.62
    return pd.DataFrame({"implied": implied.round(4), "realized": realized.round(4)})


def realized_vol(close: pd.Series, window: int = 21) -> pd.Series:
    return np.log(close).diff().rolling(window).std() * np.sqrt(252)


def vrp_research(iv_30d: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """RESEARCH ONLY: IV today minus realized vol over the NEXT window days (future data by design)."""
    return iv_30d - realized_vol(close, window).shift(-window)


def vrp_signal(iv_30d: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """Tradable: IV today minus TRAILING realized vol."""
    return iv_30d - realized_vol(close, window)


def vol_market(n: int = 1500, seed: int = 21):
    """An index with stochastic volatility and an implied vol that is its expected future vol plus a premium."""
    rng = np.random.default_rng(seed)
    v = np.empty(n)
    v[0] = 0.16
    for t in range(1, n):
        v[t] = max(0.07, v[t - 1] + 0.03 * (0.16 - v[t - 1]) + rng.normal(0, 0.012) + (rng.random() < 0.006) * 0.15)
    r = v / np.sqrt(252) * rng.standard_normal(n)
    idx = pd.bdate_range("2015-01-02", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    iv = pd.Series(v * 1.1 + 0.015 + rng.normal(0, 0.005, n), index=idx)
    return close, iv


# --------------------------------------------------------------------------------------
# 08 · hedging
# --------------------------------------------------------------------------------------
def futures_hedge_contracts(portfolio_value, beta, fut_price, multiplier, hedge_ratio=1.0) -> int:
    """Index futures to SELL: round(hedge_ratio × β × value / (futures price × multiplier))."""
    return int(round(hedge_ratio * beta * portfolio_value / (fut_price * multiplier)))


def zero_cost_call_strike(S, T, k_put, iv_fn, r=0.04, q=0.0) -> float:
    """Call strike above S whose premium equals the put's at k_put (brentq on [S, 3S])."""
    put = bsm_price(S, k_put, T, r, q, iv_fn(k_put), -1)
    return float(brentq(lambda k: bsm_price(S, k, T, r, q, iv_fn(k), 1) - put, S, 3 * S, xtol=1e-8))


def crash_path(seed: int = 0, calm: int = 500, crash: int = 15, recovery: int = 250, crash_size: float = -0.30):
    """Calm drift up at 14% vol, a crash of crash_size over `crash` days at 60% vol, then recovery at 30% vol.
    Returns (close, implied vol) with implied a little above realized."""
    rng = np.random.default_rng(seed)
    n = calm + crash + recovery
    vol = np.concatenate([np.full(calm, 0.14), np.full(crash, 0.60), np.full(recovery, 0.30)])
    drift = np.concatenate([np.full(calm, 0.10 / 252), np.full(crash, np.log(1 + crash_size) / max(crash, 1)),
                            np.full(recovery, 0.15 / 252)])
    lr = drift + vol / np.sqrt(252) * rng.standard_normal(n) * np.concatenate([np.ones(calm), np.full(crash, 0.3),
                                                                              np.ones(recovery)])
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(lr)), index=idx), pd.Series(vol * 1.1, index=idx)


def hedge_study(close: pd.Series, iv: pd.Series, method: str = "none", otm: float = 0.10, tenor_days: int = 63,
                roll_days: int = 21, hedge_ratio: float = 0.5, r: float = 0.04) -> pd.Series:
    """Equity of 1,000,000 in the index plus a hedge: 'none', 'puts' (roll 10% OTM 3-month puts monthly), 'collar'
    (the puts plus zero-cost short calls), 'futures' (short hedge_ratio of the index)."""
    S, vol = close.to_numpy(), iv.to_numpy()
    N = 1_000_000 / S[0]
    cash, legs, eq = 0.0, [], np.empty(S.size)

    def leg_value(t, legs):
        v = 0.0
        for cp, K, exp_day, qty in legs:
            tau = (exp_day - t) / 252
            v += qty * (bsm_price(S[t], K, tau, r, 0.0, vol[t], cp) if tau > 1e-9 else max(cp * (S[t] - K), 0.0))
        return v

    for t in range(S.size):
        if method in ("puts", "collar") and t % roll_days == 0:
            cash += leg_value(t, legs)
            kp = (1 - otm) * S[t]
            legs = [(-1, kp, t + tenor_days, N)]
            if method == "collar":
                kc = zero_cost_call_strike(S[t], tenor_days / 252, kp, lambda k, v=vol[t]: v, r)
                legs.append((1, kc, t + tenor_days, -N))
            cash -= leg_value(t, legs)
        if method == "futures" and t > 0:
            cash -= hedge_ratio * N * (S[t] - S[t - 1])
        eq[t] = N * S[t] + cash + leg_value(t, legs)
    return pd.Series(eq, index=close.index, name=method)


def hedge_report(curves: dict) -> pd.DataFrame:
    """total_return, max_drawdown, and cost_vs_none (total return minus the unhedged one)."""
    rows = {m: {"total_return": eq.iloc[-1] / eq.iloc[0] - 1, "max_drawdown": (eq / eq.cummax() - 1).min()}
            for m, eq in curves.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["cost_vs_none"] = df["total_return"] - df.loc["none", "total_return"]
    return df
