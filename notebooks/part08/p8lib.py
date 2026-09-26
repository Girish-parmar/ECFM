"""Helper functions for the Part 8 guided notebooks (MFAAT, Months 7–8: backtesting, validation, risk and portfolios).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

All data is SYNTHETIC: a market with known regimes, strategy return streams with KNOWN Sharpe ratios, pure-noise
strategies, and a stock universe with delistings. So every statistic can be checked against the truth, and every
"discovery" can be checked against luck. The definitions match the graded labs in labs/part08/ (which use duckdb,
scikit-learn and cvxpy; here the same ideas use only numpy, pandas and scipy).
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import os
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from math import comb

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis, norm, skew

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P8_STRICT") == "1"      # tests: a failed check raises instead of continuing
EULER = 0.5772156649015329


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
def sma(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.size >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def regime_market(n_blocks: int = 24, block: int = 125, seed: int = 0) -> pd.DataFrame:
    """Daily bars from blocks with a KNOWN regime (trend 1/0, high_vol 1/0), as in Part 7, plus volume."""
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
    vol = np.round(2e6 * np.exp(rng.normal(0, 0.3, c.size)))
    idx = pd.bdate_range("2012-01-02", periods=c.size)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": vol, "trend": tl, "high_vol": vl},
                        index=idx)


def sma_cross_signal(close, fast: int = 20, slow: int = 100) -> np.ndarray:
    f, s = sma(close, fast), sma(close, slow)
    return np.where(np.isfinite(s) & (f > s), 1.0, 0.0)


def tsmom_signal(close, lookback: int = 120, vol_n: int = 20, target_vol: float = 0.10, max_lev: float = 2.0):
    close = np.asarray(close, dtype=float)
    past = np.full(close.shape, np.nan)
    past[lookback:] = close[lookback:] / close[:-lookback] - 1
    r = np.full(close.shape, np.nan)
    r[1:] = np.diff(np.log(close))
    vol = pd.Series(r).rolling(vol_n).std().to_numpy() * np.sqrt(252)
    return np.nan_to_num(np.clip(np.sign(past) * target_vol / vol, -max_lev, max_lev))


def first_look_pnl(decided, open_, cost_bps: float = 2.0) -> np.ndarray:
    """Part 7's first-look P&L: decided[t] filled at open[t+1], held to the next open, cost on turnover."""
    decided = np.nan_to_num(np.asarray(decided, dtype=float))
    open_ = np.asarray(open_, dtype=float)
    pos = np.zeros_like(decided)
    pos[1:] = decided[:-1]
    ret = np.zeros_like(open_)
    ret[:-1] = open_[1:] / open_[:-1] - 1
    return pos * ret - np.abs(np.diff(pos, prepend=0.0)) * cost_bps / 1e4


def strategy_returns(n_days: int = 2520, sharpes=(0.8, 0.6, 0.5, 0.4, 0.3), vols=(0.10, 0.15, 0.08, 0.20, 0.12),
                     corr: float = 0.2, seed: int = 0, fat_tails: bool = True) -> pd.DataFrame:
    """Daily returns of strategies with KNOWN annual Sharpes and vols and a common pairwise correlation."""
    rng = np.random.default_rng(seed)
    n = len(sharpes)
    C = np.full((n, n), corr) + (1 - corr) * np.eye(n)
    L = np.linalg.cholesky(C)
    z = rng.standard_t(5, size=(n_days, n)) / np.sqrt(5 / 3) if fat_tails else rng.standard_normal((n_days, n))
    z = z @ L.T
    vols, sh = np.asarray(vols), np.asarray(sharpes)
    daily = z * vols / np.sqrt(252) + sh * vols / 252
    return pd.DataFrame(daily, index=pd.bdate_range("2015-01-02", periods=n_days),
                        columns=[f"S{i + 1}" for i in range(n)])


def sharpe(r) -> float:
    """Annualized Sharpe of daily returns (ddof=1); NaN without variance."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------------------
# 01 · research log, events, ledger
# --------------------------------------------------------------------------------------
def config_hash(config: dict) -> str:
    """First 12 hex characters of sha256(json.dumps(config, sort_keys=True))."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


class ResearchLog:
    """Every run is recorded (sqlite here, duckdb in the lab): the Deflated Sharpe Ratio needs ALL the trials."""

    def __init__(self, path: str = ":memory:"):
        self.con = sqlite3.connect(path)
        self.con.execute("CREATE TABLE IF NOT EXISTS runs (run_id INTEGER, strategy TEXT, config_hash TEXT, params TEXT,"
                         " data_version TEXT, sharpe REAL, max_dd REAL, n_obs INTEGER)")

    def record(self, strategy: str, params: dict, data_version: str, sharpe_: float, max_dd: float, n_obs: int) -> int:
        run_id = self.con.execute("SELECT coalesce(max(run_id), 0) + 1 FROM runs").fetchone()[0]
        h = config_hash({"strategy": strategy, "params": params, "data_version": data_version})
        self.con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (run_id, strategy, h, json.dumps(params, sort_keys=True), data_version, sharpe_, max_dd, n_obs))
        return int(run_id)

    def trials(self, strategy: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM runs" + (" WHERE strategy = ?" if strategy else "") + " ORDER BY run_id"
        return pd.read_sql_query(q, self.con, params=(strategy,) if strategy else None)


@dataclass(order=True)
class Event:
    ts: pd.Timestamp
    priority: int                        # same timestamp: 0 fill, 1 bar, 2 timer
    seq: int                             # insertion order breaks the remaining ties
    kind: str = field(compare=False)
    data: object = field(compare=False, default=None)


class EventQueue:
    """Min-heap ordered by (ts, priority, seq): deterministic replay."""

    def __init__(self):
        self._h, self._seq = [], 0

    def push(self, ts, priority: int, kind: str, data=None) -> None:
        heapq.heappush(self._h, Event(ts, priority, self._seq, kind, data))
        self._seq += 1

    def pop(self) -> Event:
        return heapq.heappop(self._h)

    def __bool__(self) -> bool:
        return bool(self._h)


class Portfolio:
    """Cash, positions, last prices and contract multipliers (ES: 50)."""

    def __init__(self, cash: float, multipliers: dict | None = None):
        self.cash, self.pos, self.last = float(cash), {}, {}
        self.mult = dict(multipliers or {})
        self.fees = 0.0

    def on_fill(self, sym: str, qty: float, price: float, fee: float = 0.0) -> None:
        m = self.mult.get(sym, 1.0)
        self.cash -= qty * price * m + fee
        self.fees += fee
        self.pos[sym] = self.pos.get(sym, 0.0) + qty

    def mark(self, sym: str, price: float) -> None:
        self.last[sym] = price

    @property
    def equity(self) -> float:
        return self.cash + sum(q * self.last.get(s, 0.0) * self.mult.get(s, 1.0) for s, q in self.pos.items())


# --------------------------------------------------------------------------------------
# 02 · fills, costs, the engine, parity
# --------------------------------------------------------------------------------------
def market_fill_price(qty: float, open_price: float, slippage_bps: float = 0.0) -> float:
    return open_price * (1 + np.sign(qty) * slippage_bps / 1e4)


def limit_fill_price(qty: float, limit: float, bar: dict) -> float | None:
    """Fill only if price trades THROUGH the limit: buy if low < limit at min(open, limit); sell if high > limit at
    max(open, limit)."""
    if qty > 0:
        return min(bar["open"], limit) if bar["low"] < limit else None
    return max(bar["open"], limit) if bar["high"] > limit else None


def limit_fill_on_touch(qty: float, limit: float, bar: dict) -> float | None:
    """The optimistic bug: fill whenever the price merely touches the limit."""
    if qty > 0:
        return min(bar["open"], limit) if bar["low"] <= limit else None
    return max(bar["open"], limit) if bar["high"] >= limit else None


def stop_fill_price(qty: float, stop: float, bar: dict) -> float | None:
    """Trigger at the stop or the GAP open, whichever is worse: buy stop if high >= stop at max(open, stop); sell stop
    if low <= stop at min(open, stop)."""
    if qty > 0:
        return max(bar["open"], stop) if bar["high"] >= stop else None
    return min(bar["open"], stop) if bar["low"] <= stop else None


def participation_cap(qty: float, bar_volume: float, max_participation: float = 0.10) -> float:
    return float(np.sign(qty) * min(abs(qty), max_participation * bar_volume))


def ib_fixed_commission(shares: float, price: float) -> float:
    """$0.005/share, min $1, max 1% of trade value; 0 for 0 shares."""
    if shares == 0:
        return 0.0
    return float(np.clip(0.005 * abs(shares), 1.0, 0.01 * abs(shares) * price))


def sqrt_impact_bps(order_shares: float, adv_shares: float, daily_vol: float, k: float = 1.0) -> float:
    """1e4 · k · σ_daily · √(|Q| / ADV)."""
    return float(1e4 * k * daily_vol * np.sqrt(abs(order_shares) / adv_shares))


def run_backtest(bars: pd.DataFrame, decided, cash: float = 1_000_000.0, symbol: str = "X", slippage_bps: float = 0.0,
                 commission=None, max_participation: float | None = None) -> pd.DataFrame:
    """Event-driven single-symbol backtest of target weights decided at each close; orders fill at the next open."""
    pf, q = Portfolio(cash), EventQueue()
    for t, ts in enumerate(bars.index):
        q.push(ts, 1, "bar", t)
    pending, rows = 0.0, []
    o, c, v = bars["open"].to_numpy(), bars["close"].to_numpy(), bars["volume"].to_numpy()
    decided = np.nan_to_num(np.asarray(decided, dtype=float))
    while q:
        t = q.pop().data
        if pending:
            qty = participation_cap(pending, v[t], max_participation) if max_participation else pending
            px = market_fill_price(qty, o[t], slippage_bps)
            pf.on_fill(symbol, qty, px, commission(qty, px) if commission else 0.0)
            pending = 0.0
        pf.mark(symbol, o[t])
        eq_open = pf.equity
        pf.mark(symbol, c[t])
        eq = pf.equity
        pending = round(float(decided[t]) * eq / c[t]) - pf.pos.get(symbol, 0.0)
        rows.append((pf.pos.get(symbol, 0.0), eq_open, eq, pf.fees))
    return pd.DataFrame(rows, index=bars.index, columns=["position", "equity_open", "equity", "fees"])


def engine_returns(bars, decided, **kw) -> np.ndarray:
    """Open-to-open returns of the engine's equity, aligned with first_look_pnl (0 on the last bar)."""
    eq = run_backtest(bars, decided, **kw)["equity_open"].to_numpy()
    r = np.zeros(len(eq))
    r[:-1] = eq[1:] / eq[:-1] - 1
    return r


# --------------------------------------------------------------------------------------
# 03 · biases and the tear sheet
# --------------------------------------------------------------------------------------
def stock_universe(n: int = 60, days: int = 2520, seed: int = 3):
    """Daily closes of n stocks over ten years; some go bust and are DELISTED (price path ends). Returns (closes with
    NaN after delisting, membership table symbol/start/end)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-02", periods=days)
    closes, rows = {}, []
    for i in range(n):
        mu, sig = rng.normal(0.02, 0.10) / 252, rng.uniform(0.2, 0.5) / np.sqrt(252)
        r = rng.normal(mu, sig, days)
        px = 50 * np.exp(np.cumsum(r))
        start = idx[0] if rng.random() < 0.8 else idx[int(rng.integers(100, 1500))]
        px[idx < start] = np.nan
        end = pd.NaT
        dead = np.flatnonzero((px < 12) & (idx >= start))
        if dead.size:
            d = int(dead[0])
            px[d:] = np.nan
            end = idx[d]
        closes[f"STK{i:02d}"] = px
        rows.append({"symbol": f"STK{i:02d}", "start": start, "end": end})
    return pd.DataFrame(closes, index=idx), pd.DataFrame(rows)


def members(membership: pd.DataFrame, date) -> list[str]:
    """Point-in-time universe: start <= date and (end is NaT or date < end); sorted."""
    d = pd.Timestamp(date)
    m = membership[(membership["start"] <= d) & (membership["end"].isna() | (d < membership["end"]))]
    return sorted(m["symbol"])


def max_drawdown(returns) -> tuple[float, int]:
    """(max drawdown <= 0 of the compounded equity, longest time under water in bars)."""
    eq = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1
    longest = run = 0
    for x in dd:
        run = run + 1 if x < 0 else 0
        longest = max(longest, run)
    return float(dd.min()), int(longest)


def tear_sheet(returns, periods: int = 252) -> dict:
    r = np.asarray(returns, dtype=float)
    cagr = np.prod(1 + r) ** (periods / r.size) - 1
    sd = r.std(ddof=1)
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    mdd, dur = max_drawdown(r)
    return {"cagr": cagr, "vol": sd * np.sqrt(periods), "sharpe": r.mean() / sd * np.sqrt(periods),
            "sortino": r.mean() / down * np.sqrt(periods), "max_dd": mdd, "dd_duration": dur,
            "calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "skew": float(skew(r)), "kurtosis": float(kurtosis(r)),
            "tail_ratio": float(np.percentile(r, 95) / abs(np.percentile(r, 5)))}


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    m = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    return m.unstack().reindex(columns=range(1, 13))


# --------------------------------------------------------------------------------------
# 04 · significance and capacity
# --------------------------------------------------------------------------------------
def sharpe_se(sr: float, T: int) -> float:
    """Lo (2002), per-period units: √((1 + SR²/2) / T)."""
    return float(np.sqrt((1 + 0.5 * sr ** 2) / T))


def psr(returns, sr_benchmark: float = 0.0) -> float:
    """Φ((SR − SR*)·√(T−1) / √(1 − γ3·SR + (γ4 − 1)/4·SR²)), per-period SR, γ4 non-excess kurtosis."""
    r = np.asarray(returns, dtype=float)
    T = r.size
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def min_track_record(returns, sr_benchmark: float = 0.0, alpha: float = 0.05) -> float:
    r = np.asarray(returns, dtype=float)
    sr = r.mean() / r.std(ddof=1)
    if sr <= sr_benchmark:
        return float("inf")
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (norm.ppf(1 - alpha) / (sr - sr_benchmark)) ** 2)


def stationary_bootstrap_sharpes(returns, n_boot: int = 500, mean_block: int = 20, seed: int = 0) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    T = r.size
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        jump = rng.random(T) < 1 / mean_block
        new = rng.integers(T, size=T)
        idx = np.empty(T, dtype=int)
        idx[0] = new[0]
        for t in range(1, T):
            idx[t] = new[t] if jump[t] else (idx[t - 1] + 1) % T
        s = r[idx]
        out[b] = s.mean() / s.std(ddof=1)
    return out


def drawdown_distribution(trade_returns, n: int = 2000, seed: int = 0) -> np.ndarray:
    r = np.asarray(trade_returns, dtype=float)
    rng = np.random.default_rng(seed)
    return np.array([max_drawdown(rng.permutation(r))[0] for _ in range(n)])


def net_sharpe_vs_capital(gross_returns, daily_turnover: float, adv_dollars: float, daily_vol: float, capitals,
                          k: float = 1.0) -> pd.Series:
    """Square-root impact cost per day = turnover × k·σ_daily·√(turnover·C / ADV); annualized net Sharpe per capital C."""
    r = np.asarray(gross_returns, dtype=float)
    out = {}
    for C in capitals:
        cost = daily_turnover * k * daily_vol * np.sqrt(daily_turnover * C / adv_dollars)
        out[C] = sharpe(r - cost)
    return pd.Series(out)


def capacity(sharpe_by_capital: pd.Series, min_sharpe: float = 0.5) -> float:
    ok = sharpe_by_capital[sharpe_by_capital >= min_sharpe]
    return float(ok.index.max()) if len(ok) else 0.0


# --------------------------------------------------------------------------------------
# 05 · search and stability
# --------------------------------------------------------------------------------------
def grid_search(objective, grid: dict) -> pd.DataFrame:
    keys = list(grid)
    rows = [dict(zip(keys, combo)) | {"score": objective(**dict(zip(keys, combo)))}
            for combo in itertools.product(*grid.values())]
    return pd.DataFrame(rows)


def random_search(objective, space: dict, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        prm = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        rows.append(prm | {"score": objective(**prm)})
    return pd.DataFrame(rows)


def plateau_scores(results: pd.DataFrame, x: str, y: str, radius: int = 1) -> pd.DataFrame:
    """Each cell of the (y × x) score table replaced by the mean of its (2r+1)² neighbourhood (clipped at the edges)."""
    tab = results.pivot(index=y, columns=x, values="score").sort_index().sort_index(axis=1)
    a = tab.to_numpy()
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = np.nanmean(a[max(0, i - radius):i + radius + 1, max(0, j - radius):j + radius + 1])
    return pd.DataFrame(out, index=tab.index, columns=tab.columns)


def pareto_front(df: pd.DataFrame, maximize: str, minimize_: str) -> pd.DataFrame:
    """Rows not dominated by any other row (another row is at least as good on both and strictly better on one),
    sorted by `minimize_`."""
    a, b = df[maximize].to_numpy(), df[minimize_].to_numpy()
    keep = [i for i in range(len(df))
            if not np.any((a >= a[i]) & (b <= b[i]) & ((a > a[i]) | (b < b[i])))]
    return df.iloc[keep].sort_values(minimize_)


# --------------------------------------------------------------------------------------
# 06 · walk-forward and cross-validation
# --------------------------------------------------------------------------------------
def walk_forward(n: int, train: int, test: int, anchored: bool = False):
    start = 0
    while start + train + test <= n:
        yield np.arange(0 if anchored else start, start + train), np.arange(start + train, start + train + test)
        start += test


def walk_forward_optimize(pnl_fn, grid: dict, n: int, train: int, test: int, anchored: bool = False) -> dict:
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
    cache = {tuple(prm.values()): pnl_fn(**prm) for prm in combos}
    oos, chosen, is_s, oos_s = [], [], [], []
    for tr, te in walk_forward(n, train, test, anchored):
        best = max(combos, key=lambda prm: np.nan_to_num(sharpe(cache[tuple(prm.values())][tr]), nan=-np.inf))
        pnl = cache[tuple(best.values())]
        chosen.append(best)
        is_s.append(sharpe(pnl[tr]))
        oos_s.append(sharpe(pnl[te]))
        oos.append(pnl[te])
    return {"oos": np.concatenate(oos), "params": chosen, "is_sharpe": is_s, "oos_sharpe": oos_s,
            "wfe": float(np.mean(oos_s) / np.mean(is_s))}


def purged_kfold(n: int, k: int = 5, label_horizon: int = 5, embargo: float = 0.01):
    """Test folds: k contiguous blocks. Train keeps i only if i + label_horizon < fold start, or i > fold end + embargo
    (ceil(embargo·n) bars)."""
    emb = int(np.ceil(embargo * n))
    idx = np.arange(n)
    for test in np.array_split(idx, k):
        lo, hi = test[0], test[-1]
        yield idx[(idx + label_horizon < lo) | (idx > hi + emb)], test


def shuffled_kfold(n: int, k: int = 5, seed: int = 0):
    """Ordinary k-fold with shuffling (sklearn KFold(shuffle=True)): the classic mistake with time series."""
    idx = np.random.default_rng(seed).permutation(n)
    for test in np.array_split(idx, k):
        test = np.sort(test)
        yield np.setdiff1d(np.arange(n), test), test


def contiguous_kfold(n: int, k: int = 5):
    """Ordinary k-fold on contiguous blocks, no purging: leaks only around the fold edges."""
    idx = np.arange(n)
    for test in np.array_split(idx, k):
        yield np.setdiff1d(idx, test), test


def cpcv_splits(n: int, n_groups: int = 6, k_test: int = 2, label_horizon: int = 5, embargo: float = 0.01) -> list:
    emb = int(np.ceil(embargo * n))
    idx = np.arange(n)
    groups = np.array_split(idx, n_groups)
    out = []
    for combo in itertools.combinations(range(n_groups), k_test):
        keep = np.ones(n, dtype=bool)
        for g in combo:
            keep &= (idx + label_horizon < groups[g][0]) | (idx > groups[g][-1] + emb)
        out.append((idx[keep], combo, np.concatenate([groups[g] for g in combo])))
    return out


def cpcv_n_paths(n_groups: int, k_test: int) -> int:
    """C(N, k)·k / N complete out-of-sample paths."""
    return comb(n_groups, k_test) * k_test // n_groups


def overlapping_labels(n: int = 1500, horizon: int = 20, seed: int = 1):
    """Pure-noise daily returns and their `horizon`-day FORWARD sums as labels: consecutive labels share most of their
    days. Nothing about the future is predictable."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n + horizon)
    y = np.array([r[t + 1:t + 1 + horizon].sum() for t in range(n)])
    return r[:n], y


def nearest_neighbour_r2(y, splits) -> float:
    """Out-of-sample R² of a 1-nearest-neighbour model whose only feature is the DATE: predict each test label with the
    label of the closest training day in time. It can only 'work' through leakage."""
    preds, trues = [], []
    for train, test in splits:
        j = np.searchsorted(train, test)
        left, right = train[np.clip(j - 1, 0, len(train) - 1)], train[np.clip(j, 0, len(train) - 1)]
        nearest = np.where(np.abs(test - left) <= np.abs(right - test), left, right)
        preds.append(y[nearest])
        trues.append(y[test])
    p_, t_ = np.concatenate(preds), np.concatenate(trues)
    return float(1 - np.sum((t_ - p_) ** 2) / np.sum((t_ - t_.mean()) ** 2))


# --------------------------------------------------------------------------------------
# 07 · deflated Sharpe and PBO
# --------------------------------------------------------------------------------------
def expected_max_sharpe(trial_srs) -> float:
    """√V · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))), V the variance of the trial Sharpes (ddof=1)."""
    s = np.asarray(trial_srs, dtype=float)
    n_, v = s.size, s.var(ddof=1)
    return float(np.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / n_) + EULER * norm.ppf(1 - 1 / (n_ * np.e))))


def deflated_sharpe(returns, trial_srs) -> tuple[float, float]:
    """(DSR, SR0): the PSR of `returns` against SR0 = expected_max_sharpe(trial_srs)."""
    sr0 = expected_max_sharpe(trial_srs)
    return psr(returns, sr0), sr0


def pbo_cscv(perf: np.ndarray, S: int = 16) -> tuple[float, np.ndarray]:
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation (T × N returns)."""
    T, N = perf.shape
    blocks = np.array_split(np.arange(T), S)

    def sr(x):
        return x.mean(0) / x.std(0, ddof=1)
    logits = []
    for is_blocks in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in is_blocks])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in is_blocks])
        best = int(np.argmax(sr(perf[is_idx])))
        w = (sr(perf[oos_idx]).argsort().argsort()[best] + 1) / (N + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits


def noise_strategies(n_days: int = 1500, n: int = 200, seed: int = 7, skilled: int = 0, skill_sharpe: float = 1.0):
    """n strategies of pure noise (Sharpe 0), the first `skilled` of them with a true annual Sharpe of skill_sharpe."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, (n_days, n))
    r[:, :skilled] += skill_sharpe * 0.01 / np.sqrt(252)
    return r


# --------------------------------------------------------------------------------------
# 08 · robustness and the validation gate
# --------------------------------------------------------------------------------------
def robustness_scorecard(run, params: dict, perturb: float = 0.2) -> pd.DataFrame:
    """run(params, delay, cost_mult) → daily P&L. Tests: each int parameter ±perturb (Sharpe >= ½ base), delay 1 bar
    (>= ½ base), costs ×2 (> 0), without the best 5 days (> 0), first and second half (> 0)."""
    base = run(params, 0, 1.0)
    sb = sharpe(base)
    rows = []
    for k, v in params.items():
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            for sgn, lab in ((-1, "-"), (1, "+")):
                s = sharpe(run(dict(params, **{k: int(round(v * (1 + sgn * perturb)))}), 0, 1.0))
                rows.append((f"{k} {lab}{int(perturb * 100)}%", s, s >= 0.5 * sb))
    s = sharpe(run(params, 1, 1.0))
    rows.append(("delay 1 bar", s, s >= 0.5 * sb))
    s = sharpe(run(params, 0, 2.0))
    rows.append(("costs x2", s, s > 0))
    s = sharpe(np.delete(base, np.argsort(base)[-5:]))
    rows.append(("without best 5 days", s, s > 0))
    h = base.size // 2
    rows += [("first half", sharpe(base[:h]), sharpe(base[:h]) > 0),
             ("second half", sharpe(base[h:]), sharpe(base[h:]) > 0)]
    return pd.DataFrame(rows, columns=["test", "value", "passed"])


GATE = {"min_oos_sharpe": 0.5, "min_dsr": 0.95, "max_pbo": 0.2, "min_robustness": 0.8, "max_drawdown": -0.25}


def validation_gate(oos_pnl, trial_srs, pbo: float, scorecard: pd.DataFrame, criteria: dict | None = None) -> dict:
    c = {**GATE, **(criteria or {})}
    r = np.asarray(oos_pnl, dtype=float)
    oos_sr = sharpe(r)
    dsr, _ = deflated_sharpe(r, trial_srs)
    rob = float(scorecard["passed"].mean())
    mdd = max_drawdown(r)[0]
    checks = {"oos_sharpe": (oos_sr, oos_sr >= c["min_oos_sharpe"]), "dsr": (dsr, dsr >= c["min_dsr"]),
              "pbo": (pbo, pbo <= c["max_pbo"]), "robustness": (rob, rob >= c["min_robustness"]),
              "max_dd": (mdd, mdd >= c["max_drawdown"])}
    return {"checks": checks, "passed": all(ok for _, ok in checks.values())}


# --------------------------------------------------------------------------------------
# 09 · the risk engine and market risk
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str = ""


class RiskRule:
    def check(self, order: dict, ctx: dict) -> RiskDecision:
        raise NotImplementedError


class MaxNotional(RiskRule):
    def __init__(self, limit: float):
        self.limit = limit

    def check(self, order, ctx):
        n_ = abs(order["qty"] * order["price"])
        return RiskDecision(n_ <= self.limit, f"notional {n_:,.0f} > {self.limit:,.0f}")


class MaxGrossExposure(RiskRule):
    def __init__(self, max_leverage: float):
        self.max_leverage = max_leverage

    def check(self, order, ctx):
        gross = ctx["gross"] + abs(order["qty"] * order["price"])
        return RiskDecision(gross <= self.max_leverage * ctx["equity"],
                            f"gross {gross / ctx['equity']:.2f}x > {self.max_leverage}x")


class MaxPositionWeight(RiskRule):
    """|current $ position + signed order notional| / equity <= max_weight."""

    def __init__(self, max_weight: float):
        self.max_weight = max_weight

    def check(self, order, ctx):
        after = ctx["positions"].get(order["symbol"], 0.0) + order["qty"] * order["price"]
        w = abs(after) / ctx["equity"]
        return RiskDecision(w <= self.max_weight, f"{order['symbol']} weight {w:.1%} > {self.max_weight:.1%}")


class DailyLossLimit(RiskRule):
    def __init__(self, max_loss_frac: float):
        self.max_loss_frac = max_loss_frac

    def check(self, order, ctx):
        dd = ctx["day_pnl"] / ctx["start_equity"]
        return RiskDecision(dd > -self.max_loss_frac, f"daily loss {dd:.2%} breaches {-self.max_loss_frac:.2%}")


class RiskEngine:
    """Chain of Responsibility: the first rejecting rule stops the chain; reason prefixed with the rule's class name."""

    def __init__(self, rules):
        self.rules, self.log = rules, []

    def check(self, order, ctx) -> RiskDecision:
        for rule in self.rules:
            d = rule.check(order, ctx)
            if not d.approved:
                reason = f"{type(rule).__name__}: {d.reason}"
                self.log.append((order["symbol"], reason))
                return RiskDecision(False, reason)
        return RiskDecision(True)


def hist_var_cvar(returns, alpha: float = 0.99) -> tuple[float, float]:
    """Historical VaR (alpha-quantile of losses) and CVaR (mean of losses >= VaR), positive numbers."""
    losses = -np.asarray(returns, dtype=float)
    var = np.quantile(losses, alpha)
    return float(var), float(losses[losses >= var].mean())


def parametric_var(weights, cov, alpha: float = 0.99, mu=None) -> float:
    w = np.asarray(weights, dtype=float)
    sd = float(np.sqrt(w @ cov @ w))
    return norm.ppf(alpha) * sd - (0.0 if mu is None else float(w @ np.asarray(mu)))


def component_var(weights, cov, alpha: float = 0.99) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return w * (cov @ w) / np.sqrt(w @ cov @ w) * norm.ppf(alpha)


def asset_returns(n_days: int = 2520, seed: int = 2) -> pd.DataFrame:
    """Daily returns of 6 assets (2 equity-like, 2 bond-like, gold, commodities) with a crisis regime where equity
    correlations jump and bonds rally."""
    rng = np.random.default_rng(seed)
    names = ["EQ_US", "EQ_INTL", "BOND_10Y", "BOND_2Y", "GOLD", "CMDTY"]
    vol = np.array([0.16, 0.18, 0.07, 0.02, 0.15, 0.20]) / np.sqrt(252)
    mu = np.array([0.07, 0.06, 0.025, 0.015, 0.03, 0.02]) / 252
    calm = np.array([[1, .7, -.2, -.1, .05, .3], [.7, 1, -.15, -.1, .1, .35], [-.2, -.15, 1, .6, .2, -.1],
                     [-.1, -.1, .6, 1, .1, -.05], [.05, .1, .2, .1, 1, .3], [.3, .35, -.1, -.05, .3, 1]])
    crisis = calm.copy()
    crisis[0, 1] = crisis[1, 0] = 0.95
    crisis[0, 5] = crisis[5, 0] = crisis[1, 5] = crisis[5, 1] = 0.7
    crisis[0, 2] = crisis[2, 0] = crisis[1, 2] = crisis[2, 1] = -0.5
    is_crisis = np.zeros(n_days, dtype=bool)
    for s in (600, 1700, 2300):
        is_crisis[s:s + 60] = True
    out = np.empty((n_days, 6))
    for regime, C in ((False, calm), (True, crisis)):
        L = np.linalg.cholesky(C)
        m = is_crisis == regime
        z = rng.standard_t(5, (m.sum(), 6)) / np.sqrt(5 / 3) @ L.T
        scale = np.where(regime, 2.2, 1.0) * np.array([1, 1, 1, 1, 1.2, 1.3]) ** regime
        drift = mu if not regime else mu * np.array([-6, -6, 3, 2, 2, -5])
        out[m] = drift + z * vol * scale
    return pd.DataFrame(out, index=pd.bdate_range("2015-01-02", periods=n_days), columns=names)


# --------------------------------------------------------------------------------------
# 10 · position sizing and Kelly
# --------------------------------------------------------------------------------------
def risk_per_trade_size(equity: float, risk_frac: float, entry: float, stop: float, multiplier: float = 1.0) -> int:
    """Units (floored) such that hitting the stop loses risk_frac of equity."""
    return int(equity * risk_frac / (abs(entry - stop) * multiplier))


def vol_target_weight(returns, target_vol: float = 0.10, lookback: int = 60, cap: float = 2.0) -> float:
    vol = np.std(np.asarray(returns, dtype=float)[-lookback:], ddof=1) * np.sqrt(252)
    return float(min(target_vol / vol, cap))


def risk_of_ruin(r_multiples, risk_frac: float, ruin: float = 0.5, n_trades: int = 500, n_sims: int = 2000,
                 seed: int = 0) -> float:
    """Share of simulated paths (R-multiples drawn with replacement, equity ×= 1 + risk_frac·R) that ever reach
    `ruin` × starting equity."""
    rng = np.random.default_rng(seed)
    R = rng.choice(np.asarray(r_multiples, dtype=float), size=(n_sims, n_trades))
    eq = np.cumprod(1 + risk_frac * R, axis=1)
    return float(np.mean(eq.min(axis=1) <= ruin))


def kelly_discrete(p_win: float, win_loss_ratio: float) -> float:
    return p_win - (1 - p_win) / win_loss_ratio


def kelly_continuous(mu: float, sigma: float, r: float = 0.0) -> float:
    return (mu - r) / sigma ** 2


def kelly_growth_simulation(mu: float = 0.08, sigma: float = 0.16, fractions=(0.25, 0.5, 1.0, 2.0), est_years: int = 5,
                            n_years: int = 20, n_paths: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Leverage = fraction × μ̂/σ² with μ̂ ESTIMATED from est_years of data; then n_years of the true process."""
    rng = np.random.default_rng(seed)
    d = 252
    res = {f: {"w": [], "dd": []} for f in fractions}
    for _ in range(n_paths):
        mu_hat = rng.normal(mu / d, sigma / np.sqrt(d), est_years * d).mean() * d
        fut = rng.normal(mu / d, sigma / np.sqrt(d), n_years * d)
        for f in fractions:
            wealth = np.cumprod(np.maximum(1 + f * mu_hat / sigma ** 2 * fut, 0.0))
            peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
            res[f]["w"].append(wealth[-1])
            res[f]["dd"].append((wealth / peak - 1).min())
    return pd.DataFrame({f: {"median_wealth": float(np.median(v["w"])), "p_loss": float(np.mean(np.array(v["w"]) < 1)),
                             "median_max_dd": float(np.median(v["dd"]))} for f, v in res.items()}).T


# --------------------------------------------------------------------------------------
# 11–12 · portfolio construction
# --------------------------------------------------------------------------------------
def inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    iv = 1 / returns.std(ddof=1)
    return iv / iv.sum()


def diversification_ratio(w, cov) -> float:
    w, cov = np.asarray(w, dtype=float), np.asarray(cov, dtype=float)
    return float(w @ np.sqrt(np.diag(cov)) / np.sqrt(w @ cov @ w))


def min_variance(cov) -> np.ndarray:
    """Σ⁻¹1 / (1'Σ⁻¹1), shorts allowed."""
    x = np.linalg.solve(np.asarray(cov, dtype=float), np.ones(len(cov)))
    return x / x.sum()


def min_variance_long_only(cov, max_weight: float = 1.0) -> np.ndarray:
    n_ = len(cov)
    cov = np.asarray(cov, dtype=float)
    res = minimize(lambda w: w @ cov @ w, np.full(n_, 1 / n_), jac=lambda w: 2 * cov @ w, method="SLSQP",
                   bounds=[(0, max_weight)] * n_, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-12, "maxiter": 500})
    return res.x


def max_sharpe(mu, cov) -> np.ndarray:
    x = np.linalg.solve(np.asarray(cov, dtype=float), np.asarray(mu, dtype=float))
    return x / x.sum()


def shrinkage_intensity(returns) -> float:
    """Ledoit–Wolf (2004) optimal intensity for shrinking the sample covariance toward μ·I (μ = average variance)."""
    X = np.asarray(returns, dtype=float)
    X = X - X.mean(0)
    T, n_ = X.shape
    S = X.T @ X / T
    mu = np.trace(S) / n_
    delta = np.sum((S - mu * np.eye(n_)) ** 2)
    beta = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / T ** 2
    return float(min(beta, delta) / delta)


def ledoit_wolf(returns) -> np.ndarray:
    """δ·μI + (1 − δ)·S with S the (1/T) sample covariance and δ = shrinkage_intensity."""
    X = np.asarray(returns, dtype=float)
    X = X - X.mean(0)
    S = X.T @ X / X.shape[0]
    d = shrinkage_intensity(returns)
    return d * np.trace(S) / S.shape[0] * np.eye(S.shape[0]) + (1 - d) * S


def risk_contributions(w, cov) -> np.ndarray:
    """w_i(Σw)_i / (w'Σw), summing to 1."""
    w, cov = np.asarray(w, dtype=float), np.asarray(cov, dtype=float)
    return w * (cov @ w) / (w @ cov @ w)


def risk_parity(cov) -> np.ndarray:
    """Equal risk contribution via Spinu's convex form: min ½y'Cy − (1/n)Σ ln y, then normalize."""
    n_ = len(cov)
    c = np.asarray(cov, dtype=float) / np.mean(np.diag(cov))
    res = minimize(lambda y: 0.5 * y @ c @ y - np.log(y).sum() / n_, np.full(n_, 1.0),
                   jac=lambda y: c @ y - 1 / (n_ * y), bounds=[(1e-9, None)] * n_, method="L-BFGS-B")
    return res.x / res.x.sum()


def hrp_order(returns: pd.DataFrame) -> list:
    """Leaf order of single-linkage clustering on the distance √(½(1 − ρ))."""
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr)).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    return list(corr.index[leaves_list(linkage(squareform(dist, checks=False), "single"))])


def cluster_variance(cov: pd.DataFrame, items) -> float:
    """Variance of the inverse-variance portfolio of `items`."""
    c = cov.loc[items, items].to_numpy()
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def hrp(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity: quasi-diagonal order, then recursive bisection with α = 1 − v_a/(v_a + v_b)."""
    cov = returns.cov()
    order = hrp_order(returns)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for a, b in zip(clusters[::2], clusters[1::2]):
            va, vb = cluster_variance(cov, a), cluster_variance(cov, b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
    return w.reindex(returns.columns)


def black_litterman(cov, w_mkt, delta: float = 2.5, tau: float = 0.05, P=None, Q=None, omega=None) -> np.ndarray:
    """π = δΣw_mkt; with views μ = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹π + P'Ω⁻¹Q], Ω default diag(PτΣP')."""
    S = np.asarray(cov, dtype=float)
    pi = delta * S @ np.asarray(w_mkt, dtype=float)
    if P is None:
        return pi
    P, Q = np.atleast_2d(np.asarray(P, dtype=float)), np.asarray(Q, dtype=float)
    tS = tau * S
    om = np.diag(np.diag(P @ tS @ P.T)) if omega is None else np.asarray(omega, dtype=float)
    A = np.linalg.inv(tS) + P.T @ np.linalg.inv(om) @ P
    b = np.linalg.solve(tS, pi) + P.T @ np.linalg.solve(om, Q)
    return np.linalg.solve(A, b)


def min_cvar(returns, alpha: float = 0.95, max_weight: float = 1.0) -> np.ndarray:
    """Rockafellar–Uryasev LP (scipy linprog): min ζ + Σu/((1−α)T) s.t. u >= −Rw − ζ, u >= 0, Σw = 1, 0 <= w <= max."""
    R = np.asarray(returns, dtype=float)
    T, n_ = R.shape
    c = np.concatenate([np.zeros(n_), [1.0], np.full(T, 1 / ((1 - alpha) * T))])
    A_ub = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
    A_eq = np.concatenate([np.ones(n_), [0.0], np.zeros(T)])[None, :]
    bounds = [(0, max_weight)] * n_ + [(None, None)] + [(0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(T), A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    return res.x[:n_]


def walk_forward_allocation(returns: pd.DataFrame, allocator, train: int = 252, rebalance: int = 21) -> dict:
    """Refit weights every `rebalance` days on the previous `train` days; hold them; OOS returns and turnover."""
    X = returns.to_numpy()
    out = np.full(len(returns), np.nan)
    prev, turns = np.zeros(X.shape[1]), []
    for start in range(train, len(returns), rebalance):
        w = np.asarray(allocator(returns.iloc[start - train:start]), dtype=float)
        turns.append(np.abs(w - prev).sum())
        prev = w
        end = min(start + rebalance, len(returns))
        out[start:end] = X[start:end] @ w
    return {"returns": pd.Series(out, index=returns.index).iloc[train:], "turnover": float(np.mean(turns))}
