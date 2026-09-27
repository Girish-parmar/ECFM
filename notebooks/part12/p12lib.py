"""Helper functions for the Part 12 guided notebooks (MFAAT, Months 11–12: the complete trading platform, capstone and
production).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

Everything runs OFFLINE, on synthetic data with known answers: a toy `quantforge` package with planted layer
violations, bars and point-in-time universe snapshots, a quote path with informed order flow, an intraday session, a
trade journal with a hidden edge, latency samples, a trading day with injected faults, and a 4-week track record in
which one strategy stops working. No broker, Docker or network is needed. The definitions match the graded labs in
labs/part12/ (the monitoring-drill helpers come from its clinic W3).
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import time
import zlib
from collections.abc import Callable
from contextlib import contextmanager
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P12_STRICT") == "1"      # tests: a failed check raises instead of continuing


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


def sharpe(r) -> float:
    """Annualized Sharpe ratio of daily returns (√252); NaN when the returns never vary (no trades)."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")




# ======================================================================================
# Synthetic data (the same as labs/part12/common.py)
# ======================================================================================
LAYERS = ["apps", "monitoring", "research", "strategy", "execution", "lib", "data", "domain", "core"]


# ------------------------------------------------------------------------------ S1 a toy platform
TOY_MODULES = {
    "core/__init__.py": "", "core/events.py": "import dataclasses\n",
    "domain/__init__.py": "", "domain/orders.py": "from quantforge.core import events\n",
    "data/__init__.py": "", "data/bars.py": "import pandas as pd\nfrom quantforge.domain.orders import Order\n",
    "lib/__init__.py": "", "lib/indicators.py": "import numpy as np\nfrom quantforge.data import bars\n",
    "execution/__init__.py": "", "execution/oms.py": "from quantforge.lib import indicators\nfrom quantforge.domain import orders\n",
    "strategy/__init__.py": "", "strategy/momentum.py": "from quantforge.lib.indicators import ema\n",
    "research/__init__.py": "", "research/backtest.py": "from quantforge.strategy import momentum\n",
    "monitoring/__init__.py": "", "monitoring/audit.py": "import hashlib\nfrom quantforge.core.events import Event\n",
    "apps/__init__.py": "", "apps/api.py": "from quantforge.monitoring import audit\nfrom quantforge.execution import oms\n",
    "brokers/__init__.py": "", "brokers/ib.py": "from quantforge.domain.orders import Order\n",
}
VIOLATIONS = {
    "strategy/shortcut.py": "from quantforge.brokers.ib import submit\n",         # strategy talks to a broker
    "lib/report.py": "import quantforge.apps.api\n",                                # lower layer imports a higher one
    "core/util.py": "def f():\n    from quantforge.execution import oms\n",          # hidden inside a function
}


def make_toy_package(root: Path, violations: bool = True) -> Path:
    """Write a small `quantforge` package under root (optionally with three planted violations); return its path."""
    pkg = Path(root) / "quantforge"
    files = {**TOY_MODULES, **(VIOLATIONS if violations else {})}
    (pkg).mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for rel, src in files.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return pkg


# --------------------------------------------------------------------------- S3–S4 bars & universe
def ohlcv(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Daily bars: open, high, low, close, volume (a trending random walk)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, n) + 0.002 * np.sin(np.arange(n) / 60)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.lognormal(np.log(1e6), 0.3, n).round()},
                        index=pd.bdate_range("2017-01-02", periods=n))


SECTORS = ["tech", "health", "energy", "financials", "industrials", "etf"]


def universe_snapshot(n: int = 300, date="2025-06-02", seed: int = 0) -> pd.DataFrame:
    """One day's point-in-time universe: symbol, date, price, adv_usd (average daily dollar volume), spread_bps,
    market_cap_bn, sector, shortable, option_oi (open interest), iv_rank (0–100)."""
    rng = np.random.default_rng(seed)
    sector = rng.choice(SECTORS, n, p=[0.25, 0.15, 0.1, 0.15, 0.15, 0.2])
    adv = np.exp(rng.normal(np.log(3e7), 1.6, n))
    return pd.DataFrame({
        "symbol": [f"X{i:03d}" for i in range(n)], "date": pd.Timestamp(date),
        "price": np.round(np.exp(rng.normal(np.log(60), 0.9, n)), 2), "adv_usd": adv,
        "spread_bps": np.round(np.clip(40 / np.sqrt(adv / 1e6) + rng.exponential(1, n), 0.5, None), 2),
        "market_cap_bn": np.round(adv / 1e7 * rng.lognormal(0, 0.5, n), 2), "sector": sector,
        "shortable": rng.random(n) > 0.1, "option_oi": np.round(adv / 2e3 * rng.lognormal(0, 1, n)).astype(int),
        "iv_rank": np.round(rng.uniform(0, 100, n), 1)})


def next_snapshot(snap: pd.DataFrame, seed: int = 1, noise: float = 0.15) -> pd.DataFrame:
    """The next day's universe: the same names with ADV, spread and IV rank moved by random noise (given)."""
    rng = np.random.default_rng(seed)
    out = snap.copy()
    out["date"] = snap["date"] + pd.tseries.offsets.BDay(1)
    out["adv_usd"] = snap["adv_usd"] * np.exp(rng.normal(0, noise, len(snap)))
    out["spread_bps"] = np.round(snap["spread_bps"] * np.exp(rng.normal(0, noise, len(snap))), 2)
    out["iv_rank"] = np.clip(snap["iv_rank"] + rng.normal(0, 10, len(snap)), 0, 100).round(1)
    return out


def latency_samples(n: int = 300, spike_at: int | None = 200, seed: int = 0) -> np.ndarray:
    """Order-ack latencies in seconds (lognormal around 40 ms) with one 400 ms spike (lesson plan S11)."""
    rng = np.random.default_rng(seed)
    x = rng.lognormal(np.log(0.04), 0.25, n)
    if spike_at is not None:
        x[spike_at] = 0.4
    return x


# ------------------------------------------------------------------------------ S5–S6 execution
def quote_path(n: int = 3600, tick: float = 0.01, seed: int = 0, p_same: float = 0.8, p_with: float = 0.3,
               p_against: float = 0.05) -> pd.DataFrame:
    """One second per row: bid, ask (spread 2–5 ticks) and the last trade (at the bid or the ask). ORDER FLOW IS
    PERSISTENT AND INFORMED: the next trade is on the same side with probability p_same, and the mid ticks WITH the
    flow with probability p_with (against it p_against). So a passive buy tends to fill when sellers are active and
    the price is falling, and goes unfilled when the price runs away. Index = seconds."""
    rng = np.random.default_rng(seed)
    at_ask = np.empty(n, dtype=bool)
    at_ask[0] = True
    for t in range(1, n):
        at_ask[t] = at_ask[t - 1] if rng.random() < p_same else not at_ask[t - 1]
    u = rng.random(n)
    step = np.where(u < p_with, 1, np.where(u < p_with + p_against, -1, 0))
    step = np.where(at_ask, step, -step)
    mid = 10_000 + np.r_[0, np.cumsum(step[:-1])]
    half = rng.integers(1, 3, n)
    bid, ask = (mid - half) * tick, (mid + half + rng.integers(0, 2, n)) * tick
    return pd.DataFrame({"bid": np.round(bid, 2), "ask": np.round(ask, 2),
                         "trade": np.round(np.where(at_ask, ask, bid), 2)})


def intraday_session(seed: int = 0) -> pd.DataFrame:
    """A 390-minute session: price (random walk from 100) and volume with the usual U shape (heavy at the open and
    the close). Index = minute timestamps."""
    rng = np.random.default_rng(seed)
    m = np.arange(390)
    shape = 1 + 2.5 * np.exp(-m / 30) + 2.0 * np.exp(-(389 - m) / 25)
    volume = np.round(shape * 20_000 * rng.lognormal(0, 0.2, 390))
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 390)))
    return pd.DataFrame({"price": price, "volume": volume},
                        index=pd.date_range("2025-06-02 09:30", periods=390, freq="min"))


# ------------------------------------------------------------------------------ S8 the journal
def trade_journal(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A trade journal with a PLANTED edge: "breakout" trades earn +0.6R on average in the trend regime and −0.3R in
    chop; "pullback" trades earn +0.1R everywhere. Columns: trade_id, entry_time, setup_tag, regime, side, qty,
    planned_qty, entry, stop, exit, exit_reason. About 3% of trades have no stop, some are oversized or outside
    the allowed hours (rule violations)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tag = str(rng.choice(["breakout", "pullback"]))
        regime = str(rng.choice(["trend", "chop"]))
        mu = {("breakout", "trend"): 0.6, ("breakout", "chop"): -0.3}.get((tag, regime), 0.1)
        r = float(np.clip(rng.normal(mu, 1.0), -1.2, 5))
        side = int(rng.choice([1, -1]))
        entry = round(float(rng.uniform(20, 200)), 2)
        risk = round(entry * float(rng.uniform(0.01, 0.03)), 2)
        stop = round(entry - side * risk, 2)
        exit_ = round(entry + side * r * risk, 2)
        minute = int(rng.integers(0, 390)) if rng.random() > 0.04 else int(rng.integers(390, 420))
        planned = int(rng.integers(1, 5)) * 100
        qty = planned * (2 if rng.random() < 0.03 else 1)
        rows.append({"trade_id": i, "entry_time": pd.Timestamp("2025-06-02 09:30") + pd.Timedelta(days=i // 8,
                     minutes=minute), "setup_tag": tag, "regime": regime, "side": side, "qty": qty,
                     "planned_qty": planned, "entry": entry, "stop": np.nan if rng.random() < 0.03 else stop,
                     "exit": exit_, "exit_reason": "stop" if r <= -1 else "target" if r >= 2 else "time"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- clinic W3: a day with faults
FAULTS = {"bad_tick": 7_000, "stale_data": 12_000, "latency_spike": 150, "runaway_order_rate": 300,
          "duplicate_fill": "F0042"}


def trading_day(seed: int = 0, faults: bool = True) -> dict:
    """A replayable session with INJECTED faults (positions in FAULTS; the truth for grading):
    ticks: DataFrame ts (seconds from the open), price — one tick per second, except a 30-second silence starting at
    FAULTS["stale_data"] and a bad print (price × 1.08) at FAULTS["bad_tick"];
    acks: order-ack latencies (s), one 400 ms spike at FAULTS["latency_spike"];
    order_counts: orders per minute (390 minutes), a runaway burst of 60 at minute FAULTS["runaway_order_rate"];
    fills: list of {fill_id, symbol, qty}, one fill delivered twice (FAULTS["duplicate_fill"]);
    broker_positions: the broker's (true) positions."""
    rng = np.random.default_rng(seed)
    n = 23_400
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.0002, n)))
    ts = np.arange(n)
    if faults:
        price[FAULTS["bad_tick"]] *= 1.08
        keep = (ts < FAULTS["stale_data"]) | (ts >= FAULTS["stale_data"] + 30)
        ts, price = ts[keep], price[keep]
    acks = rng.lognormal(np.log(0.04), 0.25, 400)
    counts = rng.poisson(3, 390).astype(float)
    if faults:
        acks[FAULTS["latency_spike"]] = 0.4
        counts[FAULTS["runaway_order_rate"]] = 60
    fills, pos = [], {}
    for i in range(80):
        sym = str(rng.choice(["SPY", "QQQ", "IWM"]))
        qty = int(rng.choice([-100, 100]))
        fills.append({"fill_id": f"F{i:04d}", "symbol": sym, "qty": qty})
        pos[sym] = pos.get(sym, 0) + qty
    if faults:
        dup = next(f for f in fills if f["fill_id"] == FAULTS["duplicate_fill"])
        fills.append(dict(dup))
    return {"ticks": pd.DataFrame({"ts": ts, "price": price}), "acks": acks, "order_counts": counts,
            "fills": fills, "broker_positions": {k: v for k, v in pos.items() if v != 0}}


# ------------------------------------------------------------------------ clinic W5: a track record
def track_record(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest daily returns (1000 days) and a 4-week live record (20 days) for three strategies with known daily
    mean / vol: momentum (0.05% / 1.0%), pairs (0.04% / 0.5%), options (0.06% / 0.8%). LIVE, momentum and pairs
    behave like their backtest; the options strategy has STOPPED WORKING and bleeds (−0.8% a day)."""
    rng = np.random.default_rng(seed)
    spec = {"momentum": (0.0005, 0.010), "pairs": (0.0004, 0.005), "options": (0.0006, 0.008)}
    bt = pd.DataFrame({k: rng.normal(m, s, 1000) for k, (m, s) in spec.items()})
    live = pd.DataFrame({k: rng.normal(-0.008 if k == "options" else m, s, 20) for k, (m, s) in spec.items()},
                        index=pd.bdate_range("2025-11-03", periods=20))
    return bt, live

# ======================================================================================
# Reference implementations: week 41 (S1–S4) integration & performance
# ======================================================================================
# ------------------------------------------------------------------------ S1 architecture rules
def module_imports(path: Path) -> set[str]:
    """Every module a Python file imports, ANYWHERE in the file (also inside functions): ast.walk over Import
    (each alias name) and ImportFrom with level 0 (the module, plus module.name for each imported name)."""
    tree = ast.parse(Path(path).read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def import_graph(pkg_dir: Path) -> dict[str, set[str]]:
    """For every .py file under the package directory: dotted module name (the package dir's name first, no
    ".__init__") → the set of imported modules that belong to the same package (start with "<package>.")."""
    pkg_dir = Path(pkg_dir)
    root = pkg_dir.name
    graph = {}
    for f in sorted(pkg_dir.rglob("*.py")):
        parts = [root, *f.relative_to(pkg_dir).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        graph[".".join(parts)] = {m for m in module_imports(f) if m == root or m.startswith(root + ".")}
    return graph


def layer_of(module: str, layers: list[str]) -> int | None:
    """Index of the module's layer (its second dotted component) in `layers` (0 = top), None if not a layer."""
    parts = module.split(".")
    return layers.index(parts[1]) if len(parts) > 1 and parts[1] in layers else None


def check_layers(graph: dict[str, set[str]], layers: list[str]) -> list[tuple[str, str]]:
    """Layered architecture: a module may import its own layer or LOWER ones (larger index) only. Return the sorted
    (importer, imported) pairs that import a HIGHER layer."""
    bad = []
    for mod, imports in graph.items():
        li = layer_of(mod, layers)
        if li is None:
            continue
        for imp in imports:
            lj = layer_of(imp, layers)
            if lj is not None and lj < li:
                bad.append((mod, imp))
    return sorted(bad)


def check_forbidden(graph: dict[str, set[str]], source: str, forbidden: list[str]) -> list[tuple[str, str]]:
    """Modules equal to or under `source` that import a module equal to or under any `forbidden` one (sorted)."""
    under = lambda m, p: m == p or m.startswith(p + ".")                                  # noqa: E731
    return sorted((mod, imp) for mod, imports in graph.items() if under(mod, source)
                  for imp in imports if any(under(imp, f) for f in forbidden))


# ---------------------------------------------------------------------------- S2 performance
class LatencyHistogram:
    """Prometheus-style histogram: cumulative bucket counts (upper bounds, +inf last), sum and count."""

    def __init__(self, buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1)):
        self.bounds = [*buckets, float("inf")]
        self.counts = [0] * len(self.bounds)
        self.total, self.n = 0.0, 0

    def observe(self, x: float) -> None:
        """Count x in every bucket whose upper bound is >= x (cumulative, like Prometheus)."""
        for i, b in enumerate(self.bounds):
            if x <= b:
                self.counts[i] += 1
        self.total += x
        self.n += 1

    def quantile(self, q: float) -> float:
        """Upper bound of the first bucket whose cumulative count reaches q·n (a conservative estimate)."""
        target = q * self.n
        return next(b for b, c in zip(self.bounds, self.counts) if c >= target)

    @contextmanager
    def time(self):
        """with hist.time(): … observes the elapsed seconds (time.perf_counter)."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - t0)


class RingBuffer:
    """Fixed memory for the last `size` values (a preallocated array and a write position): no growing DataFrames
    in the hot path."""

    def __init__(self, size: int):
        self.buf = np.full(size, np.nan)
        self.size, self.pos, self.n = size, 0, 0

    def append(self, x: float) -> None:
        self.buf[self.pos] = x
        self.pos = (self.pos + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def values(self) -> np.ndarray:
        """The stored values, OLDEST first."""
        if self.n < self.size:
            return self.buf[: self.n].copy()
        return np.r_[self.buf[self.pos:], self.buf[: self.pos]]


class StreamingEMA:
    """O(1) per update; must equal pandas ewm(span=n, adjust=False).mean()."""

    def __init__(self, n: int):
        self.alpha, self.value = 2 / (n + 1), None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return self.value


async def _measure_lag(work: Callable[[], object], in_executor: bool, tick: float = 0.01) -> float:
    lags, done = [], asyncio.Event()

    async def ticker():
        while not done.is_set():
            t0 = time.perf_counter()
            await asyncio.sleep(tick)
            lags.append(time.perf_counter() - t0 - tick)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(tick * 2)
    if in_executor:
        await asyncio.get_running_loop().run_in_executor(None, work)
    else:
        work()                                                                            # blocks the loop
    await asyncio.sleep(tick * 2)
    done.set()
    await task
    return max(lags)


def event_loop_lag(work: Callable[[], object], in_executor: bool) -> float:
    """Max scheduling delay of a 10 ms ticker while `work` runs on the loop (in_executor=False) or in a thread pool
    (True). Given — the lesson: CPU work on the loop delays everything else."""
    return asyncio.run(_measure_lag(work, in_executor))


# ------------------------------------------------------------------------ S3 strategy creator
def ema(x, n):
    return pd.Series(np.asarray(x, float)).ewm(span=n, adjust=False).mean().to_numpy()


def sma(x, n):
    return pd.Series(np.asarray(x, float)).rolling(n).mean().to_numpy()


def rsi(x, n=14):
    d = pd.Series(np.asarray(x, float)).diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn)).to_numpy()


def crossover(a, b):
    a, b = np.broadcast_to(a, np.shape(b) if np.ndim(a) == 0 else np.shape(a)), np.asarray(b)
    prev = np.r_[False, (a[:-1] <= b[:-1]) if np.ndim(b) else (a[:-1] <= b)]
    return (a > b) & prev


REGISTRY = {"indicators": {"ema": ema, "sma": sma, "rsi": rsi},
            "actions": {"crossover": crossover, "gt": lambda a, b: np.asarray(a) > b,
                        "lt": lambda a, b: np.asarray(a) < b},
            "params": {"ema": {"n": (2, 500)}, "sma": {"n": (2, 500)}, "rsi": {"n": (2, 100)}}}
REQUIRED = ("name", "entry", "exit", "sizing")


def build_series(spec, registry=REGISTRY):
    """Lesson plan S3: a number → constant; a string → that column of the data; {"ind", "params", "input"} → the
    registered indicator on d[input] (default "close")."""
    if isinstance(spec, (int, float)):
        return lambda d: spec
    if isinstance(spec, str):
        return lambda d: d[spec].to_numpy()
    ind = registry["indicators"][spec["ind"]]
    return lambda d: ind(d[spec.get("input", "close")], **spec.get("params", {}))


def build_condition(spec, registry=REGISTRY):
    """Lesson plan S3: {"all": […]}, {"any": […]}, {"not": {…}} and {"fn": name, "args": […]} → function(data) →
    boolean array."""
    if "all" in spec:
        parts = [build_condition(s, registry) for s in spec["all"]]
        return lambda d: np.logical_and.reduce([p(d) for p in parts])
    if "any" in spec:
        parts = [build_condition(s, registry) for s in spec["any"]]
        return lambda d: np.logical_or.reduce([p(d) for p in parts])
    if "not" in spec:
        inner = build_condition(spec["not"], registry)
        return lambda d: ~inner(d)
    fn = registry["actions"][spec["fn"]]
    args = [build_series(a, registry) for a in spec["args"]]
    return lambda d: fn(*[a(d) for a in args])


def validate_config(cfg: dict, registry=REGISTRY) -> list[str]:
    """Reject a config BEFORE it runs. Errors (strings): "missing <key>" for each REQUIRED key; walking the entry
    tree: "unknown action <fn>", "unknown indicator <ind>", "unknown parameter <ind>.<p>", "<ind>.<p>=<v> outside
    [lo, hi]". [] = valid."""
    errors = [f"missing {k}" for k in REQUIRED if k not in cfg]

    def walk_series(s):
        if isinstance(s, dict):
            ind = s.get("ind")
            if ind not in registry["indicators"]:
                errors.append(f"unknown indicator {ind}")
                return
            ranges = registry["params"].get(ind, {})
            for p, v in s.get("params", {}).items():
                if p not in ranges:
                    errors.append(f"unknown parameter {ind}.{p}")
                elif not ranges[p][0] <= v <= ranges[p][1]:
                    errors.append(f"{ind}.{p}={v} outside [{ranges[p][0]}, {ranges[p][1]}]")

    def walk(c):
        for key in ("all", "any"):
            if key in c:
                for s in c[key]:
                    walk(s)
                return
        if "not" in c:
            walk(c["not"])
            return
        if c.get("fn") not in registry["actions"]:
            errors.append(f"unknown action {c.get('fn')}")
        for a in c.get("args", []):
            walk_series(a)

    if "entry" in cfg:
        walk(cfg["entry"])
    return errors


def config_hash(cfg: dict) -> str:
    """First 12 hex characters of sha256(json.dumps(cfg, sort_keys=True)): stored with every trade."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_strategy(text: str, registry=REGISTRY) -> tuple[dict, Callable, str]:
    """yaml.safe_load the config, validate it (ValueError listing every error if invalid), and return (config,
    entry-signal function, config hash)."""
    cfg = yaml.safe_load(text)
    errors = validate_config(cfg, registry)
    if errors:
        raise ValueError("; ".join(errors))
    return cfg, build_condition(cfg["entry"], registry), config_hash(cfg)


# ---------------------------------------------------------------------------------- S4 screeners
def min_adv(usd: float):
    """Filter factory: average daily dollar volume >= usd. Each filter maps a snapshot to a boolean Series."""
    return lambda df: df["adv_usd"] >= usd


def price_between(lo: float, hi: float):
    return lambda df: df["price"].between(lo, hi)


def max_spread(bps: float):
    return lambda df: df["spread_bps"] <= bps


def sector_in(sectors):
    return lambda df: df["sector"].isin(set(sectors))


def min_option_oi(oi: int):
    return lambda df: df["option_oi"] >= oi


def shortable():
    return lambda df: df["shortable"].astype(bool)


def screen(df: pd.DataFrame, filters: list, limit: int | None = None) -> list[str]:
    """Symbols passing ALL filters, sorted by adv_usd descending (then symbol), at most `limit`."""
    mask = np.logical_and.reduce([f(df) for f in filters]) if filters else np.ones(len(df), bool)
    out = df[mask].sort_values(["adv_usd", "symbol"], ascending=[False, True])["symbol"].tolist()
    return out if limit is None else out[:limit]


def universe_turnover(old: list[str], new: list[str]) -> float:
    """Share of the new universe that was NOT in the old one (names you must trade into); 0 if new is empty."""
    return len(set(new) - set(old)) / len(new) if new else 0.0


class UniverseStore:
    """Stored screen results, so a live universe can be reproduced: (date, screen name) → config hash + symbols."""

    def __init__(self):
        self.rows: dict[tuple[pd.Timestamp, str], dict] = {}

    def save(self, date, name: str, cfg: dict, symbols: list[str]) -> None:
        self.rows[(pd.Timestamp(date), name)] = {"config_hash": config_hash(cfg), "symbols": list(symbols)}

    def get(self, date, name: str) -> dict | None:
        """The latest saved result ON OR BEFORE date for that screen (point-in-time), else None."""
        keys = [k for k in self.rows if k[1] == name and k[0] <= pd.Timestamp(date)]
        return self.rows[max(keys)] if keys else None

# ======================================================================================
# Week 42 (S5–S8) execution & trade management
# ======================================================================================
# ------------------------------------------------------------------------ S5 spread-aware orders
def limit_price(side: str, bid: float, ask: float, tick: float, aggressiveness: float = 0.0) -> float:
    """Lesson plan S5: BUY at bid + a·(ask − bid), SELL at ask − a·(ask − bid); round to the tick with Decimal in the
    direction that never overpays (BUY down, SELL up)."""
    px = bid + aggressiveness * (ask - bid) if side == "BUY" else ask - aggressiveness * (ask - bid)
    q = Decimal(str(tick))
    steps = (Decimal(str(px)) / q).to_integral_value(rounding=ROUND_DOWN if side == "BUY" else ROUND_UP)
    return float(steps * q)


def simulate_chase(side: str, quotes: pd.DataFrame, start: int, schedule=(0.0, 0.33, 0.67, 1.0), wait: int = 5,
                   tick: float = 0.01, markout: int = 30) -> dict:
    """Offline chase on a quote path (one row per second). For each aggressiveness in the schedule: price with
    limit_price from the quote at the current second, then watch the NEXT `wait` seconds; a BUY fills at its limit
    in the first second where ask <= limit (marketable) or the last trade <= limit (a seller hit us); SELL
    mirrored. Unfilled after the schedule → cancel. sign = +1 BUY, −1 SELL. Return {"filled", "price" (NaN if not),
    "seconds" (start → fill, or time spent), "aggr" (level that filled, NaN), "cost_bps": sign·(price − arrival
    mid)/arrival mid·1e4, "markout_bps": sign·(mid `markout` seconds after the fill − price)/price·1e4 (what the
    fill is worth soon after: negative = adverse selection); both NaN if unfilled, "all_in_bps": cost_bps if filled,
    else the cost of completing by CROSSING at the last quote watched (BUY at its ask, SELL at its bid) vs the
    arrival mid — non-fill is not free}."""
    sign = 1 if side == "BUY" else -1
    mid = (quotes["bid"] + quotes["ask"]) / 2
    mid0 = mid.iloc[start]
    t = start
    for aggr in schedule:
        q = quotes.iloc[t]
        px = limit_price(side, q["bid"], q["ask"], tick, aggr)
        for s in range(t + 1, min(t + 1 + wait, len(quotes))):
            r = quotes.iloc[s]
            hit = (r["ask"] <= px or r["trade"] <= px) if side == "BUY" else (r["bid"] >= px or r["trade"] >= px)
            if hit:
                later = mid.iloc[min(s + markout, len(quotes) - 1)]
                return {"filled": True, "price": px, "seconds": s - start, "aggr": aggr,
                        "cost_bps": sign * (px - mid0) / mid0 * 1e4, "markout_bps": sign * (later - px) / px * 1e4,
                        "all_in_bps": sign * (px - mid0) / mid0 * 1e4}
        t = min(t + wait, len(quotes) - 1)
    r = quotes.iloc[t]
    cross = r["ask"] if side == "BUY" else r["bid"]
    return {"filled": False, "price": np.nan, "seconds": t - start, "aggr": np.nan, "cost_bps": np.nan,
            "markout_bps": np.nan, "all_in_bps": sign * (cross - mid0) / mid0 * 1e4}


def chase_report(quotes: pd.DataFrame, schedules: dict[str, tuple], n_orders: int = 200, seed: int = 0,
                 wait: int = 5) -> pd.DataFrame:
    """Run simulate_chase for n_orders random (side, start) pairs (rng = default_rng(seed): sides uniform in
    BUY/SELL, starts in [0, len − 100)) for every schedule. Per schedule: fill_rate, avg_seconds, avg_cost_bps and
    avg_markout_bps (filled orders only), avg_all_in_bps (all orders). Index = schedule name."""
    rng = np.random.default_rng(seed)
    orders = [(str(rng.choice(["BUY", "SELL"])), int(rng.integers(0, len(quotes) - 100))) for _ in range(n_orders)]
    rows = {}
    for name, sched in schedules.items():
        res = pd.DataFrame([simulate_chase(s, quotes, t, sched, wait) for s, t in orders])
        f = res[res["filled"]]
        rows[name] = {"fill_rate": float(res["filled"].mean()), "avg_seconds": float(f["seconds"].mean()),
                      "avg_cost_bps": float(f["cost_bps"].mean()), "avg_markout_bps": float(f["markout_bps"].mean()),
                      "avg_all_in_bps": float(res["all_in_bps"].mean())}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- S6 algos & TCA
def twap_schedule(qty: int, start, end, slices: int) -> pd.Series:
    """Lesson plan S6: equal integer slices at `slices` times from start (end excluded); the remainder goes to the
    first slices, one share each."""
    times = pd.date_range(start, end, periods=slices + 1)[:-1]
    base = qty // slices
    return pd.Series([base + (1 if i < qty - base * slices else 0) for i in range(slices)], index=times)


def vwap_schedule(qty: int, volume_profile: pd.Series) -> pd.Series:
    """Lesson plan S6: integer sizes proportional to the volume profile, remainder by largest fractional parts."""
    raw = volume_profile / volume_profile.sum() * qty
    sizes = np.floor(raw).astype(int)
    sizes.iloc[np.argsort(-(raw - sizes).to_numpy(), kind="stable")[: int(qty - sizes.sum())]] += 1
    return sizes


def pov_child_qty(market_volume_since_last: float, participation: float, remaining: int) -> int:
    """Lesson plan S6."""
    return int(min(remaining, np.floor(participation * market_volume_since_last)))


def implementation_shortfall_bps(side: str, decision_px: float, fills: list[tuple[float, float]],
                                 fees: float = 0.0) -> float:
    """Lesson plan S6: cost vs the decision price in bps (positive = cost)."""
    qty = sum(q for _, q in fills)
    avg = sum(p * q for p, q in fills) / qty
    sign = 1 if side == "BUY" else -1
    return 1e4 * (sign * (avg - decision_px) * qty + fees) / (decision_px * qty)


def execute_schedule(side: str, schedule: pd.Series, session: pd.DataFrame, eta: float = 0.02) -> list[tuple]:
    """Fill each child at its minute's price moved AGAINST us by temporary impact η·(child / minute volume)·price
    (a child that is a big share of the minute's volume pays more). Zero-size children are skipped. Returns fills
    [(price, qty)]."""
    sign = 1 if side == "BUY" else -1
    fills = []
    for ts, q in schedule.items():
        if q <= 0:
            continue
        row = session.loc[ts]
        fills.append((row["price"] * (1 + sign * eta * q / row["volume"]), float(q)))
    return fills


# --------------------------------------------------------------------- S7 option position manager
class OptionPositionManager:
    """Management rules for a short-premium option position, checked each bar in this PRIORITY order:
    stop (cost to close >= credit·(1 + stop_mult)) → take_profit (cost to close <= credit·(1 − tp_frac)) →
    expiry (dte == 0) → ex_dividend (a short call is ITM and days_to_exdiv <= 1: early-assignment risk) →
    adjust (any |short delta| > max_delta) → roll (dte <= roll_dte). Returns (action, reason) or None."""

    def __init__(self, tp_frac: float = 0.5, stop_mult: float = 2.0, roll_dte: int = 21, max_delta: float = 0.30):
        self.tp, self.stop, self.roll_dte, self.max_delta = tp_frac, stop_mult, roll_dte, max_delta

    def evaluate(self, pos: dict):
        """pos: credit, value (cost to close now), dte, short_deltas (list), short_call_itm (bool),
        days_to_exdiv (int or None)."""
        if pos["value"] >= pos["credit"] * (1 + self.stop):
            return "close", "stop"
        if pos["value"] <= pos["credit"] * (1 - self.tp):
            return "close", "take_profit"
        if pos["dte"] == 0:
            return "close", "expiry"
        dx = pos.get("days_to_exdiv")
        if pos.get("short_call_itm") and dx is not None and dx <= 1:
            return "close", "ex_dividend"
        if any(abs(d) > self.max_delta for d in pos["short_deltas"]):
            return "adjust", "short_delta"
        if pos["dte"] <= self.roll_dte:
            return "roll", "dte"
        return None


# ---------------------------------------------------------------------------------- S8 journal
def r_multiples(journal: pd.DataFrame) -> pd.Series:
    """R = side·(exit − entry) / |entry − stop|: P&L in units of the PLANNED risk; NaN when there is no stop."""
    return journal["side"] * (journal["exit"] - journal["entry"]) / (journal["entry"] - journal["stop"]).abs()


def journal_stats(journal: pd.DataFrame, by="setup_tag") -> pd.DataFrame:
    """Lesson plan S8 on the r_multiple column (rows with NaN R dropped): trades, win_rate, avg_win_R, avg_loss_R,
    expectancy_R per group, sorted by expectancy descending."""
    j = journal.dropna(subset=["r_multiple"])
    g = j.groupby(by)["r_multiple"]
    return pd.DataFrame({"trades": g.size(), "win_rate": g.apply(lambda r: (r > 0).mean()),
                         "avg_win_R": g.apply(lambda r: r[r > 0].mean()),
                         "avg_loss_R": g.apply(lambda r: r[r <= 0].mean()),
                         "expectancy_R": g.mean()}).sort_values("expectancy_R", ascending=False)


def rule_violations(journal: pd.DataFrame, open_="09:30", close="16:00") -> pd.DataFrame:
    """One row per violation (columns trade_id, rule): "no_stop" (stop missing), "outside_hours" (entry time not in
    [open, close)), "oversized" (qty > planned_qty). Sorted by trade_id, then rule."""
    t = journal["entry_time"].dt.strftime("%H:%M")
    checks = {"no_stop": journal["stop"].isna(), "outside_hours": ~((t >= open_) & (t < close)),
              "oversized": journal["qty"] > journal["planned_qty"]}
    rows = [{"trade_id": tid, "rule": rule} for rule, mask in checks.items() for tid in journal.loc[mask, "trade_id"]]
    return pd.DataFrame(rows, columns=["trade_id", "rule"]).sort_values(["trade_id", "rule"], ignore_index=True)

# ======================================================================================
# Week 43 (S9–S12) the monitoring department
# ======================================================================================
# ------------------------------------------------------------------------- S9 observability
class StructuredLogger:
    """JSON lines with a correlation id that follows one idea from signal → intent → order → fill."""

    def __init__(self, now: Callable[[], float] = time.time):
        self.lines: list[str] = []
        self.now = now

    def log(self, level: str, msg: str, cid: str, **fields) -> None:
        """Append json.dumps({"ts", "level", "msg", "cid", **fields}, sort_keys=True, default=str)."""
        self.lines.append(json.dumps({"ts": self.now(), "level": level, "msg": msg, "cid": cid, **fields},
                                     sort_keys=True, default=str))

    def trace(self, cid: str) -> list[dict]:
        """Every record of one correlation id, in order: the story of one trade."""
        return [r for r in map(json.loads, self.lines) if r["cid"] == cid]


def make_metrics(registry: CollectorRegistry) -> dict:
    """The lesson plan S2/S9 metrics, registered on `registry` (not the global one): orders (Counter
    qf_orders_total by strategy, broker), rejects (Counter qf_rejects_total by broker), latency (Histogram
    qf_signal_to_submit_seconds with buckets 0.001, 0.005, 0.01, 0.025, 0.05, 0.1), equity (Gauge qf_equity_usd by
    account), staleness (Gauge qf_data_staleness_seconds by feed)."""
    return {"orders": Counter("qf_orders_total", "Orders sent", ["strategy", "broker"], registry=registry),
            "rejects": Counter("qf_rejects_total", "Orders rejected", ["broker"], registry=registry),
            "latency": Histogram("qf_signal_to_submit_seconds", "Bar received -> order submitted",
                                 buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1), registry=registry),
            "equity": Gauge("qf_equity_usd", "Account equity", ["account"], registry=registry),
            "staleness": Gauge("qf_data_staleness_seconds", "Age of the last tick", ["feed"], registry=registry)}


def exposition(registry: CollectorRegistry) -> str:
    """What Prometheus scrapes (given)."""
    return generate_latest(registry).decode()


SEVERITY = {"critical": 0, "warning": 1, "info": 2}


def evaluate_alerts(state: dict, rules: list[tuple[str, Callable[[dict], bool], str]]) -> list[tuple[str, str]]:
    """Fire every rule whose condition(state) is True; return (name, severity) sorted by severity (critical
    first), then name."""
    fired = [(name, sev) for name, cond, sev in rules if cond(state)]
    return sorted(fired, key=lambda x: (SEVERITY[x[1]], x[0]))


def slo_report(values: pd.Series, threshold: float, target: float = 0.999) -> dict:
    """SLO "value <= threshold for `target` of the time": {"compliance": share OK, "met": compliance >= target,
    "budget_used": share of bad samples / (1 − target) (1.0 = the whole error budget spent)}."""
    ok = float((values <= threshold).mean())
    return {"compliance": ok, "met": ok >= target, "budget_used": (1 - ok) / (1 - target)}


# -------------------------------------------------------------- S10 audit, reconciliation, approvals
def _digest(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()


class AuditLog:
    """Lesson plan S10: append-only, hash-chained. Each record: ts, event, data, prev (the previous hash, 64 zeros
    for the first), hash = sha256 of the record WITHOUT its hash (json, sorted keys)."""

    def __init__(self, now: Callable[[], float] = time.time):
        self.records: list[dict] = []
        self._last = "0" * 64
        self.now = now

    def append(self, event: str, **data) -> str:
        rec = {"ts": self.now(), "event": event, "data": data, "prev": self._last}
        rec["hash"] = _digest(rec)
        self.records.append(rec)
        self._last = rec["hash"]
        return rec["hash"]

    def verify(self) -> int:
        """Index of the first broken record (wrong prev link or wrong hash), −1 if the chain is intact."""
        prev = "0" * 64
        for i, rec in enumerate(self.records):
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec["prev"] != prev or _digest(body) != rec["hash"]:
                return i
            prev = rec["hash"]
        return -1

    def head(self) -> str:
        """The last hash: send it out daily (n8n e-mail) so history cannot be rewritten unnoticed."""
        return self._last


def reconcile_positions(internal: dict[str, float], broker: dict[str, float], tol: float = 1e-9) -> list[dict]:
    """Breaks between our books and the broker's, over the union of symbols (missing = 0): [{"symbol", "internal",
    "broker", "diff" (broker − internal)}] sorted by symbol. The BROKER is the source of truth."""
    out = []
    for s in sorted(set(internal) | set(broker)):
        a, b = internal.get(s, 0.0), broker.get(s, 0.0)
        if abs(a - b) > tol:
            out.append({"symbol": s, "internal": a, "broker": b, "diff": b - a})
    return out


def reconcile_orders(internal_open: set[str], broker_open: set[str]) -> dict[str, list[str]]:
    """Open-order breaks by id: {"unknown_at_broker": our open orders the broker does not have (sorted),
    "orphan_at_broker": broker orders we did not send, e.g. a manual TWS order (sorted)}."""
    return {"unknown_at_broker": sorted(internal_open - broker_open),
            "orphan_at_broker": sorted(broker_open - internal_open)}


class ApprovalQueue:
    """Four-eyes approvals for strategies, parameter and limit changes and large orders. Every step is audited
    (events approval_requested / approval_granted / approval_rejected)."""

    def __init__(self, audit: AuditLog):
        self.audit, self.items = audit, {}

    def request(self, kind: str, payload: dict, requester: str, reason: str) -> str:
        """Id A0001, A0002, …; status "pending"."""
        aid = f"A{len(self.items) + 1:04d}"
        self.items[aid] = {"id": aid, "kind": kind, "payload": payload, "requester": requester, "reason": reason,
                           "status": "pending"}
        self.audit.append("approval_requested", id=aid, kind=kind, requester=requester, reason=reason)
        return aid

    def decide(self, aid: str, approver: str, approve: bool, reason_code: str) -> None:
        """ValueError if the item is not pending or approver == requester (four eyes). Status "approved" or
        "rejected", with approver and reason_code recorded."""
        item = self.items[aid]
        if item["status"] != "pending":
            raise ValueError(f"{aid} is {item['status']}")
        if approver == item["requester"]:
            raise ValueError("four-eyes: the requester cannot approve")
        item.update(status="approved" if approve else "rejected", approver=approver, reason_code=reason_code)
        self.audit.append("approval_granted" if approve else "approval_rejected", id=aid, approver=approver,
                          reason_code=reason_code)


def needs_approval(order: dict, max_notional: float) -> bool:
    """Large orders need a human: |qty| · price > max_notional."""
    return abs(order["qty"]) * order["price"] > max_notional


# ------------------------------------------------------------------------- S11 anomalies
class RobustZ:
    """Lesson plan S11: streaming robust z-score (median / MAD·1.4826) over the last `window` values, computed from
    the values BEFORE x; flags only after min_obs values."""

    def __init__(self, window: int = 500, threshold: float = 6.0, min_obs: int = 50):
        self.buf, self.window, self.threshold, self.min_obs = [], window, threshold, min_obs

    def update(self, x: float) -> tuple[float, bool]:
        z = 0.0
        if len(self.buf) >= self.min_obs:
            arr = np.asarray(self.buf)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826 + 1e-12
            z = (x - med) / mad
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        return z, abs(z) > self.threshold


def fat_tail_error(realized_pnl: float, expected_sigma: float, k: float = 5.0) -> bool:
    """Lesson plan S11: a P&L move beyond k sigma is more often a data/position/fill error than a real event."""
    return abs(realized_pnl) > k * expected_sigma


class Cusum:
    """Two-sided CUSUM on standardized values z = (x − mean)/sd: s+ = max(0, s+ + z − k), s− = max(0, s− − z − k);
    alarm (and reset both) when either exceeds h. Detects a slow DRIFT that no single observation reveals."""

    def __init__(self, mean: float, sd: float, k: float = 0.5, h: float = 5.0):
        self.mean, self.sd, self.k, self.h = mean, sd, k, h
        self.pos = self.neg = 0.0

    def update(self, x: float) -> bool:
        z = (x - self.mean) / self.sd
        self.pos = max(0.0, self.pos + z - self.k)
        self.neg = max(0.0, self.neg - z - self.k)
        if self.pos > self.h or self.neg > self.h:
            self.pos = self.neg = 0.0
            return True
        return False


CONTROLLER_POLICY = {"stale_data": "pause_strategy", "latency_spike": "alert", "reject_burst": "pause_strategy",
                     "slippage_outlier": "alert", "fat_tail_pnl": "pause_all_and_reconcile",
                     "runaway_order_rate": "trip_kill_switch"}


class Controller:
    """Maps anomalies to actions (lesson plan policy; unknown anomalies → "alert"). pause_strategy pauses that
    strategy; pause_all_and_reconcile pauses every strategy in `strategies` and sets needs_reconcile;
    trip_kill_switch sets killed and pauses everything. Every decision is audited ("controller_action" with anomaly,
    strategy, action). A paused strategy resumes ONLY with an approved approval of kind "resume" for that strategy;
    nothing resumes while killed."""

    def __init__(self, strategies: list[str], audit: AuditLog, approvals: ApprovalQueue,
                 policy: dict[str, str] = CONTROLLER_POLICY):
        self.strategies, self.audit, self.approvals, self.policy = list(strategies), audit, approvals, policy
        self.paused: set[str] = set()
        self.killed = False
        self.needs_reconcile = False

    def handle(self, anomaly: str, strategy: str | None = None) -> str:
        action = self.policy.get(anomaly, "alert")
        if action == "pause_strategy" and strategy:
            self.paused.add(strategy)
        elif action == "pause_all_and_reconcile":
            self.paused |= set(self.strategies)
            self.needs_reconcile = True
        elif action == "trip_kill_switch":
            self.killed = True
            self.paused |= set(self.strategies)
        self.audit.append("controller_action", anomaly=anomaly, strategy=strategy, action=action)
        return action

    def resume(self, strategy: str, approval_id: str) -> bool:
        """True (and audited "resumed") only if not killed and the approval is approved, of kind "resume" and for
        this strategy (payload["strategy"])."""
        a = self.approvals.items.get(approval_id)
        ok = (not self.killed and a is not None and a["status"] == "approved" and a["kind"] == "resume"
              and a["payload"].get("strategy") == strategy)
        if ok:
            self.paused.discard(strategy)
            self.audit.append("resumed", strategy=strategy, approval=approval_id)
        return ok


# ---------------------------------------------------------------------------- S12 reports
def daily_report(fills: pd.DataFrame, equity: pd.Series) -> dict:
    """fills: side (+1/−1), qty, price, arrival_mid. equity: end-of-day values. {"pnl": last − previous equity,
    "return": pnl / previous, "drawdown": last / running max − 1, "n_fills", "slippage_bps": qty-weighted mean of
    side·(price − arrival_mid)/arrival_mid·1e4}."""
    slip = fills["side"] * (fills["price"] - fills["arrival_mid"]) / fills["arrival_mid"] * 1e4
    return {"pnl": float(equity.iloc[-1] - equity.iloc[-2]), "return": float(equity.iloc[-1] / equity.iloc[-2] - 1),
            "drawdown": float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1), "n_fills": int(len(fills)),
            "slippage_bps": float(np.average(slip, weights=fills["qty"]))}


def backtest_band(daily_returns, horizon: int = 20, n_boot: int = 2000, q=(0.05, 0.95), seed: int = 0
                  ) -> pd.DataFrame:
    """Expected range of a live track record: bootstrap `horizon`-day paths from the backtest's daily returns
    (rng.choice with replacement), cumulative SUM per path; per day 1..horizon the q quantiles. Columns lower,
    upper; index 1..horizon."""
    r = np.asarray(daily_returns, dtype=float)
    rng = np.random.default_rng(seed)
    paths = rng.choice(r, size=(n_boot, horizon), replace=True).cumsum(axis=1)
    lo, hi = np.quantile(paths, q, axis=0)
    return pd.DataFrame({"lower": lo, "upper": hi}, index=range(1, horizon + 1))


def inside_band(live_returns, band: pd.DataFrame, lower_only: bool = False) -> pd.Series:
    """For each live day d (1-based), is the cumulative live return within [lower, upper] of day d? With
    lower_only, only "not below the band" counts (beating the band calls for investigation, not demotion)."""
    cum = np.cumsum(np.asarray(live_returns, dtype=float))
    b = band.iloc[: len(cum)]
    ok = cum >= b["lower"].to_numpy()
    if not lower_only:
        ok &= cum <= b["upper"].to_numpy()
    return pd.Series(ok, index=b.index)


def attribution(pnl: pd.DataFrame) -> pd.DataFrame:
    """P&L by strategy (columns of daily P&L): total, share of the total, sharpe (annualized, ddof=1); sorted by
    total descending."""
    total = pnl.sum()
    return pd.DataFrame({"total": total, "share": total / total.sum(),
                         "sharpe": pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)}).sort_values("total", ascending=False)

# ======================================================================================
# Week 44 (S13–S16) production engineering
# ======================================================================================
DEPLOY_DIR = Path(__file__).resolve().parent / "deploy"      # the committed docker-compose.yml and ci.yml
CRITICAL = ("engine", "api", "ib-gateway", "db")
SECRET_VAR = re.compile(r"PASSWORD|SECRET|TOKEN|API_KEY|PASS$", re.I)


# ------------------------------------------------------------------------------ S13 deployment
def check_compose(spec: dict) -> list[str]:
    """Problems in a parsed docker-compose file ([] = passes), in this order per service (services in file order):
    "unpinned image: <svc>" — an image with no ":tag", or a tag containing "latest" (a digest "@sha256:" is pinned);
    "inline secret: <svc>.<VAR>" — an `environment` entry whose NAME matches SECRET_VAR and whose value is not a
    "${...}" reference (environment may be a mapping or a list of "KEY=value"); "no restart policy: <svc>" for CRITICAL services; "no healthcheck: db";
    "exposed port: <svc>" — any service except "proxy" publishing ports; "unhealthy dependency: <svc>" — a service
    that depends on db without condition service_healthy (a list-style depends_on counts as unconditional)."""
    problems = []
    for name, svc in spec.get("services", {}).items():
        image = svc.get("image")
        if image and "@sha256:" not in image:
            tag = image.rsplit(":", 1)[1] if ":" in image.split("/")[-1] else ""
            if not tag or "latest" in tag:
                problems.append(f"unpinned image: {name}")
        env = svc.get("environment") or {}
        if isinstance(env, list):                                                     # ["KEY=value", …] form
            env = dict(e.split("=", 1) if "=" in e else (e, "") for e in env)
        for var, val in env.items():
            if SECRET_VAR.search(str(var)) and not str(val).startswith("${"):
                problems.append(f"inline secret: {name}.{var}")
        if name in CRITICAL and "restart" not in svc:
            problems.append(f"no restart policy: {name}")
        if name == "db" and "healthcheck" not in svc:
            problems.append("no healthcheck: db")
        if svc.get("ports") and name != "proxy":
            problems.append(f"exposed port: {name}")
        dep = svc.get("depends_on", {})
        if "db" in dep and (isinstance(dep, list) or dep["db"].get("condition") != "service_healthy"):
            problems.append(f"unhealthy dependency: {name}")
    return problems


REQUIRED_CI = {"lint": "ruff check", "types": "mypy --strict", "architecture": "lint-imports",
               "coverage": "--cov-fail-under", "regression": "tests/regression"}


def check_ci(workflow: dict) -> list[str]:
    """Names of REQUIRED_CI gates missing from the "run" commands of ANY job's steps (sorted)."""
    runs = " \n".join(step.get("run", "") for job in workflow.get("jobs", {}).values() for step in job.get("steps", []))
    return sorted(k for k, needle in REQUIRED_CI.items() if needle not in runs)


# ------------------------------------------------------------------------------ S14 regression tests
def fixture_backtest(close: np.ndarray, sma_fn=sma, fast: int = 10, slow: int = 40, cost_bps: float = 5.0) -> dict:
    """The regression fixture: long when sma_fn(fast) > sma_fn(slow) (decided at the close, held the NEXT day),
    else flat; costs on position changes. Return {"total_return" (sum of daily net returns, rounded to 10 decimals),
    "n_trades" (position changes), "exposure" (share of days long)}."""
    r = np.r_[0.0, np.diff(close) / close[:-1]]
    pos = np.nan_to_num((sma_fn(close, fast) > sma_fn(close, slow)).astype(float))
    held = np.r_[0.0, pos[:-1]]
    changes = np.abs(np.diff(np.r_[0.0, held]))
    net = held * r - changes * cost_bps / 1e4
    return {"total_return": round(float(net.sum()), 10), "n_trades": int(changes.sum()), "exposure": float(held.mean())}


def regression_check(result: dict, baseline: dict, rtol: float = 1e-9) -> list[str]:
    """Keys whose value differs from the stored baseline (numbers: beyond rtol relative, 1e-12 absolute; keys
    missing on either side also differ). Sorted. [] = behaviour unchanged."""
    bad = []
    for k in sorted(set(result) | set(baseline)):
        if k not in result or k not in baseline:
            bad.append(k)
        elif not np.isclose(result[k], baseline[k], rtol=rtol, atol=1e-12):
            bad.append(k)
    return bad


# ------------------------------------------------------------------------ S15 reliability & recovery
def apply_fill(state: dict, fill: dict) -> dict:
    """New state after a fill {symbol, qty (signed), price}: positions[symbol] += qty (removed when 0),
    cash −= qty·price, and last_seq = fill["seq"]. Does not modify the input."""
    pos = dict(state["positions"])
    pos[fill["symbol"]] = pos.get(fill["symbol"], 0) + fill["qty"]
    if pos[fill["symbol"]] == 0:
        del pos[fill["symbol"]]
    return {"positions": pos, "cash": state["cash"] - fill["qty"] * fill["price"], "last_seq": fill["seq"]}


def rebuild(events: list[dict], state: dict | None = None) -> dict:
    """Replay fills (skipping seq <= the state's last_seq: idempotent) onto a state (default: no positions, cash 0,
    last_seq −1)."""
    state = state or {"positions": {}, "cash": 0.0, "last_seq": -1}
    for f in events:
        if f["seq"] > state["last_seq"]:
            state = apply_fill(state, f)
    return state


def save_snapshot(state: dict, path: Path) -> None:
    """Write the state atomically: to "<path>.tmp" first, then rename over path (a crash mid-write never leaves a
    half-written snapshot)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    tmp.replace(path)


def recover(snapshot: Path, journal: list[dict], broker_positions: dict[str, float]) -> dict:
    """Startup recovery: load the snapshot (if the file exists, else an empty state), replay the fill journal on
    top (rebuild skips what the snapshot already holds), reconcile with the broker. Return {"state", "breaks" (list
    of {symbol, internal, broker}), "may_trade": no breaks} — trading resumes only after a clean reconciliation."""
    snapshot = Path(snapshot)
    state = json.loads(snapshot.read_text()) if snapshot.exists() else None
    state = rebuild(journal, state)
    breaks = [{"symbol": s, "internal": state["positions"].get(s, 0), "broker": broker_positions.get(s, 0)}
              for s in sorted(set(state["positions"]) | set(broker_positions))
              if state["positions"].get(s, 0) != broker_positions.get(s, 0)]
    return {"state": state, "breaks": breaks, "may_trade": not breaks}


def backup(tables: dict) -> tuple[bytes, str]:
    """Compressed JSON dump (zlib) and its sha256 hex digest (store them apart)."""
    blob = zlib.compress(json.dumps(tables, sort_keys=True, default=str).encode())
    return blob, hashlib.sha256(blob).hexdigest()


def restore(blob: bytes, digest: str) -> dict:
    """Verify the checksum BEFORE restoring (ValueError "corrupt backup" if it differs), then decompress."""
    if hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("corrupt backup")
    return json.loads(zlib.decompress(blob))


# ------------------------------------------------------------------------ S16 security & op-risk
SECRET_RULES = {"anthropic_key": r"sk-ant-[A-Za-z0-9_-]{16,}", "aws_access_key": r"AKIA[0-9A-Z]{16}",
                "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                "hardcoded_password": r"(?i)password\s*[=:]\s*['\"][^'\"$]{4,}['\"]"}


def scan_secrets(files: dict[str, str]) -> list[tuple[str, int, str]]:
    """(file, 1-based line number, rule name) for every SECRET_RULES match, sorted."""
    out = []
    for name, text in files.items():
        for i, line in enumerate(text.splitlines(), start=1):
            for rule, pat in SECRET_RULES.items():
                if re.search(pat, line):
                    out.append((name, i, rule))
    return sorted(out)


def rank_threats(threats: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """threats: threat, likelihood (1–5), impact (1–5), control. Add risk = likelihood × impact; return the `top`
    rows by risk (ties: higher impact first, then threat name)."""
    t = threats.assign(risk=threats["likelihood"] * threats["impact"])
    return t.sort_values(["risk", "impact", "threat"], ascending=[False, False, True]).head(top).reset_index(drop=True)


def precautionary_check(order: dict, last_price: float, limits: dict) -> list[str]:
    """Broker-side precautionary limits, a second line behind the platform's risk engine: "size" (|qty| >
    max_qty), "notional" (|qty|·price > max_notional), "price_band" (limit price more than band_pct away from the
    last price). Returns the failed checks in that order."""
    out = []
    if abs(order["qty"]) > limits["max_qty"]:
        out.append("size")
    if abs(order["qty"]) * order["price"] > limits["max_notional"]:
        out.append("notional")
    if abs(order["price"] / last_price - 1) > limits["band_pct"]:
        out.append("price_band")
    return out

# ======================================================================================
# Weeks 45–48 (S17–S24) go-live, operations, reviews
# ======================================================================================
STAGES = ["paper", "shadow", "small_live", "scaled_live"]
CRITERIA = {"paper": {"min_days": 10}, "shadow": {"min_days": 10}, "small_live": {"min_days": 20},
            "scaled_live": {}}


# ------------------------------------------------------------------------ S17 rollout & ramp
def rollout_decision(stage: str, ev: dict) -> tuple[str, str, list[str]]:
    """ev: days, sev1_incidents, incidents, open_breaks, slippage_ratio (realized / model), inside_band (share of
    days inside the backtest band), drift_alarm (bool).
    DEMOTE one stage (not below paper) if inside_band < 0.5, drift_alarm, or incidents >= 3 → ("demote", new stage,
    reasons). At the top stage: ("hold", stage, []). Else PROMOTE one stage if ALL hold: days >= the stage's min_days, sev1_incidents == 0,
    open_breaks == 0, slippage_ratio <= 1.5, inside_band >= 0.8. Else ("hold", stage, the failed criteria).
    Reason strings: "outside band", "drift alarm", "repeated incidents", "too few days", "sev1 incident",
    "open reconciliation breaks", "slippage above model", "below band"."""
    i = STAGES.index(stage)
    dem = [r for r, bad in (("outside band", ev["inside_band"] < 0.5), ("drift alarm", ev["drift_alarm"]),
                            ("repeated incidents", ev["incidents"] >= 3)) if bad]
    if dem:
        return "demote", STAGES[max(i - 1, 0)], dem
    if i == len(STAGES) - 1:
        return "hold", stage, []                                                       # nothing above
    fails = [r for r, bad in (("too few days", ev["days"] < CRITERIA[stage]["min_days"]),
                              ("sev1 incident", ev["sev1_incidents"] > 0),
                              ("open reconciliation breaks", ev["open_breaks"] > 0),
                              ("slippage above model", ev["slippage_ratio"] > 1.5),
                              ("below band", ev["inside_band"] < 0.8)) if bad]
    if fails:
        return "hold", stage, fails
    return "promote", STAGES[i + 1], []


def capital_ramp(fraction: float, weeks_in_band: int, drawdown: float, plan: dict) -> float:
    """Next week's capital fraction. Drawdown worse than plan["max_dd"] (e.g. −0.05) → halve (not below
    plan["min"]). Else, every full plan["weeks_per_step"] weeks in the band → + plan["step"] (capped at
    plan["max"]), i.e. increase only when weeks_in_band is a positive multiple of weeks_per_step. Else unchanged."""
    if drawdown < plan["max_dd"]:
        return max(fraction / 2, plan["min"])
    if weeks_in_band > 0 and weeks_in_band % plan["weeks_per_step"] == 0:
        return min(fraction + plan["step"], plan["max"])
    return fraction


# --------------------------------------------------------------------------- S18 daily operations
HOLIDAYS = {pd.Timestamp(d) for d in ("2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
                                      "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25")}
EARLY_CLOSE = {pd.Timestamp(d) for d in ("2025-07-03", "2025-11-28", "2025-12-24")}


def session_times(day) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(open 09:30, close 16:00, or 13:00 on EARLY_CLOSE days) for a trading day; None on weekends and HOLIDAYS."""
    d = pd.Timestamp(day).normalize()
    if d.weekday() >= 5 or d in HOLIDAYS:
        return None
    close = pd.Timedelta(hours=13) if d in EARLY_CLOSE else pd.Timedelta(hours=16)
    return d + pd.Timedelta(hours=9, minutes=30), d + close


PREMARKET = {"gateway_connected": lambda s: s["gateway_connected"],
             "data_fresh": lambda s: s["data_age_s"] <= 5,
             "reconciliation_clean": lambda s: s["open_breaks"] == 0,
             "kill_switch_tested": lambda s: s["kill_switch_tested_today"],
             "risk_limits_loaded": lambda s: s["risk_limits_loaded"],
             "backup_ok": lambda s: s["last_backup_age_h"] <= 26,
             "disk_ok": lambda s: s["disk_free_pct"] >= 15}
BLOCKING = {"gateway_connected", "data_fresh", "reconciliation_clean", "kill_switch_tested", "risk_limits_loaded"}


def premarket_check(status: dict) -> dict:
    """Run every PREMARKET check. {"results": {name: bool}, "blocking_failures": sorted failed BLOCKING checks,
    "warnings": sorted failed non-blocking checks, "may_trade": no blocking failure}."""
    res = {k: bool(f(status)) for k, f in PREMARKET.items()}
    failed = {k for k, ok in res.items() if not ok}
    return {"results": res, "blocking_failures": sorted(failed & BLOCKING), "warnings": sorted(failed - BLOCKING),
            "may_trade": not (failed & BLOCKING)}


# --------------------------------------------------------------------------- S19 incidents
PHASES = ["detected", "contained", "diagnosed", "recovered", "reconciled", "closed"]
SLA_MINUTES = {"sev1": {"contained": 5, "reconciled": 60}, "sev2": {"contained": 30, "reconciled": 240},
               "sev3": {"contained": 240, "reconciled": 1440}}


class Incident:
    """Lifecycle detect → contain → diagnose → recover → reconcile → close, one step at a time, with timestamps."""

    def __init__(self, title: str, severity: str, detected_at):
        self.title, self.severity = title, severity
        self.times = {"detected": pd.Timestamp(detected_at)}

    @property
    def phase(self) -> str:
        return PHASES[len(self.times) - 1]

    def advance(self, phase: str, at) -> None:
        """Only the NEXT phase is allowed, not earlier in time than the previous one (ValueError otherwise)."""
        i = PHASES.index(self.phase)
        if i + 1 >= len(PHASES) or PHASES[i + 1] != phase:
            raise ValueError(f"{self.phase} cannot go to {phase}")
        at = pd.Timestamp(at)
        if at < self.times[self.phase]:
            raise ValueError("time goes forward")
        self.times[phase] = at

    def sla_breaches(self) -> list[str]:
        """Phases in SLA_MINUTES[severity] that were reached LATER than their target minutes after detection
        (only phases already reached are checked)."""
        t0 = self.times["detected"]
        return [p for p, m in SLA_MINUTES[self.severity].items()
                if p in self.times and (self.times[p] - t0) > pd.Timedelta(minutes=m)]


def postmortem_timeline(records: list[dict], start, end) -> list[tuple[pd.Timestamp, str]]:
    """From audit records (ts as a datetime string or Timestamp, event): (ts, event) between start and end
    inclusive, sorted by time — the backbone of a blameless post-mortem."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = [(pd.Timestamp(r["ts"]), r["event"]) for r in records if start <= pd.Timestamp(r["ts"]) <= end]
    return sorted(out)


# --------------------------------------------------------------- S20–S24 reviews & graduation
def go_live_review(ev: dict) -> tuple[bool, list[str]]:
    """Approved for small live only on evidence: every strategy's dsr >= 0.95 and pbo < 0.5 ("gate: <name>"), at
    least 10 runbooks ("runbooks"), every drill passed ("drill: <name>"), security checklist complete
    ("security"), a rollout plan ("rollout plan"). ev: strategies {name: {dsr, pbo}}, runbooks (int), drills
    {name: bool}, security_complete (bool), rollout_plan (bool). Returns (approved, sorted missing items)."""
    missing = [f"gate: {n}" for n, s in ev["strategies"].items() if not (s["dsr"] >= 0.95 and s["pbo"] < 0.5)]
    missing += [f"drill: {n}" for n, ok in ev["drills"].items() if not ok]
    if ev["runbooks"] < 10:
        missing.append("runbooks")
    if not ev["security_complete"]:
        missing.append("security")
    if not ev["rollout_plan"]:
        missing.append("rollout plan")
    return not missing, sorted(missing)


def weekly_decisions(inside: pd.DataFrame, incidents: dict[str, int]) -> dict[str, str]:
    """inside: days × strategies, True when the strategy's cumulative return was inside its backtest band.
    Per strategy: "demote" if inside on fewer than half the days or >= 2 incidents; "scale" if inside every day and
    no incident; else "hold"."""
    out = {}
    for s in inside.columns:
        share, inc = inside[s].mean(), incidents.get(s, 0)
        out[s] = "demote" if share < 0.5 or inc >= 2 else "scale" if share == 1.0 and inc == 0 else "hold"
    return out


def orders_without_risk_approval(records: list[dict]) -> list[str]:
    """Graduation check from the audit chain: ids of "order" records (data["id"]) NOT preceded by a "risk_decision"
    record with data["order_id"] == that id and data["approved"] True. [] = no order bypassed the risk engine."""
    approved, bad = set(), []
    for r in records:
        if r["event"] == "risk_decision" and r["data"].get("approved"):
            approved.add(r["data"]["order_id"])
        elif r["event"] == "order" and r["data"]["id"] not in approved:
            bad.append(r["data"]["id"])
    return bad


def graduation(records: list[dict], eod_breaks: dict, explained: set, strategies_dsr: dict) -> tuple[bool, list[str]]:
    """The four mandatory requirements: a "killswitch_trip" in the audit ("kill switch not demonstrated"), no order
    without risk approval ("orders bypassed risk: <ids joined by ,>"), every end-of-day break explained (eod_breaks:
    {date: n}; a date with n > 0 not in `explained` → "unexplained breaks: <dates joined by ,>", dates as given),
    a DSR for every strategy (None → "missing DSR: <names joined by ,>"). Returns (passed, reasons)."""
    reasons = []
    if not any(r["event"] == "killswitch_trip" for r in records):
        reasons.append("kill switch not demonstrated")
    bypass = orders_without_risk_approval(records)
    if bypass:
        reasons.append("orders bypassed risk: " + ",".join(bypass))
    unexplained = [d for d, n in eod_breaks.items() if n > 0 and d not in explained]
    if unexplained:
        reasons.append("unexplained breaks: " + ",".join(unexplained))
    missing = [s for s, v in strategies_dsr.items() if v is None]
    if missing:
        reasons.append("missing DSR: " + ",".join(missing))
    return not reasons, reasons

# ======================================================================================
# Monitoring-drill helpers (labs/part12 clinic W3)
# ======================================================================================
def bad_ticks(ticks: pd.DataFrame, window: int = 21, threshold: float = 8.0) -> list[int]:
    """Timestamps of bad prints: deviation of each price from the rolling median of the `window` ticks CENTRED on it
    (ignoring itself is not needed: the median is robust), then a RobustZ(window=500, threshold) stream over the
    deviations. A single bad print moves its own deviation only, not its neighbours'."""
    dev = ticks["price"] / ticks["price"].rolling(window, center=True, min_periods=1).median() - 1
    rz = RobustZ(window=500, threshold=threshold)
    return [int(t) for t, d in zip(ticks["ts"], dev) if rz.update(float(d))[1]]


def stale_gaps(ticks: pd.DataFrame, max_gap: float = 5.0) -> list[int]:
    """Timestamps of the last tick before a silence longer than max_gap seconds."""
    ts = ticks["ts"].to_numpy()
    return [int(ts[i]) for i in np.flatnonzero(np.diff(ts) > max_gap)]


def stream_flags(values, threshold: float = 6.0) -> list[int]:
    """Positions flagged by a RobustZ(threshold=threshold) stream (latencies, orders per minute)."""
    rz = RobustZ(threshold=threshold)
    return [i for i, v in enumerate(values) if rz.update(float(v))[1]]


def book_fills(fills: list[dict]) -> tuple[dict[str, int], list[str]]:
    """IDEMPOTENT booking: apply each fill_id once. Return (positions without zeros, ids delivered more than
    once)."""
    seen, dups, pos = set(), [], {}
    for f in fills:
        if f["fill_id"] in seen:
            dups.append(f["fill_id"])
            continue
        seen.add(f["fill_id"])
        pos[f["symbol"]] = pos.get(f["symbol"], 0) + f["qty"]
    return {k: v for k, v in pos.items() if v != 0}, dups



