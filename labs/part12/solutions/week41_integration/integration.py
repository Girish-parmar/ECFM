"""Week 41 (S1–S4) — Platform audit and performance: architecture rules checked from the import graph, latency
histograms and percentiles, ring buffers and streaming indicators, event-loop lag, the declarative strategy creator
(YAML → validated condition trees) and composable screeners with stored universes.

Rule of the week: measure before you optimize, and let the machine enforce the architecture.
Fill in every block marked "Your turn", then run:  python -m pytest week41_integration
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ------------------------------------------------------------------------ S1 architecture rules
def module_imports(path: Path) -> set[str]:
    """Every module a Python file imports, ANYWHERE in the file (also inside functions): ast.walk over Import
    (each alias name) and ImportFrom with level 0 (the module, plus module.name for each imported name)."""
    # >>> SOLUTION
    tree = ast.parse(Path(path).read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out
    # <<< SOLUTION


def import_graph(pkg_dir: Path) -> dict[str, set[str]]:
    """For every .py file under the package directory: dotted module name (the package dir's name first, no
    ".__init__") → the set of imported modules that belong to the same package (start with "<package>.")."""
    # >>> SOLUTION
    pkg_dir = Path(pkg_dir)
    root = pkg_dir.name
    graph = {}
    for f in sorted(pkg_dir.rglob("*.py")):
        parts = [root, *f.relative_to(pkg_dir).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        graph[".".join(parts)] = {m for m in module_imports(f) if m == root or m.startswith(root + ".")}
    return graph
    # <<< SOLUTION


def layer_of(module: str, layers: list[str]) -> int | None:
    """Index of the module's layer (its second dotted component) in `layers` (0 = top), None if not a layer."""
    # >>> SOLUTION
    parts = module.split(".")
    return layers.index(parts[1]) if len(parts) > 1 and parts[1] in layers else None
    # <<< SOLUTION


def check_layers(graph: dict[str, set[str]], layers: list[str]) -> list[tuple[str, str]]:
    """Layered architecture: a module may import its own layer or LOWER ones (larger index) only. Return the sorted
    (importer, imported) pairs that import a HIGHER layer."""
    # >>> SOLUTION
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
    # <<< SOLUTION


def check_forbidden(graph: dict[str, set[str]], source: str, forbidden: list[str]) -> list[tuple[str, str]]:
    """Modules equal to or under `source` that import a module equal to or under any `forbidden` one (sorted)."""
    # >>> SOLUTION
    under = lambda m, p: m == p or m.startswith(p + ".")                                  # noqa: E731
    return sorted((mod, imp) for mod, imports in graph.items() if under(mod, source)
                  for imp in imports if any(under(imp, f) for f in forbidden))
    # <<< SOLUTION


# ---------------------------------------------------------------------------- S2 performance
class LatencyHistogram:
    """Prometheus-style histogram: cumulative bucket counts (upper bounds, +inf last), sum and count."""

    def __init__(self, buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1)):
        self.bounds = [*buckets, float("inf")]
        self.counts = [0] * len(self.bounds)
        self.total, self.n = 0.0, 0

    def observe(self, x: float) -> None:
        """Count x in every bucket whose upper bound is >= x (cumulative, like Prometheus)."""
        # >>> SOLUTION
        for i, b in enumerate(self.bounds):
            if x <= b:
                self.counts[i] += 1
        self.total += x
        self.n += 1
        # <<< SOLUTION

    def quantile(self, q: float) -> float:
        """Upper bound of the first bucket whose cumulative count reaches q·n (a conservative estimate)."""
        # >>> SOLUTION
        target = q * self.n
        return next(b for b, c in zip(self.bounds, self.counts) if c >= target)
        # <<< SOLUTION

    @contextmanager
    def time(self):
        """with hist.time(): … observes the elapsed seconds (time.perf_counter)."""
        # >>> SOLUTION
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - t0)
        # <<< SOLUTION


class RingBuffer:
    """Fixed memory for the last `size` values (a preallocated array and a write position): no growing DataFrames
    in the hot path."""

    def __init__(self, size: int):
        # >>> SOLUTION
        self.buf = np.full(size, np.nan)
        self.size, self.pos, self.n = size, 0, 0
        # <<< SOLUTION

    def append(self, x: float) -> None:
        # >>> SOLUTION
        self.buf[self.pos] = x
        self.pos = (self.pos + 1) % self.size
        self.n = min(self.n + 1, self.size)
        # <<< SOLUTION

    def values(self) -> np.ndarray:
        """The stored values, OLDEST first."""
        # >>> SOLUTION
        if self.n < self.size:
            return self.buf[: self.n].copy()
        return np.r_[self.buf[self.pos:], self.buf[: self.pos]]
        # <<< SOLUTION


class StreamingEMA:
    """O(1) per update; must equal pandas ewm(span=n, adjust=False).mean()."""

    def __init__(self, n: int):
        # >>> SOLUTION
        self.alpha, self.value = 2 / (n + 1), None
        # <<< SOLUTION

    def update(self, x: float) -> float:
        # >>> SOLUTION
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return self.value
        # <<< SOLUTION


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
    # >>> SOLUTION
    if isinstance(spec, (int, float)):
        return lambda d: spec
    if isinstance(spec, str):
        return lambda d: d[spec].to_numpy()
    ind = registry["indicators"][spec["ind"]]
    return lambda d: ind(d[spec.get("input", "close")], **spec.get("params", {}))
    # <<< SOLUTION


def build_condition(spec, registry=REGISTRY):
    """Lesson plan S3: {"all": […]}, {"any": […]}, {"not": {…}} and {"fn": name, "args": […]} → function(data) →
    boolean array."""
    # >>> SOLUTION
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
    # <<< SOLUTION


def validate_config(cfg: dict, registry=REGISTRY) -> list[str]:
    """Reject a config BEFORE it runs. Errors (strings): "missing <key>" for each REQUIRED key; walking the entry
    tree: "unknown action <fn>", "unknown indicator <ind>", "unknown parameter <ind>.<p>", "<ind>.<p>=<v> outside
    [lo, hi]". [] = valid."""
    # >>> SOLUTION
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
    # <<< SOLUTION


def config_hash(cfg: dict) -> str:
    """First 12 hex characters of sha256(json.dumps(cfg, sort_keys=True)): stored with every trade."""
    # >>> SOLUTION
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    # <<< SOLUTION


def load_strategy(text: str, registry=REGISTRY) -> tuple[dict, Callable, str]:
    """yaml.safe_load the config, validate it (ValueError listing every error if invalid), and return (config,
    entry-signal function, config hash)."""
    # >>> SOLUTION
    cfg = yaml.safe_load(text)
    errors = validate_config(cfg, registry)
    if errors:
        raise ValueError("; ".join(errors))
    return cfg, build_condition(cfg["entry"], registry), config_hash(cfg)
    # <<< SOLUTION


# ---------------------------------------------------------------------------------- S4 screeners
def min_adv(usd: float):
    """Filter factory: average daily dollar volume >= usd. Each filter maps a snapshot to a boolean Series."""
    # >>> SOLUTION
    return lambda df: df["adv_usd"] >= usd
    # <<< SOLUTION


def price_between(lo: float, hi: float):
    # >>> SOLUTION
    return lambda df: df["price"].between(lo, hi)
    # <<< SOLUTION


def max_spread(bps: float):
    # >>> SOLUTION
    return lambda df: df["spread_bps"] <= bps
    # <<< SOLUTION


def sector_in(sectors):
    # >>> SOLUTION
    return lambda df: df["sector"].isin(set(sectors))
    # <<< SOLUTION


def min_option_oi(oi: int):
    # >>> SOLUTION
    return lambda df: df["option_oi"] >= oi
    # <<< SOLUTION


def shortable():
    # >>> SOLUTION
    return lambda df: df["shortable"].astype(bool)
    # <<< SOLUTION


def screen(df: pd.DataFrame, filters: list, limit: int | None = None) -> list[str]:
    """Symbols passing ALL filters, sorted by adv_usd descending (then symbol), at most `limit`."""
    # >>> SOLUTION
    mask = np.logical_and.reduce([f(df) for f in filters]) if filters else np.ones(len(df), bool)
    out = df[mask].sort_values(["adv_usd", "symbol"], ascending=[False, True])["symbol"].tolist()
    return out if limit is None else out[:limit]
    # <<< SOLUTION


def universe_turnover(old: list[str], new: list[str]) -> float:
    """Share of the new universe that was NOT in the old one (names you must trade into); 0 if new is empty."""
    # >>> SOLUTION
    return len(set(new) - set(old)) / len(new) if new else 0.0
    # <<< SOLUTION


class UniverseStore:
    """Stored screen results, so a live universe can be reproduced: (date, screen name) → config hash + symbols."""

    def __init__(self):
        self.rows: dict[tuple[pd.Timestamp, str], dict] = {}

    def save(self, date, name: str, cfg: dict, symbols: list[str]) -> None:
        # >>> SOLUTION
        self.rows[(pd.Timestamp(date), name)] = {"config_hash": config_hash(cfg), "symbols": list(symbols)}
        # <<< SOLUTION

    def get(self, date, name: str) -> dict | None:
        """The latest saved result ON OR BEFORE date for that screen (point-in-time), else None."""
        # >>> SOLUTION
        keys = [k for k in self.rows if k[1] == name and k[0] <= pd.Timestamp(date)]
        return self.rows[max(keys)] if keys else None
        # <<< SOLUTION
