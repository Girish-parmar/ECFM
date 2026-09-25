# Part 3 — Python Engineering for Trading: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 1, **Month 2 second half + Month 3** (program weeks 7–12) |
| Format | 24 sessions × 90 min (4 per week) + 6 lab clinics × 120 min + self-study (~7 hrs/week) |
| Total effort | ~36 hrs live + 12 hrs clinic + 42 hrs self-study ≈ **90 hours** |
| Environment | Python 3.12+, `uv`, JupyterLab, VS Code, Git/GitHub, Docker, PostgreSQL + TimescaleDB, DuckDB, Redis |
| Platform milestone | **M0 Foundations** (end of week 12): `quantforge` repository skeleton, `core/` (config, logging, errors, event bus, clock), `domain/` (instruments, market data, orders, fills, positions), `stats/` and `analytics/metrics.py` promoted from Part 2, data layer schemas, CI |
| Covers original items | "Stock Market Basic" 3 (basic Python), 4 (advanced Python), 5 (object-oriented programming), 6 (low-level design principles) · roadmap 3.1–3.7 · Platform roadmap items 1–2 (foundations) |

**Why this Part matters:** from Part 4 onward, every lesson adds code to one growing trading platform. Code that is untyped, untested or tangled becomes impossible to trust once real orders depend on it. This Part teaches Python as an engineer uses it for trading systems: exact money arithmetic, time zones, asynchronous I/O for broker connections, clean object models, design patterns, a proper data layer, and tests that catch mistakes before the market does.

---

## 1. Learning Objectives

By the end of Part 3 the learner will be able to:

1. **Write idiomatic Python**: types, control flow, functions, collections, comprehensions, files, exceptions, modules, with correct handling of **money (Decimal)** and **time (time zones, daylight saving)**.
2. **Use advanced features** where they help: generators for data streams, decorators (timing, retry), context managers, type hints with `mypy`, dataclasses and Pydantic validation.
3. **Write concurrent code** with `asyncio` (tasks, queues, timeouts, cancellation, back-pressure), and choose correctly between threads, processes and async.
4. **Measure and improve performance**: profiling, vectorization, `numba`, memory (`__slots__`, generators).
5. **Design object models** for trading: value objects vs entities, inheritance vs composition, abstract base classes and protocols, error hierarchies.
6. **Apply low-level design**: SOLID principles and the patterns used throughout the platform (Strategy, Template Method, Factory/Registry, Adapter, Repository, Observer, Command, State, Chain of Responsibility, Specification, Builder), documented with UML.
7. **Handle market data** with NumPy, pandas and Polars (time-zone-aware indexing, resampling, rolling windows, as-of joins), and store it in PostgreSQL/TimescaleDB, Parquet + DuckDB and Redis.
8. **Work like a professional team**: Git and pull requests, project layout, linting, testing (unit, property-based, golden files), structured logging, configuration, Docker and CI.
9. **Deliver platform milestone M0** that every later Part builds on.

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| Part 1 (order types, instruments, corporate actions, futures multipliers, time zones) | W3 domain model, W5 data layer |
| Part 2 guided notebooks (NumPy/pandas basics already seen) and the stats functions written there | W1–W2 (familiar syntax), W5–W6 (promoting code into packages) |
| Course environment installed; GitHub account | S1 |

No prior programming experience is assumed for weeks 7–8; learners with experience can take the "fast track" (extra exercises on generators, async and typing) during those weeks.

---

## 3. Six-Week Overview

| Week | Theme | Sessions | Clinic Lab | Output |
|---|---|---|---|---|
| **W1** (7) | Python basics (3.1) | S1 Setup, types, numbers & money · S2 Control flow, functions, comprehensions · S3 Collections, strings, dates & time zones · S4 Files, exceptions, modules | Trade-log parser → P&L report | `labs/part03/trade_log/` |
| **W2** (8) | Advanced Python (3.2) | S5 Iterators, generators, decorators, context managers · S6 Type hints, dataclasses, enums, Pydantic · S7 Concurrency & `asyncio` · S8 Performance & memory | Async quote simulator with back-pressure | Part 2 graded notebook due (end of Month 2) |
| **W3** (9) | Object-oriented programming (3.3) | S9 Classes, properties, dunder methods · S10 Inheritance, composition, ABCs & protocols · S11 Domain modelling for trading · S12 Errors, testing classes | Domain model with property-based tests | `domain/` draft |
| **W4** (10) | Low-level design (3.4) | S13 SOLID with trading examples · S14 Patterns I · S15 Patterns II · S16 UML & architecture | UML diagrams + pattern kata | `docs/uml/`, `core/events.py` |
| **W5** (11) | Scientific stack & data (3.5, 3.6) | S17 NumPy · S18 pandas & Polars for time series · S19 SQL, PostgreSQL & TimescaleDB · S20 Parquet, DuckDB, Redis & data validation | Data pipeline: Part 2 data → Parquet/DuckDB + Timescale | `data/` schemas and loaders |
| **W6** (12) | Software craft & M0 (3.7) | S21 Git, project layout, tooling · S22 Testing in depth · S23 Logging, config, CLI, Docker & CI · S24 M0 integration & review | M0 code review | **M0 release** |

---

## 4. Session-by-Session Plan

> Each session: **15 min recap/theory → 45 min live coding → 20 min guided lab → 10 min wrap-up and homework.**
> Exercises live in `labs/part03/`; from week 9, code goes into the `quantforge` repository through pull requests.

### Week 1 — Python Basics (3.1, item 3)

#### S1 · Setup, Types, Numbers & Money

| Block | Content |
|---|---|
| Theory | Installing Python with `uv`; virtual environments and why every project gets its own; the REPL, scripts and notebooks. Variables, core types (`int`, `float`, `bool`, `str`, `None`), dynamic typing. **Floats are binary approximations** (`0.1 + 0.2 != 0.3`), which is fine for analytics (returns, statistics) and **wrong for money, prices on a tick grid and quantities**; use `decimal.Decimal` (constructed from strings) for those. Rounding modes; rounding a price to an instrument's tick size. |
| Live coding | `uv init`, `uv add`, running scripts; float vs Decimal demonstrations; `round_to_tick` (below). |
| Lab | Commission calculator for IB's fixed schedule (from Part 8's formula) using Decimal; compare float and Decimal totals over 10,000 trades. |
| Homework | 20 short exercises (types, arithmetic, string formatting with f-strings). |

```python
from decimal import Decimal, ROUND_HALF_UP

assert 0.1 + 0.2 != 0.3                          # binary floating point
assert Decimal("0.1") + Decimal("0.2") == Decimal("0.3")

def round_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    """Exact decimal rounding to the instrument's tick size (never use float for this)."""
    return (price / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick

round_to_tick(Decimal("101.237"), Decimal("0.05"))  # Decimal('101.25')
```

#### S2 · Control Flow, Functions & Comprehensions

| Block | Content |
|---|---|
| Theory | `if/elif/else`, `for`, `while`, `break/continue`, `match` (structural pattern matching, handy for broker message types). Functions: parameters, defaults (and the mutable-default trap), `*args/**kwargs`, keyword-only arguments, return values, docstrings, pure functions vs side effects. Scope and closures (preview of decorators). List/dict/set comprehensions and when a plain loop is clearer. |
| Live coding | Position-sizing and P&L functions; a `match` statement that routes order-update messages by type. |
| Lab | Functions for simple and log returns, cumulative return, max drawdown on a Python list (then compare with the NumPy versions from Part 2). |
| Homework | Refactor a 60-line script into 5 small, tested functions. |

#### S3 · Collections, Strings, Dates & Time Zones

| Block | Content |
|---|---|
| Theory | `list`, `tuple`, `dict`, `set`: behaviour and time complexity (lookups in a dict/set are O(1), in a list O(n)); `collections` (`deque` for rolling windows, `defaultdict`, `Counter`, `namedtuple`). Strings, f-strings, parsing. **Dates and times for trading:** `datetime`, `zoneinfo`; always **time-zone-aware**; store in **UTC**, display in exchange time; daylight saving changes the UTC time of the US open (14:30 UTC in winter, 13:30 UTC in summer); exchange calendars for holidays and half-days (`exchange_calendars`). |
| Live coding | `market_open_utc` (below); a rolling window with `deque(maxlen=n)`; parsing broker timestamps. |
| Lab | Convert a mixed-time-zone trade log (ET, UTC, naive) into clean UTC; flag trades outside regular hours using an exchange calendar. |
| Homework | Build a dictionary-based order book (price → size) supporting add, cancel and best bid/ask. |

```python
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

NY, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")

def market_open_utc(d: date) -> datetime:
    """09:30 New York time on date d, in UTC (the offset changes with daylight saving)."""
    return datetime.combine(d, time(9, 30), tzinfo=NY).astimezone(UTC)

market_open_utc(date(2025, 1, 15)).hour   # 14
market_open_utc(date(2025, 7, 15)).hour   # 13
```

#### S4 · Files, Exceptions & Modules

| Block | Content |
|---|---|
| Theory | Reading/writing text, CSV and JSON; `pathlib`; context managers for files. **Exceptions:** `try/except/else/finally`, catching specific exceptions (never a bare `except:`), raising with messages, custom exception classes, "fail loudly" for trading code. Modules and packages, imports, `if __name__ == "__main__"`, the standard library tour (`csv`, `json`, `statistics`, `itertools`, `functools`). |
| Live coding | Trade-log parser with validation errors collected per line. |
| Lab | **Trade-log parser → P&L report** (clinic W1): read fills from CSV, validate, compute realized P&L per symbol and day, write a JSON report. |
| Homework | Add a command-line interface with `argparse` to the parser. |

**Clinic W1 (120 min):** the trade-log parser, including malformed-line handling, time-zone normalization and Decimal arithmetic.

---

### Week 2 — Advanced Python (3.2, item 4)

#### S5 · Iterators, Generators, Decorators & Context Managers

| Block | Content |
|---|---|
| Theory | The iterator protocol; **generators** for streaming data (a tick file too large for memory, a live feed); `yield from`; `itertools` (`islice`, `groupby`, `accumulate`). **Closures** and **decorators**: timing, logging, caching (`functools.lru_cache`), **retry with exponential back-off and jitter** (reused for broker reconnects in Part 4). **Context managers** (`with`, `contextlib.contextmanager`) for connections, files and temporary state: resources are released even when an exception occurs. |
| Live coding | Tick-file generator, `@timed`, async `@retry`, a `session` context manager (below). |
| Lab | Stream a 5-million-row tick file and build 1-minute bars with a generator pipeline; compare memory use with loading the whole file. |
| Homework | A decorator that rate-limits calls to at most N per second. |

```python
import asyncio, csv, functools, logging, random, time
from contextlib import contextmanager

log = logging.getLogger(__name__)

def read_ticks(path: str):
    """Generator: stream a large tick file row by row without loading it into memory."""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield row["ts"], float(row["price"]), int(row["size"])

def retry(times=5, base=0.5, cap=30.0, exceptions=(ConnectionError, TimeoutError)):
    """Decorator: retry an async function with exponential back-off and jitter."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    delay = min(cap, base * 2 ** attempt) * random.uniform(0.5, 1.0)
                    log.warning("%s failed (%s); retry %d in %.2fs", fn.__name__, e, attempt, delay)
                    await asyncio.sleep(delay)
        return wrapper
    return deco

@contextmanager
def session(name: str):
    """Resources are always released, even if the body raises."""
    log.info("open %s", name)
    try:
        yield name
    finally:
        log.info("close %s", name)
```

#### S6 · Type Hints, Dataclasses, Enums & Pydantic

| Block | Content |
|---|---|
| Theory | Type hints (`list[int]`, `dict[str, Decimal]`, `X | None`, `Callable`, generics), what they do (documentation, IDE help, static checking) and do not do (no runtime checks). **`mypy --strict`** in CI. **`dataclasses`**: `frozen=True` for immutable value objects, `slots=True` for memory, default factories, `__post_init__` validation. **`Enum`** for closed sets (sides, order types, asset classes). **Pydantic** for validating external data (broker JSON, config files): parse, validate, coerce, and fail with clear messages. |
| Live coding | Order request as a frozen dataclass; broker order-update message parsed with Pydantic; `mypy` catching a `Decimal` vs `float` bug. |
| Lab | Type the Week 1 trade-log parser fully and get it to pass `mypy --strict`. |
| Homework | Pydantic models for 3 real broker message types (from IB or Alpaca documentation samples). |

#### S7 · Concurrency & `asyncio`

| Block | Content |
|---|---|
| Theory | CPU-bound vs I/O-bound work. **Threads** (I/O, limited by the GIL for CPU work; note the optional free-threaded builds of Python 3.13+), **processes** (`concurrent.futures.ProcessPoolExecutor` for CPU-heavy research such as parameter sweeps), **`asyncio`** (one thread, many concurrent I/O tasks: the model used by `ib_async`, `alpaca-py` streams and the platform). Event loop, coroutines, `await`, tasks, `gather`, **queues** (bounded queues give back-pressure), **timeouts** (`asyncio.wait_for`), **cancellation**, and the golden rule: **never block the event loop** (no `time.sleep`, no heavy computation inside a coroutine). |
| Live coding | Producer/consumer quote pipeline with a bounded queue and a timeout (below); showing what a blocking call does to latency. |
| Lab | **Async quote simulator** (clinic W2): 3 simulated feeds at different rates → one consumer computing a mid-price; measure latency; add a slow consumer and observe back-pressure. |
| Homework | Graceful shutdown: cancel all tasks cleanly on Ctrl-C and log what was in flight. |

```python
async def producer(q: asyncio.Queue, n: int):
    for i in range(n):
        await q.put(("SPY", 500 + i * 0.01))
        await asyncio.sleep(0)             # yield control, like waiting on a socket
    await q.put(None)                      # sentinel: no more data

async def consumer(q: asyncio.Queue, out: list):
    while (item := await q.get()) is not None:
        out.append(item)

async def run_pipeline(n=1000, timeout=5.0):
    q: asyncio.Queue = asyncio.Queue(maxsize=100)    # bounded: back-pressure if the consumer is slow
    out: list = []
    await asyncio.wait_for(asyncio.gather(producer(q, n), consumer(q, out)), timeout)
    return out

# asyncio.run(run_pipeline())
```

#### S8 · Performance & Memory

| Block | Content |
|---|---|
| Theory | "Measure first": `timeit`, `cProfile` + `snakeviz`, `line_profiler`, `tracemalloc`/`memray`. Big-O intuition with trading examples (searching a list of orders vs a dict by order ID). **Vectorization** with NumPy; **`numba`** for loops that cannot be vectorized (recursive filters like EMA, path-dependent logic); pandas/Polars built-ins. **Memory:** `__slots__`, generators, choosing dtypes (`float32` vs `float64`), avoiding copies. When *not* to optimize. |
| Live coding | EMA in pure Python vs `numba` vs `pandas.ewm` on 1 million points; `__slots__` memory comparison. |
| Lab | Profile the Week 1 parser on a 10-million-line file; fix the top hotspot. |
| Homework | Make the Part 2 `metrics()` function 10× faster on a 5,000-asset return matrix. |

> Measured in the course environment: an EMA over 1,000,000 points took about 150–220 ms in pure Python and about 5.5 ms with `numba` (roughly 30–40× faster); 100,000 small objects used about 10.4 MB without `__slots__` and 6.4 MB with it.

**Clinic W2 (120 min):** async quote simulator; also office hours for finishing the Part 2 graded notebook (due at the end of this week).

---

### Week 3 — Object-Oriented Programming (3.3, item 5)

#### S9 · Classes, Properties & Dunder Methods

| Block | Content |
|---|---|
| Theory | Classes and instances, attributes and methods, `self`, class vs instance attributes, `@property` for derived values, `@classmethod` constructors (e.g. `Instrument.from_broker(...)`), `@staticmethod`. **Dunder methods:** `__repr__` (for logs), `__eq__` and `__hash__` (value equality; objects as dict keys), `__lt__` for sorting, `__len__`, `__iter__`, `__enter__/__exit__`. Encapsulation by convention (`_private`). |
| Live coding | A `Money` value class with currency-checked arithmetic and a clear `__repr__`. |
| Lab | `OrderBook` class from the S3 homework with `__len__`, `__iter__` (price levels) and best bid/ask properties. |
| Homework | Add `__add__` and `__sub__` to `Money` that refuse to combine different currencies. |

#### S10 · Inheritance, Composition, ABCs & Protocols

| Block | Content |
|---|---|
| Theory | Inheritance for "is-a" with shared behaviour; **composition** for "has-a" (usually better); the fragile-base-class problem. **Abstract base classes** (`abc.ABC`, `@abstractmethod`) for required interfaces (e.g. `BrokerAdapter`, `Strategy`). **`typing.Protocol`** for structural typing (anything with the right methods fits; great for test fakes). Mixins with care. Method resolution order. |
| Live coding | `BrokerAdapter` ABC with a `FakeBroker` for tests; a `PriceSource` Protocol satisfied by both a live feed and a replay feed. |
| Lab | Refactor an inheritance-heavy "strategy zoo" into composition (strategy + sizer + risk rules as separate objects). |
| Homework | Write a `Protocol` for the Part 2 volatility estimators and make 3 implementations conform. |

#### S11 · Domain Modelling for Trading

| Block | Content |
|---|---|
| Theory | **Value objects** (immutable, equal by content: `Instrument`, `Bar`, `Tick`, `Fill`) vs **entities** (identity and state over time: `Order`, `Position`, `Account`). Modelling instruments across asset classes (tick size, multiplier, currency, expiry). Quantities and prices as `Decimal`. **Position accounting:** FIFO lots vs average cost; realized vs unrealized P&L; futures multipliers; fees. The invariant every accounting implementation must satisfy: **realized + unrealized P&L = cash P&L**, for any sequence of fills. |
| Live coding | The domain module (below) and its tests. |
| Lab | **Domain model with property-based tests** (clinic W3): use `hypothesis` to generate random fill sequences and check the P&L invariant. |
| Homework | Add a `Bar` value object with validation (`low ≤ open, close ≤ high`, UTC timestamp required). |

```python
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

class AssetClass(str, Enum):
    EQUITY = "EQUITY"; FUTURE = "FUTURE"; OPTION = "OPTION"; FX = "FX"; CRYPTO = "CRYPTO"

@dataclass(frozen=True, slots=True)
class Instrument:
    """Value object: equal and hashable by content, immutable, safe as a dict key."""
    symbol: str
    asset_class: AssetClass
    currency: str = "USD"
    tick: Decimal = Decimal("0.01")
    multiplier: Decimal = Decimal("1")

    def __post_init__(self):
        if self.tick <= 0 or self.multiplier <= 0:
            raise ValueError("tick and multiplier must be positive")

@dataclass(frozen=True, slots=True)
class Future(Instrument):
    expiry: str = ""                                  # e.g. "202612"

@dataclass(frozen=True, slots=True)
class Fill:
    instrument: Instrument
    qty: Decimal                                      # + buy, - sell
    price: Decimal
    fee: Decimal = Decimal("0")

@dataclass(slots=True)
class Position:
    """Entity. FIFO lot accounting: closes match the oldest open lots first. Only exact Decimal
    multiplication and addition are used, so realized + unrealized P&L always reconciles with
    cash to the last digit (average-cost accounting needs a division)."""
    instrument: Instrument
    lots: deque = field(default_factory=deque)       # open lots: (signed qty, price), oldest first
    realized: Decimal = Decimal("0")

    @property
    def qty(self) -> Decimal:
        return sum((q for q, _ in self.lots), Decimal("0"))

    @property
    def avg_price(self) -> Decimal:                  # for display only
        q = self.qty
        return sum((lq * lp for lq, lp in self.lots), Decimal("0")) / q if q else Decimal("0")

    def apply(self, fill: Fill) -> None:
        if fill.instrument != self.instrument:
            raise ValueError("fill for a different instrument")
        m = self.instrument.multiplier
        self.realized -= fill.fee
        q = fill.qty
        while q != 0 and self.lots and (self.lots[0][0] > 0) != (q > 0):   # fill reduces the position
            lot_q, lot_p = self.lots[0]
            close = min(abs(q), abs(lot_q))
            s = 1 if lot_q > 0 else -1
            self.realized += (fill.price - lot_p) * close * s * m
            lot_q -= s * close
            q += s * close
            if lot_q == 0:
                self.lots.popleft()
            else:
                self.lots[0] = (lot_q, lot_p)
        if q != 0:                                    # add, open, or the flipped remainder
            self.lots.append((q, fill.price))

    def unrealized(self, mark: Decimal) -> Decimal:
        return sum(((mark - p) * q for q, p in self.lots), Decimal("0")) * self.instrument.multiplier
```

```python
# tests/domain/test_position.py
from decimal import Decimal as D
from hypothesis import given, settings, strategies as st

SPY = Instrument("SPY", AssetClass.EQUITY)
ES = Future("ES", AssetClass.FUTURE, tick=D("0.25"), multiplier=D("50"), expiry="202612")

def test_fifo_reduce_and_flip():
    p = Position(SPY)
    p.apply(Fill(SPY, D("100"), D("10"))); p.apply(Fill(SPY, D("100"), D("12")))
    p.apply(Fill(SPY, D("-150"), D("13")))            # closes 100 @ 10 (+300) and 50 @ 12 (+50)
    assert p.qty == 50 and p.realized == D("350")
    p.apply(Fill(SPY, D("-80"), D("9")))              # closes 50 @ 12 at 9 (-150), opens short 30 @ 9
    assert p.qty == -30 and p.avg_price == 9 and p.realized == D("200")

def test_futures_multiplier():
    f = Position(ES)
    f.apply(Fill(ES, D("2"), D("6000"))); f.apply(Fill(ES, D("-2"), D("6010.25")))
    assert f.realized == D("1025") and f.qty == 0     # 10.25 points × 2 contracts × $50

@settings(max_examples=500)
@given(st.lists(st.tuples(st.integers(-50, 50).filter(lambda x: x != 0), st.integers(90, 110)),
                min_size=1, max_size=40))
def test_pnl_reconciles_with_cash(trades):
    """Invariant: realized + unrealized P&L == cash P&L, for ANY sequence of fills."""
    p, cash = Position(SPY), D("0")
    for q, px in trades:
        p.apply(Fill(SPY, D(q), D(px)))
        cash -= D(q) * D(px)
    mark = D("100")
    assert p.realized + p.unrealized(mark) == cash + p.qty * mark
```

> Verified: all tests pass, including 500 random fill sequences. **Teaching moment found while preparing this lesson:** a first version using average-cost accounting failed the property test after only two fills (buy 1 @ 90, buy 2 @ 91 gives an average of 90.666…, which `Decimal` must round), and a cost-basis version still failed on partial closes. Exact reconciliation needs an accounting method without division, such as FIFO lots. Use this story in class: the property test found a bug that hand-written examples missed.

#### S12 · Error Handling & Testing Classes

| Block | Content |
|---|---|
| Theory | An **exception hierarchy** for the platform (`TradingError` → `BrokerError`, `RiskRejected`, `DataStale`, `ConfigError`…) so callers can catch precisely; exceptions vs return values; never swallow errors in trading code; adding context (`raise ... from e`). **Testing classes** with `pytest`: fixtures, parametrization, fakes vs mocks, testing invariants with `hypothesis` (continued from S11). |
| Live coding | `core/errors.py`; tests for `Instrument` validation and `Position` edge cases (fees, zero crossing). |
| Lab | Reach 95% coverage on `domain/` with meaningful tests (not just line coverage: every rule has a test that would fail if the rule were removed). |
| Homework | Add a `Money`-aware `Account` entity (cash by currency, positions) with tests. |

```python
class TradingError(Exception): """Base class for all platform errors."""
class ConfigError(TradingError): """Invalid or missing configuration."""
class BrokerError(TradingError): """Broker rejected or failed a request."""
class RiskRejected(TradingError): """Risk engine rejected an order."""
class DataStale(TradingError): """Market data older than allowed."""
```

**Clinic W3 (120 min):** domain model review; property-based tests for P&L, instrument validation and bar validation.

---

### Week 4 — Low-Level Design (3.4, item 6)

#### S13 · SOLID with Trading Examples

| Block | Content |
|---|---|
| Theory | **S**ingle responsibility (a strategy decides; it does not also place orders and log to a database), **O**pen/closed (new indicators and brokers without editing core code: registries), **L**iskov substitution (every `BrokerAdapter` must behave the same for the same call: contract tests in Part 4), **I**nterface segregation (small interfaces: `PriceSource`, `OrderSink`), **D**ependency inversion (strategies depend on abstractions, not on IB or Alpaca). Also DRY, KISS, YAGNI, and "make illegal states unrepresentable". |
| Live coding | Refactor a 300-line "god script" (data + signals + orders + logging) into SOLID components. |
| Lab | Code-review exercise: find SOLID violations in 5 short snippets and fix them. |
| Homework | Write the interfaces (ABCs/Protocols) for data feed, order sink, clock and logger used by M0. |

#### S14 · Design Patterns I

| Block | Content |
|---|---|
| Theory | **Strategy** (swap algorithms: sizing methods, fill models), **Template Method** (the `Strategy` base class lifecycle: `on_start`, `on_bar`, `on_fill`), **Factory & Registry** (create instruments and strategies by name from config; the Part 5 indicator registry), **Adapter** (IB and Alpaca behind one `BrokerAdapter`), **Repository** (hide whether bars come from PostgreSQL, Parquet or a broker). |
| Live coding | Registry decorator; a tiny `Strategy` base class with Template Method; `FakeBroker` adapter. |
| Lab | Pattern kata: implement a "fill model" family (next open, VWAP, limit-through) using the Strategy pattern. |
| Homework | Repository interface for bars with an in-memory implementation and tests. |

#### S15 · Design Patterns II

| Block | Content |
|---|---|
| Theory | **Observer / event bus** (market data and order updates fan out to many listeners), **Command** (orders as replayable, auditable objects), **State** (order lifecycle as a transition table; illegal transitions fail loudly), **Chain of Responsibility** (risk checks), **Specification** (composable entry rules: `&`, `|`, `~`), **Builder** (multi-leg option orders). Anti-patterns: Singleton for everything, deep inheritance trees, "manager" classes that do everything. |
| Live coding | Event bus and order state machine (below), both unit-tested. |
| Lab | Wire the event bus: a simulated feed publishes ticks; a bar builder, a logger and a stats collector subscribe. |
| Homework | Chain-of-Responsibility risk checks (max notional, max position) with tests (fully built in Part 8). |

```python
import asyncio
from collections import defaultdict
from typing import Callable

class EventBus:
    """Observer / publish-subscribe: publishers do not know who listens."""
    def __init__(self):
        self._subs: dict[type, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable) -> None:
        self._subs[event_type].append(handler)

    async def publish(self, event) -> None:
        for h in self._subs[type(event)]:
            res = h(event)
            if asyncio.iscoroutine(res):          # supports sync and async handlers
                await res

class OrderStateError(Exception): pass

class OrderStateMachine:
    """State pattern as a transition table: illegal transitions fail loudly."""
    TRANSITIONS = {
        "PENDING_NEW": {"ACCEPTED", "REJECTED"},
        "ACCEPTED": {"PARTIALLY_FILLED", "FILLED", "PENDING_CANCEL", "EXPIRED"},
        "PARTIALLY_FILLED": {"PARTIALLY_FILLED", "FILLED", "PENDING_CANCEL"},
        "PENDING_CANCEL": {"CANCELLED", "FILLED"},
        "FILLED": set(), "CANCELLED": set(), "REJECTED": set(), "EXPIRED": set(),
    }

    def __init__(self):
        self.state = "PENDING_NEW"

    def to(self, new: str) -> None:
        if new not in self.TRANSITIONS[self.state]:
            raise OrderStateError(f"{self.state} -> {new} not allowed")
        self.state = new
```

#### S16 · UML & Architecture

| Block | Content |
|---|---|
| Theory | **UML that is actually useful:** class diagrams (domain model), sequence diagrams (signal → risk → order → fill), state diagrams (order lifecycle), written as **Mermaid** in the repository so they are versioned and reviewed. **Architecture:** layers (core → domain → data → library → strategy → execution → apps), clean/hexagonal architecture (business logic in the middle, brokers and databases as adapters at the edge), dependency injection (pass collaborators in; do not create them inside). Walkthrough of [03_PLATFORM_LLD_ROADMAP.md](../03_PLATFORM_LLD_ROADMAP.md). |
| Live coding | Mermaid class, sequence and state diagrams for M0. |
| Lab | **UML + pattern kata** (clinic W4): each learner diagrams M0 and presents the design choices. |
| Homework | Architecture Decision Records (ADRs) for 3 choices (Decimal for prices, FIFO accounting, asyncio). |

**Clinic W4 (120 min):** UML review and design critique in pairs.

---

### Week 5 — Scientific Stack & Data (3.5, 3.6)

#### S17 · NumPy for Market Data

| Block | Content |
|---|---|
| Theory | Arrays, dtypes, shapes, indexing and slicing, views vs copies, **broadcasting** (e.g. demeaning a T × N return matrix), vectorized math, boolean masks, `NaN` handling (`np.nan*` functions), random numbers with `default_rng` (reproducible seeds), linear algebra (`solve`, `eigh`: Part 2 reused). |
| Live coding | Vectorized returns, rolling windows with `sliding_window_view`, covariance of 500 assets. |
| Lab | Rewrite 5 loop-based Part 2 functions with NumPy; verify identical outputs and measure speed. |
| Homework | Implement a vectorized drawdown-duration function. |

#### S18 · pandas & Polars for Time Series

| Block | Content |
|---|---|
| Theory | **pandas:** `DatetimeIndex` (time-zone-aware), selection, `resample` (with explicit `label`/`closed` so bar timestamps mean what you think), `rolling`, `groupby`, `pivot`, `merge`, **`merge_asof`** (as-of joins such as "last quote before each trade", essential for point-in-time data). Common pitfalls: chained assignment, silent index alignment, time-zone-naive data. **Polars:** expressions, lazy frames (`scan_parquet`), speed on large data; when to choose which. Plotting with matplotlib and Plotly. |
| Live coding | 5-minute bars → hourly with correct labels; trades joined to the last prior quote with `merge_asof`; the same pipeline in Polars. |
| Lab | Build daily, weekly and monthly return tables for the 10 Part 2 tickers in both pandas and Polars; compare results and run time. |
| Homework | Detect and report gaps in 1-minute data using an exchange calendar. |

```python
bars = bars.tz_convert("UTC")                       # bars: 5-min, index = bar OPEN time (UTC)
hourly = bars.resample("1h", label="left", closed="left").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})

# last quote at or before each trade (no look-ahead)
joined = pd.merge_asof(trades.sort_values("ts"), quotes.sort_values("ts"), on="ts", direction="backward")
```

#### S19 · SQL, PostgreSQL & TimescaleDB

| Block | Content |
|---|---|
| Theory | Relational modelling for trading: instruments, bars, ticks, orders, fills, positions, the audit log. SQL essentials: `SELECT`, `WHERE`, `JOIN`, `GROUP BY`, window functions (running P&L, `LAG` for returns), indexes, transactions and constraints (a fill must reference an existing order). **TimescaleDB:** hypertables for time-series tables, compression, continuous aggregates (1-minute → 1-hour bars). Python access with `psycopg` (parameterized queries only: never build SQL with string formatting). Migrations (Alembic) as versioned schema changes. |
| Live coding | Schema (below) in the course's PostgreSQL/Timescale container; inserting and querying bars and fills from Python. |
| Lab | Load 2 years of 5-minute bars for the 10 tickers; build an hourly continuous aggregate; compute daily realized P&L from a fills table with SQL window functions. |
| Homework | Query plan exercise: add the right index to make a slow bar query fast (`EXPLAIN ANALYZE`). |

```sql
CREATE TABLE instruments (
  instrument_id serial PRIMARY KEY,
  symbol        text NOT NULL,
  asset_class   text NOT NULL CHECK (asset_class IN ('EQUITY','FUTURE','OPTION','FX','CRYPTO')),
  currency      char(3) NOT NULL DEFAULT 'USD',
  tick          numeric NOT NULL CHECK (tick > 0),
  multiplier    numeric NOT NULL DEFAULT 1 CHECK (multiplier > 0),
  UNIQUE (symbol, asset_class)
);

CREATE TABLE bars (
  instrument_id int NOT NULL REFERENCES instruments,
  timeframe     text NOT NULL,                -- '1m', '5m', '1d'
  ts            timestamptz NOT NULL,         -- bar OPEN time, UTC
  open numeric, high numeric, low numeric, close numeric, volume numeric,
  source        text NOT NULL,                -- 'ib', 'alpaca_iex', ...
  PRIMARY KEY (instrument_id, timeframe, ts),
  CHECK (low <= LEAST(open, close) AND high >= GREATEST(open, close))
);
SELECT create_hypertable('bars', 'ts');       -- TimescaleDB

CREATE TABLE fills (
  fill_id         text PRIMARY KEY,           -- broker execution id (deduplicates replays)
  client_order_id text NOT NULL,
  instrument_id   int NOT NULL REFERENCES instruments,
  ts              timestamptz NOT NULL,
  qty             numeric NOT NULL,           -- + buy, - sell
  price           numeric NOT NULL,
  fee             numeric NOT NULL DEFAULT 0
);
```

#### S20 · Parquet, DuckDB, Redis & Data Validation

| Block | Content |
|---|---|
| Theory | **Research storage:** Parquet (columnar, compressed, typed), partitioned by symbol and date; **DuckDB** to query Parquet files with SQL at high speed without a server; Polars lazy scans. **Live state:** Redis for last prices, positions snapshot, kill-switch flag (fast, in memory, shared between processes). **Data validation:** schemas (Pydantic, `pandera`), checks for monotonic timestamps, OHLC consistency, duplicates, gaps. Which store for which job: PostgreSQL/Timescale (system of record: orders, fills, audit), Parquet + DuckDB (research), Redis (live, ephemeral). |
| Live coding | Write partitioned Parquet with Polars; query it with DuckDB (below); Redis set/get with expiry for a "last price" cache. |
| Lab | **Data pipeline** (clinic W5): Part 2 data → validated → partitioned Parquet + Timescale; one query per store type. |
| Homework | Validation report generator for any bar dataset. |

```python
import duckdb, polars as pl

pl.from_pandas(bars_df).write_parquet("data/bars", partition_by=["symbol", "date"])   # hive-style folders
duckdb.sql("""
    SELECT symbol, count(*) AS n, max(close) AS high_close
    FROM read_parquet('data/bars/**/*.parquet', hive_partitioning = true)
    GROUP BY symbol
""").fetchall()

sma = (pl.scan_parquet("data/bars/**/*.parquet", hive_partitioning=True)       # lazy: nothing read yet
         .with_columns(pl.col("close").rolling_mean(window_size=12).over("symbol").alias("sma12"))
         .collect())
```

> Verified: the pandas resample, `merge_asof`, partitioned Parquet write, DuckDB query and Polars lazy scan above run correctly in the course environment (tested with synthetic bars). The SQL schema is written for PostgreSQL + TimescaleDB and is run in the lab's database container; it was not executed while preparing this document.

**Clinic W5 (120 min):** data pipeline build and review.

---

### Week 6 — Software Craft & M0 (3.7)

#### S21 · Git, Project Layout & Tooling

| Block | Content |
|---|---|
| Theory | **Git:** commits that do one thing, branches, pull requests, code review etiquette, resolving conflicts, `.gitignore` (never commit data, secrets or notebooks' outputs by accident), tags for releases. **Project layout:** `src/` layout, `pyproject.toml`, `uv.lock` for reproducible installs. **Tooling:** `ruff` (lint + format), `mypy --strict`, `pre-commit` hooks (including a secret scanner), `nbstripout` for notebooks. |
| Live coding | Create the `quantforge` repository with the layout below; pre-commit running on every commit. |
| Lab | Each learner opens a pull request adding their Week 3 domain module; peers review with a checklist. |
| Homework | Address review comments; merge. |

```toml
# pyproject.toml (excerpt)
[project]
name = "quantforge"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy", "pandas", "polars", "pyarrow", "duckdb", "pydantic", "pydantic-settings",
                "psycopg[binary]", "redis", "numba", "scipy", "statsmodels", "arch", "exchange-calendars"]

[dependency-groups]
dev = ["pytest", "pytest-asyncio", "pytest-cov", "hypothesis", "mypy", "ruff", "pre-commit"]

[tool.ruff]
line-length = 110

[tool.mypy]
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["paper: tests that need a broker paper account (never run in CI)"]
```

```text
quantforge/
├── pyproject.toml, uv.lock, .pre-commit-config.yaml, Dockerfile
├── src/quantforge/
│   ├── core/        config.py, logging.py, errors.py, events.py, clock.py
│   ├── domain/      instrument.py, market.py, order.py, position.py
│   ├── data/        schemas.py, store.py (Repository), validate.py
│   ├── stats/       from Part 2: linalg, distributions, inference, timeseries, volatility, regimes
│   └── analytics/   metrics.py (from Part 2)
├── tests/           mirrors src/, plus tests/property/ and tests/golden/
├── docs/uml/        Mermaid diagrams, ADRs
└── notebooks/       research only (outputs stripped)
```

#### S22 · Testing in Depth

| Block | Content |
|---|---|
| Theory | The test pyramid for trading code: many fast unit tests, contract tests for interfaces (Part 4 brokers), a few end-to-end tests. `pytest` fixtures and parametrization; **fakes over mocks** for brokers and clocks; **property-based testing** with `hypothesis` (invariants such as P&L reconciliation, length preservation of indicators); **golden-file tests** (compare against stored reference outputs, e.g. TA-Lib values in Part 5); async tests with `pytest-asyncio`; coverage as a guide, not a goal; tests marked `paper` never run in CI. |
| Live coding | Test suite for `core/` and `domain/`; a golden-file test for a Part 2 metric. |
| Lab | Mutation-testing exercise: change one line in the domain code and check that at least one test fails (tools such as `mutmut`). |
| Homework | Raise coverage of `core/` and `domain/` to ≥ 90% with meaningful tests. |

#### S23 · Logging, Configuration, CLI, Docker & CI

| Block | Content |
|---|---|
| Theory | **Structured logging:** JSON lines with a **correlation ID** carried through a signal → order → fill chain (`contextvars` works across `asyncio` tasks); log levels; never log secrets. **Configuration:** `pydantic-settings` (environment variables and `.env`, typed, validated, secrets as `SecretStr`), separate profiles for paper and live. **CLI** with `typer` or `argparse`. **Docker:** a slim image for the platform; `docker compose` with PostgreSQL/Timescale and Redis for local development. **CI:** GitHub Actions running ruff, mypy and pytest on every pull request (extended in Part 12). |
| Live coding | `core/logging.py` and `core/config.py` (below); Dockerfile; CI workflow. |
| Lab | Break the build on purpose (type error, failing test, lint error) and fix it through the PR workflow. |
| Homework | Prepare the M0 pull request. |

```python
import contextvars, json, logging, uuid
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")

class JsonFormatter(logging.Formatter):
    """One JSON object per line; every record carries the current correlation id."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage(), "cid": correlation_id.get()}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def new_correlation_id() -> str:
    cid = uuid.uuid4().hex[:12]
    correlation_id.set(cid)
    return cid

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QF_", env_file=".env")
    env: str = "paper"                              # paper | live | backtest
    ib_port: int = 4002
    alpaca_key: SecretStr = SecretStr("")          # never printed in logs or repr
    max_order_notional: float = 50_000.0
```

#### S24 · M0 Integration & Code Review

| Block | Content |
|---|---|
| Theory | M0 acceptance checklist (Section 7). How Part 4 will build on M0 (broker adapters, data handler, OMS). |
| Activity | M0 pull-request reviews in small groups; fixes; release tag `v0.1.0`. |
| Homework | Read the Part 4 prerequisites; open IB Gateway and Alpaca paper API keys. |

**Clinic W6 (120 min):** M0 code review with instructors; architecture Q&A.

---

## 5. Lab & Code Map

| Artifact | Session(s) | Location | Used again in |
|---|---|---|---|
| Trade-log parser | S1–S4 | `labs/part03/trade_log/` | Part 8 (fills reconciliation) |
| Tick generator, `retry`, context managers | S5 | `core/utils.py` | Part 4 (streaming, reconnects) |
| Pydantic broker messages | S6 | `labs/part03/messages/` | Part 4 (adapters) |
| Async pipeline & quote simulator | S7 | `labs/part03/async_sim/` | Part 4 (live data), Part 12 (event loop health) |
| Error hierarchy | S12 | `core/errors.py` | All Parts |
| Domain model & property tests | S11–S12 | `domain/`, `tests/property/` | All Parts |
| Event bus, order state machine | S15 | `core/events.py`, `domain/order.py` | Parts 4, 8, 12 |
| UML diagrams & ADRs | S16 | `docs/uml/` | Part 12 (architecture audit) |
| Data schemas & loaders | S19–S20 | `data/`, `sql/` | Parts 4, 5, 8 |
| Logging & configuration | S23 | `core/logging.py`, `core/config.py` | All Parts |
| Part 2 statistics modules | S24 | `stats/`, `analytics/metrics.py` | Parts 5, 8, 9, 10 |

---

## 6. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Floats for prices, quantities and money | Off-by-a-cent errors, prices off the tick grid | `Decimal` from strings; `round_to_tick`; tests |
| 2 | Naive datetimes | Bars misaligned twice a year (daylight saving) | Time-zone-aware everywhere; UTC storage; lint rule |
| 3 | Mutable default arguments | State leaking between calls | `ruff` rule; `None` defaults; `field(default_factory=...)` |
| 4 | Bare `except:` or swallowed errors | Silent failures in live trading | Specific exceptions; error hierarchy; re-raise with context |
| 5 | Blocking calls inside `async` code | Frozen feeds, late orders | No `time.sleep`/heavy CPU in coroutines; loop-lag checks |
| 6 | Unbounded queues | Memory growth when consumers are slow | Bounded `asyncio.Queue` (back-pressure) |
| 7 | Optimizing without profiling | Wasted effort, unreadable code | Profile first; benchmark before/after |
| 8 | Deep inheritance for strategies | Rigid, fragile code | Composition; small interfaces |
| 9 | Mutable value objects | Instruments changing inside dict keys | `frozen=True` dataclasses |
| 10 | Average-cost accounting tested only by examples | P&L that does not reconcile | Property-based invariant test; FIFO lots |
| 11 | pandas chained assignment and silent index alignment | Wrong values, no error | `.loc`, explicit joins, copy-on-write mode |
| 12 | SQL built with string formatting | SQL injection, quoting bugs | Parameterized queries only |
| 13 | Committing secrets, data or notebook outputs | Leaked keys, huge repositories | `.gitignore`, secret scanner, `nbstripout` |
| 14 | Tests that depend on the network or the clock | Flaky CI | Fakes for brokers and clocks; `paper` marker |

---

## 7. Assessment — Platform Milestone M0

**A. Programming exercises (weeks 7–8), 20%.** Weekly auto-graded exercise sets (basics, collections, dates, files, generators, decorators, typing, asyncio).

**B. M0 pull request (end of week 12), 60%.**
1. Repository layout, `pyproject.toml` + `uv.lock`, pre-commit, Dockerfile, CI green.
2. `core/`: typed config (`pydantic-settings`), JSON logging with correlation IDs, error hierarchy, event bus, clock abstraction (live and simulated).
3. `domain/`: instruments (equity, future, option, FX, crypto), market data value objects (`Bar`, `Tick`, `Quote`) with validation, order model with the state machine, fills, FIFO positions.
4. `data/`: SQL schema (PostgreSQL/Timescale), Parquet layout, a Repository with in-memory and Parquet implementations, validation.
5. `stats/` and `analytics/metrics.py` promoted from Part 2 with their tests.
6. UML (class, sequence, state) and 3 ADRs.
7. Tests: ≥ 90% coverage on `core/` and `domain/`, including property-based tests; `mypy --strict` clean.

**C. Design review & quiz, 20%.** Oral walkthrough of design choices (SOLID, patterns, trade-offs) + quiz.

| Criterion | Points |
|---|---|
| Programming exercises | 20 |
| Code correctness (money, time, accounting, validation) | 20 |
| Design quality (SOLID, patterns used appropriately, clear interfaces) | 15 |
| Tests (coverage, property-based invariants, no flaky tests) | 15 |
| Tooling & reproducibility (typing, lint, CI, Docker, lockfile) | 10 |
| Documentation (UML, ADRs, docstrings) | 5 |
| Design review & quiz | 15 |
| **Total** | **100** |

Pass mark: 70, **and** (mandatory) the P&L reconciliation property test must pass, and CI (ruff, `mypy --strict`, pytest) must be green on the M0 pull request.

---

## 8. Further Reading

| Type | Reference |
|---|---|
| Book | Ramalho, L. (2022). *Fluent Python* (2nd ed.). O'Reilly. |
| Book | Slatkin, B. (2019). *Effective Python* (2nd ed.). Addison-Wesley. |
| Book | Hattingh, C. (2020). *Using Asyncio in Python*. O'Reilly. |
| Book | Percival, H. & Gregory, B. (2020). *Architecture Patterns with Python*. O'Reilly (free online). |
| Book | Gamma, E., Helm, R., Johnson, R. & Vlissides, J. (1994). *Design Patterns*. Addison-Wesley. |
| Book | Martin, R. C. (2017). *Clean Architecture*. Prentice Hall. |
| Book | McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly (free online). |
| Book | Gorelick, M. & Ozsvald, I. (2020). *High Performance Python* (2nd ed.). O'Reilly. |
| Book | Okken, B. (2022). *Python Testing with pytest* (2nd ed.). Pragmatic Bookshelf. |
| Book | Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly. |
| Docs | Python documentation (`decimal`, `zoneinfo`, `asyncio`, `dataclasses`, `typing`), `uv`, `ruff`, `mypy`, `pytest`, `hypothesis`, Pydantic, NumPy, pandas, Polars, DuckDB, PostgreSQL, TimescaleDB, Redis, Docker, GitHub Actions |

---

## 9. Instructor Notes

- Weeks 7–8 run at two speeds: a core track for beginners and a fast track (extra async, typing and performance exercises) for experienced programmers. Everyone meets again for OOP in week 9.
- Tell the average-cost vs FIFO story from S11: a property test caught a real bug that example-based tests missed. It sells property-based testing better than any slide.
- Enforce the pull-request workflow from week 9: no direct commits to `main`, every change reviewed by a peer.
- Keep the Part 2 graded notebook deadline at the end of week 8, and schedule office hours in clinic W2 for it.
- Nothing connects to a broker yet. Part 4 starts from M0, so the M0 review must be strict about interfaces (`BrokerAdapter`, `PriceSource`, clock) that Part 4 will implement.
