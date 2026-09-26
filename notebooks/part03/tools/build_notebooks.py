"""Build the Part 3 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part03:  python tools/build_notebooks.py
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
for d in (Path.cwd(), Path.cwd().parent):       # p3lib.py is in notebooks/part03/
    sys.path.insert(0, str(d))
from decimal import Decimal
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p3lib as p

p.use_course_style()"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 3 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_03_PYTHON_ENGINEERING.md) · graded labs in [`labs/part03/`](../../labs/part03/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
NB["01_money_and_time"] = [
    header("01", "Money, ticks and time zones", "S1 (Types, numbers & money) · S3 (Collections, dates & time zones)",
           "1. See why floats are wrong for money and prices, and use `Decimal` instead.\n"
           "2. Round prices to an instrument's tick and compute a commission exactly.\n"
           "3. Convert the New York open to UTC and watch daylight saving move it.\n"
           "4. Keep a rolling window with `deque`, and feel why dict lookups beat list scans."),
    ("code", SETUP),
    ("md", "## 1. Floats are binary approximations"),
    ("code", """print(0.1 + 0.2, 0.1 + 0.2 == 0.3)
print(Decimal("0.1") + Decimal("0.2"), Decimal("0.1") + Decimal("0.2") == Decimal("0.3"))
print(Decimal(0.1))                       # built from a float: the float's error comes along
total_f, total_d = 0.0, Decimal("0")
for _ in range(1_000_000):                # a million one-cent fees
    total_f += 0.01
    total_d += Decimal("0.01")
print(f"float total:   {total_f!r}")
print(f"Decimal total: {total_d!r}")"""),
    ("md", "## 2. Rounding to the tick\n\nExchanges only accept prices on the instrument's tick grid. Round **in Decimal**, half up."),
    ("md", YOUR_TURN),
    ("ex", """from decimal import ROUND_HALF_UP

def round_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    # ✍️ divide by the tick, quantize to a whole number with ROUND_HALF_UP, multiply back
    return ...

tests = [(Decimal("101.237"), Decimal("0.05")), (Decimal("4512.30"), Decimal("0.25")), (Decimal("0.123456"), Decimal("0.0001"))]
mine = [round_to_tick(px, tk) for px, tk in tests]
mine = p.check("round_to_tick", mine, [p.round_to_tick(px, tk) for px, tk in tests])
mine""",
     """from decimal import ROUND_HALF_UP

def round_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    return (price / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick

tests = [(Decimal("101.237"), Decimal("0.05")), (Decimal("4512.30"), Decimal("0.25")), (Decimal("0.123456"), Decimal("0.0001"))]
mine = [round_to_tick(px, tk) for px, tk in tests]
mine = p.check("round_to_tick", mine, [p.round_to_tick(px, tk) for px, tk in tests])
mine"""),
    ("code", """# The float version looks fine... until it doesn't
def round_to_tick_float(price, tick):
    return round(price / tick) * tick

bad = [(x, round_to_tick_float(x, 0.05)) for x in (101.237, 1.005, 0.35)]
bad                                          # values like 0.35000000000000003 are OFF the tick grid"""),
    ("md", "## 3. Commission in Decimal\n\nIB fixed pricing (lesson-plan figures): **$0.005 per share, minimum $1.00, maximum 1% of trade value**, rounded to the cent."),
    ("md", YOUR_TURN),
    ("ex", """def commission(shares: int, price: Decimal) -> Decimal:
    raw = Decimal("0.005") * shares
    cap = Decimal("0.01") * shares * price
    # ✍️ apply the $1.00 minimum and the 1% cap, then quantize to Decimal("0.01") with ROUND_HALF_UP
    return ...

trades = [(100, Decimal("50")), (1000, Decimal("0.5")), (500, Decimal("20")), (10_000, Decimal("3.1"))]
fees = [commission(n, px) for n, px in trades]
fees = p.check("IB fixed commission", fees, [p.ib_fixed_commission(n, px) for n, px in trades])
fees""",
     """def commission(shares: int, price: Decimal) -> Decimal:
    raw = Decimal("0.005") * shares
    cap = Decimal("0.01") * shares * price
    return min(max(raw, Decimal("1.00")), cap).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

trades = [(100, Decimal("50")), (1000, Decimal("0.5")), (500, Decimal("20")), (10_000, Decimal("3.1"))]
fees = [commission(n, px) for n, px in trades]
fees = p.check("IB fixed commission", fees, [p.ib_fixed_commission(n, px) for n, px in trades])
fees"""),
    ("code", """rng = np.random.default_rng(0)
sizes, prices_ = rng.integers(1, 5000, 10_000), rng.uniform(1, 500, 10_000).round(2)
dec = [p.ib_fixed_commission(int(n), Decimal(str(px))) for n, px in zip(sizes, prices_)]
flt = [round(min(max(0.005 * n, 1.0), 0.01 * n * px), 2) for n, px in zip(sizes, prices_)]
differ = sum(Decimal(str(f)) != d for f, d in zip(flt, dec))
print(f"Decimal total: ${sum(dec):,}   float total: ${sum(flt):,.10f}")
print(f"The float version disagrees with exact half-up rounding on {differ:,} of 10,000 trades: round() on a float "
      "rounds a binary approximation, half to even (1.005 → 1.0). Pennies, multiplied by every trade.")"""),
    ("md", "## 4. Time zones: store in UTC, think in exchange time\n\nThe US open is 09:30 in New York, which is **14:30 UTC in winter and 13:30 UTC in summer**."),
    ("md", YOUR_TURN),
    ("ex", """from datetime import date, datetime, time
from zoneinfo import ZoneInfo
NY, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")

def market_open_utc(d: date) -> datetime:
    # ✍️ combine d with 09:30, attach tzinfo=NY, convert to UTC
    return ...

days = [date(2025, 1, 15), date(2025, 7, 15), date(2025, 3, 10)]
opens = [market_open_utc(d) for d in days]
opens = p.check("market open in UTC", opens, [p.market_open_utc(d) for d in days])
opens""",
     """from datetime import date, datetime, time
from zoneinfo import ZoneInfo
NY, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")

def market_open_utc(d: date) -> datetime:
    return datetime.combine(d, time(9, 30), tzinfo=NY).astimezone(UTC)

days = [date(2025, 1, 15), date(2025, 7, 15), date(2025, 3, 10)]
opens = [market_open_utc(d) for d in days]
opens = p.check("market open in UTC", opens, [p.market_open_utc(d) for d in days])
opens"""),
    ("code", """year = pd.bdate_range("2025-01-01", "2025-12-31")
hours = [p.market_open_utc(d.date()).hour + p.market_open_utc(d.date()).minute / 60 for d in year]
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.step(year, hours, where="post")
ax.set(title="The US open in UTC through 2025", ylabel="UTC hour", yticks=[13.5, 14.5], yticklabels=["13:30", "14:30"])
plt.show()
naive = datetime(2025, 3, 10, 9, 30)
print("A naive datetime has no zone:", naive.tzinfo, "- comparing it with an aware one raises TypeError.")"""),
    ("md", "## 5. Collections: a rolling window and O(1) lookups"),
    ("md", YOUR_TURN),
    ("ex", """from collections import deque

def rolling_mean(values, n):
    win, out = deque(maxlen=n), []
    for v in values:
        win.append(v)
        # ✍️ append the mean of the window once it holds n values, else None
        ...
    return out

vals = [10, 11, 12, 13, 14, 15]
rm = rolling_mean(vals, 3)
rm = p.check("rolling mean with deque", rm, p.rolling_mean_deque(vals, 3))
rm""",
     """from collections import deque

def rolling_mean(values, n):
    win, out = deque(maxlen=n), []
    for v in values:
        win.append(v)
        out.append(sum(win) / n if len(win) == n else None)
    return out

vals = [10, 11, 12, 13, 14, 15]
rm = rolling_mean(vals, 3)
rm = p.check("rolling mean with deque", rm, p.rolling_mean_deque(vals, 3))
rm"""),
    ("code", """sizes_ = [1_000, 10_000, 100_000]
rows = []
for n in sizes_:
    symbols = [f"S{i}" for i in range(n)]
    as_list, as_set = symbols, set(symbols)
    rows.append({"n": n, "list scan (µs)": p.timeit(lambda: "missing" in as_list, number=20) * 1e6,
                 "set lookup (µs)": p.timeit(lambda: "missing" in as_set, number=20) * 1e6})
pd.DataFrame(rows).set_index("n").round(2)"""),
    ("md", """## Questions
1. Where in a trading system is a float fine, and where must you use `Decimal`?
2. Why is `Decimal(0.1)` different from `Decimal("0.1")`?
3. A strategy stores bar times as naive local times. What goes wrong twice a year?

**Graded version:** `labs/part03/week07_basics` (`round_to_tick`, the commission, New York/UTC times)."""),
]

# ---------------------------------------------------------------------------------------------- 02
NB["02_functions_and_files"] = [
    header("02", "Functions, control flow and files", "S2 (Control flow, functions & comprehensions) · S4 (Files, exceptions & modules) · Clinic W1",
           "1. Write small, pure functions for returns and drawdown, and compare them with NumPy.\n"
           "2. Route broker messages with `match`.\n3. Avoid the mutable-default trap.\n"
           "4. Parse a messy trade log, collecting an error per bad line instead of crashing."),
    ("code", SETUP),
    ("md", "## 1. Returns and drawdown on plain lists"),
    ("md", YOUR_TURN),
    ("ex", """prices = [100.0, 102.0, 99.0, 104.0, 101.0, 107.0]
# ✍️ simple returns with a list comprehension over consecutive pairs (hint: zip(prices, prices[1:]))
rets = ...
rets = p.check("simple returns", rets, p.simple_returns(prices))
rets""",
     """prices = [100.0, 102.0, 99.0, 104.0, 101.0, 107.0]
rets = [b / a - 1 for a, b in zip(prices, prices[1:])]
rets = p.check("simple returns", rets, p.simple_returns(prices))
rets"""),
    ("ex", """def max_drawdown(prices):
    peak, mdd = prices[0], 0.0
    for px in prices:
        # ✍️ update the running peak and the worst drawdown so far (px / peak − 1)
        ...
    return mdd

mdd = p.check("max drawdown", max_drawdown(prices), p.max_drawdown(prices))
mdd""",
     """def max_drawdown(prices):
    peak, mdd = prices[0], 0.0
    for px in prices:
        peak = max(peak, px)
        mdd = min(mdd, px / peak - 1)
    return mdd

mdd = p.check("max drawdown", max_drawdown(prices), p.max_drawdown(prices))
mdd"""),
    ("code", """big = list(p.prices_array(200_000))
arr = np.asarray(big)
t_py = p.timeit(p.max_drawdown, big, repeat=3)
t_np = p.timeit(lambda x: p.drawdown_vectorized(x).min(), arr, repeat=3)
print(f"pure Python {t_py * 1e3:.1f} ms   NumPy {t_np * 1e3:.2f} ms   ({t_py / t_np:.0f}× faster)")
print("Same answer:", np.isclose(p.max_drawdown(big), p.drawdown_vectorized(arr).min()))"""),
    ("md", "## 2. Routing broker messages with `match`"),
    ("md", YOUR_TURN),
    ("ex", """def route(msg: dict) -> str:
    match msg:
        case {"type": "fill", "qty": q} if q > 0:
            return "book_fill"
        case {"type": "fill"}:
            return "reject_bad_fill"
        # ✍️ add: "cancel" or "reject" → "release_order"; "heartbeat" → "ignore"; anything else → "log_unknown"
        case _:
            return ...

msgs = [{"type": "fill", "qty": 100}, {"type": "fill", "qty": 0}, {"type": "cancel", "id": 7},
        {"type": "reject", "id": 8}, {"type": "heartbeat"}, {"type": "news"}]
routed = [route(m) for m in msgs]
routed = p.check("message routing", routed, [p.route(m) for m in msgs])
routed""",
     """def route(msg: dict) -> str:
    match msg:
        case {"type": "fill", "qty": q} if q > 0:
            return "book_fill"
        case {"type": "fill"}:
            return "reject_bad_fill"
        case {"type": "cancel" | "reject"}:
            return "release_order"
        case {"type": "heartbeat"}:
            return "ignore"
        case _:
            return "log_unknown"

msgs = [{"type": "fill", "qty": 100}, {"type": "fill", "qty": 0}, {"type": "cancel", "id": 7},
        {"type": "reject", "id": 8}, {"type": "heartbeat"}, {"type": "news"}]
routed = [route(m) for m in msgs]
routed = p.check("message routing", routed, [p.route(m) for m in msgs])
routed"""),
    ("md", "## 3. The mutable-default trap"),
    ("code", """def add_fill(fill, book=[]):               # ❌ the list is created ONCE, when the function is defined
    book.append(fill)
    return book

print(add_fill("A"), add_fill("B"))           # the second call still sees "A"

def add_fill_ok(fill, book=None):             # ✅
    book = [] if book is None else book
    book.append(fill)
    return book

print(add_fill_ok("A"), add_fill_ok("B"))"""),
    ("md", "## 4. Parsing a messy trade log (clinic W1)\n\nFail loudly per line, but keep going: collect `(line number, message)` for every bad line."),
    ("code", "print(p.TRADE_CSV)"),
    ("md", YOUR_TURN),
    ("ex", """import csv, io
from decimal import InvalidOperation

def parse_trades(text):
    rows, errors = [], []
    for i, r in enumerate(csv.DictReader(io.StringIO(text)), start=2):    # line 1 is the header
        try:
            # ✍️ raise ValueError(f"bad side {r['side']!r}") unless side is BUY or SELL
            ...
            qty = int(r["qty"])
            if not r["price"]:
                raise ValueError("missing price")
            price = Decimal(r["price"])
            rows.append({"time": datetime.fromisoformat(r["time"]), "symbol": r["symbol"], "side": r["side"],
                         "qty": qty, "price": price})
        except (ValueError, InvalidOperation) as e:
            # ✍️ record (i, str(e)) in errors
            ...
    return rows, errors

from datetime import datetime
parsed = parse_trades(p.TRADE_CSV)
rows, errors = p.check("trade-log parser", parsed, p.parse_trades(p.TRADE_CSV))
errors""",
     """import csv, io
from decimal import InvalidOperation

def parse_trades(text):
    rows, errors = [], []
    for i, r in enumerate(csv.DictReader(io.StringIO(text)), start=2):    # line 1 is the header
        try:
            if r["side"] not in ("BUY", "SELL"):
                raise ValueError(f"bad side {r['side']!r}")
            qty = int(r["qty"])
            if not r["price"]:
                raise ValueError("missing price")
            price = Decimal(r["price"])
            rows.append({"time": datetime.fromisoformat(r["time"]), "symbol": r["symbol"], "side": r["side"],
                         "qty": qty, "price": price})
        except (ValueError, InvalidOperation) as e:
            errors.append((i, str(e)))
    return rows, errors

from datetime import datetime
parsed = parse_trades(p.TRADE_CSV)
rows, errors = p.check("trade-log parser", parsed, p.parse_trades(p.TRADE_CSV))
errors"""),
    ("code", """import json
report = {"valid_rows": len(rows), "errors": errors,
          "realized_pnl": {s: str(v) for s, v in p.realized_pnl(rows).items()}}      # Decimal → str keeps it exact
print(json.dumps(report, indent=2))"""),
    ("md", """## Questions
1. Why is a bare `except:` dangerous in trading code? What should happen to an unknown message type?
2. Which function in this notebook has a side effect, and how would you test it?
3. The report stores P&L as strings. Why not floats?

**Graded version:** `labs/part03/week07_basics` and the clinic CLI in `labs/part03/clinic_w1_trade_log`."""),
]

# ---------------------------------------------------------------------------------------------- 03
NB["03_generators_decorators"] = [
    header("03", "Generators, decorators and context managers", "S5 (Iterators, generators, decorators & context managers)",
           "1. Stream ticks with a generator and turn them into bars without holding the day in memory.\n"
           "2. Write a decorator that counts calls, keeping the function's name.\n"
           "3. Time a block of code with your own context manager, and retry a flaky call."),
    ("code", SETUP),
    ("md", "## 1. Generators: one tick at a time"),
    ("code", """import sys
ticks_list = list(p.tick_stream(50_000))
print(f"list of 50,000 ticks: {sys.getsizeof(ticks_list) / 1e6:.1f} MB for the list alone (plus every tuple)")
gen = p.tick_stream(50_000)
print(f"the generator:        {sys.getsizeof(gen)} bytes, whatever the length")
print(next(gen), next(gen), sep="\\n")"""),
    ("md", YOUR_TURN),
    ("ex", """def bars_from_ticks(ticks, minutes=1):
    cur = None
    for t, px, sz in ticks:
        start = t.floor(f"{minutes}min")
        if cur is not None and start != cur["start"]:
            yield cur                          # the previous bar is complete
            cur = None
        if cur is None:
            cur = {"start": start, "open": px, "high": px, "low": px, "close": px, "volume": 0}
        # ✍️ update high, low, close and volume with this tick
        ...
    if cur is not None:
        yield cur

mine = pd.DataFrame(bars_from_ticks(p.tick_stream(3000), minutes=5))
bars = p.check("bars from a tick stream", mine, pd.DataFrame(p.bars_from_ticks(p.tick_stream(3000), minutes=5)))
bars.head()""",
     """def bars_from_ticks(ticks, minutes=1):
    cur = None
    for t, px, sz in ticks:
        start = t.floor(f"{minutes}min")
        if cur is not None and start != cur["start"]:
            yield cur                          # the previous bar is complete
            cur = None
        if cur is None:
            cur = {"start": start, "open": px, "high": px, "low": px, "close": px, "volume": 0}
        cur["high"], cur["low"] = max(cur["high"], px), min(cur["low"], px)
        cur["close"] = px
        cur["volume"] += sz
    if cur is not None:
        yield cur

mine = pd.DataFrame(bars_from_ticks(p.tick_stream(3000), minutes=5))
bars = p.check("bars from a tick stream", mine, pd.DataFrame(p.bars_from_ticks(p.tick_stream(3000), minutes=5)))
bars.head()"""),
    ("code", """ax = bars.set_index("start")["close"].plot(title="5-minute closes built from a tick stream")
ax.set_xlabel(""); plt.show()"""),
    ("md", "## 2. Decorators"),
    ("md", YOUR_TURN),
    ("ex", """import functools

def count_calls(fn):
    @functools.wraps(fn)                  # keeps fn's name and docstring on the wrapper
    def wrapper(*args, **kwargs):
        # ✍️ increase wrapper.calls by one, then call fn and return its result
        return ...
    wrapper.calls = 0
    return wrapper

@count_calls
def fetch_quote(symbol):
    \"\"\"Pretend broker call.\"\"\"
    return {"symbol": symbol, "bid": 99.99, "ask": 100.01}

for s in ("SPY", "QQQ", "IWM"):
    fetch_quote(s)
last = fetch_quote("TLT")
result = (fetch_quote.calls, fetch_quote.__name__, last)
result = p.check("count_calls decorator", result, (4, "fetch_quote", {"symbol": "TLT", "bid": 99.99, "ask": 100.01}))
result""",
     """import functools

def count_calls(fn):
    @functools.wraps(fn)                  # keeps fn's name and docstring on the wrapper
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return fn(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@count_calls
def fetch_quote(symbol):
    \"\"\"Pretend broker call.\"\"\"
    return {"symbol": symbol, "bid": 99.99, "ask": 100.01}

for s in ("SPY", "QQQ", "IWM"):
    fetch_quote(s)
last = fetch_quote("TLT")
result = (fetch_quote.calls, fetch_quote.__name__, last)
result = p.check("count_calls decorator", result, (4, "fetch_quote", {"symbol": "TLT", "bid": 99.99, "ask": 100.01}))
result"""),
    ("md", "## 3. Context managers"),
    ("md", YOUR_TURN),
    ("ex", """import time
from contextlib import contextmanager

@contextmanager
def timer(results: dict, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        # ✍️ store the elapsed seconds in results[name] (runs even if the block raises)
        ...

timings = {}
with timer(timings, "sleep"):
    time.sleep(0.05)
ok = p.check("timer context manager", 0.04 < timings.get("sleep", 0) < 0.5, True)""",
     """import time
from contextlib import contextmanager

@contextmanager
def timer(results: dict, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        results[name] = time.perf_counter() - t0

timings = {}
with timer(timings, "sleep"):
    time.sleep(0.05)
ok = p.check("timer context manager", 0.04 < timings.get("sleep", 0) < 0.5, True)"""),
    ("code", """def retry(times=3, delay=0.01, exceptions=(ConnectionError,)):
    \"\"\"Retry on the given exceptions with a doubling delay; re-raise after the last try.\"\"\"
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            wait = delay
            for attempt in range(1, times + 1):
                try:
                    return fn(*a, **k)
                except exceptions as e:
                    if attempt == times:
                        raise
                    print(f"  attempt {attempt} failed ({e}); retrying in {wait:.2f}s")
                    time.sleep(wait)
                    wait *= 2
        return wrapper
    return deco

attempts = iter([ConnectionError("gateway down"), ConnectionError("timeout"), "connected"])

@retry(times=3)
def connect():
    x = next(attempts)
    if isinstance(x, Exception):
        raise x
    return x

print(connect())"""),
    ("md", """## Questions
1. Why is a generator a natural fit for a live data feed?
2. What would `fetch_quote.__name__` be without `functools.wraps`, and why does it matter in logs?
3. Which exceptions should a broker retry catch, and which must never be retried (hint: an order rejection)?

**Graded version:** `labs/part03/week08_advanced` (tick generator → bars, `count_calls`, async `retry`)."""),
]

# ---------------------------------------------------------------------------------------------- 04
NB["04_types_and_asyncio"] = [
    header("04", "Typed messages and asyncio", "S6 (Type hints, dataclasses, enums & Pydantic) · S7 (Concurrency & asyncio) · Clinic W2",
           "1. Model data with frozen dataclasses and enums.\n2. Validate broker messages with Pydantic, including a rule across fields.\n"
           "3. Run I/O-bound calls concurrently with `asyncio.gather`.\n4. See back-pressure: what a bounded queue does when the consumer is slow."),
    ("code", SETUP),
    ("md", "## 1. Dataclasses and enums"),
    ("code", """from dataclasses import dataclass, FrozenInstanceError
from enum import Enum

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    side: Side
    qty: int
    price: Decimal

f = Fill("SPY", Side.BUY, 100, Decimal("512.10"))
print(f, Side("SELL"), f == Fill("SPY", Side.BUY, 100, Decimal("512.10")))
try:
    f.qty = 200
except FrozenInstanceError as e:
    print("frozen:", e)"""),
    ("md", "## 2. Validating messages with Pydantic"),
    ("md", YOUR_TURN),
    ("ex", """from typing import Literal
from pydantic import BaseModel, Field, ValidationError, model_validator

class OrderRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)
    side: Literal["BUY", "SELL"]
    qty: int = Field(gt=0)
    order_type: Literal["MKT", "LMT"]
    limit_price: Decimal | None = None

    @model_validator(mode="after")
    def price_rules(self):
        # ✍️ LMT needs a limit_price; MKT must NOT have one (raise ValueError with a message)
        ...
        return self

payloads = [{"symbol": "SPY", "side": "BUY", "qty": 100, "order_type": "LMT", "limit_price": "512.10"},
            {"symbol": "SPY", "side": "BUY", "qty": 100, "order_type": "LMT"},
            {"symbol": "SPY", "side": "BUY", "qty": 100, "order_type": "MKT", "limit_price": "500"},
            {"symbol": "SPY", "side": "SELL", "qty": 0, "order_type": "MKT"},
            {"symbol": "QQQ", "side": "SELL", "qty": 5, "order_type": "MKT"}]

def is_valid(d):
    try:
        OrderRequest(**d)
        return True
    except ValidationError:
        return False

valid = p.check("order validation", [is_valid(d) for d in payloads], [True, False, False, False, True])
try:
    OrderRequest(**payloads[1])
except ValidationError as e:
    print(e)""",
     """from typing import Literal
from pydantic import BaseModel, Field, ValidationError, model_validator

class OrderRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12)
    side: Literal["BUY", "SELL"]
    qty: int = Field(gt=0)
    order_type: Literal["MKT", "LMT"]
    limit_price: Decimal | None = None

    @model_validator(mode="after")
    def price_rules(self):
        if self.order_type == "LMT" and self.limit_price is None:
            raise ValueError("a limit order needs a limit_price")
        if self.order_type == "MKT" and self.limit_price is not None:
            raise ValueError("a market order must not have a limit_price")
        return self

payloads = [{"symbol": "SPY", "side": "BUY", "qty": 100, "order_type": "LMT", "limit_price": "512.10"},
            {"symbol": "SPY", "side": "BUY", "qty": 100, "order_type": "LMT"},
            {"symbol": "SPY", "side": "BUY", "qty": 100, "order_type": "MKT", "limit_price": "500"},
            {"symbol": "SPY", "side": "SELL", "qty": 0, "order_type": "MKT"},
            {"symbol": "QQQ", "side": "SELL", "qty": 5, "order_type": "MKT"}]

def is_valid(d):
    try:
        OrderRequest(**d)
        return True
    except ValidationError:
        return False

valid = p.check("order validation", [is_valid(d) for d in payloads], [True, False, False, False, True])
try:
    OrderRequest(**payloads[1])
except ValidationError as e:
    print(e)"""),
    ("md", "## 3. `asyncio`: waiting for many things at once\n\nEach fake quote request takes 0.2 s of *waiting* (network), not CPU. Sequentially, 5 requests take 1 s."),
    ("code", """import asyncio, time

async def get_quote(symbol: str) -> dict:
    await asyncio.sleep(0.2)                      # the network round trip
    return {"symbol": symbol, "bid": 99.99, "ask": 100.01}

SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
t0 = time.perf_counter()
seq = [await get_quote(s) for s in SYMBOLS]
print(f"sequential: {time.perf_counter() - t0:.2f} s")"""),
    ("md", YOUR_TURN),
    ("ex", """t0 = time.perf_counter()
# ✍️ request all SYMBOLS concurrently with asyncio.gather (results in the same order)
quotes = ...
elapsed = time.perf_counter() - t0
print(f"concurrent: {elapsed:.2f} s")
quotes = p.check("asyncio.gather", quotes, seq)
fast = p.check("finished in well under a second", elapsed < 0.6, True)""",
     """t0 = time.perf_counter()
quotes = await asyncio.gather(*(get_quote(s) for s in SYMBOLS))
elapsed = time.perf_counter() - t0
print(f"concurrent: {elapsed:.2f} s")
quotes = p.check("asyncio.gather", quotes, seq)
fast = p.check("finished in well under a second", elapsed < 0.6, True)"""),
    ("md", "## 4. Back-pressure with a bounded queue (clinic W2)\n\nA fast feed (1 quote / ms) and a slow consumer (1.5 ms per quote). With an unbounded queue the backlog — and the age of the data you act on — keeps growing. A bounded queue makes the producer wait instead."),
    ("code", """async def run(maxsize: int, n: int = 300):
    q, ages, depth = asyncio.Queue(maxsize=maxsize), [], []

    async def producer():
        for i in range(n):
            await q.put(time.perf_counter())      # waits when a bounded queue is full
            await asyncio.sleep(0.001)
        await q.put(None)

    async def consumer():
        while (ts := await q.get()) is not None:
            await asyncio.sleep(0.0015)           # slow processing
            ages.append((time.perf_counter() - ts) * 1e3)
            depth.append(q.qsize())

    await asyncio.gather(producer(), consumer())
    return np.array(ages), np.array(depth)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for maxsize, label in ((0, "unbounded"), (10, "bounded (10)")):
    ages, depth = await run(maxsize)
    axes[0].plot(ages, label=f"{label}: p99 {np.percentile(ages, 99):.0f} ms")
    axes[1].plot(depth, label=label)
axes[0].set(title="Age of each quote when processed (ms)", xlabel="quote #"); axes[0].legend()
axes[1].set(title="Queue depth", xlabel="quote #"); axes[1].legend()
plt.tight_layout(); plt.show()"""),
    ("md", """## Questions
1. Why is `asyncio` a good fit for talking to brokers and data feeds, but no help for a CPU-heavy backtest?
2. The bounded queue slows the producer. In a live feed you cannot slow the exchange — what would you do instead (drop, conflate to the latest quote, …)?
3. Which Pydantic rule above would you also enforce in the risk engine, and why twice?

**Graded version:** `labs/part03/week08_advanced` (Pydantic `OrderRequest`, async pipeline) and `labs/part03/clinic_w2_async_sim`."""),
]

# ---------------------------------------------------------------------------------------------- 05
NB["05_performance"] = [
    header("05", "Performance and memory", "S8 (Performance & memory) · S17 (NumPy for market data)",
           "1. Measure first: time a Python loop against NumPy on a million prices.\n"
           "2. Vectorize a drawdown with `np.maximum.accumulate`.\n"
           "3. Compile a loop you cannot vectorize (an EMA) with `numba`.\n4. Cut a DataFrame's memory with the right dtypes."),
    ("code", SETUP),
    ("code", """x = p.prices_array(1_000_000)
print(f"{len(x):,} prices, {x.nbytes / 1e6:.0f} MB as float64")"""),
    ("md", "## 1. Vectorizing a drawdown\n\n$DD_t = P_t / \\max_{s \\le t} P_s - 1$"),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ the drawdown of every price in one vectorized line (hint: np.maximum.accumulate)
dd = ...
dd = p.check("vectorized drawdown", dd, p.drawdown_vectorized(x))
print(f"max drawdown {dd.min():.1%}")""",
     """dd = x / np.maximum.accumulate(x) - 1
dd = p.check("vectorized drawdown", dd, p.drawdown_vectorized(x))
print(f"max drawdown {dd.min():.1%}")"""),
    ("md", "## 2. Compiling a loop with numba\n\nAn EMA depends on its own previous value, so it cannot be written as one NumPy expression. `numba` compiles the plain loop to machine code."),
    ("code", """def ema_py(x, alpha):
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out"""),
    ("md", YOUR_TURN),
    ("ex", """from numba import njit
# ✍️ a compiled version of ema_py (hint: njit(ema_py))
ema_fast = ...
compiled = p.check("compiled with numba", hasattr(ema_fast, "signatures"), True)
if not hasattr(ema_fast, "signatures"):
    ema_fast = njit(ema_py)                       # not done yet: compile it for you so the rest runs
vals = p.check("numba EMA matches the loop", ema_fast(x[:100_000], 0.1), p.ema_loop(x[:100_000], 0.1))""",
     """from numba import njit
ema_fast = njit(ema_py)
compiled = p.check("compiled with numba", hasattr(ema_fast, "signatures"), True)
if not hasattr(ema_fast, "signatures"):
    ema_fast = njit(ema_py)                       # not done yet: compile it for you so the rest runs
vals = p.check("numba EMA matches the loop", ema_fast(x[:100_000], 0.1), p.ema_loop(x[:100_000], 0.1))"""),
    ("code", """small = x[:200_000]
t = {"Python loop": p.timeit(ema_py, small, 0.1, repeat=2),
     "pandas ewm": p.timeit(lambda a: pd.Series(a).ewm(alpha=0.1, adjust=False).mean(), small, repeat=3),
     "numba": p.timeit(ema_fast, small, 0.1, repeat=5)}
ax = pd.Series(t).mul(1e3).plot.barh(title="EMA of 200,000 prices (ms, lower is better)", logx=True)
ax.set_xlabel("milliseconds (log scale)"); plt.show()
print({k: f"{v * 1e3:.2f} ms" for k, v in t.items()}, "— the first numba call also paid a one-off compile time")"""),
    ("md", "## 3. Memory: choose dtypes"),
    ("code", """rng = np.random.default_rng(0)
ticks = pd.DataFrame({"symbol": rng.choice(["SPY", "QQQ", "IWM", "TLT", "GLD"], 1_000_000),
                      "price": rng.uniform(50, 500, 1_000_000), "size": rng.integers(1, 1000, 1_000_000)})
ticks.dtypes"""),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ convert symbol to "category", price to "float32" and size to "int32" (hint: .astype({...}))
compact = ...
got = (str(getattr(compact, "dtypes", {}).get("symbol")), str(getattr(compact, "dtypes", {}).get("price")),
       str(getattr(compact, "dtypes", {}).get("size")))
got = p.check("compact dtypes", got, ("category", "float32", "int32"))
if got == ("category", "float32", "int32") and compact is Ellipsis:
    compact = ticks.astype({"symbol": "category", "price": "float32", "size": "int32"})""",
     """compact = ticks.astype({"symbol": "category", "price": "float32", "size": "int32"})
got = (str(getattr(compact, "dtypes", {}).get("symbol")), str(getattr(compact, "dtypes", {}).get("price")),
       str(getattr(compact, "dtypes", {}).get("size")))
got = p.check("compact dtypes", got, ("category", "float32", "int32"))
if got == ("category", "float32", "int32") and compact is Ellipsis:
    compact = ticks.astype({"symbol": "category", "price": "float32", "size": "int32"})"""),
    ("code", """mem = pd.DataFrame({"before (MB)": ticks.memory_usage(deep=True) / 1e6,
                    "after (MB)": compact.memory_usage(deep=True) / 1e6}).drop("Index")
mem.loc["total"] = mem.sum()
print("float32 keeps about 7 significant digits: fine for a price display, NOT for money or P&L sums.")
mem.round(1)"""),
    ("md", """## Questions
1. Why measure before optimizing? Which of today's speed-ups would matter in a live loop receiving one bar a minute?
2. When is `float32` acceptable, and when is it dangerous?
3. The first numba call is slow. How would you hide that cost in a trading engine?

**Graded version:** `labs/part03/week08_advanced` (EMA) and `labs/part03/week11_data` (NumPy and pandas)."""),
]

# ---------------------------------------------------------------------------------------------- 06
NB["06_oop_and_patterns"] = [
    header("06", "Objects, domain models and design patterns", "S9–S12 (OOP and domain modelling) · S13–S16 (SOLID, patterns, UML)",
           "1. Give a value object behaviour with dunder methods (`Money`).\n2. Expose derived values as properties (`Position`).\n"
           "3. Enforce an order's life cycle with a state machine.\n4. See the Observer and Strategy patterns decouple a trading system."),
    ("code", SETUP),
    ("md", "## 1. A value object: `Money`"),
    ("md", YOUR_TURN),
    ("ex", """from dataclasses import dataclass

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "USD"

    def __add__(self, other: "Money") -> "Money":
        # ✍️ raise ValueError if the currencies differ, else return a new Money with the summed amount
        return ...

total = Money(Decimal("1.10")) + Money(Decimal("2.20"))
try:
    Money(Decimal("1"), "USD") + Money(Decimal("1"), "EUR")
    raised = False
except ValueError:
    raised = True
result = p.check("Money addition", (getattr(total, "amount", total), raised), (Decimal("3.30"), True))
result""",
     """from dataclasses import dataclass

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "USD"

    def __add__(self, other: "Money") -> "Money":
        if other.currency != self.currency:
            raise ValueError(f"cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)

total = Money(Decimal("1.10")) + Money(Decimal("2.20"))
try:
    Money(Decimal("1"), "USD") + Money(Decimal("1"), "EUR")
    raised = False
except ValueError:
    raised = True
result = p.check("Money addition", (getattr(total, "amount", total), raised), (Decimal("3.30"), True))
result"""),
    ("md", "## 2. Properties: derived values that are never stale"),
    ("md", YOUR_TURN),
    ("ex", """class Position:
    def __init__(self, symbol):
        self.symbol, self.qty, self.cost = symbol, 0, Decimal("0")

    def buy(self, qty, price):
        self.qty += qty
        self.cost += qty * price

    @property
    def avg_price(self) -> Decimal:
        # ✍️ the average cost per share (cost / qty); Decimal("0") when flat
        return ...

    def unrealized(self, mark: Decimal) -> Decimal:
        return self.qty * (mark - self.avg_price) if self.qty else Decimal("0")

    def __repr__(self):
        return f"Position({self.symbol}, qty={self.qty}, avg={self.avg_price})"

pos = Position("SPY")
pos.buy(100, Decimal("500"))
pos.buy(300, Decimal("510"))
avg = p.check("average price property", pos.avg_price, Decimal("507.5"))
print(pos, "unrealized at 512:", pos.unrealized(Decimal("512")) if isinstance(pos.avg_price, Decimal) else "…")""",
     """class Position:
    def __init__(self, symbol):
        self.symbol, self.qty, self.cost = symbol, 0, Decimal("0")

    def buy(self, qty, price):
        self.qty += qty
        self.cost += qty * price

    @property
    def avg_price(self) -> Decimal:
        return self.cost / self.qty if self.qty else Decimal("0")

    def unrealized(self, mark: Decimal) -> Decimal:
        return self.qty * (mark - self.avg_price) if self.qty else Decimal("0")

    def __repr__(self):
        return f"Position({self.symbol}, qty={self.qty}, avg={self.avg_price})"

pos = Position("SPY")
pos.buy(100, Decimal("500"))
pos.buy(300, Decimal("510"))
avg = p.check("average price property", pos.avg_price, Decimal("507.5"))
print(pos, "unrealized at 512:", pos.unrealized(Decimal("512")) if isinstance(pos.avg_price, Decimal) else "…")"""),
    ("md", "## 3. The order life cycle as a state machine\n\nIllegal transitions (a fill after a cancel, a second fill after FILLED) are bugs or broker surprises: refuse them loudly."),
    ("code", """pd.DataFrame([(s, ", ".join(sorted(t)) or "— (final)") for s, t in p.ORDER_TRANSITIONS.items()],
             columns=["state", "may go to"]).set_index("state")"""),
    ("md", YOUR_TURN),
    ("ex", """def replay(events):
    state = "NEW"
    for e in events:
        # ✍️ raise ValueError(f"{state} -> {e} is not allowed") if e is not in p.ORDER_TRANSITIONS[state]
        ...
        state = e
    return state

def outcome(events):
    try:
        return replay(events)
    except ValueError:
        return "error"

seqs = [["SUBMITTED", "PARTIAL", "PARTIAL", "FILLED"], ["SUBMITTED", "CANCELLED", "FILLED"],
        ["FILLED"], ["SUBMITTED", "REJECTED"], ["SUBMITTED", "FILLED", "PARTIAL"]]
states = p.check("order state machine", [outcome(s) for s in seqs], ["FILLED", "error", "error", "REJECTED", "error"])
states""",
     """def replay(events):
    state = "NEW"
    for e in events:
        if e not in p.ORDER_TRANSITIONS[state]:
            raise ValueError(f"{state} -> {e} is not allowed")
        state = e
    return state

def outcome(events):
    try:
        return replay(events)
    except ValueError:
        return "error"

seqs = [["SUBMITTED", "PARTIAL", "PARTIAL", "FILLED"], ["SUBMITTED", "CANCELLED", "FILLED"],
        ["FILLED"], ["SUBMITTED", "REJECTED"], ["SUBMITTED", "FILLED", "PARTIAL"]]
states = p.check("order state machine", [outcome(s) for s in seqs], ["FILLED", "error", "error", "REJECTED", "error"])
states"""),
    ("md", "## 4. Observer and Strategy: plug parts in without editing the core"),
    ("code", """from collections import defaultdict
from typing import Callable, Protocol

class EventBus:                                   # Observer: publishers don't know their subscribers
    def __init__(self):
        self.subs: dict[str, list[Callable]] = defaultdict(list)
    def subscribe(self, topic, fn):
        self.subs[topic].append(fn)
    def publish(self, topic, payload):
        for fn in self.subs[topic]:
            fn(payload)

class FillModel(Protocol):                        # Strategy: any object with this method will do
    def fill_price(self, side: str, next_open: Decimal) -> Decimal: ...

class NextOpen:
    def fill_price(self, side, next_open):
        return next_open

class NextOpenWithSlippage:
    def __init__(self, bps: Decimal):
        self.bps = bps
    def fill_price(self, side, next_open):
        sign = 1 if side == "BUY" else -1
        return (next_open * (1 + sign * self.bps / 10_000)).quantize(Decimal("0.01"))

journal, alerts = [], []
bus = EventBus()
bus.subscribe("fill", journal.append)
bus.subscribe("fill", lambda f: alerts.append(f) if f["qty"] >= 1000 else None)
for model in (NextOpen(), NextOpenWithSlippage(Decimal("5"))):
    for qty in (100, 2000):
        bus.publish("fill", {"model": type(model).__name__, "qty": qty,
                             "price": model.fill_price("BUY", Decimal("512.00"))})
print(f"journal: {len(journal)} fills, alerts: {len(alerts)} large fills")
pd.DataFrame(journal)"""),
    ("md", """## Questions
1. `Money` is frozen. What bugs does immutability prevent in a multi-strategy system?
2. Why is `avg_price` a property and not an attribute updated by hand?
3. Which SOLID principle lets you add a new fill model without touching the backtester?

**Graded version:** `labs/part03/week09_domain` (Money, Position, property tests) and `labs/part03/week10_patterns`."""),
]

# ---------------------------------------------------------------------------------------------- 07
NB["07_pandas_polars"] = [
    header("07", "Time series with pandas and Polars", "S18 (pandas & Polars for time series)",
           "1. Resample trades into OHLCV bars.\n2. Join each trade to the quote in force at that moment with `merge_asof`, and see how a careless join leaks the future.\n"
           "3. Measure effective spreads.\n4. Compute VWAP bars with Polars and compare speed."),
    ("code", SETUP),
    ("code", """trades, quotes = p.trades_and_quotes()
print(f"{len(trades)} trades, {len(quotes)} quotes, UTC timestamps")
trades.head(3)"""),
    ("md", "## 1. OHLCV bars"),
    ("md", YOUR_TURN),
    ("ex", """g = trades.set_index("time").resample("5min")
# ✍️ open/high/low/close of price (hint: g["price"].ohlc()) and the summed size as "volume"; drop empty bars
bars = ...
bars = p.check("5-minute OHLCV bars", bars, p.ohlcv(trades))
bars.head()""",
     """g = trades.set_index("time").resample("5min")
bars = g["price"].ohlc()
bars["volume"] = g["size"].sum()
bars = bars.dropna()
bars = p.check("5-minute OHLCV bars", bars, p.ohlcv(trades))
bars.head()"""),
    ("md", "## 2. As-of joins: the quote in force at each trade"),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ each trade with the LAST quote at or before it: pd.merge_asof(..., on="time", direction="backward")
tq = ...
tq = p.check("merge_asof (no look-ahead)", tq, p.with_quotes_asof(trades, quotes))
tq.head()""",
     """tq = pd.merge_asof(trades.sort_values("time"), quotes.sort_values("time"), on="time", direction="backward")
tq = p.check("merge_asof (no look-ahead)", tq, p.with_quotes_asof(trades, quotes))
tq.head()"""),
    ("code", """q2 = quotes.rename(columns={"time": "quote_time"})
nearest = pd.merge_asof(trades, q2, left_on="time", right_on="quote_time", direction="nearest")
leak = (nearest["quote_time"] > nearest["time"]).mean()
print(f"direction='nearest' used a quote from the FUTURE for {leak:.0%} of trades — a silent look-ahead.")
tq["mid"] = (tq["bid"] + tq["ask"]) / 2
tq["eff_spread_bps"] = 2 * (tq["price"] - tq["mid"]).abs() / tq["mid"] * 1e4
ax = tq.set_index("time")["eff_spread_bps"].resample("5min").mean().plot(title="Average effective spread (bps)")
ax.set_xlabel(""); plt.show()"""),
    ("md", "## 3. Polars: the same job, faster"),
    ("md", YOUR_TURN),
    ("ex", """import polars as pl
pt = pl.from_pandas(trades)
# ✍️ an expression for VWAP: sum(price × size) / sum(size)   (hint: (pl.col("price") * pl.col("size")).sum() / ...)
vwap_expr = ...
vwap_pl = None
if vwap_expr is not Ellipsis:
    out = pt.sort("time").group_by_dynamic("time", every="5m").agg(vwap_expr.alias("vwap")).to_pandas()
    vwap_pl = out.set_index("time")["vwap"]
ref = trades.set_index("time").assign(pv=lambda d: d["price"] * d["size"]).resample("5min")[["pv", "size"]].sum()
ref = (ref["pv"] / ref["size"]).dropna().rename("vwap")
vwap = p.check("VWAP with Polars", vwap_pl, ref)""",
     """import polars as pl
pt = pl.from_pandas(trades)
vwap_expr = (pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()
vwap_pl = None
if vwap_expr is not Ellipsis:
    out = pt.sort("time").group_by_dynamic("time", every="5m").agg(vwap_expr.alias("vwap")).to_pandas()
    vwap_pl = out.set_index("time")["vwap"]
ref = trades.set_index("time").assign(pv=lambda d: d["price"] * d["size"]).resample("5min")[["pv", "size"]].sum()
ref = (ref["pv"] / ref["size"]).dropna().rename("vwap")
vwap = p.check("VWAP with Polars", vwap_pl, ref)"""),
    ("code", """rng = np.random.default_rng(1)
n = 3_000_000
big = pd.DataFrame({"symbol": rng.choice([f"S{i}" for i in range(500)], n), "price": rng.uniform(10, 500, n),
                    "size": rng.integers(1, 1000, n)})
bpl = pl.from_pandas(big)
t_pd = p.timeit(lambda: big.assign(pv=big["price"] * big["size"]).groupby("symbol")[["pv", "size"]].sum(), repeat=3)
t_pl = p.timeit(lambda: bpl.group_by("symbol").agg((pl.col("price") * pl.col("size")).sum(), pl.col("size").sum()), repeat=3)
print(f"VWAP per symbol over {n:,} rows: pandas {t_pd * 1e3:.0f} ms, Polars {t_pl * 1e3:.0f} ms")"""),
    ("md", """## Questions
1. Why is `direction="nearest"` wrong for a backtest even though it looks "more accurate"?
2. A resample on naive local timestamps misbehaves twice a year. Why do the timestamps here carry UTC?
3. When would you keep pandas rather than switch to Polars?

**Graded version:** `labs/part03/week11_data` (resampling, `merge_asof`, Polars lazy SMA)."""),
]

# ---------------------------------------------------------------------------------------------- 08
NB["08_sql_duckdb_testing"] = [
    header("08", "SQL, Parquet, validation and property tests", "S19–S20 (SQL, Parquet, DuckDB & data validation) · S22 (Testing in depth)",
           "1. Store bars as partitioned Parquet and query them with DuckDB SQL.\n2. Use a window function for a moving average per symbol.\n"
           "3. Validate bars before anything trusts them.\n4. Write a property-based test that catches a float bug."),
    ("code", SETUP + """
import duckdb, tempfile
import pyarrow as pa, pyarrow.parquet as pq"""),
    ("code", """rng = np.random.default_rng(0)
dates = pd.bdate_range("2024-01-02", periods=250)
frames = []
for i, sym in enumerate(["SPY", "QQQ", "IWM", "TLT", "GLD"]):
    close = 100 * (1 + i / 10) * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    open_ = close * np.exp(rng.normal(0, 0.003, len(dates)))
    frames.append(pd.DataFrame({"date": dates, "symbol": sym, "open": open_, "close": close,
                                "high": np.maximum(open_, close) * 1.004, "low": np.minimum(open_, close) * 0.996,
                                "volume": rng.integers(1_000_000, 5_000_000, len(dates))}))
bars = pd.concat(frames, ignore_index=True)
root = Path(tempfile.mkdtemp()) / "bars"
pq.write_to_dataset(pa.Table.from_pandas(bars), root, partition_cols=["symbol"])      # hive: symbol=SPY/...
print(sorted(x.name for x in root.iterdir()))
con = duckdb.connect()
con.sql(f"CREATE VIEW bars AS SELECT * FROM read_parquet('{root}/**/*.parquet', hive_partitioning = true)")
con.sql("SELECT symbol, COUNT(*) AS n FROM bars GROUP BY symbol ORDER BY symbol").df()"""),
    ("md", "## 1. SQL over Parquet"),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ per symbol: average volume (avg_volume) and highest close (max_close), ordered by symbol
sql = ...
got = con.sql(sql).df() if isinstance(sql, str) else None
ref = bars.groupby("symbol").agg(avg_volume=("volume", "mean"), max_close=("close", "max")).reset_index()
summary = p.check("SQL aggregation", got, ref)
summary""",
     """sql = \"\"\"SELECT symbol, AVG(volume) AS avg_volume, MAX(close) AS max_close
FROM bars GROUP BY symbol ORDER BY symbol\"\"\"
got = con.sql(sql).df() if isinstance(sql, str) else None
ref = bars.groupby("symbol").agg(avg_volume=("volume", "mean"), max_close=("close", "max")).reset_index()
summary = p.check("SQL aggregation", got, ref)
summary"""),
    ("md", "## 2. Window functions: a moving average per symbol"),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ fill in the window: AVG(close) OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
window = ...
got = None
if isinstance(window, str):
    got = con.sql(f"SELECT symbol, date, AVG(close) OVER ({window}) AS sma20 FROM bars ORDER BY symbol, date").df()
    got = got.groupby("symbol").apply(lambda d: d.iloc[19:], include_groups=False)["sma20"].reset_index(drop=True)
ref = (bars.sort_values(["symbol", "date"]).groupby("symbol")["close"].rolling(20).mean().dropna()
       .reset_index(drop=True).rename("sma20"))
sma = p.check("SQL moving average", got, ref)""",
     """window = "PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW"
got = None
if isinstance(window, str):
    got = con.sql(f"SELECT symbol, date, AVG(close) OVER ({window}) AS sma20 FROM bars ORDER BY symbol, date").df()
    got = got.groupby("symbol").apply(lambda d: d.iloc[19:], include_groups=False)["sma20"].reset_index(drop=True)
ref = (bars.sort_values(["symbol", "date"]).groupby("symbol")["close"].rolling(20).mean().dropna()
       .reset_index(drop=True).rename("sma20"))
sma = p.check("SQL moving average", got, ref)"""),
    ("md", "## 3. Validate before you trust"),
    ("code", """dirty = bars.copy()
dirty.loc[10, "high"] = dirty.loc[10, "close"] * 0.9          # high below the close
dirty.loc[300, "low"] = dirty.loc[300, "open"] * 1.1          # low above the open
dirty.loc[777, "close"] = -1.0                                # impossible price
dirty.loc[900, "volume"] = -5                                  # impossible volume"""),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ a boolean Series: high < max(open, close) OR low > min(open, close) OR any price <= 0 OR volume < 0
bad = ...
bad = p.check("bad-bar detector", bad, p.bad_bars(dirty))
dirty[bad]""",
     """bad = ((dirty["high"] < dirty[["open", "close"]].max(axis=1)) | (dirty["low"] > dirty[["open", "close"]].min(axis=1))
       | (dirty[["open", "high", "low", "close"]] <= 0).any(axis=1) | (dirty["volume"] < 0))
bad = p.check("bad-bar detector", bad, p.bad_bars(dirty))
dirty[bad]"""),
    ("md", "## 4. Property-based testing with Hypothesis\n\nInstead of a few hand-picked examples, state a **property** that must hold for every input and let Hypothesis search for a counterexample."),
    ("code", """from hypothesis import given, settings, strategies as st

def round_to_tick_float(price: Decimal, tick: Decimal) -> Decimal:
    \"\"\"The buggy version: rounds in float, then turns the float back into a Decimal.\"\"\"
    return Decimal(round(float(price) / float(tick)) * float(tick))

def passes(test) -> bool:
    try:
        test()
        return True
    except AssertionError:
        return False"""),
    ("md", YOUR_TURN),
    ("ex", """def make_test(round_fn):
    @settings(max_examples=300, deadline=None)
    @given(st.decimals(min_value=Decimal("0.01"), max_value=Decimal("10000"), places=4),
           st.sampled_from([Decimal("0.01"), Decimal("0.05"), Decimal("0.25")]))
    def test(price, tick):
        r = round_fn(price, tick)
        # ✍️ assert r is on the tick grid (r % tick == 0) and within half a tick of price (abs(r - price) <= tick / 2)
        ...
    return test

result = (passes(make_test(p.round_to_tick)), not passes(make_test(round_to_tick_float)))
result = p.check("property: decimal passes, float bug caught", result, (True, True))""",
     """def make_test(round_fn):
    @settings(max_examples=300, deadline=None)
    @given(st.decimals(min_value=Decimal("0.01"), max_value=Decimal("10000"), places=4),
           st.sampled_from([Decimal("0.01"), Decimal("0.05"), Decimal("0.25")]))
    def test(price, tick):
        r = round_fn(price, tick)
        assert r % tick == 0
        assert abs(r - price) <= tick / 2
    return test

result = (passes(make_test(p.round_to_tick)), not passes(make_test(round_to_tick_float)))
result = p.check("property: decimal passes, float bug caught", result, (True, True))"""),
    ("md", """## Questions
1. Why partition Parquet by symbol (and often by date)? What does DuckDB skip when you filter on `symbol`?
2. The first 19 rows of the SQL moving average are averages of fewer than 20 closes. How would you make them NULL instead?
3. Which other properties would you test for a `Position` class (hint: realized + unrealized P&L)?

**Graded version:** `labs/part03/week11_data` (bar validation, hive Parquet, DuckDB) and the property test in `week09_domain`."""),
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
