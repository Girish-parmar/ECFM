"""Week 8 — Advanced Python (Part 3, S5–S8): generators, decorators, typing/Pydantic, asyncio, performance.

Fill in every block marked "Your turn", then run:  python -m pytest week08_advanced
"""
from __future__ import annotations

import asyncio
import csv
import functools
import random
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, field_validator, model_validator


# ------------------------------------------------------------------------ S5: generators
def read_ticks(path: str) -> Iterator[tuple[datetime, float, int]]:
    """Yield (timestamp, price, size) from a CSV with header ts,price,size (ts in ISO format with offset).
    Must be a GENERATOR (use `yield`), so huge files never load into memory."""
    # >>> SOLUTION
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield datetime.fromisoformat(row["ts"]), float(row["price"]), int(row["size"])
    # <<< SOLUTION


def bars_from_ticks(ticks: Iterable[tuple[datetime, float, int]], minutes: int = 1) -> Iterator[dict]:
    """Aggregate time-ordered ticks into bars of `minutes` minutes. Yield a dict per bar:
    {"ts": bar OPEN time (floored), "open", "high", "low", "close", "volume"}.
    Yield each bar as soon as a tick from a later bar arrives; yield the last bar at the end."""
    # >>> SOLUTION
    width = timedelta(minutes=minutes)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    bar = None
    for ts, price, size in ticks:
        start = ts - ((ts - epoch) % width)
        if bar is not None and start > bar["ts"]:
            yield bar
            bar = None
        if bar is None:
            bar = {"ts": start, "open": price, "high": price, "low": price, "close": price, "volume": size}
        else:
            bar["high"], bar["low"] = max(bar["high"], price), min(bar["low"], price)
            bar["close"], bar["volume"] = price, bar["volume"] + size
    if bar is not None:
        yield bar
    # <<< SOLUTION


# ------------------------------------------------------------------------ S5: decorators
def count_calls(fn: Callable) -> Callable:
    """Decorator: count calls in `wrapper.calls` (starting at 0) and keep the function's name and docstring."""
    # >>> SOLUTION
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return fn(*args, **kwargs)
    wrapper.calls = 0
    return wrapper
    # <<< SOLUTION


def retry(times: int = 5, base: float = 0.5, cap: float = 30.0,
          exceptions: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)) -> Callable:
    """Decorator for ASYNC functions: on one of `exceptions`, wait min(cap, base·2**attempt)·U(0.5, 1)
    seconds (asyncio.sleep) and try again, up to `times` attempts in total; then re-raise.
    Other exceptions propagate immediately."""
    # >>> SOLUTION
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions:
                    if attempt == times:
                        raise
                    await asyncio.sleep(min(cap, base * 2 ** attempt) * random.uniform(0.5, 1.0))
        return wrapper
    return deco
    # <<< SOLUTION


class TokenBucket:
    """Rate limiter: `rate` requests per second with bursts up to `capacity`.

    `delay()` returns how many seconds the caller must wait before its request is allowed
    (0.0 if a token is available now) and consumes the token. It never sleeps itself.
    `clock` returns the current time in seconds (injected so tests can control time)."""

    def __init__(self, rate: float, capacity: int, clock: Callable[[], float] = time.monotonic):
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.tokens, self.t = float(capacity), clock()

    def delay(self) -> float:
        # >>> SOLUTION
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now
        self.tokens -= 1
        return 0.0 if self.tokens >= 0 else -self.tokens / self.rate
        # <<< SOLUTION


# ------------------------------------------------------------------ S6: typing & Pydantic
class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"


class OrderRequest(BaseModel):
    """Validated order request. Rules (raise ValueError / let Pydantic raise):
    - symbol: 1–10 characters, stored in upper case
    - qty: > 0
    - LMT orders need a positive limit_price; MKT orders must NOT have a limit_price"""
    symbol: str
    side: Side
    qty: Decimal
    type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, v: str) -> str:
        # >>> SOLUTION
        if not 1 <= len(v) <= 10:
            raise ValueError("symbol must be 1-10 characters")
        return v.upper()
        # <<< SOLUTION

    @field_validator("qty")
    @classmethod
    def _qty(cls, v: Decimal) -> Decimal:
        # >>> SOLUTION
        if v <= 0:
            raise ValueError("qty must be positive")
        return v
        # <<< SOLUTION

    @model_validator(mode="after")
    def _limit(self) -> OrderRequest:
        # >>> SOLUTION
        if self.type == OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("limit orders need a positive limit_price")
        if self.type == OrderType.MARKET and self.limit_price is not None:
            raise ValueError("market orders must not have a limit_price")
        return self
        # <<< SOLUTION


# ------------------------------------------------------------------------- S7: asyncio
async def run_pipeline(n: int, maxsize: int = 10) -> list[int]:
    """Producer puts 0..n-1 on a BOUNDED asyncio.Queue(maxsize), then a None sentinel; the consumer
    collects items until the sentinel. Run both concurrently (asyncio.gather) and return the list."""
    # >>> SOLUTION
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    out: list[int] = []

    async def producer():
        for i in range(n):
            await q.put(i)
        await q.put(None)

    async def consumer():
        while (item := await q.get()) is not None:
            out.append(item)

    await asyncio.gather(producer(), consumer())
    return out
    # <<< SOLUTION


async def first_quote(sources: list[Callable[[], asyncio.Future]], timeout: float) -> object:
    """Start every source coroutine concurrently, return the FIRST result to arrive, cancel the others.
    If nothing arrives within `timeout` seconds, cancel everything and raise TimeoutError."""
    # >>> SOLUTION
    tasks = [asyncio.ensure_future(s()) for s in sources]
    try:
        done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            raise TimeoutError("no quote in time")
        return next(iter(done)).result()
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
    # <<< SOLUTION


# ------------------------------------------------------------------------- S8: performance
def ema(values: list[float], n: int) -> list[float]:
    """EMA with alpha = 2/(n+1), seeded with the SIMPLE AVERAGE of the first n values (TA-Lib style).
    Positions before n-1 are float('nan'). Must run in O(len(values))."""
    # >>> SOLUTION
    out = [float("nan")] * len(values)
    if len(values) < n:
        return out
    alpha = 2 / (n + 1)
    out[n - 1] = sum(values[:n]) / n
    for i in range(n, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out
    # <<< SOLUTION
