"""Week 14 — Market data (Part 4, S5–S8): IB pacing, chunked downloads, one canonical bar schema,
a live bar builder and a read-through cache.

No broker connection is needed: tests feed recorded-shape DataFrames and a fake clock.
Fill in every block marked "Your turn", then run:  python -m pytest week14_data
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from common import CANONICAL_COLUMNS

UTC = timezone.utc


# ------------------------------------------------------------------ S5 IB historical requests
def ib_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, str]]:
    """Plan an IB download that walks BACKWARDS from `end` to `start` (both aware UTC).
    Return [(endDateTime, durationStr)] newest first; durationStr is f'{n} D'. Every chunk is `chunk_days`
    long except the last (oldest), which covers only what is left, rounded UP to whole days."""
    # >>> SOLUTION
    out, cur = [], end
    while cur > start:
        left = cur - start
        days = min(chunk_days, -(-left // timedelta(days=1)))       # ceil
        out.append((cur, f"{days} D"))
        cur -= timedelta(days=days)
    return out
    # <<< SOLUTION


class PacingGuard:
    """IB historical-data pacing rules (small bars), checked BEFORE each request:
      1. no identical request (same `key`) within 15 s,
      2. at most 6 requests for the same contract (key[0]) within 2 s,
      3. at most 60 requests in any 600 s window.
    `wait(key)` returns how many seconds to wait so that ALL rules hold (0.0 if none is violated);
    `record(key)` stores that a request was sent now. Time comes from `clock()` (seconds)."""

    def __init__(self, clock: Callable[[], float]):
        self.clock = clock
        self.sent: deque[tuple[float, tuple]] = deque()

    def wait(self, key: tuple) -> float:
        # >>> SOLUTION
        now = self.clock()
        w = 0.0
        same = [t for t, k in self.sent if k == key and now - t < 15]
        if same:
            w = max(w, max(same) + 15 - now)
        contract = [t for t, k in self.sent if k[0] == key[0] and now - t < 2]
        if len(contract) >= 6:
            w = max(w, sorted(contract)[-6] + 2 - now)
        window = [t for t, _ in self.sent if now - t < 600]
        if len(window) >= 60:
            w = max(w, sorted(window)[-60] + 600 - now)
        return w
        # <<< SOLUTION

    def record(self, key: tuple) -> None:
        # >>> SOLUTION
        now = self.clock()
        self.sent.append((now, key))
        while self.sent and now - self.sent[0][0] >= 600:
            self.sent.popleft()
        # <<< SOLUTION


# ---------------------------------------------------------------- S6 canonical bar schema
def normalize_ib(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """`df` is `ib_async.util.df(bars)` with columns date, open, high, low, close, volume, average, barCount
    (date is tz-aware or naive UTC). Return CANONICAL_COLUMNS: ts (UTC), vwap from `average`, trade_count from
    `barCount`, source 'ib', adjusted False. IB reports volume -1 (and average/barCount -1) when there is
    none (FX MIDPOINT): turn every -1 in volume/vwap/trade_count into NaN. Sort by ts, drop duplicate ts."""
    # >>> SOLUTION
    ts = pd.to_datetime(df["date"])
    ts = ts.dt.tz_localize(UTC) if ts.dt.tz is None else ts.dt.tz_convert(UTC)
    out = pd.DataFrame({
        "ts": ts, "symbol": symbol,
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "close": df["close"].astype(float),
        "volume": df["volume"].astype(float), "vwap": df["average"].astype(float),
        "trade_count": df["barCount"].astype(float),
        "timeframe": timeframe, "source": "ib", "adjusted": False,
    })
    for c in ("volume", "vwap", "trade_count"):
        out.loc[out[c] < 0, c] = np.nan
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)[CANONICAL_COLUMNS]
    # <<< SOLUTION


def normalize_alpaca(df: pd.DataFrame, timeframe: str, feed: str = "iex", adjusted: bool = True) -> pd.DataFrame:
    """`df` is `client.get_stock_bars(req).df`: a MultiIndex (symbol, timestamp) and columns open, high, low,
    close, volume, trade_count, vwap. Return CANONICAL_COLUMNS with source f'alpaca_{feed}', sorted by
    symbol then ts."""
    # >>> SOLUTION
    out = df.reset_index().rename(columns={"timestamp": "ts"})
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for c in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        out[c] = out[c].astype(float)
    out["timeframe"], out["source"], out["adjusted"] = timeframe, f"alpaca_{feed}", adjusted
    return out.sort_values(["symbol", "ts"]).reset_index(drop=True)[CANONICAL_COLUMNS]
    # <<< SOLUTION


# ------------------------------------------------------------------------ S7 live bars
@dataclass
class LiveBar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarBuilder:
    """Aggregate trades into fixed-interval bars; `start` = bar open (UTC), aligned to the epoch.

    on_trade(ts, price, size): a trade in a LATER interval first emits the current bar via `on_bar`.
    on_timer(now): emit the current bar if its interval has ended (now >= start + interval), even when no
    new trade arrived. A bar is never emitted twice. Trades older than the current bar are ignored."""

    def __init__(self, interval: timedelta, on_bar: Callable[[LiveBar], None]):
        self.interval, self.on_bar = interval, on_bar
        self.current: LiveBar | None = None

    def floor(self, ts: datetime) -> datetime:
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        return ts - ((ts - epoch) % self.interval)

    def on_trade(self, ts: datetime, price: float, size: float) -> None:
        # >>> SOLUTION
        start = self.floor(ts)
        b = self.current
        if b is not None and start < b.start:
            return
        if b is not None and start > b.start:
            self.on_bar(b)
            b = self.current = None
        if b is None:
            self.current = LiveBar(start, price, price, price, price, size)
        else:
            b.high, b.low, b.close, b.volume = max(b.high, price), min(b.low, price), price, b.volume + size
        # <<< SOLUTION

    def on_timer(self, now: datetime) -> None:
        # >>> SOLUTION
        if self.current is not None and now >= self.current.start + self.interval:
            self.on_bar(self.current)
            self.current = None
        # <<< SOLUTION


# ------------------------------------------------------------- S8 read-through bar cache
def missing_ranges(have: list[tuple[datetime, datetime]], start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Parts of [start, end) NOT covered by the half-open intervals in `have` (unsorted, may overlap).
    Return sorted, non-empty, non-overlapping intervals."""
    # >>> SOLUTION
    out, cur = [], start
    for a, b in sorted(have):
        if b <= cur or a >= end:
            continue
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
        if cur >= end:
            break
    if cur < end:
        out.append((cur, end))
    return out
    # <<< SOLUTION


class BarCache:
    """Read-through cache: `get(symbol, start, end)` returns canonical bars with start <= ts < end.
    Fetch ONLY the missing ranges from `source(symbol, a, b)` (which returns canonical bars in [a, b)), add
    them to the stored frame and to `self.covered[symbol]`, and never store duplicate ts.
    `self.calls` counts source calls (the tests use it to prove cache hits)."""

    def __init__(self, source: Callable[[str, datetime, datetime], pd.DataFrame]):
        self.source = source
        self.frames: dict[str, pd.DataFrame] = {}
        self.covered: dict[str, list[tuple[datetime, datetime]]] = {}
        self.calls = 0

    def get(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        # >>> SOLUTION
        have = self.covered.setdefault(symbol, [])
        new = []
        for a, b in missing_ranges(have, start, end):
            self.calls += 1
            new.append(self.source(symbol, a, b))
            have.append((a, b))
        if new:
            frames = [f for f in [self.frames.get(symbol), *new] if f is not None and len(f)]
            merged = pd.concat(frames) if frames else pd.DataFrame(columns=CANONICAL_COLUMNS)
            self.frames[symbol] = merged.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df = self.frames.get(symbol, pd.DataFrame(columns=CANONICAL_COLUMNS))
        return df[(df["ts"] >= start) & (df["ts"] < end)].reset_index(drop=True)
        # <<< SOLUTION
