"""Clinic W2 — Async quote simulator with back-pressure (Part 3, S7).

Several simulated feeds publish quotes at different rates onto ONE bounded asyncio.Queue. A consumer
computes the mid price and the latency (time from `sent_at` to processing). A slow consumer makes the
queue fill up: a bounded queue then slows the producers down (back-pressure) instead of growing memory.

Run:  python quote_sim.py            Test:  python -m pytest clinic_w2_async_sim
"""
from __future__ import annotations

import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Quote:
    feed: str
    bid: float
    ask: float
    sent_at: float                     # time.perf_counter() when the quote was produced


@dataclass
class Stats:
    processed: int = 0
    per_feed: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    max_queue: int = 0
    last_mid: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        lat = sorted(self.latencies_ms)
        return {"processed": self.processed, "per_feed": self.per_feed, "max_queue": self.max_queue,
                "p50_ms": statistics.median(lat) if lat else None,
                "p99_ms": lat[int(0.99 * (len(lat) - 1))] if lat else None}


async def feed(name: str, rate_hz: float, n: int, queue: asyncio.Queue, seed: int = 0) -> None:
    """Put `n` Quote objects on `queue`, one every 1/rate_hz seconds (asyncio.sleep), around a random-walk mid
    of 100 with a 0.02 spread. Set sent_at = time.perf_counter() just before putting.
    Use `await queue.put(...)` so a full queue makes this feed wait (back-pressure)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


async def consumer(queue: asyncio.Queue, n_total: int, stats: Stats, work_ms: float = 0.0) -> None:
    """Take `n_total` quotes from `queue`. For each: record queue.qsize() into stats.max_queue (max so far),
    the mid in stats.last_mid[feed], the latency in ms, per-feed counts and stats.processed.
    Simulate processing time with `await asyncio.sleep(work_ms / 1000)` (never time.sleep!)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


async def run(feeds: dict[str, float], n_per_feed: int, maxsize: int = 50, work_ms: float = 0.0,
              timeout: float = 30.0) -> Stats:
    """Create a bounded queue, start one `feed` task per (name, rate_hz) and one consumer, and wait for all
    of them with an overall timeout (asyncio.wait_for). Return the Stats."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    fast = asyncio.run(run({"IB": 500, "ALPACA": 200, "SIM": 1000}, n_per_feed=300))
    slow = asyncio.run(run({"IB": 500, "ALPACA": 200, "SIM": 1000}, n_per_feed=300, maxsize=20, work_ms=3))
    print("fast consumer:", fast.summary())
    print("slow consumer:", slow.summary())
