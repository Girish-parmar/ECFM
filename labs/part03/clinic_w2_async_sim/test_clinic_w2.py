import asyncio

import pytest
from _loader import load

qs = load("clinic_w2_async_sim", "quote_sim")


def test_feed_puts_n_quotes():
    async def main():
        q = asyncio.Queue()
        await qs.feed("X", rate_hz=1000, n=25, queue=q)
        items = [q.get_nowait() for _ in range(q.qsize())]
        return items
    items = asyncio.run(main())
    assert len(items) == 25 and all(isinstance(i, qs.Quote) and i.feed == "X" for i in items)
    assert all(abs((i.ask - i.bid) - 0.02) < 1e-9 for i in items)


def test_run_counts_and_mids():
    stats = asyncio.run(qs.run({"A": 1000, "B": 500}, n_per_feed=100, maxsize=50))
    assert stats.processed == 200 and stats.per_feed == {"A": 100, "B": 100}
    assert set(stats.last_mid) == {"A", "B"} and all(90 < m < 110 for m in stats.last_mid.values())
    assert len(stats.latencies_ms) == 200 and min(stats.latencies_ms) >= 0


def test_backpressure_bounds_the_queue_and_raises_latency():
    fast = asyncio.run(qs.run({"A": 2000, "B": 2000}, n_per_feed=150, maxsize=10))
    slow = asyncio.run(qs.run({"A": 2000, "B": 2000}, n_per_feed=150, maxsize=10, work_ms=2))
    assert slow.max_queue <= 10 and fast.max_queue <= 10
    assert slow.summary()["p50_ms"] > fast.summary()["p50_ms"]


def test_timeout():
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        asyncio.run(qs.run({"A": 10}, n_per_feed=100, timeout=0.2))
