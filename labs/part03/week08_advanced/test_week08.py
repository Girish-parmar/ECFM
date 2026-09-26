import asyncio
import inspect
import math
from datetime import datetime, timezone
from decimal import Decimal as D

import numpy as np
import pytest
from _loader import load

ex = load("week08_advanced", "exercises")
UTC = timezone.utc


def test_read_ticks_is_a_generator(tmp_path):
    f = tmp_path / "ticks.csv"
    f.write_text("ts,price,size\n2025-03-03T14:30:05+00:00,100.5,10\n2025-03-03T14:30:40+00:00,100.7,5\n")
    gen = ex.read_ticks(str(f))
    assert inspect.isgenerator(gen)
    assert list(gen) == [(datetime(2025, 3, 3, 14, 30, 5, tzinfo=UTC), 100.5, 10),
                         (datetime(2025, 3, 3, 14, 30, 40, tzinfo=UTC), 100.7, 5)]


def test_bars_from_ticks():
    t = lambda m, s: datetime(2025, 3, 3, 14, m, s, tzinfo=UTC)   # noqa: E731
    ticks = [(t(30, 1), 10.0, 1), (t(30, 30), 12.0, 2), (t(30, 59), 11.0, 1), (t(32, 5), 9.0, 4)]
    bars = list(ex.bars_from_ticks(iter(ticks)))
    assert bars == [{"ts": t(30, 0), "open": 10.0, "high": 12.0, "low": 10.0, "close": 11.0, "volume": 4},
                    {"ts": t(32, 0), "open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 4}]
    five = list(ex.bars_from_ticks(iter(ticks), minutes=5))
    assert len(five) == 1 and five[0]["volume"] == 8
    assert list(ex.bars_from_ticks(iter([]))) == []


def test_count_calls():
    @ex.count_calls
    def f(x):
        """doc"""
        return x * 2
    assert f(2) == 4 and f(3) == 6 and f.calls == 2
    assert f.__name__ == "f" and f.__doc__ == "doc"


def test_retry(monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr(ex.asyncio, "sleep", no_sleep)
    calls = {"n": 0}

    @ex.retry(times=3)
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("down")
        return "ok"
    assert asyncio.run(flaky()) == "ok" and calls["n"] == 3

    @ex.retry(times=2)
    async def always_down():
        raise TimeoutError("still down")
    with pytest.raises(TimeoutError):
        asyncio.run(always_down())

    @ex.retry(times=5)
    async def bug():
        calls["n"] += 100
        raise KeyError("not retried")
    with pytest.raises(KeyError):
        asyncio.run(bug())
    assert calls["n"] == 103


def test_token_bucket():
    now = {"t": 0.0}
    tb = ex.TokenBucket(rate=2, capacity=2, clock=lambda: now["t"])
    assert tb.delay() == 0.0 and tb.delay() == 0.0         # burst of 2
    assert tb.delay() == pytest.approx(0.5)                   # 3rd request: wait for half a second
    now["t"] = 10.0                                           # refill (capped at capacity)
    assert [tb.delay() for _ in range(3)] == [0.0, 0.0, pytest.approx(0.5)]


def test_order_request_validation():
    o = ex.OrderRequest(symbol="spy", side="BUY", qty=D("10"), type="LMT", limit_price=D("500.1"))
    assert o.symbol == "SPY" and o.side == ex.Side.BUY
    bad = [dict(symbol="", side="BUY", qty=1), dict(symbol="SPY", side="BUY", qty=0),
           dict(symbol="SPY", side="HOLD", qty=1), dict(symbol="SPY", side="BUY", qty=1, type="LMT"),
           dict(symbol="SPY", side="BUY", qty=1, type="MKT", limit_price=D("1")),
           dict(symbol="WAYTOOLONGSYM", side="SELL", qty=1)]
    for kwargs in bad:
        with pytest.raises(ValueError):
            ex.OrderRequest(**kwargs)


def test_pipeline_and_backpressure():
    assert asyncio.run(ex.run_pipeline(500, maxsize=5)) == list(range(500))


def test_first_quote():
    async def fast():
        await asyncio.sleep(0.01)
        return "fast"

    async def slow():
        await asyncio.sleep(1)
        return "slow"

    async def main():
        t0 = asyncio.get_running_loop().time()
        res = await ex.first_quote([slow, fast], timeout=0.5)
        return res, asyncio.get_running_loop().time() - t0
    res, took = asyncio.run(main())
    assert res == "fast" and took < 0.5
    with pytest.raises(TimeoutError):
        asyncio.run(ex.first_quote([slow], timeout=0.05))


def test_ema_matches_reference_and_is_fast():
    rng = np.random.default_rng(0)
    x = list(100 + np.cumsum(rng.normal(size=5000)))
    got = ex.ema(x, 20)
    ref = np.full(len(x), np.nan); ref[19] = np.mean(x[:20]); a = 2 / 21
    for i in range(20, len(x)):
        ref[i] = a * x[i] + (1 - a) * ref[i - 1]
    assert all(math.isnan(g) for g in got[:19])
    np.testing.assert_allclose(got[19:], ref[19:], rtol=1e-12)
    big = list(range(200_000))
    import time
    t0 = time.perf_counter(); ex.ema(big, 50)
    assert time.perf_counter() - t0 < 2.0, "EMA should be O(n): do not recompute averages from scratch"
