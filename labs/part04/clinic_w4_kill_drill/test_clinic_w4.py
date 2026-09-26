import asyncio
from decimal import Decimal as D

from _loader import load

dr = load("clinic_w4_kill_drill", "drill")


def test_load_book():
    b = dr.SlowBroker("x", 0.0, {s: D(100 + 10 * i) for i, s in enumerate(dr.SYMBOLS)})
    asyncio.run(dr.load_book(b, 3, 5))
    assert b.positions == {"SPY": D("10"), "QQQ": D("-5"), "IWM": D("10")}
    assert len(b.open) == 5 and all(r.limit_price < D("100") for r in b.open.values())


def test_drill_passes_and_brokers_are_handled_concurrently(tmp_path):
    rep = asyncio.run(dr.run_drill(tmp_path, latency=0.1))
    assert rep["passed"] and rep["flat"] and rep["open_orders"] == 0 and rep["tripped"]
    assert 0.2 <= rep["elapsed"] < 0.35            # cancel phase + flatten phase, both brokers in parallel
    assert (tmp_path / "killswitch.json").exists()


def test_drill_fails_when_too_slow(tmp_path):
    rep = asyncio.run(dr.run_drill(tmp_path, latency=0.2, limit=0.3))    # 0.4 s > 0.3 s limit
    assert rep["flat"] and not rep["passed"]


class FakeSwitch:
    def __init__(self):
        self.reasons = []

    async def trip(self, reason, flatten=True):
        self.reasons.append(reason)
        return 0.0


def test_watchdog_and_watch_loop():
    now = {"t": 0.0}
    dog = dr.Watchdog(timeout=5, clock=lambda: now["t"])
    assert not dog.stale
    now["t"] = 4
    dog.beat()
    now["t"] = 9
    assert not dog.stale                            # 5 s since the beat: not MORE than timeout
    ks = FakeSwitch()

    async def fake_sleep(s):
        now["t"] += s

    assert asyncio.run(dr.watch(dog, ks, poll=1, max_polls=0, sleep=fake_sleep)) is False
    assert asyncio.run(dr.watch(dog, ks, poll=1, max_polls=10, sleep=fake_sleep)) is True
    assert ks.reasons == ["heartbeat lost"] and now["t"] == 10
