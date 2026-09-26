import asyncio

import pytest
from _loader import load

pt = load("week10_patterns", "patterns")


class Tick:
    def __init__(self, px):
        self.px = px


class Heartbeat:
    pass


def test_event_bus():
    seen = []
    bus = pt.EventBus()
    sync_h = lambda e: seen.append(("sync", e.px))            # noqa: E731

    async def async_h(e):
        seen.append(("async", e.px))
    bus.subscribe(Tick, sync_h); bus.subscribe(Tick, async_h)
    bus.subscribe(Heartbeat, lambda e: seen.append("hb"))
    asyncio.run(bus.publish(Tick(1)))
    assert seen == [("sync", 1), ("async", 1)]
    bus.unsubscribe(Tick, sync_h); bus.unsubscribe(Tick, lambda e: None)   # unknown handler: no error
    asyncio.run(bus.publish(Tick(2))); asyncio.run(bus.publish(Heartbeat()))
    assert seen[2:] == [("async", 2), "hb"]


def test_order_state_machine():
    m = pt.OrderStateMachine()
    for s in ["ACCEPTED", "PARTIALLY_FILLED", "PARTIALLY_FILLED", "PENDING_CANCEL", "FILLED"]:
        m.to(s)
    assert m.is_terminal and m.history[-1] == "FILLED" and len(m.history) == 6
    with pytest.raises(pt.OrderStateError):
        m.to("CANCELLED")
    m2 = pt.OrderStateMachine()
    assert not m2.is_terminal
    with pytest.raises(pt.OrderStateError):
        m2.to("FILLED")


def test_registry():
    reg = pt.Registry()

    @reg.register("sma")
    class SMA:
        def __init__(self, n=20):
            self.n = n
    assert reg.create("sma", n=5).n == 5 and reg.names == ["sma"]
    with pytest.raises(KeyError):
        reg.register("sma")(SMA)
    with pytest.raises(KeyError, match="sma"):
        reg.create("ema")


def test_risk_chain():
    eng = pt.RiskEngine([pt.MaxNotional(50_000), pt.MaxPosition(300)])
    assert eng.check({"qty": 100, "price": 400}, {"position": 0}) == pt.Decision(True)
    d = eng.check({"qty": 200, "price": 400}, {"position": 0})
    assert not d.approved and d.reason == "MaxNotional: notional 80000 > 50000"
    d = eng.check({"qty": 100, "price": 100}, {"position": 250})
    assert not d.approved and d.reason.startswith("MaxPosition:")
    assert eng.check({"qty": -100, "price": 100}, {"position": 250}).approved


def test_conditions():
    trend = pt.Condition(lambda c: c["close"] > c["sma"], "close>sma")
    hot = pt.Condition(lambda c: c["rsi"] > 70, "rsi>70")
    entry = trend & ~hot
    assert entry.name == "(close>sma AND NOT rsi>70)"
    assert entry({"close": 11, "sma": 10, "rsi": 50}) is True
    assert entry({"close": 11, "sma": 10, "rsi": 80}) is False
    either = trend | hot
    assert either.name == "(close>sma OR rsi>70)" and either({"close": 9, "sma": 10, "rsi": 80})


def test_fill_models():
    bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}
    assert pt.NextOpenFill().fill_price("BUY", None, bar) == 100.0
    lt = pt.LimitThroughFill()
    assert lt.fill_price("BUY", 99.5, bar) == 99.5
    assert lt.fill_price("BUY", 99.0, bar) is None                     # touched, not through
    assert lt.fill_price("BUY", 102.0, bar) == 100.0                   # marketable: fills at the open
    assert lt.fill_price("SELL", 100.8, bar) == 100.8
    assert lt.fill_price("SELL", 101.0, bar) is None
