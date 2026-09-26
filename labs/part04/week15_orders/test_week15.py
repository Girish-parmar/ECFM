import asyncio
import random
from decimal import Decimal as D

import pytest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from hypothesis import given, settings, strategies as st
from _loader import load
from common import (TIF, AssetClass, CircuitOpen, DuplicateOrder, Instrument, OrderRequest, OrderState, OrdType, Side,
                    UnsupportedOrder)

od = load("week15_orders", "orders")
S = OrderState
SPY = Instrument("SPY", AssetClass.EQUITY, exchange="ARCA")
BTC = Instrument("BTC", AssetClass.CRYPTO)
EUR = Instrument("EURUSD", AssetClass.FX)


# ------------------------------------------------------------------ connections
def test_backoff_delays():
    d = od.backoff_delays(8, base=1, cap=30, rng=random.Random(1))
    assert len(d) == 8
    for k, x in enumerate(d, start=1):
        full = min(30, 2 ** k)
        assert 0.5 * full <= x <= full
    assert od.backoff_delays(3, rng=random.Random(7)) == od.backoff_delays(3, rng=random.Random(7))


def test_with_backoff_retries_then_succeeds():
    calls, slept = {"n": 0}, []

    async def connect():
        calls["n"] += 1
        if calls["n"] < 4:
            raise ConnectionRefusedError("gateway down")
        return "connected"

    async def fake_sleep(s):
        slept.append(s)

    out = asyncio.run(od.with_backoff(connect, sleep=fake_sleep, rng=random.Random(3), base=1, cap=60))
    assert out == "connected" and calls["n"] == 4
    assert slept == od.backoff_delays(3, base=1, cap=60, rng=random.Random(3))


def test_with_backoff_gives_up_and_does_not_retry_other_errors():
    async def down():
        raise TimeoutError

    async def no_sleep(s):
        pass

    with pytest.raises(TimeoutError):
        asyncio.run(od.with_backoff(down, max_tries=3, sleep=no_sleep))
    calls = {"n": 0}

    async def bad_credentials():
        calls["n"] += 1
        raise PermissionError("401")      # an OSError subclass, so restrict retry_on: bad credentials never heal

    with pytest.raises(PermissionError):
        asyncio.run(od.with_backoff(bad_credentials, sleep=no_sleep, retry_on=(ConnectionError, TimeoutError)))
    assert calls["n"] == 1


def test_circuit_breaker():
    now = {"t": 0.0}
    cb = od.CircuitBreaker(threshold=3, reset_after=30, clock=lambda: now["t"])

    def boom():
        raise ConnectionError("503")

    for _ in range(3):
        with pytest.raises(ConnectionError):
            cb.call(boom)
    assert cb.state == "open"
    with pytest.raises(CircuitOpen):
        cb.call(lambda: "never called")
    now["t"] = 30
    assert cb.state == "half_open"
    with pytest.raises(ConnectionError):                 # trial call fails -> open again immediately
        cb.call(boom)
    assert cb.state == "open"
    now["t"] = 61
    assert cb.call(lambda: 42) == 42 and cb.state == "closed" and cb.failures == 0
    with pytest.raises(ConnectionError):
        cb.call(boom)
    assert cb.state == "closed"                          # one failure < threshold


@pytest.mark.parametrize("code, action", [(1100, "reconnect"), (504, "reconnect"), (1101, "resubscribe"),
                                          (1102, "ok"), (2104, "ok"), (2158, "ok"), (326, "fatal"),
                                          (162, "retry_later"), (201, "log")])
def test_classify_ib_code(code, action):
    assert od.classify_ib_code(code) == action


# ----------------------------------------------------------------------- mapping
def test_round_limit_to_tick():
    assert od.round_limit_to_tick(D("101.237"), D("0.01"), Side.BUY) == D("101.23")
    assert od.round_limit_to_tick(D("101.231"), D("0.01"), Side.SELL) == D("101.24")
    assert od.round_limit_to_tick(D("5012.60"), D("0.25"), Side.BUY) == D("5012.50")
    assert od.round_limit_to_tick(D("5012.50"), D("0.25"), Side.SELL) == D("5012.50")


def test_alpaca_price_ok():
    assert od.alpaca_price_ok(D("101.25")) and od.alpaca_price_ok(D("101.2500"))
    assert not od.alpaca_price_ok(D("101.255")) and not od.alpaca_price_ok(D("0"))
    assert od.alpaca_price_ok(D("0.1234")) and not od.alpaca_price_ok(D("0.12345"))


def _req(**kw):
    base = dict(instrument=SPY, side=Side.BUY, qty=D("10"), type=OrdType.LIMIT, limit_price=D("500.25"),
                client_order_id="qf-test-1")
    return OrderRequest(**{**base, **kw})


def test_to_ib_order():
    o = od.to_ib_order(_req(tif=TIF.GTC, extended_hours=True))
    assert (o.action, o.totalQuantity, o.orderType, o.lmtPrice, o.tif, o.orderRef, o.outsideRth) == \
           ("BUY", 10.0, "LMT", 500.25, "GTC", "qf-test-1", True)
    assert od.to_ib_order(_req(type=OrdType.MARKET, limit_price=None)).orderType == "MKT"
    o = od.to_ib_order(_req(side=Side.SELL, type=OrdType.STOP_LIMIT, limit_price=D("495"), stop_price=D("496")))
    assert (o.orderType, o.lmtPrice, o.auxPrice) == ("STP LMT", 495.0, 496.0)
    o = od.to_ib_order(_req(side=Side.SELL, type=OrdType.TRAIL, limit_price=None, trail_percent=D("1.5")))
    assert (o.orderType, o.trailingPercent) == ("TRAIL", 1.5)
    for bad in [_req(limit_price=None), _req(type=OrdType.STOP, limit_price=None), _req(qty=D("0"))]:
        with pytest.raises(UnsupportedOrder):
            od.to_ib_order(bad)


def test_to_alpaca_request():
    r = od.to_alpaca_request(_req(extended_hours=True))
    assert (r.symbol, r.qty, r.side, r.type, r.limit_price, r.time_in_force, r.client_order_id, r.extended_hours) == \
           ("SPY", 10.0, OrderSide.BUY, OrderType.LIMIT, 500.25, TimeInForce.DAY, "qf-test-1", True)
    r = od.to_alpaca_request(_req(instrument=BTC, type=OrdType.MARKET, limit_price=None, qty=D("0.001"), tif=TIF.GTC))
    assert (r.symbol, r.qty, r.type) == ("BTC/USD", 0.001, OrderType.MARKET)
    r = od.to_alpaca_request(_req(side=Side.SELL, type=OrdType.TRAIL, limit_price=None, trail_percent=D("2")))
    assert (r.type, r.trail_percent) == (OrderType.TRAILING_STOP, 2.0)
    bad = [_req(instrument=EUR), _req(type=OrdType.MARKET, limit_price=None, extended_hours=True),
           _req(tif=TIF.GTC, extended_hours=True), _req(instrument=BTC, tif=TIF.DAY), _req(limit_price=D("500.255"))]
    for b in bad:
        with pytest.raises(UnsupportedOrder):
            od.to_alpaca_request(b)


@settings(max_examples=150, deadline=None)
@given(t=st.sampled_from(list(OrdType)), tif=st.sampled_from(list(TIF)), ext=st.booleans(),
       qty=st.integers(1, 1000), cents=st.integers(1, 100_000), frac=st.sampled_from(["", "5"]),
       crypto=st.booleans())
def test_mapping_never_changes_meaning(t, tif, ext, qty, cents, frac, crypto):
    """Every canonical order maps to an equivalent broker order or raises UnsupportedOrder."""
    px = D(f"{cents // 100}.{cents % 100:02d}{frac}")
    req = OrderRequest(BTC if crypto else SPY, Side.BUY, D(qty), t, limit_price=px, stop_price=px,
                       trail_percent=D("1"), tif=tif, extended_hours=ext)
    try:
        r = od.to_alpaca_request(req)
    except UnsupportedOrder:
        return
    assert r.qty == qty and r.client_order_id == req.client_order_id
    assert r.time_in_force.value == tif.value.lower()
    if t in (OrdType.LIMIT, OrdType.STOP_LIMIT):
        assert D(str(r.limit_price)) == px
    if t in (OrdType.STOP, OrdType.STOP_LIMIT):
        assert D(str(r.stop_price)) == px


# ------------------------------------------------------------------- state machine
def test_status_mapping():
    assert od.map_ib_status("PreSubmitted") is S.ACCEPTED
    assert od.map_ib_status("Submitted", filled=3) is S.PARTIALLY_FILLED
    assert od.map_ib_status("Inactive") is S.REJECTED and od.map_ib_status("ApiCancelled") is S.CANCELLED
    assert od.map_alpaca_status("accepted") is S.PENDING_NEW and od.map_alpaca_status("new") is S.ACCEPTED
    assert od.map_alpaca_status("partial_fill") is S.PARTIALLY_FILLED and od.map_alpaca_status("done_for_day") is S.EXPIRED
    with pytest.raises(ValueError):
        od.map_ib_status("Weird")
    with pytest.raises(ValueError):
        od.map_alpaca_status("weird")


def test_tracker_happy_path_with_partials_and_duplicates():
    tr = od.OrderTracker()
    req = _req(qty=D("10"))
    tr.track(req)
    with pytest.raises(DuplicateOrder):
        tr.track(req)
    tr.on_status(req.client_order_id, S.ACCEPTED)
    tr.on_status(req.client_order_id, S.ACCEPTED)                        # duplicate status: ignored
    assert tr.on_fill(req.client_order_id, "e1", D("4"), D("500"))
    assert not tr.on_fill(req.client_order_id, "e1", D("4"), D("500"))   # duplicate execution
    assert tr.orders[req.client_order_id].state is S.PARTIALLY_FILLED
    tr.on_fill(req.client_order_id, "e2", D("6"), D("501"))
    t = tr.orders[req.client_order_id]
    assert t.state is S.FILLED and t.filled == 10 and t.avg_price == D("500.6")
    assert t.history == [S.PENDING_NEW, S.ACCEPTED, S.PARTIALLY_FILLED, S.FILLED]
    tr.on_status(req.client_order_id, S.PARTIALLY_FILLED)                # late status after FILLED
    assert t.state is S.FILLED and tr.anomalies == [("illegal transition", req.client_order_id, "FILLED->PARTIALLY_FILLED")]
    assert tr.position("SPY") == D("10")


def test_tracker_cancel_race_and_unknown():
    tr = od.OrderTracker()
    buy, sell = _req(client_order_id="b"), _req(client_order_id="s", side=Side.SELL, qty=D("3"))
    tr.track(buy), tr.track(sell)
    tr.on_status("b", S.ACCEPTED)
    tr.on_status("b", S.PENDING_CANCEL)
    tr.on_fill("b", "x1", D("10"), D("500"))                             # fill raced the cancel: legal
    assert tr.orders["b"].state is S.FILLED
    tr.on_status("s", S.CANCELLED)
    tr.on_fill("s", "x2", D("1"), D("502"))                              # fill after cancel: counted + anomaly
    assert tr.orders["s"].filled == 1 and ("fill after terminal", "s", "CANCELLED") in tr.anomalies
    tr.on_status("b", S.FILLED)                                          # duplicate status, no anomaly
    tr.track(_req(client_order_id="ib", qty=D("5")))
    tr.on_status("ib", S.FILLED)                                         # IB: status before executions
    tr.on_fill("ib", "x3", D("2"), D("500")), tr.on_fill("ib", "x4", D("3"), D("500"))
    assert tr.orders["ib"].state is S.FILLED and tr.orders["ib"].filled == 5
    assert not [a for a in tr.anomalies if a[1] == "ib"]
    tr.on_fill("ib", "x5", D("1"), D("500"))
    assert ("overfill", "ib") in tr.anomalies
    tr.on_status("ghost", S.ACCEPTED)
    assert ("unknown order", "ghost") in tr.anomalies
    assert tr.position("SPY") == D("15")


def test_reconcile():
    rep = od.reconcile({"a": D("10"), "b": D("5"), "c": D("1")}, {"b": D("5"), "c": D("2"), "z": D("7")},
                       {"SPY": D("100"), "QQQ": D("0")}, {"SPY": D("90"), "IWM": D("-5")})
    assert rep.only_internal == ["a"] and rep.only_broker == ["z"]
    assert rep.qty_mismatch == [("c", D("1"), D("2"))]
    assert rep.position_mismatch == [("IWM", D("0"), D("-5")), ("SPY", D("100"), D("90"))]
    assert not rep.ok
    assert od.reconcile({}, {}, {"SPY": D("0")}, {}).ok
