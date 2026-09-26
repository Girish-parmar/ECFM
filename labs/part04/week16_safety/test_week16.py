import asyncio
import contextlib
import json
from datetime import date
from decimal import Decimal as D

import pytest
from _loader import load
from common import (AssetClass, DuplicateOrder, Instrument, KillSwitchTripped, OrderRequest, OrderState, OrdType,
                    Side, UnsupportedOrder)

sf = load("week16_safety", "safety")
S = OrderState
SPY = Instrument("SPY", AssetClass.EQUITY)
QQQ = Instrument("QQQ", AssetClass.EQUITY)
run = asyncio.run


# ------------------------------------------------------------------ futures & FX
def test_futures_helpers():
    assert sf.tick_value(D("0.25"), D("50")) == D("12.50")
    assert sf.tick_value(D("0.25"), D("5")) == D("1.25")                   # MES
    assert sf.futures_pnl(D("5000"), D("5010.25"), D("2"), D("50")) == D("1025.00")
    assert sf.futures_pnl(D("80"), D("78.5"), D("-1"), D("1000")) == D("1500.0")   # short CL
    assert sf.third_friday(2026, 12) == date(2026, 12, 18)
    assert sf.third_friday(2027, 3) == date(2027, 3, 19)
    assert sf.third_friday(2026, 5) == date(2026, 5, 15)                  # month starting on a Friday
    assert sf.roll_date("202612") == date(2026, 12, 8)
    assert sf.roll_date("202612", 1) == date(2026, 12, 17)
    assert sf.roll_date("202703", 5) == date(2027, 3, 12)


def test_pip_value():
    assert sf.pip_value_usd("EURUSD", D("25000"), D("1")) == D("2.5000")
    assert sf.pip_value_usd("USDJPY", D("100000"), D("1") / D("150")) == pytest.approx(D("6.6667"), abs=D("0.0001"))


# ------------------------------------------------------------------ pre-trade checks
def _ctx(**kw):
    base = dict(tradable={"SPY", "QQQ", "GME"}, last={"SPY": D("500"), "QQQ": D("400"), "GME": D("20")},
                buying_power=D("100000"), shortable={"SPY"}, position={"QQQ": D("10")})
    return sf.TradeContext(**{**base, **kw})


def _order(sym="SPY", side=Side.BUY, qty="10", px="500", **kw):
    return OrderRequest(Instrument(sym, AssetClass.EQUITY), side, D(qty), OrdType.LIMIT if px else OrdType.MARKET,
                        limit_price=D(px) if px else None, **kw)


@pytest.mark.parametrize("order, ctx, reason", [
    (_order(), _ctx(), None),
    (_order(), _ctx(kill_switch_tripped=True), "kill switch tripped"),
    (_order("TSLA"), _ctx(), "not tradable"),
    (_order(), _ctx(session_open=False), "session closed"),
    (_order(extended_hours=True), _ctx(session_open=False), None),
    (_order(), _ctx(last={}), "no last price"),
    (_order(px="550"), _ctx(), "fat finger"),
    (_order(px="526"), _ctx(), "fat finger"),
    (_order(px="524"), _ctx(), None),
    (_order(qty="5000"), _ctx(), "qty above max"),
    (_order(qty="201"), _ctx(), "insufficient buying power"),
    (_order(px=None, qty="201"), _ctx(), "insufficient buying power"),
    (_order("QQQ", Side.SELL, "10", "400"), _ctx(), None),                 # closing a long: fine
    (_order("QQQ", Side.SELL, "11", "400"), _ctx(), "not shortable"),       # would go short
    (_order("SPY", Side.SELL, "10", "500"), _ctx(), None),                  # shortable
    (_order("GME", Side.SELL, "1", "20"), _ctx(kill_switch_tripped=True), "kill switch tripped"),
])
def test_pretrade_check(order, ctx, reason):
    assert sf.pretrade_check(order, ctx) == reason


# ------------------------------------------------- contract tests every adapter must pass
@pytest.fixture
def adapter():
    """Add your IB and Alpaca paper adapters here in quantforge (marked `paper`, never run in CI)."""
    a = sf.SimBroker({"SPY": D("500"), "QQQ": D("400")})
    run(a.connect())
    yield a
    with contextlib.suppress(NotImplementedError):            # starter: the test itself already failed
        run(a.cancel_all())
    run(a.disconnect())


def _req(inst=SPY, side=Side.BUY, qty="10", t=OrdType.LIMIT, lmt="450", stop=None, coid=None):
    kw = {"client_order_id": coid} if coid else {}
    return OrderRequest(inst, side, D(qty), t, limit_price=D(lmt) if lmt else None, stop_price=D(stop) if stop else None, **kw)


def test_contract_submit_and_cancel_far_limit(adapter):
    r = _req()
    assert run(adapter.submit(r)) == r.client_order_id
    assert adapter.state(r.client_order_id) is S.ACCEPTED
    run(adapter.cancel(r.client_order_id))
    assert adapter.state(r.client_order_id) is S.CANCELLED
    run(adapter.cancel(r.client_order_id))                                  # cancelling twice is harmless
    run(adapter.cancel("never-existed"))


def test_contract_duplicate_client_order_id_is_not_sent_twice(adapter):
    r = _req()
    run(adapter.submit(r))
    with pytest.raises(DuplicateOrder):
        run(adapter.submit(r))
    run(adapter.cancel(r.client_order_id))
    with pytest.raises(DuplicateOrder):                                     # even after it is gone
        run(adapter.submit(r))


def test_contract_unsupported_order_type(adapter):
    with pytest.raises(UnsupportedOrder):
        run(adapter.submit(_req(t=OrdType.STOP_LIMIT, stop="455")))


def test_sim_matching():
    b = sf.SimBroker({"SPY": D("500")})
    m = _req(t=OrdType.MARKET, lmt=None)
    run(b.submit(m))
    assert b.state(m.client_order_id) is S.FILLED and b.positions["SPY"] == 10 and b.fills[-1].price == 500
    lim = _req(side=Side.SELL, qty="4", lmt="505")
    stp = _req(side=Side.SELL, qty="6", t=OrdType.STOP, lmt=None, stop="495")
    run(b.submit(lim)), run(b.submit(stp))
    assert set(b.open) == {lim.client_order_id, stp.client_order_id}
    run(b.on_price("SPY", D("506")))                                        # limit sell fills AT its limit
    assert b.state(lim.client_order_id) is S.FILLED and b.fills[-1].price == D("505") and b.positions["SPY"] == 6
    run(b.on_price("SPY", D("494")))                                        # stop triggers, fills at last
    assert b.state(stp.client_order_id) is S.FILLED and b.fills[-1].price == D("494") and b.is_flat()


def test_sim_cancel_all_and_flatten():
    b = sf.SimBroker({"SPY": D("500"), "QQQ": D("400")})
    run(b.submit(_req(t=OrdType.MARKET, lmt=None)))
    run(b.submit(_req(QQQ, Side.SELL, "3", OrdType.MARKET, lmt=None)))
    for i in range(3):
        run(b.submit(_req(lmt=str(400 + i))))
    assert run(b.cancel_all()) == 3 and not b.open
    assert run(b.flatten_all()) == 2 and b.is_flat()
    assert run(b.flatten_all()) == 0


# ------------------------------------------------------------------------- kill switch
class BrokenBroker(sf.SimBroker):
    async def cancel_all(self):
        raise ConnectionError("gateway down")


def _loaded_broker():
    b = sf.SimBroker({"SPY": D("500"), "QQQ": D("400")})
    run(b.submit(_req(t=OrdType.MARKET, lmt=None)))
    run(b.submit(_req(QQQ, Side.SELL, "5", OrdType.MARKET, lmt=None)))
    for i in range(5):
        run(b.submit(_req(lmt=str(450 + i))))
    return b


def test_kill_switch_flattens_and_is_sticky(tmp_path):
    a, b = _loaded_broker(), _loaded_broker()
    notes = []

    async def notify(msg):
        notes.append(msg)

    ks = sf.KillSwitch([a, b], tmp_path / "ks.json", notify=notify)
    ks.guard()                                                              # not tripped: no error
    elapsed = run(ks.trip("daily loss limit"))
    assert 0 <= elapsed < 5 and a.is_flat() and b.is_flat()
    assert notes and "daily loss limit" in notes[0]
    with pytest.raises(KillSwitchTripped, match="daily loss limit"):
        ks.guard()
    again = sf.KillSwitch([a], tmp_path / "ks.json")                         # "after a restart"
    assert again.tripped
    run(again.trip("second reason"))                                        # idempotent, keeps first reason
    assert json.loads((tmp_path / "ks.json").read_text())["reason"] == "daily loss limit"
    with pytest.raises(ValueError):
        again.reset("  ")
    again.reset("alice")
    assert not ks.tripped
    ks.guard()
    audit = [json.loads(l) for l in (tmp_path / "ks.audit.jsonl").read_text().splitlines()]
    assert [e["event"] for e in audit] == ["trip", "trip", "reset"] and audit[2]["operator"] == "alice"


def test_kill_switch_survives_a_failing_broker(tmp_path):
    ok, broken = _loaded_broker(), BrokenBroker({"SPY": D("500")})
    ks = sf.KillSwitch([broken, ok], tmp_path / "ks.json")
    run(ks.trip("stale data"))
    assert ok.is_flat()
    audit = json.loads((tmp_path / "ks.audit.jsonl").read_text().splitlines()[0])
    assert audit["errors"] == ["gateway down"]


def test_kill_switch_without_flatten_keeps_positions(tmp_path):
    b = _loaded_broker()
    run(sf.KillSwitch([b], tmp_path / "ks.json").trip("manual", flatten=False))
    assert not b.open and b.positions["SPY"] == 10
