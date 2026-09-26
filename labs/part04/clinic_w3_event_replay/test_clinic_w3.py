from decimal import Decimal as D
from pathlib import Path

import pytest
from _loader import load
from common import OrderState

rp = load("clinic_w3_event_replay", "replay")
HERE = Path(__file__).resolve().parent
ORDERS = rp.load_orders(HERE / "bracket_orders.json")


def test_normalize_event():
    e = rp.normalize_event({"broker": "ib", "type": "orderStatus", "orderRef": "x", "status": "Submitted", "filled": 3})
    assert e == {"coid": "x", "kind": "status", "state": OrderState.PARTIALLY_FILLED}
    e = rp.normalize_event({"broker": "ib", "type": "execDetails", "orderRef": "x", "execId": "e9", "shares": 3,
                            "price": 499.95})
    assert e == {"coid": "x", "kind": "fill", "exec_id": "e9", "qty": D("3"), "price": D("499.95")}
    e = rp.normalize_event({"broker": "alpaca", "event": "canceled", "order": {"client_order_id": "y"}})
    assert e == {"coid": "y", "kind": "status", "state": OrderState.CANCELLED}
    e = rp.normalize_event({"broker": "alpaca", "event": "partial_fill", "order": {"client_order_id": "y"},
                            "execution_id": "a", "qty": "2", "price": "10.5"})
    assert e["kind"] == "fill" and e["qty"] == D("2") and e["price"] == D("10.5")
    for bad in [{"broker": "tradestation"}, {"broker": "ib", "type": "tickPrice"}]:
        with pytest.raises(ValueError):
            rp.normalize_event(bad)


@pytest.mark.parametrize("log", ["events_ib.jsonl", "events_alpaca.jsonl"])
def test_replay_final_state(log):
    s = rp.summary(rp.replay(ORDERS, rp.read_jsonl(HERE / log)))
    assert s["orders"]["br-1-entry"] == {"state": "FILLED", "filled": "10", "avg_price": "499.9700"}
    assert s["orders"]["br-1-tp"] == {"state": "FILLED", "filled": "10", "avg_price": "502.0500"}
    assert s["orders"]["br-1-sl"] == {"state": "CANCELLED", "filled": "0", "avg_price": None}
    assert s["positions"] == {"SPY": "0"}


def test_anomalies_are_recorded_not_raised():
    ib = rp.summary(rp.replay(ORDERS, rp.read_jsonl(HERE / "events_ib.jsonl")))
    alp = rp.summary(rp.replay(ORDERS, rp.read_jsonl(HERE / "events_alpaca.jsonl")))
    assert ib["anomalies"] == []                                   # duplicates are silently dropped
    assert alp["anomalies"] == [["illegal transition", "br-1-entry", "PARTIALLY_FILLED->ACCEPTED"]]


def test_cli(capsys):
    code = rp.main([str(HERE / "bracket_orders.json"), str(HERE / "events_ib.jsonl"), str(HERE / "events_alpaca.jsonl")])
    assert code == 0 and "final states match" in capsys.readouterr().out
