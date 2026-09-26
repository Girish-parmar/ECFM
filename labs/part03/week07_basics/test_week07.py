from datetime import date, datetime
from decimal import Decimal as D
from zoneinfo import ZoneInfo

import pytest
from _loader import load

ex = load("week07_basics", "exercises")


@pytest.mark.parametrize("price,tick,expected", [("101.237", "0.05", "101.25"), ("99.994", "0.01", "99.99"),
                                                 ("6000.12", "0.25", "6000.00"), ("0.125", "0.25", "0.25")])
def test_round_to_tick(price, tick, expected):
    assert ex.round_to_tick(D(price), D(tick)) == D(expected)


@pytest.mark.parametrize("shares,price,expected", [(100, "50", "1.00"), (1000, "20", "5.00"),
                                                   (10, "5", "0.50"), (-400, "30", "2.00")])
def test_ib_fixed_commission(shares, price, expected):
    assert ex.ib_fixed_commission(shares, D(price)) == D(expected)


def test_returns():
    px = [100.0, 110.0, 99.0]
    assert ex.simple_returns(px) == pytest.approx([0.10, -0.10])
    assert ex.log_returns(px) == pytest.approx([0.0953102, -0.1053605], abs=1e-7)
    assert sum(ex.log_returns(px)) == pytest.approx(__import__("math").log(99 / 100))


def test_max_drawdown():
    assert ex.max_drawdown([100, 120, 90, 130]) == pytest.approx(-0.25)
    assert ex.max_drawdown([1, 2, 3]) == 0.0


def test_route_message():
    assert ex.route_message({"type": "fill", "symbol": "SPY", "qty": 100, "price": 500.25}) == "FILL SPY 100 @ 500.25"
    assert ex.route_message({"type": "cancel", "order_id": "A1"}) == "CANCELLED A1"
    assert ex.route_message({"type": "reject", "order_id": "A1", "reason": "margin"}) == "REJECTED A1: margin"
    assert ex.route_message({"type": "heartbeat"}) == "UNKNOWN"


def test_time_zones():
    assert ex.to_utc("2025-01-15 09:30:00", "America/New_York") == datetime(2025, 1, 15, 14, 30, tzinfo=ZoneInfo("UTC"))
    assert ex.market_open_utc(date(2025, 1, 15)).hour == 14
    assert ex.market_open_utc(date(2025, 7, 15)).hour == 13
    utc = ZoneInfo("UTC")
    assert ex.is_regular_hours(datetime(2025, 7, 15, 13, 30, tzinfo=utc))       # 09:30 NY
    assert not ex.is_regular_hours(datetime(2025, 7, 15, 20, 0, tzinfo=utc))    # 16:00 NY
    assert not ex.is_regular_hours(datetime(2025, 7, 19, 15, 0, tzinfo=utc))    # Saturday
    with pytest.raises(ValueError):
        ex.is_regular_hours(datetime(2025, 7, 15, 15, 0))


def test_best_bid_ask_and_vwap():
    book = {"bids": {D("99.98"): 300, D("99.99"): 0, D("99.97"): 100}, "asks": {D("100.01"): 200, D("100.02"): 50}}
    assert ex.best_bid_ask(book) == (D("99.98"), D("100.01"))
    assert ex.best_bid_ask({"bids": {}, "asks": {}}) == (None, None)
    assert ex.vwap([(D("10"), 100), (D("11"), 300)]) == D("10.75")
    with pytest.raises(ValueError):
        ex.vwap([(D("10"), 0)])
