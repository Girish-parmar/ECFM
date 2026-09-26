from datetime import datetime, timezone
from decimal import Decimal as D

import pytest
from hypothesis import given, settings, strategies as st
from _loader import load

dm = load("week09_domain", "domain")
SPY = dm.Instrument("SPY", dm.AssetClass.EQUITY)
ES = dm.Future("ES", dm.AssetClass.FUTURE, tick=D("0.25"), multiplier=D("50"), expiry="202612")


def test_money():
    assert dm.Money(D("10"), "USD") + dm.Money(D("2.5"), "USD") == dm.Money(D("12.5"), "USD")
    assert dm.Money(D("10")) - dm.Money(D("2.5")) == dm.Money(D("7.5"))
    assert repr(dm.Money(D("12.5"), "EUR")) == "Money(12.50 EUR)"
    with pytest.raises(dm.CurrencyMismatch):
        dm.Money(D("1"), "USD") + dm.Money(D("1"), "EUR")
    with pytest.raises(dm.TradingError):
        dm.Money(D("1"), "USD") - dm.Money(D("1"), "JPY")


def test_instrument_is_a_value_object():
    assert dm.Instrument("SPY", dm.AssetClass.EQUITY) == SPY
    assert len({SPY, dm.Instrument("SPY", dm.AssetClass.EQUITY)}) == 1
    with pytest.raises(Exception):
        SPY.symbol = "QQQ"                                   # frozen
    for bad in [dict(symbol=""), dict(currency="US"), dict(tick=D("0")), dict(multiplier=D("-1"))]:
        with pytest.raises(ValueError):
            dm.Instrument(**{"symbol": "X", "asset_class": dm.AssetClass.EQUITY, **bad})


def test_bar_validation():
    ts = datetime(2025, 3, 3, 14, 30, tzinfo=timezone.utc)
    dm.Bar(ts, D("10"), D("11"), D("9.5"), D("10.5"), D("100"))
    with pytest.raises(ValueError):
        dm.Bar(ts.replace(tzinfo=None), D("10"), D("11"), D("9.5"), D("10.5"), D("100"))
    with pytest.raises(ValueError):
        dm.Bar(ts, D("10"), D("10.2"), D("9.5"), D("10.5"), D("100"))      # high < close
    with pytest.raises(ValueError):
        dm.Bar(ts, D("10"), D("11"), D("9.5"), D("10.5"), D("-1"))


def test_fifo_reduce_and_flip():
    p = dm.Position(SPY)
    p.apply(dm.Fill(SPY, D("100"), D("10"))); p.apply(dm.Fill(SPY, D("100"), D("12")))
    assert p.qty == 200 and p.avg_price == 11
    p.apply(dm.Fill(SPY, D("-150"), D("13")))               # FIFO: 100 @ 10 (+300), 50 @ 12 (+50)
    assert p.qty == 50 and p.realized == D("350")
    p.apply(dm.Fill(SPY, D("-80"), D("9")))                 # 50 @ 12 closed at 9 (-150), short 30 @ 9
    assert p.qty == -30 and p.avg_price == 9 and p.realized == D("200")
    assert p.unrealized(D("8")) == D("30")


def test_futures_multiplier_and_fees():
    f = dm.Position(ES)
    f.apply(dm.Fill(ES, D("2"), D("6000"), fee=D("4.2")))
    f.apply(dm.Fill(ES, D("-2"), D("6010.25"), fee=D("4.2")))
    assert f.qty == 0 and f.realized == D("1025") - D("8.4")
    with pytest.raises(ValueError):
        f.apply(dm.Fill(SPY, D("1"), D("1")))


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.integers(-50, 50).filter(lambda x: x != 0), st.integers(90, 110)),
                min_size=1, max_size=40))
def test_pnl_reconciles_with_cash(trades):
    """realized + unrealized == cash P&L for ANY sequence of fills (exact Decimal equality)."""
    p, cash = dm.Position(SPY), D("0")
    for q, px in trades:
        p.apply(dm.Fill(SPY, D(q), D(px)))
        cash -= D(q) * D(px)
    assert p.realized + p.unrealized(D("100")) == cash + p.qty * D("100")
