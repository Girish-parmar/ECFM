from collections import namedtuple
from decimal import Decimal as D

import pytest
from _loader import load
from common import AssetClass, Instrument, UnsupportedInstrument

fd = load("week13_foundations", "foundations")


def test_ports_and_env_guard():
    assert fd.ib_port("gateway", "paper") == 4002 and fd.ib_port("TWS", "Live") == 7496
    with pytest.raises(ValueError):
        fd.ib_port("gateway", "demo")
    fd.check_port_matches_env("paper", 4002)
    fd.check_port_matches_env("live", 7496)
    with pytest.raises(RuntimeError):
        fd.check_port_matches_env("paper", 4001)          # live port while we think we are on paper
    with pytest.raises(RuntimeError):
        fd.check_port_matches_env("live", 7497)
    with pytest.raises(ValueError):
        fd.check_port_matches_env("paper", 8080)


def test_find_secrets():
    text = "\n".join([
        "QF_ALPACA_KEY=PKABCDEFGHIJ1234567Z",             # 1 key id
        "QF_ALPACA_SECRET=<your-secret>",                  # 2 placeholder
        "db_password = 'hunter2hunter2'",                  # 3 assignment
        "api_key: ${ALPACA_KEY}",                          # 4 placeholder
        "token=changeme",                                  # 5 placeholder
        "print('hello world')",                            # 6
        "SECRET_TOKEN = abcdefgh12345",                    # 7 assignment
        "pk_short = PK123",                                # 8 too short for anything
    ])
    assert fd.find_secrets(text) == [(1, "alpaca_key_id"), (3, "assignment"), (7, "assignment")]


@pytest.mark.parametrize("inst, sym", [
    (Instrument("AAPL", AssetClass.EQUITY, exchange="NASDAQ"), "AAPL"),
    (Instrument("ES", AssetClass.FUTURE, exchange="CME", expiry="202612"), "ESZ6"),
    (Instrument("MES", AssetClass.FUTURE, exchange="CME", expiry="202703"), "MESH7"),
    (Instrument("EURUSD", AssetClass.FX), "EUR.USD"),
    (Instrument("BTC", AssetClass.CRYPTO), "BTC/USD"),
])
def test_canonical_symbol(inst, sym):
    assert fd.canonical_symbol(inst) == sym


def test_to_ib_contract():
    c = fd.to_ib_contract(Instrument("AAPL", AssetClass.EQUITY, exchange="NASDAQ"))
    assert (c.secType, c.symbol, c.exchange, c.primaryExchange, c.currency) == ("STK", "AAPL", "SMART", "NASDAQ", "USD")
    c = fd.to_ib_contract(Instrument("ES", AssetClass.FUTURE, exchange="CME", expiry="202612"))
    assert (c.secType, c.lastTradeDateOrContractMonth, c.exchange) == ("FUT", "202612", "CME")
    c = fd.to_ib_contract(Instrument("EURUSD", AssetClass.FX))
    assert (c.secType, c.symbol, c.currency, c.exchange) == ("CASH", "EUR", "USD", "IDEALPRO")
    c = fd.to_ib_contract(Instrument("BTC", AssetClass.CRYPTO))
    assert (c.secType, c.exchange) == ("CRYPTO", "PAXOS")
    for bad in [Instrument("ES", AssetClass.FUTURE, exchange="CME"), Instrument("SPX", AssetClass.OPTION)]:
        with pytest.raises(UnsupportedInstrument):
            fd.to_ib_contract(bad)


AV = namedtuple("AccountValue", "account tag value currency modelCode")


def test_ib_account_summary():
    rows = [AV("DU1", "NetLiquidation", "100250.50", "USD", ""), AV("DU1", "NetLiquidation", "92000", "EUR", ""),
            AV("DU1", "BuyingPower", "400000", "USD", ""), AV("DU1", "AccountType", "INDIVIDUAL", "", ""),
            AV("DU1", "TotalCashValue", "5000.25", "USD", "")]
    assert fd.ib_account_summary(rows) == {"NetLiquidation": D("100250.50"), "BuyingPower": D("400000"),
                                           "TotalCashValue": D("5000.25")}
    assert fd.ib_account_summary(rows, "EUR") == {"NetLiquidation": D("92000")}


def test_alpaca_account_warnings():
    ok = {"equity": "100000", "trading_blocked": False, "account_blocked": False, "pattern_day_trader": False,
          "daytrade_count": 0}
    assert fd.alpaca_account_warnings(ok) == []
    bad = {**ok, "equity": "20000", "account_blocked": True, "pattern_day_trader": True, "daytrade_count": 3}
    assert fd.alpaca_account_warnings(bad) == ["trading blocked", "pattern day trader", "day trades: 3", "low equity"]
