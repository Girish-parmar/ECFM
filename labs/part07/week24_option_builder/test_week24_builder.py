import numpy as np
import pytest
from _loader import load
from common import bsm_greeks, bsm_price

ob = load("week24_option_builder", "builder")
T = 30 / 365


def plan_condor():
    return (ob.OptionStrategy("iron condor").add(-1, 90, T, 1, 0.24).add(-1, 95, T, -1, 0.22)
            .add(1, 105, T, -1, 0.19).add(1, 110, T, 1, 0.18))


def test_builder_is_chainable_and_values_legs():
    s = ob.OptionStrategy("x")
    assert s.add(1, 100, T, 1) is s and len(s.legs) == 1 and s.legs[0] == ob.Leg(1, 100, T, 1, 0.2)
    c = plan_condor()
    want = 100 * (bsm_price(100, 90, T, 0.04, 0, 0.24, -1) - bsm_price(100, 95, T, 0.04, 0, 0.22, -1)
                  - bsm_price(100, 105, T, 0.04, 0, 0.19, 1) + bsm_price(100, 110, T, 0.04, 0, 0.18, 1))
    assert c.value(100) == pytest.approx(want)
    np.testing.assert_allclose(c.value(np.array([80.0, 100.0, 120.0]), dt=T), [-500, 0, -500])   # at expiry


def test_plan_example_numbers():
    a = plan_condor().analyze(100)
    assert a["cost"] == pytest.approx(-104.58, abs=0.01)             # a credit of about $105
    np.testing.assert_allclose(a["breakevens"], [94.0, 106.05], atol=0.06)
    assert a["max_profit"] == pytest.approx(104.58, abs=0.01) and a["max_loss"] == pytest.approx(-395.42, abs=0.01)
    assert a["pop"] == pytest.approx(0.689, abs=0.005)


def test_greeks_units_and_defined_risk():
    c = plan_condor()
    g = c.greeks(100)
    want_vega = 100 * sum(q * bsm_greeks(100, K, T, 0.04, 0, iv, cp)["vega"] / 100
                          for cp, K, q, iv in [(-1, 90, 1, .24), (-1, 95, -1, .22), (1, 105, -1, .19), (1, 110, 1, .18)])
    assert g["vega"] == pytest.approx(want_vega) and g["vega"] < 0 and g["theta"] > 0 and g["gamma"] < 0
    assert c.is_defined_risk()
    naked = ob.OptionStrategy("short strangle").add(-1, 95, T, -1).add(1, 105, T, -1)
    assert not naked.is_defined_risk()
    covered = ob.OptionStrategy("covered call").add(0, 0, 0, 1).add(1, 105, T, -1)
    assert covered.is_defined_risk() and covered.greeks(100)["delta"] < 100


def test_templates():
    iv = lambda K: 0.20 + 0.3 * (100 - K) / 100                        # noqa: E731  put skew
    strikes = np.arange(70, 131, 1.0)
    ic = ob.iron_condor(100, T, iv, strikes, short_delta=0.16, wing=5)
    Ks = [L.K for L in ic.legs]
    assert [L.qty for L in ic.legs] == [1, -1, -1, 1] and Ks[1] - Ks[0] == 5 and Ks[3] - Ks[2] == 5
    assert bsm_greeks(100, Ks[1], T, 0.04, 0, iv(Ks[1]), -1)["delta"] == pytest.approx(-0.16, abs=0.03)
    assert ic.is_defined_risk() and ic.analyze(100)["cost"] < 0
    assert ob.vertical(1, 100, 105, T, iv).name == "bull call spread" and ob.vertical(-1, 100, 95, T, iv).name == "bear put spread"
    assert ob.vertical(-1, 95, 100, T, iv).name == "bull put spread" and ob.vertical(1, 105, 100, T, iv).name == "bear call spread"
    bcs = ob.vertical(1, 100, 105, T, iv).analyze(100)
    assert bcs["max_profit"] == pytest.approx(500 - bcs["cost"], abs=1) and bcs["max_loss"] == pytest.approx(-bcs["cost"], abs=1)
    col = ob.collar(90, 110, T, iv)
    assert col.is_defined_risk() and col.analyze(100)["max_loss"] > -1500   # floor near 90 (plus/minus the net premium)


def test_calendar_is_valued_with_time_left_on_the_far_leg():
    cal = ob.calendar(100, T, 60 / 365, 1, 0.20, 0.20)
    at_near = cal.payoff_at_first_expiry(np.array([100.0]))
    far_left = 100 * bsm_price(100, 100, 30 / 365, 0.04, 0, 0.20, 1)
    assert at_near[0] == pytest.approx(far_left)                        # short leg expired worthless at the strike
    assert cal.analyze(100)["cost"] > 0 and cal.greeks(100)["vega"] > 0


def test_iv_rank_percentile_and_selector():
    hist = np.concatenate([np.linspace(0.15, 0.35, 251), [0.20]])
    assert ob.iv_rank(hist) == pytest.approx(25.0)
    assert ob.iv_percentile(hist) == pytest.approx(np.mean(hist[:-1] < 0.20) * 100)
    assert ob.iv_rank(np.full(10, 0.2)) == 50
    sel = ob.select_structure
    assert [sel(1, 10), sel(1, 50), sel(1, 90)] == ["long_call", "bull_call_spread", "bull_put_spread"]
    assert [sel(-1, 10), sel(-1, 50), sel(-1, 90)] == ["long_put", "bear_put_spread", "bear_call_spread"]
    assert [sel(0, 60), sel(0, 20, term_slope=0.01), sel(0, 20, term_slope=-0.01)] == ["iron_condor", "calendar", "no_trade"]
    assert sel(1, 90, event=True, implied_move=0.04, expected_move=0.06) == "long_straddle"
    assert sel(1, 90, event=True, implied_move=0.08, expected_move=0.06) == "bull_put_spread"


def test_combo_prices_and_orders():
    p = ob.combo_prices([1, -1, -1, 1], [0.50, 1.10, 0.95, 0.40], [0.55, 1.20, 1.05, 0.45])
    assert p["mid"] == pytest.approx(0.525 - 1.15 - 1.00 + 0.425) and p["natural"] == pytest.approx(0.55 - 1.10 - 0.95 + 0.45)
    assert p["leg_cost"] == pytest.approx(0.025 + 0.05 + 0.05 + 0.025)   # the four half-spreads you give away
    c = plan_condor()
    bag = ob.to_ib_combo(c, [11, 12, 13, 14], "SPY")
    assert bag.secType == "BAG" and [(leg.conId, leg.action, leg.ratio) for leg in bag.comboLegs] == \
        [(11, "BUY", 1), (12, "SELL", 1), (13, "SELL", 1), (14, "BUY", 1)]
    with pytest.raises(ValueError):
        ob.to_ib_combo(ob.collar(90, 110, T, lambda K: 0.2), [1, 2], "SPY")
    order = ob.to_alpaca_mleg(ob.collar(90, 110, T, lambda K: 0.2), ["SPYP", "SPYC"], 0.123)
    assert order["limit_price"] == 0.12 and [leg["side"] for leg in order["legs"]] == ["buy", "sell"]
