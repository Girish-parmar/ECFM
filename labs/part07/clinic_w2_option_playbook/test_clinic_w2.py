from datetime import date

import numpy as np
import pytest
from _loader import load
from common import bsm_greeks, load_chain

pb = load("clinic_w2_option_playbook", "playbook")
CHAIN, META = load_chain()
S, R, Q = META["spot"], META["r"], META["q"]


@pytest.fixture(scope="module")
def prepared():
    return pb.prepare(CHAIN, S, date(2026, 3, 2), R, Q)


def test_prepare_and_smile(prepared):
    c = prepared
    assert (c["bid"] > 0).all() and c["iv"].notna().all() and len(c) >= 90
    e0 = sorted(c["expiry"].unique())[0]
    f = pb.smile(c, e0, S)
    assert f(450) > f(500) > f(530) * 0.9 and f(300) == f(420)            # skew; flat beyond the listed range


def test_playbook_structures(prepared):
    book = pb.build_playbook(prepared, S, R, Q)
    assert list(book) == ["direction", "range", "hedge"] and all(s.is_defined_risk() for s in book.values())
    d, rng, h = book["direction"], book["range"], book["hedge"]
    assert d.name == "bull call spread" and d.analyze(S, R, Q)["cost"] > 0
    assert [L.qty for L in rng.legs] == [1, -1, -1, 1] and rng.legs[1].K - rng.legs[0].K == 20
    short_put = rng.legs[1]
    assert bsm_greeks(S, short_put.K, short_put.T, R, Q, short_put.iv, -1)["delta"] == pytest.approx(-0.16, abs=0.05)
    assert rng.analyze(S, R, Q)["cost"] < 0 and rng.analyze(S, R, Q)["pop"] > 0.6
    assert h.name == "collar" and h.legs[0].cp == 0 and h.greeks(S, R, Q)["delta"] < 100


def test_report_and_orders():
    md, orders, book = pb.build()
    heads = [line for line in md.splitlines() if line.startswith("#")]
    assert heads == ["# Option playbook", "## direction: bull call spread", "## range: iron condor", "## hedge: collar"]
    for name, st in book.items():
        o = orders[name]
        opt_legs = [L for L in st.legs if L.cp != 0]
        assert len(o["ib"].comboLegs) == len(opt_legs) == len(o["alpaca"]["legs"])
        assert o["combo"]["natural"] >= o["combo"]["mid"] and o["alpaca"]["limit_price"] == round(o["combo"]["mid"], 2)
        model = float(pb.ob.OptionStrategy("options only", opt_legs).value(S, R, Q)) / 100
        assert o["combo"]["mid"] == pytest.approx(model, abs=0.05)       # model at the chain's IVs ≈ quoted net mid
    assert orders["range"]["combo"]["mid"] < 0 and orders["direction"]["combo"]["mid"] > 0
    ids = [leg.conId for o in orders.values() for leg in o["ib"].comboLegs]
    assert ids == list(range(1001, 1001 + len(ids)))


def test_scenario_grid(prepared):
    book = pb.build_playbook(prepared, S, R, Q)
    g = pb.scenario_grid(book["range"], S, R, Q)
    assert g.loc[0.0, 0.0] == pytest.approx(0.0, abs=1e-9)
    assert g.loc[0.1, 0.0] < 0 and g.loc[-0.1, 0.0] < 0 and g.loc[0.0, 0.05] < 0       # short gamma, short vega
    assert np.isfinite(g.to_numpy()).all()
