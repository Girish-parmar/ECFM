from datetime import date

import numpy as np
import pytest
from _loader import load
from common import load_chain

cm = load("clinic_w1_chain_vs_broker", "compare")
CHAIN, META = load_chain()


def test_prepare_converts_units_and_filters():
    c = cm.prepare(CHAIN, date(2026, 3, 2))
    assert (c["bid"] > 0).all() and len(c) < len(CHAIN)
    row = c.iloc[0]
    assert row["ib_vega_raw"] == pytest.approx(row["ib_vega"] * 100) and row["ib_theta_raw"] == pytest.approx(row["ib_theta"] * 365)
    assert row["T"] == pytest.approx(30 / 365) and row["mid"] == pytest.approx((row["bid"] + row["ask"]) / 2)


def test_implied_rate_forward_on_clean_prices():
    pr = cm.pr
    K = np.arange(450.0, 551, 10)
    iv = 0.2 - 0.3 * np.log(K / 500)
    C, P = pr.bsm_price(500, K, 0.25, 0.05, 0.02, iv, 1), pr.bsm_price(500, K, 0.25, 0.05, 0.02, iv, -1)
    r, F = cm.implied_rate_forward(K, C, P, 0.25)
    assert r == pytest.approx(0.05, abs=1e-10) and F == pytest.approx(500 * np.exp(0.03 * 0.25), rel=1e-12)


def test_differences_are_inputs_not_bugs():
    before, inputs, after, after_otm = cm.compare()
    assert list(before.columns) == ["d_iv", "d_delta", "d_vega", "d_theta"] and len(before) == 3
    assert (before["d_iv"] > 0.02).all()                               # guessed r and q: visible differences
    assert inputs["r"].to_numpy() == pytest.approx(0.045, abs=2e-3)   # the broker's inputs, recovered from prices
    assert inputs["q"].to_numpy() == pytest.approx(0.013, abs=2e-3)
    assert (after["d_iv"] < before["d_iv"]).all()
    assert (after_otm["d_iv"] < 1e-4).all() and (after_otm["d_delta"] < 1e-4).all()
    assert (after_otm["d_vega"] < 0.05).all()                          # vega per 1.00 σ (library units)
