from datetime import datetime, timezone

import numpy as np
import pytest
from _loader import load
from common import load_golden

pr = load("week21_pricing_iv", "pricing")
G = load_golden("vollib")
ARGS = (100.0, G["K"], G["T"], G["r"], G["q"], G["sigma"], G["cp"])


def test_year_fraction():
    now = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
    exp = datetime(2026, 4, 1, 20, 0, tzinfo=timezone.utc)
    assert pr.year_fraction(now, exp) == pytest.approx((30 + 5.5 / 24) / 365)


def test_price_matches_vollib():
    np.testing.assert_allclose(pr.bsm_price(*ARGS), G["price"], rtol=1e-10, atol=1e-12)


def test_black76_matches_vollib():
    np.testing.assert_allclose(pr.black76_price(100.0, G["K"], G["T"], G["r"], G["sigma"], G["cp"]), G["black76"],
                               rtol=1e-10, atol=1e-12)


def test_greeks_match_vollib_after_unit_conversion():
    raw = pr.greeks(*ARGS)
    disp = pr.to_display(raw)
    for k in ("delta", "gamma", "vega", "theta", "rho"):
        np.testing.assert_allclose(disp[k], G[k], rtol=1e-9, atol=1e-12, err_msg=k)
    assert raw["vega"] is not disp["vega"] and np.allclose(raw["vega"], disp["vega"] * 100)   # g left unchanged


def test_put_call_parity():
    S, K, T, r, q, s = 100.0, np.array([80.0, 100, 120]), 0.5, 0.04, 0.02, 0.25
    gap = pr.parity_gap(pr.bsm_price(S, K, T, r, q, s, 1), pr.bsm_price(S, K, T, r, q, s, -1), S, K, T, r, q)
    np.testing.assert_allclose(gap, 0, atol=1e-12)


def test_implied_forward_recovers_the_dividend():
    S, T, r, q = 500.0, 0.25, 0.045, 0.013
    K = np.arange(450.0, 551.0, 5)
    iv = 0.18 - 0.2 * np.log(K / S)                         # a skew does not matter for parity
    C, P = pr.bsm_price(S, K, T, r, q, iv, 1), pr.bsm_price(S, K, T, r, q, iv, -1)
    F = pr.implied_forward(K, C, P, T, r)
    assert F == pytest.approx(S * np.exp((r - q) * T), rel=1e-12)
    assert r - np.log(F / S) / T == pytest.approx(q, abs=1e-10)          # implied dividend yield


def test_bounds_and_iv_round_trip():
    lo, hi = pr.price_bounds(100, 90, 1.0, 0.05, 0.0, 1)
    assert lo == pytest.approx(100 - 90 * np.exp(-0.05)) and hi == 100
    assert np.isnan(pr.implied_vol(5.0, 100, 90, 1.0, 0.05, 0.0, 1))    # below intrinsic
    assert np.isnan(pr.implied_vol(100.0, 100, 90, 1.0, 0.05, 0.0, 1))  # at the upper bound
    lower = np.maximum(G["cp"] * (100 * np.exp(-G["q"] * G["T"]) - G["K"] * np.exp(-G["r"] * G["T"])), 0)
    ok = ~np.isnan(G["iv"]) & (G["price"] - lower > 1e-4)   # no time value -> IV is not identifiable
    ours = np.array([pr.implied_vol(p, 100.0, K, T, r, q, int(cp)) for p, K, T, r, q, cp in
                     zip(G["price"][ok], G["K"][ok], G["T"][ok], G["r"][ok], G["q"][ok], G["cp"][ok])])
    np.testing.assert_allclose(ours, G["sigma"][ok], rtol=1e-6)


def test_newton_fast_path_and_brent_fallback():
    st = {}
    p = pr.bsm_price(100, 100, 0.5, 0.03, 0.0, 0.25, 1)
    assert pr.implied_vol(p, 100, 100, 0.5, 0.03, 0.0, 1, stats=st) == pytest.approx(0.25, abs=1e-9)
    assert st["method"] == "newton"
    st = {}
    p = pr.bsm_price(100, 300, 0.02, 0.0, 0.0, 3.0, 1)      # 1 week, far OTM, 300% vol: Newton leaves (lo, hi)
    assert pr.implied_vol(p, 100, 300, 0.02, 0.0, 0.0, 1, stats=st) == pytest.approx(3.0, rel=1e-6)
    assert st["method"] == "brent"


def test_vectorized_chain_iv():
    S, T, r, q = 100.0, 0.3, 0.03, 0.01
    K = np.linspace(50, 200, 61)
    true = 0.2 + 0.3 * np.abs(np.log(K / S))
    cp = np.where(K < S, -1, 1)
    prices = pr.bsm_price(S, K, T, r, q, true, cp)
    prices[5] = 0.0                                          # a zero price has no IV
    out = pr.implied_vol_chain(prices, S, K, T, r, q, cp)
    assert np.isnan(out[5])
    good = (np.arange(K.size) != 5) & (prices > 1e-8)
    np.testing.assert_allclose(out[good], true[good], rtol=1e-6)
