import numpy as np
import pytest
from _loader import load

nm = load("week22_greeks_numerics", "numerics")
pr = nm.pr
BASE = (100.0, 105.0, 0.4, 0.04, 0.015, 0.27)
CASES = [(*BASE, 1), (*BASE, -1), (100.0, 90.0, 0.1, 0.05, 0.0, 0.4, -1), (50.0, 50.0, 1.5, 0.02, 0.03, 0.15, 1)]


def fd(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2 * h)


@pytest.mark.parametrize("S, K, T, r, q, v, cp", CASES)
def test_second_order_greeks_match_finite_differences(S, K, T, r, q, v, cp):
    so = nm.second_order(S, K, T, r, q, v, cp)
    g = lambda **kw: pr.greeks(**{**dict(S=S, K=K, T=T, r=r, q=q, sigma=v, cp=cp), **kw})  # noqa: E731
    vol = lambda **kw: nm.second_order(**{**dict(S=S, K=K, T=T, r=r, q=q, sigma=v, cp=cp), **kw})["volga"]  # noqa: E731
    ref = {
        "vanna": fd(lambda x: g(sigma=x)["delta"], v, 1e-5),
        "volga": fd(lambda x: g(sigma=x)["vega"], v, 1e-5),
        "charm": -fd(lambda x: g(T=x)["delta"], T, 1e-6),
        "speed": fd(lambda x: g(S=x)["gamma"], S, 1e-4 * S),
        "zomma": fd(lambda x: g(sigma=x)["gamma"], v, 1e-5),
        "color": -fd(lambda x: g(T=x)["gamma"], T, 1e-6),
        "veta": -fd(lambda x: g(T=x)["vega"], T, 1e-6),
        "ultima": fd(lambda x: vol(sigma=x), v, 1e-5),
    }
    for k, want in ref.items():
        assert so[k] == pytest.approx(want, rel=1e-4, abs=1e-8), k


@pytest.mark.parametrize("S, K, T, r, q, v, cp", CASES)
def test_bump_and_revalue_matches_closed_forms(S, K, T, r, q, v, cp):
    f = nm.greeks_fd(pr.bsm_price, S, K, T, r, q, v, cp)
    g, so = pr.greeks(S, K, T, r, q, v, cp), nm.second_order(S, K, T, r, q, v, cp)
    for k in ("delta", "gamma", "vega", "theta", "rho"):
        assert f[k] == pytest.approx(g[k], rel=1e-4, abs=1e-7), k
    assert f["vanna"] == pytest.approx(so["vanna"], rel=1e-3, abs=1e-6)
    assert f["volga"] == pytest.approx(so["volga"], rel=1e-3, abs=1e-5)


def test_crr_converges_and_prices_early_exercise():
    S, K, T, r, q, v = BASE
    eu_call = pr.bsm_price(S, K, T, r, q, v, 1)
    assert nm.crr_price(S, K, T, r, q, v, 1, 2000, american=False) == pytest.approx(eu_call, abs=2e-3)
    am_put = nm.crr_price(S, K, T, r, q, v, -1, 1000)
    eu_put = pr.bsm_price(S, K, T, r, q, v, -1)
    assert 0.12 < am_put - eu_put < 0.18                          # early-exercise premium of the put
    assert abs(nm.crr_price(S, K, T, r, q, v, 1, 1000) - eu_call) < 0.005   # call ~ European (small dividend)
    assert nm.crr_price(S, K, T, 0.10, 0.0, v, -1, 500) - pr.bsm_price(S, K, T, 0.10, 0.0, v, -1) > \
        am_put - eu_put                                          # higher rates: more valuable to exercise the put


def test_monte_carlo_is_within_its_standard_error():
    S, K, T, r, q, v = BASE
    for cp in (1, -1):
        price, se = nm.mc_european(S, K, T, r, q, v, cp, 200_000, seed=1)
        assert abs(price - pr.bsm_price(S, K, T, r, q, v, cp)) < 3 * se and 0 < se < 0.03


def test_common_random_numbers_make_mc_greeks_usable():
    S, K, T, r, q, v = BASE
    true = pr.greeks(S, K, T, r, q, v, 1)["delta"]
    crn = [nm.mc_delta(S, K, T, r, q, v, 1, seed=s) for s in range(5)]
    indep = [nm.mc_delta(S, K, T, r, q, v, 1, seed=s, common_random_numbers=False) for s in range(5)]
    assert max(abs(np.array(crn) - true)) < 0.01
    assert np.std(indep) > 20 * np.std(crn)                        # independent seeds: noise swamps the bump


@pytest.mark.parametrize("S, K", [(100.0, 105.0), (100.0, 100.0), (90.0, 100.0), (117.0, 100.0)])
def test_crank_nicolson_second_order(S, K):
    T, r, q, v = 0.4, 0.04, 0.015, 0.27
    ref = pr.bsm_price(S, K, T, r, q, v, 1)
    e200 = nm.crank_nicolson_call(S, K, T, r, q, v, 200, 200) - ref
    e800 = nm.crank_nicolson_call(S, K, T, r, q, v, 800, 800) - ref
    assert abs(e800) < 2e-4 and abs(e800) <= abs(e200) / 8        # 4× finer grid: error ÷ ~16
