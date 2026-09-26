import numpy as np
import pytest
from _loader import load

sp = load("week22_surface_portfolio", "surface_portfolio")
pr = sp.pr
TRUE = (0.0025, 0.02, -0.65, 0.02, 0.10)


# ---------------------------------------------------------------------------------- SVI
def test_svi_fit_recovers_known_parameters():
    k = np.linspace(-0.35, 0.25, 41)
    T = 58 / 365
    iv = np.sqrt(sp.svi_total_var(k, *TRUE) / T)
    np.testing.assert_allclose(sp.fit_svi(k, iv, T), TRUE, atol=1e-4)
    noisy = iv + np.random.default_rng(0).normal(0, 0.001, k.size)
    fit = sp.fit_svi(k, noisy, T)
    assert np.max(np.abs(np.sqrt(sp.svi_total_var(k, *fit) / T) - iv)) < 0.003


def test_butterfly_condition():
    k = np.linspace(-1, 1, 201)
    assert (sp.butterfly_g(k, *TRUE) >= 0).all()
    bad = (-0.02, 0.4, -0.95, 0.0, 0.01)            # very steep wings with a near-zero ATM variance
    assert (sp.butterfly_g(k, *bad) < 0).any()


def test_calendar_violations_and_interpolation():
    k = np.linspace(-0.3, 0.2, 11)
    good = {30 / 365: np.array([0.0012, 0.012, -0.7, 0.02, 0.08]), 58 / 365: np.array(TRUE)}
    assert sp.calendar_violations(k, good) == []
    bad = {30 / 365: np.array([0.004, 0.012, -0.7, 0.02, 0.08]), 58 / 365: np.array(TRUE)}
    v = sp.calendar_violations(k, bad)
    assert v and all(t1 == 30 / 365 and t2 == 58 / 365 for t1, t2, _ in v)
    t_mid = 44 / 365
    w1, w2 = sp.svi_total_var(0.0, *good[30 / 365]), sp.svi_total_var(0.0, *good[58 / 365])
    want = np.sqrt((w1 + (w2 - w1) * (t_mid - 30 / 365) / (28 / 365)) / t_mid)
    assert sp.interp_iv(0.0, t_mid, good) == pytest.approx(want)
    assert sp.interp_iv(0.0, 10 / 365, good) == pytest.approx(np.sqrt(w1 / (30 / 365)))    # flat σ outside


def test_sabr():
    F, T = 5000.0, 0.5
    atm = sp.sabr_vol(F, F, T, 0.2, 1.0, -0.3, 0.5)
    near = sp.sabr_vol(F, F * (1 + 1e-9), T, 0.2, 1.0, -0.3, 0.5)
    assert atm == pytest.approx(near, rel=1e-6)                       # continuous through the money
    assert sp.sabr_vol(F, F, T, 0.2, 1.0, 0.0, 1e-8) == pytest.approx(0.2, rel=1e-6)   # no vol-of-vol: flat α
    K = np.array([4500.0, 5000, 5500])
    smile = sp.sabr_vol(F, K, T, 0.2, 1.0, -0.4, 0.6)
    assert smile[0] > smile[1] > smile[2]                             # ρ < 0: downside skew
    assert sp.sabr_vol(F, K, T, 0.2, 1.0, 0.0, 0.6)[0] > sp.sabr_vol(F, F, T, 0.2, 1.0, 0.0, 0.6)   # ν: smile


# ----------------------------------------------------------------------------- portfolio
def book():
    return [
        sp.Position("SPY_C500", "option", -10, 500.0, 100, K=500, T=30 / 365, sigma=0.17, cp=1, q=0.013),
        sp.Position("SPY", "stock", 500, 500.0, 1, beta=1.0),
        sp.Position("ESM6", "future", 1, 5000.0, 50, beta=1.0),
        sp.Position("ES_P4800", "fop", 2, 5000.0, 50, K=4800, T=60 / 365, sigma=0.19, cp=-1),
        sp.Position("TSLA", "stock", 100, 250.0, 1, beta=2.0),
    ]


def test_dollar_greeks():
    call = book()[0]
    g = sp.dollar_greeks(call)
    raw = pr.greeks(500.0, 500, 30 / 365, 0.04, 0.013, 0.17, 1)
    assert g["dollar_delta"] == pytest.approx(-1000 * raw["delta"] * 500)
    assert g["dollar_gamma"] == pytest.approx(-1000 * raw["gamma"] * 500 ** 2 * 0.01)
    assert g["vega"] == pytest.approx(-1000 * raw["vega"] / 100) and g["theta"] == pytest.approx(-1000 * raw["theta"] / 365)
    assert g["theta"] > 0                                           # short option: time decay earns
    fut = sp.dollar_greeks(book()[2])
    assert fut == {"dollar_delta": 250_000.0, "dollar_gamma": 0.0, "vega": 0.0, "theta": 0.0}
    fop = sp.dollar_greeks(book()[3])
    b76 = pr.greeks(5000.0, 4800, 60 / 365, 0.04, 0.04, 0.19, -1)    # Black-76: S = F, q = r
    assert fop["dollar_delta"] == pytest.approx(100 * b76["delta"] * 5000)


def test_book_report_and_beta_weighting():
    rep = sp.book_report(book(), spy_price=500.0)
    assert list(rep.columns) == ["dollar_delta", "dollar_gamma", "vega", "theta", "beta_dollar_delta", "spy_shares"]
    assert rep.loc["TOTAL", "dollar_delta"] == pytest.approx(rep.drop("TOTAL")["dollar_delta"].sum())
    assert rep.loc["TSLA", "beta_dollar_delta"] == 50_000 and rep.loc["TSLA", "spy_shares"] == 100
    assert rep.loc["TOTAL", "spy_shares"] == pytest.approx(rep.loc["TOTAL", "beta_dollar_delta"] / 500)


def test_scenario_grid_full_revaluation_vs_taylor():
    b = book()
    grid = sp.scenario_grid(b, [-0.2, -0.01, 0.0, 0.01], [-0.05, 0.0, 0.1])
    assert grid.shape == (4, 3) and grid.loc[0.0, 0.0] == pytest.approx(0.0, abs=1e-6)
    for ds in (-0.01, 0.01):
        assert grid.loc[ds, 0.0] == pytest.approx(sp.taylor_pnl(b, ds, 0.0), rel=0.01)   # 1% moves: close
    # Large moves: the delta-gamma (Taylor) estimate can be wrong in EITHER direction -> always fully revalue.
    T = 30 / 365
    short_put = [sp.Position("P90", "option", -10, 100.0, 100, K=90, T=T, sigma=0.2, cp=-1)]
    full = sp.scenario_grid(short_put, [-0.20], [0.0]).iloc[0, 0]
    assert full < 3 * sp.taylor_pnl(short_put, -0.20, 0.0) < 0      # OTM put: gamma GROWS as spot falls -> Taylor
    want = -1000 * (pr.bsm_price(80, 90, T, 0.04, 0, 0.2, -1) - pr.bsm_price(100, 90, T, 0.04, 0, 0.2, -1))
    assert full == pytest.approx(want)                              # understates the loss more than 3×
    straddle = [sp.Position("C", "option", -10, 100.0, 100, K=100, T=T, sigma=0.2, cp=1),
                sp.Position("P", "option", -10, 100.0, 100, K=100, T=T, sigma=0.2, cp=-1)]
    full = sp.scenario_grid(straddle, [-0.25], [0.0]).iloc[0, 0]
    assert sp.taylor_pnl(straddle, -0.25, 0.0) < 1.5 * full < 0      # ATM straddle: gamma SHRINKS -> overstates


def test_delta_hedging_study():
    runs = {n: sp.delta_hedge_pnl(rebalances=n, n_paths=4000, seed=1) for n in (5, 30, 250)}
    sd = {n: runs[n].std() for n in runs}
    assert abs(runs[250].mean()) < 0.05 and sd[5] > sd[30] > sd[250]
    assert sd[30] / sd[250] == pytest.approx(np.sqrt(250 / 30), rel=0.35)      # ≈ 1/√(rebalances)
    edge = sp.delta_hedge_pnl(iv=0.25, rv=0.20, rebalances=250, n_paths=4000, seed=1).mean()
    assert 0.45 < edge < 0.70                                     # short vol earns when implied > realized
    costly = sp.delta_hedge_pnl(rebalances=250, n_paths=4000, seed=1, cost=0.0005)
    assert costly.mean() < runs[250].mean() - 0.1                 # hedging 250 times is not free
