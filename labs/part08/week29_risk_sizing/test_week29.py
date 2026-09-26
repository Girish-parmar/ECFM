import numpy as np
import pytest
from scipy.stats import norm
from _loader import load

rs = load("week29_risk_sizing", "risk_sizing")
CTX = {"equity": 100_000, "gross": 150_000, "positions": {"SPY": 15_000}, "sector_exposure": {"Tech": 35_000},
       "day_pnl": -500, "start_equity": 100_500, "adv": {"SPY": 1_000_000, "XYZ": 50_000}}


def order(sym="SPY", qty=10, price=500.0, sector=None):
    o = {"symbol": sym, "qty": qty, "price": price}
    if sector:
        o["sector"] = sector
    return o


def test_rules():
    assert rs.MaxNotional(50_000).check(order(qty=120), CTX) == rs.RiskDecision(False, "notional 60,000 > 50,000")
    assert rs.MaxNotional(50_000).check(order(qty=-100), CTX).approved
    d = rs.MaxGrossExposure(2.0).check(order(qty=120), CTX)
    assert not d.approved and d.reason == "gross 2.10x > 2.0x"
    assert rs.MaxPositionWeight(0.2).check(order(qty=20), CTX).reason == "SPY weight 25.0% > 20.0%"
    assert rs.MaxPositionWeight(0.2).check(order(qty=-40), CTX).approved             # reduces the position
    assert rs.MaxSectorExposure(0.4).check(order("AAPL", 35, 200.0, "Tech"), CTX).reason == "sector Tech 42.0% > 40.0%"
    assert rs.MaxSectorExposure(0.4).check(order("GLD", 1000, 200.0), CTX).approved   # no sector
    assert rs.MaxADVParticipation(0.1).check(order("XYZ", 7_500, 10.0), CTX).reason == "qty 7,500 is 15.0% of ADV > 10.0%"
    bad_day = {**CTX, "day_pnl": -2_512.5}
    assert rs.DailyLossLimit(0.02).check(order(), bad_day).reason == "daily loss -2.50% breaches -2.00%"
    assert rs.DailyLossLimit(0.02).check(order(), CTX).approved


def test_engine_chain_and_log():
    eng = rs.RiskEngine([rs.MaxNotional(50_000), rs.MaxGrossExposure(2.0), rs.MaxPositionWeight(0.2)])
    assert eng.check(order(qty=10), CTX).approved
    d = eng.check(order(qty=120), CTX)
    assert not d.approved and d.reason == "MaxNotional: notional 60,000 > 50,000"          # first rejection wins
    eng.check(order(qty=20), CTX)
    assert eng.log == [("SPY", "MaxNotional: notional 60,000 > 50,000"), ("SPY", "MaxPositionWeight: SPY weight 25.0% > 20.0%")]


def test_var_measures():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 100_000)
    var, cvar = rs.hist_var_cvar(r, 0.99)
    z = norm.ppf(0.99)
    assert var == pytest.approx(z * 0.01, rel=0.03)
    assert cvar == pytest.approx(0.01 * norm.pdf(z) / (1 - 0.99), rel=0.05)      # normal expected shortfall
    cov = np.array([[0.04, 0.006], [0.006, 0.01]])
    w = np.array([0.6, 0.4])
    v = rs.parametric_var(w, cov, 0.99)
    assert v == pytest.approx(norm.ppf(0.99) * np.sqrt(w @ cov @ w))
    assert rs.parametric_var(w, cov, 0.99, mu=[0.1, 0.05]) == pytest.approx(v - 0.08)
    comp = rs.component_var(w, cov, 0.99)
    assert comp.sum() == pytest.approx(v) and comp[0] > comp[1]


def test_sizing():
    assert rs.risk_per_trade_size(100_000, 0.01, entry=100, stop=95) == 200                  # lesson-plan example
    assert rs.risk_per_trade_size(100_000, 0.01, 5000, 4980, multiplier=50) == 1              # one ES contract
    r = np.random.default_rng(1).normal(0, 0.02, 60)
    assert rs.vol_target_weight(r, 0.10) == pytest.approx(min(0.10 / (np.std(r, ddof=1) * np.sqrt(252)), 2.0))
    assert rs.vol_target_weight(r * 0.01, 0.10) == 2.0                                          # capped
    assert rs.turtle_units(1_000_000, 0.01, atr=40, point_value=50) == 5


def test_risk_of_ruin():
    trades = [2.0, -1, -1, 1.5, -1, 3.0, -1]                        # positive expectancy in R
    small, big = rs.risk_of_ruin(trades, 0.01), rs.risk_of_ruin(trades, 0.10)
    assert small == 0.0 and big > 0.1
    assert rs.risk_of_ruin([-1.0], 0.5, ruin=0.5, n_trades=2, n_sims=10) == 1.0


def test_kelly():
    assert rs.kelly_discrete(0.55, 1.0) == pytest.approx(0.10)
    assert rs.kelly_continuous(0.08, 0.16, r=0.03) == pytest.approx(1.953125)                  # lesson plan ≈ 1.95x
    mu, cov = np.array([0.08, 0.06]), np.array([[0.04, 0.01], [0.01, 0.02]])
    np.testing.assert_allclose(rs.kelly_multi(mu, cov, 0.02), np.linalg.inv(cov) @ (mu - 0.02))


def test_kelly_with_estimation_error():
    sim = rs.kelly_growth_simulation(n_paths=400, seed=0)
    assert list(sim.index) == [0.25, 0.5, 1.0, 2.0]
    assert sim.loc[2.0, "median_wealth"] < 1 < sim.loc[0.5, "median_wealth"]                   # 2x Kelly loses money
    assert sim.loc[0.5, "median_wealth"] > sim.loc[1.0, "median_wealth"]                       # half beats full
    assert sim["median_max_dd"].is_monotonic_decreasing and sim.loc[1.0, "median_max_dd"] < -0.6
