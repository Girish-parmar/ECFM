import numpy as np
import pytest
from _loader import load
from common import strategy_returns

pf = load("week30_portfolio", "portfolio")
R = strategy_returns(n_days=1500, seed=2)
COV = R.cov().to_numpy()


def test_inverse_vol_and_diversification():
    w = pf.inverse_vol_weights(R)
    assert w.sum() == pytest.approx(1) and w.idxmax() == "S3" and w.idxmin() == "S4"      # lowest / highest vol
    one = np.array([1.0, 0, 0, 0, 0])
    assert pf.diversification_ratio(one, COV) == pytest.approx(1.0)
    assert pf.diversification_ratio(w.to_numpy(), COV) > 1.5
    same = np.full((3, 3), 0.04)
    assert pf.diversification_ratio(np.ones(3) / 3, same) == pytest.approx(1.0)            # perfect correlation


def test_mean_variance():
    w = pf.min_variance(COV)
    assert w.sum() == pytest.approx(1)
    for _ in range(20):                                              # no nearby portfolio has lower variance
        d = np.random.default_rng(_).normal(0, 0.01, 5)
        d -= d.mean()
        assert (w + d) @ COV @ (w + d) >= w @ COV @ w - 1e-15
    lo = pf.min_variance_long_only(COV, max_weight=0.4)
    assert lo.sum() == pytest.approx(1, abs=1e-6) and lo.min() >= -1e-7 and lo.max() <= 0.4 + 1e-6
    mu = R.mean().to_numpy()
    t = pf.max_sharpe(mu, COV)
    sr = lambda x: x @ mu / np.sqrt(x @ COV @ x)                     # noqa: E731
    assert sr(t) >= sr(np.ones(5) / 5) and t.sum() == pytest.approx(1)


def test_shrinkage_helps_when_assets_approach_observations():
    rng = np.random.default_rng(0)
    n, T = 40, 60
    true = 0.0001 * (0.3 * np.ones((n, n)) + 0.7 * np.eye(n))
    L = np.linalg.cholesky(true)
    train, test = rng.normal(size=(T, n)) @ L.T, rng.normal(size=(5000, n)) @ L.T
    oos = lambda w: np.std(test @ w)                                  # noqa: E731
    assert oos(pf.min_variance(pf.ledoit_wolf(train))) < oos(pf.min_variance(np.cov(train.T)))


def test_risk_parity_and_hrp():
    rp = pf.risk_parity(COV)
    np.testing.assert_allclose(pf.risk_contributions(rp, COV), 0.2, atol=1e-4)      # lesson plan: 20% each
    assert pf.risk_contributions(np.ones(5) / 5, COV).max() > 0.3
    h = pf.hrp(R)
    assert list(h.index) == list(R.columns) and h.sum() == pytest.approx(1) and (h > 0).all()
    assert h.idxmax() == R.var().idxmin()                                          # most weight to the lowest variance


def test_black_litterman():
    w_mkt = np.ones(5) / 5
    pi = pf.black_litterman(COV, w_mkt, delta=2.5)
    np.testing.assert_allclose(pi, 2.5 * COV @ w_mkt)
    P, Q = np.array([[1.0, -1.0, 0, 0, 0]]), np.array([0.001])                   # S1 beats S2 by 0.1%/day
    post = pf.black_litterman(COV, w_mkt, P=P, Q=Q)
    assert (post[0] - post[1]) > (pi[0] - pi[1])
    assert (post[0] - post[1]) < 0.001 + 1e-12                                    # blended, not replaced
    confident = pf.black_litterman(COV, w_mkt, P=P, Q=Q, omega=np.array([[1e-12]]))
    assert confident[0] - confident[1] == pytest.approx(0.001, rel=1e-4)


def test_min_cvar():
    X = R.to_numpy()
    w = pf.min_cvar(X, alpha=0.95)
    assert w.sum() == pytest.approx(1, abs=1e-6) and w.min() >= -1e-7

    def cvar(x):
        loss = -(X @ x)
        v = np.quantile(loss, 0.95)
        return loss[loss >= v].mean()
    assert cvar(w) <= cvar(np.ones(5) / 5) + 1e-9 and cvar(w) <= cvar(pf.inverse_vol_weights(R).to_numpy()) + 1e-6


def test_walk_forward_allocation():
    eq = pf.walk_forward_allocation(R, lambda x: np.ones(x.shape[1]) / x.shape[1], train=252, rebalance=21)
    assert len(eq["returns"]) == len(R) - 252 and eq["returns"].notna().all()
    assert eq["turnover"] == pytest.approx(1 / len(np.arange(252, len(R), 21)))       # only the first allocation
    np.testing.assert_allclose(eq["returns"].iloc[:21], R.iloc[252:273].mean(axis=1))
    ivol = pf.walk_forward_allocation(R, lambda x: pf.inverse_vol_weights(x).to_numpy())
    assert ivol["weights"].index[0] == R.index[252] and ivol["turnover"] > eq["turnover"]
