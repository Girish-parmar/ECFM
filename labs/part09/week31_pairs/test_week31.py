import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import cointegrated_pair, drifting_beta_pair, sector_universe, simulate_ou

pp = load("week31_pairs", "pairs")
TRUE_HL = 7.0
OU = simulate_ou(3000, np.log(2) / TRUE_HL, mu=1.0, sigma=0.5, seed=0)
RW = np.cumsum(np.random.default_rng(3).normal(size=3000))
PAIR = cointegrated_pair()


def test_fit_ou_recovers_the_truth():
    f = pp.fit_ou(OU)
    assert f["half_life"] == pytest.approx(TRUE_HL, rel=0.1)
    assert f["theta"] == pytest.approx(np.log(2) / f["half_life"])
    assert f["mu"] == pytest.approx(1.0, abs=0.25) and f["sigma"] == pytest.approx(0.5, rel=0.05)
    assert pp.fit_ou(np.arange(100.0) ** 1.5)["half_life"] == np.inf                 # explosive: b > 1
    assert pp.fit_ou(RW)["half_life"] > 100                                          # a random walk barely reverts


def test_variance_ratio_and_posterior():
    assert pp.variance_ratio(OU) < 0.9 < pp.variance_ratio(RW) < 1.1
    trend = np.cumsum(np.cumsum(np.random.default_rng(1).normal(size=3000)) * 0.01 + np.random.default_rng(2).normal(size=3000))
    assert pp.variance_ratio(trend) > 1.1
    r = np.random.default_rng(0).normal(0.001, 0.01, 50)
    m, sd = pp.normal_posterior_mean(r, prior_mean=0.0, prior_sd=0.001)
    s2 = r.var(ddof=1)
    prec = 1 / 0.001 ** 2 + 50 / s2
    assert m == pytest.approx(50 * r.mean() / s2 / prec) and sd == pytest.approx(prec ** -0.5)
    assert 0 < m < r.mean()                                                           # shrunk toward the prior
    big, _ = pp.normal_posterior_mean(np.random.default_rng(0).normal(0.001, 0.01, 200_000))
    assert big == pytest.approx(0.001, abs=1e-4)                                      # data wins with enough of it


def test_half_life_ci():
    lo, hi = pp.half_life_ci(OU)
    assert lo < TRUE_HL < hi and hi - lo < 4
    short_lo, short_hi = pp.half_life_ci(OU[:250], block=20)
    assert short_hi - short_lo > hi - lo                                              # less data, wider interval
    assert pp.half_life_ci(RW[:500])[1] == np.inf                                     # "maybe no reversion at all"


def test_engle_granger_and_johansen():
    eg = pp.engle_granger(PAIR.y, PAIR.x)
    assert eg["beta"] == pytest.approx(1.5, abs=0.05) and eg["pvalue"] < 1e-6
    assert eg["half_life"] == pytest.approx(7, rel=0.15)
    np.testing.assert_allclose(eg["spread"], PAIR.y - eg["beta"] * PAIR.x - eg["alpha"])
    assert abs(eg["spread"].mean()) < 1e-10
    walk = pd.Series(4 + np.cumsum(np.random.default_rng(9).normal(0, 0.01, len(PAIR))), index=PAIR.index)
    assert pp.engle_granger(PAIR.y, walk)["pvalue"] > 0.05
    rank, v = pp.johansen_rank(PAIR[["y", "x"]])
    assert rank == 1 and -v[1] / v[0] == pytest.approx(1.5, abs=0.05)                # same hedge ratio
    assert pp.johansen_rank(pd.DataFrame({"a": PAIR.x, "b": walk}))[0] == 0


def test_bh_reject():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.001])
    # sorted 0.001, 0.01, 0.03, 0.04, 0.2 vs k·0.05/5 = 0.01, 0.02, 0.03, 0.04, 0.05 → the four smallest
    np.testing.assert_array_equal(pp.bh_reject(p), [True, True, True, False, True])
    np.testing.assert_array_equal(pp.bh_reject([0.03, 0.5]), [False, False])          # 0.03 > 1·0.05/2
    assert not pp.bh_reject(np.linspace(0.3, 1, 20)).any()


def test_screen_pairs_finds_the_true_pairs():
    prices, sectors, truth = sector_universe()
    sc = pp.screen_pairs(prices, sectors)
    assert list(sc.columns) == ["a", "b", "beta", "pvalue", "half_life", "fdr_pass", "selected"]
    assert len(sc) == 4 * 28 and sc["pvalue"].is_monotonic_increasing                # 4 sectors × C(8, 2)
    assert set(zip(sc.a[sc.selected], sc.b[sc.selected])) == set(truth)
    assert (sc.pvalue < 0.05).sum() > sc.fdr_pass.sum()                               # raw p-values let false pairs in
    small, sec_small, _ = sector_universe(n_sectors=2, per_sector=5, n_days=600, seed=1)
    assert len(pp.screen_pairs(small)) == 45 > len(pp.screen_pairs(small, sec_small)) == 20


def test_zscore_and_state_machine():
    s = np.arange(10.0)
    z = pp.rolling_zscore(s, 5)
    assert np.isnan(z[:4]).all() and z[4] == pytest.approx(2 / np.std([0, 1, 2, 3, 4], ddof=1))
    z = [np.nan, 0, 2.5, 1.0, 0.3, -2.2, np.nan, -1, -5, 0, 4.5, 3, 3, 3, 3, 1.5, 2.5]
    np.testing.assert_array_equal(pp.pairs_positions(z),
                                  [0, 0, -1, -1, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1])
    # time stop after 2 bars, then no re-entry until |z| < entry
    np.testing.assert_array_equal(pp.pairs_positions(z, max_hold=2),
                                  [0, 0, -1, -1, 0, 1, 1, 1, 0, 0, 0, -1, -1, 0, 0, 0, -1])
    assert pp.pairs_positions([3.0, 1.0])[0] == -1                                    # the first bar can trade


def test_pair_pnl():
    y, x = [0, 0.01, 0.03, 0.02], [0, 0.01, 0.01, 0.0]
    pnl = pp.pair_pnl([1, 1, 0, 0], y, x, beta=1.0, cost_bps=5, borrow_bps_year=252)
    np.testing.assert_allclose(pnl, [0, -1e-3 - 5e-5, 0.01 - 5e-5, -1e-3])
    eg = pp.engle_granger(PAIR.y, PAIR.x)
    pos = pp.pairs_positions(pp.rolling_zscore(eg["spread"], 60))
    free = pp.pair_pnl(pos, PAIR.y, PAIR.x, eg["beta"], 0, 0)
    paid = pp.pair_pnl(pos, PAIR.y, PAIR.x, eg["beta"])
    assert paid.sum() > 0 and free.sum() > paid.sum()
    assert pp.pair_pnl(pos, PAIR.y, PAIR.x, eg["beta"], 5, 500).sum() < paid.sum()   # hard-to-borrow short


def test_kalman_beats_rolling_ols_on_a_drifting_beta():
    d = drifting_beta_pair()
    beta, alpha, e, q = pp.kalman_hedge(d.y, d.x)
    assert beta[-1] == pytest.approx(2.0, abs=0.1) and (q > 0).all()
    truth = d.beta.to_numpy()
    kal_err = np.abs(beta[250:] - truth[250:]).mean()
    ols_err = np.abs(pp.rolling_ols_beta(d.y, d.x, 250)[250:] - truth[250:]).mean()
    assert kal_err < 0.05 and kal_err < ols_err
    assert e[0] == pytest.approx(d.y.iloc[0])                                         # forecast error before the update
    ro = pp.rolling_ols_beta(PAIR.y, PAIR.x, 250)
    assert np.isnan(ro[:249]).all() and np.nanmean(ro) == pytest.approx(1.5, abs=0.1)
