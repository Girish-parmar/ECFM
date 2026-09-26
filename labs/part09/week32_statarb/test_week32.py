import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from _loader import load
from common import breaking_pair, characteristics_panel, cointegrated_pair, factor_universe

sa = load("week32_statarb", "statarb")
R, REV = factor_universe()


def lesson_plan_s_scores(returns, n_factors=5, window=60):
    """The reference implementation printed in the lesson plan (S5)."""
    R = returns.iloc[-window:]
    Z = (R - R.mean()) / R.std()
    _, eigvec = np.linalg.eigh(np.corrcoef(Z.T.to_numpy()))
    F = Z.to_numpy() @ eigvec[:, ::-1][:, :n_factors]
    A = np.c_[np.ones(window), F]
    out = {}
    for col in R.columns:
        coef, *_ = np.linalg.lstsq(A, R[col].to_numpy(), rcond=None)
        X = np.cumsum(R[col].to_numpy() - A @ coef)
        b, a = np.polyfit(X[:-1], X[1:], 1)
        if not 0 < b < 1:
            continue
        out[col] = (X[-1] - a / (1 - b)) / np.sqrt((X[1:] - (a + b * X[:-1])).var(ddof=2) / (1 - b ** 2))
    return pd.Series(out)


def test_s_scores_match_the_lesson_plan():
    for end in (100, 400, 750):
        s = sa.s_scores(R.iloc[:end], n_factors=3, window=60)
        ref = lesson_plan_s_scores(R.iloc[:end], n_factors=3, window=60)
        pd.testing.assert_series_equal(s, ref, check_names=False, atol=1e-8)
    fast = sa.s_scores(R, 3, 60, max_half_life=5)
    assert set(fast.index) < set(sa.s_scores(R, 3, 60).index)


def test_s_score_rules():
    prev = pd.Series([0, 0, 0, 1, 1, -1, -1, 1.0], index=list("ABCDEFGH"))
    s = pd.Series({"A": -1.5, "B": 1.3, "C": 1.0, "D": -0.6, "E": -0.4, "F": 0.8, "G": 0.7})     # H: not reverting
    np.testing.assert_array_equal(sa.s_score_positions(s, prev), [1, -1, 0, 1, 0, -1, 0, 0])


def test_residual_backtest_is_market_neutral_and_needs_real_reversion():
    bt = sa.residual_backtest(R)
    r = bt["returns"]
    assert r.index[0] == R.index[60] and len(bt["positions"]) == len(r)
    sharpe = lambda x: x.mean() / x.std() * np.sqrt(252)                                        # noqa: E731
    assert sharpe(r) > 0.5
    assert abs(np.corrcoef(r, R.mean(axis=1).loc[r.index])[0, 1]) < 0.15                        # market neutral
    assert sa.residual_backtest(R, cost_bps=0)["returns"].sum() > r.sum()
    noise, _ = factor_universe(reversion_share=0.0)
    assert sharpe(sa.residual_backtest(noise)["returns"]) < 0                                   # costs, no edge


def test_newey_west_matches_statsmodels():
    e = np.random.default_rng(0).normal(size=300)
    x = np.zeros(300)
    for t in range(1, 300):
        x[t] = 0.7 * x[t - 1] + e[t]                                               # AR(1), positively autocorrelated
    for lags in (0, 3, 8):
        ref = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
        assert sa.newey_west_se(x, lags) == pytest.approx(ref.bse[0])
    assert sa.newey_west_se(x, 8) > sa.newey_west_se(x, 0)                           # positive autocorrelation


def test_fama_macbeth_recovers_the_planted_premium():
    rets, expo = characteristics_panel(n_stocks=200, n_months=120)
    fm = sa.fama_macbeth(rets, expo)
    assert list(fm.index) == ["const", "momentum", "size"] and list(fm.columns) == ["premium", "t_stat"]
    assert fm.loc["momentum", "premium"] == pytest.approx(0.002, abs=0.0006) and fm.loc["momentum", "t_stat"] > 4
    assert abs(fm.loc["size", "t_stat"]) < 2
    nw = sa.fama_macbeth(rets, expo, nw_lags=3)
    np.testing.assert_allclose(nw["premium"], fm["premium"])
    assert nw.loc["momentum", "t_stat"] > 4


def test_neutralize():
    rng = np.random.default_rng(0)
    names = [f"N{i}" for i in range(40)]
    sectors = pd.Series(np.repeat(["tech", "fin", "energy", "health"], 10), index=names)
    beta = pd.Series(rng.uniform(0.5, 1.5, 40), index=names)
    raw = pd.Series(rng.normal(size=40), index=names)
    sig = raw + 2 * beta + sectors.map({"tech": 1.0, "fin": -1.0, "energy": 0.0, "health": 3.0})
    out = sa.neutralize(sig, sectors, beta)
    np.testing.assert_allclose(out.groupby(sectors).mean(), 0, atol=1e-12)
    assert abs(np.cov(out, beta)[0, 1]) < 1e-12
    assert np.corrcoef(out, raw)[0, 1] > 0.9                                       # the stock-specific part survives


def test_chow_and_cusum():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    e = rng.normal(0, 0.5, 400)
    y_break = 1 + np.where(np.arange(400) < 200, 2.0, 2.5) * x + e
    F, p = sa.chow_test(y_break, x, 200)
    assert p < 1e-6
    ssr = lambda yy, xx: sm.OLS(yy, sm.add_constant(xx)).fit().ssr                                # noqa: E731
    s1, s2 = ssr(y_break[:200], x[:200]), ssr(y_break[200:], x[200:])
    assert F == pytest.approx(((ssr(y_break, x) - s1 - s2) / 2) / ((s1 + s2) / 396))
    assert sa.chow_test(1 + 2 * x + e, x, 200)[1] > 0.05
    assert sa.cusum_pvalue(1 + 2 * x + e + np.where(np.arange(400) < 250, 0, 0.4), x) < 0.01    # level shift
    assert sa.cusum_pvalue(1 + 2 * x + e, x) > 0.05
    pair = cointegrated_pair()
    # a FALSE alarm: a perfectly stable pair, but the level residuals are autocorrelated (the tests assume iid errors)
    assert sa.cusum_pvalue(pair.y, pair.x) < 0.05


def test_rolling_stability_and_retirement():
    d = breaking_pair()                                                            # dies at row 900
    st = sa.rolling_stability(d.y, d.x, window=250, step=21)
    assert list(st.columns) == ["pvalue", "half_life", "beta", "spread_vol"]
    assert st.index[0] == d.index[249] and len(st) == len(range(250, len(d) + 1, 21))
    date = sa.retirement_date(st)
    assert date is not None and d.index[900] < date < d.index[900 + 250]           # caught within one window
    ok = cointegrated_pair()
    assert sa.retirement_date(sa.rolling_stability(ok.y, ok.x)) is None
    toy = pd.DataFrame({"pvalue": [0.01, 0.3, 0.01, 0.3, 0.3], "half_life": 5.0, "beta": 1.0},
                       index=pd.bdate_range("2020-01-01", periods=5))
    assert sa.retirement_date(toy) == toy.index[4]                                  # two bad checks in a row
    assert sa.retirement_date(toy.assign(beta=[1, 1, 1.3, 1, 1])) == toy.index[2]  # p-value, then β drift: any rule
    assert sa.retirement_date(toy.assign(beta=[1, 1.3, 1, 1, 1]), patience=1) == toy.index[1]


def test_stationary_bootstrap_and_reality_check():
    idx = sa.stationary_bootstrap_indices(100_000, 10, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 100_000
    assert np.mean(np.diff(idx) == 1) == pytest.approx(0.9, abs=0.01)             # mean block length 10
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, (1000, 50))
    rc = sa.reality_check(noise)
    assert rc["naive_p"] < 0.01 and rc["p_value"] > 0.1                           # the best of 50 looks significant
    good = noise.copy()
    good[:, 7] += 0.0012
    rc = sa.reality_check(good)
    assert rc["best"] == 7 and rc["p_value"] < 0.05


def test_ensemble_positions():
    z = [0, 2.1, 1.0, 0.6, 0.3]
    pos = sa.ensemble_positions(z)
    # entry 2.25 never fires; entries 1.75 and 2.0 do (6 of 9 combinations); exits 0.75 / 0.5 / 0.25 close one by one
    np.testing.assert_allclose(pos, [0, -6 / 9, -6 / 9, -4 / 9, -2 / 9])


def test_pair_book_weights_with_name_caps():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (20_000, 3)), columns=["P1", "P2", "P3"])
    pairs = {"P1": ("A", "B"), "P2": ("A", "C"), "P3": ("D", "E")}
    free = sa.pair_book_weights(rets, pairs, name_cap=1.0)
    np.testing.assert_allclose(free, 1 / 3, atol=0.01)
    w = sa.pair_book_weights(rets, pairs, name_cap=0.5)
    assert w.sum() == pytest.approx(1) and w["P1"] + w["P2"] <= 0.5 + 1e-8        # A is in two pairs
    np.testing.assert_allclose(w, [0.25, 0.25, 0.5], atol=0.01)
    with pytest.raises(ValueError):
        sa.pair_book_weights(rets, pairs, name_cap=0.4)                            # P3 alone would need 0.6


def test_book_exposures():
    pairs = {"P1": ("A", "B"), "P2": ("A", "C")}
    ex = sa.book_exposures(pd.Series({"P1": 1.0, "P2": -1.0}), pd.Series({"P1": 0.5, "P2": 0.5}), pairs,
                           hedges=pd.Series({"P1": 1.5, "P2": 0.5}),
                           betas=pd.Series({"A": 1.0, "B": 1.2, "C": 0.8}),
                           sectors=pd.Series({"A": "tech", "B": "tech", "C": "fin"}))
    np.testing.assert_allclose(ex["weights"], [0.2 - 1 / 3, -0.3, 1 / 6])
    assert list(ex["weights"].index) == ["A", "B", "C"]
    assert ex["gross"] == pytest.approx(0.6) and ex["net"] == pytest.approx(-0.8 / 3)
    assert ex["net_beta"] == pytest.approx(-0.36)
    np.testing.assert_allclose(ex["sector_net"], [1 / 6, -0.3 + 0.2 - 1 / 3])       # fin, tech
