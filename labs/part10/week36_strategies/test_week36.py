import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import garch_returns, regime_market

st = load("week36_strategies", "strategies")
D = pd.bdate_range("2024-01-01", periods=6)
G = garch_returns(4000)


@pytest.fixture(scope="module")
def meta():
    ds = st.meta_dataset(regime_market(6000, seed=1))
    return ds, st.walk_forward_meta(ds)


def test_meta_dataset(meta):
    ds, _ = meta
    assert set(st.FEATURES) <= set(ds.columns) and {"t1", "ret", "y", "w"} <= set(ds.columns)
    assert (ds["t1"] > ds.index).all() and ds["w"].between(0, 1).all() and ds.notna().all().all()
    assert ((ds["ret"] > 0).astype(int) == ds["y"]).all()


def test_meta_labeling_beats_the_primary_after_costs(meta):
    ds, oos = meta
    assert oos.index[0] == ds.index[len(ds) // 3] and len(oos) == len(ds) - len(ds) // 3
    assert list(oos.columns) == ["prob", "size", "y", "primary", "meta"] and oos["size"].between(0, 1).all()
    s = st.meta_summary(oos, years=(oos.index[-1] - oos.index[0]).days / 365.25)
    assert s.loc["meta", "trades"] < s.loc["primary", "trades"]                      # it skips trades...
    assert s.loc["meta", "precision"] > s.loc["primary", "precision"]                # ...the worse ones
    assert s.loc["meta", "sharpe"] > s.loc["primary", "sharpe"] + 0.3


def test_meta_summary_by_hand():
    oos = pd.DataFrame({"size": [1.0, 0, 0.5, 0], "y": [1, 0, 1, 1], "primary": [0.02, -0.01, 0.01, 0.01],
                        "meta": [0.02, 0.0, 0.005, 0.0]})
    s = st.meta_summary(oos, years=1.0)
    assert list(s["trades"]) == [4, 2] and list(s["precision"]) == [0.75, 1.0]
    assert s.loc["primary", "sharpe"] == pytest.approx(np.mean([.02, -.01, .01, .01]) / np.std([.02, -.01, .01, .01], ddof=1) * 2)


def test_har_beats_naive_and_tracks_true_vol():
    x = pd.Series(np.arange(1.0, 31.0))
    h = st.har_features(x)
    assert h.index[0] == 21 and h.index[-1] == 28
    assert list(h.iloc[0]) == [22.0, 20.0, 11.5, 23.0]
    vf = st.vol_forecasts(st.har_features(G["ret"].abs()), 2000)
    assert list(vf.columns) == ["naive", "har", "lgbm", "target"]
    mse = ((vf[["naive", "har", "lgbm"]].sub(vf["target"], axis=0)) ** 2).mean()
    assert mse["har"] < 0.7 * mse["naive"] and mse["har"] < mse["lgbm"]                # the simple model wins
    assert np.corrcoef(vf["har"], G["vol"].shift(-1).loc[vf.index])[0, 1] > 0.9


def test_crash_labels_by_hand():
    ret = pd.Series([0, -.02, -.03, .01, 0, 0], index=D)
    np.testing.assert_array_equal(st.crash_labels(ret, horizon=2, threshold=0.04), [1, 0, 0, 0, np.nan, np.nan])


def test_crash_warning_cuts_risk():
    cw = st.crash_warning(G["ret"], 2000)
    assert cw.index[0] >= G.index[2000] and set(cw["label"].unique()) == {0.0, 1.0}
    assert st.pr_auc(cw["prob"], cw["label"]) > 1.8 * cw["label"].mean()              # vs the base rate, not 0.5
    ro = st.risk_off_returns(G["ret"], cw["prob"])
    assert st.max_drawdown(ro) > st.max_drawdown(G["ret"].loc[ro.index])
    toy = st.risk_off_returns(pd.Series(0.01, index=D[:3]), pd.Series([0.9, 0.1, 0.9], index=D[:3]))
    np.testing.assert_allclose(toy, [0.005, 0.01])
    assert list(toy.index) == [D[1], D[2]]


def test_rank_ic():
    pred = pd.DataFrame([[1, 2, 3], [1, 2, 3]], index=D[:2], columns=list("abc"))
    real = pd.DataFrame([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], index=D[:2], columns=list("abc"))
    np.testing.assert_allclose(st.rank_ic(pred, real), [1, -1])


def test_psi_and_drift_alarm():
    rng = np.random.default_rng(0)
    base = rng.normal(size=5000)
    assert st.psi(base, rng.normal(size=5000)) < 0.02
    assert 0.2 < st.psi(base, rng.normal(0.5, 1, 5000)) < 0.32                         # lesson plan: ≈ 0.25
    train = pd.DataFrame({"f1": rng.normal(size=2000), "f2": rng.normal(size=2000)})
    live = pd.DataFrame({"f1": np.r_[rng.normal(size=1000), rng.normal(0.7, 1, 500)], "f2": rng.normal(size=1500)},
                        index=pd.bdate_range("2020-01-01", periods=1500))
    tab = st.drift_table(train, live, window=250)
    assert list(tab.columns) == ["f1", "f2", "ks_p"] and list(tab.index) == list(live.index[249::250])
    assert st.first_alarm(tab) == live.index[1249]                                   # the first drifted block
    assert tab["ks_p"].iloc[-1] < 0.01 < tab["ks_p"].iloc[0]
    assert st.first_alarm(tab[["f2", "ks_p"]]) is None
    small = st.drift_table(train, live.iloc[:1000], window=60)                        # no drift at all, but...
    assert (small[["f1", "f2"]] > 0.1).any().any()                                    # ...60 values in 10 bins are noisy


def test_streaming_feature_parity():
    x = pd.Series(np.random.default_rng(1).normal(100, 5, 500))
    live = st.StreamingRolling(20)
    out = np.array([live.update(v) for v in x])
    np.testing.assert_allclose(out[:, 0], x.rolling(20).mean(), equal_nan=True)
    np.testing.assert_allclose(out[:, 1], x.rolling(20).std(), rtol=1e-6, equal_nan=True)


def test_shadow_mode():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300), index=pd.bdate_range("2023-01-02", periods=300))
    champion = pd.Series(1.0, index=r.index)
    oracle = np.sign(r.shift(-1)).fillna(0)
    res = st.shadow_compare(r, champion, oracle)
    assert res["promote"] and res["sharpe_challenger"] > res["sharpe_champion"]
    assert 0.3 < res["agreement"] < 0.7
    assert not st.shadow_compare(r.iloc[:40], champion.iloc[:40], oracle.iloc[:40])["promote"]    # too few days
