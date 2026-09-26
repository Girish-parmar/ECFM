import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss
from _loader import load
from common import regime_market

lb = load("week34_labels", "labels")
D = pd.bdate_range("2024-01-01", periods=6)
MKT = regime_market()


def test_daily_vol_and_triple_barrier_by_hand():
    close = pd.Series([100.0, 101, 103, 99, 98, 100], index=D)
    pd.testing.assert_series_equal(lb.daily_vol(close, 3), close.pct_change().ewm(span=3).std())
    vol = pd.Series(0.02, index=D)
    tb = lb.triple_barrier(close, [D[0], D[2], D[5]], vol, pt=1, sl=1, max_hold=3)
    assert list(tb.index) == [D[0], D[2]]                                          # the last bar has no future
    assert list(tb["t1"]) == [D[2], D[3]] and list(tb["label"]) == [1, -1]
    np.testing.assert_allclose(tb["ret"], [0.03, 99 / 103 - 1])
    vert = lb.triple_barrier(close, [D[1]], vol, max_hold=1)                        # +1.98% < 2%: time barrier
    assert vert["label"].iloc[0] == 0 and vert["t1"].iloc[0] == D[2]
    wide = lb.triple_barrier(close, [D[0]], vol, pt=2, sl=2, max_hold=5)
    assert wide["label"].iloc[0] == 0 and wide["t1"].iloc[0] == D[5]


def test_concurrency_and_uniqueness():
    t1 = pd.Series([D[2], D[3]], index=[D[0], D[1]])
    np.testing.assert_array_equal(lb.num_concurrent(t1, D[:5]), [1, 2, 2, 1, 0])
    np.testing.assert_allclose(lb.avg_uniqueness(t1, D[:5]), [2 / 3, 2 / 3])
    alone = pd.Series([D[0], D[4]], index=[D[0], D[4]])
    np.testing.assert_allclose(lb.avg_uniqueness(alone, D), [1, 1])


def _sample_uniqueness(t1, index, drawn):
    cover = pd.Series(0.0, index=index)
    for j in drawn:
        cover.loc[t1.index[j]:t1.iloc[j]] += 1
    return np.mean([(1 / cover.loc[t1.index[j]:t1.iloc[j]]).mean() for j in drawn])


def test_sequential_bootstrap_draws_more_unique_samples():
    idx = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(100, 40, replace=False))
    t1 = pd.Series([idx[s + rng.integers(3, 15)] for s in starts], index=idx[starts])
    seq, iid = [], []
    for seed in range(8):
        drawn = lb.sequential_bootstrap(t1, idx, seed=seed)
        assert len(drawn) == 40 and min(drawn) >= 0 and max(drawn) < 40
        seq.append(_sample_uniqueness(t1, idx, drawn))
        iid.append(_sample_uniqueness(t1, idx, np.random.default_rng(seed).integers(0, 40, 40)))
    assert np.mean(seq) > np.mean(iid)


def test_meta_labels_and_bet_sizing():
    close = pd.Series(np.exp(np.cumsum([0, .01, .01, -.05, .01])), index=D[:5])
    np.testing.assert_array_equal(lb.momentum_side(close, 2), [0, 0, 1, -1, -1])
    side = pd.Series([1, -1, 1, -1.0])
    ret = pd.Series([0.02, 0.02, -0.01, -0.03])
    np.testing.assert_array_equal(lb.meta_labels(side, ret), [1, 0, 0, 1])
    np.testing.assert_allclose(lb.bet_size([0.50, 0.55, 0.70, 0.90]), [0.0, 0.08, 0.34, 0.82], atol=0.005)
    np.testing.assert_allclose(lb.bet_size([0.3, 0.0, 1.0]), [0, 0, 1])                # below 50%: no bet
    np.testing.assert_allclose(lb.discretize([0.08, 0.34, 0.82]), [0.1, 0.3, 0.8])


def test_ml_features_are_point_in_time():
    feats = lb.ml_features(MKT["close"], MKT["volume"])
    assert list(feats.columns) == ["ret5", "ret20", "vol20", "vol_ratio", "er20", "volume_z"]
    cut = lb.ml_features(MKT["close"].iloc[:501], MKT["volume"].iloc[:501])
    pd.testing.assert_series_equal(cut.iloc[-1], feats.iloc[500])
    er = feats["er20"].dropna()
    assert er.between(0, 1).all()
    by_regime = feats.groupby(MKT["regime"]).mean()
    assert by_regime.loc[0, "vol20"] > by_regime.loc[1, "vol20"] and by_regime.loc[1, "er20"] > by_regime.loc[0, "er20"]


def test_models_weights_and_calibration():
    assert {k: type(lb.make_model(k)).__name__ for k in ("logit", "rf", "lgbm")} == \
        {"logit": "Pipeline", "rf": "RandomForestClassifier", "lgbm": "LGBMClassifier"}
    with pytest.raises(ValueError):
        lb.make_model("svm")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 3)), columns=list("abc"))
    y = (rng.random(3000) < 1 / (1 + np.exp(-X["a"]))).astype(int)
    w = np.r_[np.zeros(1000), np.ones(2000)]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    subset = LogisticRegression().fit(X.iloc[1000:], y.iloc[1000:]).coef_
    for model in (LogisticRegression(), make_pipeline(LogisticRegression())):         # weights reach the last step
        fitted = lb.fit_weighted(model, X, y, w)
        coef = (fitted[-1] if hasattr(fitted, "steps") else fitted).coef_
        np.testing.assert_allclose(coef, subset, rtol=1e-3)                            # weight 0 = left out
    from sklearn.ensemble import RandomForestClassifier
    tr, ca, te = slice(0, 1000), slice(1000, 2000), slice(2000, 3000)
    raw = RandomForestClassifier(50, random_state=0).fit(X.iloc[tr], y.iloc[tr])      # overconfident trees
    cal = lb.calibrate(raw, X.iloc[ca], y.iloc[ca])
    p_raw = np.clip(raw.predict_proba(X.iloc[te])[:, 1], 1e-3, 1 - 1e-3)
    assert log_loss(y.iloc[te], cal.predict_proba(X.iloc[te])[:, 1]) < log_loss(y.iloc[te], p_raw)
    sig = lb.calibrate(raw, X.iloc[ca], y.iloc[ca], method="sigmoid")
    assert sig.predict_proba(X.iloc[te]).shape == (1000, 2)


def test_reliability_table():
    t = lb.reliability_table([0.1, 0.15, 0.9], [0, 1, 1], bins=5)
    assert list(t.index) == [0, 4] and list(t["count"]) == [2, 1]
    np.testing.assert_allclose(t["mean_pred"], [0.125, 0.9])
    np.testing.assert_allclose(t["frac_pos"], [0.5, 1.0])
    assert list(lb.reliability_table([1.0], [1]).index) == [4]                          # p = 1 goes in the top bin


def test_regimes_recover_the_hidden_state():
    feats = lb.ml_features(MKT["close"], MKT["volume"])[["vol20", "er20"]]
    g = lb.gmm_regimes(feats)
    assert set(g.unique()) == {0, 1} and ((1 - g) == MKT["regime"].loc[g.index]).mean() > 0.85   # 0 = calm = trend
    h = lb.hmm_regimes(np.log(MKT["close"]).diff())
    assert ((1 - h) == MKT["regime"].loc[h.index]).mean() > 0.9
    mom = lb.momentum_side(MKT["close"]).shift(1) * np.log(MKT["close"]).diff()
    perf = lb.regime_performance(mom, h)
    assert list(perf.columns) == ["sharpe", "mean", "share"] and perf["share"].sum() == pytest.approx(1)
    assert perf.loc[0, "sharpe"] > 1 > 0 > perf.loc[1, "sharpe"]                         # momentum lives in calm trends
