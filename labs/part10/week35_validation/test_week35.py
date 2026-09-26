import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from _loader import load
from common import overlapping_dataset, planted_features

va = load("week35_validation", "validation")
X, Y, T1 = overlapping_dataset()
XP, YP = planted_features()
CVP = va.PurgedKFold(pd.Series(XP.index, index=XP.index), 5, 0.0)                # independent rows
RF = RandomForestClassifier(100, min_samples_leaf=5, random_state=0, n_jobs=-1)


def test_purged_kfold_by_hand():
    d = pd.bdate_range("2024-01-01", periods=10)
    t1 = pd.Series(d[np.minimum(np.arange(10) + 2, 9)], index=d)
    splits = list(va.PurgedKFold(t1, n_splits=2, embargo=0.1).split(np.zeros(10)))
    np.testing.assert_array_equal(splits[0][1], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(splits[0][0], [7, 8, 9])                          # 5, 6 purged/embargoed
    np.testing.assert_array_equal(splits[1][0], [0, 1, 2])                          # 3, 4 end inside the test fold
    cv = va.PurgedKFold(T1)
    assert cv.get_n_splits() == 5
    assert len(cross_val_score(LogisticRegression(), X, Y, cv=cv, scoring="roc_auc")) == 5   # sklearn compatible


def test_overlap_audit():
    for tr, te in va.PurgedKFold(T1).split(X):
        assert va.overlap_test(tr, te, T1) == {"n_overlap": 0, "passed": True}
    tr, te = next(iter(KFold(5).split(X)))
    assert not va.overlap_test(tr, te, T1)["passed"]


def test_shuffled_kfold_finds_skill_in_noise():
    shuffled = va.cv_scores(RF, X, Y, KFold(5, shuffle=True, random_state=0))
    purged = va.cv_scores(RF, X, Y, va.PurgedKFold(T1))
    assert list(purged.columns) == ["log_loss", "auc", "accuracy"] and len(purged) == 5
    assert shuffled["auc"].mean() > 0.65 > 0.55 > purged["auc"].mean()             # the "edge" is leakage
    w = np.ones(len(X))
    np.testing.assert_allclose(va.cv_scores(LogisticRegression(), X, Y, va.PurgedKFold(T1), w),
                               va.cv_scores(LogisticRegression(), X, Y, va.PurgedKFold(T1)))


def test_importances():
    mdi = va.mdi_importance(RandomForestClassifier(100, min_samples_leaf=5, random_state=0).fit(XP, YP), XP.columns)
    mda = va.mda_importance(RF, XP, YP, CVP)
    assert mdi["noise_cont"] > 0.1 and abs(mda["noise_cont"]) < 0.01                 # MDI flatters noise
    assert mda.index[0] == "signal" and mda["noise_bin"] < 0.01
    clusters = va.cluster_features(XP)
    assert clusters == {"signal": ["signal", "twin"], "noise_cont": ["noise_cont"], "noise_bin": ["noise_bin"]}
    cmda = va.mda_importance(RF, XP, YP, CVP, groups=clusters)
    assert cmda.index[0] == "signal" and cmda["signal"] > 1.5 * mda["signal"]       # substitution effect removed
    sfi = va.sfi_importance(LogisticRegression(), XP, YP, CVP)
    assert sfi[["signal", "twin"]].min() > 0.75 and sfi[["noise_cont", "noise_bin"]].max() < 0.56


def test_leakage_audit():
    lr = LogisticRegression()
    s = va.shuffled_labels_test(lr, XP, YP, CVP)
    assert s["passed"] and abs(s["auc"] - 0.5) < 0.05
    assert va.canary_test(lr, X, Y, va.PurgedKFold(T1)) == {"rank": 1, "passed": True}
    ok = va.time_shift_test(lr, XP, YP, CVP)
    assert ok["passed"] and ok["auc"] > ok["auc_shifted"]
    y_late = pd.Series(np.r_[0, YP.to_numpy()[:-1]], index=YP.index)                  # the label reacts a bar later
    bad = va.time_shift_test(lr, XP, y_late, CVP)
    assert not bad["passed"]                                                          # features are stamped too early


def test_optuna_logs_every_trial():
    best, trials = va.optuna_search(X, Y, va.PurgedKFold(T1), n_trials=4)
    assert len(trials) == 4 and {"number", "value", "num_leaves", "learning_rate", "min_child_samples"} <= set(trials)
    row = trials.loc[trials["value"].idxmin()]
    assert best == {"num_leaves": row["num_leaves"], "learning_rate": row["learning_rate"],
                    "min_child_samples": row["min_child_samples"]}
    assert trials["num_leaves"].between(4, 32).all() and trials["learning_rate"].between(0.01, 0.2).all()


def test_seed_ensemble():
    make = lambda s: RandomForestClassifier(20, random_state=s, max_features=1)         # noqa: E731
    tr, te = XP.iloc[:1500], XP.iloc[1500:]
    ens = va.seed_ensemble(make, tr, YP.iloc[:1500], te, seeds=(0, 1, 2))
    single = [make(s).fit(tr, YP.iloc[:1500]).predict_proba(te)[:, 1] for s in (0, 1, 2)]
    np.testing.assert_allclose(ens, np.mean(single, axis=0))


def test_model_registry(tmp_path):
    reg = va.ModelRegistry(tmp_path / "registry.json")
    assert reg.register("meta_mom", {"audit_passed": True, "cv_auc": 0.56, "n_trials": 40}) == 1
    assert reg.register("meta_mom", {"audit_passed": False}) == 2
    with pytest.raises(ValueError):
        reg.register("meta_mom", {"cv_auc": 0.9})                                     # the audit must be recorded
    with pytest.raises(ValueError):
        reg.promote("meta_mom", 1, "paper")                                           # no skipping shadow mode
    with pytest.raises(ValueError):
        reg.promote("meta_mom", 2, "shadow")                                          # failed audit
    for stage in ("shadow", "paper", "live"):
        reg.promote("meta_mom", 1, stage)
    assert reg.get("meta_mom", "live")["version"] == 1 and reg.get("meta_mom", "shadow") is None
    reg.register("meta_mom", {"audit_passed": True})
    for stage in ("shadow", "paper", "live"):
        reg.promote("meta_mom", 3, stage)
    again = va.ModelRegistry(tmp_path / "registry.json")                              # persisted to disk
    assert again.get("meta_mom", "live")["version"] == 3 and again.get("meta_mom", "paper")["version"] == 1
