import pytest
from _loader import load

mr = load("clinic_w3_ml_report", "ml_report")


@pytest.fixture(scope="module")
def ds():
    return mr.research_dataset()


def test_cv_comparison(ds):
    cvt = mr.cv_comparison(ds)
    assert list(cvt.index) == ["logit", "rf", "lgbm"]
    assert list(cvt.columns) == ["shuffled_auc", "purged_auc", "inflation"]
    assert cvt["purged_auc"].between(0.5, 0.62).all()                               # a real but small edge
    assert (cvt["inflation"].abs() < 0.05).all()                                    # event sampling keeps overlap low


def test_audit_passes(ds):
    assert mr.audit(ds) == {"shuffled_labels": True, "canary": True, "time_shift": True, "overlap": True,
                            "all": True}


def test_meta_models_beat_the_primary_after_costs(ds):
    tt = mr.trading_table(ds)
    assert list(tt.index) == ["primary", "logit", "rf", "lgbm"]
    assert (tt.drop("primary")["sharpe"] > tt.loc["primary", "sharpe"]).all()
    assert (tt.drop("primary")["trades"] < tt.loc["primary", "trades"]).all()
    assert mr.n_trials(tt.drop("primary"), tt) == 9                                   # 3 kinds × (2 CV + 1 run)
