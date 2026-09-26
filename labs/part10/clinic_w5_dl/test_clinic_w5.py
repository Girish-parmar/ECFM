import numpy as np
import pytest
from _loader import load
from common import garch_returns

dv = load("clinic_w5_dl", "dl_vs_gbm")
G = garch_returns(3000)
ABS = G["ret"].abs()


def test_folds():
    assert dv.folds(2999, 2000, 2) == [(2000, 2499), (2499, 2998)]


def test_every_model_forecasts_the_same_days():
    hg = dv.har_and_gbm(dv.st.har_features(ABS), 2000 - 22, 2100 - 22)
    np.testing.assert_allclose(hg["target"], ABS.iloc[2000:2100])
    out = dv.lstm_forecast(ABS.to_numpy(), 2000, 2100, epochs=1)
    np.testing.assert_allclose(out["target"], ABS.iloc[2000:2100], rtol=1e-5)
    assert len(out["pred"]) == 100 and len(out["cal_resid"]) > 200


@pytest.fixture(scope="module")
def result():
    return dv.compare(G["ret"], G["vol"])


def test_the_simple_model_wins(result):
    table, extra = result
    assert list(table.index) == ["har", "lgbm", "lstm"] and list(table.columns) == ["mse", "corr_true", "seed_std"]
    assert table["mse"].idxmin() == "har"                                          # HAR beats both on this data
    assert table.loc["lstm", "corr_true"] > 0.8 and table.loc["lstm", "seed_std"] > 0
    assert table.loc["har", "seed_std"] == 0


def test_conformal_coverage_under_regime_change(result):
    _, extra = result
    assert 0.8 < extra["coverage"] < 0.95                                           # close to 90%, not guaranteed
