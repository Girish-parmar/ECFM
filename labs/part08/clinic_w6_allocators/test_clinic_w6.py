import pandas as pd
import pytest
from _loader import load
from common import strategy_returns

al = load("clinic_w6_allocators", "allocators")


@pytest.fixture(scope="module")
def table():
    return al.compare(strategy_returns(n_days=1000, seed=0), train=252, rebalance=63)


def test_comparison_table(table):
    assert list(table.index) == list(al.allocators()) and list(table.columns) == ["sharpe", "vol", "max_dd", "turnover"]
    assert table["turnover"].idxmin() == "1/N"                                    # no drift in weights, ever
    for name in ("inverse vol", "risk parity", "min variance (LW, long-only)", "HRP"):
        assert table.loc[name, "vol"] < table.loc["1/N", "vol"], name            # risk-based = less volatile
    assert (table["max_dd"] < 0).all()


def test_recommendation_rule():
    t = pd.DataFrame({"sharpe": [0.80, 0.95, 1.20, 0.97], "turnover": [0.01, 0.03, 0.80, 0.10]},
                     index=["1/N", "risk parity", "fancy", "HRP"])
    assert al.recommend(t) == "risk parity"         # 'fancy' trades too much; within 0.1 of 0.97: RP and HRP
    assert al.recommend(t, tolerance=0.2) == "1/N"
    assert al.recommend(t, max_turnover=0.001) == "1/N"                           # nothing qualifies -> fallback
