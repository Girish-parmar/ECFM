import numpy as np
import pytest
from _loader import load
from common import regime_market

rm = load("clinic_w1_regime_map", "regime_map")
BARS = regime_market(n_blocks=40, seed=7)


@pytest.fixture(scope="module")
def signals():
    return rm.seven_strategies(BARS)


def test_seven_strategies(signals):
    assert list(signals) == ["momentum", "range-bound", "mean reversion", "either-way", "volatility", "mathematical",
                             "statistical"]
    for name, s in signals.items():
        assert s.shape == (len(BARS),) and np.isfinite(s).all(), name
    assert set(np.unique(signals["statistical"])) == {0.0, 1.0}


def test_regime_names_and_estimates():
    names = rm.regime_names(BARS[["trend", "high_vol"]])
    assert set(names) == set(rm.REGIMES)
    est = rm.estimate_regimes(BARS)
    assert list(est.columns) == ["trend", "high_vol"] and est["trend"].iloc[:27].eq(0).all()   # ADX warm-up
    assert (est["high_vol"] == BARS["high_vol"]).mean() > 0.65
    assert (est["trend"] == BARS["trend"]).mean() > 0.55
    part = rm.estimate_regimes(BARS.iloc[:1000])
    assert part.equals(est.iloc[:1000])                                   # causal: no look-ahead


def test_the_regime_map_shows_where_each_group_works(signals):
    tab = rm.regime_table(BARS, signals, BARS[["trend", "high_vol"]])
    assert list(tab.index) == list(signals) and list(tab.columns) == rm.REGIMES
    trend_cols, range_cols = rm.REGIMES[:2], rm.REGIMES[2:]
    for s in ("momentum", "either-way", "mathematical"):
        assert tab.loc[s, trend_cols].mean() > 0 > tab.loc[s, range_cols].mean(), s
    for s in ("range-bound", "mean reversion"):
        assert tab.loc[s, range_cols].mean() > 0 > tab.loc[s, trend_cols].mean(), s
    assert tab.abs().max().max() < 5                                      # realistic, not miraculous
