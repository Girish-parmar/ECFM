import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import random_walk_prices, tick_data

fe = load("week33_features", "features")
IDX = pd.bdate_range("2024-01-01", periods=7)


def test_direction_dataset():
    close = pd.Series(np.exp(np.cumsum([0, .01, -.02, .03, .01, -.01, .02])), index=IDX)
    X, y, fwd = fe.direction_dataset(close, n_lags=2)
    assert list(X.columns) == ["lag0", "lag1"] and list(X.index) == list(IDX[2:6])
    np.testing.assert_allclose(X.iloc[0], [-0.02, 0.01])
    assert list(y) == [1, 1, 0, 1]
    np.testing.assert_allclose(fwd, [0.03, 0.01, -0.01, 0.02])


def test_walk_forward_baseline():
    X, y, fwd = fe.direction_dataset(random_walk_prices(2000))
    wf = fe.walk_forward_logit(X, y, fwd)
    assert len(wf["returns"]) == len(X) - 500 and wf["returns"].index[0] == X.index[500]
    assert abs(wf["accuracy"] - 0.5) < 0.04                                       # no signal, no skill
    assert wf["always_up"] == pytest.approx(y.iloc[500:].mean())
    leak = fe.walk_forward_logit(X.assign(leak=y), y, fwd)                        # a leaked target is caught at once
    assert leak["accuracy"] > 0.99
    pos = 2 * y.iloc[500:] - 1
    np.testing.assert_allclose(leak["returns"], fwd.iloc[500:].abs() - 5e-4 * pos.diff().fillna(pos).abs())


def test_dollar_bars_by_hand():
    ts = pd.date_range("2024-01-02 09:30", periods=5, freq="s")
    ticks = pd.DataFrame({"price": 10.0, "size": [1.0, 2, 3, 4, 5]}, index=ts)   # $10, 30, 60, 100, 150 cumulated
    bars = fe.dollar_bars(ticks, 50)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert list(bars.index) == [ts[1], ts[2], ts[3], ts[4]]                        # stamped when the bar CLOSES
    assert list(bars["volume"]) == [3, 3, 4, 5]


def test_dollar_bars_are_closer_to_normal():
    ticks = tick_data()
    tb = fe.time_bars(ticks, "30min")
    db = fe.dollar_bars(ticks, (ticks["price"] * ticks["size"]).sum() / len(tb))  # the same number of bars
    assert abs(len(db) - len(tb)) <= 2 and db["volume"].sum() == ticks["size"].sum()
    st, sd = fe.bar_stats(tb["close"]), fe.bar_stats(db["close"])
    assert sd["n"] == len(db) - 1 and sd["jb"] < st["jb"] / 3


def test_cusum_events():
    close = pd.Series(np.exp(np.cumsum([0, .01, .01, .01, -.05, .02])), index=IDX[:6])
    assert list(fe.cusum_events(close, 0.025)) == [IDX[3], IDX[4]]
    assert len(fe.cusum_events(random_walk_prices(2000), 0.03)) < 2000 / 5          # far fewer, relevant samples


def test_ffd():
    np.testing.assert_allclose(fe.ffd_weights(1.0), [-1, 1])
    np.testing.assert_allclose(fe.ffd_weights(0.0), [1])
    np.testing.assert_allclose(fe.ffd_weights(0.4)[-3:], [-0.12, -0.4, 1])
    x = pd.Series(np.cumsum(np.arange(10.0)), index=pd.bdate_range("2024-01-01", periods=10))
    pd.testing.assert_series_equal(fe.frac_diff(x, 1.0), x.diff().dropna())
    logp = np.log(random_walk_prices(2000))
    d, table = fe.min_ffd(logp)
    assert table.loc[0.0, "adf_p"] > 0.1 and table.loc[1.0, "adf_p"] < 0.01
    assert 0 < d <= 0.5 and table.loc[d, "corr"] > 0.5                            # stationary AND still remembers
    assert table["corr"].is_monotonic_decreasing


def test_microstructure_features():
    rng = np.random.default_rng(0)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, 5000))
    px = pd.Series(mid + 0.05 * rng.choice([-1, 1], 5000))                       # bid-ask bounce, spread 0.10
    assert fe.roll_spread(px, 1000).iloc[-1] == pytest.approx(0.10, rel=0.15)
    assert fe.roll_spread(pd.Series(mid), 1000).iloc[-1] < 0.03
    close = pd.Series([10.0, 11, 10, 10])
    vol = pd.Series([100.0, 100, 200, 100])
    expected = np.mean([np.log(1.1) / 1100, np.log(1.1) / 2000]) * 1e9
    assert fe.amihud(close, vol, window=2).iloc[2] == pytest.approx(expected)


def test_feature_store_is_point_in_time():
    fs = fe.FeatureStore()
    days = pd.bdate_range("2024-01-01", periods=3)
    fs.put("AAA", "pe", pd.Series([10.0, 11.0, 12.0], index=days), delay=pd.Timedelta(days=1))
    fs.put("BBB", "mom", pd.Series([0.1, 0.2, 0.3], index=days))
    got = fs.get(["AAA", "BBB", "CCC"], ["pe", "mom"], days[0])
    assert np.isnan(got.loc["AAA", "pe"]) and got.loc["BBB", "mom"] == 0.1           # pe is published a day later
    assert np.isnan(got.loc["CCC"]).all()
    assert fs.get(["AAA"], ["pe"], days[1]).loc["AAA", "pe"] == 10.0
    assert fs.get(["AAA"], ["pe"], days[2] + pd.Timedelta(days=1)).loc["AAA", "pe"] == 12.0
    fs.put("AAA", "pe", pd.Series([10.5], index=days[:1]), delay=pd.Timedelta(days=1))   # a revision
    assert fs.get(["AAA"], ["pe"], days[1]).loc["AAA", "pe"] == 10.5 and len(fs.rows) == 6


def test_truncation_test_catches_look_ahead():
    x = random_walk_prices(300)
    pts = [50, 150, 250]
    assert fe.truncation_test(lambda s: s.rolling(20).mean(), x, pts) == []
    assert fe.truncation_test(lambda s: fe.frac_diff(np.log(s), 0.4, 1e-3), x, pts) == []
    assert fe.truncation_test(lambda s: s.rolling(21, center=True).mean(), x, pts) == pts
    assert fe.truncation_test(lambda s: (s - s.mean()) / s.std(), x, pts) == pts        # full-sample z-score
    df = pd.DataFrame({"a": [1.0, 3], "b": [2.0, 2], "c": [3.0, 1]})
    np.testing.assert_allclose(fe.cross_sectional_rank(df), [[1 / 3, 2 / 3, 1], [1, 2 / 3, 1 / 3]])
