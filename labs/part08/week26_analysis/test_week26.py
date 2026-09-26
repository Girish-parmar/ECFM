import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from _loader import load

an = load("week26_analysis", "analysis")
RNG = np.random.default_rng(0)


def test_point_in_time_members_and_survivorship():
    m = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "DDD"],
                      "start": pd.to_datetime(["2000-01-01", "2000-01-01", "2010-06-01", "2000-01-01"]),
                      "end": pd.to_datetime([None, "2008-10-01", None, "2015-01-01"])})
    assert an.members(m, "2005-01-01") == ["AAA", "BBB", "DDD"]
    assert an.members(m, "2008-10-01") == ["AAA", "DDD"]                 # removed ON its end date
    assert an.members(m, "2020-01-01") == ["AAA", "CCC"]
    # today's members backtested over history miss the stocks that failed:
    rets = {"AAA": 0.05, "BBB": -0.60, "CCC": 0.10, "DDD": -0.30}
    today = an.members(m, "2020-01-01")
    pit = an.members(m, "2005-01-01")
    assert np.mean([rets[s] for s in today]) > np.mean([rets[s] for s in pit])


def test_drawdown_and_tear_sheet():
    r = np.array([0.1, -0.2, 0.05, 0.05, 0.2, -0.01])
    mdd, dur = an.max_drawdown(r)
    assert mdd == pytest.approx(0.8 - 1) and dur == 3
    x = RNG.normal(0.0005, 0.01, 2520)
    ts = an.tear_sheet(x)
    assert ts["sharpe"] == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(252))
    assert ts["vol"] == pytest.approx(x.std(ddof=1) * np.sqrt(252)) and ts["max_dd"] < 0
    assert ts["calmar"] == pytest.approx(ts["cagr"] / abs(ts["max_dd"])) and abs(ts["skew"]) < 0.2
    assert ts["sortino"] > ts["sharpe"] * 1.2 and ts["tail_ratio"] == pytest.approx(1, abs=0.2)


def test_monthly_table():
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    r = pd.Series(0.001, index=idx)
    tab = an.monthly_table(r)
    assert list(tab.columns) == list(range(1, 13)) and tab.loc[2025, 1] == pytest.approx(1.001 ** 23 - 1)
    assert np.isnan(tab.loc[2025, 4])


def test_trades_mae_mfe_and_stats():
    open_ = np.array([100, 100, 101, 103, 102, 104, 104, 103, 100, 99.0])
    high = open_ + 1
    low = open_ - 1
    pos = np.array([0, 1, 1, 1, 0, 0, -1, -1, 0, 0.0])
    tr = an.trades_from_positions(pos, open_, high, low)
    assert tr[["entry_i", "exit_i", "direction"]].values.tolist() == [[2, 5, 1], [7, 9, -1]]
    assert tr["ret"].tolist() == pytest.approx([104 / 101 - 1, -(99 / 103 - 1)])
    assert tr["mae"].iloc[0] == pytest.approx(100 / 101 - 1) and tr["mfe"].iloc[0] == pytest.approx(104 / 101 - 1)
    assert tr["mfe"].iloc[1] == pytest.approx(1 - 99 / 103) and tr["mae"].iloc[1] == pytest.approx(1 - 104 / 103)
    st = an.trade_stats([0.02, -0.01, -0.01, 0.03, -0.02, -0.01, -0.01])
    assert st["win_rate"] == pytest.approx(2 / 7) and st["payoff"] == pytest.approx(0.025 / 0.012)
    assert st["profit_factor"] == pytest.approx(0.05 / 0.06) and st["max_consecutive_losses"] == 3


def test_sharpe_significance():
    assert an.sharpe_se(0.0, 100) == pytest.approx(0.1)
    x = RNG.normal(0.001, 0.01, 1000)                                    # per-period SR ≈ 0.1
    sr = x.mean() / x.std(ddof=1)
    assert an.psr(x) == pytest.approx(norm.cdf(sr / an.sharpe_se(sr, 1000)), abs=0.02)   # ≈ normal case
    assert an.psr(x, sr_benchmark=sr) == pytest.approx(0.5, abs=1e-9)
    # fat left tail lowers the PSR for the same mean and std:
    crash = x.copy()
    crash[:10] -= 0.05
    crash = (crash - crash.mean()) / crash.std(ddof=1) * x.std(ddof=1) + x.mean()
    assert an.psr(crash) < an.psr(x)
    mtr = an.min_track_record(x)
    assert 200 < mtr < 400 and an.min_track_record(-x) == float("inf")


def test_bootstrap_and_drawdown_distribution():
    x = RNG.normal(0.0005, 0.01, 750)
    b = an.stationary_bootstrap_sharpes(x, 300, 20, seed=1)
    sr = x.mean() / x.std(ddof=1)
    assert b.shape == (300,) and abs(np.median(b) - sr) < 0.02
    assert np.std(b) == pytest.approx(an.sharpe_se(sr, 750), rel=0.35)
    dd = an.drawdown_distribution(RNG.normal(0.002, 0.02, 200), n=500)
    assert (dd <= 0).all() and np.percentile(dd, 5) < np.median(dd)


def test_capacity_and_shortfall():
    gross = RNG.normal(0.0008, 0.01, 2520)
    caps = [1e5, 1e6, 1e7, 1e8, 1e9]
    s = an.net_sharpe_vs_capital(gross, daily_turnover=0.5, adv_dollars=5e7, daily_vol=0.02, capitals=caps)
    assert s.is_monotonic_decreasing and s.iloc[0] > 1
    assert an.capacity(s, 0.5) in caps[:-1] and an.capacity(s, 100) == 0.0
    assert an.implementation_shortfall_bps(100.0, 100.05, 1) == pytest.approx(5.0)
    assert an.implementation_shortfall_bps(100.0, 100.05, -1) == pytest.approx(-5.0)    # sold higher: a gain
