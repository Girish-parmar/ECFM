import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import regime_market, rsi, sma

ln = load("week23_linear_groups", "linear")
MKT = regime_market(n_blocks=12, seed=2)
O, H, L, C = (MKT[k].to_numpy() for k in ("open", "high", "low", "close"))


def ou(n=3000, theta=0.1, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] - theta * x[t - 1] + rng.normal(0, sigma)
    return x


# ------------------------------------------------------------------------------ mean reversion
def test_rsi2_rules():
    pos = ln.rsi2_reversion(C, 10, 70, 200)
    r, m = rsi(C, 2), sma(C, 200)
    entries = np.flatnonzero((pos[1:] == 1) & (pos[:-1] == 0)) + 1
    exits = np.flatnonzero((pos[1:] == 0) & (pos[:-1] == 1)) + 1
    assert entries.size > 5 and all(r[t] < 10 and C[t] > m[t] for t in entries)
    assert all(r[t] > 70 for t in exits)
    no_trend = ln.rsi2_reversion(C, 10, 70, 200, use_trend=False)
    assert no_trend.sum() > pos.sum()                                 # the filter removes trades
    capped = ln.rsi2_reversion(C, 10, 101, 200, use_trend=False, max_hold=3)   # RSI never > 101: only the time stop
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], capped, [0]])) != 0))[::2]
    assert runs.max() <= 4


def test_ibs():
    np.testing.assert_allclose(ln.ibs(np.array([11.0, 10]), np.array([9.0, 10]), np.array([9.5, 10])), [0.25, 0.5])


def test_zscore_reversion_on_mean_reverting_data():
    x = 100 * np.exp(ou())
    pos = ln.zscore_reversion(x, 20, 2.0, 0.5, max_hold=10)
    assert set(np.unique(pos)) == {-1.0, 0.0, 1.0}
    ret = np.zeros_like(x)
    ret[:-1] = x[1:] / x[:-1] - 1
    assert (pos * ret).sum() > 0                                      # on an OU process fading extremes pays
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], np.abs(pos), [0]])) != 0))[::2]
    assert runs.max() <= 11                                           # the time stop works


def test_bollinger_fade_only_trades_in_ranges():
    pos = ln.bollinger_fade(H, L, C)
    from common import adx
    a = adx(H, L, C, 14)
    entries = np.flatnonzero((pos != 0) & (np.roll(pos, 1) == 0))
    assert entries.size > 3 and (a[entries] < 20).all()
    up = np.linspace(100, 200, 300)
    assert (ln.bollinger_fade(up + 1, up - 1, up) == 0).all()         # a clean trend: ADX high, never fades


# -------------------------------------------------------------------------------- either-way
def session(prices, start="2026-01-12 14:30"):
    idx = pd.date_range(start, periods=len(prices), freq="5min", tz="UTC")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p + 0.2, "low": p - 0.2, "close": p}, index=idx)


def test_opening_range_breakout():
    s = session([100, 100.5, 99.8, 100.2, 100.1, 99.9, 100.4, 101.0, 101.5, 102.0, 102.5, 103.0])
    t = ln.orb_trade(s, minutes=30)                                  # range = first 6 bars: 99.6 .. 100.7
    assert t["direction"] == 1 and t["entry"] == 101.5 and t["stop"] == pytest.approx(99.6)
    assert not t["stopped"] and t["ret"] == pytest.approx(103.0 / 101.5 - 1)
    s = session([100, 100.5, 99.8, 100.2, 100.1, 99.9, 99.2, 99.0, 100.9, 101.0])
    t = ln.orb_trade(s, minutes=30)                                  # breaks down, then rips through the stop
    assert t["direction"] == -1 and t["stopped"] and t["exit"] == pytest.approx(100.7)
    assert t["ret"] == pytest.approx(-(100.7 / 99.0 - 1))
    assert ln.orb_trade(session([100, 100.1, 100.0, 100.2, 100.1, 100.0, 100.1, 100.2]), 30) is None


# -------------------------------------------------------------------------------- volatility
def test_vol_target_overlay():
    rng = np.random.default_rng(3)
    vol = np.where(np.arange(2000) < 1000, 0.05, 0.40) / np.sqrt(252)
    rets = rng.normal(0, vol)
    scaled = ln.vol_target_overlay(np.ones(2000), rets, target=0.10, n=20, max_lev=5)
    assert (scaled[:19] == 0).all() and scaled[19] > 0                  # 20 returns known at bar 19
    pnl = scaled[:-1] * rets[1:]                                      # position decided at t earns t+1
    for part in (pnl[100:990], pnl[1100:]):
        assert part.std() * np.sqrt(252) == pytest.approx(0.10, rel=0.15)
    np.testing.assert_allclose(ln.vix_regime_overlay([1.0, 1.0, -1.0], [15, 30, 30], [18, 25, 25]), [1.0, 0.5, -0.5])


# ------------------------------------------------------------------------------ mathematical
def test_kalman_and_hurst():
    rng = np.random.default_rng(4)
    noisy = np.linspace(0, 10, 500) + rng.normal(0, 1, 500)
    slow, fast = ln.kalman_level(noisy, 1e-5, 1e-2), ln.kalman_level(noisy, 1e-2, 1e-2)
    assert np.std(np.diff(slow)) < np.std(np.diff(fast))              # q/r controls smoothness
    walk = np.cumsum(rng.normal(size=5000))
    assert ln.hurst_exponent(walk) == pytest.approx(0.5, abs=0.07)
    assert ln.hurst_exponent(ou(5000, theta=0.2)) < 0.3
    inc = np.zeros(5000)
    for t in range(1, 5000):
        inc[t] = 0.7 * inc[t - 1] + rng.normal()
    assert ln.hurst_exponent(np.cumsum(inc)) > 0.6


# ------------------------------------------------------------------------------- statistical
def test_pairs_positions_no_lookahead_and_profit_on_a_cointegrated_pair():
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(0, 0.01, 1500)) + 4.0
    y = 1.5 * x + ou(1500, 0.1, 0.005, seed=6) - 2.0
    pos, beta = ln.pairs_positions(y, x, 60, 2.0, 0.5)
    assert np.isnan(beta[:60]).all() and np.nanmedian(beta) == pytest.approx(1.5, abs=0.2)
    spread_ret = np.diff(y) - beta[:-1] * np.diff(x)
    assert (pos[:-1] * np.nan_to_num(spread_ret)).sum() > 0
    p2, _ = ln.pairs_positions(y[:900], x[:900], 60, 2.0, 0.5)
    np.testing.assert_array_equal(p2, pos[:900])                      # the future does not change the past


def test_calendar_and_session_splits():
    idx = pd.bdate_range("2026-01-01", "2026-02-06", tz="UTC")       # weekdays (holidays ignored here)
    m = ln.turn_of_month(idx, last_days=1, first_days=3)
    assert list(idx[m].strftime("%m-%d")) == ["01-01", "01-02", "01-05", "01-30", "02-02", "02-03", "02-04"]
    for k in (5, 21, 26):                                              # flags never depend on future rows
        np.testing.assert_array_equal(ln.turn_of_month(idx[:k], 1, 3), m[:k])
    on, intra = ln.overnight_intraday(O, C)
    assert np.isnan(on[0])
    np.testing.assert_allclose((1 + on[1:]) * (1 + intra[1:]), C[1:] / C[:-1])
