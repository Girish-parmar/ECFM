import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control
from _loader import load
from common import load_golden

gr = load("week18_groups", "groups")
G = load_golden("groups")
O, H, L, C, V = G["open"], G["high"], G["low"], G["close"], G["volume"]


def close_to(ours, ref):
    ours = np.asarray(ours, dtype=float)
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref), err_msg="warm-up (NaN) positions differ")
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-8, equal_nan=True)


# ------------------------------------------------------------------------------ edge tests
def test_forward_returns_enter_next_open():
    o = np.array([100.0, 101, 102, 104, 103, 105])
    f = gr.forward_returns(o, 2)
    np.testing.assert_allclose(f, [np.log(104 / 101), np.log(103 / 102), np.log(105 / 104), np.nan, np.nan, np.nan],
                               equal_nan=True)


def test_pattern_edge_detects_a_planted_edge_and_not_noise():
    rng = np.random.default_rng(1)
    n = 3000
    lr = rng.normal(0, 0.01, n)
    sig = np.zeros(n, dtype=bool)
    sig[rng.choice(n - 10, 150, replace=False)] = True
    lr[np.flatnonzero(sig) + 2] += 0.01                         # the bar after entry drifts up after a signal
    o = 100 * np.exp(np.cumsum(lr))
    r = gr.pattern_edge(sig, o, horizon=3, n_perm=500)
    assert r["n"] == 150 and r["edge"] > 0.005 and r["p_value"] < 0.01 and r["hit_rate"] > 0.6
    noise = rng.random(n) < 0.05
    assert gr.pattern_edge(noise, o, horizon=3, n_perm=500)["p_value"] > 0.05
    empty = gr.pattern_edge(np.zeros(n), o)
    assert empty["n"] == 0 and empty["p_value"] == 1.0


def test_bh_adjust_matches_scipy():
    p = np.array([0.001, 0.2, 0.03, 0.04, 0.5, 0.0001, 0.049, 0.9])
    np.testing.assert_allclose(gr.bh_adjust(p), false_discovery_control(p, method="bh"), rtol=1e-12)
    rng = np.random.default_rng(0)
    p = rng.random(1800)                                          # 1,800 null tests: ~90 "significant" at 5% ...
    assert (p < 0.05).sum() > 50 and (gr.bh_adjust(p) < 0.10).sum() == 0   # ... but none survive BH at 10%


# --------------------------------------------------------------------------- golden vs TA-Lib
def test_golden_direction():
    close_to(gr.kama(C, 10), G["kama10"])
    a, p, m = gr.adx(H, L, C, 14)
    close_to(a, G["adx14"]), close_to(p, G["plus_di14"]), close_to(m, G["minus_di14"])


def test_golden_momentum_and_volume():
    close_to(gr.cci(H, L, C, 20), G["cci20"])
    close_to(gr.roc(C, 10), G["roc10"])
    close_to(gr.mfi(H, L, C, V, 14), G["mfi14"])
    close_to(gr.ad_line(H, L, C, V), G["ad"])


# ------------------------------------------------------------------------------ supertrend
def test_supertrend_properties():
    line, d = gr.supertrend(H, L, C, 10, 3.0)
    assert np.isnan(line[:11]).all() and (d[:11] == 0).all()
    assert set(np.unique(d[11:])) <= {-1, 1}
    assert np.all(line[11:][d[11:] == 1] <= C[11:][d[11:] == 1])        # support below price in an up-trend
    assert np.all(line[11:][d[11:] == -1] >= C[11:][d[11:] == -1])      # resistance above price in a down-trend
    cut = 900
    l2, d2 = gr.supertrend(H[:cut], L[:cut], C[:cut], 10, 3.0)
    np.testing.assert_array_equal(d[:cut], d2)                          # no look-ahead
    up = np.linspace(100, 200, 300)
    assert gr.supertrend(up + 0.5, up - 0.5, up)[1][-1] == 1
    assert gr.supertrend(up[::-1] + 0.5, up[::-1] - 0.5, up[::-1])[1][-1] == -1


# ------------------------------------------------------------------------ volatility estimators
def brownian_bars(n=2000, sigma=0.01, sigma_on=0.0, steps=390, seed=0):
    """Bars from a finely sampled driftless Brownian path: exact-ish H/L, optional overnight gap."""
    rng = np.random.default_rng(seed)
    o, h, l, c = (np.empty(n) for _ in range(4))
    px = 0.0
    for t in range(n):
        px += rng.normal(0, sigma_on)
        path = px + np.concatenate([[0.0], np.cumsum(rng.normal(0, sigma / np.sqrt(steps), steps))])
        o[t], h[t], l[t], c[t] = path[0], path.max(), path.min(), path[-1]
        px = path[-1]
    return tuple(100 * np.exp(a) for a in (o, h, l, c))


def test_estimators_recover_sigma_and_range_is_more_efficient():
    o, h, l, c = brownian_bars()
    true = 0.01 * np.sqrt(252)
    for m in ["close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"]:
        v = gr.realized_vol(o, h, l, c, n=20, method=m)
        assert np.nanmean(v) == pytest.approx(true, rel=0.06), m
    spread = {m: np.nanstd(gr.realized_vol(o, h, l, c, 20, m)) for m in ["close", "parkinson", "garman_klass"]}
    assert spread["parkinson"] < 0.6 * spread["close"] and spread["garman_klass"] < 0.6 * spread["close"]


def test_yang_zhang_handles_overnight_gaps():
    o, h, l, c = brownian_bars(sigma_on=0.008, seed=3)
    total = np.sqrt(0.01 ** 2 + 0.008 ** 2) * np.sqrt(252)
    assert np.nanmean(gr.realized_vol(o, h, l, c, 20, "yang_zhang")) == pytest.approx(total, rel=0.06)
    assert np.nanmean(gr.realized_vol(o, h, l, c, 20, "parkinson")) < 0.85 * total      # range misses the gap


def test_realized_vol_lookbacks_and_errors():
    assert np.isnan(gr.realized_vol(O, H, L, C, 20, "parkinson")[:19]).all()
    assert not np.isnan(gr.realized_vol(O, H, L, C, 20, "parkinson")[19])
    assert np.isnan(gr.realized_vol(O, H, L, C, 20, "yang_zhang")[:20]).all()
    assert not np.isnan(gr.realized_vol(O, H, L, C, 20, "close")[20])
    with pytest.raises(ValueError):
        gr.realized_vol(O, H, L, C, 20, "magic")


# ----------------------------------------------------------------------- VWAP, profile, MTF
def intraday(days=3, seed=0):
    """5-minute bars 14:30–21:00 UTC on January days (09:30–16:00 New York)."""
    idx = pd.DatetimeIndex([], tz="UTC")
    for d in pd.date_range("2026-01-12", periods=days, freq="B"):
        idx = idx.append(pd.date_range(f"{d.date()} 14:30", f"{d.date()} 20:55", freq="5min", tz="UTC"))
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    return pd.DataFrame({"close": close, "volume": rng.integers(100, 1000, len(idx)).astype(float)}, index=idx)


def test_session_and_anchored_vwap():
    df = intraday()
    vw = gr.session_vwap(df.index, df["close"].to_numpy(), df["volume"].to_numpy())
    first_bars = np.flatnonzero(df.index.strftime("%H:%M") == "14:30")
    np.testing.assert_allclose(vw[first_bars], df["close"].to_numpy()[first_bars])     # resets every session
    day1 = df.iloc[:78]
    assert vw[77] == pytest.approx((day1["close"] * day1["volume"]).sum() / day1["volume"].sum())
    av = gr.anchored_vwap(df["close"].to_numpy(), df["volume"].to_numpy(), 100)
    assert np.isnan(av[:100]).all() and av[100] == pytest.approx(df["close"].iloc[100])


def test_volume_profile():
    price = np.concatenate([np.full(50, 10.0), np.full(30, 11.0), np.full(10, 12.0), np.full(10, 9.0)]) + 0.001
    vol = np.ones(100)
    vp = gr.volume_profile(price, vol, bins=4)          # edges 9.001, 9.751, 10.501, 11.251, 12.001
    assert vp["poc"] == pytest.approx(10.126)            # 50% of the volume sits at 10
    assert vp["val"] == pytest.approx(9.751) and vp["vah"] == pytest.approx(11.251)   # + the 11 bin -> 80%


def test_align_higher_tf_has_no_lookahead():
    df = intraday()
    ident = gr.align_higher_tf(df, lambda x: x, "1h", "30min")
    t = pd.Timestamp("2026-01-12 15:35", tz="UTC")
    assert ident[t] == df.loc["2026-01-12 15:25", "close"]         # the 14:30–15:30 hour, closed at 15:30
    assert np.isnan(ident[pd.Timestamp("2026-01-12 15:25", tz="UTC")])   # first hour not finished yet
    ema = load("week17_core", "core").ema
    full = gr.align_higher_tf(df, lambda x: ema(x, 3))
    for cut in [37, 101, 170]:
        part = gr.align_higher_tf(df.iloc[:cut], lambda x: ema(x, 3))
        pd.testing.assert_series_equal(full.iloc[:cut], part, check_names=False)
    naive = df["close"].resample("1h", label="left", closed="left", offset="30min").last().reindex(df.index, method="ffill")
    assert naive[t] != ident[t]                                     # the unshifted version peeks at the future
