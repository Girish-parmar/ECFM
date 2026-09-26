import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays as np_arrays
from _loader import load
from common import load_golden

core = load("week17_core", "core")
G = load_golden("core")
O, H, L, C, V = G["open"], G["high"], G["low"], G["close"], G["volume"]
prices = np_arrays(np.float64, st.integers(40, 300), elements=st.floats(1, 1e4))


def close_to(ours, ref):
    """Golden comparison: same NaN positions, values within 1e-8."""
    ours = np.asarray(ours, dtype=float)
    assert ours.shape == ref.shape
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref), err_msg="warm-up (NaN) positions differ")
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-8, equal_nan=True)


# ---------------------------------------------------------------------------- registry
def test_registry_and_catalog():
    for name in ["sma", "ema", "rsi", "atr", "macd", "bbands", "stoch", "willr", "obv", "doji", "engulfing", "hammer"]:
        assert name in core.REGISTRY, f"{name} not registered"
    spec, fn = core.REGISTRY["macd"]
    assert fn is core.macd and spec.group == "momentum" and spec.outputs == ("macd", "signal", "hist")
    assert core.default_lookback("macd") == 33 and core.default_lookback("stoch") == 17
    with pytest.raises(ValueError):
        core.indicator("sma", "direction", lambda n: n - 1, {"n": (20, 2, 200)})(lambda x: x)
    md = core.catalog_markdown().splitlines()
    assert md[0] == "| Name | Group | Outputs | Params | Look-back |"
    assert "| sma | direction | value | n=20 | 19 |" in md
    assert "| bbands | volatility | upper, middle, lower | n=20, k=2.0 | 19 |" in md
    groups = [line.split("|")[2].strip() for line in md[2:]]
    assert groups == sorted(groups)


def test_helpers():
    x = np.array([100.0, 110.0, 99.0])
    np.testing.assert_allclose(core.log_returns(x), [np.nan, np.log(1.1), np.log(0.9)], equal_nan=True)
    r = core.rank_pct(np.array([3.0, 1.0, 2.0, 5.0, 4.0]), 3)
    np.testing.assert_allclose(r, [np.nan, np.nan, 2 / 3, 1.0, 2 / 3], equal_nan=True)


# ------------------------------------------------------------------ golden vs TA-Lib
def test_golden_moving_averages_rsi_atr():
    close_to(core.sma(C, 20), G["sma20"])
    close_to(core.ema(C, 20), G["ema20"])
    close_to(core.rsi(C, 14), G["rsi14"])
    close_to(core.atr(H, L, C, 14), G["atr14"])


def test_golden_macd_bbands_stoch():
    m, s, h = core.macd(C)
    close_to(m, G["macd"]), close_to(s, G["macd_signal"]), close_to(h, G["macd_hist"])
    u, mid, lo = core.bbands(C, 20, 2.0)
    close_to(u, G["bb_upper"]), close_to(mid, G["bb_middle"]), close_to(lo, G["bb_lower"])
    k, d = core.stoch(H, L, C)
    close_to(k, G["stoch_k"]), close_to(d, G["stoch_d"])


def test_golden_willr_obv_engulfing():
    close_to(core.willr(H, L, C, 14), G["willr14"])
    close_to(core.obv(C, V), G["obv"])
    np.testing.assert_array_equal(core.engulfing(O, H, L, C), G["engulfing"].astype(np.int8))


def test_seeding_gap_with_pandas_closes_over_time():
    """pandas ewm(adjust=False) seeds with the first value: different at first, same later (S2 teaching point)."""
    import pandas as pd
    ours = core.ema(C, 20)
    pdv = pd.Series(C).ewm(span=20, adjust=False).mean().to_numpy()
    assert abs(ours[19] - pdv[19]) > 1e-3 and abs(ours[400] - pdv[400]) < 1e-8


def test_rsi_edge_cases():
    up = np.arange(1.0, 40.0)
    assert np.all(core.rsi(up, 14)[14:] == 100.0)
    assert np.isnan(core.rsi(up[:14], 14)).all()


# ---------------------------------------------------------------- streaming ≡ vectorized
def _stream(obj, xs):
    return np.array([np.nan if (v := obj.update(float(x))) is None else v for x in xs])


@settings(max_examples=60, deadline=None)
@given(prices)
def test_streaming_ema_rsi_match_vectorized(x):
    np.testing.assert_allclose(_stream(core.StreamingEMA(20), x), core.ema(x, 20), rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(_stream(core.StreamingRSI(14), x), core.rsi(x, 14), rtol=1e-9, atol=1e-9,
                               equal_nan=True)


@settings(max_examples=60, deadline=None)
@given(prices)
def test_rolling_max_and_bollinger_match_vectorized(x):
    ref = np.full(x.shape, np.nan)
    ref[9:] = np.lib.stride_tricks.sliding_window_view(x, 10).max(axis=1)
    np.testing.assert_array_equal(_stream(core.RollingMax(10), x), ref)
    b = core.StreamingBollinger(20, 2.0)
    got = np.array([[np.nan] * 3 if (v := b.update(float(xi))) is None else v for xi in x])
    u, m, lo = core.bbands(x, 20, 2.0)
    # float equivalence is judged relative to the price scale (1e-7 of the largest price)
    np.testing.assert_allclose(got, np.column_stack([u, m, lo]), rtol=0, atol=1e-7 * np.abs(x).max(), equal_nan=True)


def test_peek_does_not_change_state():
    s = core.StreamingEMA(3)
    assert s.update(1.0) is None and s.update(2.0) is None
    assert s.peek(6.0) == 3.0 and s.value is None               # forming bar: seed would be (1+2+6)/3
    assert s.update(3.0) == 2.0
    before = s.value
    assert s.peek(10.0) == pytest.approx(0.5 * 10 + 0.5 * 2.0) and s.value == before


# ------------------------------------------------------------------------- candlesticks
def _ohlc(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def test_anatomy_and_doji():
    o, h, l, c = _ohlc([[10, 12, 9, 11], [10, 10, 10, 10], [10, 11, 9, 10.05]])
    a = core.anatomy(o, h, l, c)
    np.testing.assert_allclose(a["body"], [1, 0, 0.05])
    np.testing.assert_allclose(a["upper"], [1, 0, 0.95]) and np.testing.assert_allclose(a["lower"], [1, 0, 1])
    np.testing.assert_allclose(a["body_pct"], [1 / 3, 0, 0.025])
    np.testing.assert_array_equal(core.doji(o, h, l, c), [0, 0, 1])       # flat bar (range 0) is not a doji


def test_engulfing_rules():
    o, h, l, c = _ohlc([[10, 10.5, 9, 9.2], [9.1, 10.8, 9.0, 10.4],       # black then white engulfing -> +1
                        [10.5, 11, 10.3, 10.6], [10.7, 10.8, 10.0, 10.2],  # white then black engulfing -> -1
                        [10.2, 10.4, 9.9, 10.0], [10.05, 10.3, 9.95, 10.1]])  # small white inside: 0
    np.testing.assert_array_equal(core.engulfing(o, h, l, c), [0, 1, 0, -1, 0, 0])


def test_hammer_needs_a_prior_decline():
    closes = np.linspace(20, 11, 12)
    rows = [[x + 0.2, x + 0.3, x - 0.1, x] for x in closes]
    rows.append([10.8, 10.85, 9.5, 10.75])                                # hammer after the decline
    o, h, l, c = _ohlc(rows)
    sig = core.hammer(o, h, l, c, trend_n=5)
    assert sig[-1] == 1 and sig[:-1].sum() == 0
    up = [[x - 0.2, x + 0.1, x - 0.3, x] for x in np.linspace(11, 20, 12)] + [[20.1, 20.15, 18.8, 20.05]]
    assert core.hammer(*_ohlc(up), trend_n=5)[-1] == 0                    # same shape in an uptrend: not a hammer


def test_morning_and_evening_star():
    o, h, l, c = _ohlc([[12, 12.1, 9.9, 10], [9.7, 9.9, 9.4, 9.75], [9.9, 11.6, 9.8, 11.5]])
    np.testing.assert_array_equal(core.morning_evening_star(o, h, l, c), [0, 0, 1])
    o, h, l, c = _ohlc([[10, 12.1, 9.9, 12], [12.3, 12.6, 12.2, 12.35], [12.1, 12.2, 10.4, 10.5]])
    np.testing.assert_array_equal(core.morning_evening_star(o, h, l, c), [0, 0, -1])
    o, h, l, c = _ohlc([[12, 12.1, 9.9, 10], [9.7, 9.9, 9.4, 9.75], [9.9, 10.9, 9.8, 10.8]])   # closes below midpoint
    np.testing.assert_array_equal(core.morning_evening_star(o, h, l, c), [0, 0, 0])
