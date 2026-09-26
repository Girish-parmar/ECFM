import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import arrays, synthetic_ohlcv

pt = load("week19_patterns", "patterns")
Pivot = pt.Pivot


# ------------------------------------------------------------------------------ levels
def test_pivot_points():
    h, l, c = np.array([110.0, 0]), np.array([100.0, 0]), np.array([105.0, 0])
    cl = pt.pivot_points(h, l, c, "classic")
    assert set(cl) == {"P", "R1", "S1", "R2", "S2"} and np.isnan(cl["P"][0])
    assert (cl["P"][1], cl["R1"][1], cl["S1"][1], cl["R2"][1], cl["S2"][1]) == (105, 110, 100, 115, 95)
    fb = pt.pivot_points(h, l, c, "fibonacci")
    assert fb["R1"][1] == pytest.approx(108.82) and fb["S3"][1] == pytest.approx(95)
    cm = pt.pivot_points(h, l, c, "camarilla")
    assert cm["R4"][1] == pytest.approx(110.5) and cm["S1"][1] == pytest.approx(105 - 11 / 12)
    wd = pt.pivot_points(h, l, c, "woodie")
    assert set(wd) == {"P", "R1", "S1"} and wd["P"][1] == 105
    with pytest.raises(ValueError):
        pt.pivot_points(h, l, c, "gann")


def test_opening_range_is_known_only_after_it_completes():
    idx = pd.date_range("2026-01-12 14:30", "2026-01-12 15:25", freq="5min", tz="UTC")     # 09:30–10:25 NY
    idx = idx.append(pd.date_range("2026-01-13 14:30", "2026-01-13 15:00", freq="5min", tz="UTC"))
    high = np.arange(len(idx), dtype=float) + 100
    low = high - 1
    high[2] = 120                                                   # inside day 1's first 30 minutes
    orh, orl = pt.opening_range(idx, high, low, minutes=30)
    assert np.isnan(orh[:6]).all()                                  # 09:30..09:55 bars: range not finished
    assert orh[6] == 120 and orl[6] == 99 and orh[11] == 120        # from 10:00 on
    d2 = np.flatnonzero(idx.date == pd.Timestamp("2026-01-13").date())
    assert np.isnan(orh[d2[:6]]).all() and orh[d2[6]] == high[d2[:6]].max()


def test_kde_levels():
    rng = np.random.default_rng(0)
    swings = np.concatenate([rng.normal(100, 0.3, 30), rng.normal(110, 0.3, 15), rng.normal(95, 0.3, 5)])
    lv = pt.kde_levels(swings, bandwidth=0.5, n_levels=2)
    assert lv[0] == pytest.approx(100, abs=0.3) and lv[1] == pytest.approx(110, abs=0.3)


# ------------------------------------------------------------------------------ swings
def test_zigzag_on_a_hand_path():
    close = np.array([100, 104, 110, 108, 103, 98, 101, 106, 112, 115, 109], dtype=float)
    pv = pt.zigzag(close, close, 0.05)
    assert [(p.idx, p.confirm_idx, p.kind) for p in pv] == [(0, 2, -1), (2, 4, 1), (5, 7, -1), (9, 10, 1)]
    assert all(p.confirm_idx > p.idx for p in pv)


def test_fractals():
    h = np.array([1, 2, 5, 3, 2, 4, 6, 4, 3], dtype=float)
    l = h - 1  # noqa: E741
    got = [(p.idx, p.confirm_idx, p.kind) for p in pt.fractals(h, l, 2)]
    assert got == [(2, 4, 1), (4, 6, -1), (6, 8, 1)]


def test_swings_have_no_lookahead():
    """Pivots confirmed before a cut must be identical when the future is removed (the truncation audit)."""
    _, h, l, c, _ = arrays(synthetic_ohlcv(800, seed=3))
    for fn in (lambda a, b: pt.zigzag(a, b, 0.04), lambda a, b: pt.fractals(a, b, 3)):
        full = fn(h, l)
        for cut in (200, 431, 650):
            part = fn(h[:cut], l[:cut])
            assert [p for p in full if p.confirm_idx < cut] == [p for p in part if p.confirm_idx < cut]


# ----------------------------------------------------------------------- chart patterns
def P(idx, price, kind, lag=2):
    return Pivot(idx, idx + lag, float(price), kind)


def test_double_tops_and_bottoms():
    pv = [P(0, 90, -1), P(10, 110, 1), P(15, 100, -1), P(22, 110.4, 1), P(30, 95, -1), P(33, 111, 1)]
    got = pt.double_tops_bottoms(pv, tol=0.5)
    assert len(got) == 1 and got[0]["kind"] == "double_top" and got[0]["neckline"] == 100
    assert got[0]["known_from"] == 24 and got[0]["first"].idx == 10
    bottoms = pt.double_tops_bottoms([P(0, 90, -1), P(5, 100, 1), P(12, 90.2, -1)], tol=0.5)
    assert [b["kind"] for b in bottoms] == ["double_bottom"]
    assert pt.double_tops_bottoms([P(0, 90, -1), P(2, 100, 1), P(3, 90.2, -1)], tol=0.5) == []   # too close


def test_head_and_shoulders():
    pv = [P(0, 105, 1), P(5, 100, -1), P(10, 112, 1), P(15, 101, -1), P(20, 105.3, 1)]
    hs = pt.head_shoulders(pv, tol=0.5)
    assert len(hs) == 1 and hs[0]["kind"] == "head_shoulders" and hs[0]["head"].price == 112
    assert hs[0]["slope"] == pytest.approx(0.1) and hs[0]["known_from"] == 22
    inv = [P(0, 95, -1), P(5, 100, 1), P(10, 88, -1), P(15, 100, 1), P(20, 95.2, -1)]
    assert pt.head_shoulders(inv, tol=0.5)[0]["kind"] == "inverse_head_shoulders"
    assert pt.head_shoulders(pv, tol=0.1) == []                     # shoulders not level enough


# ---------------------------------------------------------------------------------- actions
def test_crossovers_ties_and_nan():
    a = np.array([1, 2, 3, 3, 2, np.nan, 4], dtype=float)
    b = np.array([2, 2, 2, 3, 3, 3, 3], dtype=float)
    np.testing.assert_array_equal(pt.crossover(a, b), [0, 0, 1, 0, 0, 0, 0])   # 2==2 is "not above"
    np.testing.assert_array_equal(pt.crossunder(a, b), [0, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(pt.crossover(a, 2.5), [0, 0, 1, 0, 0, 0, 0])


def test_bar_actions():
    o = np.array([10, 12.5, 11, 11.5])
    h = np.array([12, 13.0, 12.5, 12.0])
    l = np.array([9, 12.0, 10.5, 11.0])  # noqa: E741
    np.testing.assert_array_equal(pt.gap_up(o, h), [0, 1, 0, 0])
    np.testing.assert_array_equal(pt.inside_bar(h, l), [0, 0, 0, 1])
    hi = np.array([5, 5, 5, 5, 5, 5, 5.0])
    lo = np.array([1, 2, 3, 3.5, 4, 4.2, 4.5])
    np.testing.assert_array_equal(pt.nr_n(hi, lo, 4), [0, 0, 0, 1, 1, 1, 1])
    x = np.array([1, 3, 2, 3, 4, 1, 5.0])
    np.testing.assert_array_equal(pt.new_high(x, 2), [0, 0, 0, 0, 1, 0, 1])   # 3 is not above 3
    cond = np.array([0, 1, 1, 0, 1, 1, 1], dtype=bool)
    np.testing.assert_array_equal(pt.consecutive(cond), [0, 1, 2, 0, 1, 2, 3])
    np.testing.assert_array_equal(pt.bars_since(cond), [np.nan, 0, 0, 1, 0, 0, 0])


def test_condition_combinators():
    ctx = {"x": np.array([0, 5, 0, 0, 0, 5, 5, 5, 0], dtype=float)}
    hot = pt.Condition(lambda d: d["x"] > 1, "x>1")
    np.testing.assert_array_equal(hot.within(3)(ctx), [0, 1, 1, 1, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(hot.confirm(2)(ctx), [0, 0, 0, 0, 0, 0, 1, 1, 0])
    rule = hot.confirm(2) & ~pt.Condition(lambda d: d["x"] > 10, "x>10")
    assert rule.name == "(x>1 for 2 bars AND NOT x>10)" and hot.within(3).name == "x>1 within 3"
    np.testing.assert_array_equal(rule(ctx), hot.confirm(2)(ctx))
