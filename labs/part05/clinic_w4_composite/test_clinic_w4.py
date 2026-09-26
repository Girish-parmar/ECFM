import numpy as np
import pandas as pd
import pytest
from _loader import load

cp = load("clinic_w4_composite", "composite")


def test_rolling_z_uses_only_the_past():
    x = pd.Series(np.random.default_rng(0).normal(size=400))
    z = cp.rolling_z(x, 100, 30)
    assert z.iloc[:29].isna().all() and z.iloc[29:].notna().all()
    w = x.iloc[200:300]
    assert z.iloc[299] == pytest.approx((x.iloc[299] - w.mean()) / w.std())
    pd.testing.assert_series_equal(cp.rolling_z(x.iloc[:250], 100, 30), z.iloc[:250])
    assert cp.rolling_z(pd.Series(np.ones(100)), 50, 10).isna().all()


def test_composite_signs_and_missing_components():
    idx = pd.RangeIndex(200)
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(size=200), index=idx)
    comp = pd.DataFrame({"fear": a, "breadth": -a, "sparse": np.nan}, index=idx)
    s = cp.composite(comp, {"fear": -1, "breadth": 1, "sparse": 1}, window=50, min_periods=20)
    expected = -cp.rolling_z(a, 50, 20)
    pd.testing.assert_series_equal(s, expected, check_names=False)      # aligned signs agree; empty column ignored


def test_quintiles_find_the_planted_link():
    comp, fwd = cp.demo_components()
    score = cp.composite(comp, {"vix_pct": -1, "breadth": 1, "cot_index": 1})
    q = cp.quintile_table(score, fwd)
    assert list(q.index) == [1, 2, 3, 4, 5] and list(q.columns) == ["count", "mean_fwd", "hit_rate"]
    assert q["count"].sum() == score.notna().sum()
    assert q["mean_fwd"].iloc[-1] - q["mean_fwd"].iloc[0] > 0.01 and q["mean_fwd"].is_monotonic_increasing


def test_stress_report():
    rep = cp.stress_report({"SPY": (100_000, 1.0), "TLT": (50_000, -0.2)})
    assert list(rep.columns) == ["shock", "SPY", "TLT", "TOTAL"]
    assert rep.index[0] == "1987-10-19 Black Monday" and rep["TOTAL"].is_monotonic_increasing
    row = rep.loc["Equity gap -10%"]
    assert row["SPY"] == pytest.approx(-10_000) and row["TLT"] == pytest.approx(1_000) and row["TOTAL"] == pytest.approx(-9_000)
    assert len(rep) == len(cp.HISTORICAL_SHOCKS) + len(cp.HYPOTHETICAL_SHOCKS)
    only = cp.stress_report({"SPY": (1.0, 1.0)}, {"x": -0.5})
    assert only.loc["x", "TOTAL"] == -0.5
