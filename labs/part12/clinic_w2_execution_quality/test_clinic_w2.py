import pandas as pd
from _loader import load
from common import intraday_session, trade_journal

eq = load("clinic_w2_execution_quality", "execution_quality")


def test_aggressiveness_sweep():
    sw = eq.aggressiveness_sweep()
    assert list(sw.index) == list(eq.SCHEDULES) and "avg_all_in_bps" in sw
    assert sw["avg_all_in_bps"].idxmin() == "chase" and sw["avg_all_in_bps"].idxmax() == "cross"
    assert sw.loc["passive", "avg_cost_bps"] < sw.loc["chase", "avg_cost_bps"]           # cheap only when it fills
    assert sw.loc[["passive", "mid", "cross"], "fill_rate"].is_monotonic_increasing and sw.loc["passive", "fill_rate"] < 0.8


def test_pov_schedule_uses_the_previous_minute():
    s = intraday_session()
    pov = eq.pov_schedule(60_000, s, 0.1)
    assert pov.iloc[0] == 0 and pov.iloc[1] == int(0.1 * s["volume"].iloc[0]) and pov.sum() == 60_000
    tiny = eq.pov_schedule(10**9, s, 0.1)
    assert tiny.sum() < 10**9                                                           # POV may not finish


def test_algo_tca():
    t = eq.algo_tca()
    assert list(t.index) == ["single", "twap", "vwap", "pov"] and (t["completion"] == 1).all()
    assert t["mean_bps"].idxmax() == "single" and t.loc["single", "mean_bps"] > 3 * t.loc["twap", "mean_bps"]
    assert t["std_bps"].idxmin() == "pov"                                               # finishes early: less drift


def test_urgency_and_journal():
    assert eq.urgency_schedule("stop") == (1.0,) and eq.urgency_schedule("unknown") == eq.URGENCY["rebalance"]
    jf = eq.journal_findings(trade_journal())
    assert jf["best"] == ("breakout", "trend") and jf["worst"] == ("breakout", "chop")
    assert set(jf["violations"]) == {"no_stop", "outside_hours", "oversized"}
    assert isinstance(jf["by_setup_regime"], pd.DataFrame)
