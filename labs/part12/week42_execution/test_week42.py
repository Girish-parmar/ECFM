import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import intraday_session, quote_path, trade_journal

ex = load("week42_execution", "execution")


# ----------------------------------------------------------------------------------- S5
def test_limit_price_never_overpays():
    assert ex.limit_price("BUY", 100.00, 100.10, 0.01, 0.0) == 100.00
    assert ex.limit_price("BUY", 100.00, 100.10, 0.01, 0.5) == 100.05
    assert ex.limit_price("BUY", 100.00, 100.10, 0.01, 0.33) == 100.03                 # 100.033 rounds DOWN
    assert ex.limit_price("SELL", 100.00, 100.10, 0.01, 0.33) == 100.07                # 100.067 rounds UP
    assert ex.limit_price("SELL", 100.00, 100.10, 0.01, 1.0) == 100.00
    assert ex.limit_price("BUY", 0.95, 1.05, 0.05, 0.5) == 1.00


def test_chase_by_hand():
    q = pd.DataFrame({"bid": [10.00, 10.00, 10.00, 10.01, 10.02], "ask": [10.04, 10.04, 10.04, 10.05, 10.06],
                      "trade": [10.04, 10.04, 10.04, 10.05, 10.06]})
    r = ex.simulate_chase("BUY", q, 0, schedule=(0.0, 1.0), wait=2, markout=1)
    # passive at 10.00 for seconds 1–2: no seller. Re-price at second 2's ask (10.04), watch seconds 3–4: the market
    # ran away (asks 10.05, 10.06). Cancelled; completing costs the ask of the last second watched (10.06)
    assert not r["filled"] and np.isnan(r["cost_bps"]) and r["seconds"] == 4
    assert r["all_in_bps"] == pytest.approx((10.06 - 10.02) / 10.02 * 1e4)
    q2 = q.assign(trade=[10.04, 10.00, 10.04, 10.05, 10.06])
    f = ex.simulate_chase("BUY", q2, 0, schedule=(0.0, 1.0), wait=2, markout=1)
    assert f["filled"] and f["price"] == 10.00 and f["seconds"] == 1 and f["aggr"] == 0.0     # a seller hit our bid
    assert f["cost_bps"] == pytest.approx(-0.02 / 10.02 * 1e4) and f["all_in_bps"] == f["cost_bps"]
    assert f["markout_bps"] == pytest.approx((10.02 - 10.00) / 10.00 * 1e4)


def test_passive_looks_cheap_until_you_count_non_fills():
    rep = ex.chase_report(quote_path(), {"passive": (0.0,), "cross": (1.0,), "chase": (0.0, 0.33, 0.67, 1.0)})
    assert list(rep.columns) == ["fill_rate", "avg_seconds", "avg_cost_bps", "avg_markout_bps", "avg_all_in_bps"]
    assert rep.loc["passive", "fill_rate"] < 0.8 and rep.loc["chase", "fill_rate"] == 1.0
    assert rep.loc["passive", "avg_cost_bps"] < rep.loc["chase", "avg_cost_bps"] < rep.loc["cross", "avg_cost_bps"]
    assert rep.loc["chase", "avg_all_in_bps"] < rep.loc["passive", "avg_all_in_bps"]      # non-fill is not free
    assert rep.loc["cross", "avg_all_in_bps"] > 1.5 and rep.loc["chase", "avg_seconds"] > rep.loc["cross", "avg_seconds"]


# ----------------------------------------------------------------------------------- S6
def test_schedules():
    tw = ex.twap_schedule(1003, "2025-06-02 09:30", "2025-06-02 10:30", 4)
    assert list(tw) == [251, 251, 251, 250] and tw.index[1] == pd.Timestamp("2025-06-02 09:45")
    prof = pd.Series([1.0, 2.0, 1.0])
    assert list(ex.vwap_schedule(10, prof)) == [3, 5, 2]                               # 2.5, 5, 2.5 → remainder to the first
    assert ex.vwap_schedule(1001, pd.Series([3.0, 1, 1, 3])).sum() == 1001
    assert ex.pov_child_qty(12_345, 0.1, 5_000) == 1234 and ex.pov_child_qty(1e6, 0.1, 500) == 500


def test_implementation_shortfall():
    assert ex.implementation_shortfall_bps("BUY", 100.0, [(100.02, 500), (100.05, 500)], fees=5.0) == pytest.approx(4.0)
    assert ex.implementation_shortfall_bps("SELL", 100.0, [(99.9, 100)]) == pytest.approx(10.0)
    assert ex.implementation_shortfall_bps("SELL", 100.0, [(100.1, 100)]) == pytest.approx(-10.0)   # price improvement


def test_slicing_beats_one_block_and_vwap_minimizes_impact():
    s = intraday_session()
    qty, end = 60_000, s.index[-1] + pd.Timedelta(minutes=1)
    single = pd.Series([qty], index=[s.index[0]])
    tw = ex.twap_schedule(qty, s.index[0], end, 13)
    vw = ex.vwap_schedule(qty, s["volume"].resample("30min").sum())
    fills = ex.execute_schedule("BUY", tw, s)
    assert len(fills) == 13 and sum(q for _, q in fills) == qty
    row = s.loc[tw.index[0]]
    assert fills[0][0] == pytest.approx(row["price"] * (1 + 0.02 * tw.iloc[0] / row["volume"]))
    flat = s.assign(price=100.0)                                                       # impact only, no market drift
    cost = {k: ex.implementation_shortfall_bps("BUY", 100.0, ex.execute_schedule("BUY", v, flat))
            for k, v in (("single", single), ("twap", tw), ("vwap", vw))}
    assert cost["single"] > 3 * cost["twap"] and cost["vwap"] < cost["twap"]
    assert ex.execute_schedule("SELL", pd.Series([0, 100], index=s.index[:2]), flat)[0][0] < 100   # sells pay down


# ----------------------------------------------------------------------------------- S7
def test_option_position_manager():
    m = ex.OptionPositionManager()
    base = {"credit": 2.0, "value": 1.5, "dte": 35, "short_deltas": [0.16, -0.15], "short_call_itm": False,
            "days_to_exdiv": None}
    assert m.evaluate(base) is None
    assert m.evaluate({**base, "value": 6.0}) == ("close", "stop")
    assert m.evaluate({**base, "value": 1.0}) == ("close", "take_profit")
    assert m.evaluate({**base, "value": 6.0, "dte": 0}) == ("close", "stop")               # stop has priority
    assert m.evaluate({**base, "dte": 0}) == ("close", "expiry")
    assert m.evaluate({**base, "short_call_itm": True, "days_to_exdiv": 1}) == ("close", "ex_dividend")
    assert m.evaluate({**base, "short_call_itm": True, "days_to_exdiv": 5}) is None
    assert m.evaluate({**base, "short_deltas": [0.35, -0.1]}) == ("adjust", "short_delta")
    assert m.evaluate({**base, "dte": 21}) == ("roll", "dte")


# ----------------------------------------------------------------------------------- S8
def test_journal_finds_the_edge_in_a_segment():
    j = trade_journal()
    j["r_multiple"] = ex.r_multiples(j)
    row = j.dropna(subset=["stop"]).iloc[0]
    assert row["r_multiple"] == pytest.approx(row["side"] * (row["exit"] - row["entry"]) / abs(row["entry"] - row["stop"]))
    assert j.loc[j["stop"].isna(), "r_multiple"].isna().all()
    by_tag = ex.journal_stats(j)
    assert list(by_tag.columns) == ["trades", "win_rate", "avg_win_R", "avg_loss_R", "expectancy_R"]
    assert by_tag["trades"].sum() == j["stop"].notna().sum()
    assert abs(by_tag.loc["breakout", "expectancy_R"] - by_tag.loc["pullback", "expectancy_R"]) < 0.2   # looks similar...
    both = ex.journal_stats(j, ["setup_tag", "regime"])
    assert both.index[0] == ("breakout", "trend") and both.loc[("breakout", "trend"), "expectancy_R"] > 0.4
    assert both.index[-1] == ("breakout", "chop") and both.loc[("breakout", "chop"), "expectancy_R"] < 0   # ...until split


def test_rule_violations():
    j = pd.DataFrame({"trade_id": [1, 2, 3], "entry_time": pd.to_datetime(["2025-06-02 09:29", "2025-06-02 10:00",
                                                                          "2025-06-02 15:59"]),
                      "stop": [9.5, np.nan, 9.5], "qty": [100, 100, 300], "planned_qty": [100, 100, 200]})
    v = ex.rule_violations(j)
    assert v.to_dict("records") == [{"trade_id": 1, "rule": "outside_hours"}, {"trade_id": 2, "rule": "no_stop"},
                                    {"trade_id": 3, "rule": "oversized"}]
    counts = ex.rule_violations(trade_journal())["rule"].value_counts()
    assert set(counts.index) == {"no_stop", "outside_hours", "oversized"}
