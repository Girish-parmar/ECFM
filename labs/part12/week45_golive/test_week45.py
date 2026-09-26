import pandas as pd
import pytest
from _loader import load

gl = load("week45_golive", "golive")
GOOD = {"days": 12, "sev1_incidents": 0, "incidents": 1, "open_breaks": 0, "slippage_ratio": 1.1,
        "inside_band": 0.9, "drift_alarm": False}


# ----------------------------------------------------------------------------------- S17
def test_rollout_stages():
    assert gl.rollout_decision("paper", GOOD) == ("promote", "shadow", [])
    assert gl.rollout_decision("small_live", GOOD) == ("hold", "small_live", ["too few days"])     # needs 20
    assert gl.rollout_decision("shadow", {**GOOD, "open_breaks": 2, "slippage_ratio": 2.0}) == \
        ("hold", "shadow", ["open reconciliation breaks", "slippage above model"])
    assert gl.rollout_decision("small_live", {**GOOD, "drift_alarm": True, "inside_band": 0.4}) == \
        ("demote", "shadow", ["outside band", "drift alarm"])
    assert gl.rollout_decision("paper", {**GOOD, "incidents": 3})[:2] == ("demote", "paper")      # floor
    assert gl.rollout_decision("scaled_live", {**GOOD, "days": 400}) == ("hold", "scaled_live", [])


def test_capital_ramp():
    plan = {"min": 0.1, "max": 1.0, "step": 0.1, "weeks_per_step": 2, "max_dd": -0.05}
    assert gl.capital_ramp(0.1, 1, -0.01, plan) == 0.1
    assert gl.capital_ramp(0.1, 2, -0.01, plan) == pytest.approx(0.2)
    assert gl.capital_ramp(0.95, 4, 0.0, plan) == 1.0
    assert gl.capital_ramp(0.4, 4, -0.08, plan) == 0.2                                  # cut after a bad drawdown
    assert gl.capital_ramp(0.15, 0, -0.08, plan) == 0.1


# ----------------------------------------------------------------------------------- S18
def test_calendar():
    assert gl.session_times("2025-07-04") is None and gl.session_times("2025-06-07") is None      # holiday, Saturday
    o, c = gl.session_times("2025-07-03")
    assert o == pd.Timestamp("2025-07-03 09:30") and c == pd.Timestamp("2025-07-03 13:00")         # early close
    assert gl.session_times("2025-06-02 14:00")[1] == pd.Timestamp("2025-06-02 16:00")


def test_premarket_checklist_blocks_trading():
    ok = {"gateway_connected": True, "data_age_s": 1.0, "open_breaks": 0, "kill_switch_tested_today": True,
          "risk_limits_loaded": True, "last_backup_age_h": 10, "disk_free_pct": 40}
    r = gl.premarket_check(ok)
    assert r["may_trade"] and all(r["results"].values()) and r["blocking_failures"] == [] == r["warnings"]
    warn = gl.premarket_check({**ok, "disk_free_pct": 5, "last_backup_age_h": 30})
    assert warn["may_trade"] and warn["warnings"] == ["backup_ok", "disk_ok"]
    stop = gl.premarket_check({**ok, "data_age_s": 60, "kill_switch_tested_today": False})
    assert not stop["may_trade"] and stop["blocking_failures"] == ["data_fresh", "kill_switch_tested"]


# ----------------------------------------------------------------------------------- S19
def test_incident_lifecycle_and_sla():
    inc = gl.Incident("runaway order loop", "sev1", "2025-06-02 10:00")
    assert inc.phase == "detected"
    with pytest.raises(ValueError):
        inc.advance("recovered", "2025-06-02 10:02")                                    # cannot skip containment
    inc.advance("contained", "2025-06-02 10:03")
    with pytest.raises(ValueError, match="forward"):
        inc.advance("diagnosed", "2025-06-02 09:59")
    assert inc.sla_breaches() == []
    for ph, t in (("diagnosed", "10:20"), ("recovered", "10:40"), ("reconciled", "11:05"), ("closed", "11:30")):
        inc.advance(ph, f"2025-06-02 {t}")
    assert inc.phase == "closed" and inc.sla_breaches() == ["reconciled"]                 # 65 min > 60
    slow = gl.Incident("stale feed", "sev2", "2025-06-02 10:00")
    slow.advance("contained", "2025-06-02 10:45")
    assert slow.sla_breaches() == ["contained"]


def test_postmortem_timeline():
    recs = [{"ts": "2025-06-02 10:05", "event": "strategy_paused"}, {"ts": "2025-06-02 09:00", "event": "premarket_ok"},
            {"ts": "2025-06-02 10:01", "event": "anomaly_runaway_orders"}, {"ts": "2025-06-02 12:00", "event": "eod"}]
    assert gl.postmortem_timeline(recs, "2025-06-02 10:00", "2025-06-02 11:00") == [
        (pd.Timestamp("2025-06-02 10:01"), "anomaly_runaway_orders"), (pd.Timestamp("2025-06-02 10:05"), "strategy_paused")]


# ----------------------------------------------------------------------------------- S20–S24
def test_go_live_review():
    ev = {"strategies": {"mom": {"dsr": 0.97, "pbo": 0.2}, "pairs": {"dsr": 0.96, "pbo": 0.3}}, "runbooks": 12,
          "drills": {"broker_disconnect": True, "runaway": True}, "security_complete": True, "rollout_plan": True}
    assert gl.go_live_review(ev) == (True, [])
    bad = {**ev, "strategies": {**ev["strategies"], "ml": {"dsr": 0.80, "pbo": 0.6}}, "runbooks": 7,
           "drills": {"broker_disconnect": False, "runaway": True}}
    assert gl.go_live_review(bad) == (False, ["drill: broker_disconnect", "gate: ml", "runbooks"])


def test_weekly_decisions():
    inside = pd.DataFrame({"mom": [True] * 5, "pairs": [True, True, False, True, True], "ml": [False] * 4 + [True]})
    assert gl.weekly_decisions(inside, {"pairs": 0, "mom": 0}) == {"mom": "scale", "pairs": "hold", "ml": "demote"}
    assert gl.weekly_decisions(inside, {"mom": 2})["mom"] == "demote"


def test_graduation_from_the_audit_chain():
    recs = [{"event": "risk_decision", "data": {"order_id": "O1", "approved": True}},
            {"event": "order", "data": {"id": "O1"}},
            {"event": "risk_decision", "data": {"order_id": "O2", "approved": False}},
            {"event": "order", "data": {"id": "O2"}},                                    # sent although rejected!
            {"event": "order", "data": {"id": "O3"}},                                    # never checked
            {"event": "risk_decision", "data": {"order_id": "O3", "approved": True}}]    # approval came too late
    assert gl.orders_without_risk_approval(recs) == ["O2", "O3"]
    ok, why = gl.graduation(recs, {"2025-06-02": 0, "2025-06-03": 1}, set(), {"mom": 0.97, "pairs": None})
    assert not ok and why == ["kill switch not demonstrated", "orders bypassed risk: O2,O3",
                              "unexplained breaks: 2025-06-03", "missing DSR: pairs"]
    clean = recs[:2] + [{"event": "killswitch_trip", "data": {}}]
    assert gl.graduation(clean, {"2025-06-03": 1}, {"2025-06-03"}, {"mom": 0.97}) == (True, [])
