import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry
from _loader import load
from common import latency_samples

mo = load("week43_monitoring", "monitoring")


def clock():
    t = [0.0]

    def now():
        t[0] += 1.0
        return t[0]
    return now


# ----------------------------------------------------------------------------------- S9
def test_structured_logs_follow_one_trade():
    log = mo.StructuredLogger(now=clock())
    log.log("INFO", "signal", "c1", strategy="mom", symbol="SPY")
    log.log("INFO", "signal", "c2", strategy="rev", symbol="QQQ")
    log.log("INFO", "order", "c1", qty=100)
    log.log("INFO", "fill", "c1", price=501.2)
    story = log.trace("c1")
    assert [r["msg"] for r in story] == ["signal", "order", "fill"] and story[2]["price"] == 501.2
    assert story[0] == {"ts": 1.0, "level": "INFO", "msg": "signal", "cid": "c1", "strategy": "mom", "symbol": "SPY"}
    assert log.lines[0].startswith('{"cid": "c1"')                                     # sorted keys: greppable


def test_prometheus_metrics():
    reg = CollectorRegistry()
    m = mo.make_metrics(reg)
    m["orders"].labels(strategy="mom", broker="ib").inc(2)
    m["rejects"].labels(broker="ib").inc()
    m["latency"].observe(0.004)
    m["equity"].labels(account="paper").set(101_250)
    text = mo.exposition(reg)
    assert 'qf_orders_total{broker="ib",strategy="mom"} 2.0' in text
    assert 'qf_signal_to_submit_seconds_bucket{le="0.005"} 1.0' in text and 'le="0.001"} 0.0' in text
    assert 'qf_equity_usd{account="paper"} 101250.0' in text and "qf_rejects_total" in text


def test_alerts_and_slo():
    rules = [("stale_data", lambda s: s["staleness"] > 5, "critical"),
             ("broker_down", lambda s: not s["connected"], "critical"),
             ("reject_rate", lambda s: s["rejects"] / max(s["orders"], 1) > 0.2, "warning"),
             ("loop_lag", lambda s: s["loop_lag"] > 0.05, "info")]
    state = {"staleness": 9.0, "connected": False, "rejects": 3, "orders": 10, "loop_lag": 0.1}
    assert mo.evaluate_alerts(state, rules) == [("broker_down", "critical"), ("stale_data", "critical"),
                                                ("reject_rate", "warning"), ("loop_lag", "info")]
    assert mo.evaluate_alerts({"staleness": 1, "connected": True, "rejects": 0, "orders": 5, "loop_lag": 0}, rules) == []
    fresh = pd.Series([1.0] * 9990 + [10.0] * 10)
    rep = mo.slo_report(fresh, threshold=5.0, target=0.999)
    assert rep["compliance"] == pytest.approx(0.999) and rep["met"] and rep["budget_used"] == pytest.approx(1.0)
    assert not mo.slo_report(pd.Series([1.0] * 98 + [10.0] * 2), 5.0)["met"]


# ----------------------------------------------------------------------------------- S10
def test_hash_chained_audit_detects_tampering():
    audit = mo.AuditLog(now=clock())
    h1 = audit.append("intent", strategy="mom", qty=100)
    audit.append("risk_decision", approved=True)
    audit.append("order", id="O1")
    assert audit.records[0]["prev"] == "0" * 64 and audit.records[1]["prev"] == h1 and audit.verify() == -1
    assert audit.head() == audit.records[-1]["hash"] and len(h1) == 64
    audit.records[1]["data"]["approved"] = False                                        # rewrite history...
    assert audit.verify() == 1                                                           # ...pinpointed
    audit.records[1]["data"]["approved"] = True
    assert audit.verify() == -1
    audit.records[1]["hash"] = mo._digest({k: v for k, v in audit.records[1].items() if k != "hash"} | {"x": 1})
    assert audit.verify() == 1


def test_reconciliation():
    breaks = mo.reconcile_positions({"SPY": 100, "QQQ": -50, "IWM": 10}, {"SPY": 100, "QQQ": -40, "TLT": 20})
    assert breaks == [{"symbol": "IWM", "internal": 10, "broker": 0.0, "diff": -10},
                      {"symbol": "QQQ", "internal": -50, "broker": -40, "diff": 10},
                      {"symbol": "TLT", "internal": 0.0, "broker": 20, "diff": 20}]
    assert mo.reconcile_positions({"SPY": 100}, {"SPY": 100.0}) == []
    assert mo.reconcile_orders({"O1", "O2"}, {"O2", "TWS-99"}) == {"unknown_at_broker": ["O1"],
                                                                   "orphan_at_broker": ["TWS-99"]}


def test_four_eyes_approvals():
    audit = mo.AuditLog(now=clock())
    q = mo.ApprovalQueue(audit)
    aid = q.request("limit_change", {"max_gross": 2.0}, "alice", "new strategy added")
    with pytest.raises(ValueError, match="four-eyes"):
        q.decide(aid, "alice", True, "ok")
    q.decide(aid, "bob", True, "reviewed")
    assert q.items[aid]["status"] == "approved" and q.items[aid]["approver"] == "bob"
    with pytest.raises(ValueError):
        q.decide(aid, "carol", False, "late")
    assert [r["event"] for r in audit.records] == ["approval_requested", "approval_granted"] and audit.verify() == -1
    assert mo.needs_approval({"qty": -300, "price": 500}, 100_000) and not mo.needs_approval({"qty": 100, "price": 500}, 100_000)


# ----------------------------------------------------------------------------------- S11
def test_robust_z_flags_the_spike_only():
    rz = mo.RobustZ()
    flags = [i for i, x in enumerate(latency_samples()) if rz.update(x)[1]]
    assert flags == [200]                                                               # lesson plan: exactly one
    early = mo.RobustZ(min_obs=50)
    assert all(not early.update(x)[1] for x in [1.0] * 10 + [100.0])                    # not enough history yet


def test_cusum_catches_drift_that_z_scores_miss():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(0.8, 1, 200)]
    c = mo.Cusum(0.0, 1.0, k=0.5, h=8.0)
    alarms = [i for i, v in enumerate(x) if c.update(v)]
    assert alarms and 300 <= alarms[0] < 340                                            # soon after the drift starts
    rz = mo.RobustZ()
    assert not any(rz.update(v)[1] for v in x)                                           # no single value is odd
    assert mo.fat_tail_error(-6_000, 1_000) and not mo.fat_tail_error(4_000, 1_000)


def test_controller_escalation_and_resume():
    audit = mo.AuditLog(now=clock())
    q = mo.ApprovalQueue(audit)
    ctl = mo.Controller(["mom", "rev", "pairs"], audit, q)
    assert ctl.handle("latency_spike", "mom") == "alert" and not ctl.paused
    assert ctl.handle("stale_data", "mom") == "pause_strategy" and ctl.paused == {"mom"}
    assert ctl.handle("martian_attack") == "alert"
    aid = q.request("resume", {"strategy": "mom"}, "alice", "feed back")
    assert not ctl.resume("mom", aid)                                                    # not approved yet
    q.decide(aid, "bob", True, "checked feed")
    other = q.request("resume", {"strategy": "rev"}, "alice", "x")
    q.decide(other, "bob", True, "x")
    assert not ctl.resume("mom", other) and ctl.resume("mom", aid) and ctl.paused == set()
    assert ctl.handle("fat_tail_pnl") == "pause_all_and_reconcile" and ctl.needs_reconcile
    assert ctl.paused == {"mom", "rev", "pairs"}
    assert ctl.handle("runaway_order_rate", "rev") == "trip_kill_switch" and ctl.killed
    assert not ctl.resume("mom", aid)                                                    # nothing resumes after a kill
    actions = [r["data"]["action"] for r in audit.records if r["event"] == "controller_action"]
    assert actions == ["alert", "pause_strategy", "alert", "pause_all_and_reconcile", "trip_kill_switch"]
    assert audit.verify() == -1


# ----------------------------------------------------------------------------------- S12
def test_daily_report():
    fills = pd.DataFrame({"side": [1, -1], "qty": [100, 300], "price": [100.05, 99.97], "arrival_mid": [100.0, 100.0]})
    rep = mo.daily_report(fills, pd.Series([100_000.0, 102_000.0, 101_000.0]))
    assert rep["pnl"] == -1000 and rep["return"] == pytest.approx(-1000 / 102_000)
    assert rep["drawdown"] == pytest.approx(101 / 102 - 1) and rep["n_fills"] == 2
    assert rep["slippage_bps"] == pytest.approx((100 * 5 + 300 * 3) / 400)


def test_live_vs_backtest_band():
    rng = np.random.default_rng(1)
    bt = rng.normal(0.0005, 0.01, 2000)
    band = mo.backtest_band(bt, horizon=20)
    assert list(band.index) == list(range(1, 21)) and (band["lower"] < band["upper"]).all()
    assert band.loc[20, "upper"] - band.loc[20, "lower"] == pytest.approx(2 * 1.645 * 0.01 * np.sqrt(20), rel=0.1)
    assert mo.inside_band(rng.normal(0.0005, 0.01, 20), band).mean() > 0.8
    broken = mo.inside_band(np.full(20, -0.01), band)                                    # a strategy that stopped working
    assert not broken.iloc[-1] and len(broken) == 20
    lucky = np.full(20, 0.01)                                                              # far ABOVE the band
    assert not mo.inside_band(lucky, band).iloc[-1] and mo.inside_band(lucky, band, lower_only=True).all()


def test_attribution():
    pnl = pd.DataFrame({"mom": [100, -50, 80], "pairs": [10, 20, 30], "opts": [-40, -10, 0]})
    a = mo.attribution(pnl)
    assert list(a.index) == ["mom", "pairs", "opts"] and a["total"].tolist() == [130, 60, -50]
    assert a["share"].sum() == pytest.approx(1.0)
