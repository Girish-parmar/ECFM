import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from _loader import load
from common import ScriptedModel, earnings_market, make_message, research_script, rogue_script, tool_call

ao = load("week40_agents_ops", "agents_ops")
N8N = Path(__file__).parent / "n8n"
SECRET = "test-secret"


def toolbox():
    audit, queue = ao.AuditLog(), ao.ProposalQueue()
    read = {"get_quote": lambda symbol: {"symbol": symbol, "bid": 99.9, "ask": 100.1},
            "search_filings": lambda ticker, query, as_of: [{"ticker": ticker, "text": "Guidance raised."}]}
    return ao.ToolBox(read, queue, audit), queue, audit


# ----------------------------------------------------------------------------------- S5
def test_agent_proposes_and_never_trades():
    tools, queue, audit = toolbox()
    assert tools.names() == ["get_quote", "search_filings", "propose_trade"]
    assert not ao.FORBIDDEN & set(tools.names())                                    # no order-placing capability
    res = ao.run_agent(ScriptedModel(research_script()), tools, "Research ACME and propose a structure.")
    assert res["status"] == "done" and res["tool_calls"] == 3 and "Proposal submitted" in res["final_text"]
    assert list(queue.items) == ["P0001"] and queue.items["P0001"]["status"] == "pending"
    assert queue.items["P0001"]["source"] == "ai_agent" and len(queue.items["P0001"]["legs"]) == 2
    assert audit.kinds() == ["ai_turn", "tool_call", "ai_turn", "tool_call", "ai_turn", "ai_proposal", "ai_turn"]
    assert res["cost"] == pytest.approx(4 * (1500 * 5 + 200 * 25) / 1e6)


def test_rogue_model_is_denied_and_audited():
    tools, queue, audit = toolbox()
    res = ao.run_agent(ScriptedModel(rogue_script()), tools, "Buy 1000 ACME now.")
    assert res["status"] == "done" and not queue.items
    denied = [e["name"] for e in audit.events if e["kind"] == "tool_denied"]
    assert denied == ["place_order", "set_risk_limit"]
    assert tools.call("place_order", {}).startswith("DENIED")
    with pytest.raises(ValueError):
        ao.ToolBox({"place_order": lambda: None}, ao.ProposalQueue(), ao.AuditLog())      # cannot even register it


def test_agent_budgets_and_refusal():
    loop = [make_message([tool_call("get_quote", {"symbol": "SPY"}, "x")], "tool_use")]
    tools, _, _ = toolbox()
    res = ao.run_agent(ScriptedModel(loop), tools, "loop", max_tool_calls=4)
    assert res["status"] == "tool_budget" and res["tool_calls"] == 4
    expensive = [make_message([tool_call("get_quote", {"symbol": "SPY"}, "x")], "tool_use", input_tokens=200_000)]
    res = ao.run_agent(ScriptedModel(expensive), toolbox()[0], "loop", max_cost=2.0)
    assert res["status"] == "cost_budget" and res["cost"] > 2.0 and res["tool_calls"] == 2
    res = ao.run_agent(ScriptedModel([make_message("no", stop_reason="refusal")]), toolbox()[0], "x")
    assert res["status"] == "refusal" and res["tool_calls"] == 0


def test_risk_precheck_of_proposals():
    legs = [{"symbol": "ACME", "right": "P", "strike": 95, "expiry": "2026-11-20", "qty": -1},
            {"symbol": "ACME", "right": "P", "strike": 90, "expiry": "2026-11-20", "qty": 1}]
    ok = {"strategy": "bull_put_spread", "legs": legs}
    allowed = {"bull_put_spread", "iron_condor"}
    assert ao.review_proposal(ok, allowed) == (True, "ok")
    assert ao.review_proposal({**ok, "strategy": "short_straddle"}, allowed)[0] is False
    naked = {"strategy": "bull_put_spread", "legs": [legs[0]]}
    assert ao.review_proposal(naked, allowed) == (False, "undefined risk: naked short options")
    big = {"strategy": "bull_put_spread", "legs": [{**leg, "qty": leg["qty"] * 20} for leg in legs]}
    assert ao.review_proposal(big, allowed)[0] is False


def test_cost_usd():
    u = make_message("x", input_tokens=1000, output_tokens=500, cache_read=9000, cache_write=2000).usage
    expected = (1000 * 5 + 9000 * 0.5 + 2000 * 6.25 + 500 * 25) / 1e6
    assert ao.cost_usd(u) == pytest.approx(expected) and ao.cost_usd(u, batch=True) == pytest.approx(expected / 2)


# ----------------------------------------------------------------------------------- S6
def test_sign_and_verify():
    body, ts = b'{"reason":"drill"}', "1000"
    sig = ao.sign(body, SECRET, ts)
    assert len(sig) == 64 and ao.verify(body, SECRET, ts, sig, now=1030)
    assert not ao.verify(body, SECRET, ts, sig, now=1000 + 3600)                     # replayed an hour later
    assert not ao.verify(body + b" ", SECRET, ts, sig, now=1000)                     # body changed
    assert not ao.verify(body, "other", ts, sig, now=1000)
    assert not ao.verify(body, SECRET, "not-a-number", sig, now=1000)


def test_killswitch_endpoint():
    tripped = []
    app = ao.make_app(SECRET, lambda reason, flatten: tripped.append((reason, flatten)),
                      lambda: {"pnl": 1250.0, "var_99": 0.012}, now=lambda: 1000.0)
    client = TestClient(app)
    assert client.get("/reports/daily").json() == {"pnl": 1250.0, "var_99": 0.012}
    body = json.dumps({"reason": "drill"}).encode()
    ok = {"X-Timestamp": "990", "X-Signature": ao.sign(body, SECRET, "990")}
    assert client.post("/killswitch/trip", content=body, headers=ok).json() == {"status": "tripped"}
    assert tripped == [("n8n: drill", True)]
    stale = {"X-Timestamp": "100", "X-Signature": ao.sign(body, SECRET, "100")}
    assert client.post("/killswitch/trip", content=body, headers=stale).status_code == 401
    forged = {"X-Timestamp": "990", "X-Signature": "0" * 64}
    assert client.post("/killswitch/trip", content=body, headers=forged).status_code == 401
    assert client.post("/killswitch/trip", content=body).status_code == 422            # headers missing
    assert len(tripped) == 1


def test_n8n_exports_pass_and_bad_ones_fail():
    for f in ("premarket_brief", "news_alert", "eod_report", "kill_switch"):
        assert ao.validate_workflow(json.loads((N8N / f"{f}.json").read_text())) == [], f
    wf = json.loads((N8N / "kill_switch.json").read_text())
    unsigned = json.loads(json.dumps(wf))
    unsigned["connections"] = {"Telegram /kill": {"main": [[{"node": "Trip kill switch", "type": "main", "index": 0}]]},
                               "Trip kill switch": {"main": [[{"node": "Confirm", "type": "main", "index": 0}]]}}
    problems = ao.validate_workflow(unsigned)
    assert "unsigned kill switch: Trip kill switch" in problems
    assert "unreachable: Sender allowed?" in problems and "unreachable: Sign request" in problems
    leaky = json.loads((N8N / "eod_report.json").read_text())
    leaky["nodes"][2]["parameters"]["jsCode"] += "\nconst apiKey = 'sk-ant-api03-abcdefghijklmnop';"
    leaky["nodes"][1]["parameters"]["url"] = "https://example.com/report"
    assert set(ao.validate_workflow(leaky)) == {"secret in Format table", "external url: Daily report"}
    assert ao.validate_workflow({"nodes": [], "connections": {}}) == ["no trigger"]


def test_news_alert_rule():
    item = {"ticker": "ACME", "relevance": 0.9, "sentiment": -0.7}
    assert ao.should_alert(item, {"ACME"})
    assert not ao.should_alert(item, {"BOLT"}) and not ao.should_alert({**item, "sentiment": 0.5}, {"ACME"})
    assert not ao.should_alert({**item, "relevance": 0.8}, {"ACME"})


# ----------------------------------------------------------------------------------- S7
def test_post_earnings_drift_by_guidance_label():
    R, mkt, ev = earnings_market()
    tab = ao.drift_by_label(R, mkt, ev)
    assert list(tab.index) == ["lowered", "maintained", "raised"] and tab["n"].sum() == len(ev)
    assert tab.loc["raised", "car"] == pytest.approx(0.015, abs=0.006) and tab.loc["raised", "t"] > 3
    assert tab.loc["lowered", "car"] == pytest.approx(-0.015, abs=0.006) and abs(tab.loc["maintained", "t"]) < 2
    trades = ao.pead_trades(R, ev, cost_bps=10)
    assert len(trades) == (ev["label"] != "maintained").sum() and trades.mean() > 0.01
    one = ev[ev["label"] == "raised"].iloc[[0]]
    d = R.index.get_loc(one["date"].iloc[0])
    expected = R[one["ticker"].iloc[0]].iloc[d + 1:d + 6].sum() - 0.001                # enter after the event day
    assert ao.pead_trades(R, one).iloc[0] == pytest.approx(expected)
    assert ao.pead_trades(R, ev[ev["label"] == "maintained"]).empty


def test_early_warning_overlay_cuts_drawdown():
    rng = np.random.default_rng(0)
    n = 2000
    stress = np.zeros(n)
    for t in range(1, n):
        stress[t] = 0.98 * stress[t - 1] + rng.normal(0, 0.2)
    idx = pd.bdate_range("2015-01-01", periods=n)
    vol = 0.008 * np.exp(0.6 * np.r_[0, stress[:-1]])
    ret = pd.Series(0.0004 - 0.5 * vol ** 2 * 100 + vol * rng.standard_normal(n), index=idx)
    feats = pd.DataFrame({"vix_proxy": stress + rng.normal(0, 0.3, n), "credit": stress + rng.normal(0, 0.5, n)},
                         index=idx)
    score = ao.early_warning(feats)
    assert score.iloc[:250].isna().all() and score.iloc[251:].notna().all()           # past data only
    cut = ao.early_warning(feats.iloc[:1000])
    pd.testing.assert_series_equal(cut, score.iloc[:1000])                           # no look-ahead
    over = ao.overlay(ret, score)
    dd = lambda r: (r.cumsum() - r.cumsum().cummax()).min()                         # noqa: E731
    assert dd(over.iloc[251:]) > dd(ret.iloc[251:])
    toy = ao.overlay(pd.Series(0.01, index=idx[:3]), pd.Series([2.0, 0.0, 0.0], index=idx[:3]))
    np.testing.assert_allclose(toy, [0.01, 0.005, 0.01])


# ----------------------------------------------------------------------------------- S8
def test_cache_hit_rate():
    us = [make_message("x", input_tokens=100, cache_write=900).usage,
          make_message("x", input_tokens=100, cache_read=900).usage]
    assert ao.cache_hit_rate(us) == pytest.approx(900 / 2000)
    assert ao.cache_hit_rate([]) == 0.0


def test_response_cache_is_versioned():
    cache, calls = ao.ResponseCache(), []
    f = lambda t: calls.append(t) or t.upper()                                     # noqa: E731
    assert cache.get_or_call("m", "v1", "hi", f) == "HI" and cache.get_or_call("m", "v1", "hi", f) == "HI"
    cache.get_or_call("m", "v2", "hi", f)
    cache.get_or_call("m2", "v1", "hi", f)
    assert len(calls) == 3 and len(ao.ResponseCache.key("m", "v1", "hi")) == 64


def test_regression_report():
    golds, old, new = [1, 2, 3, 4], [1, 2, 0, 0], [1, 0, 3, 4]
    rep = ao.regression_report(golds, old, new, lambda a, b: a == b)
    assert rep == {"old_score": 0.5, "new_score": 0.75, "regressions": [1], "passed": False}  # better, yet broke one
    assert ao.regression_report(golds, old, golds, lambda a, b: a == b)["passed"]


def test_rate_limiter():
    clock = [0.0]
    rl = ao.RateLimiter(2, 60, now=lambda: clock[0])
    assert [rl.allow(), rl.allow(), rl.allow()] == [True, True, False]
    clock[0] = 60.0
    assert rl.allow() and rl.allow() and not rl.allow()
