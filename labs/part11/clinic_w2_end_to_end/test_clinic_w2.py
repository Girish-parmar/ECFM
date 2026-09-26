import pandas as pd
from _loader import load
from common import news_market

e2e = load("clinic_w2_end_to_end", "end_to_end")
D = pd.Timestamp("2024-03-06")                                                        # a Wednesday


def test_scored_news_uses_the_cache_and_timestamps():
    _, _, news = news_market()
    cache = e2e.nr.ScoreCache()
    doubled = pd.concat([news, news.iloc[:50]], ignore_index=True)
    s = e2e.scored_news(doubled, cache)
    assert len(s) == 850 and cache.calls == 800                                       # repeats are never re-scored
    assert (s["usable_at"] >= s["published_at"]).all() and (s["usable_at"] > s["published_at"]).any()
    assert (s["sentiment"].iloc[800:].to_numpy() == s["sentiment"].iloc[:50].to_numpy()).all()


def test_overnight_alert_window():
    t = lambda hm, day=0: D + pd.Timedelta(days=day) + pd.Timedelta(f"{hm}:00")       # noqa: E731
    scored = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "AAA", "AAA", "CCC"],
        "headline": ["old", "late yesterday", "not held", "too weak", "this morning", "after brief"],
        "usable_at": [t("08:30", -1), t("18:00", -1), t("06:00"), t("07:00"), t("08:30"), t("08:31")],
        "sentiment": [0.9, -0.7, 0.9, 0.3, 0.95, 0.9], "relevance": [1.0] * 6})
    got = e2e.overnight_alerts(scored, D, {"AAA", "CCC"})
    assert list(got["headline"]) == ["this morning", "late yesterday"]               # sorted by |sentiment|


def test_premarket_brief_format():
    alerts = pd.DataFrame({"ticker": ["AAA"], "headline": ["AAA wins contract"], "sentiment": [0.834]})
    assert e2e.premarket_brief(D, alerts, {"AAA": 200}) == \
        "Pre-market brief 2024-03-06\nAAA (pos 200): AAA wins contract [sentiment +0.83]"
    assert e2e.premarket_brief(D, alerts.iloc[:0], {}) == "Pre-market brief 2024-03-06\nNo alerts."


def test_proposal_flow_and_audit_trail():
    ok = e2e.propose_and_review(human_approves=True)
    assert ok["agent"]["status"] == "done" and ok["queue"].items["P0001"]["status"] == "approved"
    assert ok["audit"].kinds()[-2:] == ["risk_check", "human_decision"]
    no = e2e.propose_and_review(human_approves=False)
    assert no["queue"].items["P0001"]["status"] == "rejected"


def test_risk_check_stops_before_the_human(monkeypatch):
    monkeypatch.setattr(e2e, "ALLOWED", {"iron_condor"})
    out = e2e.propose_and_review(human_approves=True)
    assert out["queue"].items["P0001"]["status"] == "rejected_by_risk"
    assert "human_decision" not in out["audit"].kinds()


def test_killswitch_drill():
    d = e2e.killswitch_drill()
    assert d["status_code"] == 200 and d["tripped"] and 0 < d["elapsed_ms"] < 2000
