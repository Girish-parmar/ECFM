"""Clinic W2 — End to end: news → LLM score (point-in-time) → alert if it matters → pre-market brief → the agent
drafts a proposal → risk pre-check → human decision → audit log; plus a timed kill-switch drill through the signed
endpoint that n8n calls (Part 11, S5–S8).
Run:  python end_to_end.py      Test:  python -m pytest clinic_w2_end_to_end   (needs week39 and week40)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import ScriptedModel, news_market, recorded_llm_scores, research_script  # noqa: E402

nr = load("week39_nlp_rag", "nlp_rag")
ao = load("week40_agents_ops", "agents_ops")
ALLOWED = {"bull_put_spread", "bear_call_spread", "iron_condor"}


def scored_news(news: pd.DataFrame, cache) -> pd.DataFrame:
    """Recorded LLM scores for the headlines, each looked up through cache.score(headline_id, …) so that a repeated
    headline is never scored twice; usable_at = max(published_at, scored_at). Columns: ticker, headline,
    published_at, usable_at, sentiment, relevance."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def overnight_alerts(scored: pd.DataFrame, day: pd.Timestamp, holdings: set[str]) -> pd.DataFrame:
    """Headlines USABLE between the previous business day's 08:30 and this day's 08:30 (brief time) that pass
    ao.should_alert, sorted by |sentiment| descending."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def premarket_brief(day: pd.Timestamp, alerts: pd.DataFrame, positions: dict[str, int]) -> str:
    """The message n8n sends: a header line "Pre-market brief YYYY-MM-DD", then one line per alert
    "TICKER (pos N): headline [sentiment +0.83]", or "No alerts." when there are none."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def propose_and_review(human_approves: bool) -> dict:
    """Run the agent (ScriptedModel(research_script())) with read-only tools (get_quote, search_filings stubs), then
    ao.review_proposal on every queued proposal (ALLOWED strategies); a proposal that passes becomes "approved" or
    "rejected" by the human flag, one that fails becomes "rejected_by_risk". Audit "risk_check" (proposal_id, ok,
    reason) and "human_decision" (proposal_id, approved) for those that reach the human.
    Return {"agent": the run_agent result, "queue": the ProposalQueue, "audit": the AuditLog}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def killswitch_drill(secret: str = "drill-secret", reason: str = "drill") -> dict:
    """Sign a trip request as the n8n Code node does (ao.sign over the exact body with the current timestamp), POST
    it to ao.make_app(...) with FastAPI's TestClient, and time it. Return {"status_code", "tripped" (the kill switch
    was called with "n8n: <reason>"), "elapsed_ms"}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    R, mkt, news = news_market()
    cache = nr.ScoreCache()
    scored = scored_news(pd.concat([news, news.iloc[:50]], ignore_index=True), cache)   # 50 repeated headlines
    print(f"{len(scored)} headlines, {cache.calls} scored (the repeats came from the cache)")
    holdings = {"S01", "S04", "S07", "S12"}
    day = next(d for d in R.index[300:] if len(overnight_alerts(scored, d, holdings)))
    print("\n" + premarket_brief(day, overnight_alerts(scored, day, holdings), {t: 100 for t in holdings}))
    out = propose_and_review(human_approves=True)
    print("\nproposal:", {k: v for k, v in out["queue"].items["P0001"].items() if k != "legs"})
    print("audit trail:", out["audit"].kinds())
    print("\nkill-switch drill:", killswitch_drill())
