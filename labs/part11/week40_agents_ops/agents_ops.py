"""Week 40 (S5–S8) — AI agents with read-only tools and a proposal queue, the platform API that n8n calls (HMAC-signed
kill switch), n8n workflow checks, event analysis (post-earnings drift by guidance label, an early-warning overlay),
and the evaluation, cost and reproducibility tools every LLM feature needs.

Rule of the week: AI PROPOSES, the risk engine and a human DECIDE. Nothing here can place an order.
Fill in every block marked "Your turn", then run:  python -m pytest week40_agents_ops
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request

from _loader import load

nr = load("week39_nlp_rag", "nlp_rag")

# Illustrative prices in USD per million tokens (CHECK THE CURRENT PRICE LIST: they change).
PRICES = {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25}


# ------------------------------------------------------------------------------ S5 agents
class AuditLog:
    """Append-only event log (in the platform: a database table)."""

    def __init__(self):
        self.events: list[dict] = []

    def log(self, kind: str, **data) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def kinds(self) -> list[str]:
        return [e["kind"] for e in self.events]


class ProposalQueue:
    """Trade PROPOSALS waiting for the risk engine and a human. Ids P0001, P0002, …; status "pending"."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def submit(self, strategy: str, legs: list[dict], rationale: str, source: str) -> str:
        raise NotImplementedError("✍️ Your turn: see the docstring")


FORBIDDEN = {"place_order", "cancel_order", "set_risk_limit", "trip_killswitch"}


class ToolBox:
    """The only tools the agent can use: the read-only functions given, plus propose_trade (which only queues).
    Every call is audited. A call to anything else — above all FORBIDDEN names — is DENIED and audited."""

    def __init__(self, read_tools: dict[str, Callable[..., object]], queue: ProposalQueue, audit: AuditLog):
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def names(self) -> list[str]:
        """Tool names exposed to the model: the read tools (in order) then "propose_trade"."""
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def call(self, name: str, args: dict) -> str:
        """Run one tool: read tools → json.dumps(result, default=str) with audit "tool_call"; propose_trade(strategy,
        legs_json, rationale) → queue it (source "ai_agent"), audit "ai_proposal" with proposal_id, return
        "Proposal P0001 queued for review."; anything else → audit "tool_denied" (name) and return
        "DENIED: <name> is not available. You can only propose trades." """
        raise NotImplementedError("✍️ Your turn: see the docstring")


def cost_usd(usage, prices: dict = PRICES, batch: bool = False) -> float:
    """Cost of one response: uncached input, cache reads, cache writes and output, each × its price per million
    tokens; batch jobs cost half."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def run_agent(model: Callable[[list[dict]], object], tools: ToolBox, task: str, max_tool_calls: int = 10,
              max_cost: float = 1.0) -> dict:
    """The tool loop with guardrails. history starts with the user task. Each turn: m = model(history); add its cost;
    audit "ai_turn" (stop_reason, tool names). Stop with status "refusal" on a refusal, "done" on end_turn, and
    "cost_budget" once the cost exceeds max_cost. On "tool_use": run each tool_use block through tools.call — but
    stop with "tool_budget" (without running it) when the call would exceed max_tool_calls — then append the
    assistant turn and one user turn of tool_result blocks. Return {"status", "tool_calls", "cost",
    "final_text" (the text of the last turn, "" if none)}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def review_proposal(p: dict, allowed: set[str], max_contracts: int = 10) -> tuple[bool, str]:
    """Risk-engine pre-check of an AI proposal (the human still decides): strategy in `allowed`; |qty| per leg <=
    max_contracts; DEFINED RISK: for every (symbol, right, expiry), the long contracts cover the short ones
    (Σ qty >= 0). Return (True, "ok") or (False, reason)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------- S6 the API n8n calls
def sign(body: bytes, secret: str, ts: str) -> str:
    """Lesson plan S6: HMAC-SHA256 hex of ts + "." + body."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def verify(body: bytes, secret: str, ts: str, signature: str, now: float, max_age: float = 60) -> bool:
    """Reject stale or future timestamps (|now − ts| > max_age: replay protection), non-numeric timestamps, and bad
    signatures (hmac.compare_digest: constant time)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def make_app(secret: str, killswitch: Callable[[str, bool], None], daily_report: Callable[[], dict],
             now: Callable[[], float] = time.time) -> FastAPI:
    """FastAPI app: GET /reports/daily → daily_report() (read-only). POST /killswitch/trip with headers X-Timestamp,
    X-Signature: verify the raw body (401 "bad signature" if it fails), then killswitch("n8n: <reason>", flatten)
    from the JSON body (defaults "manual", True) and return {"status": "tripped"}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


API_BASE = "http://quantforge-api:8000"
TRIGGERS = {"n8n-nodes-base.scheduleTrigger", "n8n-nodes-base.cron", "n8n-nodes-base.webhook",
            "n8n-nodes-base.telegramTrigger"}
SECRET_PATTERNS = [r"sk-ant-[A-Za-z0-9_-]{10,}", r"xox[bpa]-[A-Za-z0-9-]{10,}", r"\b\d{8,10}:[A-Za-z0-9_-]{30,}\b",
                   r"(?i)(secret|password|api[_-]?key)\s*[=:]\s*['\"][^'\"$]{6,}['\"]"]


def upstream(wf: dict, name: str) -> set[str]:
    """All nodes with a path TO `name` through wf["connections"] (given)."""
    parents: dict[str, set[str]] = {}
    for src, outs in wf["connections"].items():
        for branch in outs.get("main", []):
            for link in branch:
                parents.setdefault(link["node"], set()).add(src)
    seen, todo = set(), [name]
    while todo:
        for p in parents.get(todo.pop(), ()):
            if p not in seen:
                seen.add(p)
                todo.append(p)
    return seen


def validate_workflow(wf: dict) -> list[str]:
    """Problems of an n8n export ([] = passes):
    "no trigger" — no node of a TRIGGERS type;
    "unreachable: <node>" — a node not reachable from a trigger (use upstream);
    "external url: <node>" — an httpRequest whose url does not start with API_BASE;
    "secret in <node>" — any SECRET_PATTERNS match in json.dumps(node parameters);
    "unsigned kill switch: <node>" — an httpRequest to /killswitch/trip without BOTH an upstream "if" node and an
    upstream "code" node whose jsCode contains createHmac."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def should_alert(item: dict, holdings: set[str]) -> bool:
    """Lesson plan news-alert rule: relevance > 0.8 and |sentiment| > 0.6 and the ticker is held."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------------ S7 events
def drift_by_label(R: pd.DataFrame, mkt: pd.Series, events: pd.DataFrame, win=(1, 5)) -> pd.DataFrame:
    """Market-model CAR over `win` (days after the event) per guidance label with nr.event_study, pooling events of
    all tickers: per label mean final CAR, t-stat (ddof=1) and n. Index = label (sorted)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def pead_trades(R: pd.DataFrame, events: pd.DataFrame, hold: int = 5, cost_bps: float = 10.0) -> pd.Series:
    """Trade the label: long after "raised", short after "lowered" (skip "maintained"), entering at the close of the
    event day and holding `hold` days (returns of days d+1..d+hold). Net return per trade = side·Σ returns −
    cost_bps/1e4 (round trip). Series indexed by event row."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def early_warning(features: pd.DataFrame, window: int = 250) -> pd.Series:
    """Composite risk score: the mean of each feature's z-score against its OWN PAST window (rolling mean and std of
    the `window` rows ending YESTERDAY: shift(1)). Higher = more stress."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def overlay(ret: pd.Series, score: pd.Series, threshold: float = 1.5, low: float = 0.5) -> pd.Series:
    """Exposure `low` on days after the score exceeds the threshold (score at t−1), else 1: risk reduction only."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------ S8 evaluation, cost, safety
def cache_hit_rate(usages) -> float:
    """Σ cache reads / Σ (uncached input + cache reads + cache writes): the share of input served from the cache."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


class ResponseCache:
    """Reproducibility: responses keyed by sha256 of "model|prompt_version|input". A new prompt version or model is a
    new key (never serve an answer produced by a different prompt)."""

    def __init__(self):
        self.store: dict[str, str] = {}

    @staticmethod
    def key(model: str, prompt_version: str, text: str) -> str:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def get_or_call(self, model: str, prompt_version: str, text: str, call: Callable[[str], str]) -> str:
        raise NotImplementedError("✍️ Your turn: see the docstring")


def regression_report(golds: list, old: list, new: list, same: Callable[[object, object], bool]) -> dict:
    """Compare a prompt/model change on the golden set: {"old_score", "new_score" (share correct), "regressions"
    (indices correct before and wrong now), "passed": no regressions and new_score >= old_score}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


class RateLimiter:
    """Per-session guard for LLM calls: at most `max_calls` in any `per_seconds` window (time injected for tests)."""

    def __init__(self, max_calls: int, per_seconds: float, now: Callable[[], float] = time.time):
        self.max_calls, self.per, self.now = max_calls, per_seconds, now
        self.calls: deque = deque()

    def allow(self) -> bool:
        """Drop calls older than the window; allow (and record) the call if fewer than max_calls remain."""
        raise NotImplementedError("✍️ Your turn: see the docstring")
