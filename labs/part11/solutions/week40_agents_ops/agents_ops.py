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
        # >>> SOLUTION
        self.events.append({"seq": len(self.events) + 1, "kind": kind, **data})
        # <<< SOLUTION

    def kinds(self) -> list[str]:
        return [e["kind"] for e in self.events]


class ProposalQueue:
    """Trade PROPOSALS waiting for the risk engine and a human. Ids P0001, P0002, …; status "pending"."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def submit(self, strategy: str, legs: list[dict], rationale: str, source: str) -> str:
        # >>> SOLUTION
        pid = f"P{len(self.items) + 1:04d}"
        self.items[pid] = {"id": pid, "strategy": strategy, "legs": legs, "rationale": rationale, "source": source,
                           "status": "pending"}
        return pid
        # <<< SOLUTION


FORBIDDEN = {"place_order", "cancel_order", "set_risk_limit", "trip_killswitch"}


class ToolBox:
    """The only tools the agent can use: the read-only functions given, plus propose_trade (which only queues).
    Every call is audited. A call to anything else — above all FORBIDDEN names — is DENIED and audited."""

    def __init__(self, read_tools: dict[str, Callable[..., object]], queue: ProposalQueue, audit: AuditLog):
        # >>> SOLUTION
        if FORBIDDEN & set(read_tools):
            raise ValueError("write tools cannot be registered")
        self.read_tools, self.queue, self.audit = read_tools, queue, audit
        # <<< SOLUTION

    def names(self) -> list[str]:
        """Tool names exposed to the model: the read tools (in order) then "propose_trade"."""
        # >>> SOLUTION
        return [*self.read_tools, "propose_trade"]
        # <<< SOLUTION

    def call(self, name: str, args: dict) -> str:
        """Run one tool: read tools → json.dumps(result, default=str) with audit "tool_call"; propose_trade(strategy,
        legs_json, rationale) → queue it (source "ai_agent"), audit "ai_proposal" with proposal_id, return
        "Proposal P0001 queued for review."; anything else → audit "tool_denied" (name) and return
        "DENIED: <name> is not available. You can only propose trades." """
        # >>> SOLUTION
        if name in self.read_tools:
            self.audit.log("tool_call", name=name, args=args)
            return json.dumps(self.read_tools[name](**args), default=str)
        if name == "propose_trade":
            pid = self.queue.submit(args["strategy"], json.loads(args["legs_json"]), args["rationale"], "ai_agent")
            self.audit.log("ai_proposal", proposal_id=pid)
            return f"Proposal {pid} queued for review."
        self.audit.log("tool_denied", name=name)
        return f"DENIED: {name} is not available. You can only propose trades."
        # <<< SOLUTION


def cost_usd(usage, prices: dict = PRICES, batch: bool = False) -> float:
    """Cost of one response: uncached input, cache reads, cache writes and output, each × its price per million
    tokens; batch jobs cost half."""
    # >>> SOLUTION
    c = (usage.input_tokens * prices["input"] + (usage.cache_read_input_tokens or 0) * prices["cache_read"]
         + (usage.cache_creation_input_tokens or 0) * prices["cache_write"]
         + usage.output_tokens * prices["output"]) / 1e6
    return c / 2 if batch else c
    # <<< SOLUTION


def run_agent(model: Callable[[list[dict]], object], tools: ToolBox, task: str, max_tool_calls: int = 10,
              max_cost: float = 1.0) -> dict:
    """The tool loop with guardrails. history starts with the user task. Each turn: m = model(history); add its cost;
    audit "ai_turn" (stop_reason, tool names). Stop with status "refusal" on a refusal, "done" on end_turn, and
    "cost_budget" once the cost exceeds max_cost. On "tool_use": run each tool_use block through tools.call — but
    stop with "tool_budget" (without running it) when the call would exceed max_tool_calls — then append the
    assistant turn and one user turn of tool_result blocks. Return {"status", "tool_calls", "cost",
    "final_text" (the text of the last turn, "" if none)}."""
    # >>> SOLUTION
    history = [{"role": "user", "content": task}]
    calls, cost = 0, 0.0
    while True:
        m = model(history)
        cost += cost_usd(m.usage)
        uses = [b for b in m.content if b.type == "tool_use"]
        tools.audit.log("ai_turn", stop_reason=m.stop_reason, tool_calls=[b.name for b in uses])
        text = " ".join(b.text for b in m.content if b.type == "text")
        if m.stop_reason == "refusal":
            return {"status": "refusal", "tool_calls": calls, "cost": cost, "final_text": text}
        if m.stop_reason != "tool_use":
            return {"status": "done", "tool_calls": calls, "cost": cost, "final_text": text}
        results = []
        for b in uses:
            if calls >= max_tool_calls:
                return {"status": "tool_budget", "tool_calls": calls, "cost": cost, "final_text": text}
            calls += 1
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": tools.call(b.name, b.input)})
        history += [{"role": "assistant", "content": [b.model_dump() for b in m.content]},
                    {"role": "user", "content": results}]
        if cost > max_cost:
            return {"status": "cost_budget", "tool_calls": calls, "cost": cost, "final_text": text}
    # <<< SOLUTION


def review_proposal(p: dict, allowed: set[str], max_contracts: int = 10) -> tuple[bool, str]:
    """Risk-engine pre-check of an AI proposal (the human still decides): strategy in `allowed`; |qty| per leg <=
    max_contracts; DEFINED RISK: for every (symbol, right, expiry), the long contracts cover the short ones
    (Σ qty >= 0). Return (True, "ok") or (False, reason)."""
    # >>> SOLUTION
    if p["strategy"] not in allowed:
        return False, f"strategy {p['strategy']} not allowed"
    if any(abs(leg["qty"]) > max_contracts for leg in p["legs"]):
        return False, f"more than {max_contracts} contracts in a leg"
    net: dict[tuple, int] = {}
    for leg in p["legs"]:
        key = (leg["symbol"], leg["right"], leg["expiry"])
        net[key] = net.get(key, 0) + leg["qty"]
    if any(v < 0 for v in net.values()):
        return False, "undefined risk: naked short options"
    return True, "ok"
    # <<< SOLUTION


# --------------------------------------------------------------------- S6 the API n8n calls
def sign(body: bytes, secret: str, ts: str) -> str:
    """Lesson plan S6: HMAC-SHA256 hex of ts + "." + body."""
    # >>> SOLUTION
    return hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    # <<< SOLUTION


def verify(body: bytes, secret: str, ts: str, signature: str, now: float, max_age: float = 60) -> bool:
    """Reject stale or future timestamps (|now − ts| > max_age: replay protection), non-numeric timestamps, and bad
    signatures (hmac.compare_digest: constant time)."""
    # >>> SOLUTION
    try:
        if abs(now - float(ts)) > max_age:
            return False
    except ValueError:
        return False
    return hmac.compare_digest(sign(body, secret, ts), signature)
    # <<< SOLUTION


def make_app(secret: str, killswitch: Callable[[str, bool], None], daily_report: Callable[[], dict],
             now: Callable[[], float] = time.time) -> FastAPI:
    """FastAPI app: GET /reports/daily → daily_report() (read-only). POST /killswitch/trip with headers X-Timestamp,
    X-Signature: verify the raw body (401 "bad signature" if it fails), then killswitch("n8n: <reason>", flatten)
    from the JSON body (defaults "manual", True) and return {"status": "tripped"}."""
    # >>> SOLUTION
    app = FastAPI()

    @app.get("/reports/daily")
    def report():
        return daily_report()

    @app.post("/killswitch/trip")
    async def trip(request: Request, x_timestamp: str = Header(...), x_signature: str = Header(...)):
        body = await request.body()
        if not verify(body, secret, x_timestamp, x_signature, now()):
            raise HTTPException(status_code=401, detail="bad signature")
        payload = json.loads(body or b"{}")
        killswitch(f"n8n: {payload.get('reason', 'manual')}", payload.get("flatten", True))
        return {"status": "tripped"}

    return app
    # <<< SOLUTION


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
    # >>> SOLUTION
    problems = []
    nodes = {n["name"]: n for n in wf["nodes"]}
    triggers = {k for k, n in nodes.items() if n["type"] in TRIGGERS}
    if not triggers:
        problems.append("no trigger")
    for name, n in nodes.items():
        ups = upstream(wf, name)
        if name not in triggers and not (ups & triggers):
            problems.append(f"unreachable: {name}")
        params = n.get("parameters", {})
        if n["type"] == "n8n-nodes-base.httpRequest":
            url = params.get("url", "")
            if not url.startswith(API_BASE):
                problems.append(f"external url: {name}")
            if "/killswitch/trip" in url:
                kinds = [nodes[u] for u in ups]
                has_if = any(u["type"] == "n8n-nodes-base.if" for u in kinds)
                has_sig = any(u["type"] == "n8n-nodes-base.code" and "createHmac" in u["parameters"].get("jsCode", "")
                              for u in kinds)
                if not (has_if and has_sig):
                    problems.append(f"unsigned kill switch: {name}")
        if any(re.search(p, json.dumps(params)) for p in SECRET_PATTERNS):
            problems.append(f"secret in {name}")
    return problems
    # <<< SOLUTION


def should_alert(item: dict, holdings: set[str]) -> bool:
    """Lesson plan news-alert rule: relevance > 0.8 and |sentiment| > 0.6 and the ticker is held."""
    # >>> SOLUTION
    return item["relevance"] > 0.8 and abs(item["sentiment"]) > 0.6 and item["ticker"] in holdings
    # <<< SOLUTION


# ------------------------------------------------------------------------------ S7 events
def drift_by_label(R: pd.DataFrame, mkt: pd.Series, events: pd.DataFrame, win=(1, 5)) -> pd.DataFrame:
    """Market-model CAR over `win` (days after the event) per guidance label with nr.event_study, pooling events of
    all tickers: per label mean final CAR, t-stat (ddof=1) and n. Index = label (sorted)."""
    # >>> SOLUTION
    rows = {}
    for label, g in events.groupby("label"):
        finals = []
        for _, e in g.iterrows():
            path, _, n = nr.event_study(R[e["ticker"]], mkt, [e["date"]], win=win)
            if n:
                finals.append(path.iloc[-1])
        f = np.array(finals)
        rows[label] = {"car": float(f.mean()), "t": float(f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))), "n": len(f)}
    return pd.DataFrame(rows).T.sort_index()
    # <<< SOLUTION


def pead_trades(R: pd.DataFrame, events: pd.DataFrame, hold: int = 5, cost_bps: float = 10.0) -> pd.Series:
    """Trade the label: long after "raised", short after "lowered" (skip "maintained"), entering at the close of the
    event day and holding `hold` days (returns of days d+1..d+hold). Net return per trade = side·Σ returns −
    cost_bps/1e4 (round trip). Series indexed by event row."""
    # >>> SOLUTION
    out = {}
    for i, e in events.iterrows():
        side = {"raised": 1, "lowered": -1}.get(e["label"])
        if side is None:
            continue
        d = R.index.get_loc(e["date"])
        out[i] = side * R[e["ticker"]].iloc[d + 1:d + 1 + hold].sum() - cost_bps / 1e4
    return pd.Series(out, dtype=float)
    # <<< SOLUTION


def early_warning(features: pd.DataFrame, window: int = 250) -> pd.Series:
    """Composite risk score: the mean of each feature's z-score against its OWN PAST window (rolling mean and std of
    the `window` rows ending YESTERDAY: shift(1)). Higher = more stress."""
    # >>> SOLUTION
    m = features.rolling(window).mean().shift(1)
    s = features.rolling(window).std().shift(1)
    return ((features - m) / s).mean(axis=1)
    # <<< SOLUTION


def overlay(ret: pd.Series, score: pd.Series, threshold: float = 1.5, low: float = 0.5) -> pd.Series:
    """Exposure `low` on days after the score exceeds the threshold (score at t−1), else 1: risk reduction only."""
    # >>> SOLUTION
    expo = pd.Series(np.where(score > threshold, low, 1.0), index=score.index).shift(1).fillna(1.0)
    return ret * expo.reindex(ret.index).fillna(1.0)
    # <<< SOLUTION


# ------------------------------------------------------------------ S8 evaluation, cost, safety
def cache_hit_rate(usages) -> float:
    """Σ cache reads / Σ (uncached input + cache reads + cache writes): the share of input served from the cache."""
    # >>> SOLUTION
    read = sum(u.cache_read_input_tokens or 0 for u in usages)
    total = sum(u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
                for u in usages)
    return read / total if total else 0.0
    # <<< SOLUTION


class ResponseCache:
    """Reproducibility: responses keyed by sha256 of "model|prompt_version|input". A new prompt version or model is a
    new key (never serve an answer produced by a different prompt)."""

    def __init__(self):
        self.store: dict[str, str] = {}

    @staticmethod
    def key(model: str, prompt_version: str, text: str) -> str:
        # >>> SOLUTION
        return hashlib.sha256(f"{model}|{prompt_version}|{text}".encode()).hexdigest()
        # <<< SOLUTION

    def get_or_call(self, model: str, prompt_version: str, text: str, call: Callable[[str], str]) -> str:
        # >>> SOLUTION
        k = self.key(model, prompt_version, text)
        if k not in self.store:
            self.store[k] = call(text)
        return self.store[k]
        # <<< SOLUTION


def regression_report(golds: list, old: list, new: list, same: Callable[[object, object], bool]) -> dict:
    """Compare a prompt/model change on the golden set: {"old_score", "new_score" (share correct), "regressions"
    (indices correct before and wrong now), "passed": no regressions and new_score >= old_score}."""
    # >>> SOLUTION
    ok_old = [same(o, g) for o, g in zip(old, golds)]
    ok_new = [same(n, g) for n, g in zip(new, golds)]
    reg = [i for i, (a, b) in enumerate(zip(ok_old, ok_new)) if a and not b]
    so, sn = float(np.mean(ok_old)), float(np.mean(ok_new))
    return {"old_score": so, "new_score": sn, "regressions": reg, "passed": not reg and sn >= so}
    # <<< SOLUTION


class RateLimiter:
    """Per-session guard for LLM calls: at most `max_calls` in any `per_seconds` window (time injected for tests)."""

    def __init__(self, max_calls: int, per_seconds: float, now: Callable[[], float] = time.time):
        self.max_calls, self.per, self.now = max_calls, per_seconds, now
        self.calls: deque = deque()

    def allow(self) -> bool:
        """Drop calls older than the window; allow (and record) the call if fewer than max_calls remain."""
        # >>> SOLUTION
        t = self.now()
        while self.calls and t - self.calls[0] >= self.per:
            self.calls.popleft()
        if len(self.calls) < self.max_calls:
            self.calls.append(t)
            return True
        return False
        # <<< SOLUTION
