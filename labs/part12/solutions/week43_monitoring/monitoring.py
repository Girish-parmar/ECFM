"""Week 43 (S9–S12) — The monitoring department: structured logs with correlation ids, Prometheus metrics and alert
rules, SLOs, a hash-chained audit log, reconciliation (broker state wins), four-eyes approvals, anomaly detectors
(robust z, fat-tail P&L, CUSUM), the controller that maps anomalies to actions, and reports with a live-vs-backtest
band.

Rule of the week: if it is not logged, measured and reconciled, it did not happen.
Fill in every block marked "Your turn", then run:  python -m pytest week43_monitoring
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


# ------------------------------------------------------------------------- S9 observability
class StructuredLogger:
    """JSON lines with a correlation id that follows one idea from signal → intent → order → fill."""

    def __init__(self, now: Callable[[], float] = time.time):
        self.lines: list[str] = []
        self.now = now

    def log(self, level: str, msg: str, cid: str, **fields) -> None:
        """Append json.dumps({"ts", "level", "msg", "cid", **fields}, sort_keys=True, default=str)."""
        # >>> SOLUTION
        self.lines.append(json.dumps({"ts": self.now(), "level": level, "msg": msg, "cid": cid, **fields},
                                     sort_keys=True, default=str))
        # <<< SOLUTION

    def trace(self, cid: str) -> list[dict]:
        """Every record of one correlation id, in order: the story of one trade."""
        # >>> SOLUTION
        return [r for r in map(json.loads, self.lines) if r["cid"] == cid]
        # <<< SOLUTION


def make_metrics(registry: CollectorRegistry) -> dict:
    """The lesson plan S2/S9 metrics, registered on `registry` (not the global one): orders (Counter
    qf_orders_total by strategy, broker), rejects (Counter qf_rejects_total by broker), latency (Histogram
    qf_signal_to_submit_seconds with buckets 0.001, 0.005, 0.01, 0.025, 0.05, 0.1), equity (Gauge qf_equity_usd by
    account), staleness (Gauge qf_data_staleness_seconds by feed)."""
    # >>> SOLUTION
    return {"orders": Counter("qf_orders_total", "Orders sent", ["strategy", "broker"], registry=registry),
            "rejects": Counter("qf_rejects_total", "Orders rejected", ["broker"], registry=registry),
            "latency": Histogram("qf_signal_to_submit_seconds", "Bar received -> order submitted",
                                 buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1), registry=registry),
            "equity": Gauge("qf_equity_usd", "Account equity", ["account"], registry=registry),
            "staleness": Gauge("qf_data_staleness_seconds", "Age of the last tick", ["feed"], registry=registry)}
    # <<< SOLUTION


def exposition(registry: CollectorRegistry) -> str:
    """What Prometheus scrapes (given)."""
    return generate_latest(registry).decode()


SEVERITY = {"critical": 0, "warning": 1, "info": 2}


def evaluate_alerts(state: dict, rules: list[tuple[str, Callable[[dict], bool], str]]) -> list[tuple[str, str]]:
    """Fire every rule whose condition(state) is True; return (name, severity) sorted by severity (critical
    first), then name."""
    # >>> SOLUTION
    fired = [(name, sev) for name, cond, sev in rules if cond(state)]
    return sorted(fired, key=lambda x: (SEVERITY[x[1]], x[0]))
    # <<< SOLUTION


def slo_report(values: pd.Series, threshold: float, target: float = 0.999) -> dict:
    """SLO "value <= threshold for `target` of the time": {"compliance": share OK, "met": compliance >= target,
    "budget_used": share of bad samples / (1 − target) (1.0 = the whole error budget spent)}."""
    # >>> SOLUTION
    ok = float((values <= threshold).mean())
    return {"compliance": ok, "met": ok >= target, "budget_used": (1 - ok) / (1 - target)}
    # <<< SOLUTION


# -------------------------------------------------------------- S10 audit, reconciliation, approvals
def _digest(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()


class AuditLog:
    """Lesson plan S10: append-only, hash-chained. Each record: ts, event, data, prev (the previous hash, 64 zeros
    for the first), hash = sha256 of the record WITHOUT its hash (json, sorted keys)."""

    def __init__(self, now: Callable[[], float] = time.time):
        self.records: list[dict] = []
        self._last = "0" * 64
        self.now = now

    def append(self, event: str, **data) -> str:
        # >>> SOLUTION
        rec = {"ts": self.now(), "event": event, "data": data, "prev": self._last}
        rec["hash"] = _digest(rec)
        self.records.append(rec)
        self._last = rec["hash"]
        return rec["hash"]
        # <<< SOLUTION

    def verify(self) -> int:
        """Index of the first broken record (wrong prev link or wrong hash), −1 if the chain is intact."""
        # >>> SOLUTION
        prev = "0" * 64
        for i, rec in enumerate(self.records):
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec["prev"] != prev or _digest(body) != rec["hash"]:
                return i
            prev = rec["hash"]
        return -1
        # <<< SOLUTION

    def head(self) -> str:
        """The last hash: send it out daily (n8n e-mail) so history cannot be rewritten unnoticed."""
        return self._last


def reconcile_positions(internal: dict[str, float], broker: dict[str, float], tol: float = 1e-9) -> list[dict]:
    """Breaks between our books and the broker's, over the union of symbols (missing = 0): [{"symbol", "internal",
    "broker", "diff" (broker − internal)}] sorted by symbol. The BROKER is the source of truth."""
    # >>> SOLUTION
    out = []
    for s in sorted(set(internal) | set(broker)):
        a, b = internal.get(s, 0.0), broker.get(s, 0.0)
        if abs(a - b) > tol:
            out.append({"symbol": s, "internal": a, "broker": b, "diff": b - a})
    return out
    # <<< SOLUTION


def reconcile_orders(internal_open: set[str], broker_open: set[str]) -> dict[str, list[str]]:
    """Open-order breaks by id: {"unknown_at_broker": our open orders the broker does not have (sorted),
    "orphan_at_broker": broker orders we did not send, e.g. a manual TWS order (sorted)}."""
    # >>> SOLUTION
    return {"unknown_at_broker": sorted(internal_open - broker_open),
            "orphan_at_broker": sorted(broker_open - internal_open)}
    # <<< SOLUTION


class ApprovalQueue:
    """Four-eyes approvals for strategies, parameter and limit changes and large orders. Every step is audited
    (events approval_requested / approval_granted / approval_rejected)."""

    def __init__(self, audit: AuditLog):
        self.audit, self.items = audit, {}

    def request(self, kind: str, payload: dict, requester: str, reason: str) -> str:
        """Id A0001, A0002, …; status "pending"."""
        # >>> SOLUTION
        aid = f"A{len(self.items) + 1:04d}"
        self.items[aid] = {"id": aid, "kind": kind, "payload": payload, "requester": requester, "reason": reason,
                           "status": "pending"}
        self.audit.append("approval_requested", id=aid, kind=kind, requester=requester, reason=reason)
        return aid
        # <<< SOLUTION

    def decide(self, aid: str, approver: str, approve: bool, reason_code: str) -> None:
        """ValueError if the item is not pending or approver == requester (four eyes). Status "approved" or
        "rejected", with approver and reason_code recorded."""
        # >>> SOLUTION
        item = self.items[aid]
        if item["status"] != "pending":
            raise ValueError(f"{aid} is {item['status']}")
        if approver == item["requester"]:
            raise ValueError("four-eyes: the requester cannot approve")
        item.update(status="approved" if approve else "rejected", approver=approver, reason_code=reason_code)
        self.audit.append("approval_granted" if approve else "approval_rejected", id=aid, approver=approver,
                          reason_code=reason_code)
        # <<< SOLUTION


def needs_approval(order: dict, max_notional: float) -> bool:
    """Large orders need a human: |qty| · price > max_notional."""
    # >>> SOLUTION
    return abs(order["qty"]) * order["price"] > max_notional
    # <<< SOLUTION


# ------------------------------------------------------------------------- S11 anomalies
class RobustZ:
    """Lesson plan S11: streaming robust z-score (median / MAD·1.4826) over the last `window` values, computed from
    the values BEFORE x; flags only after min_obs values."""

    def __init__(self, window: int = 500, threshold: float = 6.0, min_obs: int = 50):
        self.buf, self.window, self.threshold, self.min_obs = [], window, threshold, min_obs

    def update(self, x: float) -> tuple[float, bool]:
        # >>> SOLUTION
        z = 0.0
        if len(self.buf) >= self.min_obs:
            arr = np.asarray(self.buf)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826 + 1e-12
            z = (x - med) / mad
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        return z, abs(z) > self.threshold
        # <<< SOLUTION


def fat_tail_error(realized_pnl: float, expected_sigma: float, k: float = 5.0) -> bool:
    """Lesson plan S11: a P&L move beyond k sigma is more often a data/position/fill error than a real event."""
    # >>> SOLUTION
    return abs(realized_pnl) > k * expected_sigma
    # <<< SOLUTION


class Cusum:
    """Two-sided CUSUM on standardized values z = (x − mean)/sd: s+ = max(0, s+ + z − k), s− = max(0, s− − z − k);
    alarm (and reset both) when either exceeds h. Detects a slow DRIFT that no single observation reveals."""

    def __init__(self, mean: float, sd: float, k: float = 0.5, h: float = 5.0):
        self.mean, self.sd, self.k, self.h = mean, sd, k, h
        self.pos = self.neg = 0.0

    def update(self, x: float) -> bool:
        # >>> SOLUTION
        z = (x - self.mean) / self.sd
        self.pos = max(0.0, self.pos + z - self.k)
        self.neg = max(0.0, self.neg - z - self.k)
        if self.pos > self.h or self.neg > self.h:
            self.pos = self.neg = 0.0
            return True
        return False
        # <<< SOLUTION


CONTROLLER_POLICY = {"stale_data": "pause_strategy", "latency_spike": "alert", "reject_burst": "pause_strategy",
                     "slippage_outlier": "alert", "fat_tail_pnl": "pause_all_and_reconcile",
                     "runaway_order_rate": "trip_kill_switch"}


class Controller:
    """Maps anomalies to actions (lesson plan policy; unknown anomalies → "alert"). pause_strategy pauses that
    strategy; pause_all_and_reconcile pauses every strategy in `strategies` and sets needs_reconcile;
    trip_kill_switch sets killed and pauses everything. Every decision is audited ("controller_action" with anomaly,
    strategy, action). A paused strategy resumes ONLY with an approved approval of kind "resume" for that strategy;
    nothing resumes while killed."""

    def __init__(self, strategies: list[str], audit: AuditLog, approvals: ApprovalQueue,
                 policy: dict[str, str] = CONTROLLER_POLICY):
        self.strategies, self.audit, self.approvals, self.policy = list(strategies), audit, approvals, policy
        self.paused: set[str] = set()
        self.killed = False
        self.needs_reconcile = False

    def handle(self, anomaly: str, strategy: str | None = None) -> str:
        # >>> SOLUTION
        action = self.policy.get(anomaly, "alert")
        if action == "pause_strategy" and strategy:
            self.paused.add(strategy)
        elif action == "pause_all_and_reconcile":
            self.paused |= set(self.strategies)
            self.needs_reconcile = True
        elif action == "trip_kill_switch":
            self.killed = True
            self.paused |= set(self.strategies)
        self.audit.append("controller_action", anomaly=anomaly, strategy=strategy, action=action)
        return action
        # <<< SOLUTION

    def resume(self, strategy: str, approval_id: str) -> bool:
        """True (and audited "resumed") only if not killed and the approval is approved, of kind "resume" and for
        this strategy (payload["strategy"])."""
        # >>> SOLUTION
        a = self.approvals.items.get(approval_id)
        ok = (not self.killed and a is not None and a["status"] == "approved" and a["kind"] == "resume"
              and a["payload"].get("strategy") == strategy)
        if ok:
            self.paused.discard(strategy)
            self.audit.append("resumed", strategy=strategy, approval=approval_id)
        return ok
        # <<< SOLUTION


# ---------------------------------------------------------------------------- S12 reports
def daily_report(fills: pd.DataFrame, equity: pd.Series) -> dict:
    """fills: side (+1/−1), qty, price, arrival_mid. equity: end-of-day values. {"pnl": last − previous equity,
    "return": pnl / previous, "drawdown": last / running max − 1, "n_fills", "slippage_bps": qty-weighted mean of
    side·(price − arrival_mid)/arrival_mid·1e4}."""
    # >>> SOLUTION
    slip = fills["side"] * (fills["price"] - fills["arrival_mid"]) / fills["arrival_mid"] * 1e4
    return {"pnl": float(equity.iloc[-1] - equity.iloc[-2]), "return": float(equity.iloc[-1] / equity.iloc[-2] - 1),
            "drawdown": float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1), "n_fills": int(len(fills)),
            "slippage_bps": float(np.average(slip, weights=fills["qty"]))}
    # <<< SOLUTION


def backtest_band(daily_returns, horizon: int = 20, n_boot: int = 2000, q=(0.05, 0.95), seed: int = 0
                  ) -> pd.DataFrame:
    """Expected range of a live track record: bootstrap `horizon`-day paths from the backtest's daily returns
    (rng.choice with replacement), cumulative SUM per path; per day 1..horizon the q quantiles. Columns lower,
    upper; index 1..horizon."""
    # >>> SOLUTION
    r = np.asarray(daily_returns, dtype=float)
    rng = np.random.default_rng(seed)
    paths = rng.choice(r, size=(n_boot, horizon), replace=True).cumsum(axis=1)
    lo, hi = np.quantile(paths, q, axis=0)
    return pd.DataFrame({"lower": lo, "upper": hi}, index=range(1, horizon + 1))
    # <<< SOLUTION


def inside_band(live_returns, band: pd.DataFrame, lower_only: bool = False) -> pd.Series:
    """For each live day d (1-based), is the cumulative live return within [lower, upper] of day d? With
    lower_only, only "not below the band" counts (beating the band calls for investigation, not demotion)."""
    # >>> SOLUTION
    cum = np.cumsum(np.asarray(live_returns, dtype=float))
    b = band.iloc[: len(cum)]
    ok = cum >= b["lower"].to_numpy()
    if not lower_only:
        ok &= cum <= b["upper"].to_numpy()
    return pd.Series(ok, index=b.index)
    # <<< SOLUTION


def attribution(pnl: pd.DataFrame) -> pd.DataFrame:
    """P&L by strategy (columns of daily P&L): total, share of the total, sharpe (annualized, ddof=1); sorted by
    total descending."""
    # >>> SOLUTION
    total = pnl.sum()
    return pd.DataFrame({"total": total, "share": total / total.sum(),
                         "sharpe": pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)}).sort_values("total", ascending=False)
    # <<< SOLUTION
