"""Week 45–48 (S17–S24) — Going live and operating: rollout stages with promotion/demotion rules and a capital ramp,
the exchange calendar and a pre-market checklist that blocks trading, the incident lifecycle with response-time
SLAs and post-mortem timelines, the go-live review, weekly review decisions and the graduation checks.

Rule of the week: evidence, not returns, earns the next stage.
Fill in every block marked "Your turn", then run:  python -m pytest week45_golive
"""
from __future__ import annotations

import pandas as pd

STAGES = ["paper", "shadow", "small_live", "scaled_live"]
CRITERIA = {"paper": {"min_days": 10}, "shadow": {"min_days": 10}, "small_live": {"min_days": 20},
            "scaled_live": {}}


# ------------------------------------------------------------------------ S17 rollout & ramp
def rollout_decision(stage: str, ev: dict) -> tuple[str, str, list[str]]:
    """ev: days, sev1_incidents, incidents, open_breaks, slippage_ratio (realized / model), inside_band (share of
    days inside the backtest band), drift_alarm (bool).
    DEMOTE one stage (not below paper) if inside_band < 0.5, drift_alarm, or incidents >= 3 → ("demote", new stage,
    reasons). At the top stage: ("hold", stage, []). Else PROMOTE one stage if ALL hold: days >= the stage's min_days, sev1_incidents == 0,
    open_breaks == 0, slippage_ratio <= 1.5, inside_band >= 0.8. Else ("hold", stage, the failed criteria).
    Reason strings: "outside band", "drift alarm", "repeated incidents", "too few days", "sev1 incident",
    "open reconciliation breaks", "slippage above model", "below band"."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def capital_ramp(fraction: float, weeks_in_band: int, drawdown: float, plan: dict) -> float:
    """Next week's capital fraction. Drawdown worse than plan["max_dd"] (e.g. −0.05) → halve (not below
    plan["min"]). Else, every full plan["weeks_per_step"] weeks in the band → + plan["step"] (capped at
    plan["max"]), i.e. increase only when weeks_in_band is a positive multiple of weeks_per_step. Else unchanged."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------------- S18 daily operations
HOLIDAYS = {pd.Timestamp(d) for d in ("2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
                                      "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25")}
EARLY_CLOSE = {pd.Timestamp(d) for d in ("2025-07-03", "2025-11-28", "2025-12-24")}


def session_times(day) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(open 09:30, close 16:00, or 13:00 on EARLY_CLOSE days) for a trading day; None on weekends and HOLIDAYS."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


PREMARKET = {"gateway_connected": lambda s: s["gateway_connected"],
             "data_fresh": lambda s: s["data_age_s"] <= 5,
             "reconciliation_clean": lambda s: s["open_breaks"] == 0,
             "kill_switch_tested": lambda s: s["kill_switch_tested_today"],
             "risk_limits_loaded": lambda s: s["risk_limits_loaded"],
             "backup_ok": lambda s: s["last_backup_age_h"] <= 26,
             "disk_ok": lambda s: s["disk_free_pct"] >= 15}
BLOCKING = {"gateway_connected", "data_fresh", "reconciliation_clean", "kill_switch_tested", "risk_limits_loaded"}


def premarket_check(status: dict) -> dict:
    """Run every PREMARKET check. {"results": {name: bool}, "blocking_failures": sorted failed BLOCKING checks,
    "warnings": sorted failed non-blocking checks, "may_trade": no blocking failure}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------------- S19 incidents
PHASES = ["detected", "contained", "diagnosed", "recovered", "reconciled", "closed"]
SLA_MINUTES = {"sev1": {"contained": 5, "reconciled": 60}, "sev2": {"contained": 30, "reconciled": 240},
               "sev3": {"contained": 240, "reconciled": 1440}}


class Incident:
    """Lifecycle detect → contain → diagnose → recover → reconcile → close, one step at a time, with timestamps."""

    def __init__(self, title: str, severity: str, detected_at):
        self.title, self.severity = title, severity
        self.times = {"detected": pd.Timestamp(detected_at)}

    @property
    def phase(self) -> str:
        return PHASES[len(self.times) - 1]

    def advance(self, phase: str, at) -> None:
        """Only the NEXT phase is allowed, not earlier in time than the previous one (ValueError otherwise)."""
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def sla_breaches(self) -> list[str]:
        """Phases in SLA_MINUTES[severity] that were reached LATER than their target minutes after detection
        (only phases already reached are checked)."""
        raise NotImplementedError("✍️ Your turn: see the docstring")


def postmortem_timeline(records: list[dict], start, end) -> list[tuple[pd.Timestamp, str]]:
    """From audit records (ts as a datetime string or Timestamp, event): (ts, event) between start and end
    inclusive, sorted by time — the backbone of a blameless post-mortem."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------- S20–S24 reviews & graduation
def go_live_review(ev: dict) -> tuple[bool, list[str]]:
    """Approved for small live only on evidence: every strategy's dsr >= 0.95 and pbo < 0.5 ("gate: <name>"), at
    least 10 runbooks ("runbooks"), every drill passed ("drill: <name>"), security checklist complete
    ("security"), a rollout plan ("rollout plan"). ev: strategies {name: {dsr, pbo}}, runbooks (int), drills
    {name: bool}, security_complete (bool), rollout_plan (bool). Returns (approved, sorted missing items)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def weekly_decisions(inside: pd.DataFrame, incidents: dict[str, int]) -> dict[str, str]:
    """inside: days × strategies, True when the strategy's cumulative return was inside its backtest band.
    Per strategy: "demote" if inside on fewer than half the days or >= 2 incidents; "scale" if inside every day and
    no incident; else "hold"."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def orders_without_risk_approval(records: list[dict]) -> list[str]:
    """Graduation check from the audit chain: ids of "order" records (data["id"]) NOT preceded by a "risk_decision"
    record with data["order_id"] == that id and data["approved"] True. [] = no order bypassed the risk engine."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def graduation(records: list[dict], eod_breaks: dict, explained: set, strategies_dsr: dict) -> tuple[bool, list[str]]:
    """The four mandatory requirements: a "killswitch_trip" in the audit ("kill switch not demonstrated"), no order
    without risk approval ("orders bypassed risk: <ids joined by ,>"), every end-of-day break explained (eod_breaks:
    {date: n}; a date with n > 0 not in `explained` → "unexplained breaks: <dates joined by ,>", dates as given),
    a DSR for every strategy (None → "missing DSR: <names joined by ,>"). Returns (passed, reasons)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
