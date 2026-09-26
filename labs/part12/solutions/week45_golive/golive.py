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
    # >>> SOLUTION
    i = STAGES.index(stage)
    dem = [r for r, bad in (("outside band", ev["inside_band"] < 0.5), ("drift alarm", ev["drift_alarm"]),
                            ("repeated incidents", ev["incidents"] >= 3)) if bad]
    if dem:
        return "demote", STAGES[max(i - 1, 0)], dem
    if i == len(STAGES) - 1:
        return "hold", stage, []                                                       # nothing above
    fails = [r for r, bad in (("too few days", ev["days"] < CRITERIA[stage]["min_days"]),
                              ("sev1 incident", ev["sev1_incidents"] > 0),
                              ("open reconciliation breaks", ev["open_breaks"] > 0),
                              ("slippage above model", ev["slippage_ratio"] > 1.5),
                              ("below band", ev["inside_band"] < 0.8)) if bad]
    if fails:
        return "hold", stage, fails
    return "promote", STAGES[i + 1], []
    # <<< SOLUTION


def capital_ramp(fraction: float, weeks_in_band: int, drawdown: float, plan: dict) -> float:
    """Next week's capital fraction. Drawdown worse than plan["max_dd"] (e.g. −0.05) → halve (not below
    plan["min"]). Else, every full plan["weeks_per_step"] weeks in the band → + plan["step"] (capped at
    plan["max"]), i.e. increase only when weeks_in_band is a positive multiple of weeks_per_step. Else unchanged."""
    # >>> SOLUTION
    if drawdown < plan["max_dd"]:
        return max(fraction / 2, plan["min"])
    if weeks_in_band > 0 and weeks_in_band % plan["weeks_per_step"] == 0:
        return min(fraction + plan["step"], plan["max"])
    return fraction
    # <<< SOLUTION


# --------------------------------------------------------------------------- S18 daily operations
HOLIDAYS = {pd.Timestamp(d) for d in ("2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
                                      "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25")}
EARLY_CLOSE = {pd.Timestamp(d) for d in ("2025-07-03", "2025-11-28", "2025-12-24")}


def session_times(day) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(open 09:30, close 16:00, or 13:00 on EARLY_CLOSE days) for a trading day; None on weekends and HOLIDAYS."""
    # >>> SOLUTION
    d = pd.Timestamp(day).normalize()
    if d.weekday() >= 5 or d in HOLIDAYS:
        return None
    close = pd.Timedelta(hours=13) if d in EARLY_CLOSE else pd.Timedelta(hours=16)
    return d + pd.Timedelta(hours=9, minutes=30), d + close
    # <<< SOLUTION


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
    # >>> SOLUTION
    res = {k: bool(f(status)) for k, f in PREMARKET.items()}
    failed = {k for k, ok in res.items() if not ok}
    return {"results": res, "blocking_failures": sorted(failed & BLOCKING), "warnings": sorted(failed - BLOCKING),
            "may_trade": not (failed & BLOCKING)}
    # <<< SOLUTION


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
        # >>> SOLUTION
        i = PHASES.index(self.phase)
        if i + 1 >= len(PHASES) or PHASES[i + 1] != phase:
            raise ValueError(f"{self.phase} cannot go to {phase}")
        at = pd.Timestamp(at)
        if at < self.times[self.phase]:
            raise ValueError("time goes forward")
        self.times[phase] = at
        # <<< SOLUTION

    def sla_breaches(self) -> list[str]:
        """Phases in SLA_MINUTES[severity] that were reached LATER than their target minutes after detection
        (only phases already reached are checked)."""
        # >>> SOLUTION
        t0 = self.times["detected"]
        return [p for p, m in SLA_MINUTES[self.severity].items()
                if p in self.times and (self.times[p] - t0) > pd.Timedelta(minutes=m)]
        # <<< SOLUTION


def postmortem_timeline(records: list[dict], start, end) -> list[tuple[pd.Timestamp, str]]:
    """From audit records (ts as a datetime string or Timestamp, event): (ts, event) between start and end
    inclusive, sorted by time — the backbone of a blameless post-mortem."""
    # >>> SOLUTION
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = [(pd.Timestamp(r["ts"]), r["event"]) for r in records if start <= pd.Timestamp(r["ts"]) <= end]
    return sorted(out)
    # <<< SOLUTION


# --------------------------------------------------------------- S20–S24 reviews & graduation
def go_live_review(ev: dict) -> tuple[bool, list[str]]:
    """Approved for small live only on evidence: every strategy's dsr >= 0.95 and pbo < 0.5 ("gate: <name>"), at
    least 10 runbooks ("runbooks"), every drill passed ("drill: <name>"), security checklist complete
    ("security"), a rollout plan ("rollout plan"). ev: strategies {name: {dsr, pbo}}, runbooks (int), drills
    {name: bool}, security_complete (bool), rollout_plan (bool). Returns (approved, sorted missing items)."""
    # >>> SOLUTION
    missing = [f"gate: {n}" for n, s in ev["strategies"].items() if not (s["dsr"] >= 0.95 and s["pbo"] < 0.5)]
    missing += [f"drill: {n}" for n, ok in ev["drills"].items() if not ok]
    if ev["runbooks"] < 10:
        missing.append("runbooks")
    if not ev["security_complete"]:
        missing.append("security")
    if not ev["rollout_plan"]:
        missing.append("rollout plan")
    return not missing, sorted(missing)
    # <<< SOLUTION


def weekly_decisions(inside: pd.DataFrame, incidents: dict[str, int]) -> dict[str, str]:
    """inside: days × strategies, True when the strategy's cumulative return was inside its backtest band.
    Per strategy: "demote" if inside on fewer than half the days or >= 2 incidents; "scale" if inside every day and
    no incident; else "hold"."""
    # >>> SOLUTION
    out = {}
    for s in inside.columns:
        share, inc = inside[s].mean(), incidents.get(s, 0)
        out[s] = "demote" if share < 0.5 or inc >= 2 else "scale" if share == 1.0 and inc == 0 else "hold"
    return out
    # <<< SOLUTION


def orders_without_risk_approval(records: list[dict]) -> list[str]:
    """Graduation check from the audit chain: ids of "order" records (data["id"]) NOT preceded by a "risk_decision"
    record with data["order_id"] == that id and data["approved"] True. [] = no order bypassed the risk engine."""
    # >>> SOLUTION
    approved, bad = set(), []
    for r in records:
        if r["event"] == "risk_decision" and r["data"].get("approved"):
            approved.add(r["data"]["order_id"])
        elif r["event"] == "order" and r["data"]["id"] not in approved:
            bad.append(r["data"]["id"])
    return bad
    # <<< SOLUTION


def graduation(records: list[dict], eod_breaks: dict, explained: set, strategies_dsr: dict) -> tuple[bool, list[str]]:
    """The four mandatory requirements: a "killswitch_trip" in the audit ("kill switch not demonstrated"), no order
    without risk approval ("orders bypassed risk: <ids joined by ,>"), every end-of-day break explained (eod_breaks:
    {date: n}; a date with n > 0 not in `explained` → "unexplained breaks: <dates joined by ,>", dates as given),
    a DSR for every strategy (None → "missing DSR: <names joined by ,>"). Returns (passed, reasons)."""
    # >>> SOLUTION
    reasons = []
    if not any(r["event"] == "killswitch_trip" for r in records):
        reasons.append("kill switch not demonstrated")
    bypass = orders_without_risk_approval(records)
    if bypass:
        reasons.append("orders bypassed risk: " + ",".join(bypass))
    unexplained = [d for d, n in eod_breaks.items() if n > 0 and d not in explained]
    if unexplained:
        reasons.append("unexplained breaks: " + ",".join(unexplained))
    missing = [s for s, v in strategies_dsr.items() if v is None]
    if missing:
        reasons.append("missing DSR: " + ",".join(missing))
    return not reasons, reasons
    # <<< SOLUTION
