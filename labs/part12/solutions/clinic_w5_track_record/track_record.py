"""Clinic W5 & capstone — Go-live review, a 4-week track record with weekly reviews against each strategy's backtest
band, the capital ramp, and the graduation checks from the audit chain (Part 12, S17–S24).
Run:  python track_record.py      Test:  python -m pytest clinic_w5_track_record  (needs week43 and week45)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import track_record                                                      # noqa: E402

mo = load("week43_monitoring", "monitoring")
gl = load("week45_golive", "golive")
PLAN = {"min": 0.1, "max": 1.0, "step": 0.1, "weeks_per_step": 2, "max_dd": -0.03}


def inside_bands(bt: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: mo.inside_band(live returns, mo.backtest_band(backtest returns, horizon=len(live)),
    lower_only=True) — demotions are for falling BELOW the band — indexed by the live dates."""
    # >>> SOLUTION
    return pd.DataFrame({s: mo.inside_band(live[s], mo.backtest_band(bt[s], horizon=len(live)),
                                           lower_only=True).to_numpy()
                         for s in live.columns}, index=live.index)
    # <<< SOLUTION


def weekly_reviews(inside: pd.DataFrame, incidents: dict[int, dict[str, int]]) -> pd.DataFrame:
    """For weeks 1..4 (5 live days each): gl.weekly_decisions on THAT week's rows of `inside` and that week's
    incidents (incidents[week], default {}). Rows = week, columns = strategies."""
    # >>> SOLUTION
    rows = {}
    for w in range(1, len(inside) // 5 + 1):
        rows[w] = gl.weekly_decisions(inside.iloc[(w - 1) * 5: w * 5], incidents.get(w, {}))
    return pd.DataFrame(rows).T
    # <<< SOLUTION


def capital_path(decisions: pd.DataFrame, live: pd.DataFrame, start: float = 0.1) -> pd.DataFrame:
    """Capital fraction per strategy after each weekly review: "demote" → 0 (back to paper, and it stays 0);
    otherwise gl.capital_ramp(fraction, consecutive "scale" weeks so far, drawdown of the cumulative live return up
    to that week's end (cum − running max of cum, including 0 at the start), PLAN)."""
    # >>> SOLUTION
    out = {}
    for s in decisions.columns:
        frac, streak, path = start, 0, []
        for w, dec in decisions[s].items():
            cum = pd.concat([pd.Series([0.0]), live[s].iloc[: w * 5].cumsum()], ignore_index=True)
            dd = float((cum - cum.cummax()).iloc[-1])
            if dec == "demote" or frac == 0:
                frac = 0.0
            else:
                streak = streak + 1 if dec == "scale" else 0
                frac = gl.capital_ramp(frac, streak, dd, PLAN)
            path.append(frac)
        out[s] = path
    return pd.DataFrame(out, index=decisions.index)
    # <<< SOLUTION


def audit_trail(n_orders: int = 30, bypass: str | None = None) -> list[dict]:
    """A track-record audit trail (given): each order preceded by an approved risk decision, then a kill-switch
    trip at the defense. `bypass` = an order id sent WITHOUT a risk decision (to test the graduation check)."""
    mo_log = mo.AuditLog()
    for i in range(n_orders):
        oid = f"O{i:03d}"
        if oid != bypass:
            mo_log.append("risk_decision", order_id=oid, approved=True)
        mo_log.append("order", id=oid)
    mo_log.append("killswitch_trip", reason="defense demo")
    return mo_log.records


def dossier(seed: int = 0, incidents: dict | None = None) -> dict:
    """The capstone evidence pack (given): go-live review, bands, weekly decisions, capital path, graduation."""
    bt, live = track_record(seed)
    review = gl.go_live_review({"strategies": {"momentum": {"dsr": 0.97, "pbo": 0.2}, "pairs": {"dsr": 0.96, "pbo": 0.3},
                                               "options": {"dsr": 0.95, "pbo": 0.4}},
                                "runbooks": 10, "drills": {"disconnect": True, "runaway": True, "bad_tick": True},
                                "security_complete": True, "rollout_plan": True})
    inside = inside_bands(bt, live)
    decisions = weekly_reviews(inside, incidents or {})
    log_records = audit_trail()
    return {"go_live": review, "inside": inside, "decisions": decisions, "capital": capital_path(decisions, live),
            "graduation": gl.graduation(log_records, {"2025-11-07": 0, "2025-11-14": 1}, {"2025-11-14"},
                                        {"momentum": 0.97, "pairs": 0.96, "options": 0.95}),
            "audit_ok": all(r.get("hash") for r in log_records)}


if __name__ == "__main__":
    d = dossier()
    print("go-live review:", d["go_live"])
    print("\nshare of live days inside the backtest band:\n", d["inside"].mean().round(2).to_string())
    print("\nweekly decisions:\n", d["decisions"].to_string())
    print("\ncapital fraction after each review:\n", d["capital"].to_string())
    print("\ngraduation:", d["graduation"])
