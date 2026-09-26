"""Clinic W3 — Monitoring drill: replay a trading day with injected faults (bad tick, stale data, latency spike,
runaway order loop, duplicated fill). Each fault must be DETECTED by the right detector, HANDLED by the controller
with the right action, and recorded in an intact audit chain — and a clean day must raise no alarm at all.
Run:  python drill.py        Test:  python -m pytest clinic_w3_monitoring_drill   (needs week43_monitoring)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import FAULTS, trading_day                                               # noqa: E402

mo = load("week43_monitoring", "monitoring")
POLICY = {**mo.CONTROLLER_POLICY, "bad_tick": "pause_strategy", "duplicate_fill": "pause_all_and_reconcile"}


def bad_ticks(ticks: pd.DataFrame, window: int = 21, threshold: float = 8.0) -> list[int]:
    """Timestamps of bad prints: deviation of each price from the rolling median of the `window` ticks CENTRED on it
    (ignoring itself is not needed: the median is robust), then a mo.RobustZ(window=500, threshold) stream over the
    deviations. A single bad print moves its own deviation only, not its neighbours'."""
    # >>> SOLUTION
    dev = ticks["price"] / ticks["price"].rolling(window, center=True, min_periods=1).median() - 1
    rz = mo.RobustZ(window=500, threshold=threshold)
    return [int(t) for t, d in zip(ticks["ts"], dev) if rz.update(float(d))[1]]
    # <<< SOLUTION


def stale_gaps(ticks: pd.DataFrame, max_gap: float = 5.0) -> list[int]:
    """Timestamps of the last tick before a silence longer than max_gap seconds."""
    # >>> SOLUTION
    ts = ticks["ts"].to_numpy()
    return [int(ts[i]) for i in np.flatnonzero(np.diff(ts) > max_gap)]
    # <<< SOLUTION


def stream_flags(values, threshold: float = 6.0) -> list[int]:
    """Positions flagged by a mo.RobustZ(threshold=threshold) stream (latencies, orders per minute)."""
    # >>> SOLUTION
    rz = mo.RobustZ(threshold=threshold)
    return [i for i, v in enumerate(values) if rz.update(float(v))[1]]
    # <<< SOLUTION


def book_fills(fills: list[dict]) -> tuple[dict[str, int], list[str]]:
    """IDEMPOTENT booking: apply each fill_id once. Return (positions without zeros, ids delivered more than
    once)."""
    # >>> SOLUTION
    seen, dups, pos = set(), [], {}
    for f in fills:
        if f["fill_id"] in seen:
            dups.append(f["fill_id"])
            continue
        seen.add(f["fill_id"])
        pos[f["symbol"]] = pos.get(f["symbol"], 0) + f["qty"]
    return {k: v for k, v in pos.items() if v != 0}, dups
    # <<< SOLUTION


def detect(day: dict) -> list[tuple[str, object]]:
    """All detections as (anomaly, where): bad_tick (ts), stale_data (ts), latency_spike (ack index — run the
    detector on LOG latency: latencies are right-skewed, and raw z-scores flag their normal long tail),
    runaway_order_rate (minute), duplicate_fill (fill_id) — in that order."""
    # >>> SOLUTION
    _, dups = book_fills(day["fills"])
    return ([("bad_tick", t) for t in bad_ticks(day["ticks"])] + [("stale_data", t) for t in stale_gaps(day["ticks"])]
            + [("latency_spike", i) for i in stream_flags(np.log(day["acks"]))]
            + [("runaway_order_rate", i) for i in stream_flags(day["order_counts"])]
            + [("duplicate_fill", f) for f in dups])
    # <<< SOLUTION


def run_drill(day: dict, strategies=("mom", "pairs", "options")) -> dict:
    """Feed every detection to a mo.Controller (POLICY; strategy "mom" for strategy-level anomalies) and reconcile the
    idempotently booked positions with the broker. Return {"detections", "actions" ({anomaly: action}),
    "killed", "paused" (sorted), "breaks" (mo.reconcile_positions), "audit_ok" (verify() == −1)}."""
    # >>> SOLUTION
    audit = mo.AuditLog()
    ctl = mo.Controller(list(strategies), audit, mo.ApprovalQueue(audit), POLICY)
    dets = detect(day)
    actions = {a: ctl.handle(a, "mom") for a, _ in dets}
    booked, _ = book_fills(day["fills"])
    return {"detections": dets, "actions": actions, "killed": ctl.killed, "paused": sorted(ctl.paused),
            "breaks": mo.reconcile_positions(booked, day["broker_positions"]), "audit_ok": audit.verify() == -1}
    # <<< SOLUTION


if __name__ == "__main__":
    out = run_drill(trading_day())
    print("injected:", FAULTS)
    print("detected:", out["detections"])
    print("actions: ", out["actions"])
    print(f"killed={out['killed']}  paused={out['paused']}  breaks={out['breaks']}  audit_ok={out['audit_ok']}")
    print("clean day detections:", run_drill(trading_day(faults=False))["detections"])
