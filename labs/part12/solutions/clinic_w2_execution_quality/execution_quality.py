"""Clinic W2 — Execution-quality report (Part 12, S5–S8): fill rate, time to fill, cost and all-in cost by
aggressiveness across several quote paths; implementation shortfall of one block vs TWAP vs VWAP vs POV across
several sessions (with a POINT-IN-TIME volume profile); an urgency policy; and the journal findings.
Run:  python execution_quality.py     Test:  python -m pytest clinic_w2_execution_quality  (needs week42_execution)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import intraday_session, quote_path, trade_journal                       # noqa: E402

ex = load("week42_execution", "execution")
SCHEDULES = {"passive": (0.0,), "mid": (0.5,), "cross": (1.0,), "chase": (0.0, 0.33, 0.67, 1.0)}


def aggressiveness_sweep(seeds=(0, 1, 2), n_orders: int = 150) -> pd.DataFrame:
    """Mean of ex.chase_report(quote_path(seed=s), SCHEDULES, n_orders, seed=s) over the seeds (same index and
    columns)."""
    # >>> SOLUTION
    reps = [ex.chase_report(quote_path(seed=s), SCHEDULES, n_orders, seed=s) for s in seeds]
    return sum(reps) / len(reps)
    # <<< SOLUTION


def pov_schedule(qty: int, session: pd.DataFrame, participation: float = 0.1) -> pd.Series:
    """POV: at each minute (from the second one), child = ex.pov_child_qty(volume of the PREVIOUS minute,
    participation, remaining). Index = minutes; it may finish early (zeros after) or not at all."""
    # >>> SOLUTION
    remaining, out = qty, []
    vols = session["volume"].to_numpy()
    for i in range(len(session)):
        child = 0 if i == 0 else ex.pov_child_qty(vols[i - 1], participation, remaining)
        remaining -= child
        out.append(child)
    return pd.Series(out, index=session.index)
    # <<< SOLUTION


def algo_tca(seeds=(1, 2, 3, 4, 5), qty: int = 60_000) -> pd.DataFrame:
    """For each session seed: decision price = the first minute's price; shortfall (bps) of BUY schedules executed
    with ex.execute_schedule: single (all at the first minute), twap (13 slices), vwap (profile = the 30-minute
    volume of the PREVIOUS seed's session: known before the day starts), pov (10%). Rows = algo; columns
    mean_bps, std_bps, completion (share of qty executed, averaged)."""
    # >>> SOLUTION
    res = {k: [] for k in ("single", "twap", "vwap", "pov")}
    done = {k: [] for k in res}
    for s in seeds:
        sess, prev = intraday_session(s), intraday_session(s - 1)
        start, end = sess.index[0], sess.index[-1] + pd.Timedelta(minutes=1)
        profile = prev["volume"].resample("30min").sum()
        profile.index = pd.date_range(start, periods=len(profile), freq="30min")
        sched = {"single": pd.Series([qty], index=[start]), "twap": ex.twap_schedule(qty, start, end, 13),
                 "vwap": ex.vwap_schedule(qty, profile), "pov": pov_schedule(qty, sess, 0.1)}
        for k, sc in sched.items():
            fills = ex.execute_schedule("BUY", sc, sess)
            res[k].append(ex.implementation_shortfall_bps("BUY", sess["price"].iloc[0], fills))
            done[k].append(sum(q for _, q in fills) / qty)
    return pd.DataFrame({"mean_bps": {k: np.mean(v) for k, v in res.items()},
                         "std_bps": {k: np.std(v, ddof=1) for k, v in res.items()},
                         "completion": {k: np.mean(v) for k, v in done.items()}})
    # <<< SOLUTION


URGENCY = {"stop": (1.0,), "exit": (0.5, 1.0), "entry": (0.0, 0.33, 0.67, 1.0), "rebalance": (0.0, 0.0, 0.33, 0.67, 1.0)}


def urgency_schedule(signal_type: str) -> tuple:
    """The chase schedule for a signal type (URGENCY); unknown types get the most careful one ("rebalance")."""
    # >>> SOLUTION
    return URGENCY.get(signal_type, URGENCY["rebalance"])
    # <<< SOLUTION


def journal_findings(journal: pd.DataFrame) -> dict:
    """{"by_setup_regime": ex.journal_stats by (setup_tag, regime), "best": its first index, "worst": its last
    index, "violations": ex.rule_violations counts per rule (dict)}."""
    # >>> SOLUTION
    j = journal.assign(r_multiple=ex.r_multiples(journal))
    st = ex.journal_stats(j, ["setup_tag", "regime"])
    return {"by_setup_regime": st, "best": st.index[0], "worst": st.index[-1],
            "violations": ex.rule_violations(j)["rule"].value_counts().to_dict()}
    # <<< SOLUTION


if __name__ == "__main__":
    print("aggressiveness (mean of 3 quote paths):\n", aggressiveness_sweep().round(2).to_string())
    print("\nparent-order TCA, BUY 60,000 (5 sessions):\n", algo_tca().round(1).to_string())
    jf = journal_findings(trade_journal())
    print("\njournal expectancy (R):\n", jf["by_setup_regime"].round(2).to_string())
    print("violations:", jf["violations"])
