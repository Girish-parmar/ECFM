"""Clinic W3 — Chart-pattern scanner with a no-look-ahead audit (Part 5, S10–S11).

scan() finds double tops/bottoms and head & shoulders on ZigZag pivots and reports the BREAKOUT bar (close through
the neckline). truncation_audit() re-runs the scanner on data cut right after each signal: an honest scanner finds the
same signal again; a scanner that uses the pivot index instead of the confirmation index does not.
Run:  python scanner.py        Test:  python -m pytest clinic_w3_scanner   (needs week17_core and week19_patterns)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                   # noqa: E402
from common import arrays, synthetic_universe              # noqa: E402

core = load("week17_core", "core")
pt = load("week19_patterns", "patterns")


def neckline_at(pattern: dict, t: int) -> float:
    """Neckline price at bar t (given): flat for double tops/bottoms, a line through the two troughs/peaks for H&S."""
    if "neck" in pattern:
        n1, _ = pattern["neck"]
        return n1.price + pattern["slope"] * (t - n1.idx)
    return pattern["neckline"]


def breakout_index(pattern: dict, close: np.ndarray, max_wait: int = 20) -> int | None:
    """First bar t in [known_from, known_from + max_wait] (inside the data) where the close breaks the neckline:
    BELOW it for 'double_top' and 'head_shoulders', ABOVE it for 'double_bottom' and 'inverse_head_shoulders'.
    None if it does not happen."""
    # >>> SOLUTION
    down = pattern["kind"] in ("double_top", "head_shoulders")
    start = pattern["known_from"]
    for t in range(start, min(start + max_wait + 1, close.size)):
        lvl = neckline_at(pattern, t)
        if (close[t] < lvl) if down else (close[t] > lvl):
            return t
    return None
    # <<< SOLUTION


def scan(df: pd.DataFrame, pct: float = 0.04, tol_atr: float = 1.0) -> pd.DataFrame:
    """Scan one symbol: ZigZag pivots (pt.zigzag on high/low with pct), ATR(14); tolerance for each pattern =
    tol_atr × ATR at the pattern's LAST pivot's confirm_idx (known at that time). Detect pt.double_tops_bottoms
    (min_sep 5) and pt.head_shoulders; keep those with a breakout. Return a DataFrame with columns
    kind, known_from, signal_idx, level (neckline at the signal bar), sorted by signal_idx then kind."""
    # >>> SOLUTION
    _, h, l, c, _ = arrays(df)  # noqa: E741
    a = core.atr(h, l, c, 14)
    pivots = pt.zigzag(h, l, pct)
    rows = []

    def tol_for(p):
        v = a[min(p.confirm_idx, a.size - 1)]
        return tol_atr * v if np.isfinite(v) else 0.0

    found = []
    for i in range(len(pivots)):
        window = pivots[max(0, i - 4):i + 1]
        tol = tol_for(pivots[i])
        found += [p for p in pt.double_tops_bottoms(window[-3:], tol, 5) if p["second"] is pivots[i]]
        if len(window) == 5:
            found += pt.head_shoulders(window, tol)
    for p in found:
        t = breakout_index(p, c)
        if t is not None:
            rows.append({"kind": p["kind"], "known_from": p["known_from"], "signal_idx": t,
                         "level": neckline_at(p, t)})
    out = pd.DataFrame(rows, columns=["kind", "known_from", "signal_idx", "level"])
    return out.sort_values(["signal_idx", "kind"], ignore_index=True)
    # <<< SOLUTION


def scan_with_pivot_index_bug(df: pd.DataFrame, pct: float = 0.04, tol_atr: float = 1.0) -> pd.DataFrame:
    """The classic bug (given): treat a pattern as known at its last PIVOT index instead of its confirmation index."""
    orig = pt.zigzag

    def zigzag_bug(high, low, p):
        return [pt.Pivot(v.idx, v.idx, v.price, v.kind) for v in orig(high, low, p)]
    pt.zigzag = zigzag_bug
    try:
        return scan(df, pct, tol_atr)
    finally:
        pt.zigzag = orig


def truncation_audit(df: pd.DataFrame, scanner=scan) -> list[dict]:
    """For every signal of scanner(df), re-run scanner(df.iloc[:signal_idx + 1]) and check that a signal with the same
    kind and signal_idx is found. Return the signals that were NOT reproduced (as dicts); empty list = passed."""
    # >>> SOLUTION
    failures = []
    for sig in scanner(df).to_dict("records"):
        again = scanner(df.iloc[:sig["signal_idx"] + 1])
        if not ((again["kind"] == sig["kind"]) & (again["signal_idx"] == sig["signal_idx"])).any():
            failures.append(sig)
    return failures
    # <<< SOLUTION


if __name__ == "__main__":
    for sym, df in list(synthetic_universe(5, 1500).items()):
        s = scan(df)
        print(f"{sym}: {len(s)} signals {s['kind'].value_counts().to_dict()} | audit failures: "
              f"honest {len(truncation_audit(df))}, pivot-index bug {len(truncation_audit(df, scan_with_pivot_index_bug))}")
