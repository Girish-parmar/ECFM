import numpy as np
from _loader import load
from common import synthetic_universe

sc = load("clinic_w3_scanner", "scanner")
Pivot = load("week19_patterns", "patterns").Pivot
UNI = synthetic_universe(3, 1200, seed=5)


def test_breakout_index():
    top = {"kind": "double_top", "neckline": 100.0, "known_from": 3}
    close = np.array([105, 104, 103, 101, 100, 99.5, 98, 97.0])
    assert sc.breakout_index(top, close) == 5                  # 100 is not below 100
    assert sc.breakout_index(top, close, max_wait=1) is None
    bottom = {"kind": "double_bottom", "neckline": 100.0, "known_from": 1}
    assert sc.breakout_index(bottom, np.array([101.0, 99, 99.5, 100.2])) == 3   # bar 0 is before known_from
    n1, n2 = Pivot(10, 12, 100.0, -1), Pivot(20, 22, 102.0, -1)
    hs = {"kind": "head_shoulders", "neck": (n1, n2), "slope": 0.2, "known_from": 30}
    close = np.full(40, 110.0)
    close[33] = 104.5                                           # neckline at 33 = 104.6
    assert sc.breakout_index(hs, close) == 33


def test_scan_output():
    s = sc.scan(UNI["S00"])
    assert list(s.columns) == ["kind", "known_from", "signal_idx", "level"] and len(s) > 5
    assert (s["signal_idx"] >= s["known_from"]).all() and s["signal_idx"].is_monotonic_increasing
    assert set(s["kind"]) <= {"double_top", "double_bottom", "head_shoulders", "inverse_head_shoulders"}


def test_honest_scanner_passes_the_truncation_audit():
    for df in UNI.values():
        assert sc.truncation_audit(df) == []


def test_pivot_index_bug_is_caught():
    failures = sum(len(sc.truncation_audit(df, sc.scan_with_pivot_index_bug)) for df in UNI.values())
    assert failures > 0
