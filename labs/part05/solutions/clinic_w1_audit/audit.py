"""Clinic W1 — Generic property audit for every registered indicator (Part 5, S1–S3 and the M2 release checklist).

Golden tests check values against TA-Lib; this audit checks the API CONTRACT of any indicator, including ones TA-Lib
does not have: same length as the input, NaN exactly during the declared look-back, no look-ahead (values do not change
when future bars are removed) and, where expected, scale invariance (RSI of prices × 10 == RSI of prices).
Run:  python audit.py          Test:  python -m pytest clinic_w1_audit       (needs week17_core done)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                  # noqa: E402
from common import synthetic_ohlcv                        # noqa: E402

core = load("week17_core", "core")


def _outputs(result) -> list[np.ndarray]:
    """An indicator returns one array or a tuple of arrays (given)."""
    return [np.asarray(r, dtype=float) for r in (result if isinstance(result, tuple) else (result,))]


def check_length(fn, x: np.ndarray) -> tuple[bool, str]:
    """Every output has the same length as x. Return (ok, detail) with detail '' when ok, else e.g. 'len 90 != 100'."""
    # >>> SOLUTION
    for out in _outputs(fn(x)):
        if out.shape != x.shape:
            return False, f"len {out.shape[0]} != {x.shape[0]}"
    return True, ""
    # <<< SOLUTION


def check_warmup(fn, x: np.ndarray, lookback: int) -> tuple[bool, str]:
    """Every output is NaN at indices < lookback and finite at every index >= lookback (x has no NaN).
    Details: 'value before look-back at i' or 'NaN after look-back at i' (first offending index)."""
    # >>> SOLUTION
    for out in _outputs(fn(x)):
        bad = np.flatnonzero(~np.isnan(out[:lookback]))
        if bad.size:
            return False, f"value before look-back at {bad[0]}"
        bad = np.flatnonzero(~np.isfinite(out[lookback:]))
        if bad.size:
            return False, f"NaN after look-back at {bad[0] + lookback}"
    return True, ""
    # <<< SOLUTION


def check_no_lookahead(fn, x: np.ndarray, cuts=(0.5, 0.8)) -> tuple[bool, str]:
    """For each fraction f in cuts, m = int(f × len(x)): fn(x[:m]) must equal fn(x)[:m] (NaN == NaN, rtol 1e-9).
    Detail on failure: 'changes when bars after m are removed' with the first m that fails."""
    # >>> SOLUTION
    full = _outputs(fn(x))
    for f in cuts:
        m = int(f * len(x))
        for a, b in zip(full, _outputs(fn(x[:m]))):
            if not np.allclose(a[:m], b, rtol=1e-9, atol=0, equal_nan=True):
                return False, f"changes when bars after {m} are removed"
    return True, ""
    # <<< SOLUTION


def check_scale_invariant(fn, x: np.ndarray, factor: float = 10.0) -> tuple[bool, str]:
    """fn(x × factor) == fn(x) for every output (rtol 1e-7, NaN == NaN). Detail: 'not scale invariant'."""
    # >>> SOLUTION
    for a, b in zip(_outputs(fn(x)), _outputs(fn(x * factor))):
        if not np.allclose(a, b, rtol=1e-7, atol=1e-9, equal_nan=True):
            return False, "not scale invariant"
    return True, ""
    # <<< SOLUTION


def audit(indicators: dict[str, tuple], x: np.ndarray, scale_invariant: set[str] = frozenset()) -> list[dict]:
    """indicators: {name: (fn taking one price array, lookback)}. Run length, warmup, no_lookahead on each, plus
    scale_invariant for names in that set. Return rows {"name", "check", "ok", "detail"} in that order
    (check names: 'length', 'warmup', 'no_lookahead', 'scale_invariant')."""
    # >>> SOLUTION
    rows = []
    for name, (fn, lb) in indicators.items():
        checks = [("length", check_length(fn, x)), ("warmup", check_warmup(fn, x, lb)),
                  ("no_lookahead", check_no_lookahead(fn, x))]
        if name in scale_invariant:
            checks.append(("scale_invariant", check_scale_invariant(fn, x)))
        rows += [{"name": name, "check": c, "ok": ok, "detail": d} for c, (ok, d) in checks]
    return rows
    # <<< SOLUTION


def registry_close_only() -> dict[str, tuple]:
    """Registered indicators that take only a close array, with their default look-back (given)."""
    out = {}
    for name in ("sma", "ema", "rsi", "macd", "bbands"):
        _, fn = core.REGISTRY[name]
        out[name] = (fn, core.default_lookback(name))
    return out


if __name__ == "__main__":
    x = synthetic_ohlcv(600, seed=1)["close"].to_numpy()
    for r in audit(registry_close_only(), x, {"rsi"}):
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['name']:<8} {r['check']:<16} {r['detail']}")
