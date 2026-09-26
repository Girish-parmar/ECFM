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
    raise NotImplementedError("✍️ Your turn: see the docstring")


def check_warmup(fn, x: np.ndarray, lookback: int) -> tuple[bool, str]:
    """Every output is NaN at indices < lookback and finite at every index >= lookback (x has no NaN).
    Details: 'value before look-back at i' or 'NaN after look-back at i' (first offending index)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def check_no_lookahead(fn, x: np.ndarray, cuts=(0.5, 0.8)) -> tuple[bool, str]:
    """For each fraction f in cuts, m = int(f × len(x)): fn(x[:m]) must equal fn(x)[:m] (NaN == NaN, rtol 1e-9).
    Detail on failure: 'changes when bars after m are removed' with the first m that fails."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def check_scale_invariant(fn, x: np.ndarray, factor: float = 10.0) -> tuple[bool, str]:
    """fn(x × factor) == fn(x) for every output (rtol 1e-7, NaN == NaN). Detail: 'not scale invariant'."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def audit(indicators: dict[str, tuple], x: np.ndarray, scale_invariant: set[str] = frozenset()) -> list[dict]:
    """indicators: {name: (fn taking one price array, lookback)}. Run length, warmup, no_lookahead on each, plus
    scale_invariant for names in that set. Return rows {"name", "check", "ok", "detail"} in that order
    (check names: 'length', 'warmup', 'no_lookahead', 'scale_invariant')."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


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
