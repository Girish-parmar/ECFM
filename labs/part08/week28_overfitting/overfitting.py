"""Week 28 (S13–S16) — Deflated Sharpe Ratio, Probability of Backtest Overfitting, robustness tests, validation gate.

If you try N unskilled strategies, the best one looks skilled. DSR and PBO correct for the search you did; the
research log (week 25) provides the trials. Uses your week 26 psr and max_drawdown.
Fill in every block marked "Your turn", then run:  python -m pytest week28_overfitting
"""
from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from _loader import load

an = load("week26_analysis", "analysis")
EULER = 0.5772156649015329


# -------------------------------------------------------------------------- S13 DSR
def expected_max_sharpe(trial_srs) -> float:
    """Expected maximum of N unskilled trial Sharpes (per-period units), with V = variance of the trial Sharpes
    (ddof=1) and γ = Euler–Mascheroni: √V · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def deflated_sharpe(returns, trial_srs) -> tuple[float, float]:
    """(DSR, SR0): the PSR of `returns` against SR0 = expected_max_sharpe(trial_srs) instead of 0."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------------- S14 PBO
def pbo_cscv(perf: np.ndarray, S: int = 16) -> tuple[float, np.ndarray]:
    """Probability of Backtest Overfitting by CSCV. perf: T × N per-period returns of N configurations.
    Split rows into S contiguous blocks; for every choice of S/2 blocks as in-sample (itertools.combinations order):
    best = argmax of the in-sample Sharpe (mean/std ddof=1 per column); its OUT-OF-SAMPLE relative rank
    w = (rank among the N OOS Sharpes, 1 = worst) / (N + 1); logit = ln(w / (1 − w)).
    PBO = share of logits <= 0 (the in-sample winner is at or below the OOS median). Return (pbo, logits)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------- S15 robustness
def robustness_scorecard(run: Callable[[dict, int, float], np.ndarray], params: dict,
                         perturb: float = 0.2) -> pd.DataFrame:
    """run(params, delay, cost_mult) returns the daily P&L of the strategy. Base = run(params, 0, 1.0), base Sharpe
    SR_b (annualized). Tests (name: value → passed):
      'param -{p}%'/'param +{p}%' for every INT parameter k: Sharpe with that parameter × (1 ∓ perturb), rounded to int
                                   → passed if >= 0.5·SR_b
      'delay 1 bar'               → Sharpe of run(params, 1, 1.0) >= 0.5·SR_b
      'costs x2'                  → Sharpe of run(params, 0, 2.0) > 0
      'without best 5 days'       → Sharpe of the base P&L with its 5 largest days removed > 0
      'first half' / 'second half'→ Sharpe of each half of the base P&L > 0
    Return a DataFrame with columns test, value, passed (in the order above; parameters in the dict's order,
    minus before plus)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------------- S16 the gate
GATE = {"min_oos_sharpe": 0.5, "min_dsr": 0.95, "max_pbo": 0.2, "min_robustness": 0.8, "max_drawdown": -0.25}


def validation_gate(oos_pnl, trial_srs, pbo: float, scorecard: pd.DataFrame, criteria: dict | None = None) -> dict:
    """Checks (all must pass): oos_sharpe (annualized) >= min_oos_sharpe; dsr (deflated_sharpe of the OOS P&L against
    the trial Sharpes, per-period) >= min_dsr; pbo <= max_pbo; robustness (share of scorecard tests passed) >=
    min_robustness; max_dd (an.max_drawdown of the OOS P&L) >= max_drawdown. Return {"checks": {name: (value, passed)},
    "passed": bool} with names oos_sharpe, dsr, pbo, robustness, max_dd."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def dossier_markdown(name: str, gate: dict, scorecard: pd.DataFrame) -> str:
    """Validation dossier (given): verdict, gate checks and the robustness scorecard."""
    lines = [f"# Validation dossier: {name}", f"**Verdict: {'PASS' if gate['passed'] else 'FAIL'}**", "",
             "| Check | Value | Passed |", "|---|---|---|"]
    lines += [f"| {k} | {v:.3f} | {'yes' if ok else 'NO'} |" for k, (v, ok) in gate["checks"].items()]
    lines += ["", "## Robustness", "", "| Test | Sharpe | Passed |", "|---|---|---|"]
    lines += [f"| {t} | {v:.2f} | {'yes' if ok else 'NO'} |" for t, v, ok in scorecard.itertuples(index=False)]
    return "\n".join(lines) + "\n"
