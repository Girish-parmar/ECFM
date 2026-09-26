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
    # >>> SOLUTION
    s = np.asarray(trial_srs, dtype=float)
    n, v = s.size, s.var(ddof=1)
    return float(np.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e))))
    # <<< SOLUTION


def deflated_sharpe(returns, trial_srs) -> tuple[float, float]:
    """(DSR, SR0): the PSR of `returns` against SR0 = expected_max_sharpe(trial_srs) instead of 0."""
    # >>> SOLUTION
    sr0 = expected_max_sharpe(trial_srs)
    return an.psr(returns, sr0), sr0
    # <<< SOLUTION


# ------------------------------------------------------------------------------- S14 PBO
def pbo_cscv(perf: np.ndarray, S: int = 16) -> tuple[float, np.ndarray]:
    """Probability of Backtest Overfitting by CSCV. perf: T × N per-period returns of N configurations.
    Split rows into S contiguous blocks; for every choice of S/2 blocks as in-sample (itertools.combinations order):
    best = argmax of the in-sample Sharpe (mean/std ddof=1 per column); its OUT-OF-SAMPLE relative rank
    w = (rank among the N OOS Sharpes, 1 = worst) / (N + 1); logit = ln(w / (1 − w)).
    PBO = share of logits <= 0 (the in-sample winner is at or below the OOS median). Return (pbo, logits)."""
    # >>> SOLUTION
    T, N = perf.shape
    blocks = np.array_split(np.arange(T), S)

    def sr(x):
        return x.mean(0) / x.std(0, ddof=1)
    logits = []
    for is_blocks in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in is_blocks])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in is_blocks])
        best = int(np.argmax(sr(perf[is_idx])))
        w = (sr(perf[oos_idx]).argsort().argsort()[best] + 1) / (N + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits
    # <<< SOLUTION


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
    # >>> SOLUTION
    def sr(p):
        p = np.asarray(p, dtype=float)
        return p.mean() / p.std(ddof=1) * np.sqrt(252)
    base = run(params, 0, 1.0)
    sb = sr(base)
    rows = []
    for k, v in params.items():
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            for sign, lab in ((-1, "-"), (1, "+")):
                p2 = dict(params, **{k: int(round(v * (1 + sign * perturb)))})
                s = sr(run(p2, 0, 1.0))
                rows.append((f"{k} {lab}{int(perturb * 100)}%", s, s >= 0.5 * sb))
    s = sr(run(params, 1, 1.0))
    rows.append(("delay 1 bar", s, s >= 0.5 * sb))
    s = sr(run(params, 0, 2.0))
    rows.append(("costs x2", s, s > 0))
    s = sr(np.delete(base, np.argsort(base)[-5:]))
    rows.append(("without best 5 days", s, s > 0))
    h = base.size // 2
    rows += [("first half", sr(base[:h]), sr(base[:h]) > 0), ("second half", sr(base[h:]), sr(base[h:]) > 0)]
    return pd.DataFrame(rows, columns=["test", "value", "passed"])
    # <<< SOLUTION


# --------------------------------------------------------------------------- S16 the gate
GATE = {"min_oos_sharpe": 0.5, "min_dsr": 0.95, "max_pbo": 0.2, "min_robustness": 0.8, "max_drawdown": -0.25}


def validation_gate(oos_pnl, trial_srs, pbo: float, scorecard: pd.DataFrame, criteria: dict | None = None) -> dict:
    """Checks (all must pass): oos_sharpe (annualized) >= min_oos_sharpe; dsr (deflated_sharpe of the OOS P&L against
    the trial Sharpes, per-period) >= min_dsr; pbo <= max_pbo; robustness (share of scorecard tests passed) >=
    min_robustness; max_dd (an.max_drawdown of the OOS P&L) >= max_drawdown. Return {"checks": {name: (value, passed)},
    "passed": bool} with names oos_sharpe, dsr, pbo, robustness, max_dd."""
    # >>> SOLUTION
    c = {**GATE, **(criteria or {})}
    r = np.asarray(oos_pnl, dtype=float)
    oos_sr = r.mean() / r.std(ddof=1) * np.sqrt(252)
    dsr, _ = deflated_sharpe(r, trial_srs)
    rob = float(scorecard["passed"].mean())
    mdd = an.max_drawdown(r)[0]
    checks = {"oos_sharpe": (oos_sr, oos_sr >= c["min_oos_sharpe"]), "dsr": (dsr, dsr >= c["min_dsr"]),
              "pbo": (pbo, pbo <= c["max_pbo"]), "robustness": (rob, rob >= c["min_robustness"]),
              "max_dd": (mdd, mdd >= c["max_drawdown"])}
    return {"checks": checks, "passed": all(p for _, p in checks.values())}
    # <<< SOLUTION


def dossier_markdown(name: str, gate: dict, scorecard: pd.DataFrame) -> str:
    """Validation dossier (given): verdict, gate checks and the robustness scorecard."""
    lines = [f"# Validation dossier: {name}", f"**Verdict: {'PASS' if gate['passed'] else 'FAIL'}**", "",
             "| Check | Value | Passed |", "|---|---|---|"]
    lines += [f"| {k} | {v:.3f} | {'yes' if ok else 'NO'} |" for k, (v, ok) in gate["checks"].items()]
    lines += ["", "## Robustness", "", "| Test | Sharpe | Passed |", "|---|---|---|"]
    lines += [f"| {t} | {v:.2f} | {'yes' if ok else 'NO'} |" for t, v, ok in scorecard.itertuples(index=False)]
    return "\n".join(lines) + "\n"
