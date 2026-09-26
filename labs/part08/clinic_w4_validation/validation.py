"""Clinic W4 — Validation dossiers: defend a strategy, then let the gate try to break it (Part 8, S9–S16).

For each strategy: log every configuration tried (research log), walk-forward optimize, measure PBO over the
configuration matrix, compute the DSR against all logged trials, run the robustness scorecard, and apply the gate.
A 'noise miner' (the best of many random signals) is included on purpose: the gate must reject it.
Run:  python validation.py     Test:  python -m pytest clinic_w4_validation   (needs weeks 25–28)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                          # noqa: E402
from common import quick_eval_pnl, regime_market, sma_cross_signal, tsmom_signal  # noqa: E402

en = load("week25_engine", "engine")
op = load("week27_optimization", "optimization")
ov = load("week28_overfitting", "overfitting")


def make_runner(bars, signal_fn):
    """run(params, delay, cost_mult) -> daily P&L (given): signal decided at the close, optionally delayed by bars."""
    o = bars["open"].to_numpy()

    def run(params, delay=0, cost_mult=1.0):
        d = signal_fn(**params)
        if delay:
            d = np.concatenate([np.zeros(delay), d[:-delay]])
        return quick_eval_pnl(d, o, cost_bps=2.0 * cost_mult)
    return run


def strategies(bars) -> dict:
    """name -> (signal_fn(**params), grid) (given). The noise miner's only 'parameter' is a random seed."""
    c = bars["close"].to_numpy()

    def noise(seed):
        return np.sign(np.random.default_rng(seed).standard_normal(c.size))
    return {
        "tsmom": (lambda lookback, vol_n: tsmom_signal(c, lookback, vol_n),
                  {"lookback": [60, 120, 180, 250], "vol_n": [20, 60]}),
        "sma_cross": (lambda fast, slow: sma_cross_signal(c, fast, slow),
                      {"fast": [10, 20, 50], "slow": [100, 150, 200]}),
        "noise_miner": (noise, {"seed": list(range(40))}),
    }


def validate(name: str, bars, signal_fn, grid: dict, log, train: int = 750, test: int = 250) -> dict:
    """1. every configuration: pnl = run(params); record it in `log` (en.ResearchLog) with its annualized Sharpe and
          max drawdown (data_version 'synthetic-v1'); build the T × N matrix of their P&L;
       2. op.walk_forward_optimize(run with delay 0, cost 1 as pnl_fn, grid, len(bars), train, test);
       3. pbo = ov.pbo_cscv(matrix, S=10)[0];
       4. trial Sharpes (per-period!) = all Sharpes in the log for this strategy ÷ √252;
       5. scorecard = ov.robustness_scorecard(run, the params chosen in the LAST walk-forward window);
       6. gate = ov.validation_gate(stitched OOS P&L, trial Sharpes, pbo, scorecard).
    Return {"gate", "scorecard", "pbo", "wf" (the walk-forward result), "dossier" (ov.dossier_markdown)}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    bars = regime_market(n_blocks=24, seed=21)
    log = en.ResearchLog()
    for name, (fn, grid) in strategies(bars).items():
        res = validate(name, bars, fn, grid, log)
        print(res["dossier"])
    print(f"{len(log.trials())} configurations logged in total")
