"""Week 27 (S9–S12) — Search, parameter plateaus, walk-forward analysis, purged k-fold and CPCV.

The stitched OUT-OF-SAMPLE curve is the only honest result of an optimization. Every configuration you try counts as a
trial (week 28 needs the count). Optuna (TPE) is a drop-in replacement for random_search in the clinics.
Fill in every block marked "Your turn", then run:  python -m pytest week27_optimization
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from math import comb

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------ S9 search
def grid_search(objective: Callable[..., float], grid: dict[str, list]) -> pd.DataFrame:
    """Evaluate objective(**params) on every combination (itertools.product in the grid's key order).
    Return a DataFrame with one column per parameter plus 'score', in evaluation order."""
    # >>> SOLUTION
    keys = list(grid)
    rows = [dict(zip(keys, combo)) | {"score": objective(**dict(zip(keys, combo)))}
            for combo in itertools.product(*grid.values())]
    return pd.DataFrame(rows)
    # <<< SOLUTION


def random_search(objective: Callable[..., float], space: dict[str, list], n: int, seed: int = 0) -> pd.DataFrame:
    """n trials, each parameter drawn uniformly from its list of allowed values (rng = default_rng(seed), draw the
    parameters in the space's key order with rng.choice). Same output format as grid_search."""
    # >>> SOLUTION
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        p = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        rows.append(p | {"score": objective(**p)})
    return pd.DataFrame(rows)
    # <<< SOLUTION


def plateau_scores(results: pd.DataFrame, x: str, y: str, radius: int = 1) -> pd.DataFrame:
    """For a 2-parameter GRID result: pivot score to a (y × x) table, then replace each cell by the mean of the cells
    within `radius` grid steps in both directions (a (2r+1)×(2r+1) window clipped at the edges). Prefer parameters
    on a broad plateau over a sharp, isolated peak. Return the smoothed table (same index/columns)."""
    # >>> SOLUTION
    tab = results.pivot(index=y, columns=x, values="score").sort_index().sort_index(axis=1)
    a = tab.to_numpy()
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = np.nanmean(a[max(0, i - radius):i + radius + 1, max(0, j - radius):j + radius + 1])
    return pd.DataFrame(out, index=tab.index, columns=tab.columns)
    # <<< SOLUTION


# ------------------------------------------------------------------------- S11 walk-forward
def walk_forward(n: int, train: int, test: int, anchored: bool = False):
    """Yield (train_idx, test_idx) arrays: test windows of `test` bars back to back after the first `train` bars;
    rolling train windows of `train` bars (anchored: from 0). Stop when a full test window no longer fits."""
    # >>> SOLUTION
    start = 0
    while start + train + test <= n:
        tr0 = 0 if anchored else start
        yield np.arange(tr0, start + train), np.arange(start + train, start + train + test)
        start += test
    # <<< SOLUTION


def sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def walk_forward_optimize(pnl_fn: Callable[..., np.ndarray], grid: dict[str, list], n: int, train: int, test: int,
                          anchored: bool = False) -> dict:
    """pnl_fn(**params) returns the full per-bar P&L series (causal: bar t uses data up to t only).
    For each walk-forward window pick the params with the best IN-SAMPLE Sharpe on the train slice, then record their
    OUT-OF-SAMPLE P&L on the test slice. Return {"oos": stitched OOS P&L (1-D array), "params": list of chosen params
    per window, "is_sharpe": list, "oos_sharpe": list, "wfe": mean(oos_sharpe) / mean(is_sharpe)}.
    (Cache pnl_fn per params: each configuration only needs to be computed once.)"""
    # >>> SOLUTION
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
    cache = {tuple(p.values()): pnl_fn(**p) for p in combos}
    oos, chosen, is_s, oos_s = [], [], [], []
    for tr, te in walk_forward(n, train, test, anchored):
        best = max(combos, key=lambda p: np.nan_to_num(sharpe(cache[tuple(p.values())][tr]), nan=-np.inf))
        pnl = cache[tuple(best.values())]
        chosen.append(best)
        is_s.append(sharpe(pnl[tr]))
        oos_s.append(sharpe(pnl[te]))
        oos.append(pnl[te])
    return {"oos": np.concatenate(oos), "params": chosen, "is_sharpe": is_s, "oos_sharpe": oos_s,
            "wfe": float(np.mean(oos_s) / np.mean(is_s))}
    # <<< SOLUTION


# ---------------------------------------------------------------------- S12 purged CV & CPCV
def purged_kfold(n: int, k: int = 5, label_horizon: int = 5, embargo: float = 0.01):
    """Yield (train_idx, test_idx): test folds are k contiguous blocks (np.array_split). Training keeps i only if
    its label window [i, i + label_horizon] ends BEFORE the fold starts (i + label_horizon < lo) or i starts after
    the fold plus an embargo of ceil(embargo × n) bars (i > hi + emb)."""
    # >>> SOLUTION
    emb = int(np.ceil(embargo * n))
    idx = np.arange(n)
    for test in np.array_split(idx, k):
        lo, hi = test[0], test[-1]
        yield idx[(idx + label_horizon < lo) | (idx > hi + emb)], test
    # <<< SOLUTION


def cpcv_splits(n: int, n_groups: int = 6, k_test: int = 2, label_horizon: int = 5, embargo: float = 0.01) -> list:
    """Combinatorial purged CV: split 0..n−1 into n_groups contiguous groups; for EVERY combination of k_test groups
    (itertools.combinations order) the test set is their union and training is everything else, purged and embargoed
    around EACH test group as in purged_kfold. Return a list of (train_idx, test_groups tuple, test_idx)."""
    # >>> SOLUTION
    emb = int(np.ceil(embargo * n))
    idx = np.arange(n)
    groups = np.array_split(idx, n_groups)
    out = []
    for combo in itertools.combinations(range(n_groups), k_test):
        keep = np.ones(n, dtype=bool)
        for g in combo:
            lo, hi = groups[g][0], groups[g][-1]
            keep &= (idx + label_horizon < lo) | (idx > hi + emb)
        out.append((idx[keep], combo, np.concatenate([groups[g] for g in combo])))
    return out
    # <<< SOLUTION


def cpcv_n_paths(n_groups: int, k_test: int) -> int:
    """Number of complete out-of-sample backtest paths CPCV produces: C(N, k)·k / N (each group is tested
    C(N−1, k−1) times)."""
    # >>> SOLUTION
    return comb(n_groups, k_test) * k_test // n_groups
    # <<< SOLUTION
