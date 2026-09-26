"""Shared, complete helpers for the Part 5 labs (nothing to fill in here).

`synthetic_ohlcv` builds reproducible OHLCV bars with volatility clustering, overnight gaps and fat tails, so every
lab runs offline. For graded research, load your cached Part 4 Parquet bars instead (same column names).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GOLDEN = Path(__file__).resolve().parent / "golden"


def synthetic_ohlcv(n: int = 1500, seed: int = 0, start: str = "2020-01-02", freq: str = "B",
                    price0: float = 100.0, drift: float = 0.0002) -> pd.DataFrame:
    """Daily-style bars (index = bar open time, UTC) with columns open, high, low, close, volume.
    GARCH(1,1)-like volatility, Student-t shocks, overnight gaps; high >= max(open, close), low <= min(open, close)."""
    rng = np.random.default_rng(seed)
    var, lr = 0.0001, np.empty(n)
    for t in range(n):
        z = rng.standard_t(5) / np.sqrt(5 / 3)
        lr[t] = drift + np.sqrt(var) * z
        var = 0.000002 + 0.08 * lr[t] ** 2 + 0.9 * var
    close = price0 * np.exp(np.cumsum(lr))
    prev = np.concatenate([[price0], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.3, n) * np.abs(lr))          # overnight gap
    span = np.abs(lr) + rng.gamma(2.0, 0.004, n)
    high = np.maximum(open_, close) * np.exp(rng.uniform(0.1, 0.6, n) * span)
    low = np.minimum(open_, close) * np.exp(-rng.uniform(0.1, 0.6, n) * span)
    volume = np.round(1e6 * np.exp(rng.normal(0, 0.3, n)) * (1 + 40 * np.abs(lr)))
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def synthetic_universe(n_symbols: int = 20, n: int = 1000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Independent synthetic symbols S00, S01, ... (for edge studies: no pattern has a real edge here)."""
    return {f"S{i:02d}": synthetic_ohlcv(n, seed=seed * 1000 + i) for i in range(n_symbols)}


def arrays(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """(open, high, low, close, volume) as float64 arrays."""
    return tuple(df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close", "volume"))


def load_golden(name: str) -> dict[str, np.ndarray]:
    """Reference outputs computed with TA-Lib on synthetic_ohlcv(1500, seed=7) (see tools/make_golden.py)."""
    with np.load(GOLDEN / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}
