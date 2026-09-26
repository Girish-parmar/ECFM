"""Shared, complete helpers for the Part 6 labs (nothing to fill in here)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "golden"
DATA = ROOT / "data"


def load_golden(name: str) -> dict[str, np.ndarray]:
    """Reference values computed with vollib / py_vollib (see tools/make_data.py)."""
    with np.load(GOLDEN / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def load_chain() -> tuple[pd.DataFrame, dict]:
    """Recorded SPY-like chain snapshot (synthetic, IB modelGreeks units) and its metadata (spot, as-of time)."""
    chain = pd.read_csv(DATA / "spy_chain.csv", parse_dates=["expiry"])
    chain["expiry"] = chain["expiry"].dt.date
    meta = pd.read_json(DATA / "spy_chain_meta.json", typ="series").to_dict()
    return chain, meta
