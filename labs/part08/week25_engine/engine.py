"""Week 25 (S1–S4) — Research log, event-driven backtest engine, fill and cost models, parity with the first look.

Rules: an order created at the CLOSE of bar t is filled at the OPEN of bar t+1 (never on its own bar); at the same
timestamp fills are processed before bars, bars before timers; the same inputs always give the same result.
Fill in every block marked "Your turn", then run:  python -m pytest week25_engine
"""
from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- S1 research log
def config_hash(config: dict) -> str:
    """First 12 hex chars of sha256 of json.dumps(config, sort_keys=True) — the same config always hashes the same,
    whatever the key order."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


class ResearchLog:
    """Every backtest run is recorded — the Deflated Sharpe Ratio (week 28) needs the number of trials and the
    spread of their Sharpes. Table `runs`: run_id INTEGER, strategy, config_hash, params (JSON text), data_version,
    sharpe DOUBLE, max_dd DOUBLE, n_obs INTEGER."""

    def __init__(self, path: str = ":memory:"):
        self.con = duckdb.connect(path)
        self.con.execute("""CREATE TABLE IF NOT EXISTS runs (run_id INTEGER, strategy VARCHAR, config_hash VARCHAR,
                            params VARCHAR, data_version VARCHAR, sharpe DOUBLE, max_dd DOUBLE, n_obs INTEGER)""")

    def record(self, strategy: str, params: dict, data_version: str, sharpe: float, max_dd: float, n_obs: int) -> int:
        """Insert one run with run_id = 1 + the current max run_id (1 for the first); return the run_id.
        config_hash covers {"strategy": ..., "params": ..., "data_version": ...}."""
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def trials(self, strategy: str | None = None) -> pd.DataFrame:
        """All runs (of one strategy if given), ordered by run_id, as a DataFrame."""
        raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------ S2 events & ledger
@dataclass(order=True)
class Event:
    ts: pd.Timestamp
    priority: int                        # same timestamp: 0 fill, 1 bar, 2 timer
    seq: int                             # insertion order breaks remaining ties (determinism)
    kind: str = field(compare=False)
    data: object = field(compare=False, default=None)


class EventQueue:
    """Min-heap of events ordered by (ts, priority, seq)."""

    def __init__(self):
        self._h: list[Event] = []
        self._seq = 0

    def push(self, ts, priority: int, kind: str, data=None) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def pop(self) -> Event:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def __bool__(self) -> bool:
        return bool(self._h)


class Portfolio:
    """Cash, positions, last prices and contract multipliers (futures: ES 50)."""

    def __init__(self, cash: float, multipliers: dict[str, float] | None = None):
        self.cash, self.pos, self.last = float(cash), {}, {}
        self.mult = dict(multipliers or {})
        self.fees = 0.0

    def on_fill(self, sym: str, qty: float, price: float, fee: float = 0.0) -> None:
        """Cash pays qty × price × multiplier plus the fee; the position changes by qty; fees are accumulated."""
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def mark(self, sym: str, price: float) -> None:
        self.last[sym] = price

    @property
    def equity(self) -> float:
        """cash + Σ qty × last price × multiplier."""
        raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------ S3 fills & costs
def market_fill_price(qty: float, open_price: float, slippage_bps: float = 0.0) -> float:
    """Market order at the next bar's open, paying slippage: buy at open × (1 + bps/1e4), sell at open × (1 − bps/1e4)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def limit_fill_price(qty: float, limit: float, bar: dict) -> float | None:
    """Limit orders fill only if price trades THROUGH the limit (touching is not enough: queue position is unknown):
    buy if low < limit, at min(open, limit); sell if high > limit, at max(open, limit). Else None."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def stop_fill_price(qty: float, stop: float, bar: dict) -> float | None:
    """Stop orders trigger at the stop or the GAP OPEN, whichever is worse: buy stop if high >= stop, at
    max(open, stop); sell stop if low <= stop, at min(open, stop). Else None."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def participation_cap(qty: float, bar_volume: float, max_participation: float = 0.10) -> float:
    """Partial fill: at most max_participation × bar volume (sign of qty kept)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def ib_fixed_commission(shares: float, price: float) -> float:
    """IBKR Pro Fixed, US stocks: $0.005/share, min $1, max 1% of trade value (verify the current schedule); 0 for
    0 shares."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def sqrt_impact_bps(order_shares: float, adv_shares: float, daily_vol: float, k: float = 1.0) -> float:
    """Square-root impact: 1e4 · k · σ_daily · √(|Q| / ADV) basis points."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ----------------------------------------------------------------------- S2–S4 the engine
def run_backtest(bars: pd.DataFrame, decided: np.ndarray, cash: float = 1_000_000.0, symbol: str = "X",
                 slippage_bps: float = 0.0, commission=None, max_participation: float | None = None) -> pd.DataFrame:
    """Event-driven single-symbol backtest of target WEIGHTS decided at each close.
    Use an EventQueue: for every bar push a 'bar' event at its timestamp (priority 1). Processing a bar event t:
      1. if an order is pending (created at the previous close): fill it at this bar's open with market_fill_price,
         capped by participation_cap(bar volume) if max_participation is set; fee = commission(qty, price) if a
         commission function is given, else 0; ledger.on_fill;
      2. mark the position at the OPEN and record equity_open;
      3. mark at the CLOSE, record equity; the target shares = round(decided[t] × equity / close); if it differs from
         the position, the difference becomes the pending order (filled at the NEXT bar's open).
    Return a DataFrame indexed like bars with columns position (after fills), equity_open, equity, fees (cumulative)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
