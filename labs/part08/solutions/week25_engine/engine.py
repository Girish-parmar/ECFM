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
    # >>> SOLUTION
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
    # <<< SOLUTION


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
        # >>> SOLUTION
        run_id = self.con.execute("SELECT coalesce(max(run_id), 0) + 1 FROM runs").fetchone()[0]
        h = config_hash({"strategy": strategy, "params": params, "data_version": data_version})
        self.con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         [run_id, strategy, h, json.dumps(params, sort_keys=True), data_version, sharpe, max_dd, n_obs])
        return int(run_id)
        # <<< SOLUTION

    def trials(self, strategy: str | None = None) -> pd.DataFrame:
        """All runs (of one strategy if given), ordered by run_id, as a DataFrame."""
        # >>> SOLUTION
        if strategy is None:
            return self.con.execute("SELECT * FROM runs ORDER BY run_id").df()
        return self.con.execute("SELECT * FROM runs WHERE strategy = ? ORDER BY run_id", [strategy]).df()
        # <<< SOLUTION


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
        # >>> SOLUTION
        heapq.heappush(self._h, Event(ts, priority, self._seq, kind, data))
        self._seq += 1
        # <<< SOLUTION

    def pop(self) -> Event:
        # >>> SOLUTION
        return heapq.heappop(self._h)
        # <<< SOLUTION

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
        # >>> SOLUTION
        m = self.mult.get(sym, 1.0)
        self.cash -= qty * price * m + fee
        self.fees += fee
        self.pos[sym] = self.pos.get(sym, 0.0) + qty
        # <<< SOLUTION

    def mark(self, sym: str, price: float) -> None:
        self.last[sym] = price

    @property
    def equity(self) -> float:
        """cash + Σ qty × last price × multiplier."""
        # >>> SOLUTION
        return self.cash + sum(q * self.last.get(s, 0.0) * self.mult.get(s, 1.0) for s, q in self.pos.items())
        # <<< SOLUTION


# ------------------------------------------------------------------------ S3 fills & costs
def market_fill_price(qty: float, open_price: float, slippage_bps: float = 0.0) -> float:
    """Market order at the next bar's open, paying slippage: buy at open × (1 + bps/1e4), sell at open × (1 − bps/1e4)."""
    # >>> SOLUTION
    return open_price * (1 + np.sign(qty) * slippage_bps / 1e4)
    # <<< SOLUTION


def limit_fill_price(qty: float, limit: float, bar: dict) -> float | None:
    """Limit orders fill only if price trades THROUGH the limit (touching is not enough: queue position is unknown):
    buy if low < limit, at min(open, limit); sell if high > limit, at max(open, limit). Else None."""
    # >>> SOLUTION
    if qty > 0:
        return min(bar["open"], limit) if bar["low"] < limit else None
    return max(bar["open"], limit) if bar["high"] > limit else None
    # <<< SOLUTION


def stop_fill_price(qty: float, stop: float, bar: dict) -> float | None:
    """Stop orders trigger at the stop or the GAP OPEN, whichever is worse: buy stop if high >= stop, at
    max(open, stop); sell stop if low <= stop, at min(open, stop). Else None."""
    # >>> SOLUTION
    if qty > 0:
        return max(bar["open"], stop) if bar["high"] >= stop else None
    return min(bar["open"], stop) if bar["low"] <= stop else None
    # <<< SOLUTION


def participation_cap(qty: float, bar_volume: float, max_participation: float = 0.10) -> float:
    """Partial fill: at most max_participation × bar volume (sign of qty kept)."""
    # >>> SOLUTION
    return float(np.sign(qty) * min(abs(qty), max_participation * bar_volume))
    # <<< SOLUTION


def ib_fixed_commission(shares: float, price: float) -> float:
    """IBKR Pro Fixed, US stocks: $0.005/share, min $1, max 1% of trade value (verify the current schedule); 0 for
    0 shares."""
    # >>> SOLUTION
    if shares == 0:
        return 0.0
    return float(np.clip(0.005 * abs(shares), 1.0, 0.01 * abs(shares) * price))
    # <<< SOLUTION


def sqrt_impact_bps(order_shares: float, adv_shares: float, daily_vol: float, k: float = 1.0) -> float:
    """Square-root impact: 1e4 · k · σ_daily · √(|Q| / ADV) basis points."""
    # >>> SOLUTION
    return float(1e4 * k * daily_vol * np.sqrt(abs(order_shares) / adv_shares))
    # <<< SOLUTION


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
    # >>> SOLUTION
    pf = Portfolio(cash)
    q = EventQueue()
    for t, ts in enumerate(bars.index):
        q.push(ts, 1, "bar", t)
    pending = 0.0
    rows = []
    o, c, v = bars["open"].to_numpy(), bars["close"].to_numpy(), bars["volume"].to_numpy()
    while q:
        ev = q.pop()
        t = ev.data
        if pending:
            qty = participation_cap(pending, v[t], max_participation) if max_participation else pending
            px = market_fill_price(qty, o[t], slippage_bps)
            fee = commission(qty, px) if commission else 0.0
            pf.on_fill(symbol, qty, px, fee)
            pending = 0.0
        pf.mark(symbol, o[t])
        eq_open = pf.equity
        pf.mark(symbol, c[t])
        eq = pf.equity
        target = round(float(decided[t]) * eq / c[t])
        pending = target - pf.pos.get(symbol, 0.0)
        rows.append((pf.pos.get(symbol, 0.0), eq_open, eq, pf.fees))
    return pd.DataFrame(rows, index=bars.index, columns=["position", "equity_open", "equity", "fees"])
    # <<< SOLUTION
