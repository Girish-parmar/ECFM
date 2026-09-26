"""Week 16 — Safety, multi-asset and integration (Part 4, S13–S16): futures/FX helpers, pre-trade checks,
a simulated broker adapter that passes the broker contract tests, and a sticky, idempotent kill switch.

Fill in every block marked "Your turn", then run:  python -m pytest week16_safety
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from common import (AssetClass, DuplicateOrder, Instrument, KillSwitchTripped, OrderRequest, OrderState, OrdType,
                    Side, UnsupportedOrder)

S = OrderState
ZERO = Decimal("0")


# ---------------------------------------------------------------- S14 futures & FX helpers
def tick_value(tick: Decimal, multiplier: Decimal) -> Decimal:
    """Money value of one tick for one contract (ES: 0.25 × 50 = 12.50)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def futures_pnl(entry: Decimal, exit: Decimal, qty: Decimal, multiplier: Decimal) -> Decimal:
    """Signed P&L: qty > 0 long, qty < 0 short."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def pip_value_usd(pair: str, units: Decimal, usd_per_quote: Decimal) -> Decimal:
    """USD value of one pip for `units` of the BASE currency. Pip = 0.01 for JPY-quoted pairs, else 0.0001.
    The pip is worth pip × units in the QUOTE currency; convert with usd_per_quote (1 for XXXUSD pairs)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def third_friday(year: int, month: int) -> date:
    """Third Friday of the month (quarterly equity-index futures such as ES/MES expire then)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def roll_date(expiry: str, business_days_before: int = 8) -> date:
    """Roll date for an equity-index future with expiry 'YYYYMM': go back `business_days_before` weekdays
    (Mon–Fri; ignore holidays in this exercise) from its third Friday."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------ S15 pre-trade checks
@dataclass
class TradeContext:
    kill_switch_tripped: bool = False
    tradable: set[str] = field(default_factory=set)
    session_open: bool = True
    last: dict[str, Decimal] = field(default_factory=dict)
    fat_finger_pct: Decimal = Decimal("5")         # limit price may deviate at most this % from last
    max_qty: Decimal = Decimal("1000")
    buying_power: Decimal = Decimal("0")
    shortable: set[str] = field(default_factory=set)
    position: dict[str, Decimal] = field(default_factory=dict)


def pretrade_check(req: OrderRequest, ctx: TradeContext) -> str | None:
    """Run the checks IN THIS ORDER and return the first failure's reason, or None if the order may go:
     1 'kill switch tripped'
     2 'not tradable'                  symbol not in ctx.tradable
     3 'session closed'                unless req.extended_hours
     4 'no last price'                 symbol missing from ctx.last
     5 'fat finger'                    limit_price deviates more than fat_finger_pct % from last
     6 'qty above max'
     7 'insufficient buying power'     BUY only: qty × (limit_price or last) > buying_power
     8 'not shortable'                 SELL that takes the position below zero, symbol not in ctx.shortable"""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------- S16 simulated broker
@dataclass
class SimFill:
    client_order_id: str
    symbol: str
    qty: Decimal                    # signed: + buy, − sell
    price: Decimal


class SimBroker:
    """In-memory BrokerAdapter used in CI and backtests. Same async interface as the IB and Alpaca adapters:
    connect, disconnect, submit, cancel, cancel_all, flatten_all, plus sim-only on_price.

    submit(req): DuplicateOrder if the client order ID was EVER seen. Only MARKET, LIMIT and STOP are
      supported (else UnsupportedOrder). The order becomes ACCEPTED, then is matched against the last price:
        MARKET fills at last · LIMIT BUY fills if last <= limit (SELL: last >= limit) at the LIMIT price ·
        STOP BUY triggers if last >= stop (SELL: last <= stop) and fills at last.
      Unfilled orders rest in `self.open` and are re-matched on every on_price(symbol, price).
      Returns the client order ID.
    cancel(coid): a resting order becomes CANCELLED; a terminal or unknown order is left alone (no error).
    cancel_all(): cancel every resting order; return how many were cancelled.
    flatten_all(): send a MARKET order that closes every non-zero position (client ID 'flatten-<n>'); return
      the number of orders sent."""

    name = "sim"

    def __init__(self, prices: dict[str, Decimal] | None = None):
        self.last: dict[str, Decimal] = dict(prices or {})
        self.open: dict[str, OrderRequest] = {}
        self.states: dict[str, OrderState] = {}
        self.positions: dict[str, Decimal] = {}
        self.fills: list[SimFill] = []
        self.connected = False
        self._n = 0

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    def _fill(self, req: OrderRequest, price: Decimal) -> None:
        signed = req.qty if req.side is Side.BUY else -req.qty
        sym = req.instrument.symbol
        self.positions[sym] = self.positions.get(sym, ZERO) + signed
        self.fills.append(SimFill(req.client_order_id, sym, signed, price))
        self.states[req.client_order_id] = S.FILLED
        self.open.pop(req.client_order_id, None)

    def _match(self, req: OrderRequest) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    async def submit(self, req: OrderRequest) -> str:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    async def cancel(self, coid: str) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    async def cancel_all(self) -> int:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    async def flatten_all(self) -> int:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    async def on_price(self, symbol: str, price: Decimal) -> None:
        self.last[symbol] = price
        for req in [r for r in self.open.values() if r.instrument.symbol == symbol]:
            self._match(req)

    def state(self, coid: str) -> OrderState:
        return self.states[coid]

    def is_flat(self) -> bool:
        return not self.open and all(q == 0 for q in self.positions.values())


# ----------------------------------------------------------------------- S13 kill switch
class KillSwitch:
    """Sticky, idempotent, audited kill switch.

    State lives in `state_file` (JSON {"tripped": bool, "reason": str, ...}) so it survives a restart; every
    action is appended as one JSON line to `audit_file`.
    tripped      True if the state file exists and says so.
    guard()      raise KillSwitchTripped (with the stored reason) when tripped. The OMS calls it before every submit.
    trip(reason, flatten=True) -> float
                 Under an asyncio.Lock: write the state (keep the FIRST reason if already tripped), then
                 cancel_all on every adapter concurrently, then flatten_all concurrently (if flatten), using
                 asyncio.gather(..., return_exceptions=True) so one failing broker cannot stop the others.
                 Audit {"event": "trip", "reason", "elapsed", "errors": [str(e) ...]} and await notify(message)
                 if a notifier was given. Return the elapsed seconds (clock()).
    reset(operator)
                 Only a named human can reset: empty operator -> ValueError. Write tripped False, audit it."""

    def __init__(self, adapters: list, state_file: Path, audit_file: Path | None = None,
                 notify: Callable[[str], Awaitable] | None = None, clock: Callable[[], float] = time.monotonic):
        self.adapters, self.state_file, self.notify, self.clock = adapters, Path(state_file), notify, clock
        self.audit_file = Path(audit_file) if audit_file else self.state_file.with_suffix(".audit.jsonl")
        self._lock = asyncio.Lock()

    def _state(self) -> dict:
        return json.loads(self.state_file.read_text()) if self.state_file.exists() else {"tripped": False}

    def _audit(self, **record) -> None:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_file.open("a") as f:
            f.write(json.dumps({"ts": time.time(), **record}) + "\n")

    @property
    def tripped(self) -> bool:
        return bool(self._state().get("tripped"))

    def guard(self) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    async def trip(self, reason: str, flatten: bool = True) -> float:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def reset(self, operator: str) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")
