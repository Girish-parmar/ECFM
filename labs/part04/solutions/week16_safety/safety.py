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
    # >>> SOLUTION
    return tick * multiplier
    # <<< SOLUTION


def futures_pnl(entry: Decimal, exit: Decimal, qty: Decimal, multiplier: Decimal) -> Decimal:
    """Signed P&L: qty > 0 long, qty < 0 short."""
    # >>> SOLUTION
    return (exit - entry) * qty * multiplier
    # <<< SOLUTION


def pip_value_usd(pair: str, units: Decimal, usd_per_quote: Decimal) -> Decimal:
    """USD value of one pip for `units` of the BASE currency. Pip = 0.01 for JPY-quoted pairs, else 0.0001.
    The pip is worth pip × units in the QUOTE currency; convert with usd_per_quote (1 for XXXUSD pairs)."""
    # >>> SOLUTION
    pip = Decimal("0.01") if pair.upper().endswith("JPY") else Decimal("0.0001")
    return pip * units * usd_per_quote
    # <<< SOLUTION


def third_friday(year: int, month: int) -> date:
    """Third Friday of the month (quarterly equity-index futures such as ES/MES expire then)."""
    # >>> SOLUTION
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(weeks=2)
    # <<< SOLUTION


def roll_date(expiry: str, business_days_before: int = 8) -> date:
    """Roll date for an equity-index future with expiry 'YYYYMM': go back `business_days_before` weekdays
    (Mon–Fri; ignore holidays in this exercise) from its third Friday."""
    # >>> SOLUTION
    d = third_friday(int(expiry[:4]), int(expiry[4:6]))
    left = business_days_before
    while left:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            left -= 1
    return d
    # <<< SOLUTION


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
    # >>> SOLUTION
    sym = req.instrument.symbol
    if ctx.kill_switch_tripped:
        return "kill switch tripped"
    if sym not in ctx.tradable:
        return "not tradable"
    if not ctx.session_open and not req.extended_hours:
        return "session closed"
    last = ctx.last.get(sym)
    if last is None:
        return "no last price"
    if req.limit_price is not None and abs(req.limit_price - last) / last * 100 > ctx.fat_finger_pct:
        return "fat finger"
    if req.qty > ctx.max_qty:
        return "qty above max"
    if req.side is Side.BUY and req.qty * (req.limit_price or last) > ctx.buying_power:
        return "insufficient buying power"
    if req.side is Side.SELL and ctx.position.get(sym, ZERO) - req.qty < 0 and sym not in ctx.shortable:
        return "not shortable"
    return None
    # <<< SOLUTION


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
        # >>> SOLUTION
        last = self.last.get(req.instrument.symbol)
        if last is None:
            return
        buy = req.side is Side.BUY
        if req.type is OrdType.MARKET:
            self._fill(req, last)
        elif req.type is OrdType.LIMIT and (last <= req.limit_price if buy else last >= req.limit_price):
            self._fill(req, req.limit_price)
        elif req.type is OrdType.STOP and (last >= req.stop_price if buy else last <= req.stop_price):
            self._fill(req, last)
        # <<< SOLUTION

    async def submit(self, req: OrderRequest) -> str:
        # >>> SOLUTION
        coid = req.client_order_id
        if coid in self.states:
            raise DuplicateOrder(coid)
        if req.type not in (OrdType.MARKET, OrdType.LIMIT, OrdType.STOP):
            raise UnsupportedOrder(req.type)
        self.states[coid] = S.ACCEPTED
        self.open[coid] = req
        self._match(req)
        return coid
        # <<< SOLUTION

    async def cancel(self, coid: str) -> None:
        # >>> SOLUTION
        if coid in self.open:
            del self.open[coid]
            self.states[coid] = S.CANCELLED
        # <<< SOLUTION

    async def cancel_all(self) -> int:
        # >>> SOLUTION
        ids = list(self.open)
        for coid in ids:
            await self.cancel(coid)
        return len(ids)
        # <<< SOLUTION

    async def flatten_all(self) -> int:
        # >>> SOLUTION
        sent = 0
        for sym, qty in list(self.positions.items()):
            if qty == 0:
                continue
            self._n += 1
            req = OrderRequest(Instrument(sym, AssetClass.EQUITY), Side.SELL if qty > 0 else Side.BUY, abs(qty),
                               OrdType.MARKET, client_order_id=f"flatten-{self._n}")
            await self.submit(req)
            sent += 1
        return sent
        # <<< SOLUTION

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
        # >>> SOLUTION
        st = self._state()
        if st.get("tripped"):
            raise KillSwitchTripped(st.get("reason", ""))
        # <<< SOLUTION

    async def trip(self, reason: str, flatten: bool = True) -> float:
        # >>> SOLUTION
        async with self._lock:
            t0 = self.clock()
            st = self._state()
            if not st.get("tripped"):
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                self.state_file.write_text(json.dumps({"tripped": True, "reason": reason, "ts": time.time()}))
            results = await asyncio.gather(*(a.cancel_all() for a in self.adapters), return_exceptions=True)
            if flatten:
                results += await asyncio.gather(*(a.flatten_all() for a in self.adapters), return_exceptions=True)
            errors = [str(r) for r in results if isinstance(r, BaseException)]
            elapsed = self.clock() - t0
            self._audit(event="trip", reason=reason, elapsed=elapsed, errors=errors)
            if self.notify:
                await self.notify(f"KILL SWITCH: {reason} ({elapsed:.2f}s, {len(errors)} errors)")
            return elapsed
        # <<< SOLUTION

    def reset(self, operator: str) -> None:
        # >>> SOLUTION
        if not operator or not operator.strip():
            raise ValueError("reset needs the name of the operator")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({"tripped": False, "reset_by": operator, "ts": time.time()}))
        self._audit(event="reset", operator=operator)
        # <<< SOLUTION
