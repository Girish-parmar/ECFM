"""Week 15 — Connections & orders (Part 4, S9–S12): back-off, circuit breaker, IB system codes,
cross-broker order mapping, order state tracking and reconciliation.

The mappers build REAL `ib_async` and `alpaca-py` order objects, offline (nothing is sent).
Fill in every block marked "Your turn", then run:  python -m pytest week15_orders
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (LimitOrderRequest, MarketOrderRequest, StopLimitOrderRequest, StopOrderRequest,
                                     TrailingStopOrderRequest)
from ib_async import LimitOrder, MarketOrder, Order, StopLimitOrder, StopOrder

from common import (TERMINAL, TIF, TRANSITIONS, AssetClass, CircuitOpen, DuplicateOrder, OrderRequest, OrderState,
                    OrdType, Side, UnsupportedOrder)

S = OrderState


# ----------------------------------------------------------------- S9 connection management
def backoff_delays(attempts: int, base: float = 1.0, cap: float = 60.0, rng: random.Random | None = None) -> list[float]:
    """Delays before retry 1..attempts: min(cap, base × 2**k) × jitter for k = 1..attempts, where
    jitter = rng.uniform(0.5, 1.0) (a fresh draw per retry). "Equal jitter" keeps retries from many
    clients from hitting the gateway at the same instant."""
    # >>> SOLUTION
    rng = rng or random.Random()
    return [min(cap, base * 2 ** k) * rng.uniform(0.5, 1.0) for k in range(1, attempts + 1)]
    # <<< SOLUTION


async def with_backoff(connect: Callable[[], Awaitable], *, max_tries: int | None = None, base: float = 1.0,
                       cap: float = 60.0, rng: random.Random | None = None,
                       sleep: Callable[[float], Awaitable] = asyncio.sleep,
                       retry_on: tuple[type[BaseException], ...] = (ConnectionError, OSError, TimeoutError)):
    """Await `connect()` until it succeeds and return its result. On an exception in `retry_on`, sleep for the
    next delay (same formula as backoff_delays, same rng) and try again; after `max_tries` failed attempts
    re-raise the last error. Any other exception propagates immediately."""
    # >>> SOLUTION
    rng = rng or random.Random()
    attempt = 0
    while True:
        try:
            return await connect()
        except retry_on:
            attempt += 1
            if max_tries is not None and attempt >= max_tries:
                raise
            await sleep(min(cap, base * 2 ** attempt) * rng.uniform(0.5, 1.0))
    # <<< SOLUTION


class CircuitBreaker:
    """Protect a flaky REST API. States: 'closed' (calls pass), 'open' (calls fail fast with CircuitOpen),
    'half_open' (reset_after seconds after opening: ONE trial call is let through).
    `threshold` consecutive failures open the circuit; a failure in half_open re-opens it at once; any success
    closes it and resets the failure count. Time comes from `clock()`."""

    def __init__(self, threshold: int, reset_after: float, clock: Callable[[], float]):
        self.threshold, self.reset_after, self.clock = threshold, reset_after, clock
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        # >>> SOLUTION
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.reset_after else "open"
        # <<< SOLUTION

    def call(self, fn: Callable, *args, **kwargs):
        # >>> SOLUTION
        state = self.state
        if state == "open":
            raise CircuitOpen(f"circuit open; retry in {self.reset_after - (self.clock() - self.opened_at):.1f}s")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.failures += 1
            if state == "half_open" or self.failures >= self.threshold:
                self.opened_at = self.clock()
            raise
        self.failures, self.opened_at = 0, None
        return result
        # <<< SOLUTION


def classify_ib_code(code: int) -> str:
    """What the connection manager does with an IB error/system code:
    1100, 502, 504 -> 'reconnect';  1101 -> 'resubscribe' (restored, data lost);
    1102, 2104, 2106, 2158 -> 'ok' (restored with data / farm OK);  326 -> 'fatal' (client id in use);
    162 -> 'retry_later' (historical data error, e.g. pacing);  anything else -> 'log'."""
    # >>> SOLUTION
    match code:
        case 1100 | 502 | 504:
            return "reconnect"
        case 1101:
            return "resubscribe"
        case 1102 | 2104 | 2106 | 2158:
            return "ok"
        case 326:
            return "fatal"
        case 162:
            return "retry_later"
    return "log"
    # <<< SOLUTION


# ---------------------------------------------------------------- S10 order mapping
def round_limit_to_tick(price: Decimal, tick: Decimal, side: Side) -> Decimal:
    """Round a LIMIT price to the tick grid in the conservative direction: BUY rounds DOWN, SELL rounds UP
    (so rounding never makes the order pay more than intended). Result is an exact multiple of `tick`."""
    # >>> SOLUTION
    steps = price / tick
    n = steps.to_integral_value(rounding="ROUND_FLOOR" if side is Side.BUY else "ROUND_CEILING")
    return n * tick
    # <<< SOLUTION


def alpaca_price_ok(price: Decimal) -> bool:
    """Alpaca stock prices: at most 2 decimals when price >= 1, at most 4 decimals below 1; must be > 0."""
    # >>> SOLUTION
    if price <= 0:
        return False
    places = -price.normalize().as_tuple().exponent
    return places <= (2 if price >= 1 else 4)
    # <<< SOLUTION


def _check_prices(req: OrderRequest) -> None:
    need = {OrdType.LIMIT: ("limit_price",), OrdType.STOP: ("stop_price",),
            OrdType.STOP_LIMIT: ("limit_price", "stop_price"), OrdType.TRAIL: ("trail_percent",)}
    if req.qty <= 0:
        raise UnsupportedOrder("qty must be positive")
    for f in need.get(req.type, ()):
        if getattr(req, f) is None:
            raise UnsupportedOrder(f"{req.type.value} needs {f}")


def to_ib_order(req: OrderRequest) -> Order:
    """Canonical order -> ib_async order (see the mapping table in the lesson plan):
    MARKET MarketOrder · LIMIT LimitOrder · STOP StopOrder · STOP_LIMIT StopLimitOrder(action, qty, lmt, stop) ·
    TRAIL Order(orderType='TRAIL', trailingPercent=...). Quantities and prices as float.
    Always set tif=req.tif.value, orderRef=req.client_order_id, outsideRth=req.extended_hours.
    Raise UnsupportedOrder (via _check_prices) for a missing price or qty <= 0."""
    # >>> SOLUTION
    _check_prices(req)
    action, qty = req.side.value, float(req.qty)
    kw = dict(tif=req.tif.value, orderRef=req.client_order_id, outsideRth=req.extended_hours)
    match req.type:
        case OrdType.MARKET:
            return MarketOrder(action, qty, **kw)
        case OrdType.LIMIT:
            return LimitOrder(action, qty, float(req.limit_price), **kw)
        case OrdType.STOP:
            return StopOrder(action, qty, float(req.stop_price), **kw)
        case OrdType.STOP_LIMIT:
            return StopLimitOrder(action, qty, float(req.limit_price), float(req.stop_price), **kw)
        case OrdType.TRAIL:
            return Order(action=action, totalQuantity=qty, orderType="TRAIL",
                         trailingPercent=float(req.trail_percent), **kw)
    raise UnsupportedOrder(req.type)
    # <<< SOLUTION


def to_alpaca_request(req: OrderRequest):
    """Canonical order -> alpaca-py request object (MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
    StopLimitOrderRequest, TrailingStopOrderRequest(trail_percent=...)), with client_order_id set.
    Symbol: equity -> symbol, crypto -> 'BTC/USD'. Raise UnsupportedOrder when the meaning would change:
      * missing price / qty <= 0 (use _check_prices)      * FX or futures (not tradable on Alpaca)
      * extended_hours unless LIMIT with DAY               * crypto with a TIF other than GTC or IOC
      * a limit/stop price that fails alpaca_price_ok (never round silently)."""
    # >>> SOLUTION
    _check_prices(req)
    inst = req.instrument
    if inst.asset_class not in (AssetClass.EQUITY, AssetClass.CRYPTO):
        raise UnsupportedOrder(f"{inst.asset_class.value} is not tradable on Alpaca")
    if req.extended_hours and not (req.type is OrdType.LIMIT and req.tif is TIF.DAY):
        raise UnsupportedOrder("extended hours needs a LIMIT DAY order")
    if inst.asset_class is AssetClass.CRYPTO and req.tif not in (TIF.GTC, TIF.IOC):
        raise UnsupportedOrder("crypto orders must be GTC or IOC")
    for p in (req.limit_price, req.stop_price):
        if p is not None and not alpaca_price_ok(p):
            raise UnsupportedOrder(f"invalid price increment {p}")
    symbol = inst.symbol if inst.asset_class is AssetClass.EQUITY else f"{inst.symbol}/{inst.currency}"
    common = dict(symbol=symbol, qty=float(req.qty), side=OrderSide(req.side.value.lower()),
                  time_in_force=TimeInForce(req.tif.value.lower()), client_order_id=req.client_order_id)
    match req.type:
        case OrdType.MARKET:
            return MarketOrderRequest(**common)
        case OrdType.LIMIT:
            return LimitOrderRequest(**common, limit_price=float(req.limit_price), extended_hours=req.extended_hours)
        case OrdType.STOP:
            return StopOrderRequest(**common, stop_price=float(req.stop_price))
        case OrdType.STOP_LIMIT:
            return StopLimitOrderRequest(**common, limit_price=float(req.limit_price), stop_price=float(req.stop_price))
        case OrdType.TRAIL:
            return TrailingStopOrderRequest(**common, trail_percent=float(req.trail_percent))
    raise UnsupportedOrder(req.type)
    # <<< SOLUTION


# ------------------------------------------------------------ S11 order state tracking
def map_ib_status(status: str, filled: float = 0.0) -> OrderState:
    """IB orderStatus.status -> canonical state (table in the lesson plan). 'PreSubmitted'/'Submitted' become
    PARTIALLY_FILLED when filled > 0. Unknown status -> ValueError."""
    # >>> SOLUTION
    table = {"PendingSubmit": S.PENDING_NEW, "ApiPending": S.PENDING_NEW, "PreSubmitted": S.ACCEPTED,
             "Submitted": S.ACCEPTED, "Filled": S.FILLED, "PendingCancel": S.PENDING_CANCEL,
             "Cancelled": S.CANCELLED, "ApiCancelled": S.CANCELLED, "Inactive": S.REJECTED}
    if status not in table:
        raise ValueError(f"unknown IB status {status!r}")
    state = table[status]
    return S.PARTIALLY_FILLED if state is S.ACCEPTED and filled > 0 else state
    # <<< SOLUTION


def map_alpaca_status(status: str) -> OrderState:
    """Alpaca order status OR trade-update event name -> canonical state:
    pending_new, accepted -> PENDING_NEW · new -> ACCEPTED · partially_filled, partial_fill -> PARTIALLY_FILLED ·
    filled, fill -> FILLED · pending_cancel -> PENDING_CANCEL · canceled -> CANCELLED · rejected -> REJECTED ·
    expired, done_for_day -> EXPIRED. Unknown -> ValueError."""
    # >>> SOLUTION
    table = {"pending_new": S.PENDING_NEW, "accepted": S.PENDING_NEW, "new": S.ACCEPTED,
             "partially_filled": S.PARTIALLY_FILLED, "partial_fill": S.PARTIALLY_FILLED,
             "filled": S.FILLED, "fill": S.FILLED, "pending_cancel": S.PENDING_CANCEL,
             "canceled": S.CANCELLED, "rejected": S.REJECTED, "expired": S.EXPIRED, "done_for_day": S.EXPIRED}
    if status not in table:
        raise ValueError(f"unknown Alpaca status {status!r}")
    return table[status]
    # <<< SOLUTION


@dataclass
class TrackedOrder:
    req: OrderRequest
    state: OrderState = S.PENDING_NEW
    filled: Decimal = Decimal("0")
    notional: Decimal = Decimal("0")          # sum of fill qty × price
    history: list[OrderState] = field(default_factory=lambda: [S.PENDING_NEW])

    @property
    def avg_price(self) -> Decimal | None:
        return self.notional / self.filled if self.filled else None


class OrderTracker:
    """Keeps canonical order state from out-of-order, duplicated broker events. Never raises on bad events:
    it records an anomaly (tuple) in `self.anomalies` instead, so the OMS can trigger reconciliation.

    track(req)                 register; DuplicateOrder if the client order ID is known.
    on_status(coid, state)     apply a status. Unknown coid -> ('unknown order', coid). Same state again
                               (except PARTIALLY_FILLED) -> ignore silently. Illegal per TRANSITIONS ->
                               ('illegal transition', coid, 'OLD->NEW') and keep the old state.
    on_fill(coid, exec_id, qty, price)
                               FILLS ARE THE TRUTH. A repeated exec_id is ignored (return False). Otherwise add
                               to filled/notional, return True, and move the state to FILLED (filled >= order qty)
                               or PARTIALLY_FILLED. Fills after a FILLED status are normal (IB often sends the
                               status first). A fill on any OTHER terminal order (e.g. after CANCELLED) records
                               ('fill after terminal', coid, state) but still counts. Filling more than the order
                               qty records ('overfill', coid).
    position(symbol)           signed sum of filled quantities (+ BUY, − SELL) over all orders on that symbol."""

    def __init__(self):
        self.orders: dict[str, TrackedOrder] = {}
        self.exec_ids: set[str] = set()
        self.anomalies: list[tuple] = []

    def track(self, req: OrderRequest) -> TrackedOrder:
        # >>> SOLUTION
        if req.client_order_id in self.orders:
            raise DuplicateOrder(req.client_order_id)
        self.orders[req.client_order_id] = t = TrackedOrder(req)
        return t
        # <<< SOLUTION

    def _move(self, t: TrackedOrder, new: OrderState) -> bool:
        if new in TRANSITIONS[t.state]:
            t.state = new
            t.history.append(new)
            return True
        return False

    def on_status(self, coid: str, state: OrderState) -> None:
        # >>> SOLUTION
        t = self.orders.get(coid)
        if t is None:
            self.anomalies.append(("unknown order", coid))
            return
        if state == t.state and state is not S.PARTIALLY_FILLED:
            return
        if not self._move(t, state):
            self.anomalies.append(("illegal transition", coid, f"{t.state.value}->{state.value}"))
        # <<< SOLUTION

    def on_fill(self, coid: str, exec_id: str, qty: Decimal, price: Decimal) -> bool:
        # >>> SOLUTION
        if exec_id in self.exec_ids:
            return False
        t = self.orders.get(coid)
        if t is None:
            self.anomalies.append(("unknown order", coid))
            return False
        self.exec_ids.add(exec_id)
        t.filled += qty
        t.notional += qty * price
        if t.filled > t.req.qty:
            self.anomalies.append(("overfill", coid))
        if t.state is S.FILLED:
            return True
        new = S.FILLED if t.filled >= t.req.qty else S.PARTIALLY_FILLED
        if t.state in TERMINAL or not self._move(t, new):
            self.anomalies.append(("fill after terminal", coid, t.state.value))
        return True
        # <<< SOLUTION

    def position(self, symbol: str) -> Decimal:
        # >>> SOLUTION
        return sum((t.filled if t.req.side is Side.BUY else -t.filled
                    for t in self.orders.values() if t.req.instrument.symbol == symbol), Decimal("0"))
        # <<< SOLUTION


# ----------------------------------------------------------------- S12 reconciliation
@dataclass
class ReconReport:
    only_internal: list[str]
    only_broker: list[str]
    qty_mismatch: list[tuple[str, Decimal, Decimal]]
    position_mismatch: list[tuple[str, Decimal, Decimal]]

    @property
    def ok(self) -> bool:
        return not (self.only_internal or self.only_broker or self.qty_mismatch or self.position_mismatch)


def reconcile(our_orders: dict[str, Decimal], broker_orders: dict[str, Decimal],
              our_pos: dict[str, Decimal], broker_pos: dict[str, Decimal]) -> ReconReport:
    """Compare OPEN orders (client order ID -> remaining qty) and positions (symbol -> signed qty).
    A symbol missing on one side counts as 0 (so a flat position may be absent). Every list sorted.
    Tuples are (key, ours, broker's)."""
    # >>> SOLUTION
    zero = Decimal("0")
    return ReconReport(
        only_internal=sorted(our_orders.keys() - broker_orders.keys()),
        only_broker=sorted(broker_orders.keys() - our_orders.keys()),
        qty_mismatch=sorted((k, our_orders[k], broker_orders[k]) for k in our_orders.keys() & broker_orders.keys()
                            if our_orders[k] != broker_orders[k]),
        position_mismatch=sorted((s, our_pos.get(s, zero), broker_pos.get(s, zero))
                                 for s in our_pos.keys() | broker_pos.keys()
                                 if our_pos.get(s, zero) != broker_pos.get(s, zero)),
    )
    # <<< SOLUTION
