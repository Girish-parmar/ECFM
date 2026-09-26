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
    raise NotImplementedError("✍️ Your turn: see the docstring")


async def with_backoff(connect: Callable[[], Awaitable], *, max_tries: int | None = None, base: float = 1.0,
                       cap: float = 60.0, rng: random.Random | None = None,
                       sleep: Callable[[float], Awaitable] = asyncio.sleep,
                       retry_on: tuple[type[BaseException], ...] = (ConnectionError, OSError, TimeoutError)):
    """Await `connect()` until it succeeds and return its result. On an exception in `retry_on`, sleep for the
    next delay (same formula as backoff_delays, same rng) and try again; after `max_tries` failed attempts
    re-raise the last error. Any other exception propagates immediately."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


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
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def call(self, fn: Callable, *args, **kwargs):
        raise NotImplementedError("✍️ Your turn: see the docstring")


def classify_ib_code(code: int) -> str:
    """What the connection manager does with an IB error/system code:
    1100, 502, 504 -> 'reconnect';  1101 -> 'resubscribe' (restored, data lost);
    1102, 2104, 2106, 2158 -> 'ok' (restored with data / farm OK);  326 -> 'fatal' (client id in use);
    162 -> 'retry_later' (historical data error, e.g. pacing);  anything else -> 'log'."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ---------------------------------------------------------------- S10 order mapping
def round_limit_to_tick(price: Decimal, tick: Decimal, side: Side) -> Decimal:
    """Round a LIMIT price to the tick grid in the conservative direction: BUY rounds DOWN, SELL rounds UP
    (so rounding never makes the order pay more than intended). Result is an exact multiple of `tick`."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def alpaca_price_ok(price: Decimal) -> bool:
    """Alpaca stock prices: at most 2 decimals when price >= 1, at most 4 decimals below 1; must be > 0."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


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
    raise NotImplementedError("✍️ Your turn: see the docstring")


def to_alpaca_request(req: OrderRequest):
    """Canonical order -> alpaca-py request object (MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
    StopLimitOrderRequest, TrailingStopOrderRequest(trail_percent=...)), with client_order_id set.
    Symbol: equity -> symbol, crypto -> 'BTC/USD'. Raise UnsupportedOrder when the meaning would change:
      * missing price / qty <= 0 (use _check_prices)      * FX or futures (not tradable on Alpaca)
      * extended_hours unless LIMIT with DAY               * crypto with a TIF other than GTC or IOC
      * a limit/stop price that fails alpaca_price_ok (never round silently)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------ S11 order state tracking
def map_ib_status(status: str, filled: float = 0.0) -> OrderState:
    """IB orderStatus.status -> canonical state (table in the lesson plan). 'PreSubmitted'/'Submitted' become
    PARTIALLY_FILLED when filled > 0. Unknown status -> ValueError."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def map_alpaca_status(status: str) -> OrderState:
    """Alpaca order status OR trade-update event name -> canonical state:
    pending_new, accepted -> PENDING_NEW · new -> ACCEPTED · partially_filled, partial_fill -> PARTIALLY_FILLED ·
    filled, fill -> FILLED · pending_cancel -> PENDING_CANCEL · canceled -> CANCELLED · rejected -> REJECTED ·
    expired, done_for_day -> EXPIRED. Unknown -> ValueError."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


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
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def _move(self, t: TrackedOrder, new: OrderState) -> bool:
        if new in TRANSITIONS[t.state]:
            t.state = new
            t.history.append(new)
            return True
        return False

    def on_status(self, coid: str, state: OrderState) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def on_fill(self, coid: str, exec_id: str, qty: Decimal, price: Decimal) -> bool:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def position(self, symbol: str) -> Decimal:
        raise NotImplementedError("✍️ Your turn: see the docstring")


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
    raise NotImplementedError("✍️ Your turn: see the docstring")
