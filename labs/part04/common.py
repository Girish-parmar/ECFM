"""Shared, complete domain types for the Part 4 labs (nothing to fill in here).

These are a trimmed copy of what `quantforge/domain/` holds after milestone M0, so every lab speaks the same
language: an `Instrument`, a canonical `OrderRequest`, the canonical order states and the platform errors.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum


# ------------------------------------------------------------------------------ errors
class TradingError(Exception):
    """Base class for all platform errors."""


class UnsupportedInstrument(TradingError):
    pass


class UnsupportedOrder(TradingError):
    """A canonical order that a broker cannot express without changing its meaning."""


class DuplicateOrder(TradingError):
    """The same client order ID was submitted twice."""


class KillSwitchTripped(TradingError):
    pass


class CircuitOpen(TradingError):
    pass


# ------------------------------------------------------------------------- instruments
class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"
    FX = "FX"
    CRYPTO = "CRYPTO"
    OPTION = "OPTION"


@dataclass(frozen=True)
class Instrument:
    """symbol: 'AAPL' (equity), 'ES' (future root), 'EURUSD' (FX pair), 'BTC' (crypto base).
    expiry: 'YYYYMM' for futures. exchange: IB listing exchange ('NASDAQ', 'CME', ...)."""
    symbol: str
    asset_class: AssetClass
    currency: str = "USD"
    exchange: str = ""
    tick: Decimal = Decimal("0.01")
    multiplier: Decimal = Decimal("1")
    expiry: str = ""


# ------------------------------------------------------------------------------ orders
class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrdType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"
    TRAIL = "TRAIL"


class TIF(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


def new_client_order_id() -> str:
    return f"qf-{uuid.uuid4().hex[:20]}"


@dataclass(frozen=True)
class OrderRequest:
    instrument: Instrument
    side: Side
    qty: Decimal
    type: OrdType
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    trail_percent: Decimal | None = None
    tif: TIF = TIF.DAY
    extended_hours: bool = False
    strategy_id: str = "manual"
    client_order_id: str = field(default_factory=new_client_order_id)


class OrderState(str, Enum):
    PENDING_NEW = "PENDING_NEW"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


S = OrderState
TRANSITIONS: dict[OrderState, set[OrderState]] = {
    S.PENDING_NEW: {S.ACCEPTED, S.REJECTED, S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.CANCELLED},
    S.ACCEPTED: {S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.CANCELLED, S.EXPIRED},
    S.PARTIALLY_FILLED: {S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.CANCELLED, S.EXPIRED},
    S.PENDING_CANCEL: {S.CANCELLED, S.FILLED, S.PARTIALLY_FILLED},
    S.FILLED: set(), S.CANCELLED: set(), S.REJECTED: set(), S.EXPIRED: set(),
}
TERMINAL = {s for s, nxt in TRANSITIONS.items() if not nxt}

# ------------------------------------------------------------------ canonical bar schema
CANONICAL_COLUMNS = ["ts", "symbol", "open", "high", "low", "close", "volume", "vwap", "trade_count",
                     "timeframe", "source", "adjusted"]
