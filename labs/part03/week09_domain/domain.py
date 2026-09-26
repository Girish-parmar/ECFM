"""Week 9 — Object-oriented domain model for trading (Part 3, S9–S12).

This becomes `quantforge/domain/` in milestone M0. Fill in every block marked "Your turn", then run:
    python -m pytest week09_domain
The property-based test checks that realized + unrealized P&L always equals cash P&L.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


class TradingError(Exception):
    """Base class for all platform errors."""


class CurrencyMismatch(TradingError):
    """Arithmetic between different currencies."""


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    FX = "FX"
    CRYPTO = "CRYPTO"


# --------------------------------------------------------------------------- value objects
@dataclass(frozen=True, slots=True)
class Money:
    """An amount in one currency. `+` and `-` only work between the same currency
    (raise CurrencyMismatch otherwise). `repr` like: Money(12.50 USD)."""
    amount: Decimal
    currency: str = "USD"

    def __add__(self, other: Money) -> Money:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def __sub__(self, other: Money) -> Money:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def __repr__(self) -> str:
        raise NotImplementedError("✍️ Your turn: see the docstring")


@dataclass(frozen=True, slots=True)
class Instrument:
    """Immutable, hashable instrument. Validation in __post_init__:
    symbol non-empty, currency 3 letters, tick > 0, multiplier > 0 (raise ValueError)."""
    symbol: str
    asset_class: AssetClass
    currency: str = "USD"
    tick: Decimal = Decimal("0.01")
    multiplier: Decimal = Decimal("1")

    def __post_init__(self):
        pass  # ✍️ Your turn: see the docstring


@dataclass(frozen=True, slots=True)
class Future(Instrument):
    expiry: str = ""                                   # e.g. "202612"


@dataclass(frozen=True, slots=True)
class Bar:
    """OHLCV bar; `ts` is the bar OPEN time and must be timezone-aware.
    Validate: low <= min(open, close), high >= max(open, close), volume >= 0 (raise ValueError)."""
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self):
        pass  # ✍️ Your turn: see the docstring


@dataclass(frozen=True, slots=True)
class Fill:
    instrument: Instrument
    qty: Decimal                                       # + buy, - sell
    price: Decimal
    fee: Decimal = Decimal("0")


# --------------------------------------------------------------------------------- entity
@dataclass(slots=True)
class Position:
    """FIFO lot accounting. `lots` holds open lots as (signed qty, price), oldest first.

    apply(fill): a fill in the same direction adds a lot; an opposite fill closes the OLDEST lots first,
    adding (fill.price − lot price) × closed qty × direction × multiplier to `realized`; any remainder
    after the position crosses zero opens a new lot at the fill price. Fees reduce `realized`.
    Raise ValueError for a fill on a different instrument. Use only Decimal + and × (no division)."""
    instrument: Instrument
    lots: deque = field(default_factory=deque)
    realized: Decimal = Decimal("0")

    @property
    def qty(self) -> Decimal:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    @property
    def avg_price(self) -> Decimal:
        """Average entry price of the open lots (display only); 0 when flat."""
        q = self.qty
        return sum((lq * lp for lq, lp in self.lots), Decimal("0")) / q if q else Decimal("0")

    def apply(self, fill: Fill) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def unrealized(self, mark: Decimal) -> Decimal:
        """P&L of the open lots at price `mark` (includes the multiplier)."""
        raise NotImplementedError("✍️ Your turn: see the docstring")
