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
        # >>> SOLUTION
        if not isinstance(other, Money):
            return NotImplemented
        if other.currency != self.currency:
            raise CurrencyMismatch(f"{self.currency} + {other.currency}")
        return Money(self.amount + other.amount, self.currency)
        # <<< SOLUTION

    def __sub__(self, other: Money) -> Money:
        # >>> SOLUTION
        if not isinstance(other, Money):
            return NotImplemented
        if other.currency != self.currency:
            raise CurrencyMismatch(f"{self.currency} - {other.currency}")
        return Money(self.amount - other.amount, self.currency)
        # <<< SOLUTION

    def __repr__(self) -> str:
        # >>> SOLUTION
        return f"Money({self.amount:.2f} {self.currency})"
        # <<< SOLUTION


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
        # >>> SOLUTION (pass)
        if not self.symbol:
            raise ValueError("empty symbol")
        if len(self.currency) != 3 or not self.currency.isalpha():
            raise ValueError("currency must be a 3-letter code")
        if self.tick <= 0 or self.multiplier <= 0:
            raise ValueError("tick and multiplier must be positive")
        # <<< SOLUTION


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
        # >>> SOLUTION (pass)
        if self.ts.tzinfo is None:
            raise ValueError("bar timestamp must be timezone-aware")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("inconsistent OHLC")
        if self.volume < 0:
            raise ValueError("negative volume")
        # <<< SOLUTION


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
        # >>> SOLUTION
        return sum((q for q, _ in self.lots), Decimal("0"))
        # <<< SOLUTION

    @property
    def avg_price(self) -> Decimal:
        """Average entry price of the open lots (display only); 0 when flat."""
        q = self.qty
        return sum((lq * lp for lq, lp in self.lots), Decimal("0")) / q if q else Decimal("0")

    def apply(self, fill: Fill) -> None:
        # >>> SOLUTION
        if fill.instrument != self.instrument:
            raise ValueError("fill for a different instrument")
        m = self.instrument.multiplier
        self.realized -= fill.fee
        q = fill.qty
        while q != 0 and self.lots and (self.lots[0][0] > 0) != (q > 0):
            lot_q, lot_p = self.lots[0]
            close = min(abs(q), abs(lot_q))
            s = 1 if lot_q > 0 else -1
            self.realized += (fill.price - lot_p) * close * s * m
            lot_q -= s * close
            q += s * close
            if lot_q == 0:
                self.lots.popleft()
            else:
                self.lots[0] = (lot_q, lot_p)
        if q != 0:
            self.lots.append((q, fill.price))
        # <<< SOLUTION

    def unrealized(self, mark: Decimal) -> Decimal:
        """P&L of the open lots at price `mark` (includes the multiplier)."""
        # >>> SOLUTION
        return sum(((mark - p) * q for q, p in self.lots), Decimal("0")) * self.instrument.multiplier
        # <<< SOLUTION
