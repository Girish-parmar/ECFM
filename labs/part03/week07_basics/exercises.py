"""Week 7 — Python basics for trading (Part 3, S1–S4).

Fill in every function marked "Your turn", then run:  python -m pytest week07_basics
Rules for this week: use only the standard library (no numpy/pandas).
"""
from __future__ import annotations

import math
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------------- S1: money
def round_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    """Round `price` to the nearest multiple of `tick` (halves round up).

    >>> round_to_tick(Decimal("101.237"), Decimal("0.05"))
    Decimal('101.25')
    """
    raise NotImplementedError("✍️ Your turn: see the docstring")


def ib_fixed_commission(shares: int, price: Decimal) -> Decimal:
    """IBKR Pro Fixed commission for US stocks: $0.005 per share, minimum $1.00,
    maximum 1% of the trade value. Return a Decimal rounded to cents (halves up).

    >>> ib_fixed_commission(100, Decimal("50"))
    Decimal('1.00')
    >>> ib_fixed_commission(10, Decimal("5"))       # 1% cap: 10 × $5 × 1% = $0.50
    Decimal('0.50')
    """
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------- S2: functions & returns
def simple_returns(prices: list[float]) -> list[float]:
    """Simple returns p[t] / p[t-1] - 1 (one fewer element than `prices`)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def log_returns(prices: list[float]) -> list[float]:
    """Log returns ln(p[t] / p[t-1])."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def max_drawdown(prices: list[float]) -> float:
    """Worst peak-to-trough fall as a negative fraction (0.0 if prices never fall).

    >>> max_drawdown([100, 120, 90, 130])
    -0.25
    """
    raise NotImplementedError("✍️ Your turn: see the docstring")


def route_message(msg: dict) -> str:
    """Turn a broker message into a log line using `match`:

    {"type": "fill", "symbol": "SPY", "qty": 100, "price": 500.25} -> "FILL SPY 100 @ 500.25"
    {"type": "cancel", "order_id": "A1"}                            -> "CANCELLED A1"
    {"type": "reject", "order_id": "A1", "reason": "margin"}        -> "REJECTED A1: margin"
    anything else                                                   -> "UNKNOWN"
    """
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------ S3: collections, dates & time zones
def to_utc(local: str, tz_name: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS' written in time zone `tz_name` and return an aware UTC datetime."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def market_open_utc(d: date) -> datetime:
    """09:30 New York time on date `d`, in UTC (14:30 in winter, 13:30 in summer)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def is_regular_hours(ts: datetime) -> bool:
    """True if the aware datetime `ts` falls Monday–Friday, 09:30 <= NY time < 16:00 (ignore holidays).
    Raise ValueError for naive datetimes."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def best_bid_ask(book: dict[str, dict[Decimal, int]]) -> tuple[Decimal | None, Decimal | None]:
    """book = {"bids": {price: size}, "asks": {price: size}}. Ignore levels with size <= 0.
    Return (best bid, best ask); None for an empty side."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def vwap(trades: list[tuple[Decimal, int]]) -> Decimal:
    """Volume-weighted average price of (price, size) trades. Raise ValueError if total size is 0."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
