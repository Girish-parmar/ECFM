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
    # >>> SOLUTION
    return (price / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick
    # <<< SOLUTION


def ib_fixed_commission(shares: int, price: Decimal) -> Decimal:
    """IBKR Pro Fixed commission for US stocks: $0.005 per share, minimum $1.00,
    maximum 1% of the trade value. Return a Decimal rounded to cents (halves up).

    >>> ib_fixed_commission(100, Decimal("50"))
    Decimal('1.00')
    >>> ib_fixed_commission(10, Decimal("5"))       # 1% cap: 10 × $5 × 1% = $0.50
    Decimal('0.50')
    """
    # >>> SOLUTION
    shares = abs(shares)
    fee = Decimal("0.005") * shares
    fee = max(fee, Decimal("1.00"))
    fee = min(fee, Decimal("0.01") * shares * price)
    return fee.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    # <<< SOLUTION


# ------------------------------------------------------------------- S2: functions & returns
def simple_returns(prices: list[float]) -> list[float]:
    """Simple returns p[t] / p[t-1] - 1 (one fewer element than `prices`)."""
    # >>> SOLUTION
    return [b / a - 1 for a, b in zip(prices[:-1], prices[1:])]
    # <<< SOLUTION


def log_returns(prices: list[float]) -> list[float]:
    """Log returns ln(p[t] / p[t-1])."""
    # >>> SOLUTION
    return [math.log(b / a) for a, b in zip(prices[:-1], prices[1:])]
    # <<< SOLUTION


def max_drawdown(prices: list[float]) -> float:
    """Worst peak-to-trough fall as a negative fraction (0.0 if prices never fall).

    >>> max_drawdown([100, 120, 90, 130])
    -0.25
    """
    # >>> SOLUTION
    peak, worst = prices[0], 0.0
    for p in prices:
        peak = max(peak, p)
        worst = min(worst, p / peak - 1)
    return worst
    # <<< SOLUTION


def route_message(msg: dict) -> str:
    """Turn a broker message into a log line using `match`:

    {"type": "fill", "symbol": "SPY", "qty": 100, "price": 500.25} -> "FILL SPY 100 @ 500.25"
    {"type": "cancel", "order_id": "A1"}                            -> "CANCELLED A1"
    {"type": "reject", "order_id": "A1", "reason": "margin"}        -> "REJECTED A1: margin"
    anything else                                                   -> "UNKNOWN"
    """
    # >>> SOLUTION
    match msg:
        case {"type": "fill", "symbol": s, "qty": q, "price": p}:
            return f"FILL {s} {q} @ {p}"
        case {"type": "cancel", "order_id": oid}:
            return f"CANCELLED {oid}"
        case {"type": "reject", "order_id": oid, "reason": why}:
            return f"REJECTED {oid}: {why}"
        case _:
            return "UNKNOWN"
    # <<< SOLUTION


# ------------------------------------------------------------ S3: collections, dates & time zones
def to_utc(local: str, tz_name: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS' written in time zone `tz_name` and return an aware UTC datetime."""
    # >>> SOLUTION
    return datetime.strptime(local, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(tz_name)).astimezone(UTC)
    # <<< SOLUTION


def market_open_utc(d: date) -> datetime:
    """09:30 New York time on date `d`, in UTC (14:30 in winter, 13:30 in summer)."""
    # >>> SOLUTION
    return datetime.combine(d, time(9, 30), tzinfo=NY).astimezone(UTC)
    # <<< SOLUTION


def is_regular_hours(ts: datetime) -> bool:
    """True if the aware datetime `ts` falls Monday–Friday, 09:30 <= NY time < 16:00 (ignore holidays).
    Raise ValueError for naive datetimes."""
    # >>> SOLUTION
    if ts.tzinfo is None:
        raise ValueError("naive datetime")
    ny = ts.astimezone(NY)
    return ny.weekday() < 5 and time(9, 30) <= ny.time() < time(16, 0)
    # <<< SOLUTION


def best_bid_ask(book: dict[str, dict[Decimal, int]]) -> tuple[Decimal | None, Decimal | None]:
    """book = {"bids": {price: size}, "asks": {price: size}}. Ignore levels with size <= 0.
    Return (best bid, best ask); None for an empty side."""
    # >>> SOLUTION
    bids = [p for p, s in book.get("bids", {}).items() if s > 0]
    asks = [p for p, s in book.get("asks", {}).items() if s > 0]
    return (max(bids) if bids else None, min(asks) if asks else None)
    # <<< SOLUTION


def vwap(trades: list[tuple[Decimal, int]]) -> Decimal:
    """Volume-weighted average price of (price, size) trades. Raise ValueError if total size is 0."""
    # >>> SOLUTION
    volume = sum(s for _, s in trades)
    if volume == 0:
        raise ValueError("no volume")
    return sum(p * s for p, s in trades) / volume
    # <<< SOLUTION
