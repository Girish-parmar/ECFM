"""Clocks: strategies ask the clock for 'now', so the same code runs live and in backtests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class LiveClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class SimClock:
    """Backtest clock: time only moves when the engine advances it (never backwards)."""

    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("SimClock needs a timezone-aware start")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance_to(self, t: datetime) -> None:
        if t < self._now:
            raise ValueError("time cannot go backwards")
        self._now = t

    def advance(self, dt: timedelta) -> None:
        self.advance_to(self._now + dt)
