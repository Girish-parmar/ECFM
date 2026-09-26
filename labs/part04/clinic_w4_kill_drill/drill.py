"""Clinic W4 — Kill-switch drill and heartbeat watchdog (Part 4, S13; mandatory for the M1 assessment).

Two simulated brokers ("ib-sim" and "alpaca-sim") answer with network latency. Each holds open positions and
working orders. The drill trips your week 16 KillSwitch and measures the time until BOTH accounts are flat
with no working orders; the pass mark is under 5 seconds. A watchdog trips the switch on its own when the
strategy process stops sending heartbeats (a hung strategy cannot stop itself).

Run:  python drill.py            Test:  python -m pytest clinic_w4_kill_drill   (needs week16_safety done)
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                          # noqa: E402
from common import AssetClass, Instrument, OrderRequest, OrdType, Side          # noqa: E402

sf = load("week16_safety", "safety")
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "TLT"]


class SlowBroker(sf.SimBroker):
    """A SimBroker whose cancel/flatten calls take `latency` seconds, like a round trip to a real broker (given)."""

    def __init__(self, name: str, latency: float, prices: dict[str, Decimal]):
        super().__init__(prices)
        self.name, self.latency = name, latency

    async def cancel_all(self) -> int:
        await asyncio.sleep(self.latency)
        return await super().cancel_all()

    async def flatten_all(self) -> int:
        await asyncio.sleep(self.latency)
        return await super().flatten_all()


async def load_book(broker, n_positions: int, n_orders: int) -> None:
    """Open `n_positions` positions (alternating long 10 / short 5 on SYMBOLS) with MARKET orders and leave
    `n_orders` BUY LIMIT orders resting 10% below the market on SYMBOLS[0]."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


async def run_drill(state_dir: Path, latency: float = 0.2, n_positions: int = 3, n_orders: int = 5,
                    limit: float = 5.0) -> dict:
    """Build two SlowBrokers (prices 100 + 10×index for SYMBOLS), load each book, trip a KillSwitch over both
    (state file state_dir/'killswitch.json'), and return
    {"elapsed": seconds from trip start to return (time.perf_counter), "flat": all brokers flat,
     "open_orders": total resting orders left, "tripped": switch state, "passed": flat and elapsed < limit}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


class Watchdog:
    """Heartbeat monitor. The strategy calls beat(); `stale` is True when no beat arrived for more than
    `timeout` seconds (clock() time). A fresh watchdog counts as having just been beaten."""

    def __init__(self, timeout: float, clock: Callable[[], float] = time.monotonic):
        self.timeout, self.clock = timeout, clock
        self.last = clock()

    def beat(self) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")

    @property
    def stale(self) -> bool:
        raise NotImplementedError("✍️ Your turn: see the docstring")


async def watch(dog: Watchdog, switch, poll: float = 1.0, max_polls: int | None = None,
                sleep: Callable[[float], Awaitable] = asyncio.sleep) -> bool:
    """Check the watchdog every `poll` seconds. When it is stale, trip the switch with reason 'heartbeat lost'
    and return True. Return False after `max_polls` checks without a trip (None = forever)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        report = asyncio.run(run_drill(Path(d)))
    print(report)
    print("DRILL PASSED" if report["passed"] else "DRILL FAILED")
