"""Week 10 — Design patterns used throughout the platform (Part 3, S13–S16).

Fill in every block marked "Your turn", then run:  python -m pytest week10_patterns
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass


# ------------------------------------------------------------------ Observer: event bus
class EventBus:
    """Publish/subscribe by event TYPE. Handlers may be plain functions or coroutine functions;
    `publish` calls every handler subscribed to type(event), in subscription order, awaiting async ones.
    `unsubscribe` removes a handler (no error if it was not subscribed)."""

    def __init__(self):
        self._subs: dict[type, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable) -> None:
        # >>> SOLUTION
        self._subs[event_type].append(handler)
        # <<< SOLUTION

    def unsubscribe(self, event_type: type, handler: Callable) -> None:
        # >>> SOLUTION
        if handler in self._subs[event_type]:
            self._subs[event_type].remove(handler)
        # <<< SOLUTION

    async def publish(self, event) -> None:
        # >>> SOLUTION
        for h in list(self._subs[type(event)]):
            res = h(event)
            if asyncio.iscoroutine(res):
                await res
        # <<< SOLUTION


# ----------------------------------------------------------- State: order lifecycle
class OrderStateError(Exception):
    pass


class OrderStateMachine:
    """Order lifecycle as a transition table. `to(new)` moves to `new` or raises OrderStateError.
    `is_terminal` is True for FILLED, CANCELLED, REJECTED, EXPIRED. `history` lists every state visited."""
    TRANSITIONS = {
        "PENDING_NEW": {"ACCEPTED", "REJECTED"},
        "ACCEPTED": {"PARTIALLY_FILLED", "FILLED", "PENDING_CANCEL", "EXPIRED"},
        "PARTIALLY_FILLED": {"PARTIALLY_FILLED", "FILLED", "PENDING_CANCEL"},
        "PENDING_CANCEL": {"CANCELLED", "FILLED"},
        "FILLED": set(), "CANCELLED": set(), "REJECTED": set(), "EXPIRED": set(),
    }

    def __init__(self):
        self.state = "PENDING_NEW"
        self.history = ["PENDING_NEW"]

    def to(self, new: str) -> None:
        # >>> SOLUTION
        if new not in self.TRANSITIONS[self.state]:
            raise OrderStateError(f"{self.state} -> {new} not allowed")
        self.state = new
        self.history.append(new)
        # <<< SOLUTION

    @property
    def is_terminal(self) -> bool:
        # >>> SOLUTION
        return not self.TRANSITIONS[self.state]
        # <<< SOLUTION


# ------------------------------------------------------------- Factory + Registry
class Registry:
    """Name -> class registry used to build objects from configuration.

    `register(name)` is a decorator; registering the same name twice raises KeyError.
    `create(name, **params)` builds an instance; unknown names raise KeyError listing the known names."""

    def __init__(self):
        self._items: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        # >>> SOLUTION
        def deco(cls: type) -> type:
            if name in self._items:
                raise KeyError(f"{name!r} already registered")
            self._items[name] = cls
            return cls
        return deco
        # <<< SOLUTION

    def create(self, name: str, **params):
        # >>> SOLUTION
        if name not in self._items:
            raise KeyError(f"unknown {name!r}; known: {sorted(self._items)}")
        return self._items[name](**params)
        # <<< SOLUTION

    @property
    def names(self) -> list[str]:
        return sorted(self._items)


# ------------------------------------------------ Chain of Responsibility: risk checks
@dataclass(frozen=True)
class Decision:
    approved: bool
    reason: str = ""


class RiskRule(ABC):
    @abstractmethod
    def check(self, order: dict, ctx: dict) -> Decision: ...


class MaxNotional(RiskRule):
    """Reject when |qty × price| > limit. Reason text: 'notional 80000 > 50000' (integers)."""

    def __init__(self, limit: float):
        self.limit = limit

    def check(self, order, ctx):
        # >>> SOLUTION
        n = abs(order["qty"] * order["price"])
        return Decision(n <= self.limit, f"notional {n:.0f} > {self.limit:.0f}")
        # <<< SOLUTION


class MaxPosition(RiskRule):
    """Reject when |current position + order qty| > limit (ctx["position"] is the current signed position)."""

    def __init__(self, limit: float):
        self.limit = limit

    def check(self, order, ctx):
        # >>> SOLUTION
        after = ctx.get("position", 0) + order["qty"]
        return Decision(abs(after) <= self.limit, f"position {after:.0f} beyond {self.limit:.0f}")
        # <<< SOLUTION


class RiskEngine:
    """Runs rules in order. Returns Decision(True) if all approve; otherwise the FIRST rejection,
    with the reason prefixed by the rule's class name: 'MaxNotional: notional 80000 > 50000'."""

    def __init__(self, rules: list[RiskRule]):
        self.rules = rules

    def check(self, order: dict, ctx: dict) -> Decision:
        # >>> SOLUTION
        for rule in self.rules:
            d = rule.check(order, ctx)
            if not d.approved:
                return Decision(False, f"{type(rule).__name__}: {d.reason}")
        return Decision(True)
        # <<< SOLUTION


# ------------------------------------------------------- Specification: entry rules
class Condition:
    """A named rule over a data context (e.g. a dict of values) returning bool.
    Combine with & (and), | (or), ~ (not); names combine as '(a AND b)', '(a OR b)', 'NOT a'."""

    def __init__(self, fn: Callable[[dict], bool], name: str):
        self.fn, self.name = fn, name

    def __call__(self, ctx: dict) -> bool:
        return bool(self.fn(ctx))

    def __and__(self, other: Condition) -> Condition:
        # >>> SOLUTION
        return Condition(lambda c: self(c) and other(c), f"({self.name} AND {other.name})")
        # <<< SOLUTION

    def __or__(self, other: Condition) -> Condition:
        # >>> SOLUTION
        return Condition(lambda c: self(c) or other(c), f"({self.name} OR {other.name})")
        # <<< SOLUTION

    def __invert__(self) -> Condition:
        # >>> SOLUTION
        return Condition(lambda c: not self(c), f"NOT {self.name}")
        # <<< SOLUTION


# ------------------------------------------------------------ Strategy: fill models
class FillModel(ABC):
    """Decide whether (and at what price) an order fills on the NEXT bar {open, high, low, close}."""

    @abstractmethod
    def fill_price(self, side: str, limit: float | None, bar: dict) -> float | None: ...


class NextOpenFill(FillModel):
    """Market orders: fill at the next bar's open (ignore `limit`)."""

    def fill_price(self, side, limit, bar):
        # >>> SOLUTION
        return bar["open"]
        # <<< SOLUTION


class LimitThroughFill(FillModel):
    """Limit orders fill only if price trades THROUGH the limit (touching is not enough):
    BUY fills if low < limit, at min(open, limit); SELL fills if high > limit, at max(open, limit).
    Return None when there is no fill."""

    def fill_price(self, side, limit, bar):
        # >>> SOLUTION
        if side == "BUY":
            return min(bar["open"], limit) if bar["low"] < limit else None
        return max(bar["open"], limit) if bar["high"] > limit else None
        # <<< SOLUTION
