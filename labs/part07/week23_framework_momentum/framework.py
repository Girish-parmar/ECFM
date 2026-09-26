"""Week 23 (S1–S2) — Strategy framework (Template Method, registry, YAML configs, next-bar runner, first-look
evaluator) and the directional-momentum group.

Rules that every strategy obeys (tested):
  * a strategy only EMITS INTENTS (target weights); it never talks to a broker;
  * a decision made at the CLOSE of bar t is filled at the OPEN of bar t+1;
  * inside on_bar a strategy can only see bars 0..t (the context enforces it).
Fill in every block marked "Your turn", then run:  python -m pytest week23_framework_momentum
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# --------------------------------------------------------------------------- S1 framework
@dataclass
class Intent:
    t: int                 # bar index at whose CLOSE the decision was made
    symbol: str
    weight: float
    strategy: str
    reason: str


@dataclass
class StrategyContext:
    """What a strategy may touch: past bars, the current bar index and an intent sink (given)."""
    bars: pd.DataFrame
    symbol: str = "X"
    t: int = -1
    intents: list[Intent] = field(default_factory=list)

    def history(self, n: int | None = None) -> pd.DataFrame:
        """Bars 0..t (the last n of them if n is given). Never anything after t."""
        start = 0 if n is None else max(0, self.t + 1 - n)
        return self.bars.iloc[start:self.t + 1]


class Strategy(ABC):
    """Template Method: the runner calls on_start, on_bar (every bar), on_stop; subclasses implement on_bar.
    `params` holds the declared defaults: name -> (default, min, max) (the Part 8 optimizer uses the ranges)."""
    name: str = "base"
    params: dict[str, tuple] = {}

    def __init__(self, ctx: StrategyContext, **overrides):
        """self.p = defaults overridden by `overrides`. An override for an UNDECLARED parameter raises ValueError;
        a value outside [min, max] raises ValueError."""
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def on_start(self) -> None:
        pass

    @abstractmethod
    def on_bar(self) -> None: ...

    def on_stop(self) -> None:
        pass

    def target(self, weight: float, reason: str) -> None:
        """Emit an order intent for the context's symbol at the current bar."""
        raise NotImplementedError("✍️ Your turn: see the docstring")


REGISTRY: dict[str, type[Strategy]] = {}


def register(name: str):
    """Class decorator: set cls.name = name and store it in REGISTRY; a duplicate name raises ValueError."""
    def deco(cls):
        pass  # ✍️ Your turn: see the docstring
        return cls
    return deco


CONFIG_KEYS = {"strategy", "universe", "timeframe", "params", "broker"}


def load_config(path: str | Path) -> dict:
    """Read a YAML strategy config. Required keys: strategy, universe (non-empty list), timeframe; optional params
    (dict, default {}) and broker (ib | alpaca | sim, default 'sim'). Unknown keys, an unregistered strategy or a
    bad broker raise ValueError. Return the dict with defaults filled in."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def run(strategy_cls: type[Strategy], bars: pd.DataFrame, symbol: str = "X", **params) -> np.ndarray:
    """Drive one strategy over `bars`: build a context, instantiate, on_start, then for t = 0..n−1 set ctx.t = t and
    call on_bar, then on_stop. Return `decided`: decided[t] = the weight in force after the decisions made at bar t
    (the LAST intent of bar t wins; with no intent the previous weight is kept; start at 0)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def quick_eval(decided, open_, cost_bps: float = 2.0, periods_per_year: int = 252) -> dict:
    """First-look evaluator (the real backtester is Part 8). decided[t] is filled at the OPEN of bar t+1 and held
    to the next open: pos[t] = decided[t−1] (pos[0] = 0), ret[t] = open[t+1]/open[t] − 1 (0 on the last bar),
    turnover[t] = |pos[t] − pos[t−1]|, pnl = pos·ret − turnover·cost_bps/1e4, equity = cumprod(1 + pnl).
    Return {"cagr", "sharpe" (NaN if pnl has no variance), "max_dd" (<= 0), "exposure" (share of bars with pos != 0),
    "turnover" (annualized sum of turnover), "pnl" (the array)}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------ S2 momentum group
def tsmom(close, lookback: int = 252, vol_n: int = 60, target_vol: float = 0.10, periods_per_year: int = 252,
          max_lev: float = 2.0) -> np.ndarray:
    """Time-series momentum: sign(close[t]/close[t−lookback] − 1) × target_vol / realized vol, where realized vol is
    the rolling std (vol_n, ddof=1) of daily log returns × √periods_per_year; clip to ±max_lev; NaN in warm-up."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def donchian_breakout(high, low, close, entry_n: int = 20, exit_n: int = 10) -> np.ndarray:
    """Turtle-style long/short breakout (state machine). Channels use the PRIOR bars only (exclude today):
    upper = max(high[t−entry_n..t−1]), lower = min(low[t−entry_n..t−1]); exit channels likewise with exit_n.
    Flat → +1 if close > upper, −1 if close < lower. Long → 0 if close < min(low[t−exit_n..t−1]).
    Short → 0 if close > max(high[t−exit_n..t−1]). Returns positions decided at each close; 0 during warm-up
    (t < max(entry_n, exit_n))."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def cross_sectional_momentum(closes: pd.DataFrame, top_q: float = 0.1) -> pd.DataFrame:
    """closes: month-end prices (dates × symbols). 12-1 momentum = closes.shift(1)/closes.shift(12) − 1 (skip the most
    recent month). Rank across symbols each date (pct); equal-weight the symbols with rank STRICTLY > 1 − top_q
    (with 10 names and top_q = 0.2 that is exactly 2 names; '>=' would take 3); rows with no signal are all 0.
    Weights sum to 1 on dates with a signal."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def dual_momentum(closes: pd.DataFrame, cash: str, lookback: int = 252) -> pd.Series:
    """Antonacci dual momentum (daily closes, one column is the cash/T-bill proxy `cash`): each date pick the risky
    asset with the best lookback return (relative momentum); hold it only if its return beats cash's (absolute
    momentum), else hold `cash`. NaN (no pick) during warm-up. Returns the chosen symbol per date."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


@register("tsmom")
class TSMOMStrategy(Strategy):
    """Registered strategy wrapping tsmom: on each bar with enough history, target the latest tsmom weight
    (reason 'tsmom'); emit nothing during warm-up."""
    params = {"lookback": (252, 20, 500), "vol_n": (60, 10, 250), "target_vol": (0.10, 0.02, 0.40),
              "max_lev": (2.0, 0.5, 4.0)}

    def on_bar(self) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")


@register("donchian")
class DonchianStrategy(Strategy):
    """Live-style Donchian: keep the current position in self.pos (set to 0 in on_start) and update it bar by bar
    from history(max(entry_n, exit_n) + 1) with the same rules as donchian_breakout; target it every bar after the
    warm-up (reason 'donchian'). Its decisions must equal donchian_breakout on the whole series (tested)."""
    params = {"entry_n": (20, 5, 100), "exit_n": (10, 2, 50)}

    def on_start(self) -> None:
        self.pos = 0.0

    def on_bar(self) -> None:
        raise NotImplementedError("✍️ Your turn: see the docstring")
