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
        # >>> SOLUTION
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        self.p = {k: v[0] for k, v in self.params.items()}
        for k, v in overrides.items():
            lo, hi = self.params[k][1], self.params[k][2]
            if not lo <= v <= hi:
                raise ValueError(f"{self.name}: {k}={v} outside [{lo}, {hi}]")
            self.p[k] = v
        self.ctx = ctx
        # <<< SOLUTION

    def on_start(self) -> None:
        pass

    @abstractmethod
    def on_bar(self) -> None: ...

    def on_stop(self) -> None:
        pass

    def target(self, weight: float, reason: str) -> None:
        """Emit an order intent for the context's symbol at the current bar."""
        # >>> SOLUTION
        self.ctx.intents.append(Intent(self.ctx.t, self.ctx.symbol, float(weight), self.name, reason))
        # <<< SOLUTION


REGISTRY: dict[str, type[Strategy]] = {}


def register(name: str):
    """Class decorator: set cls.name = name and store it in REGISTRY; a duplicate name raises ValueError."""
    def deco(cls):
        # >>> SOLUTION (pass)
        if name in REGISTRY:
            raise ValueError(f"strategy {name!r} already registered")
        cls.name = name
        REGISTRY[name] = cls
        # <<< SOLUTION
        return cls
    return deco


CONFIG_KEYS = {"strategy", "universe", "timeframe", "params", "broker"}


def load_config(path: str | Path) -> dict:
    """Read a YAML strategy config. Required keys: strategy, universe (non-empty list), timeframe; optional params
    (dict, default {}) and broker (ib | alpaca | sim, default 'sim'). Unknown keys, an unregistered strategy or a
    bad broker raise ValueError. Return the dict with defaults filled in."""
    # >>> SOLUTION
    cfg = yaml.safe_load(Path(path).read_text())
    extra = set(cfg) - CONFIG_KEYS
    if extra:
        raise ValueError(f"unknown config keys {sorted(extra)}")
    for k in ("strategy", "universe", "timeframe"):
        if k not in cfg:
            raise ValueError(f"missing config key {k!r}")
    if cfg["strategy"] not in REGISTRY:
        raise ValueError(f"unknown strategy {cfg['strategy']!r}")
    if not isinstance(cfg["universe"], list) or not cfg["universe"]:
        raise ValueError("universe must be a non-empty list")
    cfg.setdefault("params", {})
    cfg.setdefault("broker", "sim")
    if cfg["broker"] not in ("ib", "alpaca", "sim"):
        raise ValueError(f"unknown broker {cfg['broker']!r}")
    return cfg
    # <<< SOLUTION


def run(strategy_cls: type[Strategy], bars: pd.DataFrame, symbol: str = "X", **params) -> np.ndarray:
    """Drive one strategy over `bars`: build a context, instantiate, on_start, then for t = 0..n−1 set ctx.t = t and
    call on_bar, then on_stop. Return `decided`: decided[t] = the weight in force after the decisions made at bar t
    (the LAST intent of bar t wins; with no intent the previous weight is kept; start at 0)."""
    # >>> SOLUTION
    ctx = StrategyContext(bars, symbol)
    s = strategy_cls(ctx, **params)
    s.on_start()
    decided = np.zeros(len(bars))
    w = 0.0
    for t in range(len(bars)):
        ctx.t = t
        before = len(ctx.intents)
        s.on_bar()
        if len(ctx.intents) > before:
            w = ctx.intents[-1].weight
        decided[t] = w
    s.on_stop()
    return decided
    # <<< SOLUTION


def quick_eval(decided, open_, cost_bps: float = 2.0, periods_per_year: int = 252) -> dict:
    """First-look evaluator (the real backtester is Part 8). decided[t] is filled at the OPEN of bar t+1 and held
    to the next open: pos[t] = decided[t−1] (pos[0] = 0), ret[t] = open[t+1]/open[t] − 1 (0 on the last bar),
    turnover[t] = |pos[t] − pos[t−1]|, pnl = pos·ret − turnover·cost_bps/1e4, equity = cumprod(1 + pnl).
    Return {"cagr", "sharpe" (NaN if pnl has no variance), "max_dd" (<= 0), "exposure" (share of bars with pos != 0),
    "turnover" (annualized sum of turnover), "pnl" (the array)}."""
    # >>> SOLUTION
    decided = np.nan_to_num(np.asarray(decided, dtype=float))
    open_ = np.asarray(open_, dtype=float)
    pos = np.zeros_like(decided)
    pos[1:] = decided[:-1]
    ret = np.zeros_like(open_)
    ret[:-1] = open_[1:] / open_[:-1] - 1
    turnover = np.abs(np.diff(pos, prepend=0.0))
    pnl = pos * ret - turnover * cost_bps / 1e4
    equity = np.cumprod(1 + pnl)
    dd = equity / np.maximum.accumulate(equity) - 1
    sd = pnl.std()
    return {"cagr": equity[-1] ** (periods_per_year / len(pnl)) - 1,
            "sharpe": pnl.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan,
            "max_dd": dd.min(), "exposure": float(np.mean(pos != 0)),
            "turnover": turnover.sum() / len(pnl) * periods_per_year, "pnl": pnl}
    # <<< SOLUTION


# ------------------------------------------------------------------------ S2 momentum group
def tsmom(close, lookback: int = 252, vol_n: int = 60, target_vol: float = 0.10, periods_per_year: int = 252,
          max_lev: float = 2.0) -> np.ndarray:
    """Time-series momentum: sign(close[t]/close[t−lookback] − 1) × target_vol / realized vol, where realized vol is
    the rolling std (vol_n, ddof=1) of daily log returns × √periods_per_year; clip to ±max_lev; NaN in warm-up."""
    # >>> SOLUTION
    close = np.asarray(close, dtype=float)
    past = np.full(close.shape, np.nan)
    past[lookback:] = close[lookback:] / close[:-lookback] - 1
    r = np.full(close.shape, np.nan)
    r[1:] = np.diff(np.log(close))
    vol = pd.Series(r).rolling(vol_n).std().to_numpy() * np.sqrt(periods_per_year)
    return np.clip(np.sign(past) * target_vol / vol, -max_lev, max_lev)
    # <<< SOLUTION


def donchian_breakout(high, low, close, entry_n: int = 20, exit_n: int = 10) -> np.ndarray:
    """Turtle-style long/short breakout (state machine). Channels use the PRIOR bars only (exclude today):
    upper = max(high[t−entry_n..t−1]), lower = min(low[t−entry_n..t−1]); exit channels likewise with exit_n.
    Flat → +1 if close > upper, −1 if close < lower. Long → 0 if close < min(low[t−exit_n..t−1]).
    Short → 0 if close > max(high[t−exit_n..t−1]). Returns positions decided at each close; 0 during warm-up
    (t < max(entry_n, exit_n))."""
    # >>> SOLUTION
    h, l, c = (np.asarray(a, dtype=float) for a in (high, low, close))  # noqa: E741
    pos = np.zeros(c.size)
    for t in range(max(entry_n, exit_n), c.size):
        prev = pos[t - 1]
        if prev == 0:
            if c[t] > h[t - entry_n:t].max():
                pos[t] = 1
            elif c[t] < l[t - entry_n:t].min():
                pos[t] = -1
        elif prev == 1:
            pos[t] = 0 if c[t] < l[t - exit_n:t].min() else 1
        else:
            pos[t] = 0 if c[t] > h[t - exit_n:t].max() else -1
    return pos
    # <<< SOLUTION


def cross_sectional_momentum(closes: pd.DataFrame, top_q: float = 0.1) -> pd.DataFrame:
    """closes: month-end prices (dates × symbols). 12-1 momentum = closes.shift(1)/closes.shift(12) − 1 (skip the most
    recent month). Rank across symbols each date (pct); equal-weight the symbols with rank STRICTLY > 1 − top_q
    (with 10 names and top_q = 0.2 that is exactly 2 names; '>=' would take 3); rows with no signal are all 0.
    Weights sum to 1 on dates with a signal."""
    # >>> SOLUTION
    mom = closes.shift(1) / closes.shift(12) - 1
    rank = mom.rank(axis=1, pct=True)
    w = (rank > 1 - top_q + 1e-12).astype(float)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)
    # <<< SOLUTION


def dual_momentum(closes: pd.DataFrame, cash: str, lookback: int = 252) -> pd.Series:
    """Antonacci dual momentum (daily closes, one column is the cash/T-bill proxy `cash`): each date pick the risky
    asset with the best lookback return (relative momentum); hold it only if its return beats cash's (absolute
    momentum), else hold `cash`. NaN (no pick) during warm-up. Returns the chosen symbol per date."""
    # >>> SOLUTION
    ret = closes / closes.shift(lookback) - 1
    risky = ret.drop(columns=cash)
    best = risky.fillna(-np.inf).idxmax(axis=1)             # all-NaN warm-up rows would raise; masked below
    out = pd.Series(np.nan, index=closes.index, dtype=object)
    ok = ret[cash].notna()
    best_ret = risky.max(axis=1)
    out[ok] = np.where(best_ret[ok] > ret.loc[ok, cash], best[ok], cash)
    return out
    # <<< SOLUTION


@register("tsmom")
class TSMOMStrategy(Strategy):
    """Registered strategy wrapping tsmom: on each bar with enough history, target the latest tsmom weight
    (reason 'tsmom'); emit nothing during warm-up."""
    params = {"lookback": (252, 20, 500), "vol_n": (60, 10, 250), "target_vol": (0.10, 0.02, 0.40),
              "max_lev": (2.0, 0.5, 4.0)}

    def on_bar(self) -> None:
        # >>> SOLUTION
        need = max(self.p["lookback"], self.p["vol_n"]) + 1
        hist = self.ctx.history(need)
        if len(hist) < need:
            return
        w = tsmom(hist["close"].to_numpy(), self.p["lookback"], self.p["vol_n"], self.p["target_vol"],
                  max_lev=self.p["max_lev"])[-1]
        if np.isfinite(w):
            self.target(w, "tsmom")
        # <<< SOLUTION


@register("donchian")
class DonchianStrategy(Strategy):
    """Live-style Donchian: keep the current position in self.pos (set to 0 in on_start) and update it bar by bar
    from history(max(entry_n, exit_n) + 1) with the same rules as donchian_breakout; target it every bar after the
    warm-up (reason 'donchian'). Its decisions must equal donchian_breakout on the whole series (tested)."""
    params = {"entry_n": (20, 5, 100), "exit_n": (10, 2, 50)}

    def on_start(self) -> None:
        self.pos = 0.0

    def on_bar(self) -> None:
        # >>> SOLUTION
        en, ex = self.p["entry_n"], self.p["exit_n"]
        h = self.ctx.history(max(en, ex) + 1)
        if self.ctx.t < max(en, ex):
            return
        hi, lo, c = h["high"].to_numpy(), h["low"].to_numpy(), h["close"].to_numpy()[-1]
        if self.pos == 0:
            if c > hi[-1 - en:-1].max():
                self.pos = 1.0
            elif c < lo[-1 - en:-1].min():
                self.pos = -1.0
        elif self.pos == 1 and c < lo[-1 - ex:-1].min():
            self.pos = 0.0
        elif self.pos == -1 and c > hi[-1 - ex:-1].max():
            self.pos = 0.0
        self.target(self.pos, "donchian")
        # <<< SOLUTION
