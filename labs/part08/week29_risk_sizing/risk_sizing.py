"""Week 29 (S17–S20) — Risk engine, market-risk measures, position sizing, Kelly and estimation error.

The risk engine sits between strategy intents and the OMS, in the backtest and live alike. Every rejection is logged
with a reason. Kelly is an UPPER BOUND, never a target: μ is estimated with large error.
Fill in every block marked "Your turn", then run:  python -m pytest week29_risk_sizing
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


# ----------------------------------------------------------------------- S17 risk engine
@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str = ""


class RiskRule:
    """order: {"symbol", "qty", "price", "sector"?}; ctx: {"equity", "gross", "positions" {sym: $ value},
    "sector_exposure" {sector: $}, "day_pnl", "start_equity", "adv" {sym: shares}}."""

    def check(self, order: dict, ctx: dict) -> RiskDecision:
        raise NotImplementedError


class MaxNotional(RiskRule):
    """Reject when |qty × price| > limit. Reason: 'notional 60,000 > 50,000'."""

    def __init__(self, limit: float):
        self.limit = limit

    def check(self, order, ctx):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class MaxGrossExposure(RiskRule):
    """Reject when (ctx gross + |order notional|) > max_leverage × equity. Reason: 'gross 2.10x > 2.0x'."""

    def __init__(self, max_leverage: float):
        self.max_leverage = max_leverage

    def check(self, order, ctx):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class MaxPositionWeight(RiskRule):
    """Single-name concentration: |current $ position in the symbol + order notional (signed)| / equity must be
    <= max_weight. Reason: 'SPY weight 25.0% > 20.0%'."""

    def __init__(self, max_weight: float):
        self.max_weight = max_weight

    def check(self, order, ctx):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class MaxSectorExposure(RiskRule):
    """|sector exposure after the order| / equity <= max_weight (orders without a sector pass).
    Reason: 'sector Tech 42.0% > 40.0%'."""

    def __init__(self, max_weight: float):
        self.max_weight = max_weight

    def check(self, order, ctx):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class MaxADVParticipation(RiskRule):
    """|qty| <= max_frac × ADV of the symbol (liquidity). Reason: 'qty 150,000 is 15.0% of ADV > 10.0%'."""

    def __init__(self, max_frac: float = 0.1):
        self.max_frac = max_frac

    def check(self, order, ctx):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class DailyLossLimit(RiskRule):
    """Block new orders once day_pnl / start_equity <= −max_loss_frac. Reason: 'daily loss -2.50% breaches -2.00%'."""

    def __init__(self, max_loss_frac: float):
        self.max_loss_frac = max_loss_frac

    def check(self, order, ctx):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class RiskEngine:
    """Chain of Responsibility: every rule must approve; the first rejection stops the chain with the reason prefixed
    by the rule's class name ('MaxNotional: notional ...'). Rejections are appended to self.log as
    (symbol, reason)."""

    def __init__(self, rules: list[RiskRule]):
        self.rules, self.log = rules, []

    def check(self, order: dict, ctx: dict) -> RiskDecision:
        raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------- S18 market risk
def hist_var_cvar(returns, alpha: float = 0.99) -> tuple[float, float]:
    """Historical VaR (the alpha-quantile of losses = −returns) and CVaR (mean of losses >= VaR), positive numbers."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def parametric_var(weights, cov, alpha: float = 0.99, mu=None) -> float:
    """Normal VaR of a portfolio: z_α·σ_p − μ_p with σ_p = √(w'Σw) (μ_p = w'μ, 0 if mu is None)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def component_var(weights, cov, alpha: float = 0.99) -> np.ndarray:
    """Euler decomposition of the (zero-mean) normal VaR: CVaR_i = w_i·(Σw)_i / σ_p · z_α; they sum to the VaR."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------- S19 sizing
def risk_per_trade_size(equity: float, risk_frac: float, entry: float, stop: float, multiplier: float = 1.0) -> int:
    """Units (floored) such that hitting the stop loses risk_frac of equity."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def vol_target_weight(returns, target_vol: float = 0.10, lookback: int = 60, periods: int = 252,
                      cap: float = 2.0) -> float:
    """Weight = target / annualized std (ddof=1) of the last `lookback` returns, capped at `cap`."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def turtle_units(equity: float, risk_frac: float, atr: float, point_value: float = 1.0) -> int:
    """Turtle unit: a 1-ATR move changes equity by risk_frac: floor(equity × risk_frac / (ATR × point value))."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def risk_of_ruin(r_multiples, risk_frac: float, ruin: float = 0.5, n_trades: int = 500, n_sims: int = 2000,
                 seed: int = 0) -> float:
    """Monte Carlo: each trade's result is an R-multiple drawn with replacement from r_multiples; equity ×= 1 +
    risk_frac·R. Return the share of simulated paths whose equity ever falls to `ruin` (e.g. 0.5 = −50%) or below."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# -------------------------------------------------------------------------- S20 Kelly
def kelly_discrete(p_win: float, win_loss_ratio: float) -> float:
    """f* = p − (1 − p)/b."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def kelly_continuous(mu: float, sigma: float, r: float = 0.0) -> float:
    """Growth-optimal leverage: (μ − r)/σ²."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def kelly_multi(mu, cov, r: float = 0.0) -> np.ndarray:
    """Multi-asset Kelly weights Σ⁻¹(μ − r) (use np.linalg.solve)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def kelly_growth_simulation(mu: float = 0.08, sigma: float = 0.16, fractions=(0.25, 0.5, 1.0, 2.0),
                            est_years: int = 5, n_years: int = 20, n_paths: int = 2000, seed: int = 0) -> pd.DataFrame:
    """For each path: ESTIMATE μ from est_years of simulated daily returns (sample mean × 252; σ known), set leverage
    = fraction × μ̂/σ² for each fraction, then simulate n_years of daily GBM returns (true μ, σ) with that constant
    leverage (daily return = lev·r_t, wealth ×= max(1 + lev·r_t, 0)). Use ONE rng = default_rng(seed); for each path
    draw the estimation sample first, then the future returns (shared by all fractions of that path).
    Return a DataFrame indexed by fraction with median_wealth, p_loss (share of paths ending below 1) and
    median_max_dd (median of each path's max drawdown, <= 0)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
