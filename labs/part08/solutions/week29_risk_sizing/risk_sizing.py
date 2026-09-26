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
        # >>> SOLUTION
        n = abs(order["qty"] * order["price"])
        return RiskDecision(n <= self.limit, f"notional {n:,.0f} > {self.limit:,.0f}")
        # <<< SOLUTION


class MaxGrossExposure(RiskRule):
    """Reject when (ctx gross + |order notional|) > max_leverage × equity. Reason: 'gross 2.10x > 2.0x'."""

    def __init__(self, max_leverage: float):
        self.max_leverage = max_leverage

    def check(self, order, ctx):
        # >>> SOLUTION
        gross = ctx["gross"] + abs(order["qty"] * order["price"])
        return RiskDecision(gross <= self.max_leverage * ctx["equity"],
                            f"gross {gross / ctx['equity']:.2f}x > {self.max_leverage}x")
        # <<< SOLUTION


class MaxPositionWeight(RiskRule):
    """Single-name concentration: |current $ position in the symbol + order notional (signed)| / equity must be
    <= max_weight. Reason: 'SPY weight 25.0% > 20.0%'."""

    def __init__(self, max_weight: float):
        self.max_weight = max_weight

    def check(self, order, ctx):
        # >>> SOLUTION
        after = ctx["positions"].get(order["symbol"], 0.0) + order["qty"] * order["price"]
        w = abs(after) / ctx["equity"]
        return RiskDecision(w <= self.max_weight, f"{order['symbol']} weight {w:.1%} > {self.max_weight:.1%}")
        # <<< SOLUTION


class MaxSectorExposure(RiskRule):
    """|sector exposure after the order| / equity <= max_weight (orders without a sector pass).
    Reason: 'sector Tech 42.0% > 40.0%'."""

    def __init__(self, max_weight: float):
        self.max_weight = max_weight

    def check(self, order, ctx):
        # >>> SOLUTION
        sec = order.get("sector")
        if sec is None:
            return RiskDecision(True)
        after = ctx["sector_exposure"].get(sec, 0.0) + order["qty"] * order["price"]
        w = abs(after) / ctx["equity"]
        return RiskDecision(w <= self.max_weight, f"sector {sec} {w:.1%} > {self.max_weight:.1%}")
        # <<< SOLUTION


class MaxADVParticipation(RiskRule):
    """|qty| <= max_frac × ADV of the symbol (liquidity). Reason: 'qty 150,000 is 15.0% of ADV > 10.0%'."""

    def __init__(self, max_frac: float = 0.1):
        self.max_frac = max_frac

    def check(self, order, ctx):
        # >>> SOLUTION
        frac = abs(order["qty"]) / ctx["adv"][order["symbol"]]
        return RiskDecision(frac <= self.max_frac,
                            f"qty {abs(order['qty']):,.0f} is {frac:.1%} of ADV > {self.max_frac:.1%}")
        # <<< SOLUTION


class DailyLossLimit(RiskRule):
    """Block new orders once day_pnl / start_equity <= −max_loss_frac. Reason: 'daily loss -2.50% breaches -2.00%'."""

    def __init__(self, max_loss_frac: float):
        self.max_loss_frac = max_loss_frac

    def check(self, order, ctx):
        # >>> SOLUTION
        dd = ctx["day_pnl"] / ctx["start_equity"]
        return RiskDecision(dd > -self.max_loss_frac, f"daily loss {dd:.2%} breaches {-self.max_loss_frac:.2%}")
        # <<< SOLUTION


class RiskEngine:
    """Chain of Responsibility: every rule must approve; the first rejection stops the chain with the reason prefixed
    by the rule's class name ('MaxNotional: notional ...'). Rejections are appended to self.log as
    (symbol, reason)."""

    def __init__(self, rules: list[RiskRule]):
        self.rules, self.log = rules, []

    def check(self, order: dict, ctx: dict) -> RiskDecision:
        # >>> SOLUTION
        for rule in self.rules:
            d = rule.check(order, ctx)
            if not d.approved:
                reason = f"{type(rule).__name__}: {d.reason}"
                self.log.append((order["symbol"], reason))
                return RiskDecision(False, reason)
        return RiskDecision(True)
        # <<< SOLUTION


# --------------------------------------------------------------------- S18 market risk
def hist_var_cvar(returns, alpha: float = 0.99) -> tuple[float, float]:
    """Historical VaR (the alpha-quantile of losses = −returns) and CVaR (mean of losses >= VaR), positive numbers."""
    # >>> SOLUTION
    losses = -np.asarray(returns, dtype=float)
    var = np.quantile(losses, alpha)
    return float(var), float(losses[losses >= var].mean())
    # <<< SOLUTION


def parametric_var(weights, cov, alpha: float = 0.99, mu=None) -> float:
    """Normal VaR of a portfolio: z_α·σ_p − μ_p with σ_p = √(w'Σw) (μ_p = w'μ, 0 if mu is None)."""
    # >>> SOLUTION
    w = np.asarray(weights, dtype=float)
    sd = float(np.sqrt(w @ cov @ w))
    m = 0.0 if mu is None else float(w @ np.asarray(mu))
    return norm.ppf(alpha) * sd - m
    # <<< SOLUTION


def component_var(weights, cov, alpha: float = 0.99) -> np.ndarray:
    """Euler decomposition of the (zero-mean) normal VaR: CVaR_i = w_i·(Σw)_i / σ_p · z_α; they sum to the VaR."""
    # >>> SOLUTION
    w = np.asarray(weights, dtype=float)
    sd = np.sqrt(w @ cov @ w)
    return w * (cov @ w) / sd * norm.ppf(alpha)
    # <<< SOLUTION


# ------------------------------------------------------------------------- S19 sizing
def risk_per_trade_size(equity: float, risk_frac: float, entry: float, stop: float, multiplier: float = 1.0) -> int:
    """Units (floored) such that hitting the stop loses risk_frac of equity."""
    # >>> SOLUTION
    return int(equity * risk_frac / (abs(entry - stop) * multiplier))
    # <<< SOLUTION


def vol_target_weight(returns, target_vol: float = 0.10, lookback: int = 60, periods: int = 252,
                      cap: float = 2.0) -> float:
    """Weight = target / annualized std (ddof=1) of the last `lookback` returns, capped at `cap`."""
    # >>> SOLUTION
    vol = np.std(np.asarray(returns, dtype=float)[-lookback:], ddof=1) * np.sqrt(periods)
    return float(min(target_vol / vol, cap))
    # <<< SOLUTION


def turtle_units(equity: float, risk_frac: float, atr: float, point_value: float = 1.0) -> int:
    """Turtle unit: a 1-ATR move changes equity by risk_frac: floor(equity × risk_frac / (ATR × point value))."""
    # >>> SOLUTION
    return int(equity * risk_frac / (atr * point_value))
    # <<< SOLUTION


def risk_of_ruin(r_multiples, risk_frac: float, ruin: float = 0.5, n_trades: int = 500, n_sims: int = 2000,
                 seed: int = 0) -> float:
    """Monte Carlo: each trade's result is an R-multiple drawn with replacement from r_multiples; equity ×= 1 +
    risk_frac·R. Return the share of simulated paths whose equity ever falls to `ruin` (e.g. 0.5 = −50%) or below."""
    # >>> SOLUTION
    rng = np.random.default_rng(seed)
    R = rng.choice(np.asarray(r_multiples, dtype=float), size=(n_sims, n_trades))
    eq = np.cumprod(1 + risk_frac * R, axis=1)
    return float(np.mean(eq.min(axis=1) <= ruin))
    # <<< SOLUTION


# -------------------------------------------------------------------------- S20 Kelly
def kelly_discrete(p_win: float, win_loss_ratio: float) -> float:
    """f* = p − (1 − p)/b."""
    # >>> SOLUTION
    return p_win - (1 - p_win) / win_loss_ratio
    # <<< SOLUTION


def kelly_continuous(mu: float, sigma: float, r: float = 0.0) -> float:
    """Growth-optimal leverage: (μ − r)/σ²."""
    # >>> SOLUTION
    return (mu - r) / sigma ** 2
    # <<< SOLUTION


def kelly_multi(mu, cov, r: float = 0.0) -> np.ndarray:
    """Multi-asset Kelly weights Σ⁻¹(μ − r) (use np.linalg.solve)."""
    # >>> SOLUTION
    return np.linalg.solve(np.asarray(cov, dtype=float), np.asarray(mu, dtype=float) - r)
    # <<< SOLUTION


def kelly_growth_simulation(mu: float = 0.08, sigma: float = 0.16, fractions=(0.25, 0.5, 1.0, 2.0),
                            est_years: int = 5, n_years: int = 20, n_paths: int = 2000, seed: int = 0) -> pd.DataFrame:
    """For each path: ESTIMATE μ from est_years of simulated daily returns (sample mean × 252; σ known), set leverage
    = fraction × μ̂/σ² for each fraction, then simulate n_years of daily GBM returns (true μ, σ) with that constant
    leverage (daily return = lev·r_t, wealth ×= max(1 + lev·r_t, 0)). Use ONE rng = default_rng(seed); for each path
    draw the estimation sample first, then the future returns (shared by all fractions of that path).
    Return a DataFrame indexed by fraction with median_wealth, p_loss (share of paths ending below 1) and
    median_max_dd (median of each path's max drawdown, <= 0)."""
    # >>> SOLUTION
    rng = np.random.default_rng(seed)
    d = 252
    res = {f: {"w": [], "dd": []} for f in fractions}
    for _ in range(n_paths):
        est = rng.normal(mu / d, sigma / np.sqrt(d), est_years * d)
        mu_hat = est.mean() * d
        fut = rng.normal(mu / d, sigma / np.sqrt(d), n_years * d)
        for f in fractions:
            lev = f * mu_hat / sigma ** 2
            wealth = np.cumprod(np.maximum(1 + lev * fut, 0.0))
            peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
            res[f]["w"].append(wealth[-1])
            res[f]["dd"].append((wealth / peak - 1).min())
    return pd.DataFrame({f: {"median_wealth": float(np.median(v["w"])), "p_loss": float(np.mean(np.array(v["w"]) < 1)),
                             "median_max_dd": float(np.median(v["dd"]))} for f, v in res.items()}).T
    # <<< SOLUTION
