"""Week 24 (S5–S6) — Option strategy builder, templates, structure selector and combo orders.

A strategy is a list of legs; the Builder adds them one call at a time. Quantities are in CONTRACTS (+ long, − short);
an underlying leg with qty 1 means one multiplier (100 shares). Values are in dollars (× multiplier).
Probability of profit is a MODEL estimate under a lognormal distribution, not a promise.
Fill in every block marked "Your turn", then run:  python -m pytest week24_option_builder
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from common import bsm_greeks, bsm_price


# ------------------------------------------------------------------------------ S5 builder
@dataclass(frozen=True)
class Leg:
    cp: int          # +1 call, −1 put, 0 underlying
    K: float
    T: float         # years to expiry (0 for the underlying)
    qty: int
    iv: float = 0.2


@dataclass
class OptionStrategy:
    name: str
    legs: list[Leg] = field(default_factory=list)
    multiplier: int = 100

    def add(self, cp: int, K: float, T: float, qty: int, iv: float = 0.2) -> OptionStrategy:
        """Append a Leg and return self (so calls can be chained)."""
        # >>> SOLUTION
        self.legs.append(Leg(cp, K, T, qty, iv))
        return self
        # <<< SOLUTION

    @property
    def first_expiry(self) -> float:
        return min(L.T for L in self.legs if L.cp != 0)

    def value(self, S, r: float = 0.04, q: float = 0.0, dt: float = 0.0, dvol: float = 0.0):
        """Mark-to-model value (dollars) after dt years pass and every IV shifts by dvol. Works for scalar or array S.
        Option legs with remaining time tau = T − dt > 1e-9 use bsm_price, expired ones their intrinsic value;
        the underlying leg is worth qty × S."""
        # >>> SOLUTION
        S = np.asarray(S, dtype=float)
        v = np.zeros_like(S)
        for L in self.legs:
            if L.cp == 0:
                v = v + L.qty * S
                continue
            tau = L.T - dt
            leg = bsm_price(S, L.K, tau, r, q, L.iv + dvol, L.cp) if tau > 1e-9 else np.maximum(L.cp * (S - L.K), 0.0)
            v = v + L.qty * leg
        return v * self.multiplier
        # <<< SOLUTION

    def payoff_at_first_expiry(self, S, r: float = 0.04, q: float = 0.0):
        """Value when the FIRST option expiry arrives (later legs, e.g. of a calendar, still carry time value)."""
        return self.value(S, r, q, dt=self.first_expiry)

    def greeks(self, S: float, r: float = 0.04, q: float = 0.0) -> dict[str, float]:
        """Net delta, gamma, vega (per vol POINT) and theta (per DAY) in dollars-per-unit terms × multiplier:
        delta and gamma in shares; underlying legs add qty to delta."""
        # >>> SOLUTION
        tot = dict.fromkeys(("delta", "gamma", "vega", "theta"), 0.0)
        for L in self.legs:
            if L.cp == 0:
                tot["delta"] += L.qty
                continue
            g = bsm_greeks(S, L.K, L.T, r, q, L.iv, L.cp)
            tot["delta"] += L.qty * g["delta"]
            tot["gamma"] += L.qty * g["gamma"]
            tot["vega"] += L.qty * g["vega"] / 100
            tot["theta"] += L.qty * g["theta"] / 365
        return {k: float(v * self.multiplier) for k, v in tot.items()}
        # <<< SOLUTION

    def analyze(self, S: float, r: float = 0.04, q: float = 0.0, sigma: float | None = None, grid=None) -> dict:
        """cost = value today (positive = debit paid, negative = credit received); P&L at the first expiry on a price
        grid (default 2001 points from 0.5·S to 1.5·S) = payoff_at_first_expiry − cost.
        breakevens: grid points where the P&L changes sign (take grid[1:] where sign differs from the previous point;
        round to 2 decimals); max_profit / max_loss over the grid; pop = ∫ density·1{P&L > 0} dS under a lognormal
        with vol `sigma` (default: mean IV of the option legs), drift r − q − σ²/2, horizon = first expiry."""
        # >>> SOLUTION
        grid = np.linspace(0.5 * S, 1.5 * S, 2001) if grid is None else np.asarray(grid, dtype=float)
        cost = float(self.value(S, r, q))
        pnl = self.payoff_at_first_expiry(grid, r, q) - cost
        sign = np.sign(pnl)
        T0 = self.first_expiry
        sigma = sigma or float(np.mean([L.iv for L in self.legs if L.cp != 0]))
        z = (np.log(grid / S) - (r - q - 0.5 * sigma ** 2) * T0) / (sigma * np.sqrt(T0))
        density = norm.pdf(z) / (grid * sigma * np.sqrt(T0))
        return {"cost": cost, "breakevens": grid[1:][sign[1:] != sign[:-1]].round(2),
                "max_profit": float(pnl.max()), "max_loss": float(pnl.min()),
                "pop": float(np.trapezoid(density * (pnl > 0), grid))}
        # <<< SOLUTION

    def is_defined_risk(self) -> bool:
        """Defined risk = every short option is covered: net call qty + underlying qty >= 0 (short calls covered by
        long calls or shares) AND net put qty >= 0 (short puts covered by long puts). Calendars count as covered."""
        # >>> SOLUTION
        calls = sum(L.qty for L in self.legs if L.cp == 1)
        puts = sum(L.qty for L in self.legs if L.cp == -1)
        shares = sum(L.qty for L in self.legs if L.cp == 0)
        return calls + shares >= 0 and puts >= 0
        # <<< SOLUTION


# --------------------------------------------------------------------------- S5 templates
def strike_for_delta(S, T, r, q, iv_fn, cp: int, target: float, strikes) -> float:
    """Listed strike whose BSM delta (each strike with its own iv_fn(K)) is closest to target (e.g. −0.16)."""
    # >>> SOLUTION
    K = np.asarray(strikes, dtype=float)
    d = np.array([bsm_greeks(S, k, T, r, q, iv_fn(k), cp)["delta"] for k in K])
    return float(K[np.argmin(np.abs(d - target))])
    # <<< SOLUTION


def iron_condor(S, T, iv_fn, strikes, short_delta: float = 0.16, wing: float = 5.0, r: float = 0.04, q: float = 0.0,
                qty: int = 1) -> OptionStrategy:
    """Short the put and call nearest ±short_delta, buy wings `wing` further out (legs in the order: long put, short put,
    short call, long call), each leg with iv_fn(K)."""
    # >>> SOLUTION
    kp = strike_for_delta(S, T, r, q, iv_fn, -1, -short_delta, strikes)
    kc = strike_for_delta(S, T, r, q, iv_fn, 1, short_delta, strikes)
    return (OptionStrategy("iron condor")
            .add(-1, kp - wing, T, qty, iv_fn(kp - wing)).add(-1, kp, T, -qty, iv_fn(kp))
            .add(1, kc, T, -qty, iv_fn(kc)).add(1, kc + wing, T, qty, iv_fn(kc + wing)))
    # <<< SOLUTION


def vertical(cp: int, k_long: float, k_short: float, T: float, iv_fn, qty: int = 1) -> OptionStrategy:
    """Two-leg spread: long cp at k_long, short cp at k_short. Name: 'bull call spread' (call, k_long < k_short),
    'bear call spread' (call, k_long > k_short), 'bear put spread' (put, k_long > k_short), 'bull put spread'
    (put, k_long < k_short)."""
    # >>> SOLUTION
    if cp == 1:
        name = "bull call spread" if k_long < k_short else "bear call spread"
    else:
        name = "bear put spread" if k_long > k_short else "bull put spread"
    return OptionStrategy(name).add(cp, k_long, T, qty, iv_fn(k_long)).add(cp, k_short, T, -qty, iv_fn(k_short))
    # <<< SOLUTION


def collar(k_put: float, k_call: float, T: float, iv_fn, shares_lots: int = 1) -> OptionStrategy:
    """Long shares (shares_lots × 100) + long put k_put + short call k_call (the call finances the put)."""
    # >>> SOLUTION
    return (OptionStrategy("collar").add(0, 0.0, 0.0, shares_lots)
            .add(-1, k_put, T, shares_lots, iv_fn(k_put)).add(1, k_call, T, -shares_lots, iv_fn(k_call)))
    # <<< SOLUTION


def calendar(K: float, T_near: float, T_far: float, cp: int, iv_near: float, iv_far: float) -> OptionStrategy:
    """Sell the near expiry, buy the far expiry at the same strike."""
    # >>> SOLUTION
    return OptionStrategy("calendar").add(cp, K, T_near, -1, iv_near).add(cp, K, T_far, 1, iv_far)
    # <<< SOLUTION


# ---------------------------------------------------------------------- S6 IV rank & selector
def iv_rank(iv_history, window: int = 252) -> float:
    """(IV_today − min) / (max − min) × 100 over the last `window` values (today included); 50 if max == min."""
    # >>> SOLUTION
    x = np.asarray(iv_history, dtype=float)[-window:]
    lo, hi = x.min(), x.max()
    return 50.0 if hi == lo else float((x[-1] - lo) / (hi - lo) * 100)
    # <<< SOLUTION


def iv_percentile(iv_history, window: int = 252) -> float:
    """Share (%) of the previous window−1 values strictly BELOW today's IV."""
    # >>> SOLUTION
    x = np.asarray(iv_history, dtype=float)[-window:]
    return float(np.mean(x[:-1] < x[-1]) * 100)
    # <<< SOLUTION


def select_structure(direction: int, ivr: float, term_slope: float = 0.0, event: bool = False,
                     implied_move: float | None = None, expected_move: float | None = None) -> str:
    """Documented rules (a hypothesis, tested in Part 8):
      event and implied_move < expected_move              -> 'long_straddle'   (buy the move when it is cheap)
      direction +1: ivr < 30 'long_call' · 30–70 'bull_call_spread' · >= 70 'bull_put_spread' (sell rich premium)
      direction −1: ivr < 30 'long_put'  · 30–70 'bear_put_spread'  · >= 70 'bear_call_spread'
      direction  0: ivr >= 50 'iron_condor' · else term_slope > 0 (contango) 'calendar' · else 'no_trade'."""
    # >>> SOLUTION
    if event and implied_move is not None and expected_move is not None and implied_move < expected_move:
        return "long_straddle"
    if direction != 0:
        tiers = (("long_call", "bull_call_spread", "bull_put_spread") if direction > 0
                 else ("long_put", "bear_put_spread", "bear_call_spread"))
        return tiers[0] if ivr < 30 else (tiers[1] if ivr < 70 else tiers[2])
    if ivr >= 50:
        return "iron_condor"
    return "calendar" if term_slope > 0 else "no_trade"
    # <<< SOLUTION


# ------------------------------------------------------------------------ S5 combo orders
def combo_prices(qtys, bids, asks) -> dict[str, float]:
    """Per-share prices of a combo (positive = debit). mid = Σ qty·(bid+ask)/2; natural = what crossing every leg
    costs (buy at ask, sell at bid) = Σ (qty·ask if qty > 0 else qty·bid); leg_cost = natural − mid (always >= 0)."""
    # >>> SOLUTION
    q, b, a = (np.asarray(x, dtype=float) for x in (qtys, bids, asks))
    mid = float(np.sum(q * (b + a) / 2))
    natural = float(np.sum(np.where(q > 0, q * a, q * b)))
    return {"mid": mid, "natural": natural, "leg_cost": natural - mid}
    # <<< SOLUTION


def to_ib_combo(strategy: OptionStrategy, conids: list[int], symbol: str):
    """IB combo: Contract(secType='BAG', symbol, exchange='SMART', currency='USD') with one ComboLeg per OPTION leg
    (conId from `conids` in leg order, ratio = |qty|, action BUY for qty > 0 else SELL, exchange 'SMART').
    Underlying legs are not allowed in this helper (ValueError)."""
    # >>> SOLUTION
    from ib_async import ComboLeg, Contract
    if any(L.cp == 0 for L in strategy.legs):
        raise ValueError("stock legs are not supported in this combo helper")
    legs = [ComboLeg(conId=cid, ratio=abs(L.qty), action="BUY" if L.qty > 0 else "SELL", exchange="SMART")
            for L, cid in zip(strategy.legs, conids, strict=True)]
    return Contract(secType="BAG", symbol=symbol, exchange="SMART", currency="USD", comboLegs=legs)
    # <<< SOLUTION


def to_alpaca_mleg(strategy: OptionStrategy, occ_symbols: list[str], limit_price: float, qty: int = 1) -> dict:
    """Alpaca multi-leg order payload (plain dict; check your alpaca-py version for OrderClass.MLEG support):
    {"order_class": "mleg", "qty": qty, "type": "limit", "limit_price": round(limit_price, 2), "time_in_force": "day",
     "legs": [{"symbol", "ratio_qty": |leg qty|, "side": "buy"/"sell"}, ...]} for the option legs in order."""
    # >>> SOLUTION
    opts = [L for L in strategy.legs if L.cp != 0]
    return {"order_class": "mleg", "qty": qty, "type": "limit", "limit_price": round(limit_price, 2),
            "time_in_force": "day",
            "legs": [{"symbol": s, "ratio_qty": abs(L.qty), "side": "buy" if L.qty > 0 else "sell"}
                     for L, s in zip(opts, occ_symbols, strict=True)]}
    # <<< SOLUTION
