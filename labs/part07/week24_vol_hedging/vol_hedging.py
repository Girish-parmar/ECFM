"""Week 24 (S7–S8) — Event and volatility option strategies, and hedging.

Hedging is insurance: judge it by COST vs DRAWDOWN REDUCTION of the whole portfolio, never by the hedge's own P&L.
Research functions that use future data are named *_research and must never feed a trading signal (tested).
Fill in every block marked "Your turn", then run:  python -m pytest week24_vol_hedging
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from common import bsm_price


# ------------------------------------------------------------------------ S7 events & volatility
def implied_move(straddle_price: float, spot: float) -> float:
    """Expected absolute move to expiry implied by the ATM straddle, as a fraction of spot: straddle / spot."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def straddle_rule_of_thumb(S: float, sigma: float, T: float) -> float:
    """ATM straddle ≈ √(2/π)·S·σ·√T ≈ 0.8·S·σ·√T (use √(2/π) exactly)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def event_study(implied: np.ndarray, realized: np.ndarray) -> dict:
    """Implied vs realized absolute moves over past events (e.g. earnings), both as fractions.
    Return {"n", "mean_implied", "mean_realized", "edge" = mean(realized − implied) (a long straddle's rough edge per
    unit of spot), "share_realized_above" = share of events where realized > implied}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def realized_vol(close: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling std of daily log returns over the last `window` days (ddof=1), known at each close."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def vrp_research(iv_30d: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """RESEARCH ONLY: implied vol today minus the realized vol over the NEXT `window` days (realized_vol shifted by
    −window). It uses future data by design: fine for measuring the variance risk premium, never a trading signal."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def vrp_signal(iv_30d: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """TRADABLE version: implied vol today minus TRAILING realized vol (known at the close)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------------ S8 hedging
def futures_hedge_contracts(portfolio_value: float, beta: float, fut_price: float, multiplier: float,
                            hedge_ratio: float = 1.0) -> int:
    """Index futures to SELL: round(hedge_ratio × β × value / (futures price × multiplier))."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def zero_cost_call_strike(S: float, T: float, k_put: float, iv_fn, r: float = 0.04, q: float = 0.0) -> float:
    """Call strike (above S) whose premium equals the put's premium at k_put: solve with brentq on [S, 3S]
    (each strike priced with iv_fn(K)). This is a zero-cost collar: the call you sell pays for the put you buy."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def crash_path(seed: int = 0, calm: int = 500, crash: int = 15, recovery: int = 250, crash_size: float = -0.30):
    """Synthetic index path (given): calm drift up at 14% vol, a crash of `crash_size` over `crash` days at 60% vol,
    then a recovery at 30% vol. Returns (close Series, implied-vol Series)."""
    rng = np.random.default_rng(seed)
    n = calm + crash + recovery
    vol = np.concatenate([np.full(calm, 0.14), np.full(crash, 0.60), np.full(recovery, 0.30)])
    drift = np.concatenate([np.full(calm, 0.10 / 252), np.full(crash, np.log(1 + crash_size) / max(crash, 1)),
                            np.full(recovery, 0.15 / 252)])
    lr = drift + vol / np.sqrt(252) * rng.standard_normal(n) * np.concatenate([np.ones(calm), np.full(crash, 0.3),
                                                                              np.ones(recovery)])
    idx = pd.bdate_range("2019-01-02", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(lr)), index=idx)
    iv = pd.Series(vol * 1.1, index=idx)                           # implied a bit above realized (the VRP)
    return close, iv


def hedge_study(close: pd.Series, iv: pd.Series, method: str = "none", otm: float = 0.10, tenor_days: int = 63,
                roll_days: int = 21, hedge_ratio: float = 0.5, r: float = 0.04) -> pd.Series:
    """Equity curve of a portfolio holding N = 1_000_000 / close[0] index units, plus a hedge:
      'none'     no hedge
      'puts'     every roll_days (starting day 0): sell the old puts at model value, buy N puts, strike (1 − otm)·S,
                 maturity tenor_days/252 years, priced with bsm_price at iv[t] (the cash pays for them). Rolling BEFORE
                 expiry (roll_days < tenor_days) keeps protection in place: with roll_days == tenor_days the hedge is
                 weakest exactly when it may be needed (try it).
      'collar'   the same puts plus N short calls at zero_cost_call_strike (flat iv[t]) rolled together
      'futures'  short hedge_ratio·N index units from day 0 (daily P&L −hedge_ratio·N·ΔS, no carry)
    Mark options each day at bsm_price with the remaining time (intrinsic at expiry) and iv[t]. No interest on cash.
    Return equity = N·S + cash + value of open hedge legs, indexed like close. Unknown method: ValueError."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def hedge_report(curves: dict[str, pd.Series]) -> pd.DataFrame:
    """Per method: total_return (last/first − 1), max_drawdown (<= 0), and cost_vs_none = total_return − that of
    'none' (negative = the insurance premium you paid, net of what the hedge gave back). Index = method."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
