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
    # >>> SOLUTION
    return straddle_price / spot
    # <<< SOLUTION


def straddle_rule_of_thumb(S: float, sigma: float, T: float) -> float:
    """ATM straddle ≈ √(2/π)·S·σ·√T ≈ 0.8·S·σ·√T (use √(2/π) exactly)."""
    # >>> SOLUTION
    return float(np.sqrt(2 / np.pi) * S * sigma * np.sqrt(T))
    # <<< SOLUTION


def event_study(implied: np.ndarray, realized: np.ndarray) -> dict:
    """Implied vs realized absolute moves over past events (e.g. earnings), both as fractions.
    Return {"n", "mean_implied", "mean_realized", "edge" = mean(realized − implied) (a long straddle's rough edge per
    unit of spot), "share_realized_above" = share of events where realized > implied}."""
    # >>> SOLUTION
    implied, realized = np.asarray(implied, dtype=float), np.asarray(realized, dtype=float)
    return {"n": int(implied.size), "mean_implied": float(implied.mean()), "mean_realized": float(realized.mean()),
            "edge": float(np.mean(realized - implied)), "share_realized_above": float(np.mean(realized > implied))}
    # <<< SOLUTION


def realized_vol(close: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling std of daily log returns over the last `window` days (ddof=1), known at each close."""
    # >>> SOLUTION
    return np.log(close).diff().rolling(window).std() * np.sqrt(252)
    # <<< SOLUTION


def vrp_research(iv_30d: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """RESEARCH ONLY: implied vol today minus the realized vol over the NEXT `window` days (realized_vol shifted by
    −window). It uses future data by design: fine for measuring the variance risk premium, never a trading signal."""
    # >>> SOLUTION
    return iv_30d - realized_vol(close, window).shift(-window)
    # <<< SOLUTION


def vrp_signal(iv_30d: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """TRADABLE version: implied vol today minus TRAILING realized vol (known at the close)."""
    # >>> SOLUTION
    return iv_30d - realized_vol(close, window)
    # <<< SOLUTION


# ------------------------------------------------------------------------------ S8 hedging
def futures_hedge_contracts(portfolio_value: float, beta: float, fut_price: float, multiplier: float,
                            hedge_ratio: float = 1.0) -> int:
    """Index futures to SELL: round(hedge_ratio × β × value / (futures price × multiplier))."""
    # >>> SOLUTION
    return int(round(hedge_ratio * beta * portfolio_value / (fut_price * multiplier)))
    # <<< SOLUTION


def zero_cost_call_strike(S: float, T: float, k_put: float, iv_fn, r: float = 0.04, q: float = 0.0) -> float:
    """Call strike (above S) whose premium equals the put's premium at k_put: solve with brentq on [S, 3S]
    (each strike priced with iv_fn(K)). This is a zero-cost collar: the call you sell pays for the put you buy."""
    # >>> SOLUTION
    put = bsm_price(S, k_put, T, r, q, iv_fn(k_put), -1)
    return float(brentq(lambda k: bsm_price(S, k, T, r, q, iv_fn(k), 1) - put, S, 3 * S, xtol=1e-8))
    # <<< SOLUTION


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
    # >>> SOLUTION
    if method not in ("none", "puts", "collar", "futures"):
        raise ValueError(f"unknown method {method!r}")
    S, vol = close.to_numpy(), iv.to_numpy()
    N = 1_000_000 / S[0]
    cash, legs, eq = 0.0, [], np.empty(S.size)            # legs: (cp, K, expiry_day, qty)

    def leg_value(t, legs):
        v = 0.0
        for cp, K, exp_day, qty in legs:
            tau = (exp_day - t) / 252
            v += qty * (bsm_price(S[t], K, tau, r, 0.0, vol[t], cp) if tau > 1e-9 else max(cp * (S[t] - K), 0.0))
        return v

    for t in range(S.size):
        if method in ("puts", "collar") and t % roll_days == 0:
            cash += leg_value(t, legs)                     # sell what is left of the old hedge
            kp = (1 - otm) * S[t]
            legs = [(-1, kp, t + tenor_days, N)]
            if method == "collar":
                kc = zero_cost_call_strike(S[t], tenor_days / 252, kp, lambda k, v=vol[t]: v, r)
                legs.append((1, kc, t + tenor_days, -N))
            cash -= leg_value(t, legs)                     # pay for the new hedge (≈ 0 for a zero-cost collar)
        if method == "futures" and t > 0:
            cash -= hedge_ratio * N * (S[t] - S[t - 1])
        eq[t] = N * S[t] + cash + leg_value(t, legs)
    return pd.Series(eq, index=close.index, name=method)
    # <<< SOLUTION


def hedge_report(curves: dict[str, pd.Series]) -> pd.DataFrame:
    """Per method: total_return (last/first − 1), max_drawdown (<= 0), and cost_vs_none = total_return − that of
    'none' (negative = the insurance premium you paid, net of what the hedge gave back). Index = method."""
    # >>> SOLUTION
    rows = {}
    for m, eq in curves.items():
        dd = (eq / eq.cummax() - 1).min()
        rows[m] = {"total_return": eq.iloc[-1] / eq.iloc[0] - 1, "max_drawdown": dd}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["cost_vs_none"] = df["total_return"] - df.loc["none", "total_return"]
    return df
    # <<< SOLUTION
