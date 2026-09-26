import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import bsm_price

vh = load("week24_vol_hedging", "vol_hedging")


def test_implied_move_and_rule_of_thumb():
    S, sig, T = 100.0, 0.30, 7 / 365
    straddle = bsm_price(S, S, T, 0.0, 0.0, sig, 1) + bsm_price(S, S, T, 0.0, 0.0, sig, -1)
    assert vh.straddle_rule_of_thumb(S, sig, T) == pytest.approx(straddle, rel=0.01)
    assert vh.implied_move(straddle, S) == pytest.approx(straddle / S)
    ev = vh.event_study(np.array([0.05, 0.04, 0.06]), np.array([0.03, 0.07, 0.02]))
    assert ev == pytest.approx({"n": 3, "mean_implied": 0.05, "mean_realized": 0.04, "edge": -0.01,
                                "share_realized_above": 1 / 3})


def test_vrp_research_uses_the_future_and_the_signal_does_not():
    close, iv = vh.crash_path(0)
    full_r, full_s = vh.vrp_research(iv, close), vh.vrp_signal(iv, close)
    cut = 480                                                     # stop the data just before the crash
    part_r, part_s = vh.vrp_research(iv.iloc[:cut], close.iloc[:cut]), vh.vrp_signal(iv.iloc[:cut], close.iloc[:cut])
    pd.testing.assert_series_equal(part_s, full_s.iloc[:cut])      # tradable: the past never changes
    assert full_r.iloc[cut - 21:cut].notna().all() and part_r.iloc[-21:].isna().all()   # needs 21 future days
    assert full_r.iloc[-21:].isna().all()                         # the last month needs data we do not have
    assert vh.realized_vol(close).iloc[21] == pytest.approx(np.log(close).diff().iloc[1:22].std() * np.sqrt(252))


def test_futures_hedge_and_zero_cost_collar():
    assert vh.futures_hedge_contracts(1_000_000, beta=1.1, fut_price=6000, multiplier=50) == 4   # lesson-plan example
    assert vh.futures_hedge_contracts(1_000_000, 1.1, 6000, 5, hedge_ratio=0.5) == 18              # micro (MES)
    iv = lambda K: 0.20 + 0.25 * (100 - K) / 100                 # noqa: E731  put skew
    kc = vh.zero_cost_call_strike(100, 0.25, 90, iv)
    assert kc > 100
    assert bsm_price(100, kc, 0.25, 0.04, 0, iv(kc), 1) == pytest.approx(bsm_price(100, 90, 0.25, 0.04, 0, iv(90), -1))
    flat = vh.zero_cost_call_strike(100, 0.25, 90, lambda K: 0.20)
    assert kc < flat                                              # skew makes puts dear: you give up more upside


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hedges_reduce_crash_drawdown(seed):
    close, iv = vh.crash_path(seed)
    rep = vh.hedge_report({m: vh.hedge_study(close, iv, m) for m in ("none", "puts", "collar", "futures")})
    assert list(rep.columns) == ["total_return", "max_drawdown", "cost_vs_none"] and rep.loc["none", "cost_vs_none"] == 0
    dd = rep["max_drawdown"]
    assert dd["none"] < dd["puts"] and dd["none"] < dd["collar"] and dd["none"] < dd["futures"]
    assert dd["futures"] / dd["none"] == pytest.approx(0.5, abs=0.1)     # a 50% overlay roughly halves it


def test_insurance_has_a_cost_in_a_rising_market():
    calm = vh.crash_path(3, calm=750, crash=0, recovery=0)
    rep = vh.hedge_report({m: vh.hedge_study(*calm, m) for m in ("none", "puts", "collar", "futures")})
    assert rep.loc["none", "total_return"] > 0.3
    assert (rep.loc[["puts", "collar", "futures"], "cost_vs_none"] < 0).all()


def test_rolling_at_expiry_leaves_a_gap():
    close, iv = vh.crash_path(0)
    monthly = vh.hedge_report({"none": vh.hedge_study(close, iv), "puts": vh.hedge_study(close, iv, "puts")})
    at_expiry = vh.hedge_report({"none": vh.hedge_study(close, iv),
                                 "puts": vh.hedge_study(close, iv, "puts", roll_days=63)})
    assert at_expiry.loc["puts", "max_drawdown"] < monthly.loc["puts", "max_drawdown"] - 0.05
    with pytest.raises(ValueError):
        vh.hedge_study(close, iv, "hope")
