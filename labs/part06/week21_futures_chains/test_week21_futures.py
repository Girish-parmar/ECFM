from datetime import date

import numpy as np
import pandas as pd
import pytest
from _loader import load

fc = load("week21_futures_chains", "futures_chains")


# ------------------------------------------------------------------------- roll calendar
def test_expiries_and_codes():
    assert fc.third_friday(2026, 12) == date(2026, 12, 18) and fc.third_friday(2026, 5) == date(2026, 5, 15)
    ex = fc.quarterly_expiries(date(2026, 1, 1), date(2026, 12, 31))
    assert ex == [date(2026, 3, 20), date(2026, 6, 19), date(2026, 9, 18), date(2026, 12, 18)]
    assert [fc.contract_code("ES", e) for e in ex] == ["ESH6", "ESM6", "ESU6", "ESZ6"]


def test_roll_calendar():
    ex = fc.quarterly_expiries(date(2026, 1, 1), date(2026, 12, 31))
    assert fc.roll_date(date(2026, 3, 20), 8) == date(2026, 3, 10)
    assert fc.active_contract(date(2026, 3, 9), ex) == date(2026, 3, 20)
    assert fc.active_contract(date(2026, 3, 10), ex) == date(2026, 6, 19)      # roll day: already in June
    with pytest.raises(ValueError):
        fc.active_contract(date(2026, 12, 20), ex)


def test_volume_roll_dates():
    idx = pd.date_range("2026-03-02", periods=6, freq="B")
    vol = pd.DataFrame({"H": [900, 800, 600, 300, 100, 50], "M": [100, 200, 500, 700, 900, 950],
                        "U": [0, 0, 0, 0, 10, 20]}, index=idx)
    assert fc.volume_roll_dates(vol) == [idx[3]]                               # M never loses to U here


# --------------------------------------------------------------------- continuous series
@pytest.fixture
def two_contracts():
    idx = pd.date_range("2026-03-02", periods=6, freq="B")
    return pd.DataFrame({"H": [100.0, 101, 102, 103, 104, np.nan], "M": [np.nan, 103, 104, 105, 106, 107]},
                        index=idx), [idx[3]]


def test_continuous_series(two_contracts):
    px, rolls = two_contracts
    np.testing.assert_allclose(fc.continuous(px, rolls, "none"), [100, 101, 102, 105, 106, 107])
    np.testing.assert_allclose(fc.continuous(px, rolls, "difference"), [102, 103, 104, 105, 106, 107])
    np.testing.assert_allclose(fc.continuous(px, rolls, "ratio"), np.array([100, 101, 102, 105, 106, 107]) *
                               np.array([105 / 103] * 3 + [1] * 3))
    with pytest.raises(ValueError):
        fc.continuous(px, rolls, "panama")


def test_adjustments_preserve_what_they_promise():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-02", periods=300, freq="B")
    spot = 5000 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    px = pd.DataFrame({f"C{i}": spot * np.exp(0.03 * (i + 1) * 0.25) for i in range(3)}, index=idx)
    rolls = [idx[100], idx[200]]
    raw, diff, ratio = (fc.continuous(px, rolls, m) for m in ("none", "difference", "ratio"))
    for i, d in enumerate(rolls):                                              # on a roll day the adjusted series moves
        old = px[f"C{i}"]                                                      # exactly like the contract it leaves
        assert diff.diff().loc[d] == pytest.approx(old.diff().loc[d])
        assert ratio.pct_change().loc[d] == pytest.approx(old.pct_change().loc[d])
        assert raw.diff().loc[d] - diff.diff().loc[d] == pytest.approx(px[f"C{i + 1}"].loc[d] - old.loc[d])  # the gap
    held = pd.Series(np.searchsorted(pd.DatetimeIndex(rolls), idx, side="right"), index=idx)
    same = held.diff().fillna(0) == 0
    np.testing.assert_allclose(diff.diff()[same].dropna(), raw.diff()[same].dropna())       # point changes kept
    np.testing.assert_allclose(ratio.pct_change()[same].dropna(), raw.pct_change()[same].dropna())   # % kept
    np.testing.assert_allclose(diff.iloc[-100:], raw.iloc[-100:])             # the latest contract is untouched


def test_carry():
    F = fc.fair_value(5000.0, 0.045, 0.013, 0.25)
    assert F == pytest.approx(5000 * np.exp(0.032 * 0.25))
    assert fc.implied_carry(F, fc.fair_value(5000.0, 0.045, 0.013, 0.5), 0.25, 0.5) == pytest.approx(0.032)


# --------------------------------------------------------------------------- symbology
def test_occ_round_trip():
    s = fc.occ_symbol("AAPL", date(2026, 12, 18), "C", 200)
    assert s == "AAPL261218C00200000"
    assert fc.occ_symbol("SPY", date(2026, 4, 17), "P", 512.5) == "SPY260417P00512500"
    assert fc.parse_occ("SPY260417P00512500") == {"root": "SPY", "expiry": date(2026, 4, 17), "cp": -1,
                                                  "strike": 512.5}
    with pytest.raises(ValueError):
        fc.parse_occ("SPY 260417P512")


# --------------------------------------------------------------------- chain normalization
def test_chain_from_ib_and_alpaca_share_one_schema():
    ib = fc.chain_from_ib([
        {"lastTradeDateOrContractMonth": "20260417", "strike": 500.0, "right": "C", "bid": 10.1, "ask": 10.3,
         "impliedVol": 0.17, "delta": 0.52},
        {"lastTradeDateOrContractMonth": "20260417", "strike": 500.0, "right": "P", "bid": -1, "ask": 9.2,
         "impliedVol": None, "delta": None}])
    assert list(ib.columns) == fc.CHAIN_COLUMNS and ib["expiry"].iloc[0] == date(2026, 4, 17)
    assert ib["mid"].iloc[0] == pytest.approx(10.2) and np.isnan(ib["bid"].iloc[1]) and np.isnan(ib["mid"].iloc[1])
    assert ib["cp"].tolist() == [1, -1] and np.isnan(ib["delta"].iloc[1]) and (ib["source"] == "ib").all()
    al = fc.chain_from_alpaca({
        "SPY260417P00500000": {"latest_quote": {"bid_price": 9.0, "ask_price": 9.2}, "implied_volatility": 0.18,
                               "greeks": {"delta": -0.47}},
        "SPY260417C00500000": {"latest_quote": {"bid_price": 10.0, "ask_price": 10.4}, "implied_volatility": 0.17,
                               "greeks": None}})
    assert list(al.columns) == fc.CHAIN_COLUMNS and al["cp"].tolist() == [-1, 1]        # sorted: put first
    assert al["delta"].iloc[0] == -0.47 and np.isnan(al["delta"].iloc[1]) and al["mid"].iloc[1] == pytest.approx(10.2)


# --------------------------------------------------------------------------------- selection
def test_liquidity_filter():
    c = pd.DataFrame({"bid": [0.0, 1.0, 1.0, 2.0], "ask": [0.1, 1.05, 1.5, 2.1], "oi": [10, 500, 500, 5]})
    c["mid"] = (c["bid"] + c["ask"]) / 2
    out = fc.liquidity_filter(c, max_spread_pct=0.10, min_oi=100)
    assert out.index.tolist() == [1] and out["spread_pct"].iloc[0] == pytest.approx(0.05 / 1.025)


def test_expiry_and_strike_selection():
    today = date(2026, 3, 2)
    ex = [date(2026, 3, 27), date(2026, 4, 2), date(2026, 4, 10), date(2026, 4, 17), date(2026, 5, 15)]
    assert fc.select_expiry(ex, today, 25, 50) == date(2026, 4, 17)                    # the monthly
    assert fc.select_expiry(ex, today, 25, 50, prefer="nearest") == date(2026, 3, 27)
    with pytest.raises(ValueError):
        fc.select_expiry(ex, today, 100, 200)
    strikes = np.arange(480, 521, 5.0)
    assert fc.atm_strike(strikes, 502.4) == 500 and fc.atm_strike(strikes, 502.5) == 500   # tie -> lower
    assert fc.atm_strike(strikes, 503.0) == 505
    chain = pd.DataFrame({"strike": strikes, "delta": np.linspace(-0.05, -0.85, strikes.size)})
    assert fc.strike_by_delta(chain, -0.25)["strike"] == 490
    wide = np.arange(450, 551, 5.0)                                            # ±1 move = 500 ± 28.67
    assert fc.expected_move_strikes(wide, 500.0, 0.20, 30 / 365) == (470.0, 530.0)
