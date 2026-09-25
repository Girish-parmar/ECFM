"""Tests for p1lib (run from notebooks/part01:  python -m pytest -q tests)."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p1lib as p  # noqa: E402
from make_fixtures import make  # noqa: E402


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory):
    return make(tmp_path_factory.mktemp("fixtures"))


@pytest.fixture
def offline(monkeypatch, fixtures):
    monkeypatch.setenv("P1_FIXTURES", str(fixtures))


# ---------- data & point-in-time ----------
def test_fred_series_offline(offline):
    s = p.fred_series("CPIAUCSL", start="2010-01-01")
    assert s.index.min() >= pd.Timestamp("2010-01-01") and s.name == "CPIAUCSL" and s.notna().all()


def test_vintages_first_release_and_as_of(offline):
    v = p.alfred_vintages("PAYEMS")
    assert v["realtime_end"].max() == pd.Timestamp("2262-04-01")      # "still current" marker parsed
    fr = p.first_release(v)
    obs = pd.Timestamp("2020-06-01")
    assert fr.loc[obs, "published"] == v[v["date"] == obs]["realtime_start"].min()
    # on the first-release date only the first vintage is visible; later, the revised one
    seen_first = p.as_of(v, fr.loc[obs, "published"])
    seen_late = p.as_of(v, "2021-06-30")
    assert seen_first[obs] == fr.loc[obs, "first_value"]
    assert seen_late[obs] == v[v["date"] == obs]["value"].iloc[-1]
    assert obs + pd.DateOffset(months=1) not in seen_first.index     # not yet published then
    rev = p.revisions(v)
    assert (rev["revision"] == rev["latest_value"] - rev["first_value"]).all()


# ---------- macro ----------
def test_regime_quadrants():
    idx = pd.date_range("2020-01-01", periods=8, freq="MS")
    g = pd.Series([1, 2, 3, 4, 3, 2, 1, 0], idx, dtype=float)
    i = pd.Series([4, 3, 2, 1, 2, 3, 4, 5], idx, dtype=float)
    r = p.regime(g, i, lookback=1)
    assert r["regime"].iloc[0].startswith("Goldilocks") and r["regime"].iloc[-1].startswith("Stagflation")


def test_fed_funds_example():
    post = p.fed_funds_post_meeting_rate(95.73, 4.33, 17, 30)
    assert post == pytest.approx(4.1915, abs=1e-4)
    assert -p.move_probability(post, 4.33) == pytest.approx(0.554, abs=1e-3)


def test_bond_return_sign():
    y = pd.Series([4.0, 4.1, 4.0])
    r = p.bond_return_from_yield(y, duration=8.5)
    assert r.iloc[1] == pytest.approx(-0.0085) and r.iloc[2] == pytest.approx(0.0085)


# ---------- futures ----------
def test_futures_fair_value_example():
    assert p.futures_fair_value(6000, 0.043, 0.013, 80) == pytest.approx(6039.58, abs=0.01)


def test_margin_ledger_call():
    led = p.margin_ledger([6000, 5950, 5900, 5850, 5900], entry_price=6000, contracts=1, multiplier=50,
                          initial_margin=20000, maintenance_margin=18000)
    assert led["daily_pnl"].sum() == pytest.approx(-5000)          # 100 points × $50
    # each 50-point drop costs $2,500 and takes the balance to $17,500 < $18,000: top up to $20,000
    assert list(led["margin_call"]) == [0, 2500, 2500, 2500, 0]
    assert (led["balance"] >= 18000).all() and led["balance"].iloc[-1] == pytest.approx(22500)


# ---------- options (same numbers as the verified Excel sheet in the lesson plan) ----------
def test_bsm_matches_excel_sheet():
    r = p.bsm(100, 105, 0.5, 0.04, 0.01, 0.25)
    assert r["call"] == pytest.approx(5.5482, abs=1e-4) and r["put"] == pytest.approx(8.9678, abs=1e-4)
    assert r["delta_call"] == pytest.approx(0.4568, abs=1e-4) and r["gamma"] == pytest.approx(0.02234, abs=1e-5)
    assert r["vega_per_point"] == pytest.approx(0.2792, abs=1e-4)
    assert r["theta_call_per_day"] == pytest.approx(-0.0223, abs=1e-4)
    assert abs(r["parity_error"]) < 1e-12


def test_implied_vol_roundtrip():
    price = p.bsm(100, 95, 0.25, 0.04, 0.0, 0.31)["put"]
    assert p.implied_vol(price, 100, 95, 0.25, 0.04, 0.0, "put") == pytest.approx(0.31, abs=1e-8)


def test_payoffs_and_breakevens():
    prices = np.linspace(80, 130, 501)
    spread = [{"type": "call", "qty": 1, "strike": 105, "premium": 5.5482},
              {"type": "call", "qty": -1, "strike": 115, "premium": 2.6431}]
    pnl = p.payoff_at_expiry(spread, prices)
    assert pnl.min() == pytest.approx(-(5.5482 - 2.6431)) and pnl.max() == pytest.approx(10 - (5.5482 - 2.6431))
    assert p.breakevens(prices, pnl)[0] == pytest.approx(105 + 5.5482 - 2.6431, abs=1e-6)
    covered = [{"type": "stock", "qty": 1, "premium": 100}, {"type": "call", "qty": -1, "strike": 105, "premium": 2}]
    assert p.payoff_at_expiry(covered, np.array([200.0]))[0] == pytest.approx(7)


def test_occ_symbol():
    assert p.occ_symbol("SPY", date(2026, 12, 18), "C", 600) == "SPY261218C00600000"


# ---------- microstructure ----------
def test_walk_the_book():
    bids, asks = p.synthetic_book(mid=100, tick=0.01)
    small = p.walk_the_book(asks, 100, mid=100)
    big = p.walk_the_book(asks, int(asks["size"].sum() * 0.9), mid=100)
    assert small["avg_price"] == pytest.approx(asks["price"].iloc[0]) and big["cost_bps"] > small["cost_bps"]
    with pytest.raises(ValueError):
        p.walk_the_book(asks, int(asks["size"].sum()) + 1, mid=100)


def test_spread_measures_and_roll():
    quotes, trades = p.simulate_trades_quotes(seed=1)
    m = p.spread_measures(trades, quotes)
    # trades happen exactly at bid/ask, so effective spread equals quoted spread
    assert np.allclose(m["effective_bps"], m["quoted_bps"])
    prof = m.groupby(m["ts"].dt.floor("30min"))["quoted_bps"].mean()
    assert prof.iloc[0] > prof.iloc[len(prof) // 2] < prof.iloc[-1]        # U-shape
    rng = np.random.default_rng(0)
    mid = 100 + np.cumsum(rng.normal(0, 0.02, 20000))
    assert p.roll_spread(mid + 0.05 * rng.choice([-1, 1], 20000)) == pytest.approx(0.10, abs=0.01)


def test_research_log(tmp_path):
    f = tmp_path / "log.csv"
    p.log_research({"source": "FRED", "series": "CPIAUCSL"}, f)
    df = p.log_research({"source": "FRED", "series": "UNRATE"}, f)
    assert list(df["series"]) == ["CPIAUCSL", "UNRATE"]
