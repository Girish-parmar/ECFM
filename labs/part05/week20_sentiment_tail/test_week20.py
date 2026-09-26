import numpy as np
import pandas as pd
import pytest
from _loader import load

st = load("week20_sentiment_tail", "sentiment_tail")


# --------------------------------------------------------------------------- market sentiment
def test_rolling_percentile_and_term_structure():
    x = np.array([10, 20, 15, 30, 5], dtype=float)
    np.testing.assert_allclose(st.rolling_percentile(x, 3), [np.nan, np.nan, 2 / 3, 1.0, 1 / 3], equal_nan=True)
    ratio, inv = st.term_structure_stress([15, 30, 20], [18, 25, 20])
    np.testing.assert_allclose(ratio, [15 / 18, 1.2, 1.0])
    np.testing.assert_array_equal(inv, [False, True, False])


def test_regime_hysteresis():
    s = np.array([0, 1.2, 0.8, 0.6, 0.4, np.nan, -0.9, -1.1, -0.7, -0.4, 1.5, 0.2])
    np.testing.assert_array_equal(st.regime_with_hysteresis(s, 1.0, 0.5),
                                  [0, 1, 1, 1, 0, 0, 0, -1, -1, 0, 1, 0])
    wobble = np.tile([0.95, 1.05], 20)                              # hovering around the threshold
    labels = st.regime_with_hysteresis(wobble, 1.0, 0.5)
    assert (np.diff(labels) != 0).sum() == 1                        # one switch, no daily flipping


def test_mcclellan():
    adv, dec = np.array([300, 200, 400.0]), np.array([200, 300, 100.0])
    net = adv - dec
    e10 = [100, 100 + 0.1 * (-100 - 100), 0]
    e10[2] = e10[1] + 0.1 * (300 - e10[1])
    e5 = [100, 100 + 0.05 * (-200), 0]
    e5[2] = e5[1] + 0.05 * (300 - e5[1])
    np.testing.assert_allclose(st.mcclellan(adv, dec), np.array(e10) - np.array(e5))
    assert net[0] == 100


def test_pct_above_ma():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    closes = pd.DataFrame({"A": [1, 2, 3, 4.0], "B": [4, 3, 2, 1.0], "C": [np.nan, np.nan, 5, 6.0]}, index=idx)
    out = st.pct_above_ma(closes, 2)
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 0.5 and out.iloc[3] == pytest.approx(2 / 3)


def test_cot_point_in_time_join():
    rep = pd.Series(pd.to_datetime(["2026-01-06", "2026-01-13"]))          # Tuesdays
    avail = st.cot_available_at(rep)
    assert avail.iloc[0] == pd.Timestamp("2026-01-09 20:30", tz="UTC")     # Fri 15:30 EST
    releases = pd.DataFrame({"available_at": avail, "net_long": [10.0, 20.0]})
    bars = pd.DataFrame({"ts": pd.to_datetime(["2026-01-07 21:00", "2026-01-09 21:00", "2026-01-14 21:00",
                                               "2026-01-16 21:00"], utc=True)})
    out = st.pit_join(bars, releases, ["net_long"])
    assert np.isnan(out["net_long"].iloc[0])                               # Tuesday's data NOT yet public on Wed
    np.testing.assert_array_equal(out["net_long"].to_numpy()[1:], [10, 10, 20])
    assert list(out.columns) == ["ts", "net_long"]


# --------------------------------------------------------------------------------- news
POS, NEG = {"beat", "strong", "upgrade", "good", "record"}, {"miss", "weak", "downgrade", "lawsuit", "bad"}


@pytest.mark.parametrize("text, score", [
    ("Apple beats? No: Apple BEAT estimates on strong iPhone sales", 1.0),
    ("Analyst downgrade after weak guidance", -1.0),
    ("Results were not good, lawsuit looms", -1.0),
    ("Record quarter but a lawsuit", 0.0),
    ("Company holds annual meeting", 0.0),
    ("No downgrade expected", 1.0),
    ("Not a miss. Strong demand", 1.0),                                  # negation stops at the full stop
])
def test_lexicon_score(text, score):
    assert st.lexicon_score(text, POS, NEG) == score


def test_dedupe_news():
    news = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-05 13:05", "2026-01-05 13:00", "2026-01-05 13:02", "2026-01-05 14:00"], utc=True),
        "symbol": ["AAPL", "AAPL", "MSFT", "AAPL"],
        "headline": ["Apple beats  estimates!", "APPLE beats estimates", "Apple beats estimates", "Apple beats estimates."]})
    out = st.dedupe_news(news)
    assert len(out) == 2 and out["ts"].iloc[0] == pd.Timestamp("2026-01-05 13:00", tz="UTC")
    assert set(out["symbol"]) == {"AAPL", "MSFT"}


def test_decay_index_has_no_lookahead():
    ev = np.array(["2026-01-05T12:00", "2026-01-05T20:00"], dtype="datetime64[ns]")
    sc = np.array([1.0, -1.0])
    bars = np.array(["2026-01-05T11:00", "2026-01-05T12:00", "2026-01-06T12:00", "2026-01-05T19:59"],
                    dtype="datetime64[ns]")
    out = st.decay_index(ev, sc, bars, tau_hours=24)
    assert out[0] == 0.0 and out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(np.exp(-1) - np.exp(-16 / 24))
    assert out[3] == pytest.approx(np.exp(-(7 + 59 / 60) / 24))          # the 20:00 story is not known yet


# ------------------------------------------------------------------------------ tail risk
@pytest.fixture(scope="module")
def t3():
    rng = np.random.default_rng(42)
    return rng.standard_t(3, 20000) * 0.01


def test_hill_and_evt_recover_student_t_tail(t3):
    assert st.hill(t3, k=400) == pytest.approx(1 / 3, abs=0.08)
    r = st.evt_var_es(t3, 0.99)
    assert r["xi"] == pytest.approx(1 / 3, abs=0.12)
    assert r["VaR"] == pytest.approx(np.quantile(-t3, 0.99), rel=0.03)
    assert r["ES"] > r["VaR"] > r["u"] > 0


def test_kupiec():
    lr, p = st.kupiec_pof(2, 250, 0.01)
    assert lr == pytest.approx(0.1084, abs=1e-3) and p > 0.5              # 2 exceptions in 250 days at 1%: fine
    lr, p = st.kupiec_pof(10, 250, 0.01)
    assert p < 0.001                                                      # 10 exceptions: VaR too low
    lr0, p0 = st.kupiec_pof(0, 250, 0.01)
    assert lr0 == pytest.approx(-2 * 250 * np.log(0.99)) and 0 < p0 < 1


def test_scenario_pnl():
    out = st.scenario_pnl({"SPY": (100_000, 1.0), "TSLA": (20_000, 2.0), "GLD": (30_000, -0.1)}, -0.10)
    assert out == pytest.approx({"SPY": -10_000, "TSLA": -4_000, "GLD": 300, "TOTAL": -13_700})
