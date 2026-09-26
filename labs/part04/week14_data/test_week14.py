from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import CANONICAL_COLUMNS

md = load("week14_data", "market_data")
UTC = timezone.utc
T0 = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)


def test_ib_chunks():
    end = datetime(2026, 3, 31, tzinfo=UTC)
    plan = md.ib_chunks(end - timedelta(days=25), end, 10)
    assert plan == [(end, "10 D"), (end - timedelta(days=10), "10 D"), (end - timedelta(days=20), "5 D")]
    plan = md.ib_chunks(end - timedelta(days=2, hours=3), end, 10)
    assert plan == [(end, "3 D")]
    assert md.ib_chunks(end, end, 10) == []


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_pacing_identical_request():
    clk = FakeClock()
    g = md.PacingGuard(clk)
    k = ("AAPL", "5 mins", "TRADES", "20260301")
    assert g.wait(k) == 0.0
    g.record(k)
    clk.t += 5
    assert g.wait(k) == pytest.approx(10.0)
    assert g.wait(("AAPL", "5 mins", "TRADES", "20260201")) == 0.0      # different request
    clk.t += 10
    assert g.wait(k) == 0.0


def test_pacing_six_per_two_seconds_and_sixty_per_ten_minutes():
    clk = FakeClock()
    g = md.PacingGuard(clk)
    for i in range(6):
        g.record(("AAPL", i))
        clk.t += 0.1
    assert g.wait(("AAPL", 99)) == pytest.approx(1.4)          # 6 AAPL requests in the last 0.6 s: wait until the oldest is 2 s old
    assert g.wait(("MSFT", 0)) == 0.0                          # other contract unaffected
    clk.t += 5
    for i in range(54):                                        # 60 requests in the window now
        g.record((f"S{i}", 0))
        clk.t += 1
    assert g.wait(("NEW", 0)) == pytest.approx(1000 + 600 - clk.t)


def _ib_frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02 14:35", "2026-03-02 14:30", "2026-03-02 14:30"]).tz_localize("UTC"),
        "open": [1.0850, 1.0840, 1.0840], "high": [1.0860, 1.0851, 1.0851], "low": [1.0845, 1.0838, 1.0838],
        "close": [1.0855, 1.0849, 1.0849], "volume": [-1.0, -1.0, -1.0], "average": [-1.0, -1.0, -1.0],
        "barCount": [-1, -1, -1]})


def test_normalize_ib():
    out = md.normalize_ib(_ib_frame(), "EUR.USD", "5m")
    assert list(out.columns) == CANONICAL_COLUMNS and len(out) == 2
    assert str(out["ts"].dt.tz) == "UTC" and out["ts"].is_monotonic_increasing
    assert out["volume"].isna().all() and out["vwap"].isna().all() and out["trade_count"].isna().all()
    assert (out["source"] == "ib").all() and not out["adjusted"].any()
    naive = _ib_frame().assign(date=lambda d: d["date"].dt.tz_localize(None), volume=[100, 200, 200],
                               average=[1.0, 2.0, 2.0], barCount=[5, 6, 6])
    out = md.normalize_ib(naive, "AAPL", "5m")
    assert str(out["ts"].dt.tz) == "UTC" and out["volume"].tolist() == [200.0, 100.0]


def test_normalize_alpaca():
    idx = pd.MultiIndex.from_tuples(
        [("MSFT", pd.Timestamp("2026-03-02 14:30", tz="UTC")), ("AAPL", pd.Timestamp("2026-03-02 14:35", tz="UTC")),
         ("AAPL", pd.Timestamp("2026-03-02 14:30", tz="UTC"))], names=["symbol", "timestamp"])
    df = pd.DataFrame({"open": [400, 230, 229], "high": [401, 231, 230], "low": [399, 229, 228],
                       "close": [400.5, 230.5, 229.5], "volume": [1000, 2000, 1500], "trade_count": [10, 20, 15],
                       "vwap": [400.2, 230.1, 229.2]}, index=idx)
    out = md.normalize_alpaca(df, "5m")
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out["symbol"].tolist() == ["AAPL", "AAPL", "MSFT"] and out["ts"].iloc[0] < out["ts"].iloc[1]
    assert (out["source"] == "alpaca_iex").all() and out["adjusted"].all()
    assert md.normalize_alpaca(df, "5m", feed="sip", adjusted=False)["source"].iloc[0] == "alpaca_sip"


def test_bar_builder_trades_and_timer():
    bars = []
    b = md.BarBuilder(timedelta(minutes=1), bars.append)
    b.on_trade(T0 + timedelta(seconds=5), 100.0, 10)
    b.on_trade(T0 + timedelta(seconds=30), 101.0, 5)
    b.on_trade(T0 + timedelta(seconds=50), 99.5, 1)
    b.on_trade(T0 + timedelta(seconds=65), 100.2, 2)              # next minute -> emits first bar
    assert len(bars) == 1
    first = bars[0]
    assert (first.start, first.open, first.high, first.low, first.close, first.volume) == (T0, 100.0, 101.0, 99.5, 99.5, 16)
    b.on_trade(T0 + timedelta(seconds=10), 50.0, 1)               # late trade for a closed bar: ignored
    b.on_timer(T0 + timedelta(seconds=110))                       # minute not over yet
    assert len(bars) == 1
    b.on_timer(T0 + timedelta(seconds=120))                       # boundary: emit without a new trade
    assert len(bars) == 2 and bars[1].start == T0 + timedelta(minutes=1) and bars[1].close == 100.2
    b.on_timer(T0 + timedelta(seconds=300))                       # nothing to emit twice
    assert len(bars) == 2


def test_missing_ranges():
    d = lambda h: T0 + timedelta(hours=h)
    assert md.missing_ranges([], d(0), d(10)) == [(d(0), d(10))]
    assert md.missing_ranges([(d(2), d(4)), (d(1), d(3)), (d(6), d(12))], d(0), d(10)) == [(d(0), d(1)), (d(4), d(6))]
    assert md.missing_ranges([(d(-5), d(20))], d(0), d(10)) == []


def _source(symbol, a, b):
    ts = pd.date_range(a, b, freq="5min", inclusive="left")
    px = np.arange(len(ts), dtype=float) + ts.hour.to_numpy()
    return pd.DataFrame({"ts": ts, "symbol": symbol, "open": px, "high": px + 1, "low": px - 1, "close": px,
                         "volume": 100.0, "vwap": px, "trade_count": 1.0, "timeframe": "5m", "source": "fake",
                         "adjusted": False})[CANONICAL_COLUMNS]


def test_bar_cache():
    c = md.BarCache(_source)
    a = c.get("SPY", T0, T0 + timedelta(hours=1))
    assert len(a) == 12 and c.calls == 1
    b = c.get("SPY", T0 + timedelta(minutes=30), T0 + timedelta(minutes=50))
    assert len(b) == 4 and c.calls == 1                                   # full cache hit
    d = c.get("SPY", T0 - timedelta(minutes=30), T0 + timedelta(hours=2))  # two missing pieces
    assert c.calls == 3 and len(d) == 30 and d["ts"].is_monotonic_increasing and not d["ts"].duplicated().any()
    c.get("QQQ", T0, T0 + timedelta(minutes=10))
    assert c.calls == 4
