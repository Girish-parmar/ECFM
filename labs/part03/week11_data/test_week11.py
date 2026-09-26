import numpy as np
import pandas as pd
import pytest
from _loader import load

dl = load("week11_data", "data_lab")


def minute_bars(day="2025-03-03", n=390, drop=()):
    idx = pd.date_range(f"{day} 14:30", periods=n, freq="1min", tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.05, n))
    df = pd.DataFrame({"open": close - 0.01, "close": close, "volume": 100.0}, index=idx)
    df["high"], df["low"] = df[["open", "close"]].max(axis=1) + 0.02, df[["open", "close"]].min(axis=1) - 0.02
    return df.drop(idx[list(drop)])[["open", "high", "low", "close", "volume"]]


def test_rolling_zscore_and_demean():
    x = np.arange(10, dtype=float)
    z = dl.rolling_zscore(x, 5)
    assert np.isnan(z[:4]).all() and z[4:] == pytest.approx(np.full(6, 2 / np.std([0, 1, 2, 3, 4])))
    r = np.random.default_rng(1).normal(size=(50, 3)) + [1, 2, 3]
    assert np.allclose(dl.demean_columns(r).mean(axis=0), 0)


def test_resample_ohlcv():
    bars = minute_bars()
    h = dl.resample_ohlcv(bars, "1h")
    assert h.index[0] == pd.Timestamp("2025-03-03 14:00", tz="UTC")       # 14:30-14:59 falls in the 14:00 bucket
    assert h["volume"].iloc[0] == 30 * 100 and h["open"].iloc[0] == bars["open"].iloc[0]
    assert h["close"].iloc[-1] == bars["close"].iloc[-1] and len(h) == 7


def test_last_quote_before_no_lookahead():
    q = pd.DataFrame({"ts": pd.to_datetime(["2025-03-03 14:30:00", "2025-03-03 14:30:10"], utc=True),
                      "bid": [99.9, 100.0], "ask": [100.1, 100.2]})
    t = pd.DataFrame({"ts": pd.to_datetime(["2025-03-03 14:30:09", "2025-03-03 14:30:10", "2025-03-03 14:29:59"], utc=True),
                      "price": [100.1, 100.2, 99.0]})
    j = dl.last_quote_before(t, q)
    assert list(j["ts"]) == sorted(t["ts"])
    assert np.isnan(j["bid"].iloc[0]) and j["bid"].iloc[1] == 99.9 and j["bid"].iloc[2] == 100.0


def test_validate_bars():
    good = minute_bars(n=10)
    assert dl.validate_bars(good) == []
    bad = good.copy(); bad.iloc[3, bad.columns.get_loc("high")] = 0
    bad.iloc[4, bad.columns.get_loc("volume")] = -1
    assert dl.validate_bars(bad) == ["high below open/close", "negative volume"]
    naive = good.tz_localize(None)
    assert dl.validate_bars(naive)[0] == "index not UTC"
    dup = pd.concat([good.iloc[:3], good.iloc[2:3]])
    assert "duplicate timestamps" in dl.validate_bars(dup) and "index not increasing" in dl.validate_bars(dup.iloc[::-1])


def test_find_gaps():
    bars = minute_bars(drop=(5, 6, 100))
    gaps = dl.find_gaps(bars)
    assert list(gaps) == list(bars.index[:0].append(pd.DatetimeIndex(
        [pd.Timestamp("2025-03-03 14:35", tz="UTC"), pd.Timestamp("2025-03-03 14:36", tz="UTC"),
         pd.Timestamp("2025-03-03 16:10", tz="UTC")])))


def test_parquet_duckdb_polars(tmp_path):
    root = tmp_path / "bars"
    for sym, day in [("SPY", "2025-03-03"), ("SPY", "2025-03-04"), ("QQQ", "2025-03-03")]:
        dl.write_partitioned(minute_bars(day=day), root, sym)
    assert (root / "symbol=SPY" / "date=2025-03-04").exists()
    s = dl.daily_summary(root)
    assert list(s.columns) == ["symbol", "date", "bars", "high", "low", "volume"]
    assert list(s["symbol"]) == ["QQQ", "SPY", "SPY"] and (s["bars"] == 390).all()
    sma = dl.sma_lazy(root, 20)
    assert sma.height == 3 * 390 and sma["sma20"].null_count() == 2 * 19     # one warm-up per symbol
