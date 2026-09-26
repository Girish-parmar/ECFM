"""Week 11 — Scientific stack and data layer (Part 3, S17–S20): NumPy, pandas, Polars, Parquet, DuckDB.

Fill in every block marked "Your turn", then run:  python -m pytest week11_data
Bars follow the course schema: a UTC DatetimeIndex (or `ts` column) holding the bar OPEN time.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


# ------------------------------------------------------------------------------ S17 NumPy
def rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """z[t] = (x[t] − mean(x[t−window+1..t])) / std(same window, ddof=0); NaN for the first window−1.
    Vectorized: no Python loop (hint: np.lib.stride_tricks.sliding_window_view)."""
    # >>> SOLUTION
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    w = np.lib.stride_tricks.sliding_window_view(x, window)
    out[window - 1:] = (x[window - 1:] - w.mean(axis=1)) / w.std(axis=1)
    return out
    # <<< SOLUTION


def demean_columns(returns: np.ndarray) -> np.ndarray:
    """Subtract each column's mean from a T × N array using broadcasting (no loop)."""
    # >>> SOLUTION
    return returns - returns.mean(axis=0, keepdims=True)
    # <<< SOLUTION


# ----------------------------------------------------------------------------- S18 pandas
def resample_ohlcv(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars (index = bar open time, UTC) to `rule` (e.g. '1h'), labelled by the OPEN of
    each new bar. Drop empty periods."""
    # >>> SOLUTION
    return bars.resample(rule, label="left", closed="left").agg(OHLCV_AGG).dropna(subset=["open"])
    # <<< SOLUTION


def last_quote_before(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """For each trade (columns ts, price) attach the most recent quote (columns ts, bid, ask) at or
    before the trade time — never a later one. Return trades with bid/ask columns, sorted by ts."""
    # >>> SOLUTION
    return pd.merge_asof(trades.sort_values("ts"), quotes.sort_values("ts"), on="ts", direction="backward")
    # <<< SOLUTION


def validate_bars(bars: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty if clean). Check, in this order, and add the message if violated:
    'index not UTC', 'index not increasing', 'duplicate timestamps', 'high below open/close',
    'low above open/close', 'negative volume'."""
    # >>> SOLUTION
    problems = []
    idx = bars.index
    if getattr(idx, "tz", None) is None or str(idx.tz) != "UTC":
        problems.append("index not UTC")
    if not idx.is_monotonic_increasing:
        problems.append("index not increasing")
    if idx.has_duplicates:
        problems.append("duplicate timestamps")
    if (bars["high"] < bars[["open", "close"]].max(axis=1)).any():
        problems.append("high below open/close")
    if (bars["low"] > bars[["open", "close"]].min(axis=1)).any():
        problems.append("low above open/close")
    if (bars["volume"] < 0).any():
        problems.append("negative volume")
    return problems
    # <<< SOLUTION


def find_gaps(bars: pd.DataFrame, freq: str = "1min", session_start: str = "14:30", session_end: str = "21:00") -> pd.DatetimeIndex:
    """Missing bar timestamps within each trading day's session [session_start, session_end) in UTC,
    for days that have at least one bar."""
    # >>> SOLUTION
    missing = []
    for day in pd.Index(bars.index.normalize().unique()):
        expected = pd.date_range(f"{day.date()} {session_start}", f"{day.date()} {session_end}",
                                 freq=freq, inclusive="left", tz="UTC")
        missing.append(expected.difference(bars.index))
    return missing[0].append(missing[1:]) if missing else pd.DatetimeIndex([], tz="UTC")
    # <<< SOLUTION


# -------------------------------------------------------------- S20 Parquet + DuckDB + Polars
def write_partitioned(bars: pd.DataFrame, root: str | Path, symbol: str) -> None:
    """Write bars (UTC index) to hive-style Parquet: root/symbol=<SYMBOL>/date=<YYYY-MM-DD>/*.parquet,
    with a `ts` column. (Hint: polars `write_parquet(..., partition_by=[...])`.)"""
    # >>> SOLUTION
    df = bars.reset_index(names="ts")
    df["symbol"] = symbol
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    pl.from_pandas(df).write_parquet(str(root), partition_by=["symbol", "date"])
    # <<< SOLUTION


def daily_summary(root: str | Path) -> pd.DataFrame:
    """With DuckDB over root/**/*.parquet (hive partitioning), return one row per (symbol, date) with
    columns symbol, date, bars, high, low, volume — sorted by symbol, date."""
    # >>> SOLUTION
    return duckdb.sql(f"""
        SELECT symbol, date, count(*) AS bars, max(high) AS high, min(low) AS low, sum(volume) AS volume
        FROM read_parquet('{Path(root).as_posix()}/**/*.parquet', hive_partitioning = true)
        GROUP BY symbol, date ORDER BY symbol, date
    """).df()
    # <<< SOLUTION


def sma_lazy(root: str | Path, n: int) -> pl.DataFrame:
    """Polars LAZY scan of the Parquet files; add column f'sma{n}' = rolling mean of close over n bars
    per symbol (ordered by ts); collect and return sorted by symbol, ts."""
    # >>> SOLUTION
    return (pl.scan_parquet(f"{Path(root).as_posix()}/**/*.parquet", hive_partitioning=True)
              .sort(["symbol", "ts"])
              .with_columns(pl.col("close").rolling_mean(window_size=n).over("symbol").alias(f"sma{n}"))
              .collect())
    # <<< SOLUTION
