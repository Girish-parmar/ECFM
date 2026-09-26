"""Week 33 (S1–S4) — Framing financial ML, bars and event sampling, fractional differentiation, microstructure
features and a point-in-time feature store.

Rule of the week: a value may only be used at a time when it was KNOWN. Every feature must pass the truncation test.
Fill in every block marked "Your turn", then run:  python -m pytest week33_features
"""
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.stattools import adfuller


# ------------------------------------------------------------------------- S1 baseline first
def direction_dataset(close: pd.Series, n_lags: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features at t: the last n_lags log returns (columns lag0 = r_t, lag1 = r_{t−1}, …). Target: 1 if the NEXT
    log return r_{t+1} > 0 else 0. Also return fwd = r_{t+1}. Drop rows with any NaN (start and last row)."""
    # >>> SOLUTION
    r = np.log(close).diff()
    X = pd.DataFrame({f"lag{k}": r.shift(k) for k in range(n_lags)})
    fwd = r.shift(-1)
    ok = X.notna().all(axis=1) & fwd.notna()
    return X[ok], (fwd[ok] > 0).astype(int), fwd[ok]
    # <<< SOLUTION


def walk_forward_logit(X: pd.DataFrame, y: pd.Series, fwd: pd.Series, train: int = 500, step: int = 21,
                       cost_bps: float = 5.0) -> dict:
    """Expanding-window walk-forward: at rows train, train+step, …, fit LogisticRegression() on ALL rows before and
    predict the next `step` rows. Position = +1 if p(up) > 0.5 else −1. Net return = position·fwd −
    cost_bps/1e4·|Δposition| (the first position counts from 0). Return {"accuracy": share of correct direction
    calls, "always_up": share of up days in the same rows, "returns": net return Series (OOS rows only)}."""
    # >>> SOLUTION
    pos = pd.Series(np.nan, index=X.index)
    for start in range(train, len(X), step):
        m = LogisticRegression().fit(X.iloc[:start], y.iloc[:start])
        end = min(start + step, len(X))
        pos.iloc[start:end] = np.where(m.predict_proba(X.iloc[start:end])[:, 1] > 0.5, 1.0, -1.0)
    pos = pos.iloc[train:]
    yy, f = y.iloc[train:], fwd.iloc[train:]
    net = pos * f - cost_bps / 1e4 * pos.diff().fillna(pos).abs()
    return {"accuracy": float(((pos > 0).astype(int) == yy).mean()), "always_up": float(yy.mean()), "returns": net}
    # <<< SOLUTION


# ---------------------------------------------------------------------------- S2 bars & events
def time_bars(ticks: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """OHLCV bars of fixed clock time (given), labeled by their CLOSE time; empty bars dropped."""
    g = ticks.resample(freq, label="right", closed="right")
    bars = g["price"].ohlc()
    bars["volume"] = g["size"].sum()
    return bars.dropna()


def dollar_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Lesson plan S2: a bar closes each time the cumulative traded dollar value (price × size) crosses another
    multiple of `threshold` (bar id = cumulative dollars // threshold). Columns open, high, low, close, volume;
    index = the timestamp of the bar's LAST trade (a bar is known when it closes)."""
    # >>> SOLUTION
    dv = (ticks["price"] * ticks["size"]).cumsum()
    bar_id = (dv // threshold).astype(int).to_numpy()
    g = ticks.groupby(bar_id)
    bars = g["price"].agg(open="first", high="max", low="min", close="last")
    bars["volume"] = g["size"].sum()
    bars.index = pd.DatetimeIndex(ticks.index.to_series().groupby(bar_id).last().to_numpy())
    return bars
    # <<< SOLUTION


def bar_stats(close: pd.Series) -> dict:
    """Of the bar log returns: {"jb": Jarque–Bera statistic (scipy), "ac1": lag-1 autocorrelation, "n": count}."""
    # >>> SOLUTION
    r = np.log(close).diff().dropna()
    return {"jb": float(jarque_bera(r).statistic), "ac1": float(r.autocorr(1)), "n": int(len(r))}
    # <<< SOLUTION


def cusum_events(close: pd.Series, h: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on log returns (lesson plan S2): s+ = max(0, s+ + x), s− = min(0, s− + x); when s+ > h
    record an event and reset s+ to 0; elif s− < −h record and reset s− to 0."""
    # >>> SOLUTION
    events, s_pos, s_neg = [], 0.0, 0.0
    for ts, x in np.log(close).diff().dropna().items():
        s_pos, s_neg = max(0.0, s_pos + x), min(0.0, s_neg + x)
        if s_pos > h:
            s_pos = 0.0
            events.append(ts)
        elif s_neg < -h:
            s_neg = 0.0
            events.append(ts)
    return pd.DatetimeIndex(events)
    # <<< SOLUTION


# ---------------------------------------------------------------- S3 fractional differentiation
def adf_pvalue(x) -> float:
    """ADF p-value with a constant and one lag (given; statsmodels, version-proof)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return float(adfuller(np.asarray(x, dtype=float), maxlag=1, regression="c", autolag=None)[1])


def ffd_weights(d: float, thresh: float = 1e-4) -> np.ndarray:
    """Fixed-width-window weights: w_0 = 1, w_k = −w_{k−1}(d − k + 1)/k, stop at the first |w_k| < thresh.
    Return them OLDEST FIRST (reversed), so that w @ x[t−L+1..t] is the transformed value at t."""
    # >>> SOLUTION
    w, k = [1.0], 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])
    # <<< SOLUTION


def frac_diff(x: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """FFD transform: value at t = ffd_weights(d) @ x[t−L+1..t] (past values only), indexed from x.index[L−1].
    A series shorter than the L weights gives an empty Series (np.convolve would silently swap its arguments)."""
    # >>> SOLUTION
    w = ffd_weights(d, thresh)
    if len(x) < len(w):
        return pd.Series(dtype=float, index=x.index[:0])
    vals = np.convolve(x.to_numpy(), w[::-1], mode="valid")
    return pd.Series(vals, index=x.index[len(w) - 1:])
    # <<< SOLUTION


def min_ffd(x: pd.Series, ds=np.round(np.arange(0.0, 1.01, 0.1), 2), p: float = 0.05, thresh: float = 1e-4
            ) -> tuple[float, pd.DataFrame]:
    """For each d: adf_pvalue of frac_diff(x, d) and
    its correlation with x on the same dates. Return (the smallest d with p-value < p (NaN if none),
    DataFrame indexed by d with columns adf_p, corr)."""
    # >>> SOLUTION
    rows = {}
    for d in ds:
        f = frac_diff(x, d, thresh)
        rows[float(d)] = {"adf_p": adf_pvalue(f),
                          "corr": float(np.corrcoef(f, x.loc[f.index])[0, 1])}
    table = pd.DataFrame(rows).T
    ok = table.index[table["adf_p"] < p]
    return (float(ok.min()) if len(ok) else np.nan), table
    # <<< SOLUTION


# ------------------------------------------------------------------- S3 microstructure features
def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Roll (1984) effective spread from price changes Δp: 2·√(−cov(Δp_t, Δp_{t−1})) over a rolling window (pandas
    rolling cov, ddof=1); 0 where the covariance is >= 0."""
    # >>> SOLUTION
    dp = close.diff()
    c = dp.rolling(window).cov(dp.shift(1))
    return 2 * np.sqrt((-c).clip(lower=0))
    # <<< SOLUTION


def amihud(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Amihud illiquidity: rolling mean of |log return| / dollar volume (close · volume), × 1e9 for readability."""
    # >>> SOLUTION
    return (np.log(close).diff().abs() / (close * volume)).rolling(window).mean() * 1e9
    # <<< SOLUTION


# ------------------------------------------------------------------------ S4 the feature store
class FeatureStore:
    """Point-in-time store: every value is kept with the time it became AVAILABLE, and get() returns exactly what was
    known at `as_of`. (In the platform this sits on DuckDB/Parquet; here a pandas table.)"""

    def __init__(self):
        self.rows = pd.DataFrame(columns=["symbol", "feature", "ts", "available_at", "value"])

    def put(self, symbol: str, feature: str, values: pd.Series, delay: pd.Timedelta = pd.Timedelta(0)) -> None:
        """Store a Series (index = the observation time ts); available_at = ts + delay (e.g. a report published
        later). Append rows; if (symbol, feature, ts) already exists keep the NEW value (a revision)."""
        # >>> SOLUTION
        new = pd.DataFrame({"symbol": symbol, "feature": feature, "ts": values.index,
                            "available_at": values.index + delay, "value": values.to_numpy(dtype=float)})
        rows = new if self.rows.empty else pd.concat([self.rows, new], ignore_index=True)
        self.rows = rows.drop_duplicates(["symbol", "feature", "ts"], keep="last").reset_index(drop=True)
        # <<< SOLUTION

    def get(self, symbols: list[str], features: list[str], as_of) -> pd.DataFrame:
        """DataFrame (index symbols, columns features): for each pair the value with the LATEST ts among the rows
        whose available_at <= as_of; NaN if nothing was known yet."""
        # >>> SOLUTION
        known = self.rows[self.rows["available_at"] <= pd.Timestamp(as_of)]
        last = known.sort_values("ts").groupby(["symbol", "feature"])["value"].last()
        out = pd.DataFrame(np.nan, index=symbols, columns=features)
        for (s, f), v in last.items():
            if s in out.index and f in out.columns:
                out.loc[s, f] = v
        return out
        # <<< SOLUTION


def truncation_test(feature_fn: Callable[[pd.Series], pd.Series], x: pd.Series, points: list[int]) -> list[int]:
    """Part 5 truncation test: for each integer position t in `points`, compute feature_fn on x[:t+1] and on the full
    x; the value at x.index[t] must be the same (np.isclose, NaN == NaN). Return the positions where it differs
    (an empty list = no look-ahead)."""
    # >>> SOLUTION
    full = feature_fn(x)
    bad = []
    for t in points:
        ts = x.index[t]
        a, b = feature_fn(x.iloc[:t + 1]).get(ts, np.nan), full.get(ts, np.nan)
        if not np.isclose(a, b, equal_nan=True):
            bad.append(t)
    return bad
    # <<< SOLUTION


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank of each value across the columns of its row (pandas rank(axis=1, pct=True))."""
    # >>> SOLUTION
    return df.rank(axis=1, pct=True)
    # <<< SOLUTION
