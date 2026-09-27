"""Helper functions for the Part 10 guided notebooks (MFAAT, Months 9–10: machine learning, deep learning and
reinforcement learning for trading).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

All data is SYNTHETIC with a KNOWN structure (uneven tick activity, a hidden trend/chop regime, an AR(1) series,
GARCH volatility, overlapping labels on pure noise, features with planted importances), so the notebooks can show
what a model is able to learn, and that it learns nothing from noise. The definitions match the graded labs in
labs/part10/, with three substitutions that keep the notebooks light: scikit-learn's HistGradientBoosting stands in
for LightGBM, a logged random search for Optuna, and a small NumPy Gaussian HMM for hmmlearn. The RL environments
follow the Gymnasium API (reset/step) without needing the package. The PyTorch code of the deep-learning notebooks
is in p10dl.py, so the other notebooks run without PyTorch installed.
"""
from __future__ import annotations

import json
import os
import warnings
from collections import deque
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import jarque_bera, ks_2samp, norm, spearmanr
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P10_STRICT") == "1"      # tests: a failed check raises instead of continuing


# --------------------------------------------------------------------------------------
# Plot style and the exercise checker
# --------------------------------------------------------------------------------------
def use_course_style() -> None:
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=PALETTE), "figure.figsize": (9, 4.5),
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#52514e", "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "legend.frameon": False,
    })


def _numeric(x) -> bool:
    return isinstance(x, (int, float, np.number, np.ndarray)) and not isinstance(x, bool)


def _same(got, expected, rtol: float, atol: float) -> bool:
    if isinstance(expected, Decimal):
        return isinstance(got, Decimal) and got == expected            # exact, and a float is not money
    if isinstance(expected, pd.Index):
        return isinstance(got, pd.Index) and got.equals(expected)
    if isinstance(expected, (pd.DataFrame, pd.Series)):
        test = pd.testing.assert_frame_equal if isinstance(expected, pd.DataFrame) else pd.testing.assert_series_equal
        try:
            test(got, expected, check_exact=False, rtol=rtol, atol=atol, check_dtype=False, check_names=False,
                 check_freq=False)
            return True
        except (AssertionError, TypeError, AttributeError):
            return False
    if isinstance(expected, dict):
        return (isinstance(got, dict) and set(got) == set(expected)
                and all(_same(got[k], v, rtol, atol) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)) and expected and all(_numeric(v) and np.ndim(v) == 0 for v in expected):
        expected = np.asarray(expected, dtype=float)
    if _numeric(expected):
        g, e = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
        return g.shape == e.shape and np.allclose(g, e, rtol=rtol, atol=atol, equal_nan=True)
    if isinstance(expected, (list, tuple)):
        got = list(got)
        return len(got) == len(expected) and all(_same(a, b, rtol, atol) for a, b in zip(got, expected))
    return type(got) is type(expected) and got == expected if isinstance(expected, bool) else got == expected


def check(name: str, got, expected, rtol: float = 1e-6, atol: float = 1e-9):
    """Compare your answer with the reference: exact for Decimals, text, dates and objects (recursively inside
    lists and dicts); with a tolerance for floats and arrays. Returns your value if correct, else the reference
    (so the notebook keeps running)."""
    try:
        if got is Ellipsis or (isinstance(got, (tuple, list)) and any(g is Ellipsis for g in got)):
            raise ValueError("not done yet")
        ok = bool(_same(got, expected, rtol, atol))
    except Exception:  # noqa: BLE001 - any failure means "not correct yet"
        ok = False
    if ok:
        print(f"✅ {name}: correct")
        return got
    msg = f"❌ {name}: not matching the reference yet"
    if STRICT:
        raise AssertionError(msg)
    print(msg + " — continuing with the reference answer so the rest of the notebook runs.")
    return expected


def attempt(fn, *args, **kwargs):
    """Run fn; if it raises (for example because a blank `...` is still in it), return Ellipsis instead, so
    p.check reports "not done yet" and the notebook keeps going."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - an unfinished exercise may fail in any way
        return Ellipsis


def sharpe(r) -> float:
    """Annualized Sharpe ratio of daily returns (√252); NaN when the returns never vary (no trades)."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")




# --------------------------------------------------------------------------------------
# Data generators (known structure; the same as labs/part10/common.py)
# --------------------------------------------------------------------------------------
def tick_data(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Trades with UNEVEN activity: each day has its own intensity (quiet and busy days) and a U-shaped intraday
    profile. Every trade moves the log price by the same noise scale, so a fixed-TIME bar's variance follows the
    activity while a fixed-DOLLAR bar's does not. Columns price, size; index = trade timestamps."""
    rng = np.random.default_rng(seed)
    frames, logp = [], np.log(100.0)
    days = pd.bdate_range("2024-01-02", periods=n_days)
    for day in days:
        n = int(rng.lognormal(np.log(4000), 0.6))
        u = np.sort(rng.beta(0.6, 0.6, n))                           # U shape: busy open and close
        ts = day + pd.Timedelta(hours=9.5) + pd.to_timedelta(u * 6.5 * 3600, unit="s")
        steps = rng.standard_t(5, n) * 0.0004
        path = logp + np.cumsum(steps)
        logp = path[-1]
        size = np.round(rng.lognormal(np.log(100), 0.8, n)).clip(1)
        frames.append(pd.DataFrame({"price": np.round(np.exp(path), 4), "size": size}, index=ts))
    return pd.concat(frames)


def random_walk_prices(n: int = 2000, sigma: float = 0.01, seed: int = 0) -> pd.Series:
    """A log-price random walk (no signal at all), as a price Series on business days."""
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, sigma, n))), index=pd.bdate_range("2015-01-02", periods=n))


def regime_market(n: int = 2500, seed: int = 0, drift: float = 0.002, trend_vol: float = 0.007,
                  flip_every: float = 120) -> pd.DataFrame:
    """Daily bars with a HIDDEN two-state regime (persistent, ~100 days per spell).
    trend (regime 1): calm (vol 0.7%), a drift of ±0.2%/day whose sign flips about every 120 days → momentum works.
    chop  (regime 0): volatile (vol 1.5%), no drift, returns mean-revert day to day (AR(1) φ = −0.15) → momentum fails.
    Columns close, volume, regime (the TRUTH: never use it as a feature)."""
    rng = np.random.default_rng(seed)
    regime = np.empty(n, dtype=int)
    regime[0] = 1
    for t in range(1, n):
        regime[t] = regime[t - 1] if rng.random() < 0.99 else 1 - regime[t - 1]
    sign, r = 1.0, np.zeros(n)
    for t in range(1, n):
        if rng.random() < 1 / flip_every:
            sign = -sign
        if regime[t] == 1:
            r[t] = drift * sign + rng.normal(0, trend_vol)
        else:
            r[t] = -0.15 * r[t - 1] + rng.normal(0, 0.015)
    volume = rng.lognormal(np.log(1e6), 0.3, n) * np.where(regime == 1, 1.0, 1.6)
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(r)), "volume": volume.round(), "regime": regime},
                        index=pd.bdate_range("2012-01-02", periods=n))


def ar1(n: int = 3000, phi: float = 0.6, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """x[t] = φ x[t−1] + ε: the best possible correlation between x[t] and a forecast made at t−1 is φ."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def garch_returns(n: int = 3000, omega: float = 2e-6, alpha: float = 0.09, beta: float = 0.89, seed: int = 0
                  ) -> pd.DataFrame:
    """GARCH(1,1) daily returns with Student-t shocks. Columns ret, vol (the TRUE conditional volatility, known at
    the start of the day). Volatility clusters, so it is forecastable; the sign of the return is not."""
    rng = np.random.default_rng(seed)
    var = np.empty(n)
    r = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    shocks = rng.standard_t(6, n) / np.sqrt(6 / 4)
    for t in range(n):
        if t:
            var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        r[t] = np.sqrt(var[t]) * shocks[t]
    return pd.DataFrame({"ret": r, "vol": np.sqrt(var)}, index=pd.bdate_range("2010-01-04", periods=n))


def overlapping_dataset(n: int = 1500, horizon: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """NO signal, but OVERLAPPING labels: from a random walk, features = past 5/20/60-day returns and 20-day
    volatility at t, label = 1 if the return over the NEXT `horizon` days is positive, t1 = the date that label is
    resolved. Neighbouring rows share most of their label window, which is exactly what fools a shuffled k-fold.
    Returns (X, y, t1) aligned on the dates."""
    close = random_walk_prices(n + 80, seed=seed)
    r = np.log(close).diff()
    X = pd.DataFrame({"ret5": r.rolling(5).sum(), "ret20": r.rolling(20).sum(), "ret60": r.rolling(60).sum(),
                      "vol20": r.rolling(20).std()})
    fwd = np.log(close).shift(-horizon) - np.log(close)
    t1 = pd.Series(close.index[np.minimum(np.arange(len(close)) + horizon, len(close) - 1)], index=close.index)
    ok = X.notna().all(axis=1) & fwd.notna()
    return X[ok], (fwd[ok] > 0).astype(int), t1[ok]


def planted_features(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A classification set with KNOWN importances: y depends on `signal` only. `twin` is signal plus a little noise
    (a substitute), `noise_cont` is continuous noise (high cardinality, flatters MDI), `noise_bin` is binary noise.
    Rows are independent (no overlap)."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    X = pd.DataFrame({"signal": s, "twin": s + rng.normal(0, 0.3, n), "noise_cont": rng.normal(size=n),
                      "noise_bin": rng.integers(0, 2, n).astype(float)},
                     index=pd.bdate_range("2010-01-04", periods=n))
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-1.5 * s))).astype(int), index=X.index)
    return X, y


# ------------------------------------------------------------------------- S1 baseline first
def direction_dataset(close: pd.Series, n_lags: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features at t: the last n_lags log returns (lag0 = r_t, lag1 = r_{t−1}, …). Target: 1 if the NEXT log return
    r_{t+1} > 0 else 0. Also return fwd = r_{t+1}. Rows with any NaN dropped."""
    r = np.log(close).diff()
    X = pd.DataFrame({f"lag{k}": r.shift(k) for k in range(n_lags)})
    fwd = r.shift(-1)
    ok = X.notna().all(axis=1) & fwd.notna()
    return X[ok], (fwd[ok] > 0).astype(int), fwd[ok]


def walk_forward_logit(X: pd.DataFrame, y: pd.Series, fwd: pd.Series, train: int = 500, step: int = 21,
                       cost_bps: float = 5.0) -> dict:
    """Expanding-window walk-forward: at rows train, train+step, …, fit LogisticRegression() on ALL rows before and
    predict the next `step` rows. Position = +1 if p(up) > 0.5 else −1; net = position·fwd − cost_bps/1e4·|Δposition|
    (the first position counts from 0). Return {"accuracy", "always_up" (share of up days in the same rows),
    "returns" (net, OOS rows only)}."""
    pos = pd.Series(np.nan, index=X.index)
    for start in range(train, len(X), step):
        m = LogisticRegression().fit(X.iloc[:start], y.iloc[:start])
        end = min(start + step, len(X))
        pos.iloc[start:end] = np.where(m.predict_proba(X.iloc[start:end])[:, 1] > 0.5, 1.0, -1.0)
    pos = pos.iloc[train:]
    yy, f = y.iloc[train:], fwd.iloc[train:]
    net = pos * f - cost_bps / 1e4 * pos.diff().fillna(pos).abs()
    return {"accuracy": float(((pos > 0).astype(int) == yy).mean()), "always_up": float(yy.mean()), "returns": net}


# ---------------------------------------------------------------------------- S2 bars & events
def time_bars(ticks: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """OHLCV bars of fixed clock time, labeled by their CLOSE time; empty bars dropped."""
    g = ticks.resample(freq, label="right", closed="right")
    bars = g["price"].ohlc()
    bars["volume"] = g["size"].sum()
    return bars.dropna()


def dollar_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """A bar closes each time the cumulative traded dollar value crosses another multiple of `threshold` (bar id =
    cumulative dollars // threshold). Columns open, high, low, close, volume; index = the bar's LAST trade time."""
    dv = (ticks["price"] * ticks["size"]).cumsum()
    bar_id = (dv // threshold).astype(int).to_numpy()
    g = ticks.groupby(bar_id)
    bars = g["price"].agg(open="first", high="max", low="min", close="last")
    bars["volume"] = g["size"].sum()
    bars.index = pd.DatetimeIndex(ticks.index.to_series().groupby(bar_id).last().to_numpy())
    return bars


def bar_stats(close: pd.Series) -> dict:
    """Of the bar log returns: {"jb": Jarque–Bera statistic, "ac1": lag-1 autocorrelation, "n": count}."""
    r = np.log(close).diff().dropna()
    return {"jb": float(jarque_bera(r).statistic), "ac1": float(r.autocorr(1)), "n": int(len(r))}


def cusum_events(close: pd.Series, h: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on log returns: s+ = max(0, s+ + x), s− = min(0, s− + x); when s+ > h record an event
    and reset s+ to 0; elif s− < −h record and reset s− to 0."""
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


# ---------------------------------------------------------------- S3 fractional differentiation
def adf_pvalue(x) -> float:
    """ADF p-value with a constant and one lag (statsmodels)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return float(adfuller(np.asarray(x, dtype=float), maxlag=1, regression="c", autolag=None)[1])


def ffd_weights(d: float, thresh: float = 1e-4) -> np.ndarray:
    """w_0 = 1, w_k = −w_{k−1}(d − k + 1)/k, stop at the first |w_k| < thresh. Returned OLDEST FIRST, so that
    w @ x[t−L+1..t] is the transformed value at t."""
    w, k = [1.0], 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff(x: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """FFD transform: value at t = ffd_weights(d) @ x[t−L+1..t], indexed from x.index[L−1] (empty if x is shorter)."""
    w = ffd_weights(d, thresh)
    if len(x) < len(w):
        return pd.Series(dtype=float, index=x.index[:0])
    vals = np.convolve(x.to_numpy(), w[::-1], mode="valid")
    return pd.Series(vals, index=x.index[len(w) - 1:])


def min_ffd(x: pd.Series, ds=np.round(np.arange(0.0, 1.01, 0.1), 2), p: float = 0.05, thresh: float = 1e-4
            ) -> tuple[float, pd.DataFrame]:
    """For each d: ADF p-value of frac_diff(x, d) and its correlation with x on the same dates. Return (the smallest
    d with p-value < p, NaN if none; DataFrame indexed by d with columns adf_p, corr)."""
    rows = {}
    for d in ds:
        f = frac_diff(x, d, thresh)
        rows[float(d)] = {"adf_p": adf_pvalue(f), "corr": float(np.corrcoef(f, x.loc[f.index])[0, 1])}
    table = pd.DataFrame(rows).T
    ok = table.index[table["adf_p"] < p]
    return (float(ok.min()) if len(ok) else np.nan), table


def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Roll (1984) spread from price changes Δp: 2·√(−cov(Δp_t, Δp_{t−1})) over a rolling window; 0 where cov >= 0."""
    dp = close.diff()
    c = dp.rolling(window).cov(dp.shift(1))
    return 2 * np.sqrt((-c).clip(lower=0))


def amihud(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Amihud illiquidity: rolling mean of |log return| / dollar volume, × 1e9 for readability."""
    return (np.log(close).diff().abs() / (close * volume)).rolling(window).mean() * 1e9


# ------------------------------------------------------------------------ S4 the feature store
class FeatureStore:
    """Point-in-time store: every value is kept with the time it became AVAILABLE, and get() returns exactly what was
    known at `as_of`. (In the platform this sits on DuckDB/Parquet; here a pandas table.)"""

    def __init__(self):
        self.rows = pd.DataFrame(columns=["symbol", "feature", "ts", "available_at", "value"])

    def put(self, symbol: str, feature: str, values: pd.Series, delay: pd.Timedelta = pd.Timedelta(0)) -> None:
        """Store a Series (index = observation time ts); available_at = ts + delay. A repeated (symbol, feature, ts)
        keeps the NEW value (a revision)."""
        new = pd.DataFrame({"symbol": symbol, "feature": feature, "ts": values.index,
                            "available_at": values.index + delay, "value": values.to_numpy(dtype=float)})
        rows = new if self.rows.empty else pd.concat([self.rows, new], ignore_index=True)
        self.rows = rows.drop_duplicates(["symbol", "feature", "ts"], keep="last").reset_index(drop=True)

    def get(self, symbols: list[str], features: list[str], as_of) -> pd.DataFrame:
        """Index symbols, columns features: the value with the LATEST ts among rows with available_at <= as_of."""
        known = self.rows[self.rows["available_at"] <= pd.Timestamp(as_of)]
        last = known.sort_values("ts").groupby(["symbol", "feature"])["value"].last()
        out = pd.DataFrame(np.nan, index=symbols, columns=features)
        for (s, f), v in last.items():
            if s in out.index and f in out.columns:
                out.loc[s, f] = v
        return out


def truncation_test(feature_fn: Callable[[pd.Series], pd.Series], x: pd.Series, points: list[int]) -> list[int]:
    """For each position t: feature_fn on x[:t+1] and on the full x must agree at x.index[t] (NaN == NaN). Return the
    positions where they differ (empty = no look-ahead)."""
    full = feature_fn(x)
    bad = []
    for t in points:
        ts = x.index[t]
        a, b = feature_fn(x.iloc[:t + 1]).get(ts, np.nan), full.get(ts, np.nan)
        if not np.isclose(a, b, equal_nan=True):
            bad.append(t)
    return bad


# ----------------------------------------------------------------------- S5 triple barrier
def daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """EWM standard deviation of SIMPLE returns: known at the close of t."""
    return close.pct_change().ewm(span=span).std()


def triple_barrier(close: pd.Series, events, vol: pd.Series, pt: float = 1.0, sl: float = 1.0, max_hold: int = 10
                   ) -> pd.DataFrame:
    """For each event t0 (skipped if it is the last bar or vol is NaN): path = close over the NEXT max_hold bars /
    close[t0] − 1; upper = pt·vol[t0], lower = −sl·vol[t0]. t1 = first bar touching a barrier, else the path's last
    bar. label = +1 upper, −1 lower, 0 vertical. Indexed by t0, columns t1, ret (path at t1), label."""
    out, idx = [], close.index
    for t0 in events:
        i0 = idx.get_loc(t0)
        if i0 + 1 >= len(idx) or np.isnan(vol.loc[t0]):
            continue
        path = close.iloc[i0 + 1: i0 + 1 + max_hold] / close.iloc[i0] - 1
        up, dn = pt * vol.loc[t0], -sl * vol.loc[t0]
        hit_up, hit_dn = path[path >= up].index.min(), path[path <= dn].index.min()
        t1 = min([t for t in (hit_up, hit_dn) if pd.notna(t)], default=path.index[-1])
        label = 1 if t1 == hit_up else -1 if t1 == hit_dn else 0
        out.append((t0, t1, path.loc[t1], label))
    return pd.DataFrame(out, columns=["t0", "t1", "ret", "label"]).set_index("t0")


def num_concurrent(t1: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """For every bar in `index`: how many labels are active (t0 <= bar <= t1; t1 is indexed by t0)."""
    c = pd.Series(0, index=index)
    for t0, end in t1.items():
        c.loc[t0:end] += 1
    return c


def avg_uniqueness(t1: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Per label: the mean over its bars [t0, t1] of 1 / num_concurrent (1 = no overlap)."""
    c = num_concurrent(t1, index)
    return pd.Series({t0: float((1 / c.loc[t0:end]).mean()) for t0, end in t1.items()})


def sequential_bootstrap(t1: pd.Series, index: pd.DatetimeIndex, n_draws: int | None = None, seed: int = 0
                         ) -> list[int]:
    """Draw label positions one at a time with probability proportional to the average uniqueness each label WOULD
    have given the labels drawn so far (its bars' 1/(1 + times already covered))."""
    rng = np.random.default_rng(seed)
    n_draws = len(t1) if n_draws is None else n_draws
    pos = {t: i for i, t in enumerate(index)}
    spans = [np.arange(pos[t0], pos[end] + 1) for t0, end in t1.items()]
    covered = np.zeros(len(index))
    drawn = []
    for _ in range(n_draws):
        u = np.array([np.mean(1 / (1 + covered[s])) for s in spans])
        j = int(rng.choice(len(spans), p=u / u.sum()))
        drawn.append(j)
        covered[spans[j]] += 1
    return drawn


# --------------------------------------------------------------------- S6 meta-labels & sizing
def momentum_side(close: pd.Series, lookback: int = 20) -> pd.Series:
    """The PRIMARY model: sign of the log return over the last `lookback` bars (+1/−1; 0 where undefined)."""
    return np.sign(np.log(close).diff(lookback)).fillna(0.0)


def meta_labels(side: pd.Series, barrier_ret: pd.Series) -> pd.Series:
    """1 if taking the primary model's side would have made money (side · barrier return > 0), else 0."""
    return ((side * barrier_ret) > 0).astype(int)


def bet_size(prob, n_classes: int = 2) -> np.ndarray:
    """z = (p − 1/n)/√(p(1 − p)); size = clip(2Φ(z) − 1, 0, 1)."""
    prob = np.asarray(prob, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (prob - 1 / n_classes) / np.sqrt(prob * (1 - prob))
    return np.clip(2 * norm.cdf(z) - 1, 0, 1)


def discretize(size, step: float = 0.1) -> np.ndarray:
    """Round sizes to multiples of `step`."""
    return np.round(np.asarray(size, dtype=float) / step) * step


# ---------------------------------------------------------------------- S7 supervised models
def ml_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Known at the close of t: ret5, ret20, vol20 (std, ddof=1), vol_ratio (vol20 / 100-day std), er20 (|20-day
    return| / Σ|r| over 20 days), volume_z (log volume minus its 60-day mean, over its 60-day std)."""
    r = np.log(close).diff()
    lv = np.log(volume)
    return pd.DataFrame({"ret5": r.rolling(5).sum(), "ret20": r.rolling(20).sum(), "vol20": r.rolling(20).std(),
                         "vol_ratio": r.rolling(20).std() / r.rolling(100).std(),
                         "er20": r.rolling(20).sum().abs() / r.abs().rolling(20).sum(),
                         "volume_z": (lv - lv.rolling(60).mean()) / lv.rolling(60).std()})


def make_model(kind: str, seed: int = 0):
    """"logit": StandardScaler + LogisticRegression · "rf": RandomForestClassifier(200 trees, min_samples_leaf=20,
    max_samples=0.5, random_state=seed) · "gbm": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03,
    max_leaf_nodes=8, min_samples_leaf=30, random_state=seed) (the labs use LightGBM with the same settings)."""
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression())
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_samples=0.5, random_state=seed,
                                      n_jobs=1)
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=8,
                                              min_samples_leaf=30, random_state=seed)
    raise ValueError(kind)


def fit_weighted(model, X, y, w=None):
    """Fit with sample weights (a Pipeline takes them as <last step>__sample_weight)."""
    if w is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": np.asarray(w)})
    return model.fit(X, y, sample_weight=np.asarray(w))


def calibrate(model, X_cal, y_cal, method: str = "isotonic"):
    """Calibrate an ALREADY FITTED model on a separate, later calibration set (isotonic or sigmoid = Platt)."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_cal, y_cal)


def reliability_table(prob, y, bins: int = 5) -> pd.DataFrame:
    """Equal-width probability bins on [0, 1]; per non-empty bin: mean_pred, frac_pos, count."""
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=float)
    b = np.minimum((prob * bins).astype(int), bins - 1)
    g = pd.DataFrame({"b": b, "p": prob, "y": y}).groupby("b")
    return pd.DataFrame({"mean_pred": g["p"].mean(), "frac_pos": g["y"].mean(), "count": g.size()})


# ------------------------------------------------------------------------ S8 regimes
def gmm_regimes(features: pd.DataFrame, n: int = 2, seed: int = 0) -> pd.Series:
    """GaussianMixture on the standardized features (NaN rows dropped); states relabelled 0..n−1 in ascending order
    of the mean of the FIRST feature within each state."""
    F = features.dropna()
    Z = (F - F.mean()) / F.std()
    lab = GaussianMixture(n, random_state=seed).fit_predict(Z)
    order = np.argsort([F.iloc[:, 0][lab == k].mean() for k in range(n)])
    remap = {int(old): new for new, old in enumerate(order)}
    return pd.Series([remap[int(k)] for k in lab], index=F.index)


def _normal_pdf(x, mu, var):
    return np.exp(-0.5 * (x[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def hmm_fit(x, n: int = 2, n_iter: int = 200, tol: float = 1e-8) -> dict:
    """A 1-D Gaussian HMM fitted by Baum–Welch (scaled forward–backward). Start: the observations sorted by |x| and
    split into n equal groups give the means and variances (state 0 calmest), transitions 0.95 on the diagonal.
    Return {"pi", "A", "mu", "var", "loglik"}."""
    x = np.asarray(x, dtype=float)
    T = x.size
    groups = np.array_split(np.argsort(np.abs(x)), n)
    mu = np.array([x[g].mean() for g in groups])
    var = np.array([x[g].var() + 1e-12 for g in groups])
    A = np.full((n, n), 0.05 / max(n - 1, 1)) + np.eye(n) * (0.95 - 0.05 / max(n - 1, 1))
    pi = np.full(n, 1 / n)
    prev = -np.inf
    for _ in range(n_iter):
        B = _normal_pdf(x, mu, var) + 1e-300
        alpha, c = np.empty((T, n)), np.empty(T)
        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        beta = np.ones((T, n))
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        xi = alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :] / c[1:, None, None]
        pi = gamma[0]
        A = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
        w = gamma / gamma.sum(0)
        mu = (w * x[:, None]).sum(0)
        var = (w * (x[:, None] - mu) ** 2).sum(0) + 1e-12
        loglik = float(np.log(c).sum())
        if loglik - prev < tol:
            break
        prev = loglik
    return {"pi": pi, "A": A, "mu": mu, "var": var, "loglik": loglik}


def hmm_viterbi(x, params: dict) -> np.ndarray:
    """Most likely state path given the WHOLE sample (log-space Viterbi)."""
    x = np.asarray(x, dtype=float)
    logB = np.log(_normal_pdf(x, params["mu"], params["var"]) + 1e-300)
    logA = np.log(params["A"])
    T, n = logB.shape
    delta, back = np.log(params["pi"] + 1e-300) + logB[0], np.zeros((T, n), dtype=int)
    for t in range(1, T):
        s = delta[:, None] + logA
        back[t] = s.argmax(0)
        delta = s.max(0) + logB[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def hmm_filter(x, params: dict) -> np.ndarray:
    """P(state_t | x_1..x_t): the forward probabilities, using data up to t ONLY (a real-time signal). Shape (T, n)."""
    x = np.asarray(x, dtype=float)
    B = _normal_pdf(x, params["mu"], params["var"]) + 1e-300
    out = np.empty_like(B)
    a = params["pi"] * B[0]
    out[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (out[t - 1] @ params["A"]) * B[t]
        out[t] = a / a.sum()
    return out


def hmm_regimes(returns: pd.Series, n: int = 2) -> pd.Series:
    """hmm_fit on the returns (NaN dropped), Viterbi states relabelled 0..n−1 by ascending state VARIANCE. This
    smooths over the whole sample: fine for describing the past, not a tradeable real-time signal."""
    r = returns.dropna()
    params = hmm_fit(r.to_numpy(), n)
    order = np.argsort(params["var"])
    remap = {int(old): new for new, old in enumerate(order)}
    return pd.Series([remap[int(k)] for k in hmm_viterbi(r.to_numpy(), params)], index=r.index)


def regime_performance(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Per regime (common dates): annualized Sharpe (ddof=1), mean daily return, share of days."""
    df = pd.DataFrame({"r": returns, "g": regimes}).dropna()
    g = df.groupby("g")["r"]
    return pd.DataFrame({"sharpe": g.mean() / g.std() * np.sqrt(252), "mean": g.mean(), "share": g.size() / len(df)})


# ------------------------------------------------------------------------------ S9 purged CV
class PurgedKFold:
    """Scikit-learn compatible purged k-fold: contiguous test folds (np.array_split of the positions); a training
    sample is kept only if its label ENDS before the test fold starts (t1 < first test t0) or STARTS after both the
    last test label's end and the embargo end. embargo = fraction of samples, rounded up. t1 is indexed by t0."""

    def __init__(self, t1: pd.Series, n_splits: int = 5, embargo: float = 0.01):
        self.t1, self.n_splits, self.embargo = t1, n_splits, embargo

    def get_n_splits(self, *args, **kwargs):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        t0 = self.t1.index
        n, emb = len(t0), int(np.ceil(self.embargo * len(t0)))
        for test in np.array_split(np.arange(n), self.n_splits):
            test_start, test_end = t0[test[0]], self.t1.iloc[test].max()
            emb_end = t0[min(test[-1] + emb, n - 1)]
            train = np.where((self.t1.to_numpy() < test_start) | (t0 > max(test_end, emb_end)))[0]
            yield train, test


class ShuffledKFold:
    """The common mistake: k random folds, ignoring time and overlap (sklearn KFold(shuffle=True) behaviour)."""

    def __init__(self, n_splits: int = 5, seed: int = 0):
        self.n_splits, self.seed = n_splits, seed

    def get_n_splits(self, *args, **kwargs):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        idx = np.random.default_rng(self.seed).permutation(len(X))
        for test in np.array_split(idx, self.n_splits):
            yield np.setdiff1d(np.arange(len(X)), test), np.sort(test)


def cv_scores(model, X: pd.DataFrame, y: pd.Series, cv, weights=None) -> pd.DataFrame:
    """Per (train, test) of cv.split(X): fit a fresh clone (weights sliced to the train rows), predict_proba on test.
    One row per fold: log_loss, auc, accuracy (threshold 0.5)."""
    rows = []
    for tr, te in cv.split(X):
        w = None if weights is None else np.asarray(weights)[tr]
        m = fit_weighted(clone(model), X.iloc[tr], y.iloc[tr], w)
        p = m.predict_proba(X.iloc[te])[:, 1]
        rows.append({"log_loss": log_loss(y.iloc[te], p, labels=[0, 1]), "auc": roc_auc_score(y.iloc[te], p),
                     "accuracy": accuracy_score(y.iloc[te], p > 0.5)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ S10 feature importance
def mdi_importance(fitted_forest, columns) -> pd.Series:
    """Mean Decrease Impurity: the fitted forest's feature_importances_ (in-sample, biased), sorted descending."""
    return pd.Series(fitted_forest.feature_importances_, index=list(columns)).sort_values(ascending=False)


def mda_importance(model, X: pd.DataFrame, y: pd.Series, cv, n_repeats: int = 3, seed: int = 0,
                   groups: dict[str, list[str]] | None = None) -> pd.Series:
    """Out-of-fold log-loss INCREASE when a feature (or a group of features, permuted TOGETHER) is shuffled: fit a
    clone per fold; rng = default_rng(seed); n_repeats per feature per fold; mean over all. Sorted descending."""
    rng = np.random.default_rng(seed)
    groups = groups or {c: [c] for c in X.columns}
    drops = {g: [] for g in groups}
    for tr, te in cv.split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        base = log_loss(y.iloc[te], m.predict_proba(X.iloc[te]), labels=m.classes_)
        for g, cols in groups.items():
            for _ in range(n_repeats):
                Xs = X.iloc[te].copy()
                perm = rng.permutation(len(Xs))
                Xs[cols] = Xs[cols].to_numpy()[perm]
                drops[g].append(log_loss(y.iloc[te], m.predict_proba(Xs), labels=m.classes_) - base)
    return pd.Series({g: float(np.mean(v)) for g, v in drops.items()}).sort_values(ascending=False)


def sfi_importance(model, X: pd.DataFrame, y: pd.Series, cv) -> pd.Series:
    """Single Feature Importance: mean out-of-fold AUC of the model trained on each feature ALONE. Sorted descending."""
    return pd.Series({c: float(cv_scores(model, X[[c]], y, cv)["auc"].mean()) for c in X.columns}
                     ).sort_values(ascending=False)


def cluster_features(X: pd.DataFrame, threshold: float = 0.5) -> dict[str, list[str]]:
    """Group features by distance 1 − |ρ| (average linkage, fcluster criterion "distance", t=threshold). Each cluster
    is named by its first feature (column order); members in column order."""
    d = 1 - X.corr().abs().to_numpy()
    np.fill_diagonal(d, 0.0)
    lab = fcluster(linkage(squareform(d, checks=False), "average"), t=threshold, criterion="distance")
    out: dict[str, list[str]] = {}
    for col, k in zip(X.columns, lab):
        members = [c for c, kk in zip(X.columns, lab) if kk == k]
        out.setdefault(members[0], members)
    return out


# ----------------------------------------------------------------------------- S11 audit
def shuffled_labels_test(model, X, y, cv, seed: int = 0, tol: float = 0.05) -> dict:
    """Refit under cv with y permuted: the mean AUC must fall to chance. {"auc", "passed": |auc − 0.5| < tol}."""
    ys = pd.Series(np.random.default_rng(seed).permutation(y.to_numpy()), index=y.index)
    auc = float(cv_scores(model, X, ys, cv)["auc"].mean())
    return {"auc": auc, "passed": abs(auc - 0.5) < tol}


def canary_test(model, X, y, cv, seed: int = 0) -> dict:
    """Add a deliberately LEAKED feature canary = y + N(0, 0.5) and run mda_importance (n_repeats=1): the audit
    machinery works if the canary ranks FIRST. {"rank", "passed"}."""
    rng = np.random.default_rng(seed)
    Xc = X.assign(canary=y.to_numpy() + rng.normal(0, 0.5, len(y)))
    imp = mda_importance(model, Xc, y, cv, n_repeats=1, seed=seed)
    rank = int(list(imp.index).index("canary")) + 1
    return {"rank": rank, "passed": rank == 1}


def time_shift_test(model, X, y, cv, margin: float = 0.02) -> dict:
    """Features one bar LATER (X.shift(1), bfill): information arrives later, so the CV AUC must not IMPROVE by more
    than `margin`. {"auc", "auc_shifted", "passed"}."""
    a = float(cv_scores(model, X, y, cv)["auc"].mean())
    b = float(cv_scores(model, X.shift(1).bfill(), y, cv)["auc"].mean())
    return {"auc": a, "auc_shifted": b, "passed": b <= a + margin}


def overlap_test(train_idx, test_idx, t1: pd.Series) -> dict:
    """No training label may overlap the test span [first test t0, last test t1]. {"n_overlap", "passed"}."""
    t0 = t1.index
    lo, hi = t0[test_idx].min(), t1.iloc[test_idx].max()
    tr_t0, tr_t1 = t0[train_idx], t1.iloc[train_idx].to_numpy()
    n = int(((tr_t1 >= lo) & (tr_t0 <= hi)).sum())
    return {"n_overlap": n, "passed": n == 0}


# ------------------------------------------------------------- S12 tuning, ensembles, registry
def random_search(X, y, cv, n_trials: int = 10, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Random search (the labs use Optuna's TPE) over HistGradientBoostingClassifier(max_iter=100, random_state=0)
    with max_leaf_nodes in [4, 32] (int), learning_rate in [0.01, 0.2] (log-uniform), min_samples_leaf in [10, 100]
    (int); objective = mean cv_scores log_loss (lower is better). Return (best params, DataFrame of ALL trials:
    number, value and the parameters): every trial counts for the Deflated Sharpe Ratio."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_trials):
        params = {"max_leaf_nodes": int(rng.integers(4, 33)),
                  "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
                  "min_samples_leaf": int(rng.integers(10, 101))}
        model = HistGradientBoostingClassifier(max_iter=100, random_state=0, **params)
        rows.append({"number": k, "value": float(cv_scores(model, X, y, cv)["log_loss"].mean()), **params})
    trials = pd.DataFrame(rows)
    best = trials.loc[trials["value"].idxmin()]
    return {c: best[c].item() for c in ("max_leaf_nodes", "learning_rate", "min_samples_leaf")}, trials


def seed_ensemble(make: Callable[[int], object], X_train, y_train, X_test, seeds=(0, 1, 2, 3, 4)) -> np.ndarray:
    """Average predict_proba[:, 1] of make(seed) fitted on the same data, one per seed."""
    return np.mean([make(s).fit(X_train, y_train).predict_proba(X_test)[:, 1] for s in seeds], axis=0)


class ModelRegistry:
    """A tiny file-backed registry (the platform uses MLflow): versions per model name, each with metadata and a
    stage. Stages move ONE step at a time, research → shadow → paper → live, and only if the audit passed.
    Promoting a version to "live" moves the previous live version back to "paper"."""

    STAGES = ["research", "shadow", "paper", "live"]

    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=1, default=str))

    def register(self, name: str, meta: dict) -> int:
        """New version (1, 2, …) in stage "research"; `meta` must contain "audit_passed". Return the version."""
        if "audit_passed" not in meta:
            raise ValueError("meta must record audit_passed")
        versions = self.data.setdefault(name, [])
        versions.append({"version": len(versions) + 1, "stage": "research", "meta": meta})
        self._save()
        return len(versions)

    def promote(self, name: str, version: int, stage: str) -> None:
        """Move a version to `stage`: ValueError if it is not the NEXT stage or the audit did not pass."""
        v = self.data[name][version - 1]
        if self.STAGES.index(stage) != self.STAGES.index(v["stage"]) + 1:
            raise ValueError(f"cannot go from {v['stage']} to {stage}")
        if not v["meta"]["audit_passed"]:
            raise ValueError("audit failed: cannot promote")
        if stage == "live":
            for other in self.data[name]:
                if other["stage"] == "live":
                    other["stage"] = "paper"
        v["stage"] = stage
        self._save()

    def get(self, name: str, stage: str) -> dict | None:
        """The HIGHEST version currently in `stage`, or None."""
        found = [v for v in self.data.get(name, []) if v["stage"] == stage]
        return max(found, key=lambda v: v["version"]) if found else None


# ------------------------------------------------------------- S13 meta-labeled momentum
FEATURES = ["ret5", "ret20", "vol20", "vol_ratio", "er20", "volume_z", "side"]


def meta_dataset(bars: pd.DataFrame, h: float = 0.015, pt: float = 1.0, sl: float = 1.0, max_hold: int = 10,
                 warmup: int = 100) -> pd.DataFrame:
    """CUSUM events (from bar `warmup` on) → triple-barrier labels → momentum side at t0. Columns: ml_features at t0,
    side, t1, ret (= side · barrier return), y (meta label), w (average uniqueness). NaN rows dropped."""
    close = bars["close"]
    ev = cusum_events(close, h)
    ev = ev[ev >= close.index[warmup]]
    tb = triple_barrier(close, ev, daily_vol(close), pt, sl, max_hold)
    side = momentum_side(close).loc[tb.index]
    df = ml_features(close, bars["volume"]).loc[tb.index].assign(side=side)
    df["t1"] = tb["t1"]
    df["ret"] = side * tb["ret"]
    df["y"] = meta_labels(side, tb["ret"])
    df["w"] = avg_uniqueness(tb["t1"], close.index)
    return df.dropna()


def walk_forward_meta(ds: pd.DataFrame, kind: str = "rf", start: float = 1 / 3, retrain_every: int = 100,
                      cost_bps: float = 5.0) -> pd.DataFrame:
    """From event position int(start·n), every `retrain_every` events: train make_model(kind) with weights w on the
    events whose t1 is STRICTLY BEFORE the current event's t0 (purging), predict the next block. size =
    bet_size(prob). Per OOS event: prob, size, y, primary = ret − 2·cost, meta = size·ret − 2·cost·size."""
    n, cost, rows = len(ds), cost_bps / 1e4, []
    for s in range(int(start * n), n, retrain_every):
        known = ds.iloc[:s][ds["t1"].iloc[:s] < ds.index[s]]
        m = fit_weighted(make_model(kind), known[FEATURES], known["y"], known["w"])
        block = ds.iloc[s:s + retrain_every]
        p = m.predict_proba(block[FEATURES])[:, 1]
        size = bet_size(p)
        rows.append(pd.DataFrame({"prob": p, "size": size, "y": block["y"], "primary": block["ret"] - 2 * cost,
                                  "meta": size * block["ret"] - 2 * cost * size}, index=block.index))
    return pd.concat(rows)


def meta_summary(oos: pd.DataFrame, years: float) -> pd.DataFrame:
    """Rows primary, meta: trades (meta: size > 0), precision (share of y == 1 among them), total, sharpe (per-event
    P&L, × √(events per year))."""
    k = np.sqrt(len(oos) / years)
    out = {}
    for name, mask in (("primary", np.ones(len(oos), bool)), ("meta", (oos["size"] > 0).to_numpy())):
        pnl = oos[name]
        out[name] = {"trades": int(mask.sum()), "precision": float(oos["y"][mask].mean()),
                     "total": float(pnl.sum()), "sharpe": float(pnl.mean() / pnl.std(ddof=1) * k)}
    return pd.DataFrame(out).T


# --------------------------------------------------------------- S14 volatility & rare events
def har_features(absret: pd.Series) -> pd.DataFrame:
    """HAR (Corsi 2009) on a daily volatility proxy v: d = v_t, w = mean of the last 5, m = mean of the last 22;
    target = v_{t+1}. NaN rows dropped."""
    return pd.DataFrame({"d": absret, "w": absret.rolling(5).mean(), "m": absret.rolling(22).mean(),
                         "target": absret.shift(-1)}).dropna()


def vol_forecasts(har: pd.DataFrame, split: int) -> pd.DataFrame:
    """Train on rows [:split], forecast [split:]: naive (= d), har (LinearRegression on d, w, m), gbm
    (HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=8, random_state=0)), target."""
    X, y = har[["d", "w", "m"]], har["target"]
    tr, te = slice(None, split), slice(split, None)
    lin = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
    gbm = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=8, random_state=0
                                        ).fit(X.iloc[tr], y.iloc[tr])
    return pd.DataFrame({"naive": X["d"].iloc[te], "har": lin.predict(X.iloc[te]), "gbm": gbm.predict(X.iloc[te]),
                         "target": y.iloc[te]}, index=X.index[te])


def crash_labels(ret: pd.Series, horizon: int = 5, threshold: float = 0.04) -> pd.Series:
    """1 if the cumulative log return over the NEXT `horizon` days falls below −threshold at ANY point, else 0; NaN
    where the future is incomplete."""
    cum = ret.cumsum()
    worst = pd.concat([cum.shift(-k) for k in range(1, horizon + 1)], axis=1).min(axis=1) - cum
    out = (worst < -threshold).astype(float)
    out.iloc[-horizon:] = np.nan
    return out


def crash_warning(ret: pd.Series, split: int, horizon: int = 5, threshold: float = 0.04) -> pd.DataFrame:
    """Features vol5, vol20, vol60, ret5; LogisticRegression(class_weight="balanced") trained on rows before
    split − horizon (resolved labels only); predict rows [split:]. Columns prob, label."""
    X = pd.DataFrame({"vol5": ret.rolling(5).std(), "vol20": ret.rolling(20).std(), "vol60": ret.rolling(60).std(),
                      "ret5": ret.rolling(5).sum()})
    y = crash_labels(ret, horizon, threshold)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    tr = X.index < ret.index[split - horizon]
    m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr], y[tr])
    te = X.index >= ret.index[split]
    return pd.DataFrame({"prob": m.predict_proba(X[te])[:, 1], "label": y[te]}, index=X.index[te])


def pr_auc(prob, label) -> float:
    """Average precision: compare with the base rate, never with 0.5."""
    return float(average_precision_score(label, prob))


def risk_off_returns(ret: pd.Series, prob: pd.Series, threshold: float = 0.5, low: float = 0.5) -> pd.Series:
    """Exposure 1, or `low` on days AFTER a warning (prob at t−1 > threshold): a risk cut, never a short."""
    expo = pd.Series(np.where(prob > threshold, low, 1.0), index=prob.index).shift(1).dropna()
    return ret.loc[expo.index] * expo


def max_drawdown(ret: pd.Series) -> float:
    """Of the cumulative sum of (log) returns; negative."""
    c = ret.cumsum()
    return float((c - c.cummax()).min())


def rank_ic(pred: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    """Per date: Spearman correlation across symbols between prediction and realized return."""
    return pd.Series([spearmanr(pred.loc[d], realized.loc[d]).statistic for d in pred.index], index=pred.index)


# ------------------------------------------------------------------------ S15 drift & serving
def psi(expected, actual, bins: int = 10) -> float:
    """Quantile bins of `expected` (outer edges ±inf); proportions (+1e-6) e, a; PSI = Σ (a − e) ln(a/e)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def drift_table(train: pd.DataFrame, live: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Per block of `window` live rows (index = the block's last date): PSI of each feature against the training
    sample, plus the KS p-value of the first feature ("ks_p")."""
    rows, idx = [], []
    for s in range(0, len(live) - window + 1, window):
        blk = live.iloc[s:s + window]
        row = {c: psi(train[c], blk[c]) for c in train.columns}
        row["ks_p"] = float(ks_2samp(train.iloc[:, 0], blk.iloc[:, 0]).pvalue)
        rows.append(row)
        idx.append(blk.index[-1])
    return pd.DataFrame(rows, index=idx)


def first_alarm(table: pd.DataFrame, threshold: float = 0.25):
    """The first date where ANY feature's PSI exceeds the threshold, else None."""
    hit = (table.drop(columns="ks_p", errors="ignore") > threshold).any(axis=1)
    return hit.index[hit.argmax()] if hit.any() else None


class StreamingRolling:
    """Rolling mean and std (ddof=1) of the last `window` values, updated in O(1) with running sums. Must equal
    pandas rolling on the same data: the train/serve parity test."""

    def __init__(self, window: int):
        self.window = window
        self.buf: deque = deque()
        self.s = self.s2 = 0.0

    def update(self, x: float) -> tuple[float, float]:
        """Add x (drop the oldest if full); return (mean, std), NaN until `window` values have been seen."""
        self.buf.append(x)
        self.s += x
        self.s2 += x * x
        if len(self.buf) > self.window:
            old = self.buf.popleft()
            self.s -= old
            self.s2 -= old * old
        if len(self.buf) < self.window:
            return np.nan, np.nan
        n = self.window
        mean = self.s / n
        return mean, float(np.sqrt(max((self.s2 - n * mean * mean) / (n - 1), 0.0)))


def shadow_compare(returns: pd.Series, champion: pd.Series, challenger: pd.Series, min_days: int = 60,
                   margin: float = 0.3) -> dict:
    """Both models' positions (decided at t, earning returns[t+1]) are logged; only the champion trades. Return
    sharpe_champion, sharpe_challenger, agreement (same position sign) and promote (at least min_days and challenger
    Sharpe > champion Sharpe + margin)."""
    nxt = returns.shift(-1)
    a, b = (champion * nxt).dropna(), (challenger * nxt).dropna()
    sa, sb = sharpe(a), sharpe(b)
    return {"sharpe_champion": sa, "sharpe_challenger": sb,
            "agreement": float((np.sign(champion) == np.sign(challenger)).mean()),
            "promote": bool(len(b) >= min_days and sb > sa + margin)}


# ------------------------------------------------------------------------------- S21 RL foundations
def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 0.9, tol: float = 1e-10
                    ) -> tuple[np.ndarray, np.ndarray]:
    """P: (A, S, S) transition probabilities, R: (S, A) expected rewards. Iterate V ← max_a [R[:, a] + γ P[a] V]
    until max |ΔV| < tol. Return (V, greedy policy)."""
    V = np.zeros(P.shape[1])
    while True:
        Q = R + gamma * np.einsum("ast,t->sa", P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new


def bucket(x: float, edges: np.ndarray) -> int:
    """State index of a continuous value: np.searchsorted(edges, x)."""
    return int(np.searchsorted(edges, x))


def q_learning_paths(paths: list[np.ndarray], edges: np.ndarray, epochs: int = 5, alpha: float = 0.05,
                     eps: float = 0.1, seed: int = 0) -> np.ndarray:
    """Tabular Q-learning on price paths (γ = 0, no costs): state = bucket(x_t), actions 0/1/2 = positions −1/0/+1,
    reward = position · (x_{t+1} − x_t); ε-greedy; Q[s, a] += α (reward − Q[s, a])."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((len(edges) + 1, 3))
    for _ in range(epochs):
        for x in paths:
            for t in range(len(x) - 1):
                s = bucket(x[t], edges)
                a = int(rng.integers(3)) if rng.random() < eps else int(Q[s].argmax())
                Q[s, a] += alpha * ((a - 1) * (x[t + 1] - x[t]) - Q[s, a])
    return Q


def greedy_reward(Q: np.ndarray, paths: list[np.ndarray], edges: np.ndarray) -> float:
    """Mean reward per step of the greedy policy (argmax Q) on the given paths."""
    total, n = 0.0, 0
    for x in paths:
        for t in range(len(x) - 1):
            total += (Q[bucket(x[t], edges)].argmax() - 1) * (x[t + 1] - x[t])
            n += 1
    return total / n


# --------------------------------------------------------------------------- S22 trading env
class TradingEnv:
    """Gymnasium-style environment (reset/step). Single asset; action 0/1/2 = target position −1/0/+1; reward = new
    position × r[t] (the return AFTER the decision) − cost·|position change|. Observation = the last `lookback`
    returns / scale, then the current position. `scale` defaults to the std of the returns GIVEN (for a test
    environment pass the TRAINING scale). The episode starts at t = lookback (or a random start when episode_len is
    set) and ends when the data (or episode_len) runs out."""

    n_actions = 3

    def __init__(self, returns: np.ndarray, lookback: int = 10, cost: float = 0.0005, scale: float | None = None,
                 episode_len: int | None = None):
        self.r = np.asarray(returns, dtype=float)
        self.L, self.cost, self.episode_len = lookback, cost, episode_len
        self.scale = float(np.std(self.r)) if scale is None else float(scale)

    def _obs(self):
        return np.append(self.r[self.t - self.L: self.t] / (self.scale + 1e-8), self.pos)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        if self.episode_len is None:
            self.t, self.end = self.L, len(self.r)
        else:
            self.t = int(rng.integers(self.L, len(self.r) - self.episode_len + 1))
            self.end = self.t + self.episode_len
        self.pos = 0.0
        return self._obs(), {}

    def step(self, action):
        new_pos = float(action) - 1.0
        reward = new_pos * self.r[self.t] - self.cost * abs(new_pos - self.pos)
        self.pos, self.t = new_pos, self.t + 1
        terminated = self.t >= self.end
        obs = self._obs() if not terminated else np.zeros(self.L + 1)
        return obs, float(reward), terminated, False, {}


def run_policy(env: TradingEnv, policy: Callable[[np.ndarray], int], seed: int | None = None) -> float:
    """Play one episode with policy(obs) → action; return the total reward."""
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, done, trunc, _ = env.step(policy(obs))
        done = done or trunc
        total += r
    return total


def differential_sharpe(returns, eta: float = 0.01) -> np.ndarray:
    """Moody & Saffell (2001): A_t = A + η(R_t − A), B_t = B + η(R_t² − B) (A = B = 0 at the start);
    D_t = (B_{t−1}ΔA_t − ½A_{t−1}ΔB_t) / (B_{t−1} − A_{t−1}²)^{3/2}; D_t = 0 while B_{t−1} − A_{t−1}² <= 0."""
    A = B = 0.0
    out = []
    for R in np.asarray(returns, dtype=float):
        dA, dB = R - A, R * R - B
        den = B - A * A
        out.append((B * dA - 0.5 * A * dB) / den ** 1.5 if den > 0 else 0.0)
        A, B = A + eta * dA, B + eta * dB
    return np.array(out)


def sign_state(obs: np.ndarray) -> int:
    """Tabular state from a TradingEnv observation: (last return > 0) × 3 + current position + 1 → 0..5."""
    return int(obs[-2] > 0) * 3 + int(round(obs[-1])) + 1


def q_learning_env(env: TradingEnv, n_states: int, state_fn: Callable[[np.ndarray], int] = sign_state,
                   episodes: int = 200, alpha: float = 0.05, gamma: float = 0.9, eps: float = 0.1, seed: int = 0
                   ) -> np.ndarray:
    """Q-learning on the environment: each episode env.reset(seed=seed + episode); ε-greedy (rng default_rng(seed));
    target = r + γ max Q[s'] (just r when terminated)."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, 3))
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        s, done = state_fn(obs), False
        while not done:
            a = int(rng.integers(3)) if rng.random() < eps else int(Q[s].argmax())
            obs, r, done, _, _ = env.step(a)
            s2 = state_fn(obs)
            Q[s, a] += alpha * ((r if done else r + gamma * Q[s2].max()) - Q[s, a])
            s = s2
    return Q


# --------------------------------------------------------------------------- S23 execution
def twap_schedule(X: float, N: int) -> np.ndarray:
    """Sell X in N equal slices."""
    return np.full(N, X / N)


def almgren_chriss_schedule(X: float, N: int, sigma: float, eta: float, gamma: float, lam: float) -> np.ndarray:
    """Almgren–Chriss (τ = 1): η̃ = η − γ/2; cosh κ = 1 + λσ²/(2η̃); holdings x_j = X sinh(κ(N − j))/sinh(κN),
    trades n_j = x_{j−1} − x_j. λ → 0 gives TWAP."""
    eta_t = eta - gamma / 2
    kappa = np.arccosh(1 + lam * sigma ** 2 / (2 * eta_t))
    if kappa < 1e-12:
        return twap_schedule(X, N)
    j = np.arange(N + 1)
    return -np.diff(X * np.sinh(kappa * (N - j)) / np.sinh(kappa * N))


def cost_moments(trades: np.ndarray, sigma: float, eta: float, gamma: float) -> tuple[float, float]:
    """Expected shortfall and variance of a sell schedule (τ = 1, linear impact): E = ½γX² + η̃ Σ n_k²,
    Var = σ² Σ x_k² over the holdings after each trade except the last."""
    n = np.asarray(trades, dtype=float)
    X = n.sum()
    x = X - np.cumsum(n)
    return float(0.5 * gamma * X ** 2 + (eta - gamma / 2) * np.sum(n ** 2)), float(sigma ** 2 * np.sum(x[:-1] ** 2))


class ExecutionEnv:
    """Sell X shares in N steps (Gymnasium-style). Observation [inventory / X, steps left / N]; action = fraction of
    the REMAINING inventory to sell now (everything on the last step). S_k = S_{k−1} + σξ_k − γn_k; the fill price
    of n_k is S_{k−1} − ηn_k; reward = n_k (fill − S_0), so the episode's total reward = −implementation shortfall."""

    def __init__(self, X: float = 1000, N: int = 20, sigma: float = 0.3, eta: float = 0.01, gamma: float = 0.001,
                 s0: float = 100.0):
        self.X, self.N, self.sigma, self.eta, self.gamma, self.s0 = X, N, sigma, eta, gamma, s0

    def _obs(self):
        return np.array([self.inv / self.X, (self.N - self.k) / self.N])

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.inv, self.k, self.S = float(self.X), 0, self.s0
        return self._obs(), {}

    def step(self, action):
        frac = float(np.clip(np.asarray(action, dtype=float).ravel()[0], 0.0, 1.0))
        n = self.inv if self.k == self.N - 1 else frac * self.inv
        fill = self.S - self.eta * n
        reward = n * (fill - self.s0)
        self.S += self.sigma * self.rng.standard_normal() - self.gamma * n
        self.inv -= n
        self.k += 1
        return self._obs(), float(reward), self.k >= self.N, False, {"n": n}


def schedule_policy(trades: np.ndarray) -> Callable[[np.ndarray, int], float]:
    """Turn a trade schedule into an ExecutionEnv policy: at step k sell trades[k] / remaining."""
    remaining = trades[::-1].cumsum()[::-1]

    def policy(obs, k):
        return float(trades[k] / remaining[k]) if remaining[k] > 0 else 1.0
    return policy


def simulate_execution(env: ExecutionEnv, policy: Callable[[np.ndarray, int], float], n_paths: int = 500,
                       seed: int = 0) -> np.ndarray:
    """Implementation shortfall (= −total reward) of policy(obs, k) on n_paths episodes, reset(seed=seed + i)."""
    out = []
    for i in range(n_paths):
        obs, _ = env.reset(seed=seed + i)
        total, done, k = 0.0, False, 0
        while not done:
            obs, r, done, _, _ = env.step([policy(obs, k)])
            total += r
            k += 1
        out.append(-total)
    return np.array(out)
