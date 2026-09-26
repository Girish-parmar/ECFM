"""Week 34 (S5–S8) — Triple-barrier labels, sample uniqueness, meta-labeling and bet sizing, supervised models with
calibration, and unsupervised regimes.

Labels overlap in time, so they are NOT independent: weight them by uniqueness. Probabilities feed bet sizes, so
they must be calibrated.
Fill in every block marked "Your turn", then run:  python -m pytest week34_labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from lightgbm import LGBMClassifier
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------- S5 triple barrier
def daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """EWM standard deviation of SIMPLE returns (pandas ewm(span=span).std()): known at the close of t."""
    # >>> SOLUTION
    return close.pct_change().ewm(span=span).std()
    # <<< SOLUTION


def triple_barrier(close: pd.Series, events, vol: pd.Series, pt: float = 1.0, sl: float = 1.0, max_hold: int = 10
                   ) -> pd.DataFrame:
    """Lesson plan S5. For each event t0 (skip it if it is the last bar or vol is NaN): path = close over the NEXT
    max_hold bars / close[t0] − 1; upper = pt·vol[t0], lower = −sl·vol[t0]. t1 = the first bar touching a barrier
    (>= upper or <= lower), else the last bar of the path. label = +1 upper, −1 lower, 0 vertical (time).
    DataFrame indexed by t0 with columns t1, ret (path at t1), label."""
    # >>> SOLUTION
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
    # <<< SOLUTION


def num_concurrent(t1: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """For every bar in `index`: how many labels are active, i.e. t0 <= bar <= t1 (t1 is indexed by t0)."""
    # >>> SOLUTION
    c = pd.Series(0, index=index)
    for t0, end in t1.items():
        c.loc[t0:end] += 1
    return c
    # <<< SOLUTION


def avg_uniqueness(t1: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Per label: the mean over its bars [t0, t1] of 1 / num_concurrent. 1 = no overlap with any other label."""
    # >>> SOLUTION
    c = num_concurrent(t1, index)
    return pd.Series({t0: float((1 / c.loc[t0:end]).mean()) for t0, end in t1.items()})
    # <<< SOLUTION


def sequential_bootstrap(t1: pd.Series, index: pd.DatetimeIndex, n_draws: int | None = None, seed: int = 0
                         ) -> list[int]:
    """López de Prado's sequential bootstrap: draw label positions one at a time, with probability proportional to the
    average uniqueness each label WOULD have given the labels drawn so far (its bars' 1/(1 + times already covered)).
    n_draws defaults to len(t1). Return the drawn positions (with repetitions)."""
    # >>> SOLUTION
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
    # <<< SOLUTION


# --------------------------------------------------------------------- S6 meta-labels & sizing
def momentum_side(close: pd.Series, lookback: int = 20) -> pd.Series:
    """The PRIMARY model: sign of the log return over the last `lookback` bars (+1/−1; 0 where undefined)."""
    # >>> SOLUTION
    return np.sign(np.log(close).diff(lookback)).fillna(0.0)
    # <<< SOLUTION


def meta_labels(side: pd.Series, barrier_ret: pd.Series) -> pd.Series:
    """1 if taking the primary model's side would have made money (side · barrier return > 0), else 0."""
    # >>> SOLUTION
    return ((side * barrier_ret) > 0).astype(int)
    # <<< SOLUTION


def bet_size(prob, n_classes: int = 2) -> np.ndarray:
    """Lesson plan S6: z = (p − 1/n)/√(p(1 − p)); size = clip(2Φ(z) − 1, 0, 1)."""
    # >>> SOLUTION
    prob = np.asarray(prob, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):                 # p = 0 or 1 → z = ∓inf → size 0 or 1
        z = (prob - 1 / n_classes) / np.sqrt(prob * (1 - prob))
    return np.clip(2 * norm.cdf(z) - 1, 0, 1)
    # <<< SOLUTION


def discretize(size, step: float = 0.1) -> np.ndarray:
    """Round sizes to multiples of `step` (fewer tiny re-sizings, less over-trading)."""
    # >>> SOLUTION
    return np.round(np.asarray(size, dtype=float) / step) * step
    # <<< SOLUTION


# ---------------------------------------------------------------------- S7 supervised models
def ml_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Features known at the close of t (log returns r): ret5, ret20 (sums of r), vol20 (std of r, ddof=1),
    vol_ratio (vol20 / 100-day std), er20 (efficiency ratio: |20-day log return| / Σ|r| over 20 days),
    volume_z (log volume minus its 60-day mean, over its 60-day std)."""
    # >>> SOLUTION
    r = np.log(close).diff()
    lv = np.log(volume)
    return pd.DataFrame({"ret5": r.rolling(5).sum(), "ret20": r.rolling(20).sum(), "vol20": r.rolling(20).std(),
                         "vol_ratio": r.rolling(20).std() / r.rolling(100).std(),
                         "er20": r.rolling(20).sum().abs() / r.abs().rolling(20).sum(),
                         "volume_z": (lv - lv.rolling(60).mean()) / lv.rolling(60).std()})
    # <<< SOLUTION


def make_model(kind: str, seed: int = 0):
    """"logit": make_pipeline(StandardScaler(), LogisticRegression()) · "rf": RandomForestClassifier(200 trees,
    min_samples_leaf=20, max_samples=0.5, random_state=seed) · "lgbm": LGBMClassifier(n_estimators=200,
    learning_rate=0.03, num_leaves=8, min_child_samples=30, subsample=0.7, subsample_freq=1, random_state=seed,
    verbose=-1). Anything else: ValueError."""
    # >>> SOLUTION
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression())
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_samples=0.5, random_state=seed)
    if kind == "lgbm":
        return LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                              subsample=0.7, subsample_freq=1, random_state=seed, verbose=-1)
    raise ValueError(kind)
    # <<< SOLUTION


def fit_weighted(model, X, y, w=None):
    """Fit with sample weights. A Pipeline takes them as <last step name>__sample_weight (given)."""
    if w is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": np.asarray(w)})
    return model.fit(X, y, sample_weight=np.asarray(w))


def calibrate(model, X_cal, y_cal, method: str = "isotonic"):
    """Calibrate an ALREADY FITTED model on a separate (later) calibration set: CalibratedClassifierCV around
    FrozenEstimator(model) with the given method ("isotonic" or "sigmoid" = Platt)."""
    # >>> SOLUTION
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_cal, y_cal)
    # <<< SOLUTION


def reliability_table(prob, y, bins: int = 5) -> pd.DataFrame:
    """Reliability diagram as a table: equal-width probability bins on [0, 1]; per non-empty bin: mean_pred,
    frac_pos (observed frequency), count. Index = bin number."""
    # >>> SOLUTION
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=float)
    b = np.minimum((prob * bins).astype(int), bins - 1)
    df = pd.DataFrame({"b": b, "p": prob, "y": y})
    g = df.groupby("b")
    return pd.DataFrame({"mean_pred": g["p"].mean(), "frac_pos": g["y"].mean(), "count": g.size()})
    # <<< SOLUTION


# ------------------------------------------------------------------------ S8 regimes
def gmm_regimes(features: pd.DataFrame, n: int = 2, seed: int = 0) -> pd.Series:
    """GaussianMixture(n, random_state=seed) on the standardized features (NaN rows dropped); relabel the states
    0..n−1 in ascending order of the mean of the FIRST feature within each state."""
    # >>> SOLUTION
    F = features.dropna()
    Z = (F - F.mean()) / F.std()
    lab = GaussianMixture(n, random_state=seed).fit_predict(Z)
    order = np.argsort([F.iloc[:, 0][lab == k].mean() for k in range(n)])
    remap = {int(old): new for new, old in enumerate(order)}
    return pd.Series([remap[int(k)] for k in lab], index=F.index)
    # <<< SOLUTION


def hmm_regimes(returns: pd.Series, n: int = 2, seed: int = 0) -> pd.Series:
    """GaussianHMM(n, covariance_type="full", n_iter=200, random_state=seed) on the returns (one column, NaN dropped);
    Viterbi states (predict), relabelled 0..n−1 in ascending order of state VARIANCE (0 = calmest).
    Note: this smooths over the whole sample: fine for describing the past, not a tradeable real-time signal."""
    # >>> SOLUTION
    r = returns.dropna()
    X = r.to_numpy().reshape(-1, 1)
    m = GaussianHMM(n, covariance_type="full", n_iter=200, random_state=seed).fit(X)
    lab = m.predict(X)
    order = np.argsort(m.covars_.reshape(n, -1)[:, 0])
    remap = {int(old): new for new, old in enumerate(order)}
    return pd.Series([remap[int(k)] for k in lab], index=r.index)
    # <<< SOLUTION


def regime_performance(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Per regime (common dates only): annualized Sharpe (ddof=1), mean daily return, share of days."""
    # >>> SOLUTION
    df = pd.DataFrame({"r": returns, "g": regimes}).dropna()
    g = df.groupby("g")["r"]
    return pd.DataFrame({"sharpe": g.mean() / g.std() * np.sqrt(252), "mean": g.mean(), "share": g.size() / len(df)})
    # <<< SOLUTION
