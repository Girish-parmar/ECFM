"""Week 36 (S13–S15) — ML strategies and deployment: a walk-forward meta-labeled momentum strategy, HAR vs ML
volatility forecasts, a rare-event (crash) warning used only to cut risk, rank IC, and drift monitoring, streaming
features with train/serve parity and a shadow-mode champion–challenger comparison.

A model is trained only on labels that were RESOLVED before the moment it predicts.
Fill in every block marked "Your turn", then run:  python -m pytest week36_strategies
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score

from _loader import load

fe = load("week33_features", "features")
lb = load("week34_labels", "labels")


# ------------------------------------------------------------- S13 meta-labeled momentum
def meta_dataset(bars: pd.DataFrame, h: float = 0.015, pt: float = 1.0, sl: float = 1.0, max_hold: int = 10,
                 warmup: int = 100) -> pd.DataFrame:
    """Events = fe.cusum_events(close, h) from bar `warmup` on; labels = lb.triple_barrier(close, events,
    lb.daily_vol(close), pt, sl, max_hold); side = lb.momentum_side(close) at t0. Columns: the lb.ml_features at t0,
    side, t1, ret (= side · barrier return: the primary trade's result), y (lb.meta_labels), w
    (lb.avg_uniqueness over the bars). Rows with any NaN feature dropped."""
    # >>> SOLUTION
    close = bars["close"]
    ev = fe.cusum_events(close, h)
    ev = ev[ev >= close.index[warmup]]
    tb = lb.triple_barrier(close, ev, lb.daily_vol(close), pt, sl, max_hold)
    side = lb.momentum_side(close).loc[tb.index]
    df = lb.ml_features(close, bars["volume"]).loc[tb.index].assign(side=side)
    df["t1"] = tb["t1"]
    df["ret"] = side * tb["ret"]
    df["y"] = lb.meta_labels(side, tb["ret"])
    df["w"] = lb.avg_uniqueness(tb["t1"], close.index)
    return df.dropna()
    # <<< SOLUTION


FEATURES = ["ret5", "ret20", "vol20", "vol_ratio", "er20", "volume_z", "side"]


def walk_forward_meta(ds: pd.DataFrame, kind: str = "rf", start: float = 1 / 3, retrain_every: int = 100,
                      cost_bps: float = 5.0) -> pd.DataFrame:
    """From event position int(start·n), every `retrain_every` events: train lb.make_model(kind) with weights w on the
    events whose t1 is STRICTLY BEFORE the current event's t0 (their label was already known: purging), then predict
    the next block. size = lb.bet_size(prob). Per OOS event: prob, size, y, primary = ret − 2·cost, meta =
    size·ret − 2·cost·size (cost = cost_bps/1e4, round trip on each unit traded)."""
    # >>> SOLUTION
    n = len(ds)
    cost = cost_bps / 1e4
    rows = []
    for s in range(int(start * n), n, retrain_every):
        known = ds.iloc[:s][ds["t1"].iloc[:s] < ds.index[s]]
        m = lb.fit_weighted(lb.make_model(kind), known[FEATURES], known["y"], known["w"])
        block = ds.iloc[s:s + retrain_every]
        p = m.predict_proba(block[FEATURES])[:, 1]
        size = lb.bet_size(p)
        rows.append(pd.DataFrame({"prob": p, "size": size, "y": block["y"], "primary": block["ret"] - 2 * cost,
                                  "meta": size * block["ret"] - 2 * cost * size}, index=block.index))
    return pd.concat(rows)
    # <<< SOLUTION


def meta_summary(oos: pd.DataFrame, years: float) -> pd.DataFrame:
    """Rows primary, meta. Columns: trades (primary: all; meta: size > 0), precision (share of y == 1 among those
    trades), total (sum of P&L), sharpe (mean/std(ddof=1) of the per-event P&L × √(events per year), events per year
    = len(oos)/years)."""
    # >>> SOLUTION
    k = np.sqrt(len(oos) / years)
    out = {}
    for name, mask in (("primary", np.ones(len(oos), bool)), ("meta", (oos["size"] > 0).to_numpy())):
        pnl = oos[name]
        out[name] = {"trades": int(mask.sum()), "precision": float(oos["y"][mask].mean()),
                     "total": float(pnl.sum()), "sharpe": float(pnl.mean() / pnl.std(ddof=1) * k)}
    return pd.DataFrame(out).T
    # <<< SOLUTION


# --------------------------------------------------------------- S14 volatility & rare events
def har_features(absret: pd.Series) -> pd.DataFrame:
    """HAR (Corsi 2009) on a daily volatility proxy v (here |return|): d = v_t, w = mean of the last 5, m = mean of
    the last 22 (all ending at t); target = v_{t+1}. Columns d, w, m, target; NaN rows dropped."""
    # >>> SOLUTION
    return pd.DataFrame({"d": absret, "w": absret.rolling(5).mean(), "m": absret.rolling(22).mean(),
                         "target": absret.shift(-1)}).dropna()
    # <<< SOLUTION


def vol_forecasts(har: pd.DataFrame, split: int) -> pd.DataFrame:
    """Train on rows [:split], forecast rows [split:]. Columns: naive (= d, tomorrow like today), har (LinearRegression
    on d, w, m), lgbm (LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=8, verbose=-1, random_state=0)
    on d, w, m), target."""
    # >>> SOLUTION
    X, y = har[["d", "w", "m"]], har["target"]
    tr, te = slice(None, split), slice(split, None)
    lin = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
    gbm = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=8, verbose=-1, random_state=0
                        ).fit(X.iloc[tr], y.iloc[tr])
    return pd.DataFrame({"naive": X["d"].iloc[te], "har": lin.predict(X.iloc[te]), "lgbm": gbm.predict(X.iloc[te]),
                         "target": y.iloc[te]}, index=X.index[te])
    # <<< SOLUTION


def crash_labels(ret: pd.Series, horizon: int = 5, threshold: float = 0.04) -> pd.Series:
    """1 if the cumulative log return over the NEXT `horizon` days falls below −threshold at ANY point (a drawdown from
    today's close), else 0; NaN where the future is incomplete."""
    # >>> SOLUTION
    cum = ret.cumsum()
    worst = pd.concat([cum.shift(-k) for k in range(1, horizon + 1)], axis=1).min(axis=1) - cum
    out = (worst < -threshold).astype(float)
    out.iloc[-horizon:] = np.nan
    return out
    # <<< SOLUTION


def crash_warning(ret: pd.Series, split: int, horizon: int = 5, threshold: float = 0.04) -> pd.DataFrame:
    """Features at t: vol5, vol20, vol60 (std of the last 5/20/60 returns) and ret5 (sum of the last 5). A
    LogisticRegression(class_weight="balanced") trained on rows [:split − horizon] (their labels are resolved by
    split); predict rows [split:]. Columns: prob, label (NaN rows dropped)."""
    # >>> SOLUTION
    X = pd.DataFrame({"vol5": ret.rolling(5).std(), "vol20": ret.rolling(20).std(), "vol60": ret.rolling(60).std(),
                      "ret5": ret.rolling(5).sum()})
    y = crash_labels(ret, horizon, threshold)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    cut = ret.index[split]
    tr = X.index < ret.index[split - horizon]
    m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[tr], y[tr])
    te = X.index >= cut
    return pd.DataFrame({"prob": m.predict_proba(X[te])[:, 1], "label": y[te]}, index=X.index[te])
    # <<< SOLUTION


def pr_auc(prob, label) -> float:
    """Average precision (area under the precision–recall curve): compare with the base rate, never with 0.5."""
    # >>> SOLUTION
    return float(average_precision_score(label, prob))
    # <<< SOLUTION


def risk_off_returns(ret: pd.Series, prob: pd.Series, threshold: float = 0.5, low: float = 0.5) -> pd.Series:
    """Hold the asset with exposure 1, or `low` on days AFTER a warning (prob at t−1 > threshold): a risk cut, never
    a short. Only the dates of `prob` (from its second date on)."""
    # >>> SOLUTION
    expo = pd.Series(np.where(prob > threshold, low, 1.0), index=prob.index).shift(1).dropna()
    return ret.loc[expo.index] * expo
    # <<< SOLUTION


def max_drawdown(ret: pd.Series) -> float:
    """Of the cumulative sum of (log) returns; negative."""
    c = ret.cumsum()
    return float((c - c.cummax()).min())


def rank_ic(pred: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    """Per date (row): Spearman correlation across symbols between prediction and realized return."""
    # >>> SOLUTION
    return pd.Series([spearmanr(pred.loc[d], realized.loc[d]).statistic for d in pred.index], index=pred.index)
    # <<< SOLUTION


# ------------------------------------------------------------------------ S15 drift & serving
def psi(expected, actual, bins: int = 10) -> float:
    """Lesson plan S15: quantile bins of `expected` (outer edges ±inf); proportions (+1e-6) e, a;
    PSI = Σ (a − e) ln(a/e). < 0.1 stable, 0.1–0.25 watch, > 0.25 drift."""
    # >>> SOLUTION
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))
    # <<< SOLUTION


def drift_table(train: pd.DataFrame, live: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """For every block of `window` live rows (index = the block's LAST date): PSI of each feature against the whole
    training sample, plus the KS p-value (scipy ks_2samp) of the first feature as column "ks_p". Small windows are
    noisy: 60 values in 10 bins can show PSI > 0.25 with no drift at all."""
    # >>> SOLUTION
    rows, idx = [], []
    for s in range(0, len(live) - window + 1, window):
        blk = live.iloc[s:s + window]
        row = {c: psi(train[c], blk[c]) for c in train.columns}
        row["ks_p"] = float(ks_2samp(train.iloc[:, 0], blk.iloc[:, 0]).pvalue)
        rows.append(row)
        idx.append(blk.index[-1])
    return pd.DataFrame(rows, index=idx)
    # <<< SOLUTION


def first_alarm(table: pd.DataFrame, threshold: float = 0.25):
    """The first date where ANY feature's PSI exceeds the threshold (columns other than ks_p), else None."""
    # >>> SOLUTION
    hit = (table.drop(columns="ks_p", errors="ignore") > threshold).any(axis=1)
    return hit.index[hit.argmax()] if hit.any() else None
    # <<< SOLUTION


class StreamingRolling:
    """Live feature: rolling mean and std (ddof=1) of the last `window` values, updated one value at a time in O(1)
    (running sums). Must equal pandas rolling on the same data: the train/serve parity test."""

    def __init__(self, window: int):
        self.window = window
        self.buf: deque = deque()
        self.s = self.s2 = 0.0

    def update(self, x: float) -> tuple[float, float]:
        """Add x (drop the oldest if full); return (mean, std), NaN until `window` values have been seen."""
        # >>> SOLUTION
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
        var = max((self.s2 - n * mean * mean) / (n - 1), 0.0)
        return mean, float(np.sqrt(var))
        # <<< SOLUTION


def shadow_compare(returns: pd.Series, champion: pd.Series, challenger: pd.Series, min_days: int = 60,
                   margin: float = 0.3) -> dict:
    """Shadow mode: both models' positions (decided at t, earning returns[t+1]) are logged; only the champion trades.
    Return {"sharpe_champion", "sharpe_challenger" (annualized), "agreement" (share of days with the same position
    sign), "promote": at least min_days of results and challenger Sharpe > champion Sharpe + margin}."""
    # >>> SOLUTION
    nxt = returns.shift(-1)
    a, b = (champion * nxt).dropna(), (challenger * nxt).dropna()
    sr = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else 0.0   # noqa: E731
    sa, sb = sr(a), sr(b)
    return {"sharpe_champion": sa, "sharpe_challenger": sb,
            "agreement": float((np.sign(champion) == np.sign(challenger)).mean()),
            "promote": bool(len(b) >= min_days and sb > sa + margin)}
    # <<< SOLUTION
