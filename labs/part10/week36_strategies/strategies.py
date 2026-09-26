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
    raise NotImplementedError("✍️ Your turn: see the docstring")


FEATURES = ["ret5", "ret20", "vol20", "vol_ratio", "er20", "volume_z", "side"]


def walk_forward_meta(ds: pd.DataFrame, kind: str = "rf", start: float = 1 / 3, retrain_every: int = 100,
                      cost_bps: float = 5.0) -> pd.DataFrame:
    """From event position int(start·n), every `retrain_every` events: train lb.make_model(kind) with weights w on the
    events whose t1 is STRICTLY BEFORE the current event's t0 (their label was already known: purging), then predict
    the next block. size = lb.bet_size(prob). Per OOS event: prob, size, y, primary = ret − 2·cost, meta =
    size·ret − 2·cost·size (cost = cost_bps/1e4, round trip on each unit traded)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def meta_summary(oos: pd.DataFrame, years: float) -> pd.DataFrame:
    """Rows primary, meta. Columns: trades (primary: all; meta: size > 0), precision (share of y == 1 among those
    trades), total (sum of P&L), sharpe (mean/std(ddof=1) of the per-event P&L × √(events per year), events per year
    = len(oos)/years)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------- S14 volatility & rare events
def har_features(absret: pd.Series) -> pd.DataFrame:
    """HAR (Corsi 2009) on a daily volatility proxy v (here |return|): d = v_t, w = mean of the last 5, m = mean of
    the last 22 (all ending at t); target = v_{t+1}. Columns d, w, m, target; NaN rows dropped."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def vol_forecasts(har: pd.DataFrame, split: int) -> pd.DataFrame:
    """Train on rows [:split], forecast rows [split:]. Columns: naive (= d, tomorrow like today), har (LinearRegression
    on d, w, m), lgbm (LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=8, verbose=-1, random_state=0)
    on d, w, m), target."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def crash_labels(ret: pd.Series, horizon: int = 5, threshold: float = 0.04) -> pd.Series:
    """1 if the cumulative log return over the NEXT `horizon` days falls below −threshold at ANY point (a drawdown from
    today's close), else 0; NaN where the future is incomplete."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def crash_warning(ret: pd.Series, split: int, horizon: int = 5, threshold: float = 0.04) -> pd.DataFrame:
    """Features at t: vol5, vol20, vol60 (std of the last 5/20/60 returns) and ret5 (sum of the last 5). A
    LogisticRegression(class_weight="balanced") trained on rows [:split − horizon] (their labels are resolved by
    split); predict rows [split:]. Columns: prob, label (NaN rows dropped)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def pr_auc(prob, label) -> float:
    """Average precision (area under the precision–recall curve): compare with the base rate, never with 0.5."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def risk_off_returns(ret: pd.Series, prob: pd.Series, threshold: float = 0.5, low: float = 0.5) -> pd.Series:
    """Hold the asset with exposure 1, or `low` on days AFTER a warning (prob at t−1 > threshold): a risk cut, never
    a short. Only the dates of `prob` (from its second date on)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def max_drawdown(ret: pd.Series) -> float:
    """Of the cumulative sum of (log) returns; negative."""
    c = ret.cumsum()
    return float((c - c.cummax()).min())


def rank_ic(pred: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    """Per date (row): Spearman correlation across symbols between prediction and realized return."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------ S15 drift & serving
def psi(expected, actual, bins: int = 10) -> float:
    """Lesson plan S15: quantile bins of `expected` (outer edges ±inf); proportions (+1e-6) e, a;
    PSI = Σ (a − e) ln(a/e). < 0.1 stable, 0.1–0.25 watch, > 0.25 drift."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def drift_table(train: pd.DataFrame, live: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """For every block of `window` live rows (index = the block's LAST date): PSI of each feature against the whole
    training sample, plus the KS p-value (scipy ks_2samp) of the first feature as column "ks_p". Small windows are
    noisy: 60 values in 10 bins can show PSI > 0.25 with no drift at all."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def first_alarm(table: pd.DataFrame, threshold: float = 0.25):
    """The first date where ANY feature's PSI exceeds the threshold (columns other than ks_p), else None."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


class StreamingRolling:
    """Live feature: rolling mean and std (ddof=1) of the last `window` values, updated one value at a time in O(1)
    (running sums). Must equal pandas rolling on the same data: the train/serve parity test."""

    def __init__(self, window: int):
        self.window = window
        self.buf: deque = deque()
        self.s = self.s2 = 0.0

    def update(self, x: float) -> tuple[float, float]:
        """Add x (drop the oldest if full); return (mean, std), NaN until `window` values have been seen."""
        raise NotImplementedError("✍️ Your turn: see the docstring")


def shadow_compare(returns: pd.Series, champion: pd.Series, challenger: pd.Series, min_days: int = 60,
                   margin: float = 0.3) -> dict:
    """Shadow mode: both models' positions (decided at t, earning returns[t+1]) are logged; only the champion trades.
    Return {"sharpe_champion", "sharpe_challenger" (annualized), "agreement" (share of days with the same position
    sign), "promote": at least min_days of results and challenger Sharpe > champion Sharpe + margin}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
