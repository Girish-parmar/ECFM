"""Week 35 (S9–S12) — Purged cross-validation, feature importance (MDI, MDA, SFI, clustered MDA), the leakage and
overfitting audit, hyperparameter search that logs every trial, seed ensembles and a model registry.

A shuffled k-fold on overlapping labels finds "skill" in pure noise. Purge, embargo, and audit before you believe.
Fill in every block marked "Your turn", then run:  python -m pytest week35_validation
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from _loader import load

lb = load("week34_labels", "labels")


# ------------------------------------------------------------------------------ S9 purged CV
class PurgedKFold:
    """Lesson plan S9, scikit-learn compatible: contiguous test folds (np.array_split of the positions); a training
    sample is kept only if its label ENDS before the test fold starts (t1 < first test t0) or it STARTS after both the
    last test label's end and the embargo end (t0 > max(test t1), t0 > t0[last test + embargo]). embargo = fraction
    of samples, rounded up. t1 is indexed by t0."""

    def __init__(self, t1: pd.Series, n_splits: int = 5, embargo: float = 0.01):
        self.t1, self.n_splits, self.embargo = t1, n_splits, embargo

    def get_n_splits(self, *args, **kwargs):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        raise NotImplementedError("✍️ Your turn: see the docstring")


def cv_scores(model, X: pd.DataFrame, y: pd.Series, cv, weights=None) -> pd.DataFrame:
    """For each (train, test) of cv.split(X): fit a FRESH clone (sklearn.base.clone) with lb.fit_weighted (weights
    sliced to the train rows), predict_proba on test. One row per fold: log_loss, auc, accuracy (threshold 0.5)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------ S10 feature importance
def mdi_importance(fitted_forest, columns) -> pd.Series:
    """Mean Decrease Impurity: the fitted forest's feature_importances_ (in-sample, biased), sorted descending."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def mda_importance(model, X: pd.DataFrame, y: pd.Series, cv, n_repeats: int = 3, seed: int = 0,
                   groups: dict[str, list[str]] | None = None) -> pd.Series:
    """Lesson plan S10: out-of-fold log-loss INCREASE when a feature is shuffled (fit a clone per fold; permute with
    rng = default_rng(seed), n_repeats per feature per fold; mean over all). With `groups` ({name: [columns]}),
    permute each group's columns TOGETHER with one permutation (clustered MDA). Sorted descending."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def sfi_importance(model, X: pd.DataFrame, y: pd.Series, cv) -> pd.Series:
    """Single Feature Importance: mean out-of-fold AUC of the model trained on each feature ALONE (no substitution
    effects, no joint effects). Sorted descending."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def cluster_features(X: pd.DataFrame, threshold: float = 0.5) -> dict[str, list[str]]:
    """Group features whose |correlation| is high: distance 1 − |ρ|, average linkage, fcluster(criterion="distance",
    t=threshold). Name each cluster by its first feature (column order); members in column order."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ----------------------------------------------------------------------------- S11 audit
def shuffled_labels_test(model, X, y, cv, seed: int = 0, tol: float = 0.05) -> dict:
    """Refit under cv with y randomly permuted: mean AUC must fall to chance. Return {"auc": …, "passed":
    |auc − 0.5| < tol}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def canary_test(model, X, y, cv, seed: int = 0) -> dict:
    """Add a deliberately LEAKED feature "canary" = y + N(0, 0.5) noise and run mda_importance (n_repeats=1).
    The audit machinery works if the canary ranks FIRST. Return {"rank": 1-based rank of canary, "passed"}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def time_shift_test(model, X, y, cv, margin: float = 0.02) -> dict:
    """Use the features one bar LATER (X.shift(1), first row filled by bfill): information arrives later, so the mean
    CV AUC must not IMPROVE by more than `margin`. Return {"auc": …, "auc_shifted": …, "passed"}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def overlap_test(train_idx, test_idx, t1: pd.Series) -> dict:
    """No training label may overlap the test span [first test t0, last test t1]. Return {"n_overlap", "passed"}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------- S12 tuning, ensembles, registry
def optuna_search(X, y, cv, n_trials: int = 10, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Optuna (TPESampler(seed=seed), direction "minimize") over LGBMClassifier(n_estimators=100, verbose=-1,
    random_state=0) with num_leaves in [4, 32] (int), learning_rate in [0.01, 0.2] (log), min_child_samples in
    [10, 100] (int); objective = mean cv_scores log_loss. Return (best params, DataFrame of ALL trials with columns
    number, value + the three parameters): every trial counts for the Deflated Sharpe Ratio."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def seed_ensemble(make_model: Callable[[int], object], X_train, y_train, X_test, seeds=(0, 1, 2, 3, 4)) -> np.ndarray:
    """Average predict_proba[:, 1] of make_model(seed) fitted on the same data, one per seed."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


class ModelRegistry:
    """A tiny file-backed registry (the platform uses MLflow): versions per model name, each with its metadata and a
    stage. Stages move ONE step at a time: research → shadow → paper → live, and only if the audit passed.
    Promoting a version to "live" moves the previous live version of that model back to "paper"."""

    STAGES = ["research", "shadow", "paper", "live"]

    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=1, default=str))

    def register(self, name: str, meta: dict) -> int:
        """Add a new version (1, 2, …) in stage "research" with `meta` (must contain "audit_passed"); save; return
        the version number."""
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def promote(self, name: str, version: int, stage: str) -> None:
        """Move a version to `stage`. ValueError if the stage is not the NEXT one, or if the audit did not pass."""
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def get(self, name: str, stage: str) -> dict | None:
        """The HIGHEST version currently in `stage`, or None."""
        raise NotImplementedError("✍️ Your turn: see the docstring")
