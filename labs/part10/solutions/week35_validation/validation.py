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
        # >>> SOLUTION
        t0 = self.t1.index
        n, emb = len(t0), int(np.ceil(self.embargo * len(t0)))
        for test in np.array_split(np.arange(n), self.n_splits):
            test_start, test_end = t0[test[0]], self.t1.iloc[test].max()
            emb_end = t0[min(test[-1] + emb, n - 1)]
            train = np.where((self.t1.to_numpy() < test_start) | (t0 > max(test_end, emb_end)))[0]
            yield train, test
        # <<< SOLUTION


def cv_scores(model, X: pd.DataFrame, y: pd.Series, cv, weights=None) -> pd.DataFrame:
    """For each (train, test) of cv.split(X): fit a FRESH clone (sklearn.base.clone) with lb.fit_weighted (weights
    sliced to the train rows), predict_proba on test. One row per fold: log_loss, auc, accuracy (threshold 0.5)."""
    # >>> SOLUTION
    from sklearn.base import clone
    rows = []
    for tr, te in cv.split(X):
        w = None if weights is None else np.asarray(weights)[tr]
        m = lb.fit_weighted(clone(model), X.iloc[tr], y.iloc[tr], w)
        p = m.predict_proba(X.iloc[te])[:, 1]
        rows.append({"log_loss": log_loss(y.iloc[te], p, labels=[0, 1]), "auc": roc_auc_score(y.iloc[te], p),
                     "accuracy": accuracy_score(y.iloc[te], p > 0.5)})
    return pd.DataFrame(rows)
    # <<< SOLUTION


# ------------------------------------------------------------------------ S10 feature importance
def mdi_importance(fitted_forest, columns) -> pd.Series:
    """Mean Decrease Impurity: the fitted forest's feature_importances_ (in-sample, biased), sorted descending."""
    # >>> SOLUTION
    return pd.Series(fitted_forest.feature_importances_, index=list(columns)).sort_values(ascending=False)
    # <<< SOLUTION


def mda_importance(model, X: pd.DataFrame, y: pd.Series, cv, n_repeats: int = 3, seed: int = 0,
                   groups: dict[str, list[str]] | None = None) -> pd.Series:
    """Lesson plan S10: out-of-fold log-loss INCREASE when a feature is shuffled (fit a clone per fold; permute with
    rng = default_rng(seed), n_repeats per feature per fold; mean over all). With `groups` ({name: [columns]}),
    permute each group's columns TOGETHER with one permutation (clustered MDA). Sorted descending."""
    # >>> SOLUTION
    from sklearn.base import clone
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
    # <<< SOLUTION


def sfi_importance(model, X: pd.DataFrame, y: pd.Series, cv) -> pd.Series:
    """Single Feature Importance: mean out-of-fold AUC of the model trained on each feature ALONE (no substitution
    effects, no joint effects). Sorted descending."""
    # >>> SOLUTION
    return pd.Series({c: float(cv_scores(model, X[[c]], y, cv)["auc"].mean()) for c in X.columns}
                     ).sort_values(ascending=False)
    # <<< SOLUTION


def cluster_features(X: pd.DataFrame, threshold: float = 0.5) -> dict[str, list[str]]:
    """Group features whose |correlation| is high: distance 1 − |ρ|, average linkage, fcluster(criterion="distance",
    t=threshold). Name each cluster by its first feature (column order); members in column order."""
    # >>> SOLUTION
    d = 1 - X.corr().abs().to_numpy()
    np.fill_diagonal(d, 0.0)
    lab = fcluster(linkage(squareform(d, checks=False), "average"), t=threshold, criterion="distance")
    out: dict[str, list[str]] = {}
    for col, k in zip(X.columns, lab):
        members = [c for c, kk in zip(X.columns, lab) if kk == k]
        out.setdefault(members[0], members)
    return out
    # <<< SOLUTION


# ----------------------------------------------------------------------------- S11 audit
def shuffled_labels_test(model, X, y, cv, seed: int = 0, tol: float = 0.05) -> dict:
    """Refit under cv with y randomly permuted: mean AUC must fall to chance. Return {"auc": …, "passed":
    |auc − 0.5| < tol}."""
    # >>> SOLUTION
    ys = pd.Series(np.random.default_rng(seed).permutation(y.to_numpy()), index=y.index)
    auc = float(cv_scores(model, X, ys, cv)["auc"].mean())
    return {"auc": auc, "passed": abs(auc - 0.5) < tol}
    # <<< SOLUTION


def canary_test(model, X, y, cv, seed: int = 0) -> dict:
    """Add a deliberately LEAKED feature "canary" = y + N(0, 0.5) noise and run mda_importance (n_repeats=1).
    The audit machinery works if the canary ranks FIRST. Return {"rank": 1-based rank of canary, "passed"}."""
    # >>> SOLUTION
    rng = np.random.default_rng(seed)
    Xc = X.assign(canary=y.to_numpy() + rng.normal(0, 0.5, len(y)))
    imp = mda_importance(model, Xc, y, cv, n_repeats=1, seed=seed)
    rank = int(list(imp.index).index("canary")) + 1
    return {"rank": rank, "passed": rank == 1}
    # <<< SOLUTION


def time_shift_test(model, X, y, cv, margin: float = 0.02) -> dict:
    """Use the features one bar LATER (X.shift(1), first row filled by bfill): information arrives later, so the mean
    CV AUC must not IMPROVE by more than `margin`. Return {"auc": …, "auc_shifted": …, "passed"}."""
    # >>> SOLUTION
    a = float(cv_scores(model, X, y, cv)["auc"].mean())
    b = float(cv_scores(model, X.shift(1).bfill(), y, cv)["auc"].mean())
    return {"auc": a, "auc_shifted": b, "passed": b <= a + margin}
    # <<< SOLUTION


def overlap_test(train_idx, test_idx, t1: pd.Series) -> dict:
    """No training label may overlap the test span [first test t0, last test t1]. Return {"n_overlap", "passed"}."""
    # >>> SOLUTION
    t0 = t1.index
    lo, hi = t0[test_idx].min(), t1.iloc[test_idx].max()
    tr_t0, tr_t1 = t0[train_idx], t1.iloc[train_idx].to_numpy()
    n = int(((tr_t1 >= lo) & (tr_t0 <= hi)).sum())
    return {"n_overlap": n, "passed": n == 0}
    # <<< SOLUTION


# ------------------------------------------------------------- S12 tuning, ensembles, registry
def optuna_search(X, y, cv, n_trials: int = 10, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Optuna (TPESampler(seed=seed), direction "minimize") over LGBMClassifier(n_estimators=100, verbose=-1,
    random_state=0) with num_leaves in [4, 32] (int), learning_rate in [0.01, 0.2] (log), min_child_samples in
    [10, 100] (int); objective = mean cv_scores log_loss. Return (best params, DataFrame of ALL trials with columns
    number, value + the three parameters): every trial counts for the Deflated Sharpe Ratio."""
    # >>> SOLUTION
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {"num_leaves": trial.suggest_int("num_leaves", 4, 32),
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  "min_child_samples": trial.suggest_int("min_child_samples", 10, 100)}
        model = LGBMClassifier(n_estimators=100, verbose=-1, random_state=0, **params)
        return float(cv_scores(model, X, y, cv)["log_loss"].mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = pd.DataFrame([{"number": t.number, "value": t.value, **t.params} for t in study.trials])
    return study.best_params, trials
    # <<< SOLUTION


def seed_ensemble(make_model: Callable[[int], object], X_train, y_train, X_test, seeds=(0, 1, 2, 3, 4)) -> np.ndarray:
    """Average predict_proba[:, 1] of make_model(seed) fitted on the same data, one per seed."""
    # >>> SOLUTION
    return np.mean([make_model(s).fit(X_train, y_train).predict_proba(X_test)[:, 1] for s in seeds], axis=0)
    # <<< SOLUTION


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
        # >>> SOLUTION
        if "audit_passed" not in meta:
            raise ValueError("meta must record audit_passed")
        versions = self.data.setdefault(name, [])
        versions.append({"version": len(versions) + 1, "stage": "research", "meta": meta})
        self._save()
        return len(versions)
        # <<< SOLUTION

    def promote(self, name: str, version: int, stage: str) -> None:
        """Move a version to `stage`. ValueError if the stage is not the NEXT one, or if the audit did not pass."""
        # >>> SOLUTION
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
        # <<< SOLUTION

    def get(self, name: str, stage: str) -> dict | None:
        """The HIGHEST version currently in `stage`, or None."""
        # >>> SOLUTION
        found = [v for v in self.data.get(name, []) if v["stage"] == stage]
        return max(found, key=lambda v: v["version"]) if found else None
        # <<< SOLUTION
