"""Clinic W3 — Full ML research report for a meta-labeled momentum strategy (Part 10, weeks 33–36).

CV comparison (shuffled k-fold vs purged k-fold) for three model families, the leakage audit, and the walk-forward
trading result of each family against the primary rule alone — with the number of trials you ran, for the DSR.
Run:  python ml_report.py       Test:  python -m pytest clinic_w3_ml_report   (needs weeks 33–36)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import regime_market                                                     # noqa: E402

lb = load("week34_labels", "labels")
va = load("week35_validation", "validation")
st = load("week36_strategies", "strategies")
KINDS = ("logit", "rf", "lgbm")


def research_dataset(n: int = 6000, seed: int = 1) -> pd.DataFrame:
    """The meta-labeling dataset of the week 36 strategy on the synthetic regime market (given)."""
    return st.meta_dataset(regime_market(n, seed=seed))


def cv_comparison(ds: pd.DataFrame) -> pd.DataFrame:
    """For each model kind: mean AUC under a SHUFFLED KFold(5, shuffle=True, random_state=0) and under
    va.PurgedKFold(t1, 5, 0.01), both with the uniqueness weights w. Index = kind; columns shuffled_auc, purged_auc,
    inflation (shuffled − purged)."""
    # >>> SOLUTION
    X, y, w = ds[st.FEATURES], ds["y"], ds["w"]
    rows = {}
    for kind in KINDS:
        m = lb.make_model(kind)
        a = va.cv_scores(m, X, y, KFold(5, shuffle=True, random_state=0), w)["auc"].mean()
        b = va.cv_scores(m, X, y, va.PurgedKFold(ds["t1"], 5, 0.01), w)["auc"].mean()
        rows[kind] = {"shuffled_auc": float(a), "purged_auc": float(b), "inflation": float(a - b)}
    return pd.DataFrame(rows).T
    # <<< SOLUTION


def audit(ds: pd.DataFrame, kind: str = "logit") -> dict:
    """Run the week 35 audit on the purged CV: shuffled labels, canary, time shift, and the overlap test on every
    purged fold. Return {"shuffled_labels", "canary", "time_shift", "overlap"} → passed (bool), plus "all"."""
    # >>> SOLUTION
    X, y = ds[st.FEATURES], ds["y"]
    cv = va.PurgedKFold(ds["t1"], 5, 0.01)
    m = lb.make_model(kind)
    out = {"shuffled_labels": va.shuffled_labels_test(m, X, y, cv)["passed"],
           "canary": va.canary_test(m, X, y, cv)["passed"],
           "time_shift": va.time_shift_test(m, X, y, cv)["passed"],
           "overlap": all(va.overlap_test(tr, te, ds["t1"])["passed"] for tr, te in cv.split(X))}
    out["all"] = all(out.values())
    return out
    # <<< SOLUTION


def trading_table(ds: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward (st.walk_forward_meta) for each kind; st.meta_summary with years = calendar span of the OOS events.
    Rows: "primary" (from the first run) and each kind (its meta row). Columns trades, precision, total, sharpe."""
    # >>> SOLUTION
    rows = {}
    for kind in KINDS:
        oos = st.walk_forward_meta(ds, kind)
        s = st.meta_summary(oos, (oos.index[-1] - oos.index[0]).days / 365.25)
        if not rows:
            rows["primary"] = s.loc["primary"]
        rows[kind] = s.loc["meta"]
    return pd.DataFrame(rows).T
    # <<< SOLUTION


def n_trials(cv_table: pd.DataFrame, trade_table: pd.DataFrame) -> int:
    """Trials to report to the DSR: every CV evaluation (2 per kind) + every walk-forward run (1 per kind) (given)."""
    return 2 * len(cv_table) + (len(trade_table) - 1)


if __name__ == "__main__":
    ds = research_dataset()
    print(f"{len(ds)} events, meta-label base rate {ds['y'].mean():.3f}, mean uniqueness {ds['w'].mean():.2f}\n")
    cvt = cv_comparison(ds)
    print("CV (AUC):\n", cvt.round(3).to_string())
    print("\naudit:", audit(ds))
    tt = trading_table(ds)
    print("\nwalk-forward, after costs:\n", tt.round(3).to_string())
    print(f"\ntrials to report to the Deflated Sharpe Ratio: {n_trials(cvt, tt)}")
    print(f"best meta Sharpe {tt.drop('primary')['sharpe'].max():.2f} vs primary {tt.loc['primary', 'sharpe']:.2f}")
