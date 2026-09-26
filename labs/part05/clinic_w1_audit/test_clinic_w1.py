import numpy as np
import pandas as pd
from _loader import load
from common import synthetic_ohlcv

au = load("clinic_w1_audit", "audit")
X = synthetic_ohlcv(600, seed=1)["close"].to_numpy()
sma = load("week17_core", "core").sma


# Look-ahead hall of shame: plausible-looking indicators with classic bugs
def centered_ma(x, n=5):                      # uses bars t-2..t+2
    return pd.Series(x).rolling(n, center=True).mean().to_numpy()


def dropped_warmup(x, n=5):                   # returns a shorter array
    return pd.Series(x).rolling(n).mean().dropna().to_numpy()


def backfilled(x, n=5):                       # fills the warm-up with a future value
    return pd.Series(x).rolling(n).mean().bfill().to_numpy()


def zscore_full_sample(x):                    # normalizes with the mean/std of the WHOLE series
    return (x - x.mean()) / x.std()


def momentum_points(x, n=5):                  # price difference: fine, but NOT scale invariant
    out = np.full(x.shape, np.nan)
    out[n:] = x[n:] - x[:-n]
    return out


def test_checks_catch_each_bug():
    assert au.check_length(dropped_warmup, X) == (False, "len 596 != 600")
    assert au.check_warmup(backfilled, X, 4) == (False, "value before look-back at 0")
    assert au.check_no_lookahead(centered_ma, X) == (False, "changes when bars after 300 are removed")
    assert au.check_no_lookahead(zscore_full_sample, X)[0] is False
    assert au.check_scale_invariant(momentum_points, X) == (False, "not scale invariant")
    assert au.check_warmup(lambda x: sma(x, 5), X, 4) == (True, "")
    nan_hole = lambda x: np.where(np.arange(x.size) == 50, np.nan, sma(x, 5))  # noqa: E731
    assert au.check_warmup(nan_hole, X, 4) == (False, "NaN after look-back at 50")


def test_audit_registry_is_clean():
    rows = au.audit(au.registry_close_only(), X, {"rsi"})
    assert len(rows) == 5 * 3 + 1
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert [r["check"] for r in rows if r["name"] == "rsi"] == ["length", "warmup", "no_lookahead", "scale_invariant"]


def test_audit_reports_failures_by_name():
    rows = au.audit({"centered": (centered_ma, 4), "mom": (momentum_points, 5)}, X, {"mom"})
    failed = {(r["name"], r["check"]) for r in rows if not r["ok"]}
    assert failed == {("centered", "warmup"), ("centered", "no_lookahead"), ("mom", "scale_invariant")}
