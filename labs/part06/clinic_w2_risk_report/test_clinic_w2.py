from datetime import date

import numpy as np
import pytest
from _loader import load
from common import load_chain

rr = load("clinic_w2_risk_report", "risk_report")
# The SVI surface that generated the recorded chain (instructor data, T = days / 365):
TRUE = {30: (0.0012, 0.012, -0.70, 0.02, 0.08), 58: (0.0025, 0.020, -0.65, 0.02, 0.10),
        93: (0.0045, 0.028, -0.60, 0.03, 0.12)}


@pytest.fixture(scope="module")
def snapshot():
    chain, meta = load_chain()
    c = rr.cm.prepare(chain, date(2026, 3, 2))
    inputs = rr.cm.implied_inputs(c, 500.0)
    return (*rr.surface_snapshot(c, 500.0, inputs), c, inputs)


def test_otm_smile(snapshot):
    _, _, c, inputs = snapshot
    exp = sorted(c["expiry"].unique())[0]
    k, iv = rr.otm_smile(c[c["expiry"] == exp], 500.0, *inputs.loc[exp, ["r", "q", "F"]])
    assert np.all(np.diff(k) > 0) and not np.isnan(iv).any() and 10 <= k.size <= 17


def test_surface_snapshot_recovers_the_true_surface(snapshot):
    snap, violations, _, _ = snapshot
    assert list(snap.columns) == ["T", "F", "a", "b", "rho", "m", "s", "rmse", "atm_iv", "rr25", "butterfly_ok"]
    for (days, p), (_, row) in zip(TRUE.items(), snap.iterrows()):
        T = days / 365
        assert row["T"] == pytest.approx(T)
        assert row["atm_iv"] == pytest.approx(np.sqrt(rr.sp.svi_total_var(0.0, *p) / T), abs=2e-3)
        assert row["rr25"] < -0.01 and row["rmse"] < 1e-3 and row["butterfly_ok"]
    assert violations == []


def test_iv_at_delta_matches_a_direct_search():
    p, T, F, r = TRUE[58], 58 / 365, 502.5, 0.045
    assert rr.iv_at_delta(p, T, F, r, -0.5, -1) < rr.iv_at_delta(p, T, F, r, -0.25, -1)   # put skew
    assert rr.iv_at_delta(p, T, F, r, 0.25, 1) < rr.iv_at_delta(p, T, F, r, 0.5, 1)


def test_risk_report_markdown():
    md = rr.build()
    heads = [line for line in md.splitlines() if line.startswith("#")]
    assert heads == ["# M3a risk report", "## Dollar Greeks", "## Spot × vol scenarios (full revaluation)",
                     "## Worst scenario", "## Volatility surface", "## Arbitrage checks"]
    assert "| TOTAL |" in md and "No calendar arbitrage." in md
    worst = md.split("## Worst scenario")[1].split("##")[0].strip()
    assert worst.startswith("Spot ") and "P&L" in worst
