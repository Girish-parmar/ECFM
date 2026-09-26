import numpy as np
import pytest
from _loader import load
from common import sector_universe

sc = load("clinic_w1_screen", "screen")
PRICES, SECTORS, TRUTH = sector_universe(n_sectors=5, per_sector=8, n_days=1000, pairs_per_sector=1, seed=4)


@pytest.fixture(scope="module")
def screened():
    return sc.screens(PRICES, SECTORS)


def test_screens(screened):
    sel, naive = screened
    assert set(zip(sel.a, sel.b)) == set(TRUTH)
    assert len(naive) > 0 and (naive.pvalue < 0.05).all() and not naive.selected.any()


def test_trade_pair_has_no_look_ahead():
    a, b = TRUTH[0]
    for hedge in ("fixed", "rolling", "kalman"):
        full = sc.trade_pair(PRICES[a], PRICES[b], hedge)
        cut = sc.trade_pair(PRICES[a].iloc[:900], PRICES[b].iloc[:900], hedge)
        assert len(full["pnl"]) == 250 and full["trades"] > 0
        np.testing.assert_allclose(cut["pnl"], full["pnl"][:150])                  # the future cannot change the past
    with pytest.raises(ValueError):
        sc.trade_pair(PRICES[a], PRICES[b], "magic")


def test_oos_report_fdr_pairs_beat_false_discoveries():
    rep = sc.oos_report(PRICES, SECTORS)
    assert list(rep.columns) == ["group", "pair", "hedge", "sharpe", "total", "trades"]
    assert len(rep) == 5 * 3 + 5
    m = rep.groupby(["group", "hedge"])["sharpe"].mean()
    assert m["selected", "fixed"] > m["naive_only", "fixed"] + 0.5
    assert m["selected", "fixed"] > 0.5


def test_kalman_delta_too_large_absorbs_the_spread():
    sr = {d: np.mean([sc.trade_pair(PRICES[a], PRICES[b], "kalman", delta=d)["sharpe"] for a, b in TRUTH])
          for d in (1e-5, 1e-7)}
    assert sr[1e-5] < 0 < sr[1e-7]
