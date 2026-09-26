import numpy as np
import pytest
from _loader import load
from common import regime_market

pa = load("clinic_w1_parity", "parity")
BARS = regime_market(n_blocks=8, seed=5)


@pytest.fixture(scope="module")
def sigs():
    return pa.signals(BARS)


def test_engine_returns_align_with_quick_eval(sigs):
    r = pa.engine_returns(BARS, sigs["tsmom"])
    assert r.shape == (len(BARS),) and r[-1] == 0.0


def test_parity_report(sigs):
    rep = pa.parity_report(BARS, sigs)
    assert list(rep.index) == ["tsmom", "sma_cross", "rsi2"]
    assert list(rep.columns) == ["sharpe_quick", "sharpe_engine", "corr", "total_quick", "total_engine"]
    assert (rep["corr"] > 0.99).all()
    assert np.allclose(rep["sharpe_quick"], rep["sharpe_engine"], atol=0.02)


def test_cost_sensitivity(sigs):
    cs = pa.cost_sensitivity(BARS, sigs, multiples=(0, 1, 3))
    assert list(cs.columns) == [0, 1, 3]
    assert (cs[0] > cs[1]).all() and (cs[1] > cs[3]).all()                  # every edge erodes with costs
