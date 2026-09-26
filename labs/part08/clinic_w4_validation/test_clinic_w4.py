import pytest
from _loader import load
from common import regime_market

va = load("clinic_w4_validation", "validation")
BARS = regime_market(n_blocks=24, seed=21)


@pytest.fixture(scope="module")
def results():
    log = va.en.ResearchLog()
    out = {name: va.validate(name, BARS, fn, grid, log) for name, (fn, grid) in va.strategies(BARS).items()}
    return out, log


def test_every_configuration_is_logged(results):
    _, log = results
    t = log.trials()
    assert len(t) == 8 + 9 + 40 and list(t["run_id"]) == list(range(1, 58))
    assert set(t["strategy"]) == {"tsmom", "sma_cross", "noise_miner"}


def test_noise_miner_is_rejected(results):
    out, _ = results
    nm = out["noise_miner"]
    assert not nm["gate"]["passed"] and nm["pbo"] > 0.5
    assert nm["gate"]["checks"]["robustness"][0] < 0.5 and not nm["gate"]["checks"]["dsr"][1]


def test_dossiers(results):
    out, _ = results
    for name, r in out.items():
        assert r["dossier"].startswith(f"# Validation dossier: {name}")
        assert len(r["wf"]["oos"]) == len(r["wf"]["params"]) * 250
        assert set(r["gate"]["checks"]) == {"oos_sharpe", "dsr", "pbo", "robustness", "max_dd"}
    assert out["tsmom"]["scorecard"]["passed"].mean() > out["noise_miner"]["scorecard"]["passed"].mean()
