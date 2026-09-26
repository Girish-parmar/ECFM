import numpy as np
import pytest
from scipy.stats import norm
from _loader import load

ov = load("week28_overfitting", "overfitting")


def test_expected_max_sharpe_grows_with_trials():
    rng = np.random.default_rng(0)
    few, many = rng.normal(0, 0.03, 5), rng.normal(0, 0.03, 1000)
    assert ov.expected_max_sharpe(many) > ov.expected_max_sharpe(few) > 0
    s = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    n, v, g = 5, s.var(ddof=1), 0.5772156649015329
    want = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
    assert ov.expected_max_sharpe(s) == pytest.approx(want)
    # simulation: the best of 200 unskilled strategies (1000 days each) is about as good as SR0 predicts
    sims = rng.normal(0, 0.01, (1000, 200))
    srs = sims.mean(0) / sims.std(0, ddof=1)
    assert srs.max() == pytest.approx(ov.expected_max_sharpe(srs), rel=0.35)


def test_deflated_sharpe_punishes_the_search():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0006, 0.01, 1000)                                    # annual Sharpe ≈ 1
    after_3 = ov.deflated_sharpe(r, rng.normal(0, 0.02, 3))[0]
    after_1000 = ov.deflated_sharpe(r, rng.normal(0, 0.02, 1000))[0]
    assert after_1000 < after_3 and after_1000 < 0.95
    dsr, sr0 = ov.deflated_sharpe(r, rng.normal(0, 0.02, 50))
    assert dsr == pytest.approx(ov.an.psr(r, sr0))


def _configs(skill: bool, seed: int, T=1000, N=50):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, (T, N))
    return noise + (np.linspace(0, 0.002, N) if skill else 0.0)


def test_pbo_separates_noise_from_skill():
    noise = np.mean([ov.pbo_cscv(_configs(False, s), S=10)[0] for s in range(6)])
    skill = np.mean([ov.pbo_cscv(_configs(True, s), S=10)[0] for s in range(6)])
    assert 0.3 < noise < 0.7 and skill < 0.15                              # lesson plan: ≈ 0.48 vs ≈ 0.07
    pbo, logits = ov.pbo_cscv(_configs(True, 0), S=8)
    assert logits.size == 70 and pbo == pytest.approx(np.mean(logits <= 0))


def _run_factory(edge=0.0008, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, 2000)

    def run(params, delay, cost_mult):
        lb = params["lookback"]
        drift = edge * (1 - abs(lb - 100) / 400)                          # a broad plateau around 100
        pnl = base + drift - (1 if delay else 0) * edge * 0.3
        return pnl - cost_mult * 0.0001
    return run


def test_robustness_scorecard():
    card = ov.robustness_scorecard(_run_factory(), {"lookback": 100, "threshold": 1.5})
    assert list(card.columns) == ["test", "value", "passed"]
    assert card["test"].tolist() == ["lookback -20%", "lookback +20%", "delay 1 bar", "costs x2", "without best 5 days",
                                     "first half", "second half"]
    assert card["passed"].all()
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.01, 2000)

    def fragile(params, delay, cost_mult):                              # edge only in the first half, gone with delay
        edge = np.where(np.arange(2000) < 1000, 0.0015, -0.0005) * (0 if delay else 1)
        return noise + edge - cost_mult * 0.0001
    weak = ov.robustness_scorecard(fragile, {"lookback": 100})
    failed = set(weak.loc[~weak["passed"], "test"])
    assert {"delay 1 bar", "second half"} <= failed


def test_validation_gate_and_dossier():
    rng = np.random.default_rng(4)
    good = rng.normal(0.0008, 0.008, 1500)
    card = ov.robustness_scorecard(_run_factory(), {"lookback": 100})
    g = ov.validation_gate(good, rng.normal(0, 0.01, 10), pbo=0.1, scorecard=card)
    assert list(g["checks"]) == ["oos_sharpe", "dsr", "pbo", "robustness", "max_dd"] and g["passed"]
    bad = ov.validation_gate(good, rng.normal(0, 0.01, 10), pbo=0.45, scorecard=card)
    assert not bad["passed"] and not bad["checks"]["pbo"][1] and bad["checks"]["oos_sharpe"][1]
    strict = ov.validation_gate(good, rng.normal(0, 0.01, 10), 0.1, card, criteria={"min_oos_sharpe": 5})
    assert not strict["passed"]
    md = ov.dossier_markdown("tsmom", g, card)
    assert md.startswith("# Validation dossier: tsmom") and "**Verdict: PASS**" in md and "| pbo | 0.100 | yes |" in md
