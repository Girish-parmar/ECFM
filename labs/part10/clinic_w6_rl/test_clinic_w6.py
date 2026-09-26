import pytest
from _loader import load

rd = load("clinic_w6_rl", "rl_defense")


@pytest.fixture(scope="module")
def table():
    return rd.robustness_table()


def test_robustness_table(table):
    assert list(table.index) == [0.3, 0.0, -0.3]
    assert {"agent_mean", "agent_std", "agent_min", "long_only", "flat", "momentum", "reversal",
            "best_rule"} <= set(table)
    assert list(table["best_rule"]) == ["momentum", "long_only", "reversal"]
    assert table.loc[0.0, "agent_mean"] == pytest.approx(0.0, abs=0.05)             # noise: it learns to stay out
    assert table.loc[-0.3, "agent_min"] > 0                                         # reversal: every seed learns it


def test_no_agent_beats_the_best_rule_and_seeds_matter(table):
    for _, row in table.iterrows():
        assert row["agent_mean"] <= row[row["best_rule"]] + 1e-9
    assert table.loc[0.3, "agent_std"] > 0.3                                        # one seed would have misled you


def test_cost_assumption():
    cs = rd.cost_sensitivity(seeds=range(3))
    assert cs["turnover_optimistic"] > 100 and cs["optimistic"] < 0                 # free trades → churn → losses
    assert cs["realistic"] > cs["optimistic"] and cs["turnover_realistic"] < cs["turnover_optimistic"]


def test_evaluate_uses_the_training_scale():
    tr, te = rd.regime_returns(0.3)
    assert rd.evaluate(rd.RULES["flat"], te, tr) == 0.0
    assert rd.evaluate(rd.RULES["long_only"], te, tr, cost=0.0) == pytest.approx(te[10:].sum(), rel=1e-4)
