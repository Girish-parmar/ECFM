import numpy as np
import pytest
from _loader import load
from common import synthetic_universe

es = load("clinic_w2_edge_study", "edge_study")


@pytest.fixture(scope="module")
def table():
    uni = synthetic_universe(8, 800, seed=3)
    return es.edge_table(uni, {**es.PATTERNS, "lookahead_cheat": es.lookahead_cheat}, horizons=(3,), n_perm=1000)


def test_table_shape(table):
    assert list(table.columns) == ["pattern", "side", "symbol", "horizon", "n", "edge", "p_value", "hit_rate",
                                   "q_value", "discovery"]
    assert {"engulfing", "star", "lookahead_cheat"} <= set(table["pattern"])   # a pattern with no events has no rows
    assert (table["n"] > 0).all() and (table["q_value"] >= table["p_value"]).all()
    assert not ((table["pattern"] == "lookahead_cheat") & (table["side"] == "bear")).any()   # no -1 events


def test_honest_patterns_have_no_discoveries(table):
    honest = table[table["pattern"] != "lookahead_cheat"]
    assert honest["discovery"].sum() == 0


def test_the_lookahead_cheat_looks_spectacular(table):
    cheat = table[table["pattern"] == "lookahead_cheat"]
    assert cheat["discovery"].all() and (cheat["edge"] > 0).all() and (cheat["hit_rate"] > 0.6).all()


def test_summary(table):
    s = es.summary(table)
    assert list(s.columns) == ["pattern", "side", "tests", "events", "mean_edge", "share_p05", "discoveries"]
    row = s[(s["pattern"] == "lookahead_cheat") & (s["side"] == "bull")].iloc[0]
    assert row["tests"] == 8 and row["discoveries"] == 8 and row["share_p05"] == 1.0
    sub = table[(table["pattern"] == "engulfing") & (table["side"] == "bull")]
    e = s[(s["pattern"] == "engulfing") & (s["side"] == "bull")]["mean_edge"].iloc[0]
    assert e == pytest.approx(np.average(sub["edge"], weights=sub["n"]))
