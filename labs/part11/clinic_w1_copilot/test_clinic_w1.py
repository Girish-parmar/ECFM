import numpy as np
import pytest
from _loader import load
from common import make_filings

ce = load("clinic_w1_copilot", "copilot_eval")
FILINGS, FACTS, QS = make_filings()


@pytest.fixture(scope="module")
def index():
    return ce.build_index(FILINGS)


def test_build_index_can_drop_sections(index):
    chunks, _ = index
    tuned, _ = ce.build_index(FILINGS, drop_sections=("Preamble",))
    assert len(tuned) == len(chunks) - len(FILINGS) and all(c.section != "Preamble" for c in tuned)


def test_look_at_what_ranks_first(index):
    rt = ce.retrieval_table(index[1], index[0], QS)
    assert list(rt.index) == ["bm25", "dense", "hybrid"] and list(rt.columns) == ["recall@1", "recall@5", "mrr"]
    assert rt.loc["hybrid", "recall@5"] == 1.0 and rt.loc["hybrid", "recall@1"] < 0.2    # the cover page wins
    chunks, ret = ce.build_index(FILINGS, drop_sections=("Preamble",))
    tuned = ce.retrieval_table(ret, chunks, QS)
    assert tuned.loc["hybrid", "recall@1"] > 0.8 and tuned.loc["hybrid", "mrr"] > rt.loc["hybrid", "mrr"] + 0.4


def test_answers_are_faithful_right_and_refuse_correctly(index):
    chunks, ret = index
    at = ce.answer_table(ret, chunks, QS, FACTS)
    assert len(at) == len(QS) and list(at.columns) == ["kind", "faithful", "refusal_ok", "numeric_ok"]
    assert at["faithful"].min() == 1.0 and at["refusal_ok"].all()
    assert at.loc[at["kind"] == "numeric", "numeric_ok"].all() and at.loc[at["kind"] != "numeric", "numeric_ok"].isna().all()


def test_adversarial_suite_needs_sanitizing(index):
    chunks, ret = index
    assert ce.adversarial_pass_rate(ret, chunks, QS, FACTS, sanitize=True) == 1.0
    raw = ce.adversarial_pass_rate(ret, chunks, QS, FACTS, sanitize=False)
    assert raw == pytest.approx(2 / 3)                                               # the injected filing wins
    a = ce.answer(ret, chunks, next(q for q in QS if q["ticker"] == "EPIC" and "2024" in q["question"]))
    assert a["flagged"] == 1 and not any("ignore all previous" in t.lower() for t in a["docs"].values())


def test_report():
    r = ce.report()
    assert r["n_questions"] == len(QS) and r["adversarial_raw"] < r["adversarial_sanitized"] == 1.0
    assert np.isclose(r["faithfulness"], 1.0) and r["numeric_accuracy"] == 1.0
