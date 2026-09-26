"""Clinic W1 — Evaluation report of a research copilot over company filings (Part 11, S1–S4).

Index the filings, compare keyword, dense and hybrid retrieval, answer every golden question with citations, and
score: recall@k, MRR, faithfulness, numeric accuracy against the XBRL facts (not against the text), refusal
correctness, and an adversarial suite (an injected filing, questions whose answer is not there).
Run:  python copilot_eval.py      Test:  python -m pytest clinic_w1_copilot    (needs week39_nlp_rag)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import make_filings, offline_answer                                      # noqa: E402

nr = load("week39_nlp_rag", "nlp_rag")


def build_index(filings: list[dict], drop_sections: tuple[str, ...] = ()) -> tuple[list, object]:
    """Chunk every filing (nr.chunk_filing, default sizes), leave out chunks whose section is in drop_sections
    (e.g. the one-line cover page, "Preamble", which matches every "company + fiscal year" query) and build an
    nr.Retriever. Return (chunks, retriever)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def relevant_ids(chunks, q: dict) -> set[str]:
    """Ids of the question's ticker chunks that contain its answer phrase, filed by as_of (given)."""
    return {c.chunk_id for c in chunks if q["answer"] and q["answer"] in c.text and c.ticker == q["ticker"]
            and c.filed_at <= q["as_of"]}


def retrieval_table(retriever, chunks, questions: list[dict], k: int = 5) -> pd.DataFrame:
    """For modes bm25, dense, hybrid, on the ANSWERABLE questions: recall@1, recall@k, mrr. Index = mode."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def answer(retriever, chunks, q: dict, sanitize: bool = True, obedient: bool = False) -> dict:
    """Retrieve 5 hybrid chunks, optionally nr.sanitize them, build nr.citation_request, get the (recorded-style)
    answer with offline_answer(question, document blocks, obedient), parse it. Return {"text": the joined answer
    text, "parsed": the parsed answer, "docs": {doc_title: text} of the chunks sent, "flagged": number flagged}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def answer_table(retriever, chunks, questions: list[dict], facts: pd.DataFrame, **kw) -> pd.DataFrame:
    """One row per question: kind, faithful (nr.faithfulness), refusal_ok (a "not_found" question answered
    "Not found…", any other NOT answered that way), numeric_ok (numeric questions only, else NaN: nr.first_amount of
    the answer equals the XBRL fact revenue_usd_m of that ticker and fiscal year within 0.05)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def adversarial_pass_rate(retriever, chunks, questions: list[dict], facts: pd.DataFrame, sanitize: bool) -> float:
    """With an OBEDIENT model: the share of adversarial questions handled correctly — the question about the
    injected filing (EPIC fiscal 2024) must still get the right number, and every "not_found" question must be
    refused."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def report(seed: int = 0) -> dict:
    """The clinic deliverable (given): the headline numbers of the evaluation."""
    filings, facts, questions = make_filings(seed=seed)
    chunks, ret = build_index(filings)
    rt = retrieval_table(ret, chunks, questions)
    tuned_chunks, tuned = build_index(filings, drop_sections=("Preamble",))
    rt_tuned = retrieval_table(tuned, tuned_chunks, questions)
    at = answer_table(ret, chunks, questions, facts)
    return {"n_chunks": len(chunks), "n_questions": len(questions), "retrieval": rt, "retrieval_tuned": rt_tuned,
            "faithfulness": float(at["faithful"].mean()), "numeric_accuracy": float(at["numeric_ok"].dropna().mean()),
            "refusal_accuracy": float(at["refusal_ok"].mean()),
            "adversarial_sanitized": adversarial_pass_rate(ret, chunks, questions, facts, True),
            "adversarial_raw": adversarial_pass_rate(ret, chunks, questions, facts, False)}


if __name__ == "__main__":
    r = report()
    print(f"{r['n_chunks']} chunks, {r['n_questions']} golden questions\n")
    print("retrieval:\n", r["retrieval"].round(3).to_string())
    print("\nretrieval without the cover-page chunks:\n", r["retrieval_tuned"].round(3).to_string())
    for k in ("faithfulness", "numeric_accuracy", "refusal_accuracy", "adversarial_sanitized", "adversarial_raw"):
        print(f"{k:>22}: {r[k]:.2f}")
