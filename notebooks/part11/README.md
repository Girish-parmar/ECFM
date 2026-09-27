# Part 11 guided notebooks — AI, NLP & Cowork in Trading

Guided notebooks for Part 11 ([lesson plan](../../docs/lessons/PART_11_AI_NLP_COWORK.md)), one notebook per session.
The setup, data and plotting code is written for you; learners fill in short **✍️ Your turn** cells, replacing each
`...`. Each exercise ends with `p.check(...)`, which prints ✅ or ❌. If an answer isn't right yet, the notebook
continues with the reference value, so later cells still run.

**Nothing here calls the Claude API, and nothing is billed.** Model outputs are recorded responses built as real
`anthropic.types.Message` objects, so the code handles exactly what `client.messages.create(...)` returns: `stop_reason`,
text and `tool_use` blocks, citations, and `usage` with cache counters. The data is synthetic with known answers:
- earnings-call transcripts with gold labels;
- headlines that move prices (including traps for word lists);
- 10-K filings with an XBRL-like facts table and a golden question set;
- scripted agent models, one well-behaved and one rogue;
- an earnings-event market with post-earnings drift.

One transcript and one filing contain a **planted prompt injection**, on purpose: the notebooks show how to catch it.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part11/`](../../labs/part11/), with
the same definitions. The request dictionaries they build (`extraction_request`, `batch_requests`, `citation_request`)
can be passed to the `anthropic` client for a live run. Model IDs and token prices are illustrative; check the Anthropic
documentation for current values.

| Notebook | Session | Exercises |
|---|---|---|
| `01_llm_extraction.ipynb` | S1 | Reading a response safely (stop reason, then schema); numeric grounding. Plus: three recorded models scored field by field, an injected number that grounding cannot catch |
| `02_llm_sentiment_batch_event_study.ipynb` | S2 | The word-list baseline; a point-in-time daily signal. Plus: a Message Batches request with a cached rubric, LLM vs word list in an event study, the look-ahead of using publication time |
| `03_rag_ingest_retrieval.ipynb` | S3 | Point-in-time filtering; BM25; reciprocal rank fusion. Plus: recall@k and MRR by mode, the cover page that outranks every answer |
| `04_rag_answers_eval.ipynb` | S4 | A Citations request; faithfulness; sanitizing retrieved chunks. Plus: the copilot report (numbers against XBRL facts, refusals), an obedient model with and without sanitizing |
| `05_agent_tools.ipynb` | S5 | Denying and auditing tool calls; the defined-risk pre-check. Plus: a research run, a rogue model, tool and cost budgets |
| `06_n8n_endpoints.ipynb` | S6 | HMAC verification with replay protection; the news-alert rule. Plus: the signed kill-switch endpoint with FastAPI's TestClient, validating the n8n exports in `n8n/` |
| `07_events_early_warning.ipynb` | S7 | Post-earnings drift trades; an early-warning score from each feature's own past. Plus: drift by guidance label, costs, a risk-off overlay |
| `08_eval_cost_safety.ipynb` | S8 | The cost of a response; the cache hit rate; a regression gate. Plus: a day of headlines with and without caching and batching, a response cache keyed by prompt version, a rate limiter |

`p11lib.py` holds the reference implementations the checks compare against (the definitions of the labs) and the
synthetic data and recorded responses. `n8n/` holds the four workflow exports from the labs.

## Setup

```bash
cd notebooks/part11
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

## What the notebooks show

* **Grounding catches invented numbers, not injected ones.**
  * Grounding flags 6 of the sloppy model's 10 extractions (revenue read in billions, an EPS invented where none was
    stated). Its revenue precision is 0.7.
  * The obedient model's injected "revenue of 99999" *is* in the transcript, so grounding passes it; only screening the
    source catches it.
* **The LLM beats the word list, but not everywhere.**
  * The recorded LLM acts on 67% of headlines and gets the sign right 90% of the time; the word list, 59% and 84%.
  * After good news the LLM signal earns +2.1% over 5 days, against +1.4% for the word list. After bad news the word
    list does better (−2.9% vs −1.9%).
  * Using the publication time instead of the time the batch score existed inflates the LLM's +2.1% to +2.4%.
* **Look at what ranks first.** Every answer is in the top 5, but recall@1 is 0.06: the one-line cover page matches
  every "company + fiscal year" question. Dropping it lifts recall@1 to 0.88 and MRR from 0.43 to 0.97.
* **Sanitize before answering.** The copilot scores 1.00 on faithfulness, numbers (against the XBRL facts) and
  refusals. With an obedient model, the adversarial pass rate is 0.67 raw ("EPIC revenue doubled") and 1.00 sanitized.
* **The agent cannot trade.**
  * The rogue model's order and risk-limit change are denied and audited.
  * A looping model stops after 4 tool calls, an expensive one at the cost budget.
  * A "put spread" whose long leg is a call fails the defined-risk check.
* **The kill switch is signed.** Signed: 200. Replayed or forged: 401. Unsigned: 422. The workflow validator rejects an
  export that bypasses signing or contains a pasted key.
* **Drift and costs.** The planted post-earnings drift is +1.8% after raised guidance and −1.7% after lowered (t ≈ 6).
  An early-warning overlay that halves exposure on 5% of days cuts the maximum drawdown from −29% to about −20%.
* **Cost.** Scoring 800 headlines costs about $18 without caching, $7.94 with a cached rubric (hit rate 0.93) and $3.97
  as a batch.
* **Regression gate.** A prompt change that raises the golden-set score from 0.67 to 0.83 still fails the gate: it
  broke an item the old prompt got right.

## Instructor material

- `solutions/`: the same notebooks with the answers filled in. **Remove this folder (or keep it on a private branch)
  before sharing the repository with learners.**
- `tools/build_notebooks.py`: the single source for starter and solution notebooks (it also contains the answers).
  Edit content there and rebuild with `python tools/build_notebooks.py`.

## Tests

```bash
python -m pytest -q tests     # 24 tests: solutions pass every check (strict mode), starters run with blanks,
                              # starter and solution notebooks differ only in the exercise cells
```
