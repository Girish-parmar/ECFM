# Part 11 labs — AI, NLP & Cowork in Trading

Auto-graded exercises for [Part 11](../../docs/lessons/PART_11_AI_NLP_COWORK.md) (program weeks 39–40, milestone
**M7b**). **The labs run offline and cost nothing.** Instead of calling the Claude API, they use recorded responses
built as real `anthropic.types.Message` objects, so your code handles exactly what `client.messages.create(...)`
returns: `stop_reason`, text and `tool_use` blocks, citations, `usage` with cache counters. The instructor notes ask
for recorded responses in labs. Live calls are for demos and your final run.

`common.py` provides the synthetic data, each with known answers:

* **Earnings-call transcripts** with gold labels. One contains a prompt injection.
* **Recorded extractions** from three kinds of model: careful, sloppy and obedient.
* **Timestamped headlines** that move prices, including traps for word lists.
* **Recorded LLM scores** that arrive three hours after the news.
* **10-K filings**, one of them injected, with an XBRL-like facts table and a golden question set.
* **A scripted agent model**, in a well-behaved and a rogue version.
* **An earnings-event market** with post-earnings drift.

Fill in each `raise NotImplementedError("✍️ Your turn ...")` and run the tests until they are green.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week39_nlp_rag/](week39_nlp_rag/) | S1–S4 | Extraction requests and Pydantic validation after a `stop_reason` check, numeric and evidence grounding, injection screening, field-level precision/recall. A lexicon baseline, Message Batches requests with a cached rubric, a score cache, a point-in-time daily signal, market-model event studies. Section chunking, point-in-time filtering, BM25, dense and hybrid (RRF) retrieval, cited-answer requests, sanitizing, recall@k, MRR, faithfulness | `python -m pytest week39_nlp_rag` |
| [week40_agents_ops/](week40_agents_ops/) | S5–S8 | An audited agent loop with read-only tools, a proposal queue, tool and cost budgets, and a risk pre-check. The HMAC-signed kill-switch API (FastAPI) and a validator for n8n workflow exports ([`n8n/`](week40_agents_ops/n8n/): pre-market brief, news alert, EOD report, kill switch). Post-earnings drift by guidance label and an early-warning overlay. Token cost and cache-hit rate, a versioned response cache, prompt-regression reports, a rate limiter | `python -m pytest week40_agents_ops` |
| [clinic_w1_copilot/](clinic_w1_copilot/) | Clinic W1 | Research-copilot evaluation report: retrieval by mode, faithfulness, numeric accuracy against the XBRL facts, refusal accuracy, an adversarial suite with and without sanitizing | `python copilot_eval.py` |
| [clinic_w2_end_to_end/](clinic_w2_end_to_end/) | Clinic W2 | News → point-in-time score → alert → pre-market brief → agent proposal → risk check → human decision → audit trail, and a timed kill-switch drill through the signed endpoint | `python end_to_end.py` |

## Setup

```bash
cd labs/part11
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week39_nlp_rag
python -m pytest                          # everything, a few seconds
```

Run all commands from `labs/part11` (its `conftest.py` makes the imports work). To go live, pass the request
dictionaries these labs build (`extraction_request`, `batch_requests`, `citation_request`) to the `anthropic` client.
Then store the responses and run the same checks on them.

## Things the labs make you notice

* **Grounding catches invented numbers, not injected ones.** The numeric grounding check flags half of the sloppy
  model's extractions (revenue read in billions, EPS invented where none was stated). Its revenue precision is 0.7.
  The obedient model reports the injected "revenue of 99999". That number *is* in the transcript, so grounding
  passes; only screening the source catches it.
* **The LLM beats the word list, but not everywhere.** The recorded LLM gets the headline's direction right 93% of
  the time, the word list 82% (it falls for "losses narrow" and "record sales" in a miss). After LLM-positive news,
  the 5-day abnormal return is +2.1%, against +1.4% after word-list-positive news. On bad news the word list does
  as well or better (−2.9% vs −1.9%). Measure before paying for the model.
* **Timestamps are part of the signal.** Scores that arrive three hours after the news are usable later. Using the
  publication time instead inflates the positive-news CAR from 2.1% to 2.4%: a look-ahead.
* **Look at what ranks first.**
  * BM25 and hybrid retrieval find every answer in the top 5 (dense: 94%), but recall@1 is only 0.06: the one-line
    cover page matches every "company + fiscal year" question best.
  * Dropping the cover-page chunks lifts recall@1 to 0.88 and MRR from 0.43 to 0.97.
  * On this corpus, BM25 and hybrid retrieval beat the dense stand-in.
* **Citations need unique document titles.** With one title per section, a citation can't be traced to its chunk
  (several chunks share a section). The lab puts the chunk id in the title.
* **Sanitize before answering.** With the injected filing in context, an obedient model answers "EPIC revenue
  doubled" without a citation, so faithfulness is 0. The adversarial pass rate is 0.67 raw and 1.0 sanitized.
* **The agent cannot trade.**
  * `place_order` and `set_risk_limit` can't even be registered, and a rogue model's calls are denied and audited.
  * Proposals go to a queue, then the risk pre-check (defined risk only), then a human.
  * The tool and cost budgets stop a looping model.
* **The kill switch is signed.** A signed request trips it. A replayed (stale), forged or unsigned one gets 401 or 422.
  All four n8n exports pass the validator. An export with a hard-coded key, an external URL or an unsigned
  kill-switch path fails it.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P11_SOLUTIONS=1 python -m pytest` (49 tests pass). On the blank starters, every failure is a
  `NotImplementedError`; tests that share a fixture report errors instead.
* **Before sharing with learners, remove `solutions/` and `tools/`.** The sync test then skips itself.
* Model IDs and token prices in the labs (`claude-opus-5`, `PRICES`) are illustrative, taken from the lesson plan.
  Check the Anthropic documentation for current values. The n8n exports are simplified: import them into your
  n8n version and re-export.
* Not covered offline: EDGAR downloads, pgvector, sentence-transformers, FinBERT and live n8n runs. The labs' data
  structures and checks carry over to them unchanged.
