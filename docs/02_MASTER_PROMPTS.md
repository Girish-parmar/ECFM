# Prompt Pack: Generate & Extend the MFAAT Course

Reusable prompts for regenerating, improving or expanding the **Master in Financial Analysis and Algorithmic Trading** program with any capable LLM.
Replace `{{placeholders}}` before use. Run the prompts in order: **Prompt 1** creates the structure, and **Prompts 2–6** go deep on each piece.

---

## Prompt 1 — Master Curriculum & Roadmap Generator

```text
ROLE
You are a panel of three experts working together:
(1) a quant portfolio manager with 15+ years in systematic trading,
(2) a senior Python/software architect who has built production trading systems,
(3) an instructional designer for premium executive technical programs.

TASK
Design a complete, premium, 12-month advanced program titled
"{{COURSE_NAME | Master in Financial Analysis and Algorithmic Trading}}"
priced at approximately {{PRICE | ₹10,00,000}}. The output must justify that price
through depth, rigor, hands-on engineering and a production-grade capstone.

LEARNER PROFILE
- Understands advanced concepts; comfortable with math and some programming.
- Goal: become an independent quant developer/trader able to research, build,
  validate, deploy and monitor systematic strategies end to end.

CONSTRAINTS
- Brokers: Interactive Brokers (TWS / IB Gateway via ib_async or ibapi) and Alpaca (alpaca-py).
  All code must be broker-agnostic through a common adapter interface.
- Markets: US equities/ETFs, indices, equity & index options, CME futures, Forex, crypto.
- Ignore SEBI / Indian regulation. Include only broker-enforced account rules that code must handle.
- Tooling: Python 3.12+, JupyterLab, Git, Docker, PostgreSQL/TimescaleDB, DuckDB, Parquet, Redis.
- Pedagogy: "build-as-you-learn". Every module contributes code to ONE growing trading
  platform built with Low-Level Design principles (SOLID, design patterns).
- Research rigor before ML: look-ahead bias, overfitting, costs, walk-forward, CPCV,
  deflated Sharpe and probability of backtest overfitting must be taught before ML/DL/RL.
- Risk is a mandatory gate: no order reaches a broker without pre-trade risk checks and a kill switch.
- Include AI-augmented workflows: LLMs, RAG over filings/news, AI agents ("cowork"),
  and n8n automation, with human-in-the-loop guardrails.

SEED TOPICS (must all be covered; fix naming, merge duplicates, fill gaps, reorder logically)
{{PASTE THE RAW TOPIC LISTS HERE: Stock Market Basics, Advance with Python (Jupyter),
Trade Creation & Analysis Advanced Methods, Complete Trading Platform roadmap}}

OUTPUT FORMAT (Markdown)
1. Program summary table (fee, duration, weekly effort, prerequisites, brokers, markets, tools).
2. Design philosophy: 5–7 principles that differentiate this program.
3. Architecture: Terms → Parts → Modules, with a Mermaid diagram.
4. Month-by-month roadmap table: theme, parts, platform milestone, graded deliverable.
5. Detailed curriculum: one table per Part with columns
   [# | Module | Key Topics | Labs/Notebooks]. Map every seed topic to a module
   and note which seed items it covers.
6. Strategy library: grouped (momentum, range-bound, mean reversion, either-way,
   volatility, mathematical, statistical, plus the same for options, including hedging),
   with example strategies per group and a standard strategy-spec template.
7. ML / DL / RL section organized as a 4-step matrix: Theory → Create Strategy →
   Advanced Analysis → Optimization.
8. Platform capstone: a summary of layers mapped to the seed roadmap item numbers.
9. Assessment model with weights and graduation criteria.
10. Value stack explaining what the fee includes.
11. Toolchain reference table.
12. List of improvements you made over the seed list, with one-line justification each.

QUALITY BAR
- Be specific: name concrete algorithms, libraries, metrics and tests, not generic phrases.
- Correct all spelling and terminology (e.g. "Black Swan", "Sentiment", "Hedging", "Anomaly detection").
- No filler, no motivational text. Tables over prose wherever possible.
```

---

## Prompt 2 — Module Deep-Dive (Lesson Plan + Notebook Outline)

```text
Using the MFAAT curriculum as context, create the full teaching package for:
Part {{PART_NO}} — Module {{MODULE_NO}}: "{{MODULE_NAME}}".

Deliver:
1. Learning objectives (5–8, measurable, Bloom's verbs).
2. Prerequisites (link to earlier module numbers).
3. Session plan: {{N_SESSIONS | 4}} sessions × 90 min, each with theory, live-coding and discussion blocks.
4. Theory notes with key formulas in LaTeX and intuitive explanations.
5. Jupyter notebook outline: cell-by-cell (markdown/code) for a lab using
   {{BROKER | IB and Alpaca paper accounts}} or cached Parquet data.
6. Production code to promote into the platform: file path in `quantforge/`,
   class/function signatures with type hints and docstrings, and pytest test cases.
7. Common mistakes & pitfalls (at least 5) and how to detect them in code.
8. Exercises: 3 basic, 3 intermediate, 2 advanced, plus 1 open research question.
9. Assessment rubric (100 points).
10. Further reading: books and papers (author, year).
```

---

## Prompt 3 — Platform Component (LLD) Generator

```text
You are a senior trading-systems architect. Design and implement the
"{{COMPONENT | e.g. Order Management System}}" component of the `quantforge` platform.

Context:
- Python 3.12, asyncio, pydantic v2, pytest, ruff, mypy --strict.
- Brokers: IB (ib_async) and Alpaca (alpaca-py) behind `BrokerAdapter`; plus `SimAdapter`.
- Event bus (Observer), immutable domain models, Repository for storage.
- Must work identically in backtest, paper and live modes.

Deliver:
1. Responsibilities and explicit non-responsibilities.
2. UML class diagram and one sequence diagram (Mermaid).
3. Design patterns used and why (SOLID mapping).
4. Public interface: full signatures, docstrings, error types.
5. Reference implementation (production quality, typed, logged, no placeholders).
6. Failure modes: disconnects, partial fills, duplicate events, clock skew,
   rate limits, stale data. Show how each is handled.
7. Tests: unit, contract tests per broker adapter (using fakes), and one property-based test.
8. Performance notes: complexity, memory, latency budget.
9. Configuration example (YAML).
```

---

## Prompt 4 — Strategy Specification & Research Generator

```text
Create a complete research specification for the strategy "{{STRATEGY_NAME}}"
in group "{{GROUP | Mean Reversion}}" for {{UNIVERSE | S&P 500 constituents / ES futures / EURUSD / SPX options}}.

Sections:
1. Hypothesis and economic rationale (why the edge should exist, and who is on the other side).
2. Market regime where it works and where it fails.
3. Data requirements (frequency, fields, history length, survivorship-free?).
4. Signal definition with exact formulas.
5. Entry, exit, stop-loss, target, adjustment and time-exit rules.
6. Position sizing and portfolio constraints.
7. Cost model: IB/Alpaca commissions, spread, slippage, borrow.
8. Backtest plan: in-sample/out-of-sample split, walk-forward windows, CPCV, benchmark.
9. Validation: deflated Sharpe, PBO, parameter-stability heatmap, Monte Carlo.
10. Risk limits and kill-switch triggers specific to this strategy.
11. Implementation using `quantforge` lib functions (code).
12. Go-live checklist: paper → shadow → small-size live.
```

---

## Prompt 5 — ML / DL / RL Strategy Pipeline Generator

```text
Build an end-to-end {{MODEL_TYPE | LightGBM / LSTM / PPO agent}} trading pipeline for {{UNIVERSE}}.

Include:
1. Feature engineering (technical, microstructure, fractionally-differentiated prices,
   sentiment, macro), with a leakage audit.
2. Labeling: triple-barrier + meta-labeling (ML/DL) OR environment design with
   costs, slippage and position limits (RL, Gymnasium API).
3. Validation: purged k-fold with embargo / CPCV; walk-forward retraining schedule.
4. Model training code (scikit-learn / LightGBM / PyTorch / stable-baselines3).
5. Explainability (SHAP) and uncertainty (conformal prediction or MC dropout).
6. Converting predictions into bets: sizing from probabilities.
7. Backtest with the platform's event-driven engine and a comparison to a simple baseline.
8. Hyperparameter optimization with Optuna under the same CV scheme.
9. Deployment: model registry (MLflow), ONNX export, drift monitoring and retrain triggers.
```

---

## Prompt 6 — AI Cowork / RAG / n8n Workflow Generator

```text
Design an AI-assisted trading workflow for: "{{USE_CASE | pre-market brief with news sentiment and watchlist}}".

Include:
1. Data sources (Alpaca/Benzinga news, IB news, SEC EDGAR, economic calendar).
2. RAG design: chunking, embedding model, vector store (pgvector), retrieval + re-ranking, citations.
3. LLM prompt templates (system + user) with structured JSON output schema.
4. Agent tools: get_quotes, get_chain, run_backtest, get_positions (read-only by default).
5. n8n workflow: nodes, triggers (cron / webhook), branching, error handling, notifications (Telegram/Slack/email).
6. Guardrails: an AI may propose trades, but only the platform RiskEngine may approve them;
   log every AI suggestion to the audit trail.
7. Evaluation: how to measure usefulness and accuracy of AI outputs over time.
```

---

## Tips for Best Results

1. Run **Prompt 1** once, save the output as the canonical syllabus, and paste it as context for Prompts 2–6.
2. Generate modules in dependency order (Part 1 → Part 12) so later modules can reference earlier code.
3. After each generation, ask: *"Critique this as a skeptical quant PM. What is missing, wrong or unrealistic? Then fix it."*
4. Keep all generated code in the `quantforge/` repository and require tests before merging.
