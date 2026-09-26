# ECFM — Master in Financial Analysis and Algorithmic Trading (MFAAT)

A 12-month advanced program (approx. ₹10,00,000) for quantitative finance and algorithmic trading, built on **Interactive Brokers** and **Alpaca**.

| Document | Purpose |
|---|---|
| [docs/01_COURSE_ROADMAP.md](docs/01_COURSE_ROADMAP.md) | Full program plan: 12 Parts, 4 Terms, month-by-month roadmap, detailed modules, assessment |
| [docs/02_MASTER_PROMPTS.md](docs/02_MASTER_PROMPTS.md) | Reusable LLM prompts to regenerate, improve and expand the course (curriculum, modules, platform components, strategies, ML, AI/n8n) |
| [docs/03_PLATFORM_LLD_ROADMAP.md](docs/03_PLATFORM_LLD_ROADMAP.md) | Low-level design of the `quantforge` trading platform: architecture, package map for all 54 roadmap items, interfaces, patterns, sprints |

## Lab material

| Folder | Contents |
|---|---|
| [notebooks/part01/](notebooks/part01/) | Part 1 starter notebooks (environment check, FRED/ALFRED point-in-time, macro regime dashboard, order books & spreads, futures & options calculator), shared `p1lib.py`, offline fixtures and tests |
| [notebooks/part02/](notebooks/part02/) | Part 2 guided notebooks (8, one per session) with self-checking exercises, instructor solutions, a synthetic market for offline use, and tests |

## Lesson plans (12 Parts, 12 months)

| Lesson plan | Summary |
|---|---|
| [docs/lessons/PART_01_MARKETS_ECONOMICS.md](docs/lessons/PART_01_MARKETS_ECONOMICS.md) | Detailed lesson plan for Part 1: 16 sessions on the algo-trading landscape, macro releases & point-in-time data, Fed & fed funds futures, intermarket regimes, instruments/clearing/margin, order types & corporate actions, fundamentals, microstructure (spreads, Roll, impact, LULD), futures carry & margin, options & parity, F&O strategies, Excel models (returns, BSM & Greeks) |
| [docs/lessons/PART_02_QUANT_TOOLKIT.md](docs/lessons/PART_02_QUANT_TOOLKIT.md) | Detailed lesson plan for Part 2: 8 sessions on linear algebra & PCA, calculus & convex optimization, return distributions & stylized facts, inference/bootstrap/HAC regression, stationarity & mean reversion, EWMA/GARCH volatility, cointegration/HMM/Monte Carlo, performance & risk metrics (VaR/CVaR) |
| [docs/lessons/PART_03_PYTHON_ENGINEERING.md](docs/lessons/PART_03_PYTHON_ENGINEERING.md) | Detailed lesson plan for Part 3: 24 sessions on Python basics (Decimal money, time zones), advanced Python (generators, decorators, typing, asyncio, performance), OOP & domain modelling (FIFO positions with property tests), SOLID & design patterns, NumPy/pandas/Polars, PostgreSQL/Timescale, Parquet/DuckDB/Redis, testing, logging, config, Docker, CI, and platform milestone M0 |
| [docs/lessons/PART_04_BROKER_CONNECTIVITY.md](docs/lessons/PART_04_BROKER_CONNECTIVITY.md) | Detailed lesson plan for Part 4: 16 sessions on IB + Alpaca connectivity, data, orders, kill switch, assessment |
| [docs/lessons/PART_05_ANALYTICS_LIBRARY.md](docs/lessons/PART_05_ANALYTICS_LIBRARY.md) | Detailed lesson plan for Part 5: 16 sessions on indicators (vectorized + streaming), candlestick & chart patterns, edge testing, sentiment, news NLP, tail risk |
| [docs/lessons/PART_06_FUTURES_OPTIONS_ENGINEERING.md](docs/lessons/PART_06_FUTURES_OPTIONS_ENGINEERING.md) | Detailed lesson plan for Part 6: 8 sessions on continuous futures, option chains & strike selection, BSM/Black-76, IV, 1st & 2nd-order Greeks, trees/MC/AD, SVI surface, portfolio Greeks & delta hedging |
| [docs/lessons/PART_07_STRATEGY_LIBRARY.md](docs/lessons/PART_07_STRATEGY_LIBRARY.md) | Detailed lesson plan for Part 7: 8 sessions on the strategy framework, 7 linear strategy groups, option strategy builder, directional/range/event/volatility option structures, hedging |
| [docs/lessons/PART_08_BACKTESTING_RISK_PORTFOLIO.md](docs/lessons/PART_08_BACKTESTING_RISK_PORTFOLIO.md) | Detailed lesson plan for Part 8: 24 sessions on the event-driven backtester, costs, biases, significance (PSR/DSR), walk-forward, purged CV/CPCV, PBO, validation gate, risk engine, sizing (Kelly), portfolio optimization (RP, HRP, BL, CVaR) |
| [docs/lessons/PART_09_STATISTICAL_TRADING.md](docs/lessons/PART_09_STATISTICAL_TRADING.md) | Detailed lesson plan for Part 9: 8 sessions on OU processes, cointegration & FDR pair selection, pairs strategy, Kalman hedge ratios, PCA residual stat-arb, factor models, structural breaks, market-neutral pairs portfolio |
| [docs/lessons/PART_10_MACHINE_LEARNING.md](docs/lessons/PART_10_MACHINE_LEARNING.md) | Detailed lesson plan for Part 10: 24 sessions on financial ML (dollar bars, fractional differentiation, triple-barrier, meta-labeling, purged CV, MDA/SHAP, leakage audit, drift), deep learning (LSTM/TCN/Transformers, conformal) and RL (Gymnasium envs, PPO for trading, execution, hedging) |
| [docs/lessons/PART_11_AI_NLP_COWORK.md](docs/lessons/PART_11_AI_NLP_COWORK.md) | Detailed lesson plan for Part 11: 8 sessions on LLM structured extraction, batch LLM sentiment & event studies, point-in-time RAG over SEC filings with citations, research agents with read-only tools, n8n automation with signed webhooks, event & crash analysis, LLM evaluation/cost/safety |
| [docs/lessons/PART_12_TRADING_PLATFORM_CAPSTONE.md](docs/lessons/PART_12_TRADING_PLATFORM_CAPSTONE.md) | Detailed lesson plan for Part 12: 24 sessions on LLD integration, performance, config-driven strategy creator, screeners, spread-aware OMS & execution algos, trade journal, monitoring department (observability, hash-chained audit, anomaly detection), Docker/CI/CD, DR, security, go-live and the 4-week capstone track record |
