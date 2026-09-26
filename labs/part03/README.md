# Part 3 labs — Python Engineering for Trading

Auto-graded exercises for [Part 3](../../docs/lessons/PART_03_PYTHON_ENGINEERING.md) (program weeks 7–12).
Every lab is a Python file with docstrings describing each task and a `pytest` file that grades it.
Fill in each `raise NotImplementedError("✍️ Your turn ...")` (or `pass  # ✍️ Your turn`) and run the tests until they are green.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week07_basics/](week07_basics/) | S1–S4 | Decimal tick rounding, IB fixed commission, returns & drawdown, `match` routing, UTC/New York times, best bid/ask, VWAP | `python -m pytest week07_basics` |
| [week08_advanced/](week08_advanced/) | S5–S8 | Tick generator → bars, decorators (`count_calls`, async `retry`), token-bucket rate limiter, Pydantic `OrderRequest`, async pipeline, EMA | `python -m pytest week08_advanced` |
| [week09_domain/](week09_domain/) | S9–S12 | `Money`, `Instrument`, `Future`, `Bar`, `Fill`, FIFO `Position` with a Hypothesis property test (realized + unrealized = cash P&L) | `python -m pytest week09_domain` |
| [week10_patterns/](week10_patterns/) | S13–S16 | Event bus (Observer), order state machine, registry/factory, risk rule chain, composable entry conditions (Specification), fill models (Strategy) | `python -m pytest week10_patterns` |
| [week11_data/](week11_data/) | S17–S20 | Rolling z-score, resampling, `merge_asof`, bar validation & gap finder, hive Parquet, DuckDB summary, Polars lazy SMA; [`schema.sql`](week11_data/schema.sql) for PostgreSQL + TimescaleDB | `python -m pytest week11_data` |
| [clinic_w1_trade_log/](clinic_w1_trade_log/) | Clinic W1 | CLI: trade-log CSV → validated fills (errors per line) → FIFO P&L per symbol and New York date → JSON report | `python trade_log.py sample_fills.csv report.json` |
| [clinic_w2_async_sim/](clinic_w2_async_sim/) | Clinic W2 | Multi-feed `asyncio` quote simulator with a bounded queue: see back-pressure and latency p50/p99 with a slow consumer | `python quote_sim.py` |
| [m0_template/](m0_template/) | S21–S24, W12 | Skeleton for milestone **M0**: `quantforge` package (`core/` errors, config, JSON logging, clocks), `pyproject.toml` (ruff, mypy strict, pytest), pre-commit, CI workflow, ADR and UML folders | see [its README](m0_template/README.md) |

## Setup

```bash
cd labs/part03
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week07_basics            # one lab
python -m pytest --ignore=m0_template     # all labs
```

Run all commands from `labs/part03` (the `conftest.py` there makes the test imports work).
Nothing needs a network connection or a broker account; the data is synthetic or in the folder.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: every block between
  `# >>> SOLUTION` and `# <<< SOLUTION` is replaced by `raise NotImplementedError` (or `pass` for `# >>> SOLUTION (pass)`).
  After editing a solution run `python tools/make_starters.py`; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P3_SOLUTIONS=1 python -m pytest --ignore=m0_template` (56 tests pass).
* **Before sharing with learners, remove `solutions/` and `tools/`.** The sync test then skips itself.
* `week11_data/schema.sql` is reference DDL for the S18–S19 database sessions; it is not run by the tests
  (it needs PostgreSQL with the TimescaleDB extension, e.g. the `timescale/timescaledb` Docker image).
