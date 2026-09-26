# Part 3 guided notebooks — Python Engineering for Trading

Guided notebooks for Part 3 ([lesson plan](../../docs/lessons/PART_03_PYTHON_ENGINEERING.md)). The setup, data and plotting
code is written for you; learners fill in short **✍️ Your turn** cells, replacing each `...`. Each exercise ends with
`p.check(...)`, which prints ✅ or ❌. If an answer isn't right yet, the notebook continues with the reference value, so
later cells still run.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part03/`](../../labs/part03/): here
you *see* why a rule exists (a float off the tick grid, a queue that ages your quotes, a join that leaks the future);
there you build the tested version.

| Notebook | Sessions | Exercises |
|---|---|---|
| `01_money_and_time.ipynb` | S1, S3 | `round_to_tick` in `Decimal`; IB fixed commission; the New York open in UTC; a `deque` rolling mean. Plus: float drift, the DST step in the open's UTC time, list vs set lookups |
| `02_functions_and_files.ipynb` | S2, S4, clinic W1 | Returns with a comprehension; drawdown loop; `match` routing of broker messages; a trade-log parser that collects an error per bad line. Plus: Python vs NumPy speed, the mutable-default trap, a JSON P&L report |
| `03_generators_decorators.ipynb` | S5 | Ticks → bars with a generator; a `count_calls` decorator with `functools.wraps`; a `timer` context manager. Plus: generator vs list memory, a retry decorator with backoff |
| `04_types_and_asyncio.ipynb` | S6, S7, clinic W2 | A Pydantic `OrderRequest` with a cross-field rule; concurrent quotes with `asyncio.gather`. Plus: frozen dataclasses and enums, back-pressure with a bounded vs unbounded queue |
| `05_performance.ipynb` | S8, S17 | Vectorized drawdown; a `numba`-compiled EMA; compact dtypes. Plus: loop vs pandas vs numba timings, memory before/after |
| `06_oop_and_patterns.ipynb` | S9–S16 | `Money.__add__` with a currency check; an `avg_price` property; the order state machine. Plus: an Observer event bus and swappable fill models (Strategy) |
| `07_pandas_polars.ipynb` | S18 | OHLCV resampling; `merge_asof` without look-ahead; VWAP with Polars. Plus: how `direction="nearest"` leaks the future, effective spreads, pandas vs Polars speed |
| `08_sql_duckdb_testing.ipynb` | S19, S20, S22 | DuckDB SQL over hive-partitioned Parquet; a window-function moving average; a bad-bar detector; a Hypothesis property test that catches the float rounding bug |

`p3lib.py` holds the reference implementations the checks compare against, the synthetic data and the chart style.
`p.check` compares `Decimal`s, text, dates and objects **exactly**, and floats and arrays with a tolerance. A float
answer to a money question does not pass.

## Setup

```bash
cd notebooks/part03
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

All data is synthetic and generated with fixed seeds, so every notebook runs offline.

## What the notebooks show

* A million one-cent fees summed as floats give 10000.000000171856. Rounding commissions with float `round()`
  disagrees with exact half-up rounding on 2,236 of 10,000 trades.
* A float tick-rounder returns 0.35000000000000003, which is off the tick grid. The Hypothesis property test finds
  this bug on its own.
* The US open moves from 14:30 UTC to 13:30 UTC in March. Naive timestamps get this wrong twice a year.
* Checking whether an item is in a set is 10,000× faster than scanning a list of 100,000 symbols.
* With a slow consumer, an unbounded queue lets quotes age to a p99 of about 330 ms before they are processed. A
  bounded queue keeps them near 27 ms.
* numba runs the EMA loop about 110× faster than plain Python. Choosing the right dtypes cuts a tick table to a third
  of its memory.
* `merge_asof(direction="nearest")` uses a quote from the future for about half of the trades.

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
