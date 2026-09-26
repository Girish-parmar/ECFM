# Part 4 guided notebooks — Broker Connectivity & Trading Infrastructure

Guided notebooks for Part 4 ([lesson plan](../../docs/lessons/PART_04_BROKER_CONNECTIVITY.md)). The setup, data and plotting
code is written for you; learners fill in short **✍️ Your turn** cells, replacing each `...`. Each exercise ends with
`p.check(...)`, which prints ✅ or ❌. If an answer isn't right yet, the notebook continues with the reference value, so
later cells still run.

**Nothing here connects to a broker.** Account rows, bars, ticks, IB message codes and order events are synthetic and
shaped like what `ib_async` and `alpaca-py` return. You can work through every notebook without a paper account, an
API key or a running IB Gateway. These notebooks are the **exploratory** companion to the auto-graded labs in
[`labs/part04/`](../../labs/part04/): here you *see* why a rule exists (a reconnect storm, a fill that beats a cancel, a
join off by five hours); there you build the tested version with the real SDK objects. The labs also include the
`paper/` scripts for the clinics with real paper accounts.

| Notebook | Sessions | Exercises |
|---|---|---|
| `01_env_config_secrets.ipynb` | S1–S2 | IB Gateway/TWS ports; a paper/live port guard; a pre-commit secret scanner. Plus: `SecretStr` settings from the environment, `clientId` clashes (error 326) |
| `02_contracts_and_accounts.ipynb` | S3–S4 | Canonical symbols (`ESZ6`, `EUR.USD`, `BTC/USD`); IB account rows → `Decimal` summary; Alpaca account warnings (PDT, blocked, buying power). Plus: IB contract fields per asset class, the multi-currency overwrite bug, both accounts in one table |
| `03_historical_data.ipynb` | S5–S6 | IB download chunks walking back in time; a request pacer; IB and Alpaca bars → one UTC schema. Plus: what pacing costs, naive time zones, IEX vs consolidated volume |
| `04_live_bars_and_cache.ipynb` | S7–S8 | A tick → bar builder; the missing ranges of a read-through cache. Plus: why the builder needs a timer, cache hit rate, stale-feed detection |
| `05_connections.ipynb` | S9 | Back-off with full jitter; a circuit breaker; sorting IB message codes. Plus: 500 clients reconnecting with and without jitter, a breaker over a 10-minute outage |
| `06_order_mapping.ipynb` | S10 | Side-aware tick rounding; canonical order → IB fields; IB status → canonical state. Plus: the Alpaca mapping and its extended-hours rule, a Hypothesis property test |
| `07_order_lifecycle.ipynb` | S11–S12 | Apply a messy update stream to the order state machine; positions from executions; reconciliation after a crash. Plus: the naive last-status-wins bug, a bracket order with OCO children |
| `08_safety_multiasset.ipynb` | S13–S16 | Futures P&L; the third-Friday expiry; the pre-trade check chain; a kill-switch flatten plan. Plus: pip values, roll dates, a sticky kill switch across a restart, time-to-flat with `asyncio.gather`, broker contract tests |

`p4lib.py` holds the reference implementations the checks compare against, the synthetic data, a small simulated broker
and the chart style. `p.check` compares `Decimal`s, text, dates and objects **exactly** (also inside lists and dicts),
and floats and arrays with a tolerance.

## Setup

```bash
cd notebooks/part04
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

All data is synthetic and generated with fixed seeds, so every notebook runs offline.

## What the notebooks show

* If you filter IB account rows by tag but not by currency, the EUR row overwrites net liquidation: $1,520 instead of
  $1,003,127.44.
* Downloading 1,040 chunks takes 5.8 minutes under the 6-per-2-seconds pacing rule. Under the 60-per-10-minutes rule
  as well, it takes 2.8 hours.
* If you treat IB's naive New York times as UTC, 18 bars still join the Alpaca feed, and every one of them joins to the
  wrong bar. Alpaca's IEX feed carries about 2.8% of consolidated volume.
* Without a timer, a bar is published up to 28 seconds late after a quiet spell, and the last bar is never published.
  A read-through cache turns 200 research requests into 41 downloads.
* 500 clients reconnecting in lockstep take 199 seconds and 2,750 attempts to get back in. With full jitter it takes
  21 seconds and 1,504 attempts. A circuit breaker cuts calls to a dead broker from 120 to 12.
* Three quarters of the messages arriving through IB's error callback are harmless information.
* Half-up rounding raises a buy limit above the intended price in 5,025 of 10,000 cases. Side-aware rounding never
  does.
* With a naive event handler, a duplicated fill shows 140 shares on a 100-share order. It also misses a fill that beat
  a cancel, and lets a rejected order look live.
* Flattening 56 requests one at a time at 100 ms each misses the 5-second kill-switch target (5.6 s). Sending them
  concurrently takes 0.2 s.
* The contract tests catch an adapter that isn't idempotent: a retried order would buy twice.

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
