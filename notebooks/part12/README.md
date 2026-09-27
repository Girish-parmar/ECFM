# Part 12 guided notebooks — The complete trading platform, capstone and production

Guided notebooks for Part 12 ([lesson plan](../../docs/lessons/PART_12_TRADING_PLATFORM_CAPSTONE.md)), ten notebooks,
two sessions each; the last one also covers the track record, the weekly reviews and graduation (weeks 6–8).
The setup, data and plotting code is written for you; learners fill in short **✍️ Your turn** cells, replacing each
`...`. Each exercise ends with `p.check(...)`, which prints ✅ or ❌. If an answer isn't right yet, the notebook
continues with the reference value, so later cells still run.

**Everything runs offline: no broker, Docker or network is needed.** The data is synthetic with known answers:
- a small `quantforge` package with three planted architecture violations;
- daily bars and point-in-time universe snapshots;
- a quote path with informed order flow, and an intraday session with U-shaped volume;
- a trade journal with a hidden edge (breakouts work only in trends);
- a trading day with five injected faults (`p.FAULTS` is the answer key);
- a 4-week live record in which one strategy stops working.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part12/`](../../labs/part12/), with
the same definitions. `deploy/` holds the paper stack's `docker-compose.yml` and the CI workflow that notebook 07 checks.

| Notebook | Sessions | Exercises |
|---|---|---|
| `01_architecture_latency.ipynb` | S1–S2 | Imports from the syntax tree; histogram quantiles; a streaming EMA. Plus: layer and broker-access violations, event-loop lag on and off the loop |
| `02_strategy_creator_screeners.ipynb` | S3–S4 | Building conditions from YAML; validating a config; a screen. Plus: a config hash, universe turnover by noise, a point-in-time universe store |
| `03_execution_tca.ipynb` | S5–S6 | Tick-safe limit prices with `Decimal`; implementation shortfall. Plus: passive vs chase vs cross with markouts, single order vs TWAP, VWAP and POV |
| `04_options_journal.ipynb` | S7–S8 | Option management rules in priority order; R-multiples. Plus: the edge by setup and regime, rule violations |
| `05_observability_audit.ipynb` | S9–S10 | SLOs and error budgets; verifying a hash-chained audit log; position reconciliation. Plus: correlation ids, Prometheus metrics, alerts, four-eyes approvals |
| `06_anomaly_controller_reports.ipynb` | S11–S12 | A streaming robust z-score; idempotent fill booking; the backtest band. Plus: log vs raw latency, CUSUM on a drift, the fault drill with the controller, the daily report, attribution |
| `07_deploy_ci_regression.ipynb` | S13–S14 | CI gates; a backtest regression check. Plus: compose rules on the committed, teaching and "quick fix" files, a refactor and four bugs |
| `08_recovery_security.ipynb` | S15–S16 | Idempotent state rebuild; a secret scanner; broker-side precautionary limits. Plus: crash recovery at ten points, a manual order that blocks trading, verified backups, a threat model |
| `09_rollout_operations.ipynb` | S17–S18 | The capital ramp; the trading calendar; the pre-market checklist. Plus: promotion and demotion between stages, a 16-week ramp |
| `10_incidents_track_record.ipynb` | S19–S24 | The incident lifecycle; weekly decisions; orders that bypassed risk. Plus: SLA breaches, a post-mortem timeline, the go-live review, the capital path, graduation |

`p12lib.py` holds the reference implementations the checks compare against (the definitions of the labs) and the
synthetic data.

## Setup

```bash
cd notebooks/part12
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

## What the notebooks show

* **Grep misses what the syntax tree finds.** All three planted violations are found from the import graph, including
  `core/util.py` importing the execution layer inside a function, which a grep of top-level import lines misses.
* **Never block the event loop.** A 200 ms computation on the loop delays a 10 ms ticker by about 200 ms; in an
  executor, by about 1 ms.
* **Passive fills are adversely selected.** Passive orders fill about two times in three and earn about 1.7 bps when
  they do, but about half of that is gone 30 seconds later. All-in, chasing from passive to cross is the cheapest
  schedule (−0.6 bps) and crossing at once the most expensive (+1.9 bps).
* **Slice big orders.** Buying 60,000 shares at once costs about 167 bps; TWAP 47, VWAP 41, POV about 23.
* **Edges hide in segments.** By setup the two look alike (+0.25R and +0.21R). By regime, breakouts earn +0.61R in
  trends and lose −0.26R in chop.
* **Monitoring.** One 400 ms spike spends more than the whole 99.9% error budget.
  * Editing an audit record breaks its hash; re-hashing it breaks the next link.
  * On clean days, raw latency z-scores reach 5.6 (the alarm is at 6), log-scale ones only 3.7.
  * CUSUM catches a 0.8σ drift 7 observations in; a z-score never fires.
* **The drill.** All five injected faults are detected and handled, the audit chain stays intact, and a clean day
  raises nothing. Booking the re-sent fill twice leaves SPY at −200 against the broker's −300.
* **Regression tests catch "improvements".** A one-bar look-ahead in the SMA raises the fixture's total return from
  1.187 to 1.280 with the same 47 trades; the regression check flags it. An SMA refactor passes.
* **Recovery.** The state is correct after a crash at any of ten points. A manual order placed in TWS blocks trading
  until the break is explained, and a backup with one flipped bit is refused.
* **The track record.** Share of live days not below the backtest band: momentum 0.95, pairs 1.00, options 0.25.
  Options is demoted at the week-2 review, still at its 10% starting size. A sev1 incident reconciled after 73 minutes
  breaches its 60-minute target.

## Instructor material

- `solutions/`: the same notebooks with the answers filled in. **Remove this folder (or keep it on a private branch)
  before sharing the repository with learners.**
- `tools/build_notebooks.py`: the single source for starter and solution notebooks (it also contains the answers).
  Edit content there and rebuild with `python tools/build_notebooks.py`.

## Tests

```bash
python -m pytest -q tests     # 30 tests: solutions pass every check (strict mode), starters run with blanks,
                              # starter and solution notebooks differ only in the exercise cells
```
