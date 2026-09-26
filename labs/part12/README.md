# Part 12 labs — Complete Trading Platform: Capstone & Production

Auto-graded exercises for [Part 12](../../docs/lessons/PART_12_TRADING_PLATFORM_CAPSTONE.md) (program weeks 41–48,
milestones **M8**, **M9** and the capstone). The capstone is your own `quantforge` platform. These labs isolate each
operating discipline so you can practise and test it offline, on synthetic data with known answers.
`common.py` provides:

* a toy package with planted layer violations
* bars and universe snapshots
* a quote path with informed order flow
* intraday sessions
* a trade journal with a hidden edge
* a trading day with injected faults
* a 4-week track record in which one strategy stops working

Fill in each `raise NotImplementedError("✍️ Your turn ...")` and run the tests until they are green.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week41_integration/](week41_integration/) | S1–S4 | Architecture rules from the import graph (an `ast`-based import-linter), latency histograms and percentiles, ring buffers, streaming EMA, event-loop lag, the YAML strategy creator with validation and config hashes, composable screeners, universe turnover, point-in-time universe storage | `python -m pytest week41_integration` |
| [week42_execution/](week42_execution/) | S5–S8 | Tick-safe spread-aware limit prices, a chase simulator (fill rate, cost, markout, all-in cost including non-fills), TWAP / VWAP / POV and implementation shortfall, an option position manager (stop, take profit, expiry, ex-dividend, delta adjust, roll), R-multiples, expectancy tables, rule-violation checks | `python -m pytest week42_execution` |
| [week43_monitoring/](week43_monitoring/) | S9–S12 | Structured logs with correlation ids, Prometheus metrics, alert rules and SLOs, a hash-chained audit log, position and order reconciliation, four-eyes approvals, robust z-scores, fat-tail and CUSUM detectors, the controller (alert → pause → pause all and reconcile → kill; resume only with approval), daily reports, the live-vs-backtest band, attribution | `python -m pytest week43_monitoring` |
| [week44_production/](week44_production/) | S13–S16 | Checks for the committed [`deploy/`](week44_production/deploy/) compose file and CI workflow, a backtest regression fixture, event-sourced state with atomic snapshots and crash recovery behind a reconciliation gate, verified backups, secret scanning, threat ranking, broker-side precautionary limits | `python -m pytest week44_production` |
| [week45_golive/](week45_golive/) | S17–S24 | Rollout stages (promote, hold, demote), capital-ramp rules, the exchange calendar with early closes, a pre-market checklist that blocks trading, the incident lifecycle with SLAs, post-mortem timelines, the go-live review, weekly decisions, the graduation checks from the audit chain | `python -m pytest week45_golive` |
| [clinic_w2_execution_quality/](clinic_w2_execution_quality/) | Clinic W2 | Execution-quality report: aggressiveness sweep, parent-order TCA (single vs TWAP vs point-in-time VWAP vs POV), urgency policy, journal findings | `python execution_quality.py` |
| [clinic_w3_monitoring_drill/](clinic_w3_monitoring_drill/) | Clinic W3 | Replay a day with five injected faults: detect each one, respond with the right controller action, keep reconciliation clean with idempotent fills, and raise no alarm on clean days | `python drill.py` |
| [clinic_w5_track_record/](clinic_w5_track_record/) | Clinic W5 & capstone | Go-live review, a 4-week track record against backtest bands, weekly decisions, the capital ramp, and graduation from the audit chain | `python track_record.py` |

## Setup

```bash
cd labs/part12
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week41_integration
python -m pytest                          # everything, about 10 seconds
```

Run all commands from `labs/part12` (its `conftest.py` makes the imports work).

## Things the labs make you notice

* **Architecture rules must see everything.** The checker finds all three planted violations, including an import
  hidden inside a function that a quick grep of the file header misses.
* **Never block the event loop.** A 200 ms computation on the loop delays everything else by 200 ms. Moved to an
  executor, the delay is about 1 ms.
* **Passive orders look cheap until you count non-fills.**
  * Order flow in the quote path is informed.
  * A passive limit saves about 1.7 bps when it fills, but it fills only about 66% of the time. Including the cost of
    completing unfilled orders, it is roughly break-even.
  * A passive-first chase schedule is the best all-in (−0.6 bps). Crossing costs +1.9 bps.
* **Slice big orders.** Buying 60,000 shares at once costs about 167 bps. TWAP and VWAP cost about 45 bps, but their
  day-long exposure to price drift makes the result vary (±70 bps). POV at 10% finishes early in the heavy morning
  volume: 23 ± 5 bps.
* **The edge hides in a segment.** In the journal, breakout and pullback setups look alike overall (+0.2R). Split by
  regime, breakouts earn +0.61R in trends and lose 0.26R in chop.
* **Different detectors catch different faults.**
  * The robust z-score flags exactly the one latency spike, but on raw latencies it also raises a false alarm on a
    clean day: latencies are skewed, so detect on their log.
  * A slow 0.8σ drift is invisible to any z-score; CUSUM catches it within about 40 observations.
  * In the drill, all five injected faults are caught, and eight clean days raise no alarm.
* **History cannot be rewritten.** Changing one audit record breaks the hash chain at exactly that record.
* **Idempotency beats reconciliation.** A fill delivered twice would create a position break. Booking by fill id
  prevents it.
* **Deployment checks catch what reviewers miss.** The lesson plan's own `timescaledb:latest-pg16` placeholder is
  flagged as unpinned. The regression fixture catches both an off-by-one SMA and a one-bar look-ahead.
* **Recovery must pass a gate.** Recovery from a crash at any point rebuilds the exact state. A manual order placed in
  TWS blocks trading until it is reconciled.
* **Four weeks is short.** A strategy that bleeds 0.8% a day falls below its backtest band in week 2 and goes back to
  paper. One that decays by 0.4% a day can stay inside the band for the whole month. That is why the capstone grades
  behaviour and explanations, not returns.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P12_SOLUTIONS=1 python -m pytest` (65 tests pass). On the blank starters, every failure is a
  `NotImplementedError`; tests that share a fixture report errors instead.
* **Before sharing with learners, remove `solutions/` and `tools/`.** The sync test then skips itself.
* These labs don't replace the capstone operations: the VPS deployment, a Grafana/Streamlit dashboard, broker
  connections and the 4-week paper or live track record stay hands-on. The labs give each of those a tested core.
