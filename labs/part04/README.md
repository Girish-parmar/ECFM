# Part 4 labs — Broker Connectivity & Trading Infrastructure

Auto-graded exercises for [Part 4](../../docs/lessons/PART_04_BROKER_CONNECTIVITY.md) (program weeks 13–16, milestone **M1**).
**Every test runs offline:** the IB (`ib_async`) and Alpaca (`alpaca-py`) order and contract objects are built but never sent,
and brokers are replaced by recorded-shape data, fake clocks and a simulated broker. The scripts in [`paper/`](paper/)
are the only code that talks to real (paper) accounts.

Fill in each `raise NotImplementedError("✍️ Your turn ...")` (or `pass  # ✍️ Your turn`) and run the tests until they are green.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week13_foundations/](week13_foundations/) | S1–S4 | IB ports and a paper/live port guard, a secret scanner for pre-commit, canonical symbols (`ESZ6`, `EUR.USD`, `BTC/USD`), `Instrument` → IB contract, IB account summary, Alpaca account warnings | `python -m pytest week13_foundations` |
| [week14_data/](week14_data/) | S5–S8 | IB download plan (walk back in chunks), IB pacing guard (15 s / 6 per 2 s / 60 per 10 min), IB and Alpaca bars → one canonical schema, live `BarBuilder` with timer flush, read-through `BarCache` | `python -m pytest week14_data` |
| [week15_orders/](week15_orders/) | S9–S12 | Back-off with jitter, circuit breaker, IB system-code handling, tick rounding, canonical order → `ib_async` / `alpaca-py` orders (with a Hypothesis "never changes meaning" test), status mapping, `OrderTracker` (duplicates, out-of-order, fill races), reconciliation report | `python -m pytest week15_orders` |
| [week16_safety/](week16_safety/) | S13–S16 | Futures and FX helpers (tick value, pip value, third Friday, roll date), pre-trade check chain, `SimBroker` passing the broker **contract tests**, sticky / idempotent / audited `KillSwitch` | `python -m pytest week16_safety` |
| [clinic_w3_event_replay/](clinic_w3_event_replay/) | Clinic W3 | Replay the same bracket order's IB and Alpaca event logs (with duplicates, a status before its fill, a late `new`) through your tracker and prove the final states match | `python replay.py bracket_orders.json events_ib.jsonl events_alpaca.jsonl` |
| [clinic_w4_kill_drill/](clinic_w4_kill_drill/) | Clinic W4 | Kill-switch drill over two slow simulated brokers (must be flat in < 5 s) and a heartbeat watchdog that trips the switch when the strategy hangs | `python drill.py` |
| [paper/](paper/) | Clinics W1–W2 | Real paper accounts: `connect_both.py` (both accounts side by side) and `download_bars.py` (5-min bars from both brokers → canonical Parquet), using your week 13–14 functions | see below |

Shared, complete types (`Instrument`, `OrderRequest`, `OrderState`, transitions, errors, canonical bar columns) are in
[`common.py`](common.py). The clinics import your week 15 / week 16 code, so finish those weeks first.

## Setup

```bash
cd labs/part04
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week13_foundations       # one lab
python -m pytest                          # everything
```

Run all commands from `labs/part04` (its `conftest.py` makes the imports work).

## Paper accounts (Clinics W1–W2)

Paper accounts only in Part 4. Put the keys in `labs/part04/.env` (git-ignored) or in environment variables:

```bash
QF_ENV=paper
QF_IB_PORT=4002            # Gateway paper (TWS paper: 7497); a live port is refused
QF_IB_CLIENT_ID=11         # unique per process
QF_ALPACA_KEY=...          # Alpaca PAPER key id
QF_ALPACA_SECRET=...
```

```bash
python paper/connect_both.py                              # add --skip-ib or --skip-alpaca
python paper/download_bars.py --symbols SPY AAPL --days 30 --out data/part04
```

Before committing, run your week 13 `find_secrets` over the files you changed.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P4_SOLUTIONS=1 python -m pytest` (79 tests pass).
* **Before sharing with learners, remove `solutions/` and `tools/`.** The sync test then skips itself.
* The `paper/` scripts were import-checked only; they were not run against live IB or Alpaca paper servers here.
  Run them once before the clinic, because both SDKs change often (the lesson plan's instructor notes say the same).
