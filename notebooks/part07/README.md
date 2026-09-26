# Part 7 guided notebooks — Strategy Library

Guided notebooks for Part 7 ([lesson plan](../../docs/lessons/PART_07_STRATEGY_LIBRARY.md)), one per session.
The setup, data and plotting code is written for you; learners fill in short **✍️ Your turn** cells, replacing each
`...`. Each exercise ends with `p.check(...)`, which prints ✅ or ❌. If an answer isn't right yet, the notebook continues
with the reference value, so later cells still run.

**No strategy here is presented as profitable.** Each one is a hypothesis with a first-look evaluation (next-bar
fills, costs, nothing more); the honest backtest, validation and sizing come in Part 8.

All data is synthetic:
- a market built from half-year blocks whose regime is **known** (trending or range-bound, calm or volatile), so every
  strategy can be scored inside each regime;
- a universe with persistent drifts;
- a cointegrated pair (and one that breaks);
- an implied-vol history, earnings events, a market with stochastic volatility, and a crash path.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part07/`](../../labs/part07/). Here
you *see* why a rule exists: a same-bar fill that turns a worthless signal into a Sharpe of 8.5, a 71% win rate that
still loses money, a hedge that looks useless until the crash. In the labs you build the tested version.

| Notebook | Session | Exercises |
|---|---|---|
| `01_framework_quick_eval.ipynb` | S1 | A registered strategy that only emits intents; the next-bar first-look P&L; the next-bar execution (truncation) test. Plus: parameter validation, a same-bar fill, a strategy that peeks |
| `02_momentum.ipynb` | S2 | Volatility-scaled TSMOM; Donchian exits; 12-1 cross-sectional momentum. Plus: the regime table, first-look equity curves |
| `03_mean_reversion.ipynb` | S3 | IBS with the flat-bar guard; a z-score time stop; RSI(2) with a trend filter. Plus: what filters and time stops do in trends |
| `04_vol_math_stat.ipynb` | S4 | A volatility-target overlay; the Hurst exponent; a pairs trade whose hedge ratio never sees the future. Plus: Hurst by regime, a pair that breaks |
| `05_option_builder.ipynb` | S5 | Valuing a multi-leg strategy; breakevens; the defined-risk rule. Plus: iron condor P&L over time, Greeks, why POP is not an edge |
| `06_directional_options.ipynb` | S6 | Strikes by delta; IV rank vs IV percentile; combo mid vs natural price. Plus: three ways to be bullish, the structure selector |
| `07_range_event_vol.ipynb` | S7 | The implied move and the 0.8·S·σ·√T rule; an event study; a tradable VRP signal. Plus: a truncation test on research vs tradable VRP, a premium seller's worst month |
| `08_hedging.ipynb` | S8 | Futures hedge sizing; the zero-cost collar strike; a portfolio-level hedge report. Plus: no hedge vs puts vs collar vs futures through a crash and in calm years |

`p7lib.py` holds the reference implementations the checks compare against (the same definitions as the labs), the
synthetic data and the chart style.

## Setup

```bash
cd notebooks/part07
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

All data is generated with fixed seeds, so every notebook runs offline.

## What the notebooks show

* **Same-bar fills.** "Today closed up, so be long" has a Sharpe of −0.6 when filled at the next open. Filled at the
  open of the bar that produced it, it shows +8.5. The next-bar execution test catches a strategy that peeks at
  tomorrow.
* **Momentum by regime.** TSMOM earns in trends (Sharpe 1.8–1.9) and bleeds in ranges. In a universe with persistent
  drifts, 12-1 momentum's winners beat the equal-weight average with a Sharpe of 1.15.
* **Mean reversion.** It earns in ranges and loses in trends. A trend filter cuts RSI(2)'s worst 20-day loss from −30%
  to −11%. A time stop cuts the z-score trade's from −34% to −22%.
* **Volatility targeting.** On buy and hold in this market it cuts the maximum drawdown from −92% to −58%.
* **Hurst exponent.** Range-bound blocks average H = 0.39, trending blocks 0.46 and a random walk 0.49, with wide
  scatter. A steady drift doesn't raise H.
* **Pairs.** A pairs trade with a rolling, look-ahead-free hedge ratio has a Sharpe of 1.0. After the relationship
  breaks, it gives back three times what it made.
* **Iron condor.** A 16-delta condor wins 71% of the time and still averages −$51, because its maximum loss is nearly
  4× its maximum profit.
* **Legging costs.** Crossing all four legs of the condor gives up about 8% of its credit.
* **Volatility risk premium.** The research version of the VRP fails the truncation test, and the tradable version
  passes. Selling monthly straddles wins 59% of months with a +0.6% average, and the worst month loses 10.6%.
* **Collars.** On a skewed smile, the call that pays for a 10% OTM put is at +4.6% (a flat smile would allow +13.9%).
* **Hedging through a crash.** Every hedge cuts the drawdown (−40% unhedged, −18% to −28% hedged). In the calm years
  the puts and futures cost 2–4% of return.

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
