# Part 1 starter notebooks — Financial Markets & Economic Foundations

Ready-to-run notebooks for the Part 1 labs ([lesson plan](../../docs/lessons/PART_01_MARKETS_ECONOMICS.md)).
In Part 1 learners **run** these notebooks and change only the cells marked **✏️ Change me**;
writing Python from scratch starts in Part 3. Excel remains the main lab tool this month, and
notebook 04 is the cross-check for the Excel models.

| Notebook | Sessions | What it does | Exports (in `outputs/`) |
|---|---|---|---|
| `00_environment_check.ipynb` | Week 1 setup | Checks packages, FRED access, API key, plotting; creates the research log | – |
| `01_fred_alfred.ipynb` | S2 | Macro series (CPI, core PCE, unemployment, industrial production, payrolls); **first release vs revised** payrolls from ALFRED vintages; point-in-time `as_of`; release-surprise table | `macro_monthly.csv` |
| `02_macro_dashboard.ipynb` | S3–S4, clinic W1 | Yield-curve spreads with recession shading; fed funds futures → implied FOMC probability; credit spread and dollar; rolling stock–bond correlation; rule-based growth × inflation regime | `macro_dashboard_summary.csv`, `macro_regime_history.csv` |
| `03_spreads.ipynb` | S8–S9, clinic W3 | Walking the order book (cost vs size); quoted, effective and realized spreads and price impact; intraday spread pattern; Roll estimator; multi-stock comparison | `spread_comparison.csv` |
| `04_derivatives_calculator.ipynb` | S10–S13, S15 | Futures fair value and basis; daily variation-margin ledger; Black–Scholes price and Greeks (matches the lesson's Excel sheet); implied volatility; payoff builder for 6 strategies with breakevens; OCC symbols; leveraged-ETF decay | – |

`p1lib.py` holds the shared functions (data access, point-in-time helpers, pricing, spreads, chart style).

## Setup

```bash
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
export FRED_API_KEY=...                    # free key from fred.stlouisfed.org (needed for ALFRED vintages in 01)
jupyter lab
```

Downloaded FRED data is cached in `data/` for 24 hours. Every notebook appends to `research_log.csv`.

## Data sources and caveats

- **FRED** (`fred_series`) returns today's **revised** values: fine for describing the present, wrong for historical backtests.
- **ALFRED** (`alfred_vintages`, `as_of`, `first_release`) returns every published vintage: use it for anything point-in-time.
- **Notebook 03 uses synthetic trades and quotes by default** (seeded, so everyone sees the same numbers). For the graded lab set `DATA_DIR` to the course microstructure files (`<TICKER>_quotes.csv` with `ts,bid,ask`; `<TICKER>_trades.csv` with `ts,price,size[,side]`).
- Futures margins, fee levels and contract details in the examples are illustrative; check the broker/exchange before relying on them.

## Offline mode and tests

`tests/make_fixtures.py` creates **synthetic** stand-ins for every FRED/ALFRED series the notebooks use
(realistic shapes, made-up numbers). Point `P1_FIXTURES` at such a folder to run without internet:

```bash
python tests/make_fixtures.py /tmp/p1_fixtures
P1_FIXTURES=/tmp/p1_fixtures jupyter lab       # charts will show synthetic data
python -m pytest -q tests                      # unit tests + executes all 5 notebooks offline
```

The live FRED/ALFRED download paths could not be exercised when these notebooks were built (no access to
fred.stlouisfed.org from the build environment); run `00_environment_check.ipynb` once with internet access
before the first class.
