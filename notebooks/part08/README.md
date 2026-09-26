# Part 8 guided notebooks — Backtesting, Optimization, Risk & Portfolio

Guided notebooks for Part 8 ([lesson plan](../../docs/lessons/PART_08_BACKTESTING_RISK_PORTFOLIO.md)), twelve notebooks
for the 24 sessions, two sessions each. The setup, data and plotting code is written for you; learners fill in short
**✍️ Your turn** cells, replacing each `...`. Each exercise ends with `p.check(...)`, which prints ✅ or ❌. If an
answer isn't right yet, the notebook continues with the reference value, so later cells still run.

All data is synthetic with a known truth, so every statistic can be checked against reality and every discovery
against luck:
- the known-regime market from Part 7;
- five strategy return streams with **known** Sharpe ratios;
- hundreds of pure-noise strategies;
- a stock universe with delistings;
- six assets with crisis episodes.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part08/`](../../labs/part08/). The
labs use DuckDB, scikit-learn and cvxpy; here the same definitions use only numpy, pandas and scipy. The helper library
uses sqlite for the research log, and its Ledoit–Wolf and minimum-CVaR code were checked against scikit-learn and cvxpy.

| Notebook | Sessions | Exercises |
|---|---|---|
| `01_research_log_and_engine.ipynb` | S1–S2 | Configuration hash; the ledger with multipliers; deterministic event order. Plus: logging every trial, the event-driven engine |
| `02_fills_costs_parity.ipynb` | S3–S4 | Limit fills that need trade-through; stop fills at the gap; square-root impact. Plus: adverse selection of limit fills, parity and cost sensitivity |
| `03_biases_and_tearsheet.ipynb` | S5–S6 | A point-in-time universe; max drawdown and time under water; Sortino and Calmar. Plus: what survivorship bias adds, a full tear sheet, a monthly heatmap |
| `04_significance_and_capacity.ipynb` | S7–S8 | The Probabilistic Sharpe Ratio; minimum track record length; capacity. Plus: Sharpe error bars and bootstrap intervals, net Sharpe vs capital |
| `05_search_and_stability.ipynb` | S9–S10 | Grid search; plateau scores; the Pareto front of Sharpe vs turnover. Plus: peak vs plateau out of sample, grid vs random search |
| `06_walk_forward_and_cv.ipynb` | S11–S12 | Walk-forward windows; purged, embargoed k-fold; the number of CPCV paths. Plus: walk-forward efficiency, a date-only model with R² 0.89 on noise under shuffled k-fold |
| `07_deflated_sharpe_and_pbo.ipynb` | S13–S14 | The expected maximum Sharpe of N trials; the Deflated Sharpe Ratio; the CSCV rank logit behind PBO. Plus: the best of 200 noise strategies, PBO for noise vs skill |
| `08_robustness_and_gate.ipynb` | S15–S16 | Sharpe without the best days; parameter perturbation; the validation gate. Plus: the full robustness scorecard, a good strategy that fails the gate |
| `09_risk_engine_and_var.ipynb` | S17–S18 | A concentration rule; the risk-engine chain; historical VaR and CVaR. Plus: component VaR, a VaR backtest, calm vs crisis correlations |
| `10_position_sizing_and_kelly.ipynb` | S19–S20 | Fixed-fractional size; risk of ruin; the Kelly fraction. Plus: ruin vs bet size, Kelly with an estimated edge |
| `11_combining_and_mean_variance.ipynb` | S21–S22 | Inverse-vol weights; minimum variance; Ledoit–Wolf shrinkage. Plus: diversification ratio, unstable max-Sharpe weights, a walk-forward comparison |
| `12_risk_based_allocation.ipynb` | S23–S24 | Risk contributions; the HRP bisection step; Black–Litterman with a view. Plus: risk parity, minimum CVaR, seven allocators walk-forward through three crises |

`p8lib.py` holds the reference implementations the checks compare against (the same definitions as the labs), the
synthetic data and the chart style.

## Setup

```bash
cd notebooks/part08
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

All data is generated with fixed seeds, so every notebook runs offline.

## What the notebooks show

* **Limit-order adverse selection.** Filling limit orders on a touch shows +16 bp per fill. Requiring the price to
  trade 3 ticks through shows −2 bp: the fills you lose are the best ones.
* **Parity.** The event-driven engine and the vectorized first look agree (correlation 1.0, differences of a few
  basis points from whole shares and compounding).
* **Survivorship bias.** Backtesting only today's survivors adds 2.7% a year of CAGR over a point-in-time universe.
* **Estimation error.** Ten years of data misses the true Sharpe by up to 0.47, and the 95% interval is ±0.6. The
  minimum track record needed at 95% ranges from 3.3 years to over 900 years across the five strategies, depending on
  how each sample happened to come out.
* **Parameter search.** On the in-sample half, the best TSMOM parameters lie on a ridge (lookback 160). Peak and plateau
  agree and hold out of sample (Sharpe 1.0), and in-sample and out-of-sample Sharpe correlate 0.71 across 48 trials.
  Random search with the same budget finds a better in-sample point than the grid.
* **Leakage in cross-validation.** A 1-nearest-neighbour model whose only feature is the date scores R² = +0.89 on pure
  noise under shuffled k-fold. Purging and an embargo bring it to −0.80.
* **Multiple testing.** The best of 200 noise strategies has a Sharpe of 1.19 and a PSR of 0.998, but a Deflated Sharpe
  of 0.41. PBO averages 0.42 for noise and 0.03 when real skill is present.
* **The validation gate.** TSMOM passes all 9 robustness tests with an out-of-sample Sharpe of 0.90 and still fails the
  gate (DSR 0.915, PBO 0.29): not enough evidence yet.
* **VaR.** A rolling normal 99% VaR is breached 41 times where 20 were expected. Equity correlation rises from 0.70 to
  0.94 in crises.
* **Kelly.** A 40%-win, 2:1 system has a Kelly fraction of 10%, and betting that much halves the account in 46% of
  paths. With an estimated edge, half Kelly beats full Kelly, and double Kelly loses money in 64% of paths.
* **Allocation.** Sample max-Sharpe weights flip sign between halves and blow up walk-forward. Out of sample, 1/N has
  the best Sharpe among the strategy allocators. Among asset allocators, shrunk minimum variance, HRP and risk parity
  have the best Sharpe ratios and shallowest drawdowns, and 1/N the highest return.

## Instructor material

- `solutions/`: the same notebooks with the answers filled in. **Remove this folder (or keep it on a private branch)
  before sharing the repository with learners.**
- `tools/build_notebooks.py`: the single source for starter and solution notebooks (it also contains the answers).
  Edit content there and rebuild with `python tools/build_notebooks.py`.

## Tests

```bash
python -m pytest -q tests     # 36 tests: solutions pass every check (strict mode), starters run with blanks,
                              # starter and solution notebooks differ only in the exercise cells
```
