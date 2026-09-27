# Part 9 guided notebooks — Advanced Statistical Trading

Guided notebooks for Part 9 ([lesson plan](../../docs/lessons/PART_09_STATISTICAL_TRADING.md)), one notebook per session.
The setup, data and plotting code is written for you; learners fill in short **✍️ Your turn** cells, replacing each
`...`. Each exercise ends with `p.check(...)`, which prints ✅ or ❌. If an answer isn't right yet, the notebook
continues with the reference value, so later cells still run.

All data is synthetic with **known parameters**, so every estimator can be checked against the truth:
- OU processes with a known half-life;
- truly cointegrated pairs hidden among unrelated stocks in a sector universe;
- a pair whose hedge ratio drifts from 1 to 2, and a pair whose relationship dies on day 900;
- a factor universe in which half the stocks have a reverting residual;
- a characteristics panel with a planted momentum premium.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part09/`](../../labs/part09/): the
same definitions, but here you also *see* what they do (a rolling z-score hiding a break, a Kalman filter swallowing the
spread, the best of 50 noise strategies looking significant).

| Notebook | Session | Exercises |
|---|---|---|
| `01_ou_bayes.ipynb` | S1 | An OU fit through its AR(1) form; the variance ratio; a normal–normal posterior mean. Plus: half-life intervals from 1 and 5 years of data, why a random walk looks slightly mean-reverting |
| `02_cointegration_screen.ipynb` | S2 | Engle–Granger; Benjamini–Hochberg. Plus: correlated random walks that aren't a pair, screening within sectors vs across the whole universe |
| `03_pairs_strategy.ipynb` | S3 | A rolling z-score; the pairs state machine (entry, exit, stop, time stop, no re-entry); pair P&L with costs and borrow. Plus: cost sensitivity, rolling vs formation z-scores on a pair that breaks |
| `04_kalman_hedge.ipynb` | S4 | Rolling OLS β; the Kalman filter update. Plus: tracking a drifting hedge ratio, a too-fast filter that absorbs the spread |
| `05_pca_residual.ipynb` | S5 | The Avellaneda–Lee s-score of one stock; the s-score trading rules. Plus: PCA factors, a dollar-neutral residual book with and without residual reversion |
| `06_factor_models.ipynb` | S6 | A Newey–West standard error; Fama–MacBeth; neutralization against sectors and beta. Plus: how many months a premium needs |
| `07_stability_breaks.ipynb` | S7 | The Chow test; a pair-retirement rule; White's Reality Check. Plus: CUSUM false alarms, a rolling stability dashboard, what retirement saves |
| `08_pairs_portfolio.ipynb` | S8 | Parameter ensembles; pair-book weights with per-name caps; stock-level exposures of a book of pairs. Plus: the entry/exit Sharpe grid, a 9-pair book out of sample |

`p9lib.py` holds the reference implementations the checks compare against (the same definitions as the labs), the
synthetic data and the chart style.

## Setup

```bash
cd notebooks/part09
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

All data is synthetic and generated with fixed seeds, so every notebook runs offline.

## What the notebooks show

* A half-life estimated from one year of data has a 90% interval of 3.3 to 7.6 days (true 7). With five years, the
  interval is 5.5 to 8.5.
* Two random walks whose returns are 0.74 correlated are not cointegrated (p = 0.61). Within sectors, all 4 true pairs
  pass Benjamini–Hochberg and nothing else does. Screening all 496 pairs instead of 112 loses one true pair.
* Out of sample, the textbook pair makes a Sharpe of 1.9 before costs, 1.2 at 5 bp a leg, and turns negative at 20 bp.
  A borrow fee of 300 bp a year costs only about 0.1.
* After a pair breaks, the rolling z-score re-scales and never reaches the stop (max |z| 3.2), so the strategy keeps
  trading a dead pair. A z-score on the formation mean and std reaches 16, and with a time stop the post-break loss
  disappears.
* The Kalman filter tracks a hedge ratio drifting from 1 to 2 within about 0.03. Rolling OLS is off by 0.24 to 0.48,
  whatever the window. On a pair with a constant β, δ ≥ 1e-6 lets the hedge absorb the spread: nothing trades.
* The s-score book earns a Sharpe of about 1.1 net of costs, with a market correlation of 0.004. When no residual
  reverts, it loses its costs (−1.7). Its positions are barely tilted toward the reverting stocks.
* Fama–MacBeth recovers the planted momentum premium (0.26% a month, t = 3.4). With 12 months of data the sign is wrong.
* Chow and CUSUM reject stability for a perfectly healthy cointegrated pair (CUSUM for 100 out of 100), because the
  spread is autocorrelated. A rolling-statistics retirement rule leaves it alone and retires the dying pair, turning
  −2.9% after the break into +0.7%.
* The best of 50 noise strategies has a naive p-value of 0.007. White's Reality Check gives 0.24.
* On one pair, the entry/exit grid ranges from a Sharpe of −0.1 to 1.6. A 9-parameter ensemble beats the average member
  on 8 of 8 pairs. A book of 9 nearly independent pairs makes 3.6. That is idealized: real pairs break, and they fall together
  in a crowded unwind.

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
