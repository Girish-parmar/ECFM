# Part 2 guided notebooks — Quantitative Toolkit

Guided notebooks for the Part 2 labs ([lesson plan](../../docs/lessons/PART_02_QUANT_TOOLKIT.md)).
Loading and plotting code is provided; learners fill in short **✍️ Your turn** cells (1–5 lines, replacing `...`).
Each exercise ends with `p.check(...)`, which prints ✅ or ❌. If an answer is not right yet, the notebook
continues with the reference value so later cells still run.

| Notebook | Session | Exercises |
|---|---|---|
| `01_linear_algebra_pca.ipynb` | S1 | portfolio volatility `√(wᵀΣw)`; PCA explained variance; Cholesky factor (then correlated simulation, condition numbers) |
| `02_calculus_optimization.ipynb` | S2 | delta-gamma approximation; minimum-variance weights via `solve`; Newton step for implied volatility (plus `cvxpy` long-only portfolio and efficient frontier) |
| `03_distributions_stylized_facts.ipynb` | S3, clinic W1 | log returns; total return from summed log returns; 4σ tail count; monthly kurtosis (plus QQ plot, t fit, the stylized-facts table) |
| `04_inference_regression.ipynb` | S4 | t-stat of a mean return; years of data needed for a given Sharpe; significant results before/after BH-FDR (plus block bootstrap, CAPM with HAC errors) |
| `05_stationarity_mean_reversion.ipynb` | S5 | variance ratio; half-life of a spread (plus ADF/KPSS table, spurious regression, AR(1) vs zero forecast) |
| `06_volatility_garch.ipynb` | S6, clinic W2 | EWMA recursion; QLIKE loss (plus GARCH-t/GJR fits, out-of-sample forecast competition, realized volatility) |
| `07_cointegration_hmm_montecarlo.ipynb` | S7 | Engle–Granger hedge ratio; volatility by regime from the **filtered** (no look-ahead) HMM probability; Monte Carlo drawdown probability |
| `08_metrics_var.ipynb` | S8, clinic W2 | Sharpe; maximum drawdown; historical VaR/CVaR (plus tear-sheet table, VaR methods, the smoothed-returns Sharpe pitfall) |

`p2lib.py` holds the data loader, the reference implementations the checks compare against, and the chart style.

## Data

Set `P2_DATA` to the course price file (CSV or Parquet; one row per date, one column per ticker:
`SPY, QQQ, IWM, TLT, GLD, XLE, AAPL, JPM, XOM, SMLCAP`). Without it, the notebooks use a **synthetic market**
with realistic behaviour (fat tails, negative skew, volatility clustering, a leverage effect, a common market
factor, a cointegrated XLE/XOM pair), so everything runs offline. Use the course data for graded work.
The 5-minute bars in notebook 06 are always synthetic in Part 2.

## Instructor material

- `solutions/` — the same notebooks with the answers filled in. **Remove this folder (or keep it on a private
  branch) before sharing the repository with learners.**
- `tools/build_notebooks.py` — single source for starter and solution notebooks (it also contains the answers).
  Edit content there and rebuild with `python tools/build_notebooks.py`.

## Tests

```bash
python -m pytest -q tests     # 24 tests: solutions pass every check (strict mode), starters run with blanks,
                              # starter and solution notebooks differ only in the exercise cells
```
