# Part 9 labs — Advanced Statistical Trading

Auto-graded exercises for [Part 9](../../docs/lessons/PART_09_STATISTICAL_TRADING.md) (program weeks 31–32,
milestone **M5b**). Everything runs offline on synthetic data from `common.py`. Every generator has known true
parameters, so the tests can check that your estimators recover them. The generators include OU processes,
cointegrated pairs, a pair whose beta drifts, a pair whose relationship dies, a sector universe with planted
cointegrated pairs, a factor universe with reverting residuals and a characteristics panel with a planted premium.

Fill in each `raise NotImplementedError("✍️ Your turn ...")` and run the tests until they are green.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week31_pairs/](week31_pairs/) | S1–S4 | OU fit via AR(1), variance ratio, conjugate normal posterior, block-bootstrap CI of the half-life, Engle–Granger and Johansen, Benjamini–Hochberg, a within-sector pair screen with FDR and a half-life filter, rolling z-score, the pairs state machine (entry, exit, stop, time stop, no re-entry after a stop), pair P&L with costs and borrow, the Kalman hedge and rolling OLS beta | `python -m pytest week31_pairs` |
| [week32_statarb/](week32_statarb/) | S5–S8 | Avellaneda–Lee s-scores (checked against the lesson-plan code), s-score trading rules and a dollar-neutral residual backtest, Newey–West, Fama–MacBeth, neutralization against sector and beta, Chow and CUSUM, rolling stability and a retirement rule, stationary bootstrap and White's Reality Check, parameter ensembles, pair-book weights with per-name caps, stock-level exposures of a book of pairs | `python -m pytest week32_statarb` |
| [clinic_w1_screen/](clinic_w1_screen/) | Clinic W1 | Screen a sector universe in the formation period, then trade the selected pairs out of sample with a fixed, a rolling and a Kalman hedge, next to the pairs that a naive screen would add | `python screen.py` |
| [clinic_w2_book/](clinic_w2_book/) | Clinic W2 | A market-neutral book of 19 pairs: inverse-vol weights with name caps, daily gross, net and net beta, and a risk report with VaR and a crowded unwind | `python book.py` |

Later labs use earlier ones: week 32 uses your week 31 `engle_granger` and `pairs_positions`, and the clinics use the
weeks they follow.

## Setup

```bash
cd labs/part09
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week31_pairs
python -m pytest                          # everything
```

Run all commands from `labs/part09` (its `conftest.py` makes the imports work).

## Things the labs make you notice

* **Screen within sectors, and correct for many tests.** In the 4-sector universe, 7 pairs have a raw p < 0.05 and only
  the 4 true pairs pass BH-FDR. If you screen all 496 pairs instead of the 112 within sectors, FDR drops one of the
  true pairs: more tests cost power.
* **False discoveries do not trade.** In clinic W1, the FDR-selected pairs have a mean out-of-sample Sharpe of about
  1.2. The extra pairs a naive screen adds (raw p < 0.05) have about 0.1.
* **A Kalman hedge is not free.** When the true β is constant, δ = 1e-5 lets the hedge chase the spread. Its mean
  out-of-sample Sharpe is about −1.0, against +0.5 with δ = 1e-7 (lesson plan, mistake 8). When β really drifts
  (week 31), the Kalman filter tracks it far better than a 250-day rolling OLS.
* **Break tests assume iid errors.** Chow and CUSUM work on a regression with iid noise. On the price levels of a
  perfectly stable cointegrated pair, CUSUM still rejects, because the spread is autocorrelated. Monitor rolling
  p-values, half-lives and β instead: the retirement rule catches a pair that dies at day 900 within one window.
* **The best of 50 noise strategies looks significant.** Its naive p-value is 0.007. White's Reality Check gives 0.24.
* **Residual reversion pays only where it exists.** The s-score book makes a Sharpe of about 1 when half the stocks
  have reverting residuals, and loses its costs (Sharpe about −1.7) when none do. It is market neutral either way.
* **Diversification is not a hedge.** Clinic W2's 19 pairs make a Sharpe of about 4.7 on this idealized data (the
  pairs are independent and never break; real ones are neither). A crowded unwind, in which every pair loses 3σ on
  the same day, costs about 7 times the historical 99% VaR.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P9_SOLUTIONS=1 python -m pytest` (31 tests pass). On the blank starters, every failure is a
  `NotImplementedError`; tests that share a fixture report errors instead.
* **Before sharing with learners, remove `solutions/` and `tools/`.** The sync test then skips itself.
* The pairs P&L uses log-price legs and a fixed capital of 1 + |β| per unit spread. Borrow costs, costs of re-hedging a
  dynamic β and legging risk are simplified; the lesson plan's OMS pair order covers the execution side.
