# Part 10 guided notebooks — Machine Learning, Deep Learning & Reinforcement Learning

Guided notebooks for Part 10 ([lesson plan](../../docs/lessons/PART_10_MACHINE_LEARNING.md)): twelve notebooks for the
24 sessions, two sessions each. The setup, data and plotting code is written for you; learners fill in short
**✍️ Your turn** cells, replacing each `...`. Each exercise ends with `p.check(...)`, which prints ✅ or ❌. If an
answer isn't right yet, the notebook continues with the reference value, so later cells still run.

All data is synthetic with a **known structure**, so you can see what a model is able to learn, and that it learns
nothing from noise:
- uneven tick activity;
- a hidden trend/chop regime;
- an AR(1) series;
- GARCH volatility;
- overlapping labels on a random walk;
- features with planted importances.

The rule for the whole Part: **a model must beat a simple baseline out of sample, after costs, under purged
validation.** Several notebooks are built so that the simple model wins.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part10/`](../../labs/part10/). The
labs use LightGBM, Optuna, hmmlearn and Gymnasium; the notebooks keep the same definitions with lighter substitutes:
- scikit-learn's `HistGradientBoosting` in place of LightGBM;
- a logged random search in place of Optuna;
- a small NumPy Gaussian HMM in place of hmmlearn;
- environments with the Gymnasium `reset`/`step` API, without the package.

PyTorch is needed only for notebooks 09 and 10.

| Notebook | Sessions | Exercises |
|---|---|---|
| `01_baseline_bars.ipynb` | S1–S2 | The direction dataset; dollar bars; the CUSUM filter. Plus: a walk-forward logistic baseline that only learns the base rate, time vs dollar bars |
| `02_fracdiff_feature_store.ipynb` | S3–S4 | Fractional-differencing weights; the Roll spread; a point-in-time feature store query. Plus: the minimum stationary d, a revision the store forgets, the truncation test |
| `03_labels_meta.ipynb` | S5–S6 | Triple-barrier labels; average uniqueness; bet sizing. Plus: the sequential bootstrap, momentum win rates by hidden regime |
| `04_models_regimes.ipynb` | S7–S8 | A reliability table; the HMM forward filter. Plus: three models against the true probability, calibrating a class-weighted model, GMM vs HMM regimes |
| `05_purged_cv_importance.ipynb` | S9–S10 | The purged k-fold split; MDA (and clustered MDA). Plus: AUC 0.7 on pure noise with shuffled folds, MDI vs MDA vs SFI |
| `06_audit_tuning_registry.ipynb` | S11–S12 | The shuffled-labels test; the overlap test; a seed ensemble. Plus: the canary and time-shift tests, a search whose best trial is a coin flip, registry gates |
| `07_ml_strategies.ipynb` | S13–S14 | Purged walk-forward retraining; HAR features; risk-off exposure. Plus: meta-labeled momentum, HAR vs gradient boosting, a crash warning judged by PR-AUC |
| `08_serving_drift.ipynb` | S15–S16 | PSI; a streaming rolling mean and std. Plus: PSI noise on small windows, a drift monitor on a known shift, shadow mode, the project-defense checklist |
| `09_pytorch_sequences.ipynb` | S17–S18 | A window dataset; split-conformal intervals. Plus: MLP, LSTM and TCN against the naive rule, what per-window normalization costs, conformal coverage after a volatility change |
| `10_dl_strategies.ipynb` | S19–S20 | A causal convolution; autoencoder anomaly scores; the pinball loss. Plus: MC dropout, quantile forecasts, seed variance |
| `11_rl_foundations_env.ipynb` | S21–S22 | Value iteration; the trading-environment reward; the differential Sharpe ratio. Plus: Q-learning on one path vs many, rule baselines |
| `12_rl_agents_execution.ipynb` | S23–S24 | The Q-learning update; the Almgren–Chriss schedule. Plus: agents across seeds, regimes and costs, the execution frontier, a simulated execution environment |

`p10lib.py` holds the reference implementations the checks compare against (the definitions of the labs), the
synthetic data and the chart style. `p10dl.py` holds the PyTorch code for notebooks 09 and 10.

## Setup

```bash
cd notebooks/part10
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

All data is synthetic and generated with fixed seeds, so every notebook runs offline, on a CPU.

## What the notebooks show

* **Baselines.** On a random walk, the walk-forward logistic regression's accuracy (51.9%) is exactly one minus the share
  of up days: it says "down" every day. On the regime market it scores below "always up".
* **Bars.** Dollar bars' returns are close to normal (Jarque–Bera 1.3); 30-minute bars' are not (16.1).
* **Memory.** Fractional differencing with d = 0.3 makes the log price stationary and keeps a 0.97 correlation with it.
  Returns keep 0.05.
* **Point in time.** A simple feature store that overwrites revisions forgets a number that was public for two months.
* **Overlap.** CUSUM-sampled labels have a mean uniqueness of 0.66; a label on every day, 0.27.
* **Calibration.** A class-weighted model's "probabilities" average 0.35 on a 10% event. Isotonic calibration brings
  them to 0.10.
* **Regimes.** The HMM's Viterbi path matches the hidden regime on 97% of days, with 24 switches (the truth has 28).
  The real-time filter scores 94% with 70 switches; a GMM, 92% with 42.
* **Validation.** A random forest scores an AUC of 0.70 on pure noise under shuffled k-fold, and 0.47 under purged
  k-fold.
* **Importance.** MDI gives continuous noise 7% of the importance and MDA about zero. Clustered MDA gives the signal
  cluster 0.37.
* **Tuning.** The best of 10 hyperparameter trials has a log loss of 0.6932 — a coin flip (ln 2).
* **Meta-labeling.** On the labs' 24-year market, meta-labeling lifts momentum's net Sharpe from −0.16 to 0.52 with a
  model whose AUC is only 0.54. The model bets 5 times more in the hidden trend regime than in chop.
* **Volatility.** HAR tracks the true GARCH volatility with a correlation of 0.97; gradient boosting reaches 0.62.
* **Crash warning.** The warning's PR-AUC is 2.2 times the 4% base rate. Halving exposure after a warning cuts the
  worst drawdown from −26% to −22%.
* **Drift.** With no drift at all, a PSI on 60 values exceeds 0.25 in 17% of tries.
* **Deep learning.** No network beats "tomorrow ∝ today" on an AR(1) with φ = 0.6, and per-window normalization costs
  them about 0.07 of correlation. Conformal intervals cover 89%, and 52% once the volatility doubles.
* **RL overfitting.** Q-learning memorizes one price path (0.36 per step in sample, 0.13 on new paths). Trained on 200
  paths, it keeps 0.15.
* **RL defense.**
  * On momentum data, five agents score from −0.55 to 2.13; the one-line momentum rule makes 2.13.
  * Agents trained with free trades lose at 30 bp; agents trained with costs stay flat.
* **Execution.** Almgren–Chriss pays 8.5% more expected cost than TWAP for 35% less variance.

## Instructor material

- `solutions/`: the same notebooks with the answers filled in. **Remove this folder (or keep it on a private branch)
  before sharing the repository with learners.**
- `tools/build_notebooks.py`: the single source for starter and solution notebooks (it also contains the answers).
  Edit content there and rebuild with `python tools/build_notebooks.py`.

## Tests

```bash
python -m pytest -q tests     # 36 tests, about 6 minutes: solutions pass every check (strict mode), starters run
                              # with blanks, starter and solution notebooks differ only in the exercise cells
```
