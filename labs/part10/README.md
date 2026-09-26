# Part 10 labs — Machine Learning, Deep Learning & Reinforcement Learning

Auto-graded exercises for [Part 10](../../docs/lessons/PART_10_MACHINE_LEARNING.md) (program weeks 33–38, milestones
**M6** and **M7a**). Everything runs offline, on CPU, on synthetic data from `common.py`. Each generator has a
**known** structure: uneven tick activity, a hidden trend/chop regime, an AR(1) series, GARCH volatility,
overlapping labels on pure noise, and features with planted importances. So the tests can check what a model can
learn, and that it learns nothing from noise.

The rule for the whole Part: **every model must beat a simple baseline out of sample, after costs, under purged
validation.** Several labs are built so that the simple model wins.

Fill in each `raise NotImplementedError("✍️ Your turn ...")` and run the tests until they are green.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week33_features/](week33_features/) | S1–S4 | Logistic walk-forward baseline, dollar bars vs time bars, CUSUM events, fractional differentiation and the minimum d, Roll spread and Amihud, a point-in-time feature store (publication delays, revisions) and the truncation test | `python -m pytest week33_features` |
| [week34_labels/](week34_labels/) | S5–S8 | Triple-barrier labels, concurrency and uniqueness, sequential bootstrap, meta-labels and bet sizing, point-in-time ML features, weighted logit / random forest / LightGBM, isotonic and Platt calibration, reliability tables, GMM and HMM regimes | `python -m pytest week34_labels` |
| [week35_validation/](week35_validation/) | S9–S12 | Purged k-fold with embargo (scikit-learn compatible), MDI / MDA / SFI and clustered MDA, the leakage audit (shuffled labels, canary, time shift, overlap), Optuna search that logs every trial, seed ensembles, a model registry with stages | `python -m pytest week35_validation` |
| [week36_strategies/](week36_strategies/) | S13–S15 | Walk-forward meta-labeled momentum with purged retraining, HAR vs LightGBM volatility, a crash warning used only to cut risk (PR-AUC), rank IC, PSI and KS drift monitoring, streaming features with train/serve parity, shadow-mode champion vs challenger | `python -m pytest week36_strategies` |
| [week37_deep/](week37_deep/) | S17–S20 | Window datasets (window or training-period normalization), splits with gaps, a training loop with early stopping, MLP, LSTM and causal TCN, pinball loss, split-conformal intervals, MC dropout, an autoencoder anomaly score | `python -m pytest week37_deep` |
| [week38_rl/](week38_rl/) | S21–S24 | Value iteration, tabular Q-learning on one path vs many, a Gymnasium trading environment (timing, costs, no look-ahead scale), the differential Sharpe ratio, a Q-learning agent, TWAP vs Almgren–Chriss and an execution environment | `python -m pytest week38_rl` |
| [clinic_w3_ml_report/](clinic_w3_ml_report/) | Clinic W3 | The ML research report: shuffled vs purged CV for three model families, the full audit, walk-forward results against the primary rule, the trial count for the DSR | `python ml_report.py` |
| [clinic_w5_dl/](clinic_w5_dl/) | Clinic W5 | LSTM vs LightGBM vs HAR on next-day volatility: walk-forward, the same target days, seed variance, conformal coverage | `python dl_vs_gbm.py` |
| [clinic_w6_rl/](clinic_w6_rl/) | Clinic W6 | RL agent defense: 5 seeds × 3 regimes against four rules, and what an optimistic cost assumption does. Optional PPO (Stable-Baselines3) | `python rl_defense.py [--ppo]` |

Later labs use earlier ones: week 34 uses the week 33 CUSUM filter, week 35 uses week 34's models, week 36 uses
weeks 33–34, and the clinics use the weeks they follow.

## Setup

```bash
cd labs/part10
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week33_features
python -m pytest                          # everything, about a minute on a laptop CPU
```

Run all commands from `labs/part10` (its `conftest.py` makes the imports work).

## Things the labs make you notice

* **A shuffled k-fold finds skill in noise.** On a random walk with overlapping 20-day labels, a random forest scores
  an AUC of 0.77 under shuffled k-fold and 0.48 under purged k-fold. With CUSUM-sampled events (clinic W3, mean
  uniqueness 0.68), the gap almost disappears. Event sampling is itself a defence.
* **MDI flatters noise.** A continuous noise feature gets 17% of the MDI importance and about zero MDA. Two
  near-copies of the signal share their MDA (0.18 for the stronger one); clustered MDA gives the cluster 0.40.
* **Meta-labeling helps, a little.** On the regime market, the momentum rule's win rate is 48% in chop and 60% in
  trend. The walk-forward meta-model has an AUC of about 0.55. Yet it lifts the net Sharpe from −0.16 to 0.52
  (random forest), with fewer and better trades. That is 9 trials to report to the DSR.
* **Simple models win where they should.**
  * HAR forecasts GARCH volatility with a correlation of 0.97 to the true volatility. LightGBM gets 0.71.
  * On an AR(1) series with φ = 0.6, the LSTM reaches a correlation of 0.54 and the "tomorrow like today" rule 0.61.
  * In clinic W5, HAR beats both LightGBM and a 3-seed LSTM on next-day volatility.
* **Conformal intervals need exchangeability.** They cover 90% on the AR(1) series, and 86% when the calibration
  period's volatility regime differs from the test period's.
* **Rare events are judged by PR-AUC.** The crash warning's PR-AUC is 2.3 times the 4% base rate. Halving exposure
  after a warning reduces the drawdown; it is never used as a short signal.
* **PSI on small windows is noisy.** Two samples from the same distribution give a PSI of about 0.005, and a 0.5σ
  shift gives about 0.25. With 60 values in 10 bins, PSI exceeds 0.25 with no drift at all.
* **RL overfits one history.** Q-learning on a single path earns 0.36 per step in sample and 0.13 on new paths.
  Trained on 200 paths, it does better on new paths.
* **An RL agent must be judged across seeds, regimes and costs** (clinic W6).
  * On momentum data, five Q-learning seeds average 0.97 with a spread of ±0.99. The momentum rule makes 2.13.
  * On noise the agent learns to stay out.
  * Trained with free trades, it trades 600 times and loses 1.06 at real costs.
  * PPO (30k steps, optional) makes 1.6–2.1: at best the momentum rule it rediscovers.
* **Execution is where RL has a benchmark.** Almgren–Chriss accepts about 8% more expected cost than TWAP for about
  35% less variance. No nearby schedule beats it on its own mean–variance objective.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P10_SOLUTIONS=1 python -m pytest` (64 tests pass, about a minute). On the blank starters,
  every failure is a `NotImplementedError`; tests that share a fixture report errors instead.
* **Before sharing with learners, remove `solutions/` and `tools/`.** The sync test then skips itself.
* The lesson plan's MLflow registry, SHAP values, ONNX export and the hedging environment are left out to keep the
  labs offline and light. The registry here is a JSON file with the same stages; MDA stands in for SHAP.
