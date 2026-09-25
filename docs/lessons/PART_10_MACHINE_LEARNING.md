# Part 10 — Machine Learning, Deep Learning & Reinforcement Learning: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 3, **Month 9 + first half of Month 10** (program weeks 33–38); Part 11 (AI, NLP & Cowork) follows in weeks 39–40 |
| Format | 24 sessions × 90 min (4 per week) + 6 lab clinics × 120 min + self-study (~7 hrs/week) |
| Total effort | ~36 hrs live + 12 hrs clinic + 42 hrs self-study ≈ **90 hours** |
| Compute | CPU is enough for ML (weeks 1–4); GPU credits for deep learning and RL (weeks 5–6) |
| Platform milestones | **M6** (end of week 36): feature store, labeling, purged-CV ML pipeline, model registry, drift monitoring, rare-event model · **M7a** (end of week 38): DL and RL research modules with walk-forward retraining |
| Covers original items | "Trade Creation" 5–16 (ML, DL, RL: theory → create strategy → advanced analysis → optimization) · "Machine learning basic" · Platform roadmap 44, 45, 50 |

**The rule for this whole Part:** every ML, DL and RL model must beat a **simple baseline** (e.g. the Part 7 rule it is meant to improve, or a logistic regression) **out of sample, after costs, under purged cross-validation, and through the Part 8 validation gate**. Most will not. Learning to show that honestly is the skill.

---

## 1. Learning Objectives

By the end of Part 10 the learner will be able to:

1. **Frame** trading problems for ML correctly (what to predict, at which events, over which horizon) and explain why financial ML is harder than typical ML.
2. **Build** leakage-free datasets: information-driven bars, event sampling, stationary-but-memory-preserving features (fractional differentiation), point-in-time joins, a feature store.
3. **Label** with the triple-barrier method and **meta-labeling**, handle overlapping labels with sample weights, and turn probabilities into bet sizes.
4. **Train, validate and interpret** tree ensembles and linear models with purged k-fold/CPCV, calibrated probabilities, MDA/SHAP importance, and leakage audits.
5. **Create** ML strategies (direction with meta-labeling, cross-sectional ranking, regime switching, volatility forecasting, rare-event warning) and deploy them with drift monitoring.
6. **Build** deep-learning models in PyTorch (MLP, LSTM/GRU, TCN, Transformers, autoencoders) with window-level normalization, early stopping, uncertainty estimates and walk-forward retraining.
7. **Design** reinforcement-learning trading environments (state, action, reward, costs) and train and evaluate agents (DQN, PPO, SAC) for trading, execution and hedging, avoiding reward hacking and overfitting.
8. **Optimize** all three model families under the same validation discipline, and decide with evidence when *not* to use them.

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| 2.1–2.3 Math, statistics, time series | All weeks |
| 3.2 Advanced Python (vectorization, `numba`, multiprocessing) | W1–W2 |
| Part 5 indicators, sentiment, streaming indicators | W1 features, S15 deployment |
| Part 7 strategies (primary models for meta-labeling) | S6, S13 |
| Part 8 walk-forward, purged k-fold, CPCV, DSR, PBO, validation gate, sizing | W3 onward |
| Part 9 factor models, OU, Kalman | S14, S19 |

---

## 3. Six-Week Overview

| Week | Theme | Sessions | Clinic Lab | Platform Output |
|---|---|---|---|---|
| **W1** | Data & features for ML | S1 Framing financial ML · S2 Bars & event sampling · S3 Features I: fractional differentiation & microstructure · S4 Features II & the feature store | Feature store for 50 ETFs with a leakage audit | `research/ml/bars.py`, `features/`, `feature_store.py` |
| **W2** | Labels & models | S5 Triple-barrier labels & sample weights · S6 Meta-labeling & bet sizing · S7 Supervised models · S8 Unsupervised learning & regimes | Meta-labeled momentum strategy, first version | `research/ml/labels.py`, `models.py`, `regimes.py` |
| **W3** | Validation & interpretation | S9 Purged CV for ML · S10 Feature importance · S11 Leakage & overfitting audit · S12 Hyperparameters, ensembles & model registry | Full CV + importance + audit report | `research/ml/cv.py`, `importance.py`, `audit.py`, MLflow registry |
| **W4** | ML strategies & deployment | S13 ML strategy I: direction + meta-labeling · S14 ML strategy II: ranking, regimes, volatility, rare events · S15 Deployment & drift · S16 ML project defense & M6 | Month-9 ML project through the validation gate | `strategy/library/ml_*.py`, `research/ml/serving.py`, `monitoring/drift.py`, **M6 release** |
| **W5** | Deep learning | S17 PyTorch for time series · S18 Sequence models & uncertainty · S19 DL strategies & analysis · S20 DL optimization & honest comparison | DL vs LightGBM baseline, walk-forward | `research/dl/` |
| **W6** | Reinforcement learning | S21 RL foundations · S22 Trading environment design · S23 RL strategies: trading, execution, hedging · S24 RL optimization & M7a | RL agent vs rule baseline, robustness across seeds and regimes | `research/rl/`, **M7a release** |

**The four-step matrix (original items 5–16), and where each step is taught:**

| Family | Theory | Create strategy | Advanced analysis | Optimization |
|---|---|---|---|---|
| Machine learning | S1–S8 | S13–S14 | S9–S11 | S12, S15 |
| Deep learning | S17–S18 | S19 | S19 | S20 |
| Reinforcement learning | S21 | S22–S23 | S23 | S24 |

---

## 4. Session-by-Session Plan

> Each session: **15 min recap/theory → 45 min live coding → 20 min guided lab → 10 min wrap-up and homework.**
> Notebooks live in `notebooks/part10/`; promoted code lives in `quantforge/research/{ml,dl,rl}/`.

### Week 1 — Data & Features for ML

#### S1 · Framing Financial ML (item 5)

| Block | Content |
|---|---|
| Theory | Why finance is hard for ML: very low signal-to-noise, non-stationarity (the data-generating process changes), few independent samples (overlapping labels), adversarial markets (edges decay when found), heavy tails. **Problem framing:** direction classification, return regression, cross-sectional ranking, regime classification, volatility forecasting, rare-event detection, meta-labeling (should I take this trade?). **When not to use ML:** when a simple rule does as well, when there is too little data, when you cannot explain the inputs. The ML research workflow on top of the Part 8 research log. |
| Live coding | Baseline first: logistic regression on 5 lagged returns for SPY direction under a simple walk-forward; record it in the research log. |
| Lab | Evaluate the baseline on 10 ETFs; accuracy vs 50% and vs "always up"; profitability after costs. |
| Homework | Read López de Prado (2018), chapters 1–2. |

#### S2 · Financial Data Structures & Event Sampling (item 5)

| Block | Content |
|---|---|
| Theory | Time bars sample too much in quiet periods and too little in busy ones. **Information-driven bars:** tick, volume and **dollar bars** have returns closer to i.i.d. normal. **Event-based sampling:** the **CUSUM filter** samples only when something happened (cumulative move > h), so the model learns at relevant moments and samples overlap less. A bar is known when it **closes**; its timestamp must be the close time. |
| Live coding | `dollar_bars` and `cusum_events` (below). |
| Lab | Compare time, volume and dollar bars from Part 4 tick data: normality (Jarque–Bera) and autocorrelation of returns. |
| Homework | Imbalance bars (tick-imbalance) as an extension. |

```python
def dollar_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """ticks: index=ts, columns price, size. A bar closes each time traded dollar value reaches threshold."""
    dv = (ticks["price"] * ticks["size"]).cumsum()
    bar_id = (dv // threshold).astype(int)
    g = ticks.groupby(bar_id.to_numpy())
    bars = g["price"].agg(open="first", high="max", low="min", close="last")
    bars["volume"] = g["size"].sum()
    bars["ts"] = g.apply(lambda d: d.index[-1], include_groups=False)   # known when it CLOSES
    return bars.set_index("ts")

def cusum_events(close: pd.Series, h: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on log returns: sample an event when cumulative drift exceeds h."""
    events, s_pos, s_neg = [], 0.0, 0.0
    for ts, x in np.log(close).diff().dropna().items():
        s_pos, s_neg = max(0.0, s_pos + x), min(0.0, s_neg + x)
        if s_pos > h:
            s_pos = 0.0; events.append(ts)
        elif s_neg < -h:
            s_neg = 0.0; events.append(ts)
    return pd.DatetimeIndex(events)
```

#### S3 · Features I: Fractional Differentiation & Microstructure (item 5)

| Block | Content |
|---|---|
| Theory | The stationarity vs memory dilemma: prices have memory but are non-stationary; returns are stationary but have almost no memory. **Fractional differentiation** (fixed-width window) with the minimum `d` that passes the ADF test keeps most of the memory. **Feature families:** multi-horizon returns and volatilities, Part 5 indicators normalized (z-scores, percentile ranks), volatility estimators, **microstructure features** (Roll spread, Amihud illiquidity, Kyle's lambda, order-flow imbalance, VPIN intro), calendar features. Every feature documented with its look-back and point-in-time source. |
| Live coding | `ffd_weights` and `frac_diff` (below); search for the minimum `d` per asset. |
| Lab | Minimum `d` for 20 assets; ADF p-value and correlation with the original series at each `d`. |
| Homework | Amihud and Roll-spread features from daily bars; Kyle's lambda from Part 4 intraday data. |

```python
def ffd_weights(d: float, thresh: float = 1e-4) -> np.ndarray:
    """Fixed-width-window fractional differentiation weights (López de Prado 2018, ch. 5)."""
    w, k = [1.0], 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k); k += 1
    return np.array(w[::-1])

def frac_diff(x: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    w = ffd_weights(d, thresh)
    vals = np.convolve(x.to_numpy(), w[::-1], mode="valid")      # uses only past values
    return pd.Series(vals, index=x.index[len(w) - 1:])
```

> Verified: on a simulated log-price random walk, the log price is non-stationary (ADF p ≈ 0.56), while its FFD transform with d = 0.4 is stationary (ADF p < 1e-4) yet still 0.74 correlated with the original.

#### S4 · Features II & the Feature Store (item 5)

| Block | Content |
|---|---|
| Theory | **Cross-sectional features** (ranks across the universe each date), **macro and sentiment** features joined point-in-time (Part 5 `available_at`), options-derived features (IV rank, skew, term structure from Part 6), stat-arb features (s-scores from Part 9). **Feature store:** each value stored with the time it became available; `get_features(symbols, as_of)` must return exactly what was known then; same code for batch (training) and streaming (live, Part 5 streaming indicators). Scaling fitted inside each training fold only. |
| Live coding | DuckDB/Parquet feature store with an `as_of` API and feature metadata. |
| Lab | Build the store for 50 ETFs (≈ 40 features); run the Part 5 truncation test on every feature. |
| Homework | Feature documentation page generated from metadata. |

**Clinic W1:** feature store + leakage audit report (truncation test, `as_of` consistency, scaling inside folds).

---

### Week 2 — Labels & Models

#### S5 · Triple-Barrier Labels & Sample Weights (item 6)

| Block | Content |
|---|---|
| Theory | Fixed-horizon labels ignore the path and use a fixed threshold in changing volatility. **Triple barrier:** upper (profit) and lower (stop) barriers scaled by current volatility, plus a vertical (time) barrier; the label is which is touched first; `t1` records when the label is resolved (needed for purging). **Overlapping labels** are not independent: compute concurrency and **uniqueness weights**; sequential bootstrap for bagging. Class balance. |
| Live coding | `triple_barrier` (below) on CUSUM events; uniqueness weights. |
| Lab | Label distribution vs barrier widths and max holding period; average uniqueness. |
| Homework | Return-attribution weights (weight samples by absolute return). |

```python
def triple_barrier(close: pd.Series, events: pd.DatetimeIndex, vol: pd.Series,
                   pt=1.0, sl=1.0, max_hold=10) -> pd.DataFrame:
    """Label each event by the first barrier touched: +1 upper, -1 lower, 0 vertical (time).
    Barriers are pt/sl multiples of volatility at the event. t1 = when the label is resolved."""
    out, idx = [], close.index
    for t0 in events:
        i0 = idx.get_loc(t0)
        if i0 + 1 >= len(idx) or np.isnan(vol.loc[t0]):
            continue
        path = close.iloc[i0 + 1: i0 + 1 + max_hold] / close.iloc[i0] - 1
        up, dn = pt * vol.loc[t0], -sl * vol.loc[t0]
        hit_up, hit_dn = path[path >= up].index.min(), path[path <= dn].index.min()
        t1 = min([t for t in (hit_up, hit_dn) if pd.notna(t)], default=path.index[-1])
        label = 1 if t1 == hit_up else -1 if t1 == hit_dn else 0
        out.append((t0, t1, path.loc[t1], label))
    return pd.DataFrame(out, columns=["t0", "t1", "ret", "label"]).set_index("t0")
```

#### S6 · Meta-Labeling & Bet Sizing (item 6)

| Block | Content |
|---|---|
| Theory | **Meta-labeling:** a primary model (a Part 7 rule, a discretionary view, or another model) decides the **side**; a secondary ML model predicts whether acting on that signal will be profitable, i.e. the **size** (0 = skip). Advantages: keeps an interpretable primary model, raises precision, reduces overfitting of the side decision. **Bet sizing** from calibrated probabilities (a probability of 0.55 should mean a small bet); averaging active bets; discretization to avoid over-trading. |
| Live coding | `meta_labels` and `bet_size` (below) on a Part 7 momentum primary. |
| Lab | Precision, recall and F1 of the primary alone vs primary + meta-model; P&L after costs. |
| Homework | Meta-label a Part 9 pairs strategy. |

```python
def meta_labels(side: pd.Series, barrier_ret: pd.Series) -> pd.Series:
    """1 if taking the primary model's side would have made money, else 0."""
    return ((side * barrier_ret) > 0).astype(int)

def bet_size(prob: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Calibrated probability of success -> position size in [0, 1] (López de Prado 2018, ch. 10)."""
    from scipy.stats import norm
    z = (prob - 1 / n_classes) / np.sqrt(prob * (1 - prob))
    return np.clip(2 * norm.cdf(z) - 1, 0, 1)

bet_size(np.array([0.50, 0.55, 0.70, 0.90]))   # -> [0.00, 0.08, 0.34, 0.82]
```

#### S7 · Supervised Models (item 5)

| Block | Content |
|---|---|
| Theory | Regularized linear/logistic regression (L1/L2/elastic net) as the honest baseline; decision trees; **random forests** (with `max_samples` set from average uniqueness); **gradient boosting** (LightGBM, XGBoost): depth, learning rate, subsampling, early stopping on a *purged* validation fold; SVMs and kNN (why they rarely help here). Bias–variance in a low-signal world (prefer high bias). Class imbalance. **Probability calibration** (Platt, isotonic) because bet sizing needs real probabilities. |
| Live coding | Logistic vs random forest vs LightGBM on the S6 meta-labels with sample weights; reliability diagrams. |
| Lab | Which model family wins after costs and purged CV (preview of S9)? |
| Homework | Calibrate the best model; compare bet-sized P&L before and after calibration. |

#### S8 · Unsupervised Learning & Market Regimes (item 5)

| Block | Content |
|---|---|
| Theory | Clustering for **regime detection**: k-means and Gaussian mixtures on volatility/trend/correlation features, **hidden Markov models** (Part 2) with persistent states; using regime labels as features or to switch strategies. PCA for feature compression and to reduce multicollinearity (links to Part 9 PCA). Clustering features before importance analysis (S10). |
| Live coding | GMM and HMM regimes on SPY features; regime-conditional performance of Part 7 strategies. |
| Lab | Do regimes found in 2010–2019 still separate performance in 2020–2025? |
| Homework | Regime-switching allocator (turn strategies on/off by regime), evaluated walk-forward. |

**Clinic W2:** first version of a meta-labeled momentum strategy with triple-barrier labels, weights and a calibrated LightGBM.

---

### Week 3 — Validation & Interpretation

#### S9 · Cross-Validation for Financial ML (item 7)

| Block | Content |
|---|---|
| Theory | Revisit Part 8 purging and embargo, now with labels that span `[t0, t1]`: a training sample is purged if its label window overlaps the test fold. A scikit-learn-compatible splitter so every `sklearn` tool (grid search, `cross_val_score`, calibration) respects it. **CPCV** for a distribution of OOS paths. **Metrics:** log-loss (for probabilities), AUC, precision at the operating threshold, and always the **trading metric** (net Sharpe of bet-sized positions). |
| Live coding | `PurgedKFold` (below); `cross_val_score` with it; CPCV paths from Part 8. |
| Lab | Naive k-fold vs purged k-fold vs CPCV on the same model: how inflated is naive CV? |
| Homework | Plot the CPCV OOS Sharpe distribution for the S13 strategy. |

```python
class PurgedKFold:
    """scikit-learn compatible K-fold for labels spanning [t0, t1]: purge train samples whose
    labels overlap the test fold and embargo a fraction of samples after it."""
    def __init__(self, t1: pd.Series, n_splits=5, embargo=0.01):
        self.t1, self.n_splits, self.embargo = t1, n_splits, embargo      # t1 indexed by t0
    def get_n_splits(self, *args, **kwargs):
        return self.n_splits
    def split(self, X, y=None, groups=None):
        t0 = self.t1.index
        n, emb = len(t0), int(np.ceil(self.embargo * len(t0)))
        for test in np.array_split(np.arange(n), self.n_splits):
            test_start, test_end = t0[test[0]], self.t1.iloc[test].max()
            emb_end = t0[min(test[-1] + emb, n - 1)]
            train = np.where((self.t1.to_numpy() < test_start) | (t0 > max(test_end, emb_end)))[0]
            yield train, test
```

#### S10 · Feature Importance (item 7)

| Block | Content |
|---|---|
| Theory | **MDI** (impurity-based, in-sample, biased to high-cardinality features); **MDA** (permutation importance, out of sample under purged CV: the one to trust); **SFI** (single-feature importance, no substitution effects); **SHAP** values for local and global explanations. **Substitution effects:** correlated features share importance, so cluster features first (clustered MDA). Importance is for understanding and pruning, not proof of an edge. |
| Live coding | `mda_importance` (below) with purged CV; SHAP summary for LightGBM; clustered MDA. |
| Lab | Compare MDI, MDA, SHAP rankings; prune to the top clusters and re-evaluate OOS. |
| Homework | Plant a noise feature and a leaked feature; which methods catch each? |

```python
from sklearn.metrics import log_loss

def mda_importance(model, X, y, cv, n_repeats=3, seed=0):
    """Mean Decrease Accuracy under purged CV: out-of-fold log-loss increase when a feature is shuffled."""
    rng = np.random.default_rng(seed)
    drops = {c: [] for c in X.columns}
    for tr, te in cv.split(X):
        m = model.fit(X.iloc[tr], y.iloc[tr])
        base = log_loss(y.iloc[te], m.predict_proba(X.iloc[te]), labels=m.classes_)
        for c in X.columns:
            for _ in range(n_repeats):
                Xs = X.iloc[te].copy()
                Xs[c] = rng.permutation(Xs[c].to_numpy())
                drops[c].append(log_loss(y.iloc[te], m.predict_proba(Xs), labels=m.classes_) - base)
    return pd.Series({c: np.mean(v) for c, v in drops.items()}).sort_values(ascending=False)
```

> Verified on a random-walk price series (no real signal): all MDA importances are within ±0.02 of zero, as they should be.

#### S11 · Leakage & Overfitting Audit (item 7)

| Block | Content |
|---|---|
| Theory | Audits every ML strategy must pass: **shuffled-labels test** (performance must collapse to chance), **future-feature canary** (a deliberately leaked feature must be flagged), **time-shift test** (shifting features one bar later should not *improve* results), truncation test, scaler-fit-inside-fold check, duplicate/overlap check between train and test. Then the Part 8 tools: every model/feature-set/hyperparameter trial goes into the research log, so **DSR** and **PBO** reflect the true search size. |
| Live coding | `audit.py` running all tests and producing a report. |
| Lab | Audit the S13 draft; fix whatever fails. |
| Homework | Add an audit step to CI for any file under `strategy/library/ml_*`. |

#### S12 · Hyperparameters, Ensembles & Model Registry (item 8)

| Block | Content |
|---|---|
| Theory | Optuna under purged CV/CPCV (never on the test period); small search spaces; early stopping inside folds. **Ensembles:** bagging with sequential bootstrap, averaging seeds, stacking (with purged out-of-fold predictions). **MLflow model registry:** model, feature list and versions, training window, CV scores, audit report, research-log trial count; stages (research → shadow → paper → live). |
| Live coding | Optuna + `PurgedKFold` study for LightGBM; register the best model with its artifacts. |
| Lab | Ensemble of 5 seeds vs single model: OOS stability. |
| Homework | Stacked model (logistic over LightGBM + linear) with purged out-of-fold predictions. |

**Clinic W3:** full ML research report: CV comparison, importance, audit results, DSR/PBO from the log.

---

### Week 4 — ML Strategies & Deployment

#### S13 · ML Strategy I: Direction with Meta-Labeling (item 6)

| Block | Content |
|---|---|
| Theory | End-to-end strategy: CUSUM events → primary signal (Part 7 momentum or mean reversion) → triple-barrier meta-labels → features from the store → calibrated LightGBM → bet sizing → Part 8 sizing and risk engine. Retraining schedule (walk-forward, e.g. monthly with an expanding window). |
| Live coding | `MLMetaStrategy(Strategy)` running in the Part 8 backtester with periodic retraining. |
| Lab | Primary alone vs primary + meta-model: net Sharpe, drawdown, turnover, DSR. |
| Homework | Start the Month-9 project (Section 7, part A). |

#### S14 · ML Strategy II: Ranking, Regimes, Volatility & Rare Events (items 6, 50)

| Block | Content |
|---|---|
| Theory | **Cross-sectional ranking:** predict next-period relative returns (LightGBM regression or `LGBMRanker`), long top / short bottom decile, sector-neutral (Part 9 factor neutralization). **Regime classifier** as a strategy switch (S8). **Volatility forecasting** with ML vs GARCH/HAR (volatility is far more predictable than returns; used for sizing and options). **Rare-event / crash / big-move prediction (item 50):** heavily imbalanced labels (e.g. next-5-day drawdown > 5%); features from Part 5 (VIX term structure, credit spreads, breadth); precision–recall curves, not accuracy; used as a **risk-reduction trigger**, never as a standalone short signal. |
| Live coding | Cross-sectional LightGBM ranker; HAR vs LightGBM volatility forecast; crash-warning classifier with PR-AUC. |
| Lab | Does the crash-warning model reduce drawdown when it scales exposure down (Part 8 backtest, crash library)? |
| Homework | Volatility forecast plugged into Part 8 volatility-target sizing; compare with EWMA volatility. |

#### S15 · Deployment, Drift & Retraining (item 8)

| Block | Content |
|---|---|
| Theory | **Train/serve parity:** live features from the Part 5 streaming implementations must equal batch features (test it). Inference latency budget; ONNX export for fast inference. **Drift monitoring:** Population Stability Index and KS tests on features and predictions, live precision vs CV precision, calibration drift. **Retraining triggers:** schedule, drift alarm, performance decay; champion–challenger with **shadow mode** (model runs and logs but does not trade). Rollback via the registry. |
| Live coding | Serving wrapper, PSI monitor (below), shadow-mode runner connected to Part 12 monitoring hooks. |
| Lab | Replay 2022 through a model trained on 2015–2021: when do the drift alarms fire? |
| Homework | Alert rule that pauses an ML strategy through the risk engine when drift exceeds a threshold. |

```python
def psi(expected: np.ndarray, actual: np.ndarray, bins=10) -> float:
    """Population Stability Index: < 0.1 stable, 0.1–0.25 watch, > 0.25 significant drift."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))
```

> Verified: two samples from the same distribution give PSI ≈ 0.004; a 0.5σ shift in the mean gives PSI ≈ 0.25 (the drift threshold).

#### S16 · ML Project Defense & M6 Release

| Block | Content |
|---|---|
| Theory | Presentation standard: problem framing, data and features, labels, model, CV and audits, importance, trading results vs baseline, DSR/PBO, deployment plan, failure modes. |
| Lab | Project defenses (Section 7, part A). |

**Clinic W4:** M6 code review: feature store, labeling, CV, audit, registry, drift monitor.

---

### Week 5 — Deep Learning

#### S17 · PyTorch for Financial Time Series (item 9)

| Block | Content |
|---|---|
| Theory | Tensors, autograd, `nn.Module`, optimizers (AdamW), loss functions (MSE, cross-entropy, quantile/pinball loss), regularization (dropout, weight decay, early stopping on a *time-ordered* validation period), gradient clipping. **Datasets without leakage:** sliding windows ending at *t* predicting a target after *t*; **normalize each window with its own statistics** (or with statistics from the training period only); gaps between train/validation/test equal to the look-back so windows do not overlap across splits. Reproducibility (seeds, deterministic ops). |
| Live coding | `WindowDataset`, training loop with early stopping (below). |
| Lab | MLP vs LightGBM on the same windows of features. |
| Homework | Quantile regression head (5%/50%/95%) for return distribution forecasts. |

```python
import torch
from torch import nn
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    """Windows X[t-L+1..t] -> y[t], where y is already the FUTURE target aligned to t.
    Each window is normalized with its own statistics only (no look-ahead)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        self.X, self.y, self.L = X.astype(np.float32), y.astype(np.float32), lookback
    def __len__(self):
        return len(self.X) - self.L + 1
    def __getitem__(self, i):
        w = self.X[i: i + self.L]
        w = (w - w.mean(0)) / (w.std(0) + 1e-8)
        return torch.from_numpy(w), torch.tensor(self.y[i + self.L - 1])

def train(model, train_dl, val_dl, epochs=50, lr=1e-3, patience=5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn, best, bad, best_state = nn.MSELoss(), np.inf, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = np.mean([loss_fn(model(xb), yb).item() for xb, yb in val_dl])
        if val < best - 1e-6:
            best, bad, best_state = val, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break                              # early stopping on the validation PERIOD
    model.load_state_dict(best_state)
    return best
```

#### S18 · Sequence Models & Uncertainty (items 9, 10)

| Block | Content |
|---|---|
| Theory | **LSTM/GRU** (gates, vanishing gradients), **TCN** (dilated causal convolutions: strictly no future), **Transformers for time series** (attention, positional encoding, patching as in PatchTST; causal masks), CNNs on price "images" (and why they mostly learn what indicators already give). **Uncertainty:** MC dropout, deep ensembles, **conformal prediction** (distribution-free intervals with a coverage guarantee under exchangeability; use a recent calibration window because markets are not exchangeable over long periods). Uncertainty feeds bet sizing: act less when the model is unsure. |
| Live coding | `LSTMRegressor` (below), a TCN block, split-conformal intervals. |
| Lab | LSTM vs TCN vs LightGBM on next-day volatility (a predictable target): accuracy and conformal coverage. |
| Homework | Small causal Transformer on the same task; compare parameters, training time and accuracy. |

```python
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=32, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):                          # x: (batch, time, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

def conformal_interval(cal_residuals: np.ndarray, pred: np.ndarray, alpha=0.1):
    """Split conformal prediction: ~(1 - alpha) coverage if calibration and test data are exchangeable."""
    n = len(cal_residuals)
    q = np.quantile(np.abs(cal_residuals), np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    return pred - q, pred + q
```

> Verified: on a simulated AR(1) series (φ = 0.6, so the best possible correlation with the next value is 0.6), the LSTM with window-level normalization and early stopping reaches a test correlation of 0.53; split-conformal 90% intervals give 91% empirical coverage.

#### S19 · Deep-Learning Strategies & Advanced Analysis (items 10, 11)

| Block | Content |
|---|---|
| Theory | Where DL can add value: many related series (train one model across the universe), high-frequency or order-book data, alternative data (text, from Part 11), multi-task targets (return + volatility), **autoencoders** for denoising and **anomaly detection** (reconstruction error as a risk signal, feeds Part 12 monitoring). Creating a DL strategy: predictions → calibrated probabilities or quantiles → bet sizing → Part 8 engine. **Analysis:** walk-forward retraining (expensive: plan compute), stability across seeds, explainability (integrated gradients, SHAP DeepExplainer), regime breakdown. |
| Live coding | Universe-wide LSTM/TCN for 5-day direction on 50 ETFs; autoencoder anomaly score on cross-asset returns. |
| Lab | DL strategy through the Part 8 backtester with monthly walk-forward retraining; seed-variance report. |
| Homework | Use the autoencoder anomaly score as a risk-off trigger; evaluate on the crash library. |

#### S20 · Deep-Learning Optimization & the Honest Comparison (item 12)

| Block | Content |
|---|---|
| Theory | Hyperparameter search (Optuna with pruning) inside walk-forward folds; architecture choices; regularization; **seed ensembling**; learning-rate schedules; knowledge **distillation** to a smaller, faster model; **ONNX export** and inference latency tests for live use. **Honest comparison:** DL vs LightGBM vs logistic baseline under identical data, validation and costs; count every DL trial in the research log for DSR. |
| Live coding | Optuna study for the S19 model; ONNX export and latency benchmark. |
| Lab | Final comparison table (accuracy, net Sharpe, DSR, compute cost, latency) across model families. |
| Homework | Written recommendation: when is DL worth it in this platform? |

**Clinic W5:** DL model vs LightGBM baseline, walk-forward, with seed variance and conformal intervals.

---

### Week 6 — Reinforcement Learning

#### S21 · RL Foundations (item 13)

| Block | Content |
|---|---|
| Theory | Markov decision processes (state, action, reward, transition, discount); Bellman equations; value iteration; **Q-learning** and **DQN** (experience replay, target networks); **policy gradients**, actor–critic, **PPO** (clipped objective, stable), **SAC** (continuous actions, entropy regularization). Exploration vs exploitation. Why RL is harder in trading: non-stationary environment, sparse and noisy rewards, a single history (easy to overfit), no true simulator of market impact. |
| Live coding | Tabular Q-learning on a toy mean-reverting price; DQN with `stable-baselines3` on a simple environment. |
| Lab | Show Q-learning overfitting a single simulated path vs generalizing across many simulated paths. |
| Homework | Read Sutton & Barto, chapters 3, 6 and 13 (summaries). |

#### S22 · Trading Environment Design (item 14)

| Block | Content |
|---|---|
| Theory | Gymnasium API (`reset`, `step`, spaces). **State:** window of features (from the feature store), current position, unrealized P&L, time features. **Actions:** discrete target positions (−1/0/+1) or continuous weights; keep them simple. **Reward:** net P&L after costs, differential Sharpe ratio, drawdown or risk penalties; beware **reward hacking** (the agent exploits a simulator flaw, e.g. free trades or look-ahead). **Timing:** the action at *t* is rewarded with the return **after** *t*. Episodes: random start points, fixed length; train/validation/test split by time; augment with bootstrapped or simulated paths. Validate the environment with `gymnasium.utils.env_checker`. |
| Live coding | `TradingEnv` (below); environment tests (no look-ahead, costs charged, correct timing). |
| Lab | Train PPO on a series with known structure; confirm it beats long-only and approaches a hindsight benchmark on unseen data; then run it on real ETF data and compare with a Part 7 rule. |
| Homework | Add a continuous-action version with volatility-scaled positions and a drawdown penalty. |

```python
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """Single asset. Action = target position in {-1, 0, +1}. Reward = position × NEXT return
    minus costs on position changes. Observation = last `lookback` returns + current position."""
    metadata = {"render_modes": []}

    def __init__(self, returns: np.ndarray, lookback=10, cost=0.0005):
        super().__init__()
        self.r, self.L, self.cost = returns.astype(np.float32), lookback, cost
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(lookback + 1,), dtype=np.float32)

    def _obs(self):
        return np.append(self.r[self.t - self.L: self.t] / (self.r.std() + 1e-8), self.pos).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.pos = self.L, 0.0
        return self._obs(), {}

    def step(self, action):
        new_pos = float(action) - 1.0                                   # 0,1,2 -> -1,0,+1
        reward = new_pos * self.r[self.t] - self.cost * abs(new_pos - self.pos)   # r[t] comes AFTER the decision
        self.pos, self.t = new_pos, self.t + 1
        terminated = self.t >= len(self.r)
        obs = self._obs() if not terminated else np.zeros(self.L + 1, dtype=np.float32)
        return obs, float(reward), terminated, False, {}

# from stable_baselines3 import PPO
# model = PPO("MlpPolicy", TradingEnv(train_returns), seed=0).learn(30_000)
```

> Verified: the environment passes `gymnasium.utils.env_checker.check_env`. On simulated returns with positive autocorrelation (φ = 0.3), PPO trained for 30k steps earns a cumulative log reward of 2.5 on the unseen test period, against 0.16 for long-only and 3.2 for a perfect-hindsight momentum benchmark. On real data, expect far less.

#### S23 · RL Strategies: Trading, Execution & Hedging (items 14, 15)

| Block | Content |
|---|---|
| Theory | Three uses, from most to least promising in practice: **optimal execution** (split a parent order to minimize implementation shortfall; benchmark against Almgren–Chriss and TWAP/VWAP from Part 12), **hedging** (deep hedging: learn a hedge policy for an options book under transaction costs, benchmark against Part 6 delta hedging), and **directional trading** (hardest). **Advanced analysis:** results across ≥ 5 seeds, across regimes and across assets; sensitivity to cost assumptions; reward-hacking checks; sim-to-real gap (paper trading the policy through the OMS). |
| Live coding | Execution environment (inventory, time left, spread, impact from the Part 8 model) with PPO vs TWAP vs Almgren–Chriss. |
| Lab | Hedging environment for a short ATM call: learned policy vs BSM delta hedging at 5 bps costs (reuse the Part 6 simulator). |
| Homework | Seed-and-regime robustness table for the learner's RL agent. |

#### S24 · RL Optimization & M7a Release (item 16)

| Block | Content |
|---|---|
| Theory | Hyperparameter tuning (learning rate, entropy coefficient, `n_steps`, network size) with Optuna, evaluated on a **validation period**, never the test period; curriculum learning (start with low costs/simple markets); ensembles of agents; policy distillation into a simple rule where possible (interpretability). Final honest comparison of RL vs Part 7 rules vs Part 10 ML on the same data and costs, through the Part 8 gate. |
| Live coding | Optuna + PPO tuning loop with validation-period evaluation; agent ensemble. |
| Lab | Final RL comparison table; decide whether any RL agent goes to paper trading. |
| Homework | Final assessment (Section 7, part B). |

**Clinic W6:** RL agent defense: environment tests, seeds/regimes robustness, baseline comparison, paper-trading plan.

---

## 5. Notebook Map (`notebooks/part10/`)

| Notebook | Session(s) | Promoted to |
|---|---|---|
| `01_framing_baseline.ipynb` | S1 | `research/ml/baseline.py` |
| `02_bars_cusum.ipynb` | S2 | `research/ml/bars.py` |
| `03_fracdiff_microstructure.ipynb` | S3 | `research/ml/features/` |
| `04_feature_store.ipynb` | S4 | `research/ml/feature_store.py` |
| `05_triple_barrier_weights.ipynb` | S5 | `research/ml/labels.py`, `weights.py` |
| `06_meta_labeling_betsize.ipynb` | S6 | `research/ml/labels.py`, `lib/sizing.py` |
| `07_supervised_models.ipynb` | S7 | `research/ml/models.py` |
| `08_regimes.ipynb` | S8 | `research/ml/regimes.py` |
| `09_purged_cv.ipynb` | S9 | `research/ml/cv.py` |
| `10_feature_importance.ipynb` | S10 | `research/ml/importance.py` |
| `11_leakage_audit.ipynb` | S11 | `research/ml/audit.py` |
| `12_optuna_ensembles_mlflow.ipynb` | S12 | `research/ml/tuning.py`, registry config |
| `13_ml_meta_strategy.ipynb` | S13 | `strategy/library/ml_meta.py` |
| `14_ranking_vol_rare_events.ipynb` | S14 | `strategy/library/ml_rank.py`, `research/rare_events/` |
| `15_serving_drift.ipynb` | S15 | `research/ml/serving.py`, `monitoring/drift.py` |
| `17_pytorch_windows.ipynb` | S17 | `research/dl/data.py`, `train.py` |
| `18_sequence_models_conformal.ipynb` | S18 | `research/dl/models.py`, `research/dl/uncertainty.py` |
| `19_dl_strategy_autoencoder.ipynb` | S19 | `strategy/library/dl_*.py`, `monitoring/anomaly.py` |
| `20_dl_optimization_onnx.ipynb` | S20 | `research/dl/tuning.py`, `research/dl/export.py` |
| `21_rl_foundations.ipynb` | S21 | – (teaching only) |
| `22_trading_env.ipynb` | S22 | `research/rl/envs/trading.py` |
| `23_rl_execution_hedging.ipynb` | S23 | `research/rl/envs/{execution,hedging}.py` |
| `24_rl_tuning.ipynb` | S24 | `research/rl/tuning.py` |

---

## 6. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Random train/test split on time series | Excellent CV, useless live | Time-ordered splits, purging, embargo |
| 2 | Scaler fitted on all data | Subtle leakage | Fit scalers inside each training fold; window-level normalization |
| 3 | Overlapping labels treated as independent | Overconfident CV, overfitting | Uniqueness weights, purged CV, sequential bootstrap |
| 4 | Fixed-horizon labels with fixed thresholds | Labels dominated by volatility regime | Triple barrier with volatility-scaled barriers |
| 5 | Accuracy as the metric | 55% accuracy, negative P&L | Log-loss, precision at threshold, **net trading metric** |
| 6 | Uncalibrated probabilities used for sizing | Oversized bets on weak signals | Calibration + reliability diagrams |
| 7 | Trusting MDI importance | Noise features look important | MDA under purged CV, SHAP, planted-noise test |
| 8 | Not logging trials | DSR meaningless | Every model/feature/hyperparameter trial in the research log |
| 9 | Batch vs live feature mismatch | Live predictions differ from backtest | Train/serve parity test with streaming indicators |
| 10 | No drift monitoring | Silent decay | PSI/KS monitors, shadow mode, retraining triggers |
| 11 | DL without a strong baseline | Complexity with no gain | LightGBM and logistic baselines on identical data |
| 12 | DL validation windows overlapping training windows | Leakage through shared bars | Gap of `lookback` bars between splits |
| 13 | RL reward uses the same-bar return | Agent "predicts" the past | Reward from the return *after* the action; env tests |
| 14 | RL evaluated on one seed / one path | Lucky agent | ≥ 5 seeds, multiple regimes, simulated paths |
| 15 | Crash model used as a short signal | Costly false alarms | Use rare-event models only to reduce risk |

---

## 7. Assessment — Milestones M6 & M7a

**Part A (end of week 36, M6): ML strategy project.**
1. Feature store with point-in-time guarantees and a documented feature list.
2. Event sampling, triple-barrier labels, sample weights, meta-labels on a Part 7/9 primary.
3. Calibrated model with purged CV/CPCV; MDA/SHAP importance; full leakage audit passing.
4. Strategy through the Part 8 backtester and **validation gate** (DSR from the full research log, PBO), compared with the primary alone and a logistic baseline.
5. Deployment plan: registry entry, drift monitors, retraining schedule, shadow-mode run of ≥ 1 week.

**Part B (end of week 38, M7a): DL and RL.**
1. One DL model vs the Part A model on the same data and costs, with walk-forward retraining, seed variance and conformal intervals.
2. One RL agent (trading, execution or hedging) with a tested environment, robustness across ≥ 5 seeds and ≥ 3 regimes, and comparison with a rule-based benchmark (TWAP/Almgren–Chriss, BSM delta hedge, or a Part 7 rule).
3. A one-page recommendation per family: adopt, keep researching, or reject, with evidence.

| Criterion | Points |
|---|---|
| Data, features & feature store (point-in-time, documented) | 10 |
| Labeling, weights & meta-labeling | 10 |
| Validation (purged CV/CPCV, calibration) & leakage audit | 15 |
| Interpretation (MDA/SHAP) & overfitting statistics (DSR/PBO) | 10 |
| ML strategy results vs baselines through the gate | 15 |
| Deployment: registry, drift monitoring, shadow mode | 10 |
| DL model: correctness, baseline comparison, uncertainty | 10 |
| RL agent: environment design, robustness, benchmark comparison | 10 |
| Code quality, reproducibility, honest conclusions | 10 |
| **Total** | **100** |

Pass mark: 70, **and** (mandatory) the leakage audit must pass for every submitted model, and every model must be compared with a simple baseline on identical data and costs.

---

## 8. Further Reading

| Type | Reference |
|---|---|
| Book | López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (bars, fractional differentiation, triple barrier, meta-labeling, purged CV, feature importance, bet sizing) |
| Book | López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge. (clustered importance, denoising) |
| Book | Jansen, S. (2020). *Machine Learning for Algorithmic Trading* (2nd ed.). Packt. |
| Book | Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. |
| Book | Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press. |
| Book | Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. |
| Paper | Gu, S., Kelly, B. & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *RFS*, 33(5). |
| Paper | Krauss, C., Do, X. A. & Huck, N. (2017). "Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500." *EJOR*, 259(2). |
| Paper | Fischer, T. & Krauss, C. (2018). "Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions." *EJOR*, 270(2). |
| Paper | Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*, 7(2). (HAR) |
| Paper | Lundberg, S. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*. (SHAP) |
| Paper | Angelopoulos, A. & Bates, S. (2023). "Conformal Prediction: A Gentle Introduction." *Foundations and Trends in ML*, 16(4). |
| Paper | Nie, Y. et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR*. (PatchTST) |
| Paper | Moody, J. & Saffell, M. (2001). "Learning to Trade via Direct Reinforcement." *IEEE Transactions on Neural Networks*, 12(4). (differential Sharpe) |
| Paper | Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347. |
| Paper | Buehler, H., Gonon, L., Teichmann, J. & Wood, B. (2019). "Deep Hedging." *Quantitative Finance*, 19(8). |
| Docs | scikit-learn, LightGBM, SHAP, Optuna, MLflow, PyTorch, ONNX Runtime, Gymnasium, Stable-Baselines3 |

---

## 9. Instructor Notes

- Open the Part with the S1 baseline and close each week by comparing against it. Learners should see how hard it is to beat simple models honestly.
- Enforce the research log from the first model; DSR in the assessment counts DL and RL trials too.
- Provide GPU credits for weeks 5–6, but keep all labs runnable on CPU with smaller models so nobody is blocked.
- For RL, always show a result on simulated data with known structure first (it proves the pipeline works) before real data (where results are usually weak).
- No ML/DL/RL strategy trades live in this Part. Shadow mode and paper only; live requires the Part 8 gate plus a completed shadow period.
