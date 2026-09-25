# Part 8 — Backtesting, Optimization, Risk & Portfolio: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 3, **Month 7 + first half of Month 8** (program weeks 25–30); Part 9 (Statistical Trading) follows in weeks 31–32 |
| Format | 24 sessions × 90 min (4 per week) + 6 lab clinics × 120 min + self-study (~6 hrs/week) |
| Total effort | ~36 hrs live + 12 hrs clinic + 36 hrs self-study ≈ **84 hours** |
| Brokers | Backtests on cached IB/Alpaca data; risk engine and sizing wired into the Part 4 OMS for paper trading |
| Platform milestones | **M4** (end of week 28): event-driven backtester, cost models, analysis, optimizer, validation gate · **M5a** (end of week 30): risk engine, position sizing, portfolio construction |
| Covers original items | "Advance with Python" 29–35 · "Trade Creation" 4 & 8 (optimization) · Platform roadmap 19, 22, 23, 42, 43, 51 |

**Why this Part matters most:** almost every strategy looks good in a naive backtest. This Part teaches how to build a backtest that tells the truth, how to optimize without fooling yourself, and how to size and combine strategies so that one bad month does not end the account. Every strategy from Part 7 onward must pass the **validation gate** built here before it can trade, even on paper.

---

## 1. Learning Objectives

By the end of Part 8 the learner will be able to:

1. **Build** an event-driven backtester that reuses the live `Strategy`, OMS and risk code, so backtest and live differ only in the data feed and broker adapter.
2. **Model** fills, commissions (IB, Alpaca), spreads, slippage and market impact realistically, and estimate strategy capacity.
3. **Detect and prevent** look-ahead, survivorship, selection and data-snooping biases.
4. **Analyze** performance at portfolio and trade level, and **test significance** (Sharpe standard error, bootstrap, Probabilistic Sharpe Ratio).
5. **Optimize** parameters (grid, random, Bayesian, genetic, multi-objective) and **validate** with walk-forward, purged k-fold and combinatorial purged CV.
6. **Quantify overfitting** with the Deflated Sharpe Ratio and the Probability of Backtest Overfitting, and run robustness tests.
7. **Design** a layered risk engine (pre-trade checks, exposure limits, VaR/CVaR, drawdown circuit breakers) as a Chain of Responsibility.
8. **Size** positions with fixed-risk, volatility-target and (fractional) Kelly methods, understanding the impact of estimation error.
9. **Construct and optimize** multi-strategy portfolios (inverse volatility, risk parity, HRP, mean-variance with shrinkage, Black–Litterman, CVaR) and compare them out of sample.

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| 2.2 Statistics (bootstrap, hypothesis testing, multiple testing) | W2–W4 |
| 2.4 Performance & risk metrics (Sharpe, drawdown, VaR/CVaR) | W2, W5 |
| 2.1 Math (linear algebra, convex optimization) | W6 |
| Part 4 SimAdapter, OMS, pre-trade checks, kill switch | W1, W5 |
| Part 5 indicators & EVT tail risk | W1, W5 |
| Part 6 portfolio Greeks & scenario grids | W5 |
| Part 7 strategy framework, `quick_eval`, strategy library | All weeks (the strategies being tested) |

---

## 3. Six-Week Overview

| Week | Theme | Sessions | Clinic Lab | Platform Output |
|---|---|---|---|---|
| **W1** | The backtest engine | S1 Philosophy & taxonomy · S2 Event-driven architecture · S3 Fill & cost models · S4 Framework comparison & parity | Port 3 Part 7 strategies to the engine; parity with `quick_eval` and vectorbt | `research/backtest/engine.py`, `brokers/sim/`, `research/backtest/costs.py` |
| **W2** | Truth-telling analysis | S5 Biases · S6 Performance analysis · S7 Statistical significance · S8 Capacity & backtest-vs-paper reconciliation | Bias hunt: 6 bugged backtests to fix | `analytics/performance.py`, `analytics/significance.py` |
| **W3** | Optimization & validation | S9 Search spaces & objectives · S10 Bayesian, genetic, multi-objective · S11 Walk-forward · S12 Purged k-fold & CPCV | Walk-forward optimization of a Part 7 strategy | `research/optimize/` |
| **W4** | Overfitting control & the gate | S13 Deflated Sharpe · S14 PBO (CSCV) · S15 Robustness tests · S16 Validation gate & M4 release | Full validation dossier for one strategy | `research/validate/gate.py`, **M4 release** |
| **W5** | Risk & sizing | S17 Risk framework & engine · S18 Market-risk measures & stress · S19 Sizing I (fixed-risk, volatility) · S20 Sizing II (Kelly) | Risk engine blocking bad orders on paper | `lib/risk.py`, `risk/engine.py`, `lib/sizing.py` |
| **W6** | Portfolio | S21 Combining strategies · S22 Mean-variance & shrinkage · S23 Risk parity, HRP, Black–Litterman, CVaR · S24 Integration & M5a | Multi-strategy portfolio, out-of-sample comparison | `portfolio/`, **M5a release** |

---

## 4. Session-by-Session Plan

> Each session: **15 min recap/theory → 45 min live coding → 20 min guided lab → 10 min wrap-up and homework.**
> Notebooks live in `notebooks/part08/`; promoted code lives in `quantforge/research/`, `quantforge/risk/`, `quantforge/lib/`, `quantforge/portfolio/`.

### Week 1 — The Backtest Engine

#### S1 · Backtesting Philosophy & Taxonomy (item 29)

| Block | Content |
|---|---|
| Theory | What a backtest can and cannot tell you: it is a **simulation of a hypothesis on one historical path**, not a forecast. Taxonomy: vectorized (fast, great for research and screening, easy to get wrong with path-dependent logic) vs **event-driven** (slower, mirrors live execution, handles orders, partial fills, stops, portfolio constraints); bar vs tick vs quote-driven simulation. The research workflow: hypothesis → spec → first look → backtest → validation gate → paper → small live. **Research log discipline:** record *every* configuration tried (needed later for the Deflated Sharpe Ratio). |
| Live coding | Research-log utility: every backtest run writes config hash, data version, code commit, metrics to DuckDB. |
| Lab | Re-run 3 Part 7 strategies through `quick_eval` and log them. |
| Homework | Read Bailey, Borwein, López de Prado & Zhu (2014), "Pseudo-Mathematics and Financial Charlatanism". |

#### S2 · Event-Driven Architecture (item 42)

| Block | Content |
|---|---|
| Theory | Components: data replay (from the Part 4 `DataHandler`), **SimClock**, priority event queue (fills before bars before timers at the same timestamp), the **unchanged** Part 7 `Strategy`, the Part 4 OMS, the `SimAdapter` (simulated exchange), and a portfolio/accounting ledger (cash, positions, mark-to-market, fees, multi-currency, futures multipliers, option expiry and assignment). Determinism: same inputs → identical results (seeded randomness). Performance: batching, avoiding pandas inside the inner loop. |
| Live coding | Event queue and portfolio ledger (below); the main loop wiring Strategy → Risk → OMS → SimAdapter. |
| Lab | Run the Part 7 TSMOM strategy on the engine; compare equity with `quick_eval` (should match within costs). |
| Homework | Add futures multipliers and currency conversion to the ledger; unit tests. |

```python
import heapq
from dataclasses import dataclass, field
import pandas as pd

@dataclass(order=True)
class Event:
    ts: pd.Timestamp
    priority: int                        # same timestamp: 0 fill, 1 bar, 2 timer
    kind: str = field(compare=False)
    data: object = field(compare=False)

class EventQueue:
    def __init__(self): self._h = []
    def push(self, e: Event): heapq.heappush(self._h, e)
    def pop(self) -> Event: return heapq.heappop(self._h)
    def __bool__(self): return bool(self._h)

class Portfolio:
    def __init__(self, cash: float):
        self.cash, self.pos, self.last = cash, {}, {}
    def on_fill(self, sym, qty, price, fee):
        self.cash -= qty * price + fee
        self.pos[sym] = self.pos.get(sym, 0) + qty
    def mark(self, sym, price):
        self.last[sym] = price
    @property
    def equity(self):
        return self.cash + sum(q * self.last.get(s, 0.0) for s, q in self.pos.items())
```

#### S3 · Fill & Cost Models (item 29)

| Block | Content |
|---|---|
| Theory | **Fill rules on bars:** market orders at next bar open (+ slippage); limit orders fill only if the price trades *through* the limit (touching is not enough: queue position is unknown); stops trigger at the stop price or the gap open, whichever is worse; bracket and OCO logic; partial fills by volume participation cap (e.g. ≤ 10% of bar volume). **Costs:** IB commissions (fixed vs tiered, plus exchange/regulatory fees), Alpaca (no commission on US stocks, but spread and regulatory fees), options per-contract fees, futures round-turn fees, short borrow fees, financing. **Slippage and impact:** half-spread, fixed bps, **square-root impact** `cost ≈ k·σ_daily·√(Q/ADV)`. Always check current broker fee schedules; numbers change. |
| Live coding | Fill model classes + commission and impact functions (below); plug into `SimAdapter`. |
| Lab | Same strategy at 0, 1×, 2× and 3× costs: how quickly does each Part 7 strategy's edge disappear? |
| Homework | Option fill model: fill at mid ± fraction of spread per leg; combo fills. |

```python
def ib_fixed_commission(shares, price):
    """IBKR Pro Fixed, US stocks: $0.005/share, min $1.00, max 1% of trade value (verify current schedule)."""
    return float(np.clip(0.005 * abs(shares), 1.0, 0.01 * abs(shares) * price))

def sqrt_impact_bps(order_shares, adv_shares, daily_vol, k=1.0):
    """Square-root market-impact model: cost ≈ k · σ_daily · sqrt(Q / ADV), in basis points."""
    return 1e4 * k * daily_vol * np.sqrt(abs(order_shares) / adv_shares)

sqrt_impact_bps(10_000, 1_000_000, 0.02)   # 1% of ADV at 2% daily vol -> 20 bps
```

#### S4 · Framework Comparison & Parity Testing

| Block | Content |
|---|---|
| Theory | vectorbt (fast vectorized research, parameter sweeps), backtrader (classic event-driven), NautilusTrader (high-performance event-driven, production-grade), zipline-reloaded (pipeline research). When to use which; why the course keeps its own engine for the live path (same code live). **Parity testing:** the same strategy on two engines must agree; differences must be explainable (fill timing, costs). |
| Live coding | SMA crossover in `quick_eval`, our engine and vectorbt; reconcile trade by trade. |
| Lab | Parity report for 3 strategies; list every difference and its cause. |
| Homework | Vectorized parameter sweep in vectorbt for fast screening; the engine for final runs. |

**Clinic W1:** port 3 Part 7 strategies (momentum, mean reversion, options) to the engine; parity report; cost sensitivity chart.

---

### Week 2 — Truth-Telling Analysis

#### S5 · Backtesting Biases & Pitfalls

| Block | Content |
|---|---|
| Theory | **Look-ahead** (same-bar fills, unshifted higher-TF data, future-adjusted prices, pivots, restated fundamentals). **Survivorship** (today's S&P 500 members backtested over 20 years), **point-in-time universes** and delisted securities. **Corporate actions** (splits, dividends, symbol changes). **Selection bias** and **data snooping** (the strategy you publish is the best of many tried). **Storytelling** (explaining a random result after the fact). Unrealistic assumptions: infinite liquidity, no borrow constraints, trading at the close price that produced the signal. |
| Live coding | Demonstrations: the same strategy on survivorship-biased vs point-in-time universe; same-bar vs next-bar fills. |
| Lab | "Bias hunt": 6 bugged notebooks, each hiding one bias; find, fix and quantify each. |
| Homework | Build a point-in-time S&P 500 membership table and a universe function `members(date)`. |

#### S6 · Performance Analysis (items 30, 51)

| Block | Content |
|---|---|
| Theory | **Portfolio level:** CAGR, volatility, Sharpe, Sortino, Calmar, max drawdown and duration, time under water, skew/kurtosis, tail ratio, rolling metrics, monthly return table, benchmark-relative (alpha, beta, information ratio, up/down capture). **Trade level:** win rate, payoff ratio, expectancy, profit factor, **MAE/MFE** (maximum adverse/favourable excursion: are stops and targets sensible?), holding time, consecutive losses, P&L by hour/weekday/regime. Tear sheets with `quantstats`; our own `performance.py` for the platform. |
| Live coding | `performance.py` tear sheet + trade analytics; MAE/MFE scatter. |
| Lab | Full tear sheets for the 3 strategies; use MAE/MFE to redesign one strategy's stop. |
| Homework | Regime breakdown table (Part 7 regime taxonomy) added to the tear sheet. |

#### S7 · Statistical Significance

| Block | Content |
|---|---|
| Theory | A Sharpe ratio is an estimate with error. **Standard error** of per-period Sharpe for i.i.d. returns ≈ √((1 + SR²/2)/T) (Lo 2002); autocorrelation makes it worse. **Bootstrap** (stationary block bootstrap for dependent returns). **Monte Carlo trade reshuffling** for the distribution of max drawdown. **Probabilistic Sharpe Ratio (PSR):** probability that the true Sharpe exceeds a benchmark, adjusting for track-record length, skewness and fat tails. Minimum track record length. |
| Live coding | PSR (below), block bootstrap confidence intervals, drawdown distribution. |
| Lab | For each strategy: Sharpe 95% CI, PSR vs 0, drawdown 95th percentile. Which results are distinguishable from luck? |
| Homework | Minimum track record length: how many months of paper trading are needed to confirm each strategy? |

```python
from scipy.stats import norm, skew, kurtosis

def sharpe_se(sr, T):
    """Standard error of a per-period Sharpe ratio, i.i.d. normal returns (Lo 2002)."""
    return np.sqrt((1 + 0.5 * sr**2) / T)

def psr(returns, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012), per-period units.
    Adjusts for track-record length, skewness and kurtosis."""
    r = np.asarray(returns); T = r.size
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    return norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1)
                    / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
```

#### S8 · Capacity & Backtest-vs-Paper Reconciliation

| Block | Content |
|---|---|
| Theory | **Capacity:** the capital at which impact costs eat the edge; estimate from ADV participation and the S3 impact model. **Implementation shortfall** (decision price vs fill price). Reconciling backtest with the Part 7 paper trades: fill timing, slippage, missed fills, partial fills; tuning the fill model from real paper/live fills. |
| Live coding | Capacity curve (net Sharpe vs capital); shortfall report from the paper-trade log. |
| Lab | Capacity for a small-cap mean-reversion strategy vs an ETF momentum strategy. |
| Homework | Calibrate slippage parameters from the learner's own paper fills. |

**Clinic W2:** bias hunt competition + significance report for the 3 strategies.

---

### Week 3 — Optimization & Validation

#### S9 · Search Spaces, Objectives & Stability (item 31)

| Block | Content |
|---|---|
| Theory | Parameter spaces from the Part 5 registry (`params` ranges). Objective choice: Sharpe, Calmar, CAGR/MaxDD, PSR, custom utility with turnover penalty; never optimize raw return. Grid vs random search (random is better in high dimensions). **Parameter stability:** prefer broad plateaus over sharp peaks; heatmaps and neighbourhood-averaged objectives. Fewer parameters is better (every parameter is a chance to overfit). |
| Live coding | Grid and random search with plateau scoring (average objective over neighbours). |
| Lab | 2-parameter heatmaps for 3 strategies; pick parameters by plateau, not peak. |
| Homework | Count the effective number of trials in your search (needed for DSR in S13). |

#### S10 · Bayesian, Genetic & Multi-Objective Optimization (items 31, "Trade Creation" 4/8)

| Block | Content |
|---|---|
| Theory | **Bayesian optimization** with Optuna (TPE sampler), pruning bad trials early, conditional search spaces. **Genetic algorithms** for discrete/rule-structure search (and why they overfit easily). **Multi-objective** (Sharpe vs drawdown vs turnover) and the Pareto front. Budgets and reproducibility (seeds, logged trials). |
| Live coding | Optuna study with a walk-forward objective (from S11) and all trials logged to the research log. |
| Lab | Bayesian vs random search: best found vs trials used; overfitting grows with trials. |
| Homework | Multi-objective study; choose a point on the Pareto front and justify it. |

```python
import optuna

def objective(trial):
    params = {"lookback": trial.suggest_int("lookback", 60, 300, step=20),
              "vol_n": trial.suggest_int("vol_n", 20, 120, step=10)}
    return walk_forward_score(strategy="tsmom", params=params)   # OOS metric only (S11)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200)
research_log.record_trials(study.trials_dataframe())          # every trial counts for DSR
```

#### S11 · Walk-Forward Analysis (item 31)

| Block | Content |
|---|---|
| Theory | In-sample (IS) optimization, out-of-sample (OOS) test, roll forward. **Rolling vs anchored** windows; window-length trade-offs; re-optimization cadence; **walk-forward efficiency** (OOS performance / IS performance). The stitched OOS equity curve is the only honest result. Parameter drift over time as a diagnostic. |
| Live coding | Walk-forward splitter (below) and optimizer loop; stitched OOS equity. |
| Lab | Walk-forward TSMOM and RSI(2): IS vs OOS Sharpe per window; walk-forward efficiency. |
| Homework | Compare anchored vs rolling windows for one strategy. |

```python
def walk_forward(n, train, test, anchored=False):
    """Yields (train_idx, test_idx). Rolling by default; anchored keeps the start fixed."""
    start = 0
    while start + train + test <= n:
        tr0 = 0 if anchored else start
        yield np.arange(tr0, start + train), np.arange(start + train, start + train + test)
        start += test
```

#### S12 · Purged K-Fold & Combinatorial Purged Cross-Validation

| Block | Content |
|---|---|
| Theory | Why ordinary k-fold leaks in finance: labels span several bars (e.g. 5-day forward returns), so training samples near the test fold share information. **Purging** removes training samples whose label window overlaps the test fold; the **embargo** removes a buffer after it (serial correlation). **CPCV** (López de Prado 2018): split into N groups, test on every combination of k groups, and get *many* OOS paths instead of one, giving a distribution of OOS Sharpe ratios. (Used heavily again in Part 10 for ML.) |
| Live coding | Purged k-fold with embargo (below); CPCV path construction. |
| Lab | Compare OOS Sharpe estimates from naive k-fold vs purged k-fold vs CPCV for an overfit strategy. |
| Homework | CPCV distribution of OOS Sharpe for your strategy; report its 5th percentile. |

```python
def purged_kfold(n, k=5, label_horizon=5, embargo=0.01):
    """Time-series k-fold: drop training samples whose labels overlap the test fold (purge)
    and a buffer right after it (embargo)."""
    emb = int(np.ceil(embargo * n))
    for test in np.array_split(np.arange(n), k):
        lo, hi = test[0], test[-1]
        train = np.array([i for i in range(n) if (i + label_horizon < lo) or (i > hi + emb)])
        yield train, test
```

**Clinic W3:** walk-forward optimization of one Part 7 strategy with Optuna; stitched OOS curve; parameter-drift chart.

---

### Week 4 — Overfitting Control & the Validation Gate

#### S13 · The Deflated Sharpe Ratio

| Block | Content |
|---|---|
| Theory | If you try N unskilled strategies, the best one's Sharpe is expected to be well above zero. The **expected maximum Sharpe** under the null grows with N and with the variance of trial Sharpes. **Deflated Sharpe Ratio (DSR)** = PSR measured against that expected maximum instead of zero. Needs the research log (number of trials, variance of their Sharpes). Related: Harvey & Liu "haircut" Sharpe, multiple-testing adjustments. |
| Live coding | DSR (below) computed from the research log. |
| Lab | DSR for every strategy in the learner's log. How many survive at 95%? |
| Homework | Explain in one page why a strategy with Sharpe 1.5 found after 1,000 trials can be less convincing than Sharpe 1.0 found after 3. |

```python
def deflated_sharpe(returns, trial_srs):
    """DSR: PSR against the Sharpe expected from the BEST of N unskilled trials (per-period units)."""
    n, v, g = len(trial_srs), np.var(trial_srs, ddof=1), 0.5772156649   # Euler–Mascheroni
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
    return psr(returns, sr0), sr0
```

#### S14 · Probability of Backtest Overfitting (CSCV)

| Block | Content |
|---|---|
| Theory | **CSCV** (Bailey, Borwein, López de Prado & Zhu 2017): split the T × N matrix of returns (N configurations) into S blocks; for every half/half split, pick the best configuration in-sample and find its rank out of sample. **PBO** = fraction of splits where the IS winner ranks below the OOS median. PBO near 0.5 means the optimization chose by noise. Also: performance degradation (IS vs OOS regression) and probability of loss. |
| Live coding | `pbo_cscv` (below). |
| Lab | PBO for the S9/S10 searches. Compare with a synthetic example: pure noise vs configurations with real, graded skill. |
| Homework | Add PBO and DSR to the validation dossier template. |

```python
import itertools

def pbo_cscv(perf: np.ndarray, S=16):
    """Probability of Backtest Overfitting. perf: T x N per-period returns of N configurations."""
    T, N = perf.shape
    blocks = np.array_split(np.arange(T), S)
    sr = lambda x: x.mean(0) / x.std(0, ddof=1)
    logits = []
    for is_blocks in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in is_blocks])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in is_blocks])
        best = np.argmax(sr(perf[is_idx]))
        w = (sr(perf[oos_idx]).argsort().argsort()[best] + 1) / (N + 1)   # relative OOS rank
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits
```

> Verified on synthetic data (50 configurations × 1,000 periods, S = 10, 20 random seeds): pure noise gives an average PBO of ≈ 0.48 (the 0.5 of pure chance), while configurations with graded real skill average ≈ 0.07. PBO from a single dataset is itself noisy (noise ranged 0.05–0.76 across seeds), so read it together with DSR and walk-forward results.

#### S15 · Robustness Tests

| Block | Content |
|---|---|
| Theory | A robust strategy survives small changes. **Parameter perturbation** (±10–20%), **execution delay** (enter one bar later), **cost stress** (2× costs), **data perturbation** (bootstrap paths, added noise, synthetic data from a fitted model), **different periods and universes** (other markets, other decades), **regime subsets**, removing the best 5 trades. Report as a robustness scorecard. |
| Live coding | Robustness runner that executes all tests from a strategy config. |
| Lab | Scorecard for 3 strategies; identify which break and why. |
| Homework | Add a Monte Carlo "synthetic history" test using a block bootstrap of the underlying returns. |

#### S16 · The Validation Gate & M4 Release

| Block | Content |
|---|---|
| Theory | The **gate** every strategy must pass before paper trading (and later live). Default criteria (configurable): walk-forward OOS Sharpe > 0.5 net of costs; DSR ≥ 0.95; PBO ≤ 0.2; passes ≥ 80% of robustness tests; max drawdown within risk budget; capacity ≥ planned capital; parity with engine and paper fills. A **validation dossier** is generated automatically. |
| Live coding | `gate.py` producing a pass/fail dossier (Markdown + charts). |
| Lab | Run the gate on all Part 7 strategies. |
| Homework | Month-7 project (Section 7, part A). |

**Clinic W4:** validation dossier review. Each learner defends one strategy; peers try to break it.

---

### Week 5 — Risk Management & Position Sizing

#### S17 · Risk Framework & the Risk Engine (items 23, 32)

| Block | Content |
|---|---|
| Theory | Risk taxonomy: market, liquidity, model, operational, counterparty/broker, concentration. **Limits hierarchy:** account → strategy → instrument; hard vs soft limits. **Pre-trade checks** as a **Chain of Responsibility** (extends Part 4 S15): notional, gross/net exposure, sector and single-name concentration, Greeks limits (Part 6), order rate, price sanity, liquidity (≤ x% of ADV). **In-trade controls:** drawdown circuit breakers (strategy- and account-level), daily loss limit linked to the kill switch. Risk engine sits between strategy intents and the OMS, in backtest and live alike. |
| Live coding | `RiskEngine` with pluggable rules (below); integration test through the backtester and paper OMS. |
| Lab | Try to break it on paper: oversized order, concentration breach, daily loss breach; every rejection must be logged with a reason. |
| Homework | Add `MaxSectorExposure` and `MaxVegaExposure` rules with tests. |

```python
from dataclasses import dataclass

@dataclass
class RiskDecision:
    approved: bool
    reason: str = ""

class RiskRule:
    def check(self, order, ctx) -> RiskDecision: raise NotImplementedError

class MaxNotional(RiskRule):
    def __init__(self, limit): self.limit = limit
    def check(self, order, ctx):
        n = abs(order["qty"] * order["price"])
        return RiskDecision(n <= self.limit, f"notional {n:,.0f} > {self.limit:,.0f}")

class MaxGrossExposure(RiskRule):
    def __init__(self, max_leverage): self.max_leverage = max_leverage
    def check(self, order, ctx):
        gross = ctx["gross"] + abs(order["qty"] * order["price"])
        return RiskDecision(gross <= self.max_leverage * ctx["equity"],
                            f"gross {gross / ctx['equity']:.2f}x > {self.max_leverage}x")

class DailyLossLimit(RiskRule):
    def __init__(self, max_loss_frac): self.max_loss_frac = max_loss_frac
    def check(self, order, ctx):
        dd = ctx["day_pnl"] / ctx["start_equity"]
        return RiskDecision(dd > -self.max_loss_frac, f"daily loss {dd:.2%} breaches {-self.max_loss_frac:.2%}")

class RiskEngine:
    """Chain of Responsibility: every rule must approve; the first rejection stops the chain."""
    def __init__(self, rules): self.rules = rules
    def check(self, order, ctx) -> RiskDecision:
        for rule in self.rules:
            d = rule.check(order, ctx)
            if not d.approved:
                return RiskDecision(False, f"{type(rule).__name__}: {d.reason}")
        return RiskDecision(True)
```

#### S18 · Market-Risk Measures & Stress Testing

| Block | Content |
|---|---|
| Theory | VaR and CVaR (historical, parametric, Monte Carlo, EVT from Part 5) at portfolio level; component and marginal VaR (which position contributes most). Correlation risk (correlations rise in crises), liquidity-adjusted VaR. **Stress tests:** historical crash library (Part 5 S15), hypothetical shocks, Greeks-based and full-revaluation scenarios for options books (Part 6 S8). Risk dashboards and daily risk report. |
| Live coding | Historical VaR/CVaR (below), component VaR, stress runner over the crash library. |
| Lab | Daily risk report for a sample multi-strategy book: VaR, CVaR, top contributors, worst stress scenario. |
| Homework | VaR backtest (Kupiec) for the learner's portfolio. |

```python
def hist_var_cvar(returns, alpha=0.99):
    """Historical VaR and CVaR (Expected Shortfall), as positive loss fractions."""
    losses = -np.asarray(returns)
    var = np.quantile(losses, alpha)
    return var, losses[losses >= var].mean()
```

#### S19 · Position Sizing I: Fixed Risk & Volatility (items 22, 33)

| Block | Content |
|---|---|
| Theory | Sizing matters as much as signals. **Fixed fractional** (x% of equity per position), **fixed risk per trade** (lose ≤ x% of equity if the stop is hit; stop from ATR), **volatility targeting** (scale exposure to a target annual volatility), **Turtle units** (ATR-normalized). Futures and options: risk per contract via multipliers and max loss. **Risk of ruin** and why small risk per trade (0.25–1%) is standard. Pyramiding and scaling rules. |
| Live coding | Sizing functions (below) wired into the risk engine as a sizing stage. |
| Lab | Same strategy with 4 sizing methods: CAGR, max drawdown, Sharpe, worst month. |
| Homework | Monte Carlo risk-of-ruin calculator from a trade-return distribution. |

```python
def risk_per_trade_size(equity, risk_frac, entry, stop, multiplier=1.0):
    """Units such that hitting the stop loses `risk_frac` of equity."""
    return int(equity * risk_frac / (abs(entry - stop) * multiplier))

def vol_target_weight(returns, target_vol=0.10, lookback=60, periods=252, cap=2.0):
    """Portfolio weight that scales recent volatility to the target (capped leverage)."""
    vol = pd.Series(returns).rolling(lookback).std().iloc[-1] * np.sqrt(periods)
    return float(min(target_vol / vol, cap))

risk_per_trade_size(100_000, 0.01, entry=100, stop=95)    # -> 200 shares
```

#### S20 · Position Sizing II: Kelly & Estimation Error (items 22, 33)

| Block | Content |
|---|---|
| Theory | **Kelly criterion:** growth-optimal bet size. Discrete `f* = p − (1 − p)/b`; continuous `f* = (μ − r)/σ²`; multi-asset `f* = Σ⁻¹(μ − r)`. Why **full Kelly is dangerous in practice:** μ is estimated with huge error, returns are fat-tailed, drawdowns at full Kelly are brutal (50% drawdown probability is high). **Fractional Kelly** (¼–½) and Kelly as an upper bound, not a target. Optimal f (Vince). Bet sizing from ML probabilities (preview of Part 10). |
| Live coding | Kelly functions (below); simulation of growth and drawdown at ¼, ½, 1 and 2× Kelly, with μ estimated from a short sample. |
| Lab | Show that with estimation error, 2× Kelly loses money over time while ½ Kelly grows steadily. |
| Homework | Final Kelly cap rule for the platform, justified from the simulation. |

```python
def kelly_discrete(p_win, win_loss_ratio):
    return p_win - (1 - p_win) / win_loss_ratio

def kelly_continuous(mu, sigma, r=0.0):
    """Growth-optimal leverage for one asset: annual excess drift / variance."""
    return (mu - r) / sigma**2

kelly_continuous(0.08, 0.16, r=0.03)   # ≈ 1.95x leverage at full Kelly; use a fraction of this
```

**Clinic W5:** risk engine on paper. Run 2 strategies on paper with the full risk chain and sizing; show at least 3 correctly rejected orders and a daily risk report.

---

### Week 6 — Portfolio Construction & Optimization

#### S21 · Combining Strategies (item 34)

| Block | Content |
|---|---|
| Theory | Diversification comes from **low correlation of strategy returns**, not from the number of strategies. Correlation and clustering of strategy returns (and how they converge in crises). Allocation methods, from simple to complex: equal weight, **inverse volatility**, equal risk contribution; why simple often beats optimized out of sample (DeMiguel et al. 2009: 1/N). Rebalancing frequency, drift bands and turnover. Capital allocation across strategies vs across assets within a strategy. |
| Live coding | Strategy-return matrix from the research log; correlation clustering; equal-weight and inverse-volatility portfolios. |
| Lab | Portfolio of 5 validated strategies: equal weight vs inverse volatility; diversification ratio. |
| Homework | Drift-band rebalancing vs calendar rebalancing: turnover and tracking difference. |

#### S22 · Mean-Variance & Covariance Estimation (item 35)

| Block | Content |
|---|---|
| Theory | Markowitz mean-variance, minimum variance, maximum Sharpe; the efficient frontier. Why naive MVO fails: **error maximization** (it bets most on the assets with the biggest estimation errors). Covariance estimation: sample covariance is noisy when assets ≈ observations; **Ledoit–Wolf shrinkage**, exponential weighting, factor models. Constraints (long-only, weight caps, turnover, sector) with `cvxpy`. |
| Live coding | Minimum variance (below) with Ledoit–Wolf covariance; constrained MVO in `cvxpy`. |
| Lab | Sample vs shrunk covariance: out-of-sample volatility of the minimum-variance portfolio. |
| Homework | Turnover-penalized MVO; compare weight stability month to month. |

```python
from sklearn.covariance import LedoitWolf

def min_variance(cov):
    inv = np.linalg.inv(cov)
    w = inv @ np.ones(len(cov))
    return w / w.sum()

cov = LedoitWolf().fit(returns).covariance_      # shrunk covariance; returns: T x N array
w_mv = min_variance(cov)
```

#### S23 · Risk Parity, HRP, Black–Litterman & CVaR (item 35)

| Block | Content |
|---|---|
| Theory | **Equal risk contribution (risk parity):** every asset/strategy contributes the same risk; convex formulation (Spinu 2013). **Hierarchical Risk Parity** (López de Prado 2016): cluster by correlation, order, recursively split capital by inverse cluster variance; no matrix inversion, robust. **Black–Litterman:** start from equilibrium returns, blend in views with confidence. **CVaR optimization** (Rockafellar–Uryasev, a linear program) for fat-tailed returns. Library comparison: `riskfolio-lib`, PyPortfolioOpt. **Evaluate out of sample only.** |
| Live coding | Risk parity and HRP (below); Black–Litterman with one view; CVaR LP in `cvxpy`. |
| Lab | Walk-forward comparison of 1/N, inverse volatility, min variance, risk parity, HRP, CVaR on the 5-strategy portfolio: OOS Sharpe, drawdown, turnover. |
| Homework | Write a one-page recommendation for the platform default allocator. |

```python
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def risk_parity(cov):
    """Equal risk contribution (long-only), convex log-barrier form (Spinu 2013):
    minimize 0.5·w'Σw − (1/n)·Σ log w, then normalize."""
    n = len(cov)
    c = cov / np.mean(np.diag(cov))                       # rescale for numerical stability
    res = minimize(lambda y: 0.5 * y @ c @ y - np.log(y).sum() / n, np.full(n, 1.0),
                   jac=lambda y: c @ y - 1 / (n * y), bounds=[(1e-9, None)] * n, method="L-BFGS-B")
    return res.x / res.x.sum()

def hrp(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (López de Prado 2016)."""
    cov, corr = returns.cov(), returns.corr()
    dist = np.sqrt(0.5 * (1 - corr)).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    order = list(corr.index[leaves_list(linkage(squareform(dist, checks=False), "single"))])
    w = pd.Series(1.0, index=order)

    def cluster_var(items):
        c = cov.loc[items, items].to_numpy()
        ivp = 1 / np.diag(c); ivp /= ivp.sum()
        return ivp @ c @ ivp

    clusters = [order]
    while clusters:                                       # recursive bisection
        clusters = [c[i:j] for c in clusters
                    for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for a, b in zip(clusters[::2], clusters[1::2]):
            va, vb = cluster_var(a), cluster_var(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
    return w.reindex(returns.columns)
```

> Verified on 5 synthetic assets: the risk-parity weights give equal risk contributions (20% each); HRP weights sum to 1 and put the most weight on the lowest-variance asset.

#### S24 · Integration & M5a Release

| Block | Content |
|---|---|
| Theory | The full live loop: strategies → intents → **allocator** (portfolio weights) → **sizer** → **risk engine** → kill switch → OMS → broker; the same chain inside the backtester. Configuration of risk budgets per strategy. Monitoring hooks for Part 12. |
| Live coding | End-to-end run: 3 validated strategies, HRP allocation, volatility-target sizing, risk chain, on the backtester and on paper with identical config. |
| Lab | Compare the backtest of the last 2 weeks with the paper results over the same 2 weeks. |
| Homework | Final assessment (Section 7, part B). |

**Clinic W6:** portfolio defense. Each learner presents their allocator choice with out-of-sample evidence.

---

## 5. Notebook Map (`notebooks/part08/`)

| Notebook | Session(s) | Promoted to |
|---|---|---|
| `01_research_log.ipynb` | S1 | `research/log.py` |
| `02_event_engine.ipynb` | S2 | `research/backtest/engine.py`, `research/backtest/ledger.py` |
| `03_fills_costs.ipynb` | S3 | `research/backtest/costs.py`, `brokers/sim/fills.py` |
| `04_parity.ipynb` | S4 | `tests/backtest/test_parity.py` |
| `05_bias_hunt.ipynb` | S5 | `data/universe.py` (point-in-time membership) |
| `06_tearsheet.ipynb` | S6 | `analytics/performance.py` |
| `07_significance.ipynb` | S7 | `analytics/significance.py` |
| `08_capacity_shortfall.ipynb` | S8 | `analytics/capacity.py` |
| `09_search_stability.ipynb` | S9 | `research/optimize/search.py` |
| `10_optuna_multiobjective.ipynb` | S10 | `research/optimize/bayes.py` |
| `11_walk_forward.ipynb` | S11 | `research/optimize/walk_forward.py` |
| `12_purged_cv_cpcv.ipynb` | S12 | `research/validate/cv.py` |
| `13_deflated_sharpe.ipynb` | S13 | `analytics/significance.py` |
| `14_pbo_cscv.ipynb` | S14 | `research/validate/pbo.py` |
| `15_robustness.ipynb` | S15 | `research/validate/robustness.py` |
| `16_validation_gate.ipynb` | S16 | `research/validate/gate.py` |
| `17_risk_engine.ipynb` | S17 | `lib/risk.py`, `risk/engine.py` |
| `18_var_stress.ipynb` | S18 | `risk/measures.py`, `risk/stress.py` |
| `19_sizing_fixed_vol.ipynb` | S19 | `lib/sizing.py` |
| `20_kelly.ipynb` | S20 | `lib/sizing.py` |
| `21_combining_strategies.ipynb` | S21 | `portfolio/allocate.py` |
| `22_mvo_shrinkage.ipynb` | S22 | `portfolio/optimize.py` |
| `23_rp_hrp_bl_cvar.ipynb` | S23 | `portfolio/optimize.py` |
| `24_integration.ipynb` | S24 | `apps/run_portfolio.py` |

---

## 6. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Different code for backtest and live | Live results never match | One `Strategy`/OMS/risk path; parity tests |
| 2 | Limit orders filled on touch | Mean-reversion looks great, fails live | Fill only when price trades through the limit |
| 3 | Costs ignored or fixed at zero | Edge disappears live | Cost model mandatory; 2× cost robustness test |
| 4 | Survivorship-biased universe | Inflated long-only returns | Point-in-time membership; include delisted names |
| 5 | Optimizing on the full history | Great backtest, poor live | Walk-forward / CPCV; OOS results only |
| 6 | Not logging failed trials | DSR impossible, false confidence | Research log records every run |
| 7 | Picking the peak parameter | Fragile performance | Plateau scoring; perturbation tests |
| 8 | Ordinary k-fold on overlapping labels | Leakage, inflated OOS | Purging + embargo |
| 9 | Reporting Sharpe without uncertainty | Overconfidence | SE, bootstrap CI, PSR, DSR |
| 10 | Full Kelly sizing | Huge drawdowns, ruin | Fractional Kelly as a cap; volatility targeting |
| 11 | Sample covariance in MVO | Extreme, unstable weights | Shrinkage, HRP, constraints |
| 12 | Allocator chosen in-sample | Disappointing OOS portfolio | Walk-forward comparison of allocators |
| 13 | Risk checks only in live code | Backtest ignores limits, live differs | Risk engine inside the backtester too |
| 14 | Correlations assumed stable | Diversification vanishes in crises | Stress correlations; crash-library tests |

---

## 7. Assessment — Milestones M4 & M5a

**Part A (end of week 28, M4): Validation dossier.** For one original strategy (from the Part 7 library or the learner's own idea):
1. Event-driven backtest with realistic costs; parity with vectorbt; cost sensitivity.
2. Walk-forward optimization (Optuna), stitched OOS curve, parameter drift.
3. CPCV OOS Sharpe distribution, PSR, DSR (from the complete research log), PBO.
4. Robustness scorecard and capacity estimate.
5. Gate verdict (pass or fail); a well-argued **fail** scores as well as a pass.

**Part B (end of week 30, M5a): Risk-managed multi-strategy portfolio.**
1. Risk engine (≥ 8 rules) inside backtester and paper OMS; kill-switch linkage.
2. Sizing layer (volatility target + fixed risk + Kelly cap).
3. Walk-forward comparison of ≥ 5 allocators on ≥ 4 strategies; chosen allocator justified.
4. Daily risk report (VaR, CVaR, stress, exposures) generated automatically.
5. Two weeks of paper trading with backtest-vs-paper reconciliation.

| Criterion | Points |
|---|---|
| Backtest engine realism (fills, costs, parity) | 15 |
| Optimization & walk-forward correctness | 15 |
| Overfitting statistics (DSR, PBO, CPCV) computed and interpreted correctly | 15 |
| Robustness & capacity analysis | 10 |
| Risk engine & sizing (design, tests, paper evidence) | 15 |
| Portfolio construction, OOS comparison, justification | 15 |
| Backtest-vs-paper reconciliation | 5 |
| Code quality, reproducibility (seeds, logs, configs), documentation | 10 |
| **Total** | **100** |

Pass mark: 70, **and** (mandatory) the dossier's DSR must be computed from the learner's full research log, and the risk engine must be active in both backtest and paper runs.

---

## 8. Further Reading

| Type | Reference |
|---|---|
| Book | López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (purged CV, CPCV, backtest overfitting, HRP) |
| Book | Chan, E. (2021). *Quantitative Trading* (2nd ed.). Wiley. |
| Book | Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley. (walk-forward) |
| Book | Grinold, R. & Kahn, R. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill. |
| Book | Vince, R. (2007). *The Handbook of Portfolio Mathematics*. Wiley. (Kelly, optimal f) |
| Paper | Lo, A. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*, 58(4). |
| Paper | Bailey, D. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2). (PSR) |
| Paper | Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5). |
| Paper | Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 20(4). |
| Paper | Harvey, C. & Liu, Y. (2015). "Backtesting." *Journal of Portfolio Management*, 42(1). |
| Paper | Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3. |
| Paper | Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix." *Journal of Portfolio Management*, 30(4). |
| Paper | López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample." *Journal of Portfolio Management*, 42(4). (HRP) |
| Paper | DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal Versus Naive Diversification." *RFS*, 22(5). |
| Paper | Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3). |
| Paper | Black, F. & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5). |
| Paper | Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights." SSRN 2297383. |
| Docs | vectorbt, backtrader, NautilusTrader, Optuna, `cvxpy`, `riskfolio-lib`, `quantstats`; IB and Alpaca fee schedules |

---

## 9. Instructor Notes

- The single most important habit: **log every trial**. Enforce it from S1; the DSR in the assessment is computed from the log, so hiding failed runs is impossible.
- Reward honest failure. Most learner strategies should fail the gate; that is the gate working.
- Keep fee schedules as data files with a "last verified" date; brokers change them.
- Risk-engine labs run on paper accounts only. Live trading is unlocked per learner after M5a review, starting with small size and the kill switch armed.
- Budget compute: CPCV and PBO over large searches are expensive; provide pre-computed return matrices for labs and let learners run full versions as homework.
