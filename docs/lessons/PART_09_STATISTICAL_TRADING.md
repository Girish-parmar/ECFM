# Part 9 — Advanced Statistical Trading: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 3, **Month 8, second half** (program weeks 31–32), right after Part 8 |
| Format | 8 sessions × **120 min** (4 per week) + 2 lab clinics × 120 min + self-study (~7 hrs/week) |
| Total effort | ~16 hrs live + 4 hrs clinic + 14 hrs self-study ≈ **34 hours** |
| Instruments | US equities & sector ETFs (IB, Alpaca; shorting needs borrow availability), index futures (IB) |
| Platform milestone | **M5b**: `research/statarb/` (OU, cointegration, pair screening, Kalman hedge, PCA residuals, factor models) and a market-neutral pairs portfolio running through the Part 8 risk engine |
| Covers original items | "Trade Creation" 1–4 (advanced statistics, statistical strategy creation, analysis, optimization) · "Advance with Python" 23 (statistical strategies) · Platform roadmap 47 (statistical arbitrage) |

**Where this fits:** Part 7 introduced pairs trading and seasonality as one-session previews. Part 9 goes deep: the stochastic models behind mean reversion, rigorous pair selection under multiple testing, dynamic hedge ratios, multi-asset residual stat-arb, factor models, and detecting when a relationship breaks. Every strategy passes through the Part 8 backtester, validation gate and risk engine.

---

## 1. Learning Objectives

By the end of Part 9 the learner will be able to:

1. **Model** prices and spreads with stochastic processes (GBM, Ornstein–Uhlenbeck, jump diffusion) and estimate their parameters, including the mean-reversion half-life.
2. **Select** trading pairs and baskets with distance, correlation and cointegration methods (Engle–Granger, Johansen) while controlling false discoveries.
3. **Build** a complete pairs strategy: spread, z-score, entry/exit/stop/time-stop, dollar- or beta-neutral sizing, borrow costs.
4. **Estimate** time-varying hedge ratios with a Kalman filter and trade the filter's forecast error.
5. **Implement** multi-asset residual stat-arb (PCA/ETF factors, OU s-scores) and cross-sectional factor strategies (Fama–MacBeth).
6. **Detect** structural breaks and relationship decay, and use Bayesian inference to express uncertainty in parameters.
7. **Optimize** statistical strategies robustly (parameter ensembles, walk-forward re-estimation) and combine many pairs into a risk-balanced, market-neutral portfolio.

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| 2.2 Statistics (regression, hypothesis tests, multiple testing) | S2, S6, S7 |
| 2.3 Time series (stationarity, ADF, cointegration, OU, Hurst, HMM, Kalman) | All sessions |
| 2.1 Linear algebra (eigen-decomposition, PCA) | S5, S6 |
| Part 4 shortability and borrow checks | S3, S5 |
| Part 7 S4 pairs preview, strategy framework | S3 |
| Part 8 backtester, costs, walk-forward, DSR/PBO, risk engine, portfolio methods | S3, S7, S8 |

---

## 3. Two-Week Overview

| Week | Theme | Sessions | Clinic Lab | Platform Output |
|---|---|---|---|---|
| **W1** | Mean reversion & pairs | S1 Stochastic processes & Bayesian inference · S2 Cointegration & pair selection · S3 The pairs strategy · S4 Kalman dynamic hedge ratio | Screen a sector universe, pick pairs with FDR control, backtest the best 5 | `research/statarb/ou.py`, `cointegration.py`, `screen.py`, `pairs.py`, `kalman.py` |
| **W2** | Multi-asset stat-arb, factors, robustness | S5 PCA residual stat-arb · S6 Factor models & cross-sectional strategies · S7 Breaks, stability & multiple testing · S8 Optimization, portfolio of pairs & M5b | Market-neutral pairs portfolio through the validation gate and risk engine | `research/statarb/pca.py`, `factors.py`, `stability.py`, **M5b release** |

---

## 4. Session-by-Session Plan

> Each 120-min session: **25 min theory → 60 min live coding → 25 min guided lab → 10 min wrap-up and homework.**
> Notebooks live in `notebooks/part09/`; promoted code lives in `quantforge/research/statarb/`. Offline, auto-graded exercises for both weeks and clinics W1 and W2 are in [`labs/part09/`](../../labs/part09/).

### Week 1 — Mean Reversion & Pairs

#### S1 · Stochastic Processes & Bayesian Inference (9.1)

| Block | Content |
|---|---|
| Theory | **GBM** (the default price model and why returns, not prices, are stationary). **Ornstein–Uhlenbeck** `dX = θ(μ − X)dt + σdW`: speed θ, long-run mean μ, equilibrium std `σ/√(2θ)`, **half-life `ln 2 / θ`** (the single most useful number in mean-reversion trading: it sets look-back windows and time stops). Estimation through the exact AR(1) discretization. **Jump diffusion** (Merton): why spreads sometimes jump and never come back (mergers, earnings). **Hurst exponent** and variance-ratio tests as mean-reversion diagnostics. **Bayesian inference:** posterior of a mean return and of a Sharpe ratio with conjugate priors; shrinking noisy estimates toward a prior; credible intervals for θ. |
| Live coding | `fit_ou` (below); simulate OU paths and recover parameters; Bayesian posterior of mean daily return. |
| Lab | Half-life of 20 ETF-pair spreads; compare OU half-life with Hurst and variance-ratio conclusions. |
| Homework | Bootstrap confidence interval for the half-life; how wide is it with 1 vs 5 years of data? |

```python
def fit_ou(x, dt=1.0):
    """Fit dX = theta(mu - X)dt + sigma dW using the exact AR(1) form X[t+1] = a + b X[t] + e."""
    x0, x1 = x[:-1], x[1:]
    b, a = np.polyfit(x0, x1, 1)
    resid = x1 - (a + b * x0)
    theta = -np.log(b) / dt                       # requires 0 < b < 1 (mean-reverting)
    mu = a / (1 - b)
    sigma = resid.std(ddof=2) * np.sqrt(2 * theta / (1 - b**2))
    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life": np.log(2) / theta}
```

> Verified: on 5,000 simulated OU steps with a true half-life of 6.9 bars, the fit returns 6.5 bars (well within the estimation error).

#### S2 · Cointegration & Pair Selection (9.2)

| Block | Content |
|---|---|
| Theory | Correlation ≠ cointegration: correlated returns can drift apart forever; cointegrated prices share a stationary spread. **Engle–Granger** (OLS hedge ratio, then ADF on residuals with the correct critical values; the test is asymmetric in y/x). **Johansen** (several series at once, rank of cointegration, basket weights from eigenvectors). **Selection pipeline:** economic link first (same industry, same index, ETF vs holdings, dual listings, share classes) → distance method (Gatev et al.: sum of squared differences of normalized prices) or correlation clustering → cointegration test → half-life filter (e.g. 2–30 days) → liquidity and borrow filter. **Multiple testing:** 500 stocks give ~125,000 pairs, so ~6,000 "cointegrated" by chance at 5%; apply BH-FDR (Part 5) and require out-of-sample confirmation. |
| Live coding | `engle_granger` (below), Johansen with `statsmodels`, pair screener with FDR. |
| Lab | Screen the S&P 500 sector by sector (point-in-time universe from Part 8): how many pairs survive FDR in formation period, and how many stay cointegrated in the next year? |
| Homework | Distance-method screen; compare its picks with the cointegration screen. |

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def engle_granger(y: pd.Series, x: pd.Series):
    """OLS hedge ratio (y = alpha + beta·x), cointegration p-value, spread and its half-life."""
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    alpha, beta = ols.params.iloc[0], ols.params.iloc[1]
    spread = y - beta * x - alpha
    return {"alpha": alpha, "beta": beta, "pvalue": coint(y, x)[1], "spread": spread,
            "half_life": fit_ou(spread.to_numpy())["half_life"]}

# Johansen for baskets: trace statistics vs 95% critical values
jo = coint_johansen(prices[["XLE", "XOM", "CVX", "COP"]].to_numpy(), det_order=0, k_ar_diff=1)
rank = int((jo.lr1 > jo.cvt[:, 1]).sum())        # number of cointegrating relationships
weights = jo.evec[:, 0]                          # basket weights of the strongest relationship
```

> Verified: on a synthetic cointegrated pair (true β = 1.5, spread half-life 6.6), the function returns β ≈ 1.47, p < 1e-12 and half-life ≈ 6.8; on an unrelated random walk, p ≈ 0.23 (correctly not cointegrated).

#### S3 · The Pairs Strategy, End to End (9.2)

| Block | Content |
|---|---|
| Theory | Spread definition (price vs log-price; fixed vs rolling hedge ratio). **Z-score** with a look-back tied to the half-life (≈ 2–4 half-lives). **Rules:** enter at |z| > 2, exit at |z| < 0.5, stop at |z| > 4 (the relationship may be breaking), **time stop** at k × half-life. **Sizing:** dollar-neutral vs beta-neutral vs volatility-balanced legs; round lots; per-pair risk budget. **Costs:** two legs × (commission + spread), **borrow fees** and hard-to-borrow recalls on the short leg, dividends on short positions. Execution: legs as simultaneous orders; handling one leg filling and the other not (hedge immediately or unwind). Formation period vs trading period; re-estimation schedule. |
| Live coding | `pairs_positions` state machine (below) → Part 7 `Strategy` subclass → Part 8 backtester with borrow costs. |
| Lab | Backtest the 5 best pairs from S2 on the out-of-sample year; compare fixed vs rolling hedge ratio and with/without time stop. |
| Homework | Leg-risk handling in the OMS: a "pair order" that unwinds the filled leg if the other leg does not fill within N seconds (paper test). |

```python
def pairs_positions(z, entry=2.0, exit_=0.5, stop=4.0, max_hold=None):
    """Spread position from the z-score: +1 = long spread (long y, short beta·x), -1 = short spread.
    Enter beyond ±entry, exit inside ±exit_, stop beyond ±stop, optional time stop (bars)."""
    pos, held = np.zeros(len(z)), 0
    for t in range(1, len(z)):
        p = pos[t - 1]
        if p == 0:
            if entry < abs(z[t]) < stop:
                p, held = -np.sign(z[t]), 0
        else:
            held += 1
            if abs(z[t]) < exit_ or abs(z[t]) > stop or (max_hold and held >= max_hold):
                p = 0
        pos[t] = p
    return pos      # act on the NEXT bar (Part 7 timing rule)
```

#### S4 · Dynamic Hedge Ratio with the Kalman Filter (9.2)

| Block | Content |
|---|---|
| Theory | Hedge ratios drift (business mix, index changes, leverage). Rolling OLS lags and depends on window length. **State-space regression:** `yₜ = αₜ + βₜ·xₜ + εₜ`, with `(αₜ, βₜ)` following a random walk. The filter gives an adaptive β, the one-step **forecast error** `eₜ` (the spread) and its variance `qₜ`, so `eₜ/√qₜ` is a self-scaling z-score. Tuning: `δ` (how fast β may move) vs observation noise `r`; too fast and the hedge absorbs the spread (no signal), too slow and it lags. Link to Part 7's Kalman trend filter. |
| Live coding | `kalman_hedge` (below); compare with rolling OLS on a pair with a drifting relationship. |
| Lab | Kalman vs rolling OLS vs fixed β on the S3 pairs: OOS Sharpe, turnover, hedge stability. |
| Homework | Tune `δ` by walk-forward (Part 8 S11); plot OOS Sharpe vs `δ`. |

```python
def kalman_hedge(y, x, delta=1e-4, r=1e-3):
    """Dynamic regression y_t = alpha_t + beta_t·x_t + e_t; (alpha, beta) follow a random walk.
    Returns beta_t, alpha_t, forecast error e_t and its variance q_t (trade z = e / sqrt(q))."""
    n = len(y)
    theta, P = np.zeros(2), np.eye(2)
    Vw = delta / (1 - delta) * np.eye(2)                 # state noise
    beta, alpha, e, q = (np.empty(n) for _ in range(4))
    for t in range(n):
        F = np.array([1.0, x[t]])
        P = P + Vw                                       # predict
        q[t] = F @ P @ F + r
        e[t] = y[t] - F @ theta                          # forecast error = spread
        K = P @ F / q[t]                                 # Kalman gain
        theta = theta + K * e[t]                         # update
        P = P - np.outer(K, F) @ P
        alpha[t], beta[t] = theta
    return beta, alpha, e, q
```

> Verified: with a true hedge ratio drifting linearly from 1.0 to 2.0 over 2,000 bars, the filter tracks it (β ≈ 1.16 at bar 200 when the true value is 1.10, and β ≈ 2.01 at the end).

**Clinic W1:** sector screen → FDR-controlled pair list → out-of-sample backtests of the top 5 with fixed, rolling and Kalman hedges → one-page summary of which approach held up.

---

### Week 2 — Multi-Asset Stat-Arb, Factors & Robustness

#### S5 · PCA / Residual Statistical Arbitrage (9.2, item 47)

| Block | Content |
|---|---|
| Theory | From pairs to portfolios: explain each stock's return by a few common factors and trade the **idiosyncratic residual**. Factors from **PCA** of the correlation matrix (eigenportfolios) or from **sector ETFs**. Avellaneda & Lee (2010): cumulate residuals over a 60-day window, fit an OU process, compute the **s-score** `(X − μ)/σ_eq`; open when |s| > 1.25, close near 0.5/0.75; trade only names with fast mean reversion (short half-life). The portfolio is automatically close to market- and factor-neutral. Capacity and turnover; crowding (Aug 2007 "quant quake"). |
| Live coding | `s_scores` (below) on a 100-stock universe; daily signal loop in the backtester. |
| Lab | PCA factors vs sector-ETF factors: s-score strategy OOS, turnover, factor exposures of the resulting book. |
| Homework | Add a volume-adjusted version (trading-time residuals, as in the paper) and compare. |

```python
def s_scores(returns: pd.DataFrame, n_factors=5, window=60):
    """Residual s-scores (Avellaneda & Lee 2010): regress each stock on the top PCA factors,
    cumulate the residuals, fit an OU process, return (X_last - mu) / sigma_equilibrium."""
    R = returns.iloc[-window:]
    Z = (R - R.mean()) / R.std()
    eigval, eigvec = np.linalg.eigh(np.corrcoef(Z.T.to_numpy()))
    F = Z.to_numpy() @ eigvec[:, ::-1][:, :n_factors]          # factor returns (T x k)
    A = np.c_[np.ones(window), F]
    out = {}
    for col in R.columns:
        coef, *_ = np.linalg.lstsq(A, R[col].to_numpy(), rcond=None)
        X = np.cumsum(R[col].to_numpy() - A @ coef)
        b, a = np.polyfit(X[:-1], X[1:], 1)
        if not 0 < b < 1:
            continue                                            # not mean-reverting in this window
        mu = a / (1 - b)
        sig_eq = np.sqrt((X[1:] - (a + b * X[:-1])).var(ddof=2) / (1 - b**2))
        out[col] = (X[-1] - mu) / sig_eq
    return pd.Series(out)
```

#### S6 · Factor Models & Cross-Sectional Strategies (9.2)

| Block | Content |
|---|---|
| Theory | Time-series factor models (CAPM, Fama–French 3/5 factors + momentum) for **risk attribution and neutralization**. **Cross-sectional** models: each period, regress next-period returns on stock characteristics (value, momentum, quality, low-volatility, size, short-term reversal); **Fama–MacBeth** averages the per-period coefficients to estimate factor premia with standard errors (use Newey–West when periods overlap). Building long-short factor portfolios; **factor neutralization** of any signal (residualize against sector and beta). **Other statistical regularities:** lead–lag (large caps leading small caps, ETFs leading components), seasonality and calendar effects (turn of month, pre-holiday, overnight vs intraday), tested with the Part 5 edge framework. |
| Live coding | `fama_macbeth` (below); neutralize the S5 signal against sector and beta. |
| Lab | Premia of 5 characteristics on the point-in-time S&P 500, 2010–2025; which are significant after FDR and in the second half of the sample? |
| Homework | Long-short decile portfolio for the strongest factor; turnover and cost-adjusted Sharpe via Part 8. |

```python
def fama_macbeth(returns: pd.DataFrame, exposures: dict[str, pd.DataFrame]):
    """Each date t: regress r_{i,t+1} on exposures at t across stocks; average the slopes.
    returns/exposures: dates x symbols. Use Newey-West errors if holding periods overlap."""
    lambdas = []
    for t, t1 in zip(returns.index[:-1], returns.index[1:]):
        X = pd.DataFrame({k: v.loc[t] for k, v in exposures.items()}).dropna()
        y = returns.loc[t1, X.index]
        lambdas.append(sm.OLS(y, sm.add_constant(X)).fit().params)
    L = pd.DataFrame(lambdas)
    return pd.DataFrame({"premium": L.mean(), "t_stat": L.mean() / (L.std(ddof=1) / np.sqrt(len(L)))})
```

> Verified: with a planted momentum premium of 0.20% per month across 100 stocks and 60 months, the function recovers 0.20% (t ≈ 16.6) and finds no premium for a pure-noise "size" factor (t ≈ 0.3).

#### S7 · Advanced Analysis: Breaks, Stability & Multiple Testing (9.3)

| Block | Content |
|---|---|
| Theory | Statistical relationships die. **Structural-break tests:** Chow (known date), CUSUM and Bai–Perron (unknown dates). **Relationship monitoring:** rolling cointegration p-values, rolling half-life, hedge-ratio drift, spread volatility regime; an automatic "retire this pair" rule. **Regime filtering** with an HMM (Part 2): trade reversion only in the calm regime. **Multiple hypothesis correction** at the strategy level: Bonferroni vs BH-FDR vs White's Reality Check / Hansen's SPA (bootstrap tests for "best of many strategies"), tying back to the Part 8 DSR. |
| Live coding | Rolling stability dashboard for a pair (p-value, half-life, β, spread vol); CUSUM test with `statsmodels`; retirement rule. |
| Lab | Replay 2015–2025 for 20 pairs: how long does a typical pair stay tradeable? Does the retirement rule improve OOS results? |
| Homework | White's Reality Check (stationary bootstrap) on the learner's family of pair strategies. |

#### S8 · Robust Optimization, Portfolio of Pairs & M5b Release (9.4)

| Block | Content |
|---|---|
| Theory | Parameters to optimize: formation window, z look-back (in half-lives), entry/exit/stop thresholds, `δ` for Kalman, number of PCA factors. **Robust choices:** plateau selection (Part 8 S9), **parameter ensembles** (average the positions of several nearby parameter sets instead of picking one), walk-forward re-estimation of pairs and hedge ratios. **Portfolio of pairs:** many small, weakly correlated bets; allocate with inverse volatility / risk parity / HRP (Part 8 S21–S23); caps per sector and per name (a stock can appear in several pairs); keep net beta ≈ 0 and gross leverage within limits. Capacity and borrow constraints at portfolio level. |
| Live coding | Ensemble pairs strategy; portfolio of 20 pairs with HRP allocation and the Part 8 risk engine (beta, gross, sector, borrow rules). |
| Lab | Full validation dossier (Part 8 gate): walk-forward, DSR (using the whole research log), PBO, robustness, capacity. |
| Homework | Final assessment (Section 7). |

**Clinic W2:** market-neutral pairs portfolio defense: dossier, risk report (beta, sector, gross/net), and 1 week of paper trading through the OMS with leg-risk handling.

---

## 5. Notebook Map (`notebooks/part09/`)

| Notebook | Session | Promoted to |
|---|---|---|
| `01_ou_bayes.ipynb` | S1 | `research/statarb/ou.py`, `analytics/bayes.py` |
| `02_cointegration_screen.ipynb` | S2 | `research/statarb/cointegration.py`, `research/statarb/screen.py` |
| `03_pairs_strategy.ipynb` | S3 | `research/statarb/pairs.py`, `strategy/library/pairs.py`, `execution/pair_order.py` |
| `04_kalman_hedge.ipynb` | S4 | `research/statarb/kalman.py` |
| `05_pca_residual.ipynb` | S5 | `research/statarb/pca.py`, `strategy/library/residual_statarb.py` |
| `06_factor_models.ipynb` | S6 | `research/statarb/factors.py` |
| `07_stability_breaks.ipynb` | S7 | `research/statarb/stability.py`, `analytics/reality_check.py` |
| `08_pairs_portfolio.ipynb` | S8 | `portfolio/pairs_book.py` |

---

## 6. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Trading correlated but not cointegrated pairs | Spreads drift apart, stops hit | Cointegration + half-life filters; OOS confirmation |
| 2 | Testing thousands of pairs without correction | Many "great" pairs fail next year | BH-FDR, economic-link prefilter, formation/trading split |
| 3 | Hedge ratio estimated on the same data it is traded on | Look-ahead, inflated results | Formation period strictly before trading period; walk-forward |
| 4 | Z-score look-back unrelated to half-life | Too many or too few signals | Set look-back ≈ 2–4 half-lives |
| 5 | No stop or time stop | One broken pair wipes out many winners | |z| stop, k × half-life time stop, retirement rule |
| 6 | Ignoring borrow cost and availability | Short leg unavailable or expensive | Borrow data from IB/Alpaca in costs and pre-trade checks |
| 7 | Legging risk | Unhedged exposure when one leg fails | Pair order with unwind logic |
| 8 | Kalman `δ` too large | Hedge absorbs the spread, no signal | Walk-forward tuning of `δ`; check spread variance |
| 9 | Using price levels in PCA | Spurious factors | PCA on standardized returns |
| 10 | Survivorship-biased universe | Pairs of survivors look stable | Point-in-time universe (Part 8 S5) |
| 11 | Treating a market-neutral book as riskless | Losses in crowded unwinds (e.g. Aug 2007) | Stress tests, gross leverage caps, crowding awareness |
| 12 | Same stock in many pairs | Hidden concentration | Per-name caps in the risk engine |
| 13 | Ignoring corporate events | Spread jumps permanently (mergers, spin-offs) | Event calendar filter; jump detection |
| 14 | Overlapping Fama–MacBeth periods with plain t-stats | Overstated significance | Newey–West standard errors |

---

## 7. Assessment — Platform Milestone M5b

**Task:** a market-neutral statistical-arbitrage book, validated and running on paper.

1. **Research library:** OU fitting, Engle–Granger and Johansen tests, FDR pair screener, Kalman hedge, PCA s-scores, Fama–MacBeth, stability monitors; unit tests on simulated data with known parameters.
2. **Pairs portfolio:** ≥ 15 pairs selected in a formation period with FDR control; strategy with z-score rules, time stop and retirement rule; Kalman or rolling hedge (justified).
3. **Validation dossier** through the Part 8 gate: walk-forward, DSR from the full research log, PBO, robustness, capacity, borrow-cost sensitivity.
4. **Risk:** HRP or risk-parity allocation; net beta, gross, sector and per-name limits enforced by the risk engine; stress tests including an Aug-2007-style crowded unwind.
5. **Paper trading:** 1 week through the OMS with pair orders and leg-risk handling; backtest-vs-paper reconciliation.

| Criterion | Points |
|---|---|
| Statistical foundations (OU, cointegration, Kalman) correct and tested | 20 |
| Pair selection with multiple-testing control and OOS evidence | 15 |
| Strategy design (rules, costs, borrow, leg risk) | 15 |
| Multi-asset extension (PCA residuals or factor model) | 10 |
| Validation dossier (walk-forward, DSR, PBO, robustness) | 15 |
| Portfolio construction & market-neutral risk control | 15 |
| Code quality, reproducibility, documentation | 10 |
| **Total** | **100** |

Pass mark: 70, **and** (mandatory) the formation and trading periods must be strictly separated, with FDR applied to the pair screen.

---

## 8. Further Reading

| Type | Reference |
|---|---|
| Book | Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley. |
| Book | Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley. (mean reversion, Kalman) |
| Book | Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley. |
| Book | Tsay, R. (2010). *Analysis of Financial Time Series* (3rd ed.). Wiley. |
| Book | Hamilton, J. (1994). *Time Series Analysis*. Princeton. (cointegration, Kalman filter) |
| Paper | Engle, R. & Granger, C. (1987). "Co-Integration and Error Correction." *Econometrica*, 55(2). |
| Paper | Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors." *Econometrica*, 59(6). |
| Paper | Gatev, E., Goetzmann, W. & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *RFS*, 19(3). |
| Paper | Avellaneda, M. & Lee, J.-H. (2010). "Statistical Arbitrage in the US Equities Market." *Quantitative Finance*, 10(7). |
| Paper | Khandani, A. & Lo, A. (2007). "What Happened to the Quants in August 2007?" *Journal of Investment Management*, 5(4). |
| Paper | Fama, E. & MacBeth, J. (1973). "Risk, Return, and Equilibrium: Empirical Tests." *JPE*, 81(3). |
| Paper | Fama, E. & French, K. (2015). "A Five-Factor Asset Pricing Model." *JFE*, 116(1). |
| Paper | White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*, 68(5). |
| Paper | Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *JBES*, 23(4). |
| Paper | Bai, J. & Perron, P. (2003). "Computation and Analysis of Multiple Structural Change Models." *Journal of Applied Econometrics*, 18(1). |
| Docs | `statsmodels` (coint, Johansen, CUSUM, state-space), Kenneth French data library, IB/Alpaca shortable and borrow data |

---

## 9. Instructor Notes

- Run the S2 screen live and count the "significant" pairs before and after FDR. The drop is the lesson.
- Use a point-in-time universe for every lab; survivorship bias is especially flattering to pairs.
- Borrow data matters: show one pair whose backtest profit disappears once hard-to-borrow fees are included.
- The Aug-2007 quant quake is the stress case for S5/S8; show how many "independent" stat-arb books lost at the same time.
- Paper trading only. Pair orders need special care in the OMS; test leg-risk handling deliberately (submit one leg with an unfillable limit price).
