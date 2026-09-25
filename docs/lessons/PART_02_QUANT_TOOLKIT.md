# Part 2 — Quantitative Toolkit: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 1, **Month 2, first half** (program weeks 5–6); Part 3.1–3.2 (Python basics and advanced Python) follows in weeks 7–8 |
| Format | 8 sessions × **120 min** (4 per week) + 2 lab clinics × 120 min + self-study (~7 hrs/week) |
| Total effort | ~16 hrs live + 4 hrs clinic + 14 hrs self-study ≈ **34 hours** |
| Tools | **Guided Jupyter notebooks**: the course provides the data loading and plotting code; learners write the short formulas and analysis cells (a gentle start before Python is taught formally in Part 3). Libraries used: NumPy, pandas, SciPy, statsmodels, `arch`, `hmmlearn`, `cvxpy`. Excel remains available for cross-checks. |
| Data | Daily prices for 10 US tickers (e.g. SPY, QQQ, IWM, TLT, GLD, XLE, AAPL, JPM, XOM, a volatile small cap), provided as Parquet/CSV; intraday 5-minute bars for 2 tickers; FRED series from Part 1 |
| Graded deliverables | **Stylized-facts & volatility notebook** on the 10 tickers (due end of Month 2, after Part 3.1) · a tested `metrics` module · quiz |
| Covers original items | "Stock Market Basic" 2 (statistics in financial markets) · roadmap 2.1–2.4 · foundations used by Parts 5–10 (and "Trade Creation" 1, advanced statistics, in Part 9) |

**Why this Part matters:** almost every mistake in quantitative trading is a statistical mistake: trusting a mean estimated from too little data, assuming normal tails, regressing one trending series on another, forgetting that volatility clusters, or reporting a Sharpe ratio without its uncertainty. This Part gives learners the tools *and* the scepticism, using real market data from the first lab.

---

## 1. Learning Objectives

By the end of Part 2 the learner will be able to:

1. **Use linear algebra** for portfolios: covariance matrices, portfolio variance, eigen-decomposition and PCA, Cholesky simulation, and explain why ill-conditioned covariance matrices break optimizers.
2. **Use calculus and optimization**: derivatives as sensitivities (Greeks), Taylor approximations and their limits, Lagrange multipliers (minimum-variance portfolio), root finding, and convex optimization.
3. **Describe return distributions** correctly: simple vs log returns, moments, fat tails, skewness, and the "stylized facts" of financial returns.
4. **Make statistical inferences** with honest uncertainty: standard errors, confidence intervals, hypothesis tests, multiple testing, bootstrap for dependent data, and regression with robust standard errors.
5. **Analyze time series**: stationarity tests, autocorrelation, random walk vs mean reversion (variance ratio, Hurst, half-life), ARIMA, spurious regression.
6. **Model and forecast volatility** with EWMA and GARCH-family models, and evaluate forecasts properly.
7. **Recognize regimes and relationships**: cointegration, Ornstein–Uhlenbeck processes, hidden Markov models, the Kalman filter idea, and Monte Carlo simulation.
8. **Compute and interpret performance and risk metrics** (Sharpe, Sortino, Calmar, drawdowns, VaR/CVaR, beta, information ratio), including their pitfalls.

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| Algebra, logs/exponentials, basic probability | All sessions |
| Part 1 S2–S4 (macro data, point-in-time), S9 (microstructure, Roll estimator), S11 and S15 (options, Black–Scholes in Excel) | S2, S3, S5 |
| Part 1 S14 (returns, volatility and drawdown in Excel) | S3, S8 |
| Course Jupyter environment installed (end of Part 1) | All labs |

A short self-study refresher (matrices, derivatives, expectation/variance) is provided before week 5 for learners who need it.

---

## 3. Two-Week Overview

| Week | Theme | Sessions | Clinic Lab | Output |
|---|---|---|---|---|
| **W1** | Math & statistics | S1 Linear algebra for portfolios · S2 Calculus & optimization · S3 Return distributions & stylized facts · S4 Estimation, inference & regression | Stylized-facts lab on the 10 tickers | `stats/linalg.py`, `stats/distributions.py`, `stats/inference.py` |
| **W2** | Time series, volatility & metrics | S5 Stationarity, autocorrelation & mean reversion · S6 Volatility modelling · S7 Cointegration, regimes, filters & simulation · S8 Performance & risk metrics | Volatility forecasting + metrics module with tests | `stats/timeseries.py`, `stats/volatility.py`, `analytics/metrics.py` |

These modules become part of `quantforge` in Part 3 (M0) and are reused throughout the program.

---

## 4. Session-by-Session Plan

> Each 120-min session: **25 min theory → 60 min worked examples (live notebook) → 25 min guided lab → 10 min wrap-up and homework.**
> Guided notebooks live in `notebooks/part02/`.

### Week 1 — Math & Statistics

#### S1 · Linear Algebra for Portfolios (2.1)

| Block | Content |
|---|---|
| Theory | Vectors as portfolios (weights) and return series; matrices as data (T × N returns) and as covariance. **Portfolio variance** `σ²ₚ = wᵀΣw`; correlation vs covariance; diversification and why it fails when correlations rise. **Positive semi-definiteness** (a valid covariance matrix; why pairwise-estimated correlations can break it). **Eigen-decomposition and PCA:** principal components as uncorrelated "factors"; the first component of a stock universe is usually "the market" (used again in Part 9). **Conditioning:** small eigenvalues make `Σ⁻¹` unstable, which is why naive mean-variance optimization gives extreme weights (Part 8 shrinkage). **Cholesky factor** to simulate correlated shocks. |
| Worked examples | Portfolio volatility from a covariance matrix; PCA of 10 stocks driven by one common factor (the first component explains most of the variance); correlated simulation via Cholesky (sample covariance of 200,000 draws matches the target). |
| Lab | PCA on the 10 course tickers: variance explained by PC1–PC3; interpret PC1 loadings; compare 2019 vs 2020 (correlations rise in a crisis). |
| Homework | Show numerically how the condition number of the covariance matrix changes with the number of assets for a fixed 2-year sample. |

```python
def portfolio_vol(w, cov, periods=252):
    return float(np.sqrt(w @ cov @ w * periods))          # cov of daily returns -> annual vol

def pca_explained(returns: pd.DataFrame, k=3):
    """Share of variance explained by the top-k principal components of standardized returns."""
    eigval, eigvec = np.linalg.eigh(np.corrcoef(returns.T.to_numpy()))
    eigval, eigvec = eigval[::-1], eigvec[:, ::-1]           # largest first
    return eigval[:k] / eigval.sum(), eigvec[:, :k]

def correlated_normals(cov, n, seed=0):
    """z ~ N(0, I)  ->  L z ~ N(0, cov), with L the Cholesky factor of cov."""
    L = np.linalg.cholesky(cov)
    return np.random.default_rng(seed).standard_normal((n, len(cov))) @ L.T
```

#### S2 · Calculus & Optimization (2.1)

| Block | Content |
|---|---|
| Theory | Derivatives as **sensitivities** (delta = ∂V/∂S, duration as ∂P/∂y); partial derivatives and gradients. **Taylor expansion:** `ΔV ≈ Δ·ΔS + ½Γ·ΔS²` and where it fails (large moves, jumps). **Convexity** and why it matters (option payoffs, bond prices, risk measures). **Constrained optimization with Lagrange multipliers:** the minimum-variance portfolio `w = Σ⁻¹1 / (1ᵀΣ⁻¹1)`. **Numerical methods:** root finding (bisection, Newton; implied volatility in Part 6), gradient descent, and **convex optimization** with `cvxpy` (the same minimum-variance problem, now with constraints such as no short selling). |
| Worked examples | Delta and delta-gamma approximations vs full repricing of an at-the-money call (below); minimum-variance weights in closed form vs `cvxpy`; Newton's method. |
| Lab | Long-only minimum-variance portfolio of the 10 tickers with `cvxpy`; compare with the unconstrained closed form (which may short). |
| Homework | Add a maximum-weight constraint (e.g. 25%) and a target-return constraint; plot the efficient frontier. |

```python
def min_variance_weights(cov):
    """Closed form from the Lagrangian: w = Σ⁻¹1 / (1ᵀΣ⁻¹1)."""
    x = np.linalg.solve(cov, np.ones(len(cov)))              # solve, never invert explicitly
    return x / x.sum()

import cvxpy as cp
w = cp.Variable(len(cov))
cp.Problem(cp.Minimize(cp.quad_form(w, cov)), [cp.sum(w) == 1, w >= 0]).solve()   # long-only version
```

> Verified: for a 3-asset example, the closed form and `cvxpy` (without the long-only constraint) give identical weights. For an at-the-money 3-month call (σ = 20%), the delta-gamma approximation matches full repricing for a 1-point move (0.579) but overshoots a 15-point move by about 1 point (12.83 vs 11.82), while delta alone is off by 3.4 points.

#### S3 · Return Distributions & Stylized Facts (2.2)

| Block | Content |
|---|---|
| Theory | **Simple vs log returns:** log returns add over time; simple returns add across assets in a portfolio; use each where it is exact. Moments: mean, variance, **skewness**, **kurtosis** (excess kurtosis = kurtosis − 3). **Fat tails:** QQ plots, Jarque–Bera test, counting 4σ days vs what a normal distribution predicts; Student-t fits (degrees of freedom ≈ 3–5 for daily stock returns). **Volatility scaling** `σ·√T` and when it misleads (autocorrelation, fat tails, jumps). **Stylized facts of returns** (Cont 2001): heavy tails, little autocorrelation in returns but strong autocorrelation in **absolute/squared returns** (volatility clustering), leverage effect (volatility rises after falls), aggregational Gaussianity (monthly returns closer to normal than daily), gain/loss asymmetry. |
| Worked examples | Moments, Jarque–Bera, 4σ-day count and Student-t fit (below) on SPY and on a volatile single stock. |
| Lab | **Stylized-facts lab** (clinic W1): verify each stylized fact on the 10 tickers with a table and plots. |
| Homework | Compare daily, weekly and monthly returns of SPY: how do skewness and kurtosis change with the horizon? |

```python
from scipy import stats

def describe_returns(r):
    r = np.asarray(r)
    return {"mean": r.mean(), "std": r.std(ddof=1), "skew": stats.skew(r),
            "excess_kurtosis": stats.kurtosis(r), "jb_pvalue": stats.jarque_bera(r).pvalue}

def tail_count(r, k=4.0):
    """Number of |z| > k observations vs the number a normal distribution predicts."""
    r = np.asarray(r)
    z = (r - r.mean()) / r.std(ddof=1)
    return int((np.abs(z) > k).sum()), float(2 * stats.norm.sf(k) * len(r))

df, loc, scale = stats.t.fit(r)        # degrees of freedom of a Student-t fit
```

> Verified on 6,000 simulated Student-t (3 d.f.) returns: 36 observations beyond 4σ vs 0.38 expected under normality; the t fit recovers ≈ 3.2 degrees of freedom.

#### S4 · Estimation, Inference & Regression (2.2)

| Block | Content |
|---|---|
| Theory | **Sampling error:** the standard error of a mean return is `σ/√T`, so with 16% volatility even 10 years of data gives a ±10%-per-year 95% interval for the mean. **t-statistic of a strategy** ≈ annual Sharpe × √years; a true Sharpe of 0.5 needs about 16 years for t ≈ 2 on average. **Confidence intervals and hypothesis tests;** what a p-value is and is not. **Multiple testing:** test 100 random strategies and about 5 look "significant" at 5%; Bonferroni and Benjamini–Hochberg (used heavily in Parts 5, 8 and 9). **Bootstrap:** i.i.d. bootstrap and **block bootstrap** for dependent data. **Regression:** OLS, CAPM beta and alpha, R²; **heteroskedasticity- and autocorrelation-robust (Newey–West/HAC) standard errors**; correlation vs causation. **Robust statistics:** median, MAD, winsorizing. |
| Worked examples | t-stat vs Sharpe × √years; years of data needed (below); block-bootstrap confidence interval for the mean; CAPM regression with HAC standard errors. |
| Lab | Beta and alpha of the 10 tickers vs SPY with ordinary and HAC standard errors; which alphas are significant, and how many survive BH-FDR? |
| Homework | Generate 200 random-signal "strategies" on SPY and count how many pass p < 0.05, before and after FDR correction. |

```python
def mean_return_tstat(r):
    r = np.asarray(r)
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))      # ≈ annual Sharpe × sqrt(years)

def years_needed(sharpe_annual, t_target=2.0):
    """Average years of data for a true Sharpe to reach a t-stat of t_target."""
    return (t_target / sharpe_annual) ** 2                   # Sharpe 0.5 -> 16 years

def block_bootstrap_ci(r, stat=np.mean, block=20, n_boot=2000, alpha=0.05, seed=0):
    """Moving-block bootstrap: resample blocks of consecutive days to keep short-range dependence."""
    r = np.asarray(r); n = len(r); rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    boots = [stat(r[np.concatenate([np.arange(s, s + block)
                                    for s in rng.choice(starts, n // block + 1)])[:n]])
             for _ in range(n_boot)]
    return np.quantile(boots, [alpha / 2, 1 - alpha / 2])

import statsmodels.api as sm
capm = sm.OLS(stock, sm.add_constant(market)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
```

**Clinic W1 (120 min):** stylized-facts lab. Deliverable: one table (rows = 10 tickers; columns = skew, excess kurtosis, JB p-value, 4σ count vs normal, t d.f., autocorrelation of returns and of |returns|, leverage correlation) plus 3 plots, with a one-paragraph interpretation per fact.

---

### Week 2 — Time Series, Volatility & Metrics

#### S5 · Stationarity, Autocorrelation & Mean Reversion (2.3)

| Block | Content |
|---|---|
| Theory | **Stationarity** and why most models need it: prices are not stationary, returns roughly are. **ACF/PACF**; Ljung–Box test. **Unit-root tests:** ADF (null: unit root) and **KPSS** (null: stationary), used together because each can mislead alone. **Random walk vs mean reversion vs trend:** Lo–MacKinlay **variance ratio**, **Hurst exponent**, AR(1) coefficient and **half-life**. **ARMA/ARIMA:** fitting, information criteria, and honest out-of-sample evaluation against a naive forecast (daily returns are hard to beat). **Spurious regression:** regressing one random walk on another gives "significant" results; the fix is differencing or cointegration (S7). |
| Worked examples | ADF/KPSS on prices vs returns; variance ratio and half-life for a random walk vs an AR(1) spread (below); spurious regression demonstration. |
| Lab | For the 10 tickers and 5 spreads (e.g. XLE − XOM): stationarity tests, variance ratios, half-lives; which look mean-reverting, and how confident can we be? |
| Homework | ARIMA forecast of daily SPY returns vs a zero forecast, out of sample: which wins? |

```python
from statsmodels.tsa.stattools import adfuller, kpss

def variance_ratio(log_prices, q=5):
    """Lo–MacKinlay variance ratio: ≈ 1 random walk, < 1 mean reversion, > 1 trending."""
    r1 = np.diff(log_prices)
    rq = log_prices[q:] - log_prices[:-q]
    return rq.var(ddof=1) / (q * r1.var(ddof=1))

def ou_half_life(x):
    """Half-life of mean reversion from an AR(1) fit x[t+1] = a + b·x[t]."""
    b, a = np.polyfit(x[:-1], x[1:], 1)
    return np.log(2) / -np.log(b)
```

> Verified: a simulated random walk gives VR ≈ 0.98; an AR(1) spread with φ = 0.9 gives VR ≈ 0.78. Across 20 simulated samples of 3,000 days, the half-life estimate averages 6.7 vs a true 6.6, but individual samples range from 5.7 to 7.6, which shows how noisy such estimates are. Regressing one simulated random walk on another unrelated one gave t = 4.1: a textbook spurious regression.

#### S6 · Volatility Modelling & Forecasting (2.3)

| Block | Content |
|---|---|
| Theory | Volatility is **not constant** and it **clusters**, which makes it far more predictable than returns. **Estimators:** rolling standard deviation, **EWMA** (RiskMetrics, λ = 0.94), range-based estimators (Parkinson, Garman–Klass, Yang–Zhang; implemented in Part 5), **realized volatility** from intraday returns. **ARCH/GARCH(1,1):** `σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁`; persistence α + β; long-run variance `ω / (1 − α − β)`; Student-t innovations for fat tails; **GJR-GARCH** for the leverage effect. **Forecast evaluation:** compare against realized variance with **QLIKE** and MSE, out of sample; Mincer–Zarnowitz regression. Uses in trading: position sizing (Part 8), option pricing inputs (Part 6), regime signals (Part 7). |
| Worked examples | EWMA volatility; GARCH(1,1)-t and GJR fits with the `arch` package; multi-step variance forecasts converging to the long-run level (below). |
| Lab | Volatility-forecast competition on the 10 tickers: rolling 21-day std vs EWMA vs GARCH-t, evaluated out of sample with QLIKE against 5-minute realized variance (2 tickers) or squared returns (the rest). |
| Homework | Is there a leverage effect? Compare GJR's asymmetry parameter across the 10 tickers (equity indices vs bonds vs gold). |

```python
from arch import arch_model

def ewma_vol(r, lam=0.94, periods=252):
    """RiskMetrics EWMA volatility; the value at t uses returns up to t-1 (a forecast for t)."""
    r = np.asarray(r); var = np.empty(len(r)); var[0] = r[:20].var()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(var * periods)

def qlike(realized_var, forecast_var):
    """QLIKE loss for variance forecasts (robust to noise in the realized proxy); lower is better."""
    ratio = realized_var / forecast_var
    return float(np.mean(ratio - np.log(ratio) - 1))

res = arch_model(100 * returns, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")   # scale to % for stability
gjr = arch_model(100 * returns, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
forecast_var = res.forecast(horizon=5).variance.iloc[-1]
```

> Verified: on 4,000 simulated GARCH(1,1)-t observations (true α = 0.10, β = 0.88, ν = 6), the `arch` fit recovers α ≈ 0.11, β ≈ 0.88, ν ≈ 5.4, and multi-step forecasts rise smoothly toward the long-run variance.

#### S7 · Cointegration, Regimes, Filters & Simulation (2.3)

| Block | Content |
|---|---|
| Theory | **Cointegration:** two non-stationary prices with a stationary combination; Engle–Granger in two steps (deep dive in Part 9). **Ornstein–Uhlenbeck process** as the continuous-time model of mean reversion; half-life links to S5. **Regime models:** **hidden Markov models** (e.g. calm vs stressed regimes with different means and volatilities); persistence of states; decoding the most likely state path (and why a real-time filter must not use future data). **Kalman filter** intuition: a recursive estimate of a hidden quantity (trend, hedge ratio) updated by each new observation (used in Parts 7 and 9). **Monte Carlo simulation:** geometric Brownian motion, bootstrap paths from history, and scenario analysis. |
| Worked examples | Two-state Gaussian HMM on returns with calm and stress periods (below); GBM Monte Carlo checked against its theoretical mean; Engle–Granger on a cointegrated simulated pair. |
| Lab | HMM regimes on SPY (2005–2025): regime timeline, regime statistics, and a warning about using `predict` on the full sample for a backtest (look-ahead; filter forward instead). |
| Homework | Monte Carlo: probability that a portfolio with 10% expected return and 18% volatility has a drawdown worse than 25% within 5 years, using GBM vs bootstrapped historical paths. |

```python
from hmmlearn.hmm import GaussianHMM

hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=0).fit(x)  # x: (T, 1) returns
states = hmm.predict(x)      # smoothed states use the WHOLE sample: fine for description, not for trading signals

def gbm_paths(S0, mu, sigma, T, steps, n_paths, seed=0):
    dt = T / steps
    z = np.random.default_rng(seed).standard_normal((n_paths, steps))
    return S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1))
```

> Verified: on simulated returns alternating calm (0.7% daily vol) and stressed (2.5%) periods, the 2-state HMM recovers both volatilities and labels 99.4% of days correctly. 20,000 GBM paths (μ = 7%, σ = 20%, 1 year) give a mean terminal value of 107.27 vs the theoretical 107.25.

#### S8 · Performance & Risk Metrics (2.4)

| Block | Content |
|---|---|
| Theory | **Return metrics:** CAGR, arithmetic vs geometric mean (volatility drag ≈ σ²/2). **Risk-adjusted:** Sharpe (with the risk-free rate), Sortino (downside deviation), Calmar (CAGR / max drawdown), Omega. **Drawdowns:** maximum drawdown, duration, time under water. **Shape:** skewness, kurtosis, tail ratio. **Relative:** beta, alpha, tracking error, information ratio, up/down capture. **Risk:** VaR and CVaR (Expected Shortfall), historical vs parametric normal vs **Cornish–Fisher** (adjusts for skew and kurtosis, but unreliable when tails are very heavy) vs Monte Carlo. **Pitfalls:** annualizing with √252 assumes independent returns (autocorrelated or smoothed returns inflate Sharpe; Lo 2002), metrics have sampling error (S4), max drawdown grows with the length of the sample, and VaR says nothing about losses beyond it (use CVaR). |
| Worked examples | `metrics()` and `var_cvar()` (below) on a sample strategy vs SPY; the three VaR methods on moderately and very fat-tailed data. |
| Lab | Metrics module with unit tests against a reference library (e.g. `quantstats`) on the 10 tickers; one-page tear sheet per ticker. |
| Homework | Show how Sharpe changes when monthly returns of a smoothed (illiquid-asset-like) series are used instead of daily, and explain why. |

```python
def metrics(r, rf=0.0, periods=252, benchmark=None):
    r = np.asarray(r); ex = r - rf / periods
    eq = np.cumprod(1 + r); dd = eq / np.maximum.accumulate(eq) - 1
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    cagr = eq[-1] ** (periods / len(r)) - 1
    out = {"cagr": cagr, "vol": r.std(ddof=1) * np.sqrt(periods),
           "sharpe": ex.mean() / ex.std(ddof=1) * np.sqrt(periods),
           "sortino": ex.mean() / downside * np.sqrt(periods),
           "max_dd": dd.min(), "calmar": cagr / abs(dd.min()),
           "skew": stats.skew(r), "excess_kurt": stats.kurtosis(r),
           "tail_ratio": abs(np.quantile(r, 0.95) / np.quantile(r, 0.05))}
    if benchmark is not None:
        b = np.asarray(benchmark)
        beta = np.cov(r, b)[0, 1] / b.var(ddof=1)
        active = r - b
        out.update(beta=beta, alpha_ann=(r.mean() - beta * b.mean()) * periods,
                   info_ratio=active.mean() / active.std(ddof=1) * np.sqrt(periods))
    return out

def var_cvar(r, alpha=0.99, method="historical"):
    """One-period VaR and CVaR as positive loss fractions."""
    r = np.asarray(r); mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    if method == "historical":
        var = -np.quantile(r, 1 - alpha)
        return var, -r[r <= -var].mean()
    if method == "normal":
        return -(mu + z * sd), -(mu - sd * stats.norm.pdf(z) / (1 - alpha))
    if method == "cornish_fisher":                        # VaR only
        s, k = stats.skew(r), stats.kurtosis(r)
        zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        return -(mu + zcf * sd), None
    raise ValueError(method)
```

> Verified on 200,000 simulated Student-t returns with 8 degrees of freedom (moderately fat tails): 99% VaR is 2.91% historical, 2.69% normal (too low) and 3.14% Cornish–Fisher. With 4 degrees of freedom (very fat tails) the Cornish–Fisher estimate breaks down (11.4% vs 3.8% historical), which is why historical, EVT (Part 5) or Monte Carlo methods are preferred for heavy tails.

**Clinic W2 (120 min):** volatility-forecast competition results + metrics module review (tests passing against the reference library).

---

## 5. Notebook Map (`notebooks/part02/`)

| Notebook | Session | Promoted to (in Part 3, M0) | Used again in |
|---|---|---|---|
| `01_linear_algebra_pca.ipynb` | S1 | `stats/linalg.py` | Part 8 (covariance, MVO), Part 9 (PCA stat-arb) |
| `02_calculus_optimization.ipynb` | S2 | `stats/optimize.py` | Part 6 (Greeks, IV root finding), Part 8 (portfolio optimization) |
| `03_distributions_stylized_facts.ipynb` | S3 | `stats/distributions.py` | Part 5 (tail risk), Part 8 (significance) |
| `04_inference_regression.ipynb` | S4 | `stats/inference.py` | Part 5 (edge tests), Part 8 (PSR/DSR), Part 9 (factor models) |
| `05_stationarity_mean_reversion.ipynb` | S5 | `stats/timeseries.py` | Parts 7 and 9 (mean reversion, pairs) |
| `06_volatility_garch.ipynb` | S6 | `stats/volatility.py` | Parts 5, 6, 8, 10 |
| `07_cointegration_hmm_montecarlo.ipynb` | S7 | `stats/regimes.py`, `stats/simulate.py` | Parts 9, 10 |
| `08_metrics_var.ipynb` | S8 | `analytics/metrics.py` | Part 8 (`analytics/performance.py`) and every later tear sheet |

---

## 6. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Adding simple returns over time or log returns across assets | Wrong cumulative or portfolio returns | Log returns over time, simple returns across assets |
| 2 | Assuming normal tails | Risk underestimated; "impossible" days happen yearly | QQ plots, tail counts, Student-t or EVT |
| 3 | Trusting a mean return from a few years of data | Overconfident expectations | Standard errors; years-needed calculation |
| 4 | Testing many ideas and reporting the best p-value | False discoveries | Count tests; BH-FDR; out-of-sample confirmation |
| 5 | i.i.d. bootstrap on dependent data | Confidence intervals too narrow | Block bootstrap |
| 6 | Plain OLS standard errors on financial data | Overstated significance | HAC (Newey–West) standard errors |
| 7 | Regressing price levels on price levels | Spurious "relationships" | Stationarity tests; differences or cointegration |
| 8 | Using ADF alone | Contradictory conclusions | ADF and KPSS together; look at the data |
| 9 | Constant-volatility assumption | Bad sizing and VaR in stress | EWMA/GARCH; volatility forecasts evaluated out of sample |
| 10 | Evaluating volatility forecasts in-sample | Overfit models look best | Out-of-sample QLIKE vs a simple benchmark |
| 11 | HMM states decoded on the full sample used as signals | Look-ahead | Forward filtering only for trading use |
| 12 | Inverting an ill-conditioned covariance matrix | Extreme portfolio weights | `solve`, shrinkage (Part 8), condition-number checks |
| 13 | √252 annualization of autocorrelated returns | Inflated Sharpe | Check autocorrelation; Lo (2002) adjustment |
| 14 | Cornish–Fisher VaR with very fat tails | Absurd VaR numbers | Historical, EVT or Monte Carlo VaR |

---

## 7. Assessment

**A. Stylized-facts & volatility notebook (graded; due end of Month 2, after Part 3.1), 50%.** On the 10 tickers:
1. Stylized facts table and plots with interpretation (S3 clinic).
2. Stationarity and mean-reversion analysis of prices, returns and 5 spreads (S5).
3. Volatility-forecast comparison (rolling, EWMA, GARCH-t, GJR) out of sample with QLIKE (S6).
4. Regime analysis with an HMM, used only in a forward-filtered way (S7).
5. Conclusions stated with their uncertainty (standard errors, confidence intervals).

**B. Metrics module, 30%.** `metrics()` and `var_cvar()` (plus drawdown duration and up/down capture), unit-tested against a reference library, documented formulas and units.

**C. Quiz, 20%.** Linear algebra and optimization, distributions, inference and multiple testing, time-series tests, volatility models, metrics.

| Criterion | Points |
|---|---|
| Stylized facts: correct measures, clear interpretation | 15 |
| Time-series analysis: correct tests, sensible conclusions, no spurious regressions | 15 |
| Volatility modelling: correct fitting, honest out-of-sample evaluation | 15 |
| Uncertainty: standard errors, bootstrap, multiple-testing awareness throughout | 5 |
| Metrics module: correctness vs reference, tests, documentation | 30 |
| Quiz | 20 |
| **Total** | **100** |

Pass mark: 70, **and** (mandatory) the volatility comparison must be out of sample, and the metrics module must pass its reference tests.

---

## 8. Further Reading

| Type | Reference |
|---|---|
| Book | Ruppert, D. & Matteson, D. (2015). *Statistics and Data Analysis for Financial Engineering* (2nd ed.). Springer. |
| Book | Tsay, R. (2010). *Analysis of Financial Time Series* (3rd ed.). Wiley. |
| Book | Campbell, J., Lo, A. & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton. |
| Book | Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. |
| Book | Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press (free online). |
| Book | Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. |
| Paper | Cont, R. (2001). "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues." *Quantitative Finance*, 1(2). |
| Paper | Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3). |
| Paper | Glosten, L., Jagannathan, R. & Runkle, D. (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance*, 48(5). (GJR) |
| Paper | Lo, A. & MacKinlay, A. C. (1988). "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test." *RFS*, 1(1). (variance ratio) |
| Paper | Granger, C. & Newbold, P. (1974). "Spurious Regressions in Econometrics." *Journal of Econometrics*, 2(2). |
| Paper | Newey, W. & West, K. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3). |
| Paper | Patton, A. (2011). "Volatility Forecast Comparison Using Imperfect Volatility Proxies." *Journal of Econometrics*, 160(1). (QLIKE) |
| Paper | Hamilton, J. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2). (regime switching) |
| Paper | Lo, A. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*, 58(4). |
| Docs | NumPy, SciPy, statsmodels, `arch`, `hmmlearn`, `cvxpy` |

---

## 9. Instructor Notes

- Start every session with real data and a question ("Is SPY's return distribution normal?") before the theory; learners remember the surprise.
- Keep the guided notebooks' "write it yourself" cells short (1–5 lines) this month; the full coding load starts in Part 3.
- Emphasize uncertainty in every lab: every estimate gets a standard error or a confidence interval, and every "significant" result gets asked "how many things did you try?".
- The graded notebook is due at the end of Month 2 so learners can finish it with the Python basics from Part 3.1; the analyses are designed in Part 2.
- Re-use the same 10 tickers throughout the program so learners build intuition about their behaviour.
