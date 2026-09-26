"""Build the Part 2 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part02:  python tools/build_notebooks.py
Cells are ("md", text), ("code", code) or ("ex", starter_code, solution_code).
Edit the content here and rebuild, so starter and solution notebooks never drift apart.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
KERNEL = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}

SETUP = """import sys
from pathlib import Path
for d in (Path.cwd(), Path.cwd().parent):       # p2lib.py is in notebooks/part02/
    sys.path.insert(0, str(d))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p2lib as p

p.use_course_style()
pd.set_option("display.float_format", "{:,.4f}".format)
prices = p.load_prices()          # dates × 10 tickers (course data via P2_DATA, else synthetic)
rets = p.log_returns(prices)      # daily log returns
prices.tail(3)"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 2 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_02_QUANT_TOOLKIT.md)

**You will:**
{goals}

How these notebooks work: the loading and plotting code is written for you. Cells marked **✍️ Your turn** need 1–5 lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
NB["01_linear_algebra_pca"] = [
    header("01", "Linear algebra for portfolios", "S1 (Linear algebra for portfolios)",
           "1. Compute portfolio volatility from a covariance matrix.\n2. Find the hidden factors in returns with PCA.\n"
           "3. Simulate correlated returns with a Cholesky factor.\n4. See why some covariance matrices are dangerous to invert."),
    ("code", SETUP),
    ("md", "## 1. Covariance matrix and portfolio volatility\n\n$\\sigma_p = \\sqrt{w^\\top \\Sigma w \\cdot 252}$ for daily returns."),
    ("code", "cov = rets.cov()                   # daily covariance matrix (10 × 10)\ncov.round(6)"),
    ("md", YOUR_TURN),
    ("ex", """w = np.full(len(cov), 1 / len(cov))          # equal weights
# ✍️ annualized portfolio volatility: sqrt(wᵀ Σ w × 252)   (hint: use cov.to_numpy() and @)
port_vol = ...
port_vol = p.check("portfolio volatility", port_vol, p.portfolio_vol(w, cov))""",
     """w = np.full(len(cov), 1 / len(cov))          # equal weights
port_vol = np.sqrt(w @ cov.to_numpy() @ w * 252)
port_vol = p.check("portfolio volatility", port_vol, p.portfolio_vol(w, cov))"""),
    ("code", """single = rets.std() * np.sqrt(252)
print(f"Average single-asset volatility: {single.mean():.1%}")
print(f"Equal-weight portfolio volatility: {port_vol:.1%}   (diversification benefit)")
print(f"Check from the portfolio's own return series: {(rets @ w).std() * np.sqrt(252):.1%}")"""),
    ("md", "## 2. PCA: hidden factors in returns\n\nEigen-decompose the correlation matrix. The largest eigenvalue's share of the total is the variance explained by the first principal component (usually \"the market\")."),
    ("md", YOUR_TURN),
    ("ex", """corr = rets.corr().to_numpy()
# ✍️ 1) eigval, eigvec = np.linalg.eigh(corr)   2) reverse both so the largest comes first
#    3) explained = share of the total variance of the top 3 components
eigval, eigvec = ..., ...
explained = ...
explained = p.check("PCA: variance explained by top 3", explained, p.pca_explained(rets)[0])""",
     """corr = rets.corr().to_numpy()
eigval, eigvec = np.linalg.eigh(corr)
eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
explained = eigval[:3] / eigval.sum()
explained = p.check("PCA: variance explained by top 3", explained, p.pca_explained(rets)[0])"""),
    ("code", """_, loadings = p.pca_explained(rets, k=3)
pc1 = pd.Series(loadings[:, 0], index=rets.columns)
pc1 = pc1 * np.sign(pc1.sum())                     # an eigenvector's sign is arbitrary: make PC1 mostly positive
ax = pc1.sort_values().plot.barh(title="PC1 loadings (the 'market' factor)")
ax.axvline(0, color="#52514e", lw=1); plt.show()
calm, crisis = rets.loc["2019"], rets.loc["2020-02-15":"2020-06-30"]
print(f"PC1 explains {p.pca_explained(calm)[0][0]:.0%} in 2019 vs {p.pca_explained(crisis)[0][0]:.0%} in the 2020 crisis window")"""),
    ("md", "## 3. Simulating correlated returns (Cholesky)\n\nIf $L L^\\top = \\Sigma$ and $z \\sim N(0, I)$, then $L z \\sim N(0, \\Sigma)$."),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ the lower-triangular Cholesky factor of the covariance matrix (hint: np.linalg.cholesky)
L = ...
L = p.check("Cholesky factor", L, np.linalg.cholesky(cov.to_numpy()))""",
     """L = np.linalg.cholesky(cov.to_numpy())
L = p.check("Cholesky factor", L, np.linalg.cholesky(cov.to_numpy()))"""),
    ("code", """z = np.random.default_rng(0).standard_normal((200_000, len(cov)))
sim = z @ L.T
err = np.abs(np.cov(sim.T) - cov.to_numpy()).max() / np.abs(cov.to_numpy()).max()
print(f"Largest covariance error of the simulation, relative to the largest covariance: {err:.2%}")"""),
    ("md", "## 4. Ill-conditioned covariance matrices\n\nThe condition number (largest / smallest eigenvalue) tells you how unstable $\\Sigma^{-1}$ is. It grows as you add assets relative to the amount of data."),
    ("code", """window = rets.loc["2023":"2024"]
for n in [3, 5, 10]:
    c = np.linalg.cond(window.iloc[:, :n].cov())
    print(f"{n:>2} assets, {len(window)} days: condition number {c:,.0f}")"""),
    ("md", """## Questions
1. Why is the portfolio volatility lower than the average single-asset volatility? When would it not be?
2. What does PC1's share rising in a crisis tell you about diversification when you need it most?
3. Why does a high condition number lead to extreme weights in mean-variance optimization (Part 8)?"""),
]

# ---------------------------------------------------------------------------------------------- 02
NB["02_calculus_optimization"] = [
    header("02", "Calculus and optimization", "S2 (Calculus & optimization)",
           "1. Approximate option P&L with delta and gamma, and see where it fails.\n2. Solve the minimum-variance portfolio in closed form.\n"
           "3. Add constraints with `cvxpy`.\n4. Find an implied volatility with Newton's method."),
    ("code", SETUP),
    ("md", "## 1. Taylor approximation: delta and gamma\n\n$\\Delta V \\approx \\Delta \\cdot \\Delta S + \\tfrac{1}{2} \\Gamma \\cdot \\Delta S^2$"),
    ("code", """S, K, T, r, q, vol = 100.0, 100.0, 0.25, 0.04, 0.0, 0.20      # ✏️ change me
base = p.bsm_call(S, K, T, r, q, vol)
print({k: round(v, 4) for k, v in base.items()})"""),
    ("md", YOUR_TURN),
    ("ex", """moves = np.array([1.0, 5.0, 15.0])
# ✍️ delta-gamma approximation of the change in call value for each move (use base["delta"], base["gamma"])
approx = ...
approx = p.check("delta-gamma approximation", approx, p.delta_gamma_pnl(base["delta"], base["gamma"], moves))""",
     """moves = np.array([1.0, 5.0, 15.0])
approx = base["delta"] * moves + 0.5 * base["gamma"] * moves**2
approx = p.check("delta-gamma approximation", approx, p.delta_gamma_pnl(base["delta"], base["gamma"], moves))"""),
    ("code", """dS = np.linspace(-25, 25, 201)
full = np.array([p.bsm_call(S + x, K, T, r, q, vol)["price"] for x in dS]) - base["price"]
fig, ax = plt.subplots()
ax.plot(dS, full, label="Full repricing")
ax.plot(dS, base["delta"] * dS, label="Delta only", ls="--")
ax.plot(dS, p.delta_gamma_pnl(base["delta"], base["gamma"], dS), label="Delta + gamma", ls=":")
ax.set(title="Change in call value vs move in the underlying", xlabel="ΔS"); ax.legend(); plt.show()
pd.DataFrame({"move": moves, "full": [p.bsm_call(S + m, K, T, r, q, vol)["price"] - base["price"] for m in moves],
              "delta-gamma": approx})"""),
    ("md", "## 2. Minimum-variance portfolio (Lagrange multipliers)\n\n$w = \\dfrac{\\Sigma^{-1}\\mathbf{1}}{\\mathbf{1}^\\top \\Sigma^{-1} \\mathbf{1}}$ — compute it with `np.linalg.solve`, never by inverting the matrix."),
    ("md", YOUR_TURN),
    ("ex", """cov_ann = rets.cov().to_numpy() * 252
# ✍️ x = solve(Σ, 1), then normalize so the weights sum to 1
w_mv = ...
w_mv = p.check("minimum-variance weights", w_mv, p.min_variance_weights(cov_ann))""",
     """cov_ann = rets.cov().to_numpy() * 252
x = np.linalg.solve(cov_ann, np.ones(len(cov_ann)))
w_mv = x / x.sum()
w_mv = p.check("minimum-variance weights", w_mv, p.min_variance_weights(cov_ann))"""),
    ("md", "## 3. Adding constraints with `cvxpy` (no short selling)"),
    ("code", """import cvxpy as cp
x = cp.Variable(len(cov_ann))
cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(cov_ann))), [cp.sum(x) == 1, x >= 0]).solve()
weights = pd.DataFrame({"closed form (may short)": w_mv, "long-only (cvxpy)": x.value}, index=rets.columns)
weights["long-only (cvxpy)"] = weights["long-only (cvxpy)"].clip(lower=0).round(4)
vols = {c: np.sqrt(weights[c] @ cov_ann @ weights[c]) for c in weights}
print({k: f"{v:.2%}" for k, v in vols.items()})
weights"""),
    ("code", """# Efficient frontier (long-only): minimum variance for each target return
mu = rets.mean().to_numpy() * 252
targets = np.linspace(mu.min(), mu.max(), 25)
frontier = []
for m in targets:
    x = cp.Variable(len(mu))
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(cov_ann))), [cp.sum(x) == 1, x >= 0, mu @ x == m])
    prob.solve()
    if x.value is not None:
        frontier.append((np.sqrt(prob.value), m))
f = np.array(frontier)
fig, ax = plt.subplots()
ax.plot(f[:, 0] * 100, f[:, 1] * 100, label="Efficient frontier (long-only)")
ax.scatter(np.sqrt(np.diag(cov_ann)) * 100, mu * 100, color="#52514e", s=20, zorder=3, label="Single assets")
for i, t in enumerate(rets.columns):
    ax.annotate(t, (np.sqrt(cov_ann[i, i]) * 100, mu[i] * 100), fontsize=8, color="#52514e", xytext=(3, 3), textcoords="offset points")
ax.set(xlabel="volatility (%)", ylabel="mean log return (%/yr)", title="Efficient frontier"); ax.legend(); plt.show()"""),
    ("md", "## 4. Root finding: implied volatility with Newton's method\n\nUpdate: $\\sigma \\leftarrow \\sigma - \\dfrac{C(\\sigma) - C_{market}}{\\text{vega}(\\sigma)}$"),
    ("md", YOUR_TURN),
    ("ex", """market_price, sigma = 5.0, 0.30                 # ✏️ market call price; starting guess

def newton_step(sigma):
    c = p.bsm_call(S, K, T, r, q, sigma)
    # ✍️ return the next sigma (use c["price"], c["vega"], market_price)
    return ...

for _ in range(20):
    nxt = newton_step(sigma)
    if nxt is Ellipsis:
        break                                        # not done yet
    sigma = nxt
from scipy.optimize import brentq
sigma = p.check("implied volatility", sigma,
                brentq(lambda v: p.bsm_call(S, K, T, r, q, v)["price"] - market_price, 1e-4, 5))
print(f"Implied volatility: {sigma:.4%}")""",
     """market_price, sigma = 5.0, 0.30                 # ✏️ market call price; starting guess

def newton_step(sigma):
    c = p.bsm_call(S, K, T, r, q, sigma)
    return sigma - (c["price"] - market_price) / c["vega"]

for _ in range(20):
    nxt = newton_step(sigma)
    if nxt is Ellipsis:
        break                                        # not done yet
    sigma = nxt
from scipy.optimize import brentq
sigma = p.check("implied volatility", sigma,
                brentq(lambda v: p.bsm_call(S, K, T, r, q, v)["price"] - market_price, 1e-4, 5))
print(f"Implied volatility: {sigma:.4%}")"""),
    ("md", """## Questions
1. For which size of move is the delta-gamma approximation good enough? What does that mean for risk reports during crashes?
2. Why do the closed-form weights include short positions, and why does the long-only answer have higher volatility?
3. When can Newton's method fail for implied volatility? (Hint: vega near zero.)"""),
]

# ---------------------------------------------------------------------------------------------- 03
NB["03_distributions_stylized_facts"] = [
    header("03", "Return distributions and stylized facts", "S3 (Return distributions & stylized facts) · Clinic W1",
           "1. Compute log returns and see why they add up over time.\n2. Measure fat tails and skew.\n"
           "3. Verify the stylized facts of returns on 10 tickers (clinic deliverable)."),
    ("code", SETUP),
    ("md", "## 1. Log returns\n\n$r_t = \\ln(P_t / P_{t-1})$"),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ daily log returns of `prices` (drop the first empty row)
my_rets = ...
my_rets = p.check("log returns", my_rets, p.log_returns(prices))""",
     """my_rets = np.log(prices / prices.shift(1)).dropna()
my_rets = p.check("log returns", my_rets, p.log_returns(prices))"""),
    ("ex", """# ✍️ total simple return of SPY over the whole sample, computed from the SUM of its log returns
total_from_logs = ...
total_from_logs = p.check("total return from log returns", total_from_logs, prices["SPY"].iloc[-1] / prices["SPY"].iloc[0] - 1)""",
     """total_from_logs = np.exp(my_rets["SPY"].sum()) - 1
total_from_logs = p.check("total return from log returns", total_from_logs, prices["SPY"].iloc[-1] / prices["SPY"].iloc[0] - 1)"""),
    ("md", "## 2. Moments, fat tails, normality"),
    ("code", """moments = pd.DataFrame({t: p.describe(rets[t]) for t in rets}).T
moments"""),
    ("md", YOUR_TURN),
    ("ex", """r = rets["SPY"]
# ✍️ standardize SPY returns (z = (r - mean) / std) and count days with |z| > 4
z = ...
n_tail = ...
n_tail = p.check("SPY days beyond 4 sigma", n_tail, p.tail_count(r)[0])
print(f"A normal distribution would predict about {p.tail_count(r)[1]:.2f} such days in {len(r)} days.")""",
     """r = rets["SPY"]
z = (r - r.mean()) / r.std()
n_tail = int((z.abs() > 4).sum())
n_tail = p.check("SPY days beyond 4 sigma", n_tail, p.tail_count(r)[0])
print(f"A normal distribution would predict about {p.tail_count(r)[1]:.2f} such days in {len(r)} days.")"""),
    ("code", """from scipy import stats
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
(osm, osr), (slope, icpt, _) = stats.probplot(rets["SPY"], dist="norm")
axes[0].scatter(osm, osr, s=6, color=p.PALETTE[0]); axes[0].plot(osm, slope * osm + icpt, color="#52514e", lw=1)
axes[0].set(title="QQ plot vs normal: SPY", xlabel="normal quantiles", ylabel="return quantiles")
df, loc, scale = stats.t.fit(rets["SPY"])
x = np.linspace(rets["SPY"].min(), rets["SPY"].max(), 400)
axes[1].hist(rets["SPY"], bins=150, density=True, color="#b7d3f6", label="SPY daily returns")
axes[1].plot(x, stats.norm.pdf(x, rets["SPY"].mean(), rets["SPY"].std()), label="Normal fit")
axes[1].plot(x, stats.t.pdf(x, df, loc, scale), label=f"Student-t fit (df = {df:.1f})")
axes[1].set(title="Distribution of SPY returns", yscale="log", ylim=(1e-2, None)); axes[1].legend()
plt.tight_layout(); plt.show()"""),
    ("md", "## 3. The stylized facts (clinic W1 table)"),
    ("code", """rows = {}
for t in rets:
    r = rets[t]
    rows[t] = {"skew": p.describe(r)["skew"], "excess kurtosis": p.describe(r)["excess_kurtosis"],
               "JB p-value": p.describe(r)["jb_pvalue"], ">4σ days": p.tail_count(r)[0],
               ">4σ expected": p.tail_count(r)[1], "t d.f.": stats.t.fit(r)[0],
               "acf(r)": p.autocorr(r), "acf(|r|)": p.autocorr(r.abs()),
               "leverage corr(r, next |r|)": r.corr(r.abs().shift(-1))}
facts = pd.DataFrame(rows).T
facts"""),
    ("code", """lags = range(1, 21)
acf_r = [rets["SPY"].autocorr(k) for k in lags]
acf_abs = [rets["SPY"].abs().autocorr(k) for k in lags]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
axes[0].bar(lags, acf_r, color=p.PALETTE[0]); axes[0].set_title("Autocorrelation of returns (SPY)")
axes[1].bar(lags, acf_abs, color=p.PALETTE[0]); axes[1].set_title("Autocorrelation of |returns| (SPY)")
for ax in axes: ax.axhline(0, color="#52514e", lw=1); ax.set_xlabel("lag (days)")
plt.tight_layout(); plt.show()"""),
    ("md", "## 4. Aggregational Gaussianity: longer horizons look more normal"),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ monthly log returns of SPY = sum of daily log returns in each month (resample("ME")), then excess kurtosis
monthly = ...
kurt_monthly = ...
kurt_monthly = p.check("SPY monthly excess kurtosis", kurt_monthly, stats.kurtosis(rets["SPY"].resample("ME").sum()))
print(f"Excess kurtosis: daily {stats.kurtosis(rets['SPY']):.2f} vs monthly {kurt_monthly:.2f}")""",
     """monthly = rets["SPY"].resample("ME").sum()
kurt_monthly = stats.kurtosis(monthly)
kurt_monthly = p.check("SPY monthly excess kurtosis", kurt_monthly, stats.kurtosis(rets["SPY"].resample("ME").sum()))
print(f"Excess kurtosis: daily {stats.kurtosis(rets['SPY']):.2f} vs monthly {kurt_monthly:.2f}")"""),
    ("md", """## Questions (clinic W1 deliverable)
Write one paragraph per stylized fact using the table: fat tails, volatility clustering, (almost) no autocorrelation in returns,
leverage effect, aggregational Gaussianity. Which tickers break a pattern, and why might they?"""),
]

# ---------------------------------------------------------------------------------------------- 04
NB["04_inference_regression"] = [
    header("04", "Estimation, inference and regression", "S4 (Estimation, inference & regression)",
           "1. Quantify how uncertain a mean return (and a Sharpe ratio) really is.\n2. Use the block bootstrap for dependent data.\n"
           "3. Estimate CAPM betas with robust (HAC) standard errors.\n4. See multiple testing create false discoveries."),
    ("code", SETUP),
    ("md", "## 1. How much data does a mean return need?"),
    ("md", YOUR_TURN),
    ("ex", """r = rets["SPY"]
# ✍️ t-statistic of the mean daily return: mean / (std / sqrt(n))
t_stat = ...
t_stat = p.check("t-stat of SPY's mean return", t_stat, p.tstat_mean(r))
years = len(r) / 252
print(f"t = {t_stat:.2f};  annual Sharpe × sqrt(years) = {r.mean() / r.std() * np.sqrt(252) * np.sqrt(years):.2f}")""",
     """r = rets["SPY"]
t_stat = r.mean() / (r.std() / np.sqrt(len(r)))
t_stat = p.check("t-stat of SPY's mean return", t_stat, p.tstat_mean(r))
years = len(r) / 252
print(f"t = {t_stat:.2f};  annual Sharpe × sqrt(years) = {r.mean() / r.std() * np.sqrt(252) * np.sqrt(years):.2f}")"""),
    ("ex", """# ✍️ years of data needed, on average, for a true Sharpe of 0.5 and of 1.0 to reach t = 2  (t ≈ SR·√years)
years_05, years_10 = ..., ...
years_05 = p.check("years needed (Sharpe 0.5)", years_05, p.years_needed(0.5))
years_10 = p.check("years needed (Sharpe 1.0)", years_10, p.years_needed(1.0))""",
     """years_05, years_10 = (2 / 0.5) ** 2, (2 / 1.0) ** 2
years_05 = p.check("years needed (Sharpe 0.5)", years_05, p.years_needed(0.5))
years_10 = p.check("years needed (Sharpe 1.0)", years_10, p.years_needed(1.0))"""),
    ("md", "## 2. Confidence intervals: normal theory vs block bootstrap"),
    ("code", """se = r.std() / np.sqrt(len(r))
normal_ci = (r.mean() - 1.96 * se, r.mean() + 1.96 * se)
boot_ci = p.block_bootstrap_ci(r, block=20)
print("95% CI for the mean daily return, annualized (%):")
print(f"  normal theory:   {normal_ci[0]*252:.2%} to {normal_ci[1]*252:.2%}")
print(f"  block bootstrap: {boot_ci[0]*252:.2%} to {boot_ci[1]*252:.2%}")"""),
    ("md", "## 3. CAPM regressions with HAC standard errors"),
    ("code", """import statsmodels.api as sm
rows = {}
for t in rets.columns.drop("SPY"):
    X = sm.add_constant(rets["SPY"])
    plain = sm.OLS(rets[t], X).fit()
    hac = sm.OLS(rets[t], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    rows[t] = {"beta": hac.params["SPY"], "alpha (ann.)": hac.params["const"] * 252,
               "alpha t (plain)": plain.tvalues["const"], "alpha t (HAC)": hac.tvalues["const"], "R²": hac.rsquared}
capm = pd.DataFrame(rows).T
capm"""),
    ("md", "## 4. Multiple testing\n\n200 \"strategies\" that trade on random coin flips. None has any real edge."),
    ("code", """from scipy import stats
rng = np.random.default_rng(1)
pvals = []
for _ in range(200):
    signal = rng.choice([-1, 1], size=len(r))              # random long/short every day
    strat = signal * r.to_numpy()
    pvals.append(stats.ttest_1samp(strat, 0).pvalue)
pvals = np.array(pvals)"""),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ 1) how many p-values are below 0.05?
#    2) apply Benjamini–Hochberg: stats.false_discovery_control(pvals, method="bh"), count adjusted p < 0.05
n_sig = ...
n_sig_fdr = ...
n_sig, n_sig_fdr = p.check("significant before / after FDR", (n_sig, n_sig_fdr), p.fdr_counts(pvals))
print(f"'Significant' strategies: {n_sig} of 200 before correction, {n_sig_fdr} after BH-FDR")""",
     """n_sig = int((pvals < 0.05).sum())
n_sig_fdr = int((stats.false_discovery_control(pvals, method="bh") < 0.05).sum())
n_sig, n_sig_fdr = p.check("significant before / after FDR", (n_sig, n_sig_fdr), p.fdr_counts(pvals))
print(f"'Significant' strategies: {n_sig} of 200 before correction, {n_sig_fdr} after BH-FDR")"""),
    ("md", """## Questions
1. With 10+ years of data, can you tell whether SPY's true Sharpe is 0.3 or 0.6? Use the confidence interval.
2. Why do HAC t-statistics differ from plain OLS ones? Which alphas survive?
3. You tested 200 ideas and 9 "worked". What should you report, and what should you do next?"""),
]

# ---------------------------------------------------------------------------------------------- 05
NB["05_stationarity_mean_reversion"] = [
    header("05", "Stationarity and mean reversion", "S5 (Stationarity, autocorrelation & mean reversion)",
           "1. Test prices and returns for stationarity (ADF and KPSS together).\n2. Measure mean reversion with the variance ratio and half-life.\n"
           "3. See a spurious regression happen."),
    ("code", SETUP),
    ("md", "## 1. Unit-root tests: ADF (null = unit root) and KPSS (null = stationary)"),
    ("code", """import warnings
from statsmodels.tsa.stattools import adfuller, kpss
spread = np.log(prices["XOM"]) - 0.9 * np.log(prices["XLE"])      # a candidate mean-reverting spread
series = {"SPY log price": np.log(prices["SPY"]), "SPY log return": rets["SPY"], "XOM − 0.9·XLE spread": spread}
rows = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                                 # KPSS warns when p is outside its table
    for name, s in series.items():
        rows[name] = {"ADF p-value": adfuller(s.dropna())[1], "KPSS p-value": kpss(s.dropna(), nlags="auto")[1]}
pd.DataFrame(rows).T"""),
    ("md", "## 2. Variance ratio and half-life"),
    ("md", YOUR_TURN),
    ("ex", """q = 5
lp = np.log(prices["SPY"]).to_numpy()
# ✍️ variance ratio = var(q-day changes) / (q × var(1-day changes)) of the log price
vr_spy = ...
vr_spy = p.check("variance ratio (SPY)", vr_spy, p.variance_ratio(lp, q))
print(f"SPY VR = {vr_spy:.3f} (≈1 random walk);  spread VR = {p.variance_ratio(spread.to_numpy(), q):.3f}")""",
     """q = 5
lp = np.log(prices["SPY"]).to_numpy()
r1, rq = np.diff(lp), lp[q:] - lp[:-q]
vr_spy = rq.var(ddof=1) / (q * r1.var(ddof=1))
vr_spy = p.check("variance ratio (SPY)", vr_spy, p.variance_ratio(lp, q))
print(f"SPY VR = {vr_spy:.3f} (≈1 random walk);  spread VR = {p.variance_ratio(spread.to_numpy(), q):.3f}")"""),
    ("ex", """x = spread.to_numpy()
# ✍️ fit x[t+1] = a + b·x[t] with np.polyfit(x[:-1], x[1:], 1) (returns [b, a]); half-life = ln 2 / (−ln b)
b, a = ..., ...
hl = ...
hl = p.check("spread half-life (days)", hl, p.half_life(x))""",
     """x = spread.to_numpy()
b, a = np.polyfit(x[:-1], x[1:], 1)
hl = np.log(2) / -np.log(b)
hl = p.check("spread half-life (days)", hl, p.half_life(x))"""),
    ("code", """z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
ax = z.plot(title=f"Spread z-score (60-day window); half-life ≈ {hl:.1f} days")
for lvl in (-2, 2): ax.axhline(lvl, color="#8a8984", lw=1, ls="--")
ax.axhline(0, color="#52514e", lw=1); ax.set_xlabel(""); plt.show()"""),
    ("md", "## 3. Spurious regression\n\nRegress one random walk on another, independent one."),
    ("code", """import statsmodels.api as sm
rng = np.random.default_rng(3)
a_walk, b_walk = np.cumsum(rng.normal(size=1000)), np.cumsum(rng.normal(size=1000))
levels = sm.OLS(a_walk, sm.add_constant(b_walk)).fit()
diffs = sm.OLS(np.diff(a_walk), sm.add_constant(np.diff(b_walk))).fit()
print(f"Levels:      t = {levels.tvalues[1]:6.1f},  R² = {levels.rsquared:.2f}   <- 'significant' but meaningless")
print(f"Differences: t = {diffs.tvalues[1]:6.1f},  R² = {diffs.rsquared:.3f}")"""),
    ("md", "## 4. Can an AR model beat a zero forecast out of sample?"),
    ("code", """split = int(len(rets) * 0.7)
train, test = rets["SPY"].iloc[:split], rets["SPY"].iloc[split:]
phi, c = np.polyfit(train.iloc[:-1], train.iloc[1:], 1)
forecast = c + phi * test.shift(1).dropna()
actual = test.iloc[1:]
print(f"AR(1) coefficient {phi:.3f}")
print(f"Out-of-sample MSE: AR(1) {np.mean((actual - forecast)**2):.3e}  vs  zero forecast {np.mean(actual**2):.3e}")"""),
    ("md", """## Questions
1. Why use ADF and KPSS together? What does it mean when both reject?
2. How confident are you in the half-life estimate? (Try the first and second half of the sample.)
3. Name a real-world pair of price series that would give a spurious regression."""),
]

# ---------------------------------------------------------------------------------------------- 06
NB["06_volatility_garch"] = [
    header("06", "Volatility modelling and forecasting", "S6 (Volatility modelling & forecasting) · Clinic W2",
           "1. Build an EWMA volatility estimate.\n2. Fit GARCH-t and GJR-GARCH models.\n"
           "3. Run an honest out-of-sample volatility forecast competition with QLIKE."),
    ("code", SETUP),
    ("md", "## 1. EWMA (RiskMetrics)\n\n$\\sigma^2_t = \\lambda\\,\\sigma^2_{t-1} + (1-\\lambda)\\,r^2_{t-1}$ with $\\lambda = 0.94$"),
    ("md", YOUR_TURN),
    ("ex", """r = rets["SPY"].to_numpy()

def ewma_next(prev_var, prev_ret, lam=0.94):
    # ✍️ return tomorrow's variance forecast: λ·prev_var + (1 − λ)·prev_ret²
    return ...

var = np.empty(len(r)); var[0] = r[:20].var()
for t in range(1, len(r)):
    nxt = ewma_next(var[t - 1], r[t - 1])
    if nxt is Ellipsis:
        break                                        # not done yet
    var[t] = nxt
lam = 0.94
var = p.check("EWMA variance", var, p.ewma_var(r, lam))""",
     """r = rets["SPY"].to_numpy()

def ewma_next(prev_var, prev_ret, lam=0.94):
    return lam * prev_var + (1 - lam) * prev_ret ** 2

var = np.empty(len(r)); var[0] = r[:20].var()
for t in range(1, len(r)):
    nxt = ewma_next(var[t - 1], r[t - 1])
    if nxt is Ellipsis:
        break                                        # not done yet
    var[t] = nxt
lam = 0.94
var = p.check("EWMA variance", var, p.ewma_var(r, lam))"""),
    ("code", """vol = pd.DataFrame({"EWMA (λ = 0.94)": np.sqrt(var * 252),
                    "21-day rolling": rets["SPY"].rolling(21).std().shift(1) * np.sqrt(252)}, index=rets.index)
ax = (vol * 100).plot(title="SPY volatility estimates (% annualized)"); ax.set_xlabel(""); plt.show()"""),
    ("md", "## 2. GARCH-t and GJR-GARCH (leverage effect)"),
    ("code", """from arch import arch_model
y = 100 * rets["SPY"]                               # percent returns keep the optimizer stable
garch = arch_model(y, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
gjr = arch_model(y, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
pars = garch.params
persistence = pars["alpha[1]"] + pars["beta[1]"]
long_run = np.sqrt(pars["omega"] / (1 - persistence) * 252) / 100
print(f"GARCH-t: persistence α+β = {persistence:.3f}, long-run volatility ≈ {long_run:.1%}")
print(f"GJR asymmetry γ = {gjr.params['gamma[1]']:.3f} (t = {gjr.tvalues['gamma[1]']:.1f}): bad news raises volatility more")
pd.DataFrame({"GARCH-t": garch.params, "GJR-t": gjr.params})"""),
    ("md", "## 3. Out-of-sample forecast competition\n\nQLIKE loss: mean of $\\;\\text{RV}/\\hat\\sigma^2 - \\ln(\\text{RV}/\\hat\\sigma^2) - 1$ (lower is better)."),
    ("md", YOUR_TURN),
    ("ex", """def my_qlike(realized_var, forecast_var):
    # ✍️ QLIKE loss (see the formula above); both inputs are arrays
    return ...

test_rv, test_fc = np.array([1.0, 2.0, 0.5]), np.array([1.2, 1.5, 0.7])
p.check("QLIKE", my_qlike(test_rv, test_fc), p.qlike(test_rv, test_fc));""",
     """def my_qlike(realized_var, forecast_var):
    ratio = np.asarray(realized_var) / np.asarray(forecast_var)
    return float(np.mean(ratio - np.log(ratio) - 1))

test_rv, test_fc = np.array([1.0, 2.0, 0.5]), np.array([1.2, 1.5, 0.7])
p.check("QLIKE", my_qlike(test_rv, test_fc), p.qlike(test_rv, test_fc));"""),
    ("code", """split = int(len(y) * 0.7)
fc = {}
for name, spec in {"GARCH-t": dict(o=0), "GJR-t": dict(o=1)}.items():
    res = arch_model(y, vol="GARCH", p=1, q=1, dist="t", **spec).fit(last_obs=y.index[split], disp="off")
    fc[name] = res.forecast(start=y.index[split], horizon=1, reindex=False).variance["h.1"].shift(1)   # forecast for the NEXT day
fc["EWMA"] = pd.Series(p.ewma_var(rets["SPY"].to_numpy()) * 1e4, index=y.index)
fc["21-day rolling"] = (y.rolling(21).var()).shift(1)
realized = y ** 2 + 1e-4                              # squared return as the (noisy) realized-variance proxy
table = {}
for name, f in fc.items():
    both = pd.concat([realized, f], axis=1).dropna()
    both = both.loc[both.index > y.index[split]]          # evaluate on the test period only
    table[name] = {"QLIKE": p.qlike(both.iloc[:, 0], both.iloc[:, 1]), "MSE": np.mean((both.iloc[:, 0] - both.iloc[:, 1]) ** 2)}
pd.DataFrame(table).T.sort_values("QLIKE")"""),
    ("md", "## 4. Realized volatility from intraday data (synthetic 5-minute bars)"),
    ("code", """i5 = p.intraday_5min("SPY")
rv = (i5["ret"] ** 2).groupby(i5.index.normalize()).sum()      # daily realized variance
rv_vol = np.sqrt(rv * 252)
ax = (rv_vol * 100).plot(title="Daily realized volatility from 5-minute returns (% annualized)"); ax.set_xlabel(""); plt.show()
print(f"Autocorrelation of daily realized variance: {pd.Series(rv.values).autocorr():.2f} (volatility is predictable)")"""),
    ("md", """## Questions
1. Which model wins out of sample, and by how much? Would the ranking change with a different split?
2. Is there a leverage effect in TLT or GLD? (Change `rets["SPY"]` in section 2.)
3. Why is volatility so much easier to forecast than returns?"""),
]

# ---------------------------------------------------------------------------------------------- 07
NB["07_cointegration_hmm_montecarlo"] = [
    header("07", "Cointegration, regimes and simulation", "S7 (Cointegration, regimes, filters & simulation)",
           "1. Run an Engle–Granger cointegration test.\n2. Find calm and stressed regimes with a hidden Markov model, without look-ahead.\n"
           "3. Estimate drawdown risk with Monte Carlo simulation."),
    ("code", SETUP),
    ("md", "## 1. Engle–Granger: is XOM cointegrated with XLE?"),
    ("md", YOUR_TURN),
    ("ex", """y, x = np.log(prices["XOM"]), np.log(prices["XLE"])
# ✍️ step 1: hedge ratio = OLS slope of y on x (hint: np.polyfit(x, y, 1)[0])
beta = ...
beta = p.check("hedge ratio", beta, p.hedge_ratio(y, x))""",
     """y, x = np.log(prices["XOM"]), np.log(prices["XLE"])
beta = np.polyfit(x, y, 1)[0]
beta = p.check("hedge ratio", beta, p.hedge_ratio(y, x))"""),
    ("code", """from statsmodels.tsa.stattools import coint
stat, pval, _ = coint(y, x)
resid = y - beta * x
print(f"Hedge ratio {beta:.3f};  Engle–Granger p-value {pval:.2e};  residual half-life {p.half_life(resid):.1f} days")
print(f"For comparison, SPY vs QQQ: p-value {coint(np.log(prices['SPY']), np.log(prices['QQQ']))[1]:.2f}")"""),
    ("md", "## 2. Regimes with a hidden Markov model\n\n`model.predict` uses the **whole sample** (fine for describing history, look-ahead for trading). `p.hmm_forward_filter` only uses data up to each day."),
    ("code", """from hmmlearn.hmm import GaussianHMM
xr = rets["SPY"].to_numpy().reshape(-1, 1)
hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=0).fit(xr)
stress_state = int(np.argmax([np.sqrt(c[0, 0]) for c in hmm.covars_]))
smoothed = hmm.predict_proba(xr)[:, stress_state]                    # uses the future
filtered = p.hmm_forward_filter(hmm, xr)[:, stress_state]            # uses only the past
probs = pd.DataFrame({"Smoothed (uses future data)": smoothed, "Filtered (real-time)": filtered}, index=rets.index)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
probs["Smoothed (uses future data)"].plot(ax=axes[0], title="P(stressed regime): smoothed", lw=1)
probs["Filtered (real-time)"].plot(ax=axes[1], title="P(stressed regime): filtered", color=p.PALETTE[1], lw=1)
for ax in axes: ax.set_xlabel(""); ax.set_ylim(-0.05, 1.05)
plt.tight_layout(); plt.show()
print(f"Days where the two disagree on the regime: {((smoothed > 0.5) != (filtered > 0.5)).mean():.1%}")"""),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ annualized volatility of SPY on calm days (filtered < 0.5) and on stressed days (filtered > 0.5)
r = rets["SPY"].to_numpy()
vol_calm = ...
vol_stress = ...
vol_calm, vol_stress = p.check("regime volatilities", (vol_calm, vol_stress), p.regime_vols(r, filtered))
print(f"Calm: {vol_calm:.1%}   Stressed: {vol_stress:.1%}")""",
     """r = rets["SPY"].to_numpy()
vol_calm = r[filtered <= 0.5].std(ddof=1) * np.sqrt(252)
vol_stress = r[filtered > 0.5].std(ddof=1) * np.sqrt(252)
vol_calm, vol_stress = p.check("regime volatilities", (vol_calm, vol_stress), p.regime_vols(r, filtered))
print(f"Calm: {vol_calm:.1%}   Stressed: {vol_stress:.1%}")"""),
    ("md", "## 3. Monte Carlo: how likely is a 25% drawdown within 5 years?"),
    ("code", """MU, SIGMA, YEARS, N = 0.10, 0.18, 5, 20_000          # ✏️ change me
paths = p.gbm_paths(100, MU, SIGMA, YEARS, 252 * YEARS, N, seed=1)
print(f"Mean terminal value {paths[:, -1].mean():.1f} vs theory {100 * np.exp(MU * YEARS):.1f}")"""),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ maximum drawdown of each path (along axis=1), then the share of paths worse than −25%
dd = ...
prob_dd = ...
prob_dd = p.check("P(max drawdown < −25%)", prob_dd, p.mc_drawdown_prob(paths, -0.25))""",
     """dd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
prob_dd = (dd < -0.25).mean()
prob_dd = p.check("P(max drawdown < −25%)", prob_dd, p.mc_drawdown_prob(paths, -0.25))"""),
    ("code", """# Same question with bootstrapped historical SPY returns (keeps fat tails and crashes)
rng = np.random.default_rng(2)
hist = rets["SPY"].to_numpy()
boot = 100 * np.exp(np.cumsum(rng.choice(hist, size=(5000, 252 * YEARS)), axis=1))
print(f"P(max drawdown < −25%): GBM {prob_dd:.1%}  vs  bootstrap of history {p.mc_drawdown_prob(boot, -0.25):.1%}")"""),
    ("md", """## Questions
1. Why is SPY vs QQQ not cointegrated even though they are highly correlated?
2. When does the filtered regime probability react later than the smoothed one? Why does that matter for a trading rule?
3. Why do the GBM and bootstrap drawdown probabilities differ?"""),
]

# ---------------------------------------------------------------------------------------------- 08
NB["08_metrics_var"] = [
    header("08", "Performance and risk metrics", "S8 (Performance & risk metrics) · Clinic W2",
           "1. Compute Sharpe, Sortino, drawdown, VaR and CVaR yourself.\n2. Build a tear-sheet table for 10 tickers.\n"
           "3. Compare VaR methods and see the Sharpe-annualization pitfall."),
    ("code", SETUP),
    ("code", """simple = p.simple_returns(prices)          # metrics use SIMPLE returns (they add across assets)
r = simple["SPY"]
RF = 0.02                                    # ✏️ risk-free rate (annual)"""),
    ("md", YOUR_TURN),
    ("ex", """# ✍️ annualized Sharpe: mean(excess) / std(excess) × sqrt(252), with excess = r − RF/252
sharpe = ...
sharpe = p.check("Sharpe", sharpe, p.sharpe(r, RF))""",
     """ex = r - RF / 252
sharpe = ex.mean() / ex.std() * np.sqrt(252)
sharpe = p.check("Sharpe", sharpe, p.sharpe(r, RF))"""),
    ("ex", """equity = (1 + r).cumprod()
# ✍️ maximum drawdown: min over time of equity / running maximum − 1
mdd = ...
mdd = p.check("maximum drawdown", mdd, p.max_drawdown(equity))""",
     """equity = (1 + r).cumprod()
mdd = (equity / equity.cummax() - 1).min()
mdd = p.check("maximum drawdown", mdd, p.max_drawdown(equity))"""),
    ("ex", """# ✍️ historical 99% VaR (the loss exceeded on 1% of days, as a positive number) and CVaR (average loss beyond it)
var99 = ...
cvar99 = ...
var99, cvar99 = p.check("VaR and CVaR (99%)", (var99, cvar99), p.hist_var_cvar(r, 0.99))
print(f"99% one-day VaR {var99:.2%}, CVaR {cvar99:.2%}")""",
     """var99 = -np.quantile(r, 0.01)
cvar99 = -r[r <= -var99].mean()
var99, cvar99 = p.check("VaR and CVaR (99%)", (var99, cvar99), p.hist_var_cvar(r, 0.99))
print(f"99% one-day VaR {var99:.2%}, CVaR {cvar99:.2%}")"""),
    ("md", "## Tear-sheet table for all tickers"),
    ("code", """table = pd.DataFrame({t: p.metrics(simple[t], RF, benchmark=simple["SPY"]) for t in simple}).T
table.style.format("{:.2f}").format("{:.1%}", subset=["CAGR", "Volatility", "Max drawdown", "VaR 99%", "CVaR 99%", "Alpha (ann.)"])"""),
    ("code", """TICKER = "AAPL"     # ✏️ change me
eq = (1 + simple[TICKER]).cumprod()
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
eq.plot(ax=axes[0], title=f"{TICKER}: growth of $1")
((eq / eq.cummax() - 1) * 100).plot(ax=axes[1], title="Drawdown (%)", color=p.PALETTE[1])
for ax in axes: ax.set_xlabel("")
plt.tight_layout(); plt.show()"""),
    ("md", "## VaR methods: historical vs normal vs Cornish–Fisher"),
    ("code", """from scipy import stats
mu, sd, s, k = r.mean(), r.std(), stats.skew(r), stats.kurtosis(r)
z = stats.norm.ppf(0.01)
zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
pd.Series({"historical": var99, "normal": -(mu + z * sd), "Cornish–Fisher": -(mu + zcf * sd)},
          name="99% one-day VaR").map("{:.2%}".format)"""),
    ("md", "## The annualization pitfall: smoothed returns inflate Sharpe"),
    ("code", """smoothed = 0.5 * r + 0.5 * r.shift(1)            # e.g. stale prices of an illiquid asset
for name, x in {"SPY": r, "smoothed SPY": smoothed.dropna()}.items():
    monthly = (1 + x).resample("ME").prod() - 1
    print(f"{name:<14} Sharpe from daily {p.sharpe(x, RF):.2f}  |  from monthly {p.sharpe(monthly, RF, periods=12):.2f}"
          f"  |  daily autocorrelation {x.autocorr():.2f}")"""),
    ("md", """## Questions
1. Which ticker has the best Sharpe but the worst drawdown? What does that say about using one metric?
2. When does the normal VaR underestimate risk the most?
3. Why does smoothing raise the daily-based Sharpe but not the monthly one?

**Graded (Part 2, B):** turn these formulas into a tested `metrics` module (see the lesson plan, Section 7)."""),
]


def build():
    (ROOT / "solutions").mkdir(exist_ok=True)
    for name, cells in NB.items():
        for kind in ("starter", "solution"):
            n = nbf.v4.new_notebook()
            n.metadata.update(KERNEL)
            out = []
            if kind == "solution":
                out.append(nbf.v4.new_markdown_cell("> **INSTRUCTOR SOLUTIONS** — do not share with learners before the session."))
            for c in cells:
                if c[0] == "md":
                    out.append(nbf.v4.new_markdown_cell(c[1]))
                elif c[0] == "code":
                    out.append(nbf.v4.new_code_cell(c[1]))
                else:
                    cell = nbf.v4.new_code_cell(c[1] if kind == "starter" else c[2])
                    cell.metadata["tags"] = ["exercise"]
                    out.append(cell)
            n.cells = out
            path = ROOT / (f"{name}.ipynb" if kind == "starter" else f"solutions/{name}_solution.ipynb")
            nbf.write(n, path)
    print(f"built {len(NB)} starter + {len(NB)} solution notebooks")


if __name__ == "__main__":
    build()
