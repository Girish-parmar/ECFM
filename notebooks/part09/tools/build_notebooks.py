"""Build the Part 9 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part09:  python tools/build_notebooks.py
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
for d in (Path.cwd(), Path.cwd().parent):       # p9lib.py is in notebooks/part09/
    sys.path.insert(0, str(d))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p9lib as p

p.use_course_style()"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 9 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_09_STATISTICAL_TRADING.md) · graded labs in [`labs/part09/`](../../labs/part09/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.
All data is synthetic with known parameters (true half-lives, true pairs, true hedge ratios, planted premia), so every estimator can be checked against the truth.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
NB["01_ou_bayes"] = [
    header("01", "Mean reversion, half-lives and Bayesian shrinkage", "S1 (Stochastic processes & Bayesian inference)",
           "1. Fit an Ornstein–Uhlenbeck process and read off its half-life.\n"
           "2. See how uncertain a half-life is with one year vs five years of data.\n"
           "3. Tell mean reversion from a random walk with the variance ratio.\n"
           "4. Shrink a noisy mean return toward a prior."),
    ("code", SETUP),
    ("md", "## 1. The Ornstein–Uhlenbeck process\n\n"
           "`dX = θ(μ − X)dt + σdW`: the further `X` is from `μ`, the harder it is pulled back. The **half-life** `ln 2 / θ` is how long a "
           "deviation takes to halve, the single most useful number in mean-reversion trading: it sets the look-back of the z-score and the "
           "time stop. Here are three paths with the same noise and different speeds."),
    ("code", """fig, ax = plt.subplots(figsize=(10, 3.6))
for hl in (3, 15, 80):
    ax.plot(p.simulate_ou(500, np.log(2) / hl, 0.0, 1.0, seed=1), lw=1, label=f"half-life {hl} bars")
ax.axhline(0, color="black", lw=0.6); ax.legend(); ax.set_title("OU paths: faster reversion, tighter band"); plt.show()"""),
    ("md", "## 2. Fitting it\n\n"
           "The exact discretization is an AR(1): `X[t+1] = a + b·X[t] + e`. Regress with `np.polyfit(x[:-1], x[1:], 1)` (it returns `b, a`), "
           "then `θ = −ln(b)`, `μ = a/(1 − b)`, `σ = std(residuals, ddof=2)·√(2θ/(1 − b²))`, `half_life = ln 2/θ`. If `b` is not in `(0, 1)` "
           "the series doesn't mean-revert: return `θ = 0`, `half_life = inf` and NaN for `μ` and `σ`."),
    ("md", YOUR_TURN),
    ("ex", """def fit_ou(x):
    x = np.asarray(x, dtype=float)
    b, a = np.polyfit(x[:-1], x[1:], 1)
    if not 0 < b < 1:
        return {"theta": 0.0, "mu": np.nan, "sigma": np.nan, "half_life": np.inf}
    resid = x[1:] - (a + b * x[:-1])
    theta = ...                                   # ✍️
    return {"theta": theta, "mu": ..., "sigma": ...,      # ✍️ mu and sigma
            "half_life": np.log(2) / theta}

x = p.simulate_ou(5000, np.log(2) / 7, 0.0, 1.0, seed=3)
rw = np.cumsum(np.random.default_rng(1).normal(size=2000))
mine = [p.attempt(fit_ou, x), p.attempt(fit_ou, rw)]
mine = p.check("fit_ou", mine, [p.fit_ou(x), p.fit_ou(rw)])
print(f"true half-life 7.0 → estimated {mine[0]['half_life']:.2f}; a random walk → {mine[1]['half_life']:.0f} bars")""",
     """def fit_ou(x):
    x = np.asarray(x, dtype=float)
    b, a = np.polyfit(x[:-1], x[1:], 1)
    if not 0 < b < 1:
        return {"theta": 0.0, "mu": np.nan, "sigma": np.nan, "half_life": np.inf}
    resid = x[1:] - (a + b * x[:-1])
    theta = -np.log(b)
    return {"theta": theta, "mu": a / (1 - b), "sigma": resid.std(ddof=2) * np.sqrt(2 * theta / (1 - b ** 2)),
            "half_life": np.log(2) / theta}

x = p.simulate_ou(5000, np.log(2) / 7, 0.0, 1.0, seed=3)
rw = np.cumsum(np.random.default_rng(1).normal(size=2000))
mine = [p.attempt(fit_ou, x), p.attempt(fit_ou, rw)]
mine = p.check("fit_ou", mine, [p.fit_ou(x), p.fit_ou(rw)])
print(f"true half-life 7.0 → estimated {mine[0]['half_life']:.2f}; a random walk → {mine[1]['half_life']:.0f} bars")"""),
    ("md", "The random walk's half-life comes out long but **finite**: in a finite sample the AR(1) slope is biased below 1, so every "
           "random walk looks a little mean-reverting. Always ask how uncertain the estimate is. A block bootstrap gives a 90% interval:"),
    ("code", """for years in (1, 5):
    xx = p.simulate_ou(252 * years, np.log(2) / 7, 0.0, 1.0, seed=4)
    lo, hi = p.half_life_ci(xx)
    print(f"{years} year(s) of data: half-life {p.fit_ou(xx)['half_life']:.1f}, 90% interval {lo:.1f} to {hi:.1f}  (true 7.0)")"""),
    ("md", "## 3. The variance ratio\n\n"
           "For a random walk, the variance of `q`-step changes is `q` times that of 1-step changes. Mean reversion makes long-horizon changes "
           "smaller than that; trending makes them larger. `VR(q) = Var(x[t+q] − x[t]) / (q · Var(x[t+1] − x[t]))` (ddof=1): about 1 for a "
           "random walk, below 1 for mean reversion, above 1 for a trend."),
    ("md", YOUR_TURN),
    ("ex", """def variance_ratio(x, q=5):
    x = np.asarray(x, dtype=float)
    return ...                                    # ✍️

trend = np.cumsum(p.simulate_ou(2000, np.log(2) / 5, 0.0, 1.0, seed=6))   # persistent increments: a trending series
cases = {"OU (half-life 7)": x, "random walk": rw, "trending": trend}
mine = {k: p.attempt(variance_ratio, v) for k, v in cases.items()}
mine = p.check("variance_ratio", mine, {k: p.variance_ratio(v) for k, v in cases.items()})
pd.Series(mine).round(3)""",
     """def variance_ratio(x, q=5):
    x = np.asarray(x, dtype=float)
    return float(np.var(x[q:] - x[:-q], ddof=1) / (q * np.var(np.diff(x), ddof=1)))

trend = np.cumsum(p.simulate_ou(2000, np.log(2) / 5, 0.0, 1.0, seed=6))   # persistent increments: a trending series
cases = {"OU (half-life 7)": x, "random walk": rw, "trending": trend}
mine = {k: p.attempt(variance_ratio, v) for k, v in cases.items()}
mine = p.check("variance_ratio", mine, {k: p.variance_ratio(v) for k, v in cases.items()})
pd.Series(mine).round(3)"""),
    ("md", "## 4. Bayesian shrinkage of a mean return\n\n"
           "One year of daily returns says very little about the true mean. With a normal prior (mean `m0`, sd `s0`) and the data variance "
           "treated as known (`s² = var(r, ddof=1)`), the posterior is normal with\n\n"
           "`precision = 1/s0² + n/s²` and `mean = (m0/s0² + n·x̄/s²) / precision`, sd `= √(1/precision)`.\n\n"
           "The posterior mean sits between the prior and the sample mean, closer to whichever is more precise."),
    ("md", YOUR_TURN),
    ("ex", """def normal_posterior_mean(returns, prior_mean=0.0, prior_sd=0.001):
    r = np.asarray(returns, dtype=float)
    n, s2 = r.size, r.var(ddof=1)
    prec = ...                                    # ✍️
    return float(...), float(np.sqrt(1 / prec))   # ✍️ the posterior mean

rng = np.random.default_rng(2)
one_year, ten_years = rng.normal(0.0004, 0.01, 252), rng.normal(0.0004, 0.01, 2520)
cases = [(one_year,), (ten_years,), (one_year, 0.0, 0.01)]
mine = [p.attempt(normal_posterior_mean, *c) for c in cases]
mine = p.check("normal_posterior_mean", mine, [p.normal_posterior_mean(*c) for c in cases])
pd.DataFrame({"sample mean (bp/day)": [c[0].mean() * 1e4 for c in cases], "posterior mean": [m * 1e4 for m, _ in mine],
              "posterior sd": [s * 1e4 for _, s in mine]}, index=["1 year, tight prior", "10 years, tight prior", "1 year, vague prior"]).round(2)""",
     """def normal_posterior_mean(returns, prior_mean=0.0, prior_sd=0.001):
    r = np.asarray(returns, dtype=float)
    n, s2 = r.size, r.var(ddof=1)
    prec = 1 / prior_sd ** 2 + n / s2
    return float((prior_mean / prior_sd ** 2 + n * r.mean() / s2) / prec), float(np.sqrt(1 / prec))

rng = np.random.default_rng(2)
one_year, ten_years = rng.normal(0.0004, 0.01, 252), rng.normal(0.0004, 0.01, 2520)
cases = [(one_year,), (ten_years,), (one_year, 0.0, 0.01)]
mine = [p.attempt(normal_posterior_mean, *c) for c in cases]
mine = p.check("normal_posterior_mean", mine, [p.normal_posterior_mean(*c) for c in cases])
pd.DataFrame({"sample mean (bp/day)": [c[0].mean() * 1e4 for c in cases], "posterior mean": [m * 1e4 for m, _ in mine],
              "posterior sd": [s * 1e4 for _, s in mine]}, index=["1 year, tight prior", "10 years, tight prior", "1 year, vague prior"]).round(2)"""),
    ("md", "The true mean is 4 bp a day, but with 1% daily volatility even ten years pin it down only to about ±2 bp (this sample "
           "says 0.8). A skeptical prior (\"most edges are near zero\", sd 10 bp) shrinks one year's estimate by about a quarter "
           "(3.0 → 2.2 bp) and ten years' by only a few percent: the more data, the less the prior matters. With a vague prior "
           "(sd 100 bp), the posterior is simply the sample mean.\n\n"
           "## Wrap-up\n\n"
           "* The half-life is the unit of time for everything mean-reverting; estimate it with an interval.\n"
           "* Variance ratios and half-lives are diagnostics, noisy on short samples.\n"
           "* Shrink noisy estimates toward a sensible prior.\n"
           "* Graded version: `labs/part09/week31_pairs` (OU, variance ratio, posterior, block-bootstrap half-life CI)."),
]

# ---------------------------------------------------------------------------------------------- 02
NB["02_cointegration_screen"] = [
    header("02", "Cointegration and screening for pairs", "S2 (Cointegration & pair selection)",
           "1. See why correlated returns don't make a pair.\n"
           "2. Run the Engle–Granger test and recover a known hedge ratio.\n"
           "3. Control false discoveries across many candidate pairs.\n"
           "4. Screen a universe within sectors, and see what screening everything costs."),
    ("code", SETUP),
    ("md", "## 1. Correlation is not cointegration\n\n"
           "Two stocks can move together day by day (correlated **returns**) and still drift apart forever. A pair trade needs the **prices** "
           "to share a stationary spread: cointegration."),
    ("code", """rng = np.random.default_rng(5)
common = np.cumsum(rng.normal(0, 0.01, 1500))
a = pd.Series(4 + common + np.cumsum(rng.normal(0, 0.006, 1500)))
b = pd.Series(4 + common + np.cumsum(rng.normal(0, 0.006, 1500)))
pair = p.cointegrated_pair()                          # y = 0.5 + 1.5·x + an OU spread with half-life 7
print(f"correlated random walks: return correlation {np.corrcoef(np.diff(a), np.diff(b))[0, 1]:.2f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].plot(a.to_numpy() - b.to_numpy()); axes[0].set_title("spread of two correlated random walks: wanders off")
axes[1].plot((pair.y - 1.5 * pair.x).to_numpy()); axes[1].set_title("spread of a cointegrated pair: comes back")
plt.tight_layout(); plt.show()"""),
    ("md", "## 2. Engle–Granger\n\n"
           "Regress `y = α + β·x` by OLS (`sm.OLS(y, sm.add_constant(x)).fit()`, whose `params` are `[α, β]`), form the spread "
           "`y − β·x − α`, and test it for a unit root with the cointegration critical values (`statsmodels`' `coint(y, x)[1]` is the "
           "p-value). Add the spread's OU half-life. Return a dict with `alpha`, `beta`, `pvalue`, `spread`, `half_life`."),
    ("md", YOUR_TURN),
    ("ex", """import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def engle_granger(y, x):
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    alpha, beta = float(ols.params.iloc[0]), float(ols.params.iloc[1])
    spread = ...                                  # ✍️
    return {"alpha": alpha, "beta": beta, "pvalue": float(coint(y, x)[1]), "spread": spread,
            "half_life": p.fit_ou(spread.to_numpy())["half_life"]}

mine = [p.attempt(engle_granger, pair.y, pair.x), p.attempt(engle_granger, a, b)]
ref = [p.engle_granger(pair.y, pair.x), p.engle_granger(a, b)]
mine = p.check("engle_granger", [{k: v for k, v in m.items() if k != "spread"} if m is not Ellipsis else m for m in mine],
               [{k: v for k, v in r.items() if k != "spread"} for r in ref])
pd.DataFrame(mine, index=["cointegrated pair (β 1.5, half-life 7)", "correlated random walks"]).round(4)""",
     """import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def engle_granger(y, x):
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    alpha, beta = float(ols.params.iloc[0]), float(ols.params.iloc[1])
    spread = y - beta * x - alpha
    return {"alpha": alpha, "beta": beta, "pvalue": float(coint(y, x)[1]), "spread": spread,
            "half_life": p.fit_ou(spread.to_numpy())["half_life"]}

mine = [p.attempt(engle_granger, pair.y, pair.x), p.attempt(engle_granger, a, b)]
ref = [p.engle_granger(pair.y, pair.x), p.engle_granger(a, b)]
mine = p.check("engle_granger", [{k: v for k, v in m.items() if k != "spread"} if m is not Ellipsis else m for m in mine],
               [{k: v for k, v in r.items() if k != "spread"} for r in ref])
pd.DataFrame(mine, index=["cointegrated pair (β 1.5, half-life 7)", "correlated random walks"]).round(4)"""),
    ("md", "## 3. Many pairs, many tests\n\n"
           "A sector of 8 stocks has 28 pairs; 500 stocks have 124,750. At a 5% threshold, one in twenty unrelated pairs passes by luck. "
           "**Benjamini–Hochberg** controls the false discovery rate: sort the p-values, find the **largest** `k` with `p_(k) <= k·q/m`, and "
           "reject the `k` smallest. Return a boolean array in the input order."),
    ("md", YOUR_TURN),
    ("ex", """def bh_reject(pvalues, q=0.05):
    pv = np.asarray(pvalues, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    ok = ...                                      # ✍️ sorted p-values <= rank·q/m, rank = 1..m
    out = np.zeros(m, dtype=bool)
    if ok.any():
        k = np.max(np.flatnonzero(ok))
        out[order[:k + 1]] = True
    return out

cases = [[0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216], list(np.random.default_rng(0).uniform(size=50)) + [1e-5]]
mine = [p.attempt(bh_reject, c) for c in cases]
mine = p.check("bh_reject", mine, [p.bh_reject(c) for c in cases])
print(f"case 1: {int(np.sum(mine[0]))} discoveries ({int(np.sum(np.array(cases[0]) < 0.05))} below 0.05);  "
      f"case 2: {int(np.sum(mine[1]))} discovery among 51 tests ({int(np.sum(np.array(cases[1]) < 0.05))} below 0.05)")""",
     """def bh_reject(pvalues, q=0.05):
    pv = np.asarray(pvalues, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    ok = pv[order] <= np.arange(1, m + 1) * q / m
    out = np.zeros(m, dtype=bool)
    if ok.any():
        k = np.max(np.flatnonzero(ok))
        out[order[:k + 1]] = True
    return out

cases = [[0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216], list(np.random.default_rng(0).uniform(size=50)) + [1e-5]]
mine = [p.attempt(bh_reject, c) for c in cases]
mine = p.check("bh_reject", mine, [p.bh_reject(c) for c in cases])
print(f"case 1: {int(np.sum(mine[0]))} discoveries ({int(np.sum(np.array(cases[0]) < 0.05))} below 0.05);  "
      f"case 2: {int(np.sum(mine[1]))} discovery among 51 tests ({int(np.sum(np.array(cases[1]) < 0.05))} below 0.05)")"""),
    ("md", "## 4. Screen a universe\n\n"
           "Four sectors of eight stocks (a market factor, a sector factor and noise); in each sector **one** pair is truly cointegrated. "
           "Screen **within sectors** first (economic link), test every pair with Engle–Granger, apply BH-FDR, then keep half-lives between 2 "
           "and 30 days (`p.screen_pairs`)."),
    ("code", """prices, sectors, truth = p.sector_universe()
screen = p.screen_pairs(prices, sectors)
print("true pairs:", truth)
display(screen.head(8).round(4))
print(f"{len(screen)} pairs tested within sectors: {(screen.pvalue < 0.05).sum()} with raw p < 0.05, {screen.fdr_pass.sum()} pass FDR, "
      f"{screen.selected.sum()} selected")"""),
    ("code", """everything = p.screen_pairs(prices)               # no economic link: all 496 pairs
found = [tuple(r) for r in everything[everything.fdr_pass][["a", "b"]].to_numpy()]
print(f"{len(everything)} pairs tested across the whole universe: {(everything.pvalue < 0.05).sum()} raw p < 0.05, "
      f"{everything.fdr_pass.sum()} pass FDR; true pairs missed: {[t for t in truth if t not in found]}")"""),
    ("md", "Within sectors, all four true pairs pass and nothing else does. Screening everything quadruples the number of tests, so FDR must "
           "be stricter, and one true pair is lost. The economic prefilter isn't just a shortcut: it buys statistical power.\n\n"
           "## Wrap-up\n\n"
           "* Cointegration of prices, not correlation of returns.\n"
           "* Economic link first, then Engle–Granger (or Johansen for baskets), then FDR, then a half-life filter.\n"
           "* Graded version: `labs/part09/week31_pairs` (Johansen too) and Clinic W1 (screen, then trade out of sample)."),
]

# ---------------------------------------------------------------------------------------------- 03
_ZCHECK = """
cases = [(spread, W), ([1.0, 2.0, 4.0, 3.0, 5.0], 3)]
mine = [p.attempt(rolling_zscore, *c) for c in cases]
mine = p.check("rolling_zscore", mine, [p.rolling_zscore(*c) for c in cases])
z = mine[0]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(spread.index[F:], z[F:], lw=1)
for lvl, c in ((2, "#eb6834"), (0.5, "#1baf7a"), (4, "#e34948")):
    ax.axhline(lvl, color=c, lw=0.8, ls="--"); ax.axhline(-lvl, color=c, lw=0.8, ls="--")
ax.set_title("trading period: z-score with entry ±2, exit ±0.5, stop ±4"); plt.show()"""

_POS_HEAD = """def pairs_positions(z, entry=2.0, exit_=0.5, stop=4.0, max_hold=None):
    z = np.asarray(z, dtype=float)
    pos, held, blocked = np.zeros(z.size), 0, False
    for t in range(z.size):
        cur = pos[t - 1] if t else 0.0
        if np.isnan(z[t]):
            pos[t] = cur                                  # no z-score yet: stay as you are
            continue
        if cur == 0:
            blocked = blocked and abs(z[t]) >= entry      # after a stop, wait until |z| is back below entry
"""
_POS_TAIL = """        pos[t] = cur
    return pos

toy = [np.nan, 0.3, 2.5, 1.8, 0.4, -2.2, -4.5, -3.0, -1.5, -2.5, 0.1]
slow = [2.5, 2.4, 2.3, 2.2, 2.1, 2.6, 1.0, 2.5]
cases = [(toy,), (slow, 2.0, 0.5, 4.0, 3), (z, 2.0, 0.5, 4.0, W)]
mine = [p.attempt(pairs_positions, *c) for c in cases]
mine = p.check("pairs_positions", mine, [p.pairs_positions(*c) for c in cases])
display(pd.DataFrame({"z": toy, "position": mine[0]}).T)
display(pd.DataFrame({"z": slow, "position (time stop 3 bars)": mine[1]}).T)"""

_PNL_HEAD = """def pair_pnl(pos, y, x, beta, cost_bps=5.0, borrow_bps_year=50.0):
    pos = np.asarray(pos, dtype=float)
    dy, dx = np.diff(np.asarray(y, float), prepend=np.nan), np.diff(np.asarray(x, float), prepend=np.nan)
    held = np.concatenate([[0.0], pos[:-1]])              # decided at yesterday's close, held today
    beta = np.asarray(beta, dtype=float)
    gross = 1 + np.abs(beta)
"""
_PNL_TAIL = """    trade_cost = np.abs(np.diff(pos, prepend=0.0)) * cost_bps / 1e4 * 2
    trade_cost = np.concatenate([[0.0], trade_cost[:-1]])  # the trade fills on the next bar
    return ret - trade_cost - short_w * borrow_bps_year / 1e4 / 252

pos = mine[2]
cases = [(pos, pair.y, pair.x, eg["beta"]), (pos, pair.y, pair.x, eg["beta"], 0.0, 300.0),
         ([0, 1, 1, -1, 0], [0, 0.01, 0.03, 0.02, 0.02], [0, 0.01, 0.01, 0.0, 0.01], 1.0, 10.0, 252.0)]
mine = [p.attempt(pair_pnl, *c) for c in cases]
mine = p.check("pair_pnl", mine, [p.pair_pnl(*c) for c in cases])
pnl = mine[0][F:]
tr = pos[F - 1:]
print(f"out of sample ({len(pnl)} days): Sharpe {p.sharpe(pnl):.2f}, total {pnl.sum():.1%}, "
      f"{int(((tr[1:] != 0) & (tr[1:] != tr[:-1])).sum())} trades, in a position {np.mean(pos[F:] != 0):.0%} of the days")
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(pair.index[F:], np.cumsum(pnl)); ax.set_title("cumulative P&L of the pair, out of sample (5 bp per leg, 50 bp/yr borrow)"); plt.show()"""

NB["03_pairs_strategy"] = [
    header("03", "The pairs strategy, end to end", "S3 (The pairs strategy, end to end)",
           "1. Build a z-score whose look-back is tied to the half-life.\n"
           "2. Write the pairs state machine: entry, exit, stop, time stop and no re-entry after a stop.\n"
           "3. Compute the P&L of both legs, with trading costs and borrow fees.\n"
           "4. Trade out of sample, then watch what a dying relationship does to a rolling and a fixed z-score."),
    ("code", SETUP),
    ("md", "## 1. Formation, then trading\n\n"
           "Estimate everything on a **formation** period (here the first 750 days) and trade only after it. The spread uses the "
           "formation hedge ratio; the z-score look-back is about three half-lives, long enough to estimate a mean and a volatility, "
           "short enough to follow slow drift."),
    ("code", """pair = p.cointegrated_pair()                       # true β 1.5, spread half-life 7 days
F = 750
eg = p.engle_granger(pair.y.iloc[:F], pair.x.iloc[:F])
hl = eg["half_life"]
W = round(3 * hl)
spread = pair.y - eg["beta"] * pair.x - eg["alpha"]
print(f"formation: β {eg['beta']:.3f}, p-value {eg['pvalue']:.1e}, half-life {hl:.1f} days → z look-back {W} days")"""),
    ("md", "## 2. The z-score\n\n"
           "`z = (s − rolling mean) / rolling std` over the last `window` values **including today** (std with ddof=1, the pandas "
           "default), NaN during the warm-up. `pd.Series.rolling(window)` gives you both the mean and the std."),
    ("md", YOUR_TURN),
    ("ex", """def rolling_zscore(spread, window):
    s = pd.Series(np.asarray(spread, dtype=float))
    r = s.rolling(window)
    return (...).to_numpy()                               # ✍️
""" + _ZCHECK,
     """def rolling_zscore(spread, window):
    s = pd.Series(np.asarray(spread, dtype=float))
    r = s.rolling(window)
    return ((s - r.mean()) / r.std()).to_numpy()
""" + _ZCHECK),
    ("md", "## 3. The state machine\n\n"
           "`+1` is long the spread (long `y`, short `β·x`), `−1` short it. Decisions are made at the close and filled on the next bar.\n\n"
           "* **Flat:** enter `−sign(z)` when `entry < |z| < stop` (beyond the stop, the relationship may already be breaking).\n"
           "* **In a trade** (count `held` bars after the entry bar): exit when `|z| < exit_`. **Stop out** when `|z| > stop` or, with a "
           "time stop, when `held >= max_hold`.\n"
           "* After a stop or time stop, **don't re-enter** until `|z|` has come back below `entry`: a pair that just stopped you out is "
           "not a fresh signal.\n"
           "* A NaN z-score (warm-up) leaves the position unchanged."),
    ("md", YOUR_TURN),
    ("ex", _POS_HEAD + """            if not blocked and ...:                       # ✍️ the entry condition
                cur, held = -np.sign(z[t]), 0
        else:
            held += 1
            if ...:                                       # ✍️ the normal exit
                cur = 0.0
            elif ...:                                     # ✍️ the stop or the time stop
                cur, blocked = 0.0, True
""" + _POS_TAIL,
     _POS_HEAD + """            if not blocked and entry < abs(z[t]) < stop:
                cur, held = -np.sign(z[t]), 0
        else:
            held += 1
            if abs(z[t]) < exit_:
                cur = 0.0
            elif abs(z[t]) > stop or (max_hold is not None and held >= max_hold):
                cur, blocked = 0.0, True
""" + _POS_TAIL),
    ("md", "In the first table, the stop at `z = −4.5` blocks the next bar (`−3.0` is still beyond 2); `−1.5` clears the block and "
           "`−2.5` is a fresh entry. In the second, the time stop closes the trade after 3 bars and nothing re-enters until `|z|` has "
           "dipped below 2.\n\n"
           "## 4. P&L with both legs, costs and borrow\n\n"
           "Work in **log prices**, so leg returns are `Δy` and `Δx`. A unit spread position holds 1 of `y` against `β` of `x`: gross "
           "capital `1 + |β|`. The position decided yesterday earns\n\n"
           "`ret = held · (Δy − β·Δx) / (1 + |β|)`\n\n"
           "(use `np.nan_to_num`: the first `Δ` is NaN). Costs: `|Δpos| · cost_bps/1e4 · 2` legs, charged on the bar the trade fills. "
           "Borrow: while you hold a position you pay `borrow_bps_year/1e4/252` a day on the **short** leg, whose weight is "
           "`1/(1 + |β|)` when short the spread (short `y`) and `|β|/(1 + |β|)` when long it (short `x`)."),
    ("md", YOUR_TURN),
    ("ex", _PNL_HEAD + """    ret = np.nan_to_num(...)                              # ✍️ the spread return on capital 1 + |β|
    short_w = ...                                         # ✍️ weight of the short leg (0 when flat)
""" + _PNL_TAIL,
     _PNL_HEAD + """    ret = np.nan_to_num(held * (dy - beta * dx) / gross)
    short_w = np.where(held < 0, 1 / gross, np.where(held > 0, np.abs(beta) / gross, 0.0))
""" + _PNL_TAIL),
    ("md", "## 5. Costs decide\n\n"
           "Every round trip pays four fills (two legs, in and out). A fast-reverting pair trades often, so the cost per fill matters "
           "far more than the borrow fee:"),
    ("code", """grid = {c: {b: p.sharpe(p.pair_pnl(pos, pair.y, pair.x, eg["beta"], c, b)[F:]) for b in (0, 50, 300)} for c in (0, 5, 10, 20)}
pd.DataFrame(grid).T.rename_axis("cost per leg (bp) ↓ · borrow (bp/yr) →").round(2)"""),
    ("md", "Sharpe 1.9 before costs, 1.2 at 5 bp a leg, 0.5 at 10 bp and negative at 20 bp. Even a hard-to-borrow fee of 300 bp a year "
           "costs only about 0.1 of Sharpe here.\n\n"
           "## 6. When the relationship dies\n\n"
           "`p.breaking_pair()` is cointegrated for 900 days, then its spread becomes a random walk (a merger, a new business line). "
           "Trade it from day 750 with two yardsticks: the **rolling** z-score, and a **formation** z-score that keeps the formation "
           "mean and std fixed. Try both with and without a time stop of three half-lives."),
    ("code", """bp = p.breaking_pair()
eb = p.engle_granger(bp.y.iloc[:F], bp.x.iloc[:F])
Wb = round(3 * eb["half_life"])
sb = bp.y - eb["beta"] * bp.x - eb["alpha"]
yardsticks = {"rolling z": p.rolling_zscore(sb, Wb),
              "formation z": ((sb - sb.iloc[:F].mean()) / sb.iloc[:F].std()).to_numpy()}
rows = {}
for name, zz in yardsticks.items():
    for mh in (None, Wb):
        pl = p.pair_pnl(p.pairs_positions(zz, max_hold=mh), bp.y, bp.x, eb["beta"])
        rows[(name, "no time stop" if mh is None else f"time stop {mh} bars")] = {
            "P&L 750–900 (healthy)": pl[F:900].sum(), "P&L after break": pl[900:].sum(),
            "max |z| after break": np.nanmax(np.abs(zz[900:]))}
display(pd.DataFrame(rows).T.round(3))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharex=True)
for ax, (name, zz) in zip(axes, yardsticks.items()):
    ax.plot(bp.index[F:], zz[F:], lw=0.8); ax.axvline(bp.index[900], color="black", lw=1)
    ax.axhline(4, color="#e34948", ls="--", lw=0.8); ax.axhline(-4, color="#e34948", ls="--", lw=0.8); ax.set_title(name)
    ax.tick_params(axis="x", labelrotation=30)
plt.tight_layout(); plt.show()"""),
    ("md", "The rolling z-score **re-scales** to the wider spread: after the break it never reaches 4, so the stop never fires and the "
           "strategy keeps trading a dead pair. With a time stop it does worse, because every time stop is followed by a fresh "
           "entry. The formation z-score sees the spread run away (|z| up to 16): the stop fires and blocks re-entry, and with a "
           "time stop the post-break loss disappears. Stops need a fixed yardstick, and even then they are a last line of defence: "
           "notebook 07 **monitors** the relationship and retires the pair.\n\n"
           "## Wrap-up\n\n"
           "* Formation first, then trade; never estimate on the data you trade.\n"
           "* Look-back ≈ 3 half-lives; time stop ≈ 3 half-lives; no re-entry after a stop.\n"
           "* Two legs, in and out: costs dominate borrow for a fast pair.\n"
           "* A rolling z-score hides a break.\n"
           "* Graded version: `labs/part09/week31_pairs` (`rolling_zscore`, `pairs_positions`, `pair_pnl`) and Clinic W1."),
]

# ---------------------------------------------------------------------------------------------- 04
_ROLS_CHECK = """
d = p.drifting_beta_pair()                                # true β moves from 1.0 to 2.0 over 2,000 days
cases = [(d.y, d.x, 250), (d.y, d.x, 60)]
mine = [p.attempt(rolling_ols_beta, *c) for c in cases]
mine = p.check("rolling_ols_beta", mine, [p.rolling_ols_beta(*c) for c in cases])
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(d.index, d.beta, color="black", lw=2, label="true β")
ax.plot(d.index, mine[0], lw=1, label="rolling OLS, 250 days"); ax.plot(d.index, mine[1], lw=0.8, label="rolling OLS, 60 days")
ax.set_ylim(-1, 4); ax.legend(); ax.set_title("rolling OLS: long wrong swings, or noise"); plt.show()"""

_KF_HEAD = """def kalman_hedge(y, x, delta=1e-4, r=1e-3):
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = y.size
    theta, P = np.zeros(2), np.eye(2)                     # the state (α, β) and its covariance
    Vw = delta / (1 - delta) * np.eye(2)                  # how far (α, β) may move in one bar
    beta, alpha, e, q = (np.empty(n) for _ in range(4))
    for t in range(n):
        F = np.array([1.0, x[t]])
        P = P + Vw                                        # predict: the state may have moved
"""
_KF_TAIL = """        theta = theta + K * e[t]                          # update
        P = P - np.outer(K, F) @ P
        alpha[t], beta[t] = theta
    return beta, alpha, e, q

cases = [(d.y, d.x), (d.y, d.x, 1e-6, 0.25)]
mine = [p.attempt(kalman_hedge, *c) for c in cases]
mine = p.check("kalman_hedge", mine, [p.kalman_hedge(*c) for c in cases])
kb = mine[0][0]
print(f"bar 200: true β {d.beta.iloc[200]:.3f}, Kalman {kb[200]:.3f};  last bar: true {d.beta.iloc[-1]:.3f}, Kalman {kb[-1]:.3f}")"""

NB["04_kalman_hedge"] = [
    header("04", "A dynamic hedge ratio with the Kalman filter", "S4 (Dynamic hedge ratio with the Kalman filter)",
           "1. See a rolling-OLS hedge ratio lag (or jitter around) a drifting one.\n"
           "2. Write the Kalman filter for a regression whose coefficients move.\n"
           "3. Compare both with the true β.\n"
           "4. See a filter that adapts too fast swallow the spread it should trade."),
    ("code", SETUP),
    ("md", "## 1. Hedge ratios drift\n\n"
           "Business mix, index membership and leverage change, so the hedge ratio does too. The simplest adaptive estimate is a "
           "**rolling OLS** β over the last `window` bars (including today): `cov(y, x) / var(x)` over the window, which is exactly "
           "the OLS slope with an intercept. Pandas has `rolling(window).cov(other)` and `rolling(window).var()`."),
    ("md", YOUR_TURN),
    ("ex", """def rolling_ols_beta(y, x, window):
    y, x = pd.Series(np.asarray(y, float)), pd.Series(np.asarray(x, float))
    return (...).to_numpy()                               # ✍️
""" + _ROLS_CHECK,
     """def rolling_ols_beta(y, x, window):
    y, x = pd.Series(np.asarray(y, float)), pd.Series(np.asarray(x, float))
    return (y.rolling(window).cov(x) / x.rolling(window).var()).to_numpy()
""" + _ROLS_CHECK),
    ("md", "Neither window works. Within a window `x` wanders only a little, so the slope is poorly identified: a stretch where `x` "
           "and the spread noise happen to move together sends the 250-day estimate far off for months. The 60-day window reacts "
           "faster but jumps around from week to week.\n\n"
           "## 2. The Kalman filter\n\n"
           "Treat `(α_t, β_t)` as a hidden state that follows a random walk, and `y_t = α_t + β_t·x_t + e_t` as a noisy observation of "
           "it. Each bar, with `F = [1, x_t]`:\n\n"
           "1. **Predict:** `P = P + Vw` (the state may have moved; `Vw = δ/(1 − δ)·I`).\n"
           "2. **Forecast error**, *before* the update: `e_t = y_t − F·θ`. Its variance is `q_t = F P Fᵀ + r`.\n"
           "3. **Gain:** `K = P Fᵀ / q_t`.\n"
           "4. **Update:** `θ = θ + K e_t`, `P = P − K F P`.\n\n"
           "`e_t` is the spread as seen with yesterday's hedge, and `e_t / √q_t` is a z-score that scales itself."),
    ("md", YOUR_TURN),
    ("ex", _KF_HEAD + """        q[t] = ...                                        # ✍️ variance of the forecast error
        e[t] = ...                                        # ✍️ forecast error, before the update
        K = ...                                           # ✍️ Kalman gain
""" + _KF_TAIL,
     _KF_HEAD + """        q[t] = F @ P @ F + r
        e[t] = y[t] - F @ theta
        K = P @ F / q[t]
""" + _KF_TAIL),
    ("code", """ests = {"Kalman (δ 1e-4)": kb, "rolling OLS 60": p.rolling_ols_beta(d.y, d.x, 60),
        "rolling OLS 250": p.rolling_ols_beta(d.y, d.x, 250), "rolling OLS 500": p.rolling_ols_beta(d.y, d.x, 500)}
err = pd.Series({k: np.nanmean(np.abs(v - d.beta.to_numpy())[500:]) for k, v in ests.items()}, name="mean |β error| after bar 500")
display(err.round(3).to_frame())
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(d.index, d.beta, color="black", lw=2, label="true β")
ax.plot(d.index, kb, lw=1, label="Kalman"); ax.plot(d.index, ests["rolling OLS 250"], lw=1, label="rolling OLS 250")
ax.set_ylim(0, 3); ax.legend(); ax.set_title("the filter tracks the drifting hedge ratio"); plt.show()"""),
    ("md", "The filter stays within a few hundredths of the true β all the way from 1 to 2; rolling OLS is about ten times further "
           "off, whatever the window.\n\n"
           "## 3. Too fast, and the hedge eats the signal\n\n"
           "`δ` sets how fast β may move. On a pair whose true β is **constant** (1.5), trade the filter's own z-score `e/√q` with "
           "`pairs_positions`, hedging each day with **yesterday's** β, over the same out-of-sample period as notebook 03. The "
           "observation noise `r` is the variance of the formation spread."),
    ("code", """pair = p.cointegrated_pair()
F = 750
r = float(p.engle_granger(pair.y.iloc[:F], pair.x.iloc[:F])["spread"].var())
rows = {}
for delta in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
    b, _, e, q = p.kalman_hedge(pair.y, pair.x, delta, r)
    zk = e / np.sqrt(q)
    pos = p.pairs_positions(zk)
    pl = p.pair_pnl(pos, pair.y, pair.x, np.r_[b[0], b[:-1]])[F:]          # yesterday's hedge: no look-ahead
    tr = pos[F - 1:]
    rows[f"δ = {delta:g}"] = {"std of z": np.std(zk[F:]), "max |z|": np.abs(zk[F:]).max(),
                              "trades": int(((tr[1:] != 0) & (tr[1:] != tr[:-1])).sum()), "OOS Sharpe": p.sharpe(pl)}
pd.DataFrame(rows).T.round(2)"""),
    ("md", "With a tiny `δ` the hedge barely moves and `e/√q` is close to the spread divided by its formation volatility: few trades, "
           "but good ones. As `δ` grows, β chases every wiggle of `y`, the forecast error shrinks and so does the z-score: from "
           "`δ = 1e-6`, `|z|` never reaches 2 and nothing trades (NaN Sharpe). The filter has explained the spread away. A drifting relationship needs a "
           "larger `δ` than a stable one; choose it by walk-forward (homework), never on the test period.\n\n"
           "## Wrap-up\n\n"
           "* Rolling OLS trades lag for noise; the Kalman filter adapts with far less of both when β really drifts.\n"
           "* Use the forecast error with yesterday's hedge, and lag the hedge in the P&L.\n"
           "* Too large a `δ` absorbs the spread (lesson plan, common mistake 8).\n"
           "* Graded version: `labs/part09/week31_pairs` (`kalman_hedge`, `rolling_ols_beta`); Clinic W1 compares fixed, rolling and "
           "Kalman hedges out of sample."),
]

# ---------------------------------------------------------------------------------------------- 05
_SS_HEAD = """def s_score(X, max_half_life=None):
    X = np.asarray(X, dtype=float)
    b, a = np.polyfit(X[:-1], X[1:], 1)                   # AR(1): X[t+1] = a + b·X[t] + e
    if not 0 < b < 1 or (max_half_life is not None and np.log(2) / -np.log(b) > max_half_life):
        return np.nan                                     # not reverting (fast enough): no s-score
    resid = X[1:] - (a + b * X[:-1])
"""
_SS_TAIL = """    return float((X[-1] - mu) / sig_eq)

mine = {mh: pd.Series({c: p.attempt(s_score, X[c], mh) for c in X.columns}) for mh in (None, 10)}
mine = p.check("s_score", mine, {mh: p.s_scores(R, k, window, mh).reindex(R.columns) for mh in (None, 10)})
s = mine[10].dropna()
print(f"{mine[None].notna().sum()} stocks have an AR(1) slope in (0, 1); {len(s)} also have a half-life of 10 days or less")
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.bar(s.index, s.to_numpy(), color=np.where(s.abs() > 1.25, "#eb6834", "#8a8984"))
ax.axhline(1.25, color="#eb6834", ls="--", lw=0.8); ax.axhline(-1.25, color="#eb6834", ls="--", lw=0.8)
ax.tick_params(axis="x", labelrotation=90, labelsize=7); ax.set_title("s-scores today: beyond ±1.25 opens a trade"); plt.show()"""

_SP_HEAD = """def s_score_positions(s, prev, open_=1.25, close_long=0.5, close_short=0.75):
    out = {}
    for name, cur in prev.items():
        v = s.get(name, np.nan)
        if np.isnan(v):
            out[name] = 0.0                               # no s-score today (not reverting): close
"""
_SP_TAIL = """        else:
            out[name] = 0.0 if v < close_short else -1.0
    return pd.Series(out, dtype=float)

toy_s = pd.Series({"A": -1.5, "B": 1.4, "C": -0.3, "D": 0.6, "E": 0.9, "F": -0.8})
toy_prev = pd.Series({"A": 0.0, "B": 0.0, "C": 1.0, "D": -1.0, "E": -1.0, "F": 1.0, "G": 1.0})
cases = [(toy_s, toy_prev), (s, pd.Series(0.0, index=R.columns))]
mine = [p.attempt(s_score_positions, *c) for c in cases]
mine = p.check("s_score_positions", mine, [p.s_score_positions(*c) for c in cases])
display(pd.DataFrame({"s": toy_s, "yesterday": toy_prev, "today": mine[0]}).T)
print(f"today, from flat: {int((mine[1] > 0).sum())} longs and {int((mine[1] < 0).sum())} shorts")"""

NB["05_pca_residual"] = [
    header("05", "PCA residual statistical arbitrage", "S5 (PCA / residual statistical arbitrage)",
           "1. Extract statistical factors with PCA.\n"
           "2. Turn each stock's cumulative residual into an Avellaneda–Lee s-score.\n"
           "3. Write the open and close rules.\n"
           "4. Backtest a dollar-neutral residual book, with and without residual reversion in the data."),
    ("code", SETUP),
    ("md", "## 1. From pairs to portfolios\n\n"
           "A pair hedges one stock with another. Residual stat-arb hedges each stock with a few **common factors** and trades what is "
           "left. `p.factor_universe()` has 60 stocks driven by 3 factors; for half of them (`reverting`), the idiosyncratic price has a "
           "small OU component with a 3-day half-life, hidden under daily noise a little larger than its own moves. The other half is pure noise. PCA of the "
           "correlation matrix finds the factors:"),
    ("code", """R, reverting = p.factor_universe()
ev = np.linalg.eigvalsh(np.corrcoef(R.to_numpy().T))[::-1]
share = ev / ev.sum()
print(f"{R.shape[1]} stocks × {R.shape[0]} days; variance explained by the first 5 components: {np.round(share[:5], 3)}")
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(1, 16), share[:15]); ax.set_title("scree plot: one market factor, two small ones, then noise"); plt.show()"""),
    ("md", "## 2. Residuals and the s-score\n\n"
           "Over the last 60 days: standardize the returns, take the top `k` eigenvectors as **eigenportfolios**, regress each stock's raw "
           "returns on `[1, factors]`, and cumulate the residuals into a price-like series `X`. This part is done for you:"),
    ("code", """window, k = 60, 3
Rw = R.iloc[-window:]
Z = ((Rw - Rw.mean()) / Rw.std()).to_numpy()
_, vec = np.linalg.eigh(np.corrcoef(Z.T))                  # eigenvalues in ascending order
factors = Z @ vec[:, ::-1][:, :k]                          # returns of the top-k eigenportfolios
A = np.c_[np.ones(window), factors]
coef, *_ = np.linalg.lstsq(A, Rw.to_numpy(), rcond=None)
X = pd.DataFrame(np.cumsum(Rw.to_numpy() - A @ coef, axis=0), columns=R.columns)   # cumulative residuals
X.iloc[:, :6].plot(figsize=(10, 3), lw=1, title="cumulative residuals X of six stocks (60 days)"); plt.show()"""),
    ("md", "Now the per-stock step. Fit `X` as an AR(1) (`b, a`); skip the stock if it doesn't revert, or reverts too slowly. The "
           "OU equilibrium is `μ = a/(1 − b)` with standard deviation `σ_eq = √(var(AR residuals, ddof=2) / (1 − b²))`, and the "
           "**s-score** is `(X_last − μ)/σ_eq`: how many equilibrium standard deviations the stock is away from its fair level "
           "relative to the factors."),
    ("md", YOUR_TURN),
    ("ex", _SS_HEAD + """    mu = ...                                              # ✍️ the equilibrium level
    sig_eq = ...                                          # ✍️ the equilibrium std
""" + _SS_TAIL,
     _SS_HEAD + """    mu = a / (1 - b)
    sig_eq = np.sqrt(resid.var(ddof=2) / (1 - b ** 2))
""" + _SS_TAIL),
    ("md", "## 3. Trading rules (Avellaneda & Lee, 2010)\n\n"
           "Per stock, given yesterday's position: **flat** → buy (`+1`) if `s < −1.25`, sell (`−1`) if `s > 1.25`; **long** → close once "
           "`s > −0.5`; **short** → close once `s < 0.75`. A stock with no s-score today is closed. Return a Series in the order of "
           "`prev`."),
    ("md", YOUR_TURN),
    ("ex", _SP_HEAD + """        elif cur == 0:
            out[name] = ...                               # ✍️ open long below −open_, short above +open_, else flat
        elif cur > 0:
            out[name] = ...                               # ✍️ close the long once v > −close_long
""" + _SP_TAIL,
     _SP_HEAD + """        elif cur == 0:
            out[name] = 1.0 if v < -open_ else -1.0 if v > open_ else 0.0
        elif cur > 0:
            out[name] = 0.0 if v > -close_long else 1.0
""" + _SP_TAIL),
    ("md", "## 4. The residual book\n\n"
           "`p.residual_backtest` runs this every day: s-scores from the last 60 days (3 factors, half-life ≤ 10), positions by your "
           "rules, **dollar-neutral** weights (+0.5 spread over the longs, −0.5 over the shorts), earned the next day, 5 bp per unit "
           "of turnover. Compare with a universe where **no** residual reverts."),
    ("code", """bt = p.residual_backtest(R)
bt_free = p.residual_backtest(R, cost_bps=0)
R0, _ = p.factor_universe(reversion_share=0.0)
bt_none = p.residual_backtest(R0)
market = R.mean(axis=1).loc[bt["returns"].index]
pd.DataFrame({"Sharpe": [p.sharpe(bt["returns"]), p.sharpe(bt_free["returns"]), p.sharpe(bt_none["returns"])],
              "correlation with the market": [np.corrcoef(bt["returns"], market)[0, 1], np.nan, np.nan]},
             index=["half the residuals revert, 5 bp", "half revert, no costs", "none revert, 5 bp"]).round(3)"""),
    ("code", """pos = bt["positions"].abs().to_numpy()
print(f"on average {(bt['positions'] > 0).sum(axis=1).mean():.1f} longs and {(bt['positions'] < 0).sum(axis=1).mean():.1f} shorts; "
      f"{pos[:, reverting].sum() / pos.sum():.0%} of the position-days are in the reverting half")
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(bt["returns"].cumsum(), label="reverting residuals"); ax.plot(bt_none["returns"].cumsum(), label="no reversion")
ax.legend(); ax.set_title("residual stat-arb book, net of costs"); plt.show()"""),
    ("md", "The book earns a Sharpe of about 1 net of costs with essentially zero market correlation, and loses its costs when "
           "there is nothing to find. Note the second line: the positions are barely tilted toward the stocks that really revert. "
           "Sixty days can't tell a 3-day OU component from noise stock by stock; the edge comes from averaging many weak bets. "
           "That is also why costs and crowding (August 2007) matter so much for this strategy.\n\n"
           "## Wrap-up\n\n"
           "* Hedge each stock with factors, trade the residual; the book is close to market- and factor-neutral by construction.\n"
           "* The s-score is an OU z-score of the cumulative residual; trade only fast reverters.\n"
           "* Many small bets, high turnover: the cost assumption decides the result.\n"
           "* Graded version: `labs/part09/week32_statarb` (`s_scores`, `s_score_positions`, `residual_backtest`)."),
]

# ---------------------------------------------------------------------------------------------- 06
_NW_CHECK = """
x = p.simulate_ou(500, np.log(2) / 5, 0.001, 0.01, seed=2)       # a persistent series (half-life 5)
iid = np.random.default_rng(3).normal(0.001, 0.01, 500)
cases = [(x, 0), (x, 5), (x, 10), (iid, 5)]
mine = [p.attempt(newey_west_se, *c) for c in cases]
mine = p.check("newey_west_se", mine, [p.newey_west_se(*c) for c in cases])
hac = sm.OLS(x, np.ones(x.size)).fit(cov_type="HAC", cov_kwds={"maxlags": 10, "use_correction": False}).bse[0]
pd.Series({"naive std/√n": x.std(ddof=1) / np.sqrt(x.size), "Newey–West, 5 lags": mine[1], "Newey–West, 10 lags": mine[2],
           "statsmodels HAC, 10 lags": hac}, name="standard error of the mean").to_frame()"""

_FM_HEAD = """def fama_macbeth(returns, exposures, nw_lags=0):
    lambdas = []
    for t, t1 in zip(returns.index[:-1], returns.index[1:]):
        X = pd.DataFrame({k: v.loc[t] for k, v in exposures.items()}).dropna()      # characteristics known at t
"""
_FM_TAIL = """        lambdas.append(sm.OLS(y, sm.add_constant(X)).fit().params)
    L = pd.DataFrame(lambdas)                             # one row of slopes per month
    se = L.std(ddof=1) / np.sqrt(len(L)) if nw_lags == 0 else L.apply(lambda c: p.newey_west_se(c.to_numpy(), nw_lags))
"""
_FM_CHECK = """
rets, expo = p.characteristics_panel()                    # 100 stocks × 60 months; true momentum premium 0.20%/month
cases = [(rets, expo), (rets, expo, 3)]
mine = [p.attempt(fama_macbeth, *c) for c in cases]
mine = p.check("fama_macbeth", mine, [p.fama_macbeth(*c) for c in cases])
display(mine[0].assign(premium=lambda d: d.premium * 100).rename(columns={"premium": "premium (%/month)"}).round(3))"""

_NEU_CHECK = """
rng = np.random.default_rng(0)
names = [f"N{i:02d}" for i in range(40)]
sectors = pd.Series([f"sec{i % 4}" for i in range(40)], index=names)
beta = pd.Series(rng.normal(1, 0.3, 40), index=names)
signal = pd.Series(rng.normal(0, 1, 40), index=names) + sectors.map({"sec0": 1.0, "sec1": 0.0, "sec2": -0.5, "sec3": 0.3}) + 2 * (beta - 1)
mine = p.check("neutralize", p.attempt(neutralize, signal, sectors, beta), p.neutralize(signal, sectors, beta))
pd.DataFrame({"raw signal": [*signal.groupby(sectors).mean(), np.corrcoef(signal, beta)[0, 1]],
              "neutralized": [*mine.groupby(sectors).mean(), np.corrcoef(mine, beta)[0, 1]]},
             index=[*sorted(sectors.unique()), "correlation with beta"]).round(3)"""

NB["06_factor_models"] = [
    header("06", "Factor models and cross-sectional premia", "S6 (Factor models & cross-sectional strategies)",
           "1. Compute a Newey–West standard error for an autocorrelated series.\n"
           "2. Estimate factor premia with Fama–MacBeth regressions and recover a planted premium.\n"
           "3. See how many months a premium needs before it is significant.\n"
           "4. Neutralize a signal against sectors and beta."),
    ("code", SETUP + "\nimport statsmodels.api as sm"),
    ("md", "## 1. Standard errors of autocorrelated means\n\n"
           "`std/√n` assumes independent observations. When the series is persistent (overlapping returns, strategy P&L, monthly "
           "slopes), neighbouring errors don't cancel and the true uncertainty is larger. **Newey–West** adds the autocovariances "
           "`γ_j = (1/n) Σ (x_t − x̄)(x_{t−j} − x̄)` with declining (Bartlett) weights:\n\n"
           "`se = √[(γ0 + 2 Σ_{j=1..L} (1 − j/(L+1)) γ_j) / n]`\n\n"
           "The code computes `γ0`; add the lag terms (`d[j:] @ d[:-j] / n` is `γ_j`)."),
    ("md", YOUR_TURN),
    ("ex", """def newey_west_se(x, lags):
    x = np.asarray(x, dtype=float)
    n, d = x.size, x - x.mean()
    s = d @ d / n                                         # γ0
    for j in range(1, lags + 1):
        s += ...                                          # ✍️ 2 · (1 − j/(lags+1)) · γj
    return float(np.sqrt(s / n))
""" + _NW_CHECK,
     """def newey_west_se(x, lags):
    x = np.asarray(x, dtype=float)
    n, d = x.size, x - x.mean()
    s = d @ d / n                                         # γ0
    for j in range(1, lags + 1):
        s += 2 * (1 - j / (lags + 1)) * (d[j:] @ d[:-j]) / n
    return float(np.sqrt(s / n))
""" + _NW_CHECK),
    ("md", "For this persistent series the honest standard error is about 2.5 times the naive one, so a naive t-statistic would be "
           "2.5 times too large. Your function matches `statsmodels`' HAC estimator.\n\n"
           "## 2. Fama–MacBeth\n\n"
           "Each month `t`, regress the **next** month's returns across stocks on the characteristics known at `t` (with a constant). "
           "The average slope is the premium; its t-statistic uses the standard deviation of the monthly slopes (`/√n`), or "
           "Newey–West when the slopes are autocorrelated. `p.characteristics_panel()` plants a momentum premium of 0.20% a month and "
           "gives \"size\" none."),
    ("md", YOUR_TURN),
    ("ex", _FM_HEAD + """        y = ...                                           # ✍️ next month's returns of the same stocks
""" + _FM_TAIL + """    return pd.DataFrame({"premium": ..., "t_stat": ...})  # ✍️
""" + _FM_CHECK,
     _FM_HEAD + """        y = returns.loc[t1, X.index]
""" + _FM_TAIL + """    return pd.DataFrame({"premium": L.mean(), "t_stat": L.mean() / se})
""" + _FM_CHECK),
    ("md", "Momentum comes out at about 0.26% a month (true 0.20%) with t ≈ 3.4; size, which has no premium, is insignificant. "
           "Each monthly slope is noisy (100 stocks with 6% idiosyncratic volatility each); only the average over many months is informative. "
           "How many months do you need?"),
    ("code", """rows = {}
for months in (12, 24, 36, 60):
    fm = p.fama_macbeth(rets.iloc[:months], {k: v.iloc[:months] for k, v in expo.items()})
    rows[f"first {months} months"] = {"momentum premium (%)": fm.loc["momentum", "premium"] * 100,
                                      "momentum t": fm.loc["momentum", "t_stat"], "size t": fm.loc["size", "t_stat"]}
pd.DataFrame(rows).T.round(2)"""),
    ("md", "After 12 months the estimate has the wrong sign; after 24 it happens to look significant, after 36 it doesn't; only the "
           "full 60 months give a clear answer. On average the t-statistic grows with √(months), but a short sample can mislead "
           "in either direction. A real but small premium needs years of data (which is why the lab also asks whether a premium "
           "survives in the second half of the sample).\n\n"
           "## 3. Neutralization\n\n"
           "A raw signal often hides bets you didn't mean to make: whole sectors, or high-beta stocks. **Neutralize** it: regress the "
           "signal across stocks on one dummy per sector (no extra constant) plus beta, and keep the residual. Build the design matrix "
           "with `pd.get_dummies(..., dtype=float)` and fit with `sm.OLS(signal, X).fit().resid`."),
    ("md", YOUR_TURN),
    ("ex", """def neutralize(signal, sectors, beta):
    D = pd.get_dummies(sectors.loc[signal.index], dtype=float)                # one column per sector
    X = pd.concat([D, beta.loc[signal.index].rename("beta")], axis=1)
    return ...                                            # ✍️ the OLS residual
""" + _NEU_CHECK,
     """def neutralize(signal, sectors, beta):
    D = pd.get_dummies(sectors.loc[signal.index], dtype=float)                # one column per sector
    X = pd.concat([D, beta.loc[signal.index].rename("beta")], axis=1)
    return sm.OLS(signal, X).fit().resid
""" + _NEU_CHECK),
    ("md", "After neutralization, every sector averages exactly zero and the signal is uncorrelated with beta: a long-short book built "
           "on it is sector- and beta-neutral before any optimizer gets involved.\n\n"
           "## Wrap-up\n\n"
           "* Use Newey–West whenever the observations overlap or persist.\n"
           "* Fama–MacBeth: one cross-sectional regression per period, then average; premia need long samples.\n"
           "* Neutralize signals against sectors and beta before trading them.\n"
           "* Graded version: `labs/part09/week32_statarb` (`newey_west_se`, `fama_macbeth`, `neutralize`)."),
]

# ---------------------------------------------------------------------------------------------- 07
_CHOW_CHECK = """
bp, good = p.breaking_pair(), p.cointegrated_pair()        # bp dies at day 900; good never does
rng = np.random.default_rng(1)
xi = rng.normal(size=1500)
yi = 1 + 2 * xi + rng.normal(size=1500)                    # a textbook regression with iid errors
cases = {"breaking pair, break at 900": (bp.y, bp.x, 900), "healthy pair, 'break' at 750": (good.y, good.x, 750),
         "iid regression, 'break' at 750": (yi, xi, 750)}
mine = {k: p.attempt(chow_test, *c) for k, c in cases.items()}
mine = p.check("chow_test", mine, {k: p.chow_test(*c) for k, c in cases.items()})
pd.DataFrame(mine, index=["F", "p-value"]).T"""

_RET_CHECK = """
stab_good = p.rolling_stability(good.y, good.x)
cases = [(stab,), (stab_good,), (stab, 0.2, 30, 0.25, 1), (stab_good, 0.05)]
mine = [p.attempt(retirement_date, *c) for c in cases]
mine = p.check("retirement_date", mine, [p.retirement_date(*c) for c in cases])
pd.Series(mine, index=["breaking pair", "healthy pair", "breaking pair, patience 1", "healthy pair, max p 0.05"],
          name="retired on").to_frame()"""

_RC_HEAD = """from scipy.stats import norm

def reality_check(perf, n_boot=500, mean_block=10, seed=0):
    P = np.asarray(perf, dtype=float)
    T = P.shape[0]
    m = P.mean(axis=0)
"""
_RC_TAIL = """    best = int(np.argmax(m))
    t = m[best] / (P[:, best].std(ddof=1) / np.sqrt(T))
    return {"best": best, "p_value": float((Vs >= V).mean()), "naive_p": float(norm.sf(t))}

noise = np.random.default_rng(0).normal(0, 0.01, (1000, 50))     # 50 strategies, 4 years, no edge at all
edge = noise.copy()
edge[:, 7] += 0.0012                                              # strategy 7 has a real edge (Sharpe ≈ 1.9)
mine = [p.attempt(reality_check, noise), p.attempt(reality_check, edge)]
mine = p.check("reality_check", mine, [p.reality_check(noise), p.reality_check(edge)])
pd.DataFrame(mine, index=["50 noise strategies", "one has a real edge"]).round(4)"""

NB["07_stability_breaks"] = [
    header("07", "Breaks, stability and multiple testing", "S7 (Advanced analysis: breaks, stability & multiple testing)",
           "1. Test for a break at a known date (Chow) and an unknown one (CUSUM), and see when they mislead.\n"
           "2. Monitor a pair with rolling statistics and write an automatic retirement rule.\n"
           "3. Measure what retirement saves on a pair that dies.\n"
           "4. Test \"the best of 50 strategies\" honestly with White's Reality Check."),
    ("code", SETUP + "\nimport statsmodels.api as sm\nfrom scipy.stats import f as f_dist"),
    ("md", "## 1. The Chow test\n\n"
           "Is `y = α + βx` the same before and after a **known** date? Fit it on the whole sample (`SSR_pooled`) and on the two "
           "segments (`SSR_1`, `SSR_2`). With `k = 2` parameters and `n` rows:\n\n"
           "`F = ((SSR_pooled − SSR_1 − SSR_2) / k) / ((SSR_1 + SSR_2) / (n − 2k))`, p-value from `F(k, n − 2k)`."),
    ("md", YOUR_TURN),
    ("ex", """def chow_test(y, x, break_idx):
    y, x = np.asarray(y, float), np.asarray(x, float)

    def ssr(yy, xx):
        return sm.OLS(yy, sm.add_constant(xx)).fit().ssr

    n, k = y.size, 2
    s1, s2 = ssr(y[:break_idx], x[:break_idx]), ssr(y[break_idx:], x[break_idx:])
    F = ...                                               # ✍️
    return float(F), float(f_dist.sf(F, k, n - 2 * k))
""" + _CHOW_CHECK,
     """def chow_test(y, x, break_idx):
    y, x = np.asarray(y, float), np.asarray(x, float)

    def ssr(yy, xx):
        return sm.OLS(yy, sm.add_constant(xx)).fit().ssr

    n, k = y.size, 2
    s1, s2 = ssr(y[:break_idx], x[:break_idx]), ssr(y[break_idx:], x[break_idx:])
    F = ((ssr(y, x) - s1 - s2) / k) / ((s1 + s2) / (n - 2 * k))
    return float(F), float(f_dist.sf(F, k, n - 2 * k))
""" + _CHOW_CHECK),
    ("md", "The real break is unmistakable, and the iid regression correctly shows none. But the **healthy** pair also \"breaks\" at "
           "the 5% level. The Chow test assumes independent errors; a cointegrated spread is autocorrelated by design (that is the "
           "whole trade), so the test is over-confident. CUSUM, for a break at an **unknown** date, has the same problem:"),
    ("code", """print(f"CUSUM p-value — breaking pair: {p.cusum_pvalue(bp.y, bp.x):.1e}, healthy pair: {p.cusum_pvalue(good.y, good.x):.1e}, "
      f"iid regression: {p.cusum_pvalue(yi, xi):.2f}")
false_alarms = sum(p.cusum_pvalue(g.y, g.x) < 0.05 for g in (p.cointegrated_pair(seed=s) for s in range(100)))
print(f"100 healthy cointegrated pairs: CUSUM rejects stability for {false_alarms} of them")"""),
    ("md", "## 2. Monitor instead\n\n"
           "A practical alternative: re-run Engle–Granger every month (21 days) on the last 250 days and watch the p-value, the "
           "half-life, the hedge ratio and the spread volatility (`p.rolling_stability`). The vertical line is the true break."),
    ("code", """stab = p.rolling_stability(bp.y, bp.x)
fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, col in zip(axes, stab.columns):
    ax.plot(stab.index, stab[col]); ax.axvline(bp.index[900], color="black", lw=1); ax.set_title(col)
    ax.tick_params(axis="x", labelrotation=45)
axes[1].set_yscale("log")
plt.tight_layout(); plt.show()"""),
    ("md", "Turn the dashboard into a rule. A check is **broken** when `pvalue > max_p`, or `half_life > max_hl`, or the hedge ratio "
           "has drifted more than `max_beta_drift` from the first check (`|β/β_first − 1|`). Retire the pair at the first check where "
           "the rule has been broken on `patience` **consecutive** checks; return `None` if it never is."),
    ("md", YOUR_TURN),
    ("ex", """def retirement_date(stab, max_p=0.2, max_hl=30, max_beta_drift=0.25, patience=2):
    bad = (...).to_numpy()                                # ✍️ True where any of the three rules is broken
    run = 0
    for date, b in zip(stab.index, bad):
        run = run + 1 if b else 0
        if run >= patience:
            return date
    return None
""" + _RET_CHECK,
     """def retirement_date(stab, max_p=0.2, max_hl=30, max_beta_drift=0.25, patience=2):
    bad = ((stab["pvalue"] > max_p) | (stab["half_life"] > max_hl)
           | ((stab["beta"] / stab["beta"].iloc[0] - 1).abs() > max_beta_drift)).to_numpy()
    run = 0
    for date, b in zip(stab.index, bad):
        run = run + 1 if b else 0
        if run >= patience:
            return date
    return None
""" + _RET_CHECK),
    ("md", "The rule catches the dead pair within one 250-day window of the break, and leaves the healthy pair alone. Tighten it and "
           "it fires on one bad month of a healthy relationship: with patience 1, the breaking pair is retired in March 2017, more "
           "than two years *before* its break; with `max_p` 0.05, the healthy pair is retired in February 2017.\n\n"
           "## 3. What retirement saves\n\n"
           "Trade the breaking pair as in notebook 03 (formation 750 days, rolling z, which never stops out), then again with the "
           "position forced to flat from the retirement date on."),
    ("code", """F = 750
eb = p.engle_granger(bp.y.iloc[:F], bp.x.iloc[:F])
z = p.rolling_zscore(bp.y - eb["beta"] * bp.x - eb["alpha"], round(3 * eb["half_life"]))
pos = p.pairs_positions(z)
cut = bp.index.get_loc(mine[0])
pos_retired = np.where(np.arange(len(pos)) >= cut, 0.0, pos)
pl, pl_retired = p.pair_pnl(pos, bp.y, bp.x, eb["beta"]), p.pair_pnl(pos_retired, bp.y, bp.x, eb["beta"])
print(f"break at day 900, retired at day {cut}; P&L after the break: {pl[900:].sum():+.1%} without retirement, "
      f"{pl_retired[900:].sum():+.1%} with it")"""),
    ("md", "## 4. The best of many strategies\n\n"
           "Try 50 strategies and report the best one's t-test, and you will almost always find \"significance\". **White's Reality "
           "Check** asks the right question: is the *best* mean larger than what the best of 50 would show by luck?\n\n"
           "* `V = max_k √T·mean_k` (the best strategy's scaled mean).\n"
           "* Each bootstrap draws one set of rows for **all** strategies (`p.stationary_bootstrap_indices(T, mean_block, rng)`, which "
           "keeps short-range dependence), and records `V* = max_k √T·(mean*_k − mean_k)`: the best of 50 when every true mean is zero.\n"
           "* p-value = share of `V* >= V`."),
    ("md", YOUR_TURN),
    ("ex", _RC_HEAD + """    V = ...                                               # ✍️
    rng = np.random.default_rng(seed)
    Vs = np.array([... for _ in range(n_boot)])           # ✍️ one V* per bootstrap
""" + _RC_TAIL,
     _RC_HEAD + """    V = np.sqrt(T) * m.max()
    rng = np.random.default_rng(seed)
    Vs = np.array([np.sqrt(T) * (P[p.stationary_bootstrap_indices(T, mean_block, rng)].mean(axis=0) - m).max()
                   for _ in range(n_boot)])
""" + _RC_TAIL),
    ("md", "The best of 50 noise strategies has a naive p-value below 1%. The Reality Check says 0.24: exactly what you'd expect from "
           "the best of 50 coins. With a real edge in strategy 7, both agree. Keep a log of *every* variant you try (Part 8's research "
           "log) so the \"50\" is honest.\n\n"
           "## Wrap-up\n\n"
           "* Chow and CUSUM assume iid errors; on a cointegrated spread they cry wolf.\n"
           "* Monitor rolling p-value, half-life, β and spread volatility; retire pairs by rule, not by feel.\n"
           "* Correct for every strategy you tried: FDR for many pairs, the Reality Check (or Hansen's SPA) for the best of many.\n"
           "* Graded version: `labs/part09/week32_statarb` (`chow_test`, `cusum_pvalue`, `rolling_stability`, `retirement_date`, "
           "`reality_check`)."),
]

# ---------------------------------------------------------------------------------------------- 08
_ENS_CHECK = """
cases = [([0, 2.1, 1.0, 0.6, 0.3],), (z,), (z, (1.75, 2.0, 2.25), (0.25, 0.5, 0.75), 4.0, W)]
mine = [p.attempt(ensemble_positions, *c) for c in cases]
mine = p.check("ensemble_positions", mine, [p.ensemble_positions(*c) for c in cases])
print("toy:", np.round(mine[0], 3), "\\nOOS Sharpe of the ensemble on this pair:",
      round(p.sharpe(p.pair_pnl(mine[1], pair.y, pair.x, eg["beta"])[F:]), 2))"""

_BW_HEAD = """def pair_book_weights(pair_returns, pairs, name_cap=0.2, max_iter=1000):
    w = 1 / pair_returns.std(ddof=1)
    w = w / w.sum()                                       # inverse volatility, summing to 1
    for _ in range(max_iter):
        load = {}
        for k, (a, b) in pairs.items():                   # a stock's load: the weights of all pairs it is in
            load[a] = load.get(a, 0.0) + w[k]
            load[b] = load.get(b, 0.0) + w[k]
        if max(load.values()) <= name_cap * (1 + 1e-9):
            return w
"""
_BW_TAIL = """        w = w * f
        w = w / w.sum()
    raise ValueError(f"name cap {name_cap} is infeasible for these pairs")

formation_rets = rets.iloc[:F]
cases = [(formation_rets, pairs, 0.15), (formation_rets, pairs, 1.0)]
mine = [p.attempt(pair_book_weights, *c) for c in cases]
mine = p.check("pair_book_weights", mine, [p.pair_book_weights(*c) for c in cases])
w = mine[0]
display(pd.DataFrame({"inverse vol": mine[1], "with a 15% name cap": w}).round(3))
try:
    p.pair_book_weights(formation_rets, pairs, 0.1)
except ValueError as err:
    print("cap 0.10:", err)"""

_EX_HEAD = """def book_exposures(pair_pos, pair_w, pairs, hedges, betas, sectors):
    w = {}
    for k, (a, b) in pairs.items():
        s, h = pair_pos[k] * pair_w[k], hedges[k]
"""
_EX_TAIL = """    return {"weights": w, "gross": gross, "net": float(w.sum()), "net_beta": float((w * betas.loc[w.index]).sum()),
            "sector_net": w.groupby(sectors.loc[w.index]).sum().sort_index()}

toy = (pd.Series({"P1": 1.0, "P2": -1.0}), pd.Series({"P1": 0.5, "P2": 0.5}), {"P1": ("A", "B"), "P2": ("A", "C")},
       pd.Series({"P1": 1.5, "P2": 0.5}), pd.Series({"A": 1.0, "B": 1.2, "C": 0.8}), pd.Series({"A": "tech", "B": "tech", "C": "fin"}))
today = (positions.iloc[-1], w, pairs, hedges, betas, sectors)
mine = [p.attempt(book_exposures, *toy), p.attempt(book_exposures, *today)]
mine = p.check("book_exposures", mine, [p.book_exposures(*toy), p.book_exposures(*today)])
ex = mine[1]
print(f"last day: {len(ex['weights'])} stocks, gross {ex['gross']:.3f}, net {ex['net']:+.4f}, net beta {ex['net_beta']:+.4f}")
ex["sector_net"].round(4).to_frame("net weight by sector")"""

NB["08_pairs_portfolio"] = [
    header("08", "Robust parameters and a portfolio of pairs", "S8 (Robust optimization, portfolio of pairs & M5b release)",
           "1. See how much a pair's Sharpe depends on the entry and exit thresholds.\n"
           "2. Replace one parameter choice with an ensemble of nearby ones.\n"
           "3. Allocate a book of pairs with inverse volatility and per-name caps.\n"
           "4. Look through the pairs to stock-level gross, net, beta and sector exposure."),
    ("code", SETUP),
    ("md", "## 1. One pair, 25 parameter choices\n\n"
           "The pair and the formation are those of notebook 03. Here is its out-of-sample Sharpe for every (entry, exit) combination:"),
    ("code", """pair = p.cointegrated_pair()
F = 750
eg = p.engle_granger(pair.y.iloc[:F], pair.x.iloc[:F])
W = round(3 * eg["half_life"])
z = p.rolling_zscore(pair.y - eg["beta"] * pair.x - eg["alpha"], W)
grid = {ex: {en: p.sharpe(p.pair_pnl(p.pairs_positions(z, en, ex), pair.y, pair.x, eg["beta"])[F:])
             for en in (1.5, 1.75, 2.0, 2.25, 2.5)} for ex in (0.0, 0.25, 0.5, 0.75, 1.0)}
pd.DataFrame(grid).rename_axis("entry ↓ · exit →").round(2)"""),
    ("md", "Sharpe ranges from about −0.1 to 1.6 on the *same* pair and data. Picking the best cell is fitting noise (Part 8's "
           "overfitting lessons). A robust alternative: **average the positions** of several nearby parameter sets. The result is "
           "a fractional position in `[−1, 1]` that scales in and out."),
    ("md", YOUR_TURN),
    ("ex", """def ensemble_positions(z, entries=(1.75, 2.0, 2.25), exits=(0.25, 0.5, 0.75), stop=4.0, max_hold=None):
    return ...                                            # ✍️ the mean of p.pairs_positions over every (entry, exit)
""" + _ENS_CHECK,
     """def ensemble_positions(z, entries=(1.75, 2.0, 2.25), exits=(0.25, 0.5, 0.75), stop=4.0, max_hold=None):
    return np.mean([p.pairs_positions(z, e, x, stop, max_hold) for e in entries for x in exits], axis=0)
""" + _ENS_CHECK),
    ("md", "On one pair, the ensemble lands inside the range of its nine members. Over several pairs, compare it with the members' "
           "worst, average and best, and with the textbook choice (2.0, 0.5), which you'd have picked in advance:"),
    ("code", """rows = {}
for seed in range(10, 18):
    pr = p.cointegrated_pair(seed=seed)
    e2 = p.engle_granger(pr.y.iloc[:F], pr.x.iloc[:F])
    zz = p.rolling_zscore(pr.y - e2["beta"] * pr.x - e2["alpha"], round(3 * e2["half_life"]))
    member = [p.sharpe(p.pair_pnl(p.pairs_positions(zz, en, ex), pr.y, pr.x, e2["beta"])[F:])
              for en in (1.75, 2.0, 2.25) for ex in (0.25, 0.5, 0.75)]
    rows[f"pair {seed}"] = {"worst member": min(member), "mean member": np.mean(member), "best member": max(member),
                            "(2.0, 0.5)": member[4], "ensemble": p.sharpe(p.pair_pnl(p.ensemble_positions(zz), pr.y, pr.x, e2["beta"])[F:])}
table = pd.DataFrame(rows).T
display(table.round(2))
print(f"ensemble beats the mean member on {(table['ensemble'] > table['mean member']).sum()} of {len(table)} pairs")"""),
    ("md", "The ensemble beats the average member on every pair and is never the worst. The textbook (2.0, 0.5) is sometimes "
           "better and sometimes far worse (pair 16). You don't know in advance which cell will be lucky; the ensemble doesn't need "
           "to.\n\n"
           "## 2. A book of pairs\n\n"
           "Five sectors of eight stocks with two true pairs per sector. Screen within sectors on the first 500 days, then trade every "
           "selected pair with the ensemble (look-back and time stop 3 half-lives) over the last 500."),
    ("code", """prices, sectors, truth = p.sector_universe(n_sectors=5, per_sector=8, pairs_per_sector=2, seed=3)
F = 500
screen = p.screen_pairs(prices.iloc[:F], sectors)
sel = screen[screen.selected]
pairs, hedges, rets, positions = {}, {}, {}, {}
for a, b in sel[["a", "b"]].to_numpy():
    key, y, x = f"{a}/{b}", prices[a], prices[b]
    e = p.engle_granger(y.iloc[:F], x.iloc[:F])
    hl = float(np.clip(e["half_life"], 2, 30))
    zz = p.rolling_zscore(y - e["beta"] * x - e["alpha"], max(10, round(3 * hl)))
    positions[key] = p.ensemble_positions(zz, max_hold=max(10, round(3 * hl)))
    rets[key] = p.pair_pnl(positions[key], y, x, e["beta"])
    pairs[key], hedges[key] = (a, b), e["beta"]
rets, positions, hedges = pd.DataFrame(rets, index=prices.index), pd.DataFrame(positions, index=prices.index), pd.Series(hedges)
print(f"{len(pairs)} pairs selected; not true pairs: {[k for k, ab in pairs.items() if ab not in truth]}; "
      f"true pairs missed: {[t for t in truth if t not in pairs.values()]}")
print("stocks in more than one pair:", pd.Series([n for ab in pairs.values() for n in ab]).value_counts().loc[lambda s: s > 1].to_dict())"""),
    ("md", "One stock appears in two pairs, so its risk is counted twice. Weight the pairs by **inverse volatility** (formation "
           "returns), then cap every stock's **load** (the total weight of the pairs it appears in): repeat {scale each pair by "
           "`min(1, cap/load_a, cap/load_b)`; renormalize to 1} until no load exceeds the cap. If that never happens within "
           "`max_iter` rounds, the cap is infeasible."),
    ("md", YOUR_TURN),
    ("ex", _BW_HEAD + """        f = pd.Series(...)                                # ✍️ per pair key: min(1, cap/load[a], cap/load[b])
""" + _BW_TAIL,
     _BW_HEAD + """        f = pd.Series({k: min(1.0, name_cap / load[a], name_cap / load[b]) for k, (a, b) in pairs.items()})
""" + _BW_TAIL),
    ("code", """oos = rets.iloc[F:]
book = oos @ w
single = oos.apply(p.sharpe)
corr = oos.corr().to_numpy()[np.triu_indices(len(pairs), 1)].mean()
print(f"book Sharpe {p.sharpe(book):.2f}; single pairs: mean {single.mean():.2f} (from {single.min():.2f} to {single.max():.2f}); "
      f"average correlation between pairs {corr:+.3f}")
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(oos.cumsum(), lw=0.7, color="#8a8984"); ax.plot(book.cumsum(), lw=2.5, color="#2a78d6")
ax.set_title("single pairs (grey) and the book (blue), out of sample"); plt.show()"""),
    ("md", "Nine weakly correlated bets with Sharpes around 1 add up to a book of about 3.6. Be suspicious of that number: here the "
           "pairs are independent **by construction** and none of them breaks. Real pairs share sectors and factors, and they "
           "fall together when crowded books unwind (Clinic W2 simulates one).\n\n"
           "## 3. Look through to the stocks\n\n"
           "Risk limits apply to stocks, not pairs. Pair `k` with position `s`, weight `w` and hedge `h` on names `(a, b)` puts "
           "`s·w/(1 + |h|)` on `a` and `−s·w·h/(1 + |h|)` on `b`; a stock collects weight from every pair it is in. Report the stock "
           "weights (zeros dropped), **gross** `Σ|w|`, **net** `Σw`, **net beta** `Σ w·β` and the net weight per sector. The "
           "stock betas are estimated on the formation period against an equal-weighted market."),
    ("code", """moves = prices.diff().iloc[1:F]
market = moves.mean(axis=1)
betas = moves.apply(lambda c: np.cov(c, market)[0, 1] / market.var())"""),
    ("md", YOUR_TURN),
    ("ex", _EX_HEAD + """        w[a] = w.get(a, 0.0) + ...                        # ✍️ the long leg y
        w[b] = w.get(b, 0.0) + ...                        # ✍️ the short leg, −h·x
    w = pd.Series(w).sort_index()
    w = w[w != 0]
    gross = ...                                           # ✍️
""" + _EX_TAIL,
     _EX_HEAD + """        w[a] = w.get(a, 0.0) + s / (1 + abs(h))
        w[b] = w.get(b, 0.0) - s * h / (1 + abs(h))
    w = pd.Series(w).sort_index()
    w = w[w != 0]
    gross = float(w.abs().sum())
""" + _EX_TAIL),
    ("code", """daily = pd.DataFrame([{k: v for k, v in p.book_exposures(positions.loc[d], w, pairs, hedges, betas, sectors).items()
                       if k in ("gross", "net", "net_beta")} for d in positions.index[F:]], index=positions.index[F:])
display(daily.describe().loc[["mean", "min", "max"]].round(3))
daily.plot(figsize=(10, 3), lw=1, title="the book's gross, net and net beta, day by day"); plt.show()"""),
    ("md", "Each pair is close to dollar- and beta-neutral (most hedge ratios are near 1), so the book stays near zero net and zero net beta "
           "while its gross exposure moves with the number of open trades. That is what \"market-neutral\" should look like "
           "before you lean on it.\n\n"
           "## Wrap-up\n\n"
           "* Don't pick the best parameter cell; average nearby ones.\n"
           "* Many small, weakly correlated pairs; weight by inverse volatility and cap every stock's load.\n"
           "* Report and limit exposures at the stock level: gross, net, net beta, sector.\n"
           "* A diversified book of pairs is not a hedge against a crowded unwind.\n"
           "* Graded version: `labs/part09/week32_statarb` (`ensemble_positions`, `pair_book_weights`, `book_exposures`) and "
           "Clinic W2 (a 19-pair book, its risk report and a crowded-unwind scenario)."),
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
