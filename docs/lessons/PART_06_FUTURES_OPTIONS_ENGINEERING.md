# Part 6 — Futures & Options Engineering: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 2, **Month 6, first half** (program weeks 21–22); Part 7 (Strategy Library) follows in weeks 23–24 |
| Format | 8 sessions × **120 min** (4 per week, longer than usual because the material is dense) + 2 lab clinics × 120 min + self-study (~7 hrs/week) |
| Total effort | ~16 hrs live + 4 hrs clinic + 14 hrs self-study ≈ **34 hours** |
| Brokers | IB: futures, futures options (FOP), equity/index options, model Greeks · Alpaca: equity options chain, snapshots with Greeks/IV |
| Platform milestone | **M3a**: `futures/roll.py`, `options/chain.py`, `options/pricing.py`, `options/greeks.py`, `options/iv.py`, `options/surface.py`, `options/portfolio.py` |
| Covers original items | "Advance with Python" 12–16 · Platform roadmap 30 (option strike helpers) and the Greeks engine used by 33–41 |

---

## 1. Learning Objectives

By the end of Part 6 the learner will be able to:

1. **Build** continuous futures series (difference- and ratio-adjusted) with explicit, testable roll rules, and compute fair value, basis and carry.
2. **Retrieve and filter** option chains from IB and Alpaca, and **select strikes** by delta, % OTM, expected move or premium, with liquidity filters.
3. **Implement** Black–Scholes–Merton and Black-76 pricing, all first-order Greeks and eight second-order Greeks, with documented unit conventions.
4. **Solve** for implied volatility robustly (Newton + Brent fallback, no-arbitrage bounds, implied forwards).
5. **Price** American and path-dependent options with binomial trees, Monte Carlo (with variance reduction) and finite differences, and compute Greeks by bump-and-revalue and automatic differentiation.
6. **Fit** an arbitrage-aware volatility surface (SVI per expiry; SABR for futures options) and explain skew, term structure and sticky-strike vs sticky-delta dynamics.
7. **Aggregate** portfolio Greeks in dollar terms, beta-weight them, run spot × vol scenario grids, and **simulate** delta hedging.
8. **Validate** the Greeks engine against IB model Greeks and an independent library (`py_vollib`).

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| 1.5 Derivatives (forwards, futures, payoffs, put-call parity) | All sessions |
| 1.6 F&O strategies (conceptual) | S2, S8 |
| 2.1 Math (calculus, optimization) | S3–S7 |
| 2.2–2.3 Statistics & time series (lognormal returns, volatility) | S3, S6 |
| 4.2 IB contracts, 4.3 Alpaca, 4.9 futures specs | S1, S2 |
| 5.7 Volatility estimators (realized vol) | S7, S8 |

---

## 3. Two-Week Overview

| Week | Theme | Sessions | Clinic Lab | Platform Output |
|---|---|---|---|---|
| **W1** | Futures, chains, pricing, IV | S1 Futures library · S2 Option chain & strike library · S3 BSM/Black-76 & first-order Greeks · S4 Implied volatility | Build a live SPY/SPX chain with our own IV and Greeks; compare with IB model Greeks | `futures/roll.py`, `options/chain.py`, `options/pricing.py`, `options/iv.py` |
| **W2** | Higher-order Greeks, numerics, surfaces, portfolio | S5 Second-order Greeks · S6 Numerical methods · S7 Volatility surface · S8 Portfolio Greeks & hedging | Surface fit + portfolio risk report + delta-hedging study | `options/greeks.py`, `options/surface.py`, `options/portfolio.py`, **M3a release** |

---

## 4. Conventions (fixed for the whole program)

| Quantity | Internal (library) unit | Display unit | IB `modelGreeks` unit |
|---|---|---|---|
| Time to expiry `T` | Years, ACT/365 calendar time | Days (DTE) | – |
| Volatility `σ` | Decimal (0.20) | % (20.0) | Decimal |
| Delta | Per 1.00 move in underlying | Same | Same |
| Gamma | Per 1.00 move | Same | Same |
| Vega | Per 1.00 change in σ | ÷ 100 → per vol point | Per vol point |
| Theta | Per year (−∂V/∂T) | ÷ 365 → per calendar day | Per day |
| Rho | Per 1.00 change in r | ÷ 100 → per 1% | – |
| Charm, color, veta | Change **as time passes** (−∂/∂T), like theta | ÷ 365 → per day | – |
| Rates & dividends | Continuous compounding | – | – |

Every function takes `cp = +1` (call) or `−1` (put). **Black-76** (futures options) reuses BSM with `S = F` and `q = r`.

---

## 5. Session-by-Session Plan

> Each 120-min session: **25 min theory → 60 min live coding → 25 min guided lab → 10 min wrap-up and homework.**
> Notebooks live in `notebooks/part06/`; promoted code lives in `quantforge/futures/` and `quantforge/options/`.

### Week 1 — Futures, Chains, Pricing, Implied Volatility

#### S1 · Module 6.1 — Futures Library

| Block | Content |
|---|---|
| Theory | Contract specs revisited: multiplier, tick, expiry rules (ES/NQ: quarterly H/M/U/Z, third Friday; CL: monthly, a few business days before the 25th of the prior month; check each exchange's calendar). Month codes `F G H J K M N Q U V X Z`. **Fair value** `F = S·e^{(r−q)T}`; basis `F − S`; carry and roll yield; contango vs backwardation. **Continuous contracts** for research: unadjusted (price jumps at rolls, which ruins indicators), **difference back-adjusted** (keeps point P&L, can produce negative prices), **ratio back-adjusted** (keeps % returns). Rule: signals may use adjusted series, but P&L and orders always use the actual contract. **Roll rules:** calendar (N days before expiry), volume or open-interest crossover, first-notice-day for physically delivered contracts. Calendar spreads and IB combo (`BAG`) orders. IB: `reqContractDetails(Future("ES", exchange="CME"))` lists active months; `includeExpired=True` for history; `ContFuture` for data only. |
| Live coding | Chain discovery from IB; roll schedule; back-adjusted continuous series (below); implied carry from two contract months. |
| Lab | Build ES and CL continuous series (both adjustments) from 3 years of individual contracts; show how a 50/200 SMA crossover gives different signals on unadjusted vs adjusted data. |
| Homework | `RollCalendar` class: given a root symbol, returns the active contract for any date under a configurable rule; unit tests on known roll dates. |

```python
def back_adjust(front, nxt, roll_idx, method="difference"):
    """Stitch two contracts into one continuous series, adjusting history at the roll bar."""
    gap = nxt[roll_idx] - front[roll_idx]
    ratio = nxt[roll_idx] / front[roll_idx]
    hist = front[:roll_idx] + gap if method == "difference" else front[:roll_idx] * ratio
    return np.concatenate([hist, nxt[roll_idx:]])

def implied_carry(f_near, f_far, t_near, t_far):
    """Annualized carry (r - q) implied by two futures prices on the same underlying."""
    return np.log(f_far / f_near) / (t_far - t_near)
```

#### S2 · Module 6.2 — Option Chain & Strike Library (item 30)

| Block | Content |
|---|---|
| Theory | **Chain retrieval.** IB: `reqSecDefOptParams` → expirations, strikes, trading class, multiplier per exchange (filter `SMART`); build `Option` contracts, `qualifyContracts`, `reqTickers` → `modelGreeks` (IV, delta, gamma, vega, theta, underlying price). Alpaca: `TradingClient.get_option_contracts(GetOptionContractsRequest(...))` for contract lists; `OptionHistoricalDataClient.get_option_chain(OptionChainRequest(underlying_symbol=...))` for snapshots with quotes, IV and Greeks. **OCC symbology** (`AAPL261218C00200000`). Exercise styles: US equity/ETF options are American with physical settlement; SPX/XSP index options are European and cash-settled (cleanest for surface work). **Expiry selection** by DTE window (weeklies, monthlies, 0DTE). **Strike selection methods:** ATM (nearest to forward), by delta, by % OTM, by expected move `S·σ·√T`, by target premium. **Liquidity filters:** open interest, volume, bid-ask spread as % of mid, no zero bids. |
| Live coding | `OptionChain` object from both brokers → one DataFrame schema; strike selectors (below). |
| Lab | For SPY, select: the 30-DTE 25-delta put, the 16-delta call, strikes ±1 expected move, and a 45-DTE ATM straddle; print prices and liquidity. |
| Homework | `select_expiry(dte_min, dte_max, prefer="monthly")` and a liquidity score used as a tie-breaker. |

```python
from ib_async import Stock, Option

und = ib.qualifyContracts(Stock("SPY", "SMART", "USD"))[0]
params = next(p for p in ib.reqSecDefOptParams(und.symbol, "", und.secType, und.conId)
              if p.exchange == "SMART")
expiry = sorted(params.expirations)[5]
spot = ib.reqTickers(und)[0].marketPrice()
strikes = [k for k in params.strikes if 0.9 * spot < k < 1.1 * spot]
opts = ib.qualifyContracts(*[Option("SPY", expiry, k, right, "SMART", tradingClass=params.tradingClass)
                             for k in strikes for right in ("C", "P")])
tickers = ib.reqTickers(*opts)          # t.modelGreeks: impliedVol, delta, gamma, vega, theta

def occ_symbol(root, expiry, cp, strike):
    """OCC symbol, e.g. occ_symbol('AAPL', date(2026, 12, 18), 'C', 200) -> 'AAPL261218C00200000'."""
    return f"{root}{expiry:%y%m%d}{cp}{int(round(strike * 1000)):08d}"

def strike_by_delta(strikes, S, T, r, q, iv_by_strike, cp, target):
    """Strike whose BSM delta (using each strike's own IV) is closest to target, e.g. -0.25."""
    deltas = np.array([bsm_greeks(S, K, T, r, q, iv, cp)["delta"]
                       for K, iv in zip(strikes, iv_by_strike)])
    i = int(np.argmin(np.abs(deltas - target)))
    return strikes[i], deltas[i]
```

#### S3 · Module 6.3a — BSM, Black-76 & First-Order Greeks

| Block | Content |
|---|---|
| Theory | BSM assumptions and where each breaks. Derivation sketch (risk-neutral expectation). `d₁ = [ln(S/K) + (r − q + σ²/2)T] / (σ√T)`, `d₂ = d₁ − σ√T`. Prices, put-call parity `C − P = S·e^{−qT} − K·e^{−rT}`. **Dividends:** continuous yield for indices; for single stocks, discrete dividends are handled via the **implied forward** from put-call parity (`F = K + e^{rT}(C − P)` at the ATM strike). Black-76 for futures options. First-order Greeks: formulas, intuition, shapes vs strike and time; unit conventions (Section 4). |
| Live coding | Vectorized BSM price and Greeks (below); validation against `py_vollib` and put-call parity. |
| Lab | Plot delta, gamma, vega, theta vs strike for 7, 30 and 90 DTE; explain why gamma and theta concentrate near ATM close to expiry. |
| Homework | `implied_forward(chain)` using the 3 strikes nearest ATM; compare with `S·e^{(r−q)T}` from a dividend forecast. |

```python
import numpy as np
from scipy.stats import norm
N, n = norm.cdf, norm.pdf

def _d1d2(S, K, T, r, q, sigma):
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqT)
    return d1, d1 - sigma * sqT

def bsm_price(S, K, T, r, q, sigma, cp):
    """cp = +1 call, -1 put. Black-76: pass S=F and q=r."""
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    return cp * (S * np.exp(-q * T) * N(cp * d1) - K * np.exp(-r * T) * N(cp * d2))
```

#### S4 · Module 6.3b — Implied Volatility

| Block | Content |
|---|---|
| Theory | IV as the market's quote currency. **No-arbitrage bounds** (a price outside them has no IV). Newton–Raphson using vega (fast, fails for deep ITM/OTM where vega → 0) with a **Brent fallback** (guaranteed within a bracket). Jäckel's "Let's Be Rational" (machine precision, used in `py_vollib`). Which price: mid, bid, ask (bid/ask IVs give an IV spread; very wide spreads mean IV is unreliable). **American options:** BSM IV on American puts is biased upward; solve with a tree (S6) or de-Americanize. Vectorizing over a full chain. |
| Live coding | `implied_vol()` (below); chain-wide IV smile; compare to IB `modelGreeks.impliedVol` and `ib.calculateImpliedVolatility`. |
| Lab | IV smile for SPX (European) vs SPY (American) at the same expiry; explain the differences. |
| Homework | Vectorized Newton over a whole chain with a per-element Brent fallback mask; benchmark vs the scalar loop. |

```python
from scipy.optimize import brentq

def implied_vol(price, S, K, T, r, q, cp, lo=1e-4, hi=5.0, tol=1e-10):
    """Newton-Raphson with a Brent fallback. NaN if price violates no-arbitrage bounds."""
    intrinsic = max(cp * (S * np.exp(-q * T) - K * np.exp(-r * T)), 0.0)
    upper = S * np.exp(-q * T) if cp == 1 else K * np.exp(-r * T)
    if not (intrinsic < price < upper):
        return np.nan
    sigma = min(max(np.sqrt(2 * abs(np.log(S / K) + (r - q) * T) / T), 0.05), 2.0)
    for _ in range(20):
        diff = bsm_price(S, K, T, r, q, sigma, cp) - price
        if abs(diff) < tol:
            return sigma
        vega = bsm_greeks(S, K, T, r, q, sigma, cp)["vega"]
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if not lo < sigma < hi:
            break
    return brentq(lambda s: bsm_price(S, K, T, r, q, s, cp) - price, lo, hi, xtol=tol)
```

**Clinic W1 (120 min):** live SPY chain (IB and Alpaca) → our IV and Greeks vs IB `modelGreeks` and Alpaca snapshot Greeks. Deliverable: a table of differences per strike, with each explained (price used, rate, dividends, American vs European model).

---

### Week 2 — Higher-Order Greeks, Numerical Methods, Surfaces, Portfolio

#### S5 · Module 6.4 — Second-Order Greeks

| Block | Content |
|---|---|
| Theory | What each measures and when it matters to a trader: **vanna** (∂Δ/∂σ: delta changes when vol moves; key for skew and dealer hedging flows), **volga/vomma** (∂vega/∂σ: convexity to vol; wings), **charm** (delta decay over time; weekend and expiry-day hedging), **veta** (vega decay), **speed** (∂Γ/∂S), **zomma** (∂Γ/∂σ), **color** (gamma decay), **ultima** (∂volga/∂σ). Verify closed forms with finite differences, always. |
| Live coding | Full Greeks function (below) + finite-difference test for every Greek. |
| Lab | Heatmaps of vanna and charm across strike × DTE; identify where a short-strangle book is most exposed. |
| Homework | Add a `greeks_fd(pricer, ...)` generic bump-and-revalue function that works for any pricer (used in S6). |

```python
def bsm_greeks(S, K, T, r, q, sigma, cp):
    """Raw Greeks (see Section 4 for units)."""
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    sqT, dq, dr = np.sqrt(T), np.exp(-q * T), np.exp(-r * T)
    pdf = n(d1)
    g = {
        "delta": cp * dq * N(cp * d1),
        "gamma": dq * pdf / (S * sigma * sqT),
        "vega": S * dq * pdf * sqT,
        "theta": (-S * dq * pdf * sigma / (2 * sqT)
                  - cp * r * K * dr * N(cp * d2) + cp * q * S * dq * N(cp * d1)),
        "rho": cp * K * T * dr * N(cp * d2),
    }
    # second order; charm, color, veta follow theta's convention: change as time PASSES (-d/dT)
    g["vanna"] = -dq * pdf * d2 / sigma
    g["volga"] = g["vega"] * d1 * d2 / sigma
    g["charm"] = cp * q * dq * N(cp * d1) - dq * pdf * (2 * (r - q) * T - d2 * sigma * sqT) / (2 * T * sigma * sqT)
    g["speed"] = -g["gamma"] / S * (d1 / (sigma * sqT) + 1)
    g["zomma"] = g["gamma"] * (d1 * d2 - 1) / sigma
    g["color"] = dq * pdf / (2 * S * T * sigma * sqT) * (
        2 * q * T + 1 + (2 * (r - q) * T - d2 * sigma * sqT) / (sigma * sqT) * d1)
    g["veta"] = S * dq * pdf * sqT * (q + (r - q) * d1 / (sigma * sqT) - (1 + d1 * d2) / (2 * T))
    g["ultima"] = -g["vega"] / sigma**2 * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
    return g
```

> Verified: price, delta, gamma, vega, theta and rho match `py_vollib` to within 1e-14 (after unit scaling); all eight second-order Greeks match central finite differences to at least 4 significant figures.

#### S6 · Module 6.5a — Numerical Pricing Methods

| Block | Content |
|---|---|
| Theory | When closed forms are not enough: American exercise, discrete dividends, path dependence, stochastic volatility. **Binomial tree (CRR)** with early-exercise check; convergence and odd/even oscillation; trinomial trees. **Bjerksund–Stensland (2002)**: fast closed-form approximation for American options. **Monte Carlo:** risk-neutral GBM paths, antithetic variates, control variates (use the BSM price of a related option), standard error reporting; Longstaff–Schwartz for American (overview). **Finite differences:** Crank–Nicolson on the BSM PDE, grid and boundary choices. **Greeks numerically:** bump-and-revalue (step size vs noise, common random numbers for MC) and **automatic differentiation** (JAX or PyTorch autograd gives exact Greeks of any differentiable pricer). |
| Live coding | CRR American pricer and antithetic Monte Carlo (below); compare with BSM; AD Greeks with `jax.grad` on the BSM formula. |
| Lab | Early-exercise premium of American puts vs moneyness and rates; convergence plot of CRR (steps) and MC (paths). |
| Homework | Crank–Nicolson pricer for a European call; show second-order convergence in grid size. |

```python
def crr_american(S, K, T, r, q, sigma, cp, steps=500):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)              # risk-neutral up probability
    disc = np.exp(-r * dt)
    ST = S * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    V = np.maximum(cp * (ST - K), 0.0)
    for _ in range(steps):
        ST = ST[:-1] / u
        V = np.maximum(disc * (p * V[:-1] + (1 - p) * V[1:]), cp * (ST - K))   # early exercise
    return V[0]

def mc_european(S, K, T, r, q, sigma, cp, n_paths=200_000, seed=0):
    z = np.random.default_rng(seed).standard_normal(n_paths // 2)
    z = np.concatenate([z, -z])                                   # antithetic variates
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    pay = np.exp(-r * T) * np.maximum(cp * (ST - K), 0.0)
    pairs = 0.5 * (pay[: n_paths // 2] + pay[n_paths // 2:])     # antithetic pairs are the i.i.d. units
    return pay.mean(), pairs.std(ddof=1) / np.sqrt(pairs.size)    # price, standard error
```

> Verified (S=100, K=105, T=0.4, r=4%, q=1.5%, σ=27%): MC is within 1 standard error of BSM; the American put is worth ~0.15 more than the European, while the American call is almost equal to the European (small dividend yield).

#### S7 · Module 6.5b — Volatility Surface

| Block | Content |
|---|---|
| Theory | Smile and skew (equity index put skew: crash fear, supply/demand), term structure (event bumps: earnings, FOMC), log-moneyness `k = ln(K/F)` and total variance `w = σ²T`. **Static no-arbitrage:** no butterfly arbitrage (density ≥ 0) and no calendar arbitrage (`w` non-decreasing in T at fixed k). **Raw SVI** per expiry: `w(k) = a + b[ρ(k − m) + √((k − m)² + s²)]`; fit with bounded least squares weighted by vega or 1/spread; check constraints. **SABR** (Hagan approximation) for futures options, with the meaning of α, β, ρ, ν. Local volatility (Dupire) and **Heston** stochastic volatility: concepts and calibration overview. Surface dynamics: sticky-strike vs sticky-delta, and what each implies for delta hedging (links to vanna). |
| Live coding | SVI fit per SPX expiry (below); arbitrage checks; interpolation in total variance across expiries. |
| Lab | Fit the SPX surface for 6 expiries; plot the smile fits and the term structure of ATM IV and 25-delta risk reversal; flag any calendar-arbitrage violations. |
| Homework | Store daily fitted parameters in the database; plot a 3-month history of skew (25Δ put IV − 25Δ call IV). |

```python
from scipy.optimize import least_squares

def svi_total_var(k, a, b, rho, m, s):
    """Raw SVI total implied variance w(k) = sigma^2 * T, k = ln(K/F)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))

def fit_svi(k, iv, T, weights=None):
    w = iv**2 * T
    wt = np.ones_like(w) if weights is None else weights
    res = least_squares(lambda p: wt * (svi_total_var(k, *p) - w),
                        x0=[w.min(), 0.1, -0.5, 0.0, 0.1],
                        bounds=([-1.0, 1e-6, -0.999, -1.0, 1e-4], [1.0, 5.0, 0.999, 1.0, 2.0]))
    a, b, rho, m, s = res.x
    if a + b * s * np.sqrt(1 - rho**2) < 0:
        raise ValueError("SVI fit gives negative total variance")
    return res.x
```

> Verified: on a synthetic smile generated from known SVI parameters, `fit_svi` recovers them to 4 decimal places.

#### S8 · Portfolio Greeks, Scenario Risk & Delta Hedging

| Block | Content |
|---|---|
| Theory | **Dollar Greeks** per position, with multipliers (100 for US equity options, 50 for ES futures options): dollar delta `Δ·S·mult·qty`; gamma as the change in dollar delta for a 1% move `Γ·S²·0.01·mult·qty`; vega $ per vol point; theta $ per day. **Beta-weighted delta** to SPY for mixed books. **Scenario grid:** full revaluation over spot shocks × vol shocks (not a Taylor approximation; compare both). **Delta hedging:** the P&L of a hedged option ≈ ½·Γ·S²·(σ²_realized − σ²_implied)·dt summed over time; hedge frequency trades P&L variance against transaction costs; gamma scalping preview (Part 7). Validation against IB `modelGreeks` for the whole book. |
| Live coding | `OptionPosition` / `Book` aggregation; spot × vol grid; delta-hedging simulator (below). |
| Lab | Delta-hedging study: short 30-DTE ATM call hedged 5, 30 and 250 times; P&L mean and standard deviation; then implied vol 25% vs realized 20%. |
| Homework | Add hedging costs (half-spread + commission) to the simulator and find the rebalance frequency that minimizes cost + risk for a given risk aversion. |

```python
def delta_hedge_pnl(S0=100., K=100., T=30/365, r=0.04, q=0.0, iv=0.20, rv=0.20,
                    rebalances=30, n_paths=5000, seed=0):
    """Sell 1 call at implied vol `iv`, delta-hedge `rebalances` times while the stock
    moves with realized vol `rv`. Returns the final P&L of each path."""
    rng = np.random.default_rng(seed)
    dt = T / rebalances
    S = np.full(n_paths, S0)
    delta = np.full(n_paths, bsm_greeks(S0, K, T, r, q, iv, 1)["delta"])
    cash = bsm_price(S0, K, T, r, q, iv, 1) - delta * S            # premium in, buy delta shares
    for i in range(1, rebalances + 1):
        S = S * np.exp((r - q - 0.5 * rv**2) * dt + rv * np.sqrt(dt) * rng.standard_normal(n_paths))
        cash = cash * np.exp(r * dt) + delta * S * (np.exp(q * dt) - 1)   # interest + dividends
        tau = T - i * dt
        new_delta = bsm_greeks(S, K, tau, r, q, iv, 1)["delta"] if tau > 1e-12 else (S > K).astype(float)
        cash -= (new_delta - delta) * S
        delta = new_delta
    return cash + delta * S - np.maximum(S - K, 0.0)
```

> Verified: with implied = realized vol, mean P&L ≈ 0 and its standard deviation falls roughly as 1/√(rebalances) (0.85 → 0.36 → 0.13 for 5 → 30 → 250); with implied 25% vs realized 20%, the short-vol hedger earns ≈ +0.57 per option.

**Clinic W2:** build the M3a risk report for a sample book (SPY options, ES futures, one ES futures option): dollar Greeks, beta-weighted delta, spot × vol grid, fitted surface snapshot, and differences vs IB `modelGreeks`.

---

## 6. Notebook Map (`notebooks/part06/`)

| Notebook | Session | Promoted to |
|---|---|---|
| `01_futures_chain_continuous.ipynb` | S1 | `futures/roll.py`, `futures/continuous.py` |
| `02_option_chain_strikes.ipynb` | S2 | `options/chain.py`, `options/select.py` |
| `03_bsm_black76_greeks.ipynb` | S3 | `options/pricing.py` |
| `04_implied_vol.ipynb` | S4 | `options/iv.py` |
| `05_second_order_greeks.ipynb` | S5 | `options/greeks.py` |
| `06_trees_mc_fd_autodiff.ipynb` | S6 | `options/numerical.py` |
| `07_vol_surface_svi_sabr.ipynb` | S7 | `options/surface.py` |
| `08_portfolio_greeks_hedging.ipynb` | S8 | `options/portfolio.py`, `research/hedging.py` |

---

## 7. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Mixing vega/theta units (per 1.00 vs per point/day) | Greeks 100× or 365× off vs broker | One unit table (Section 4); tests against IB `modelGreeks` |
| 2 | Indicators on an unadjusted continuous futures series | False signals at every roll | Use adjusted series for signals; roll-date regression test |
| 3 | Trading the adjusted price | Orders at non-existent prices | Orders always reference the actual contract month |
| 4 | Using spot instead of the forward for ATM/moneyness | Skewed smiles, wrong ATM strike | Implied forward from put-call parity |
| 5 | BSM IV for American puts | IV biased high for ITM puts | Tree-based IV or European underlyings (SPX) |
| 6 | IV from stale or one-sided quotes | Spikes and holes in the smile | Filter zero bids, wide spreads; use mid only when spread < threshold |
| 7 | Newton IV diverging for deep OTM options | NaN / huge IVs | Bounds check + Brent fallback |
| 8 | Calendar time vs trading time mixed | Theta and IV inconsistent near weekends | One time convention; document it |
| 9 | Fitting SVI without constraints | Negative variance, butterfly arbitrage | Bounded fit + arbitrage checks |
| 10 | Taylor (delta-gamma) risk for large moves | Underestimated crash loss | Full revaluation scenario grid |
| 11 | Forgetting multipliers in portfolio Greeks | Risk off by 50–100× | Multiplier from contract details, never hard-coded |
| 12 | MC Greeks with different random numbers per bump | Noisy, useless Greeks | Common random numbers or AD |
| 13 | Dividend ignored for single-stock options | Early-exercise and parity errors | Dividend forecast or implied forward per expiry |
| 14 | Assuming Alpaca and IB Greeks use the same model | "Mismatches" that are just model differences | Compare against our own engine with the same inputs |

---

## 8. Assessment — Platform Milestone M3a

**Task:** ship `quantforge.futures` and `quantforge.options` v0.3 plus a risk report.

1. Continuous futures builder with two adjustment methods and a tested roll calendar for ES, NQ, CL.
2. Chain loader for IB and Alpaca into one schema; strike and expiry selectors; liquidity filters.
3. Pricing (BSM, Black-76, CRR American, MC), IV solver, first- and second-order Greeks with finite-difference and `py_vollib` tests.
4. SVI surface fitter with arbitrage checks, run daily on SPX.
5. Portfolio Greeks, beta-weighted delta, spot × vol scenario grid.
6. **Validation report:** our IV and Greeks vs IB `modelGreeks` for ≥ 200 options across 3 underlyings, with differences explained.

| Criterion | Points |
|---|---|
| Futures library: continuous series, roll rules, carry | 10 |
| Chain & strike selection (both brokers, liquidity) | 10 |
| Pricing & IV correctness (tests, bounds, American handling) | 20 |
| Second-order Greeks, verified by finite differences | 10 |
| Volatility surface (SVI fit, arbitrage checks) | 15 |
| Portfolio Greeks & scenario risk | 15 |
| Validation vs IB model Greeks, with differences explained | 10 |
| Code quality, units documented, coverage ≥ 90% | 10 |
| **Total** | **100** |

Pass mark: 70, **and** the unit-convention test suite (Section 4) must pass (mandatory).

---

## 9. Further Reading

| Type | Reference |
|---|---|
| Book | Hull, J. C. (2021). *Options, Futures, and Other Derivatives* (11th ed.). Pearson. |
| Book | Natenberg, S. (2014). *Option Volatility and Pricing* (2nd ed.). McGraw-Hill. |
| Book | Haug, E. G. (2007). *The Complete Guide to Option Pricing Formulas* (2nd ed.). McGraw-Hill. (all Greeks, Bjerksund–Stensland) |
| Book | Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley. (SVI) |
| Book | Sinclair, E. (2013). *Volatility Trading* (2nd ed.). Wiley. (hedging P&L) |
| Book | Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. |
| Paper | Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *JPE*, 81(3). |
| Paper | Black, F. (1976). "The Pricing of Commodity Contracts." *JFE*, 3(1–2). |
| Paper | Cox, J., Ross, S. & Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *JFE*, 7(3). |
| Paper | Hagan, P. et al. (2002). "Managing Smile Risk." *Wilmott Magazine*. (SABR) |
| Paper | Gatheral, J. & Jacquier, A. (2014). "Arbitrage-Free SVI Volatility Surfaces." *Quantitative Finance*, 14(1). |
| Paper | Jäckel, P. (2015). "Let's Be Rational." *Wilmott*, 2015(75). |
| Paper | Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility." *RFS*, 6(2). |
| Docs | IB TWS API: option chains (`reqSecDefOptParams`), option computations (`modelGreeks`); Alpaca Options Trading & Market Data API; `py_vollib`/`vollib`; QuantLib-Python |

---

## 10. Instructor Notes

- Run S2/S4/clinic labs during US market hours; outside hours, IB `modelGreeks` may be missing or stale. Keep recorded chain snapshots (Parquet) as a fallback.
- Emphasize that "our Greeks differ from the broker's" is usually a difference in inputs (price, rate, dividends, time convention, model), not a bug. The clinic W1 difference table teaches exactly this.
- SPX options require an index-options data subscription on IB; XSP (mini-SPX) or recorded snapshots are alternatives.
- AD with JAX: enable 64-bit (`jax.config.update("jax_enable_x64", True)`) or Greeks will disagree with float64 closed forms at around 1e-7.
- No live options trading in Part 6. Option strategies are built in Part 7 and traded live only after the Part 8 risk engine is reviewed.
