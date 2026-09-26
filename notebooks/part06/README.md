# Part 6 guided notebooks — Futures & Options Engineering

Guided notebooks for Part 6 ([lesson plan](../../docs/lessons/PART_06_FUTURES_OPTIONS_ENGINEERING.md)), one per session.
The setup, data and plotting code is written for you; learners fill in short **✍️ Your turn** cells, replacing each
`...`. Each exercise ends with `p.check(...)`, which prints ✅ or ❌. If an answer isn't right yet, the notebook continues
with the reference value, so later cells still run.

All data is synthetic with known parameters, so every estimate can be compared with the truth:
- a futures strip with a known 3% carry;
- a 30-day option chain priced from a known SVI smile, with realistic spreads, zero bids and open interest;
- simulated paths with a known volatility.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part06/`](../../labs/part06/). Here you
*see* why a rule exists: a roll gap that isn't a return, a Newton step that shoots off, a Taylor estimate that misses
40% of a crash. In the labs you build the tested version, checked against py_vollib golden values.

**Units** follow the lesson plan (Section 4): T in years (ACT/365), σ as a decimal, vega per 1.00 σ, theta per year
(−∂V/∂T), rho per 1.00 r, continuous compounding, `cp = +1` call / `−1` put. Notebook 03 converts to broker display
units.

| Notebook | Session | Exercises |
|---|---|---|
| `01_futures_continuous.ipynb` | S1 | Fair value and implied carry; difference back-adjustment; the contract actually held on a date. Plus: the roll gaps in an unadjusted series, what difference and ratio adjustment preserve |
| `02_option_chain.ipynb` | S2 | OCC symbols; a liquidity filter; the ATM strike on the forward. Plus: why "at the money" is at the forward, expected-move strikes |
| `03_pricing_and_greeks.ipynb` | S3 | The BSM price; delta and vega; display units. Plus: put–call parity, Greeks against spot at three expiries, Black-76 = BSM with S = F, q = r |
| `04_implied_vol.ipynb` | S4 | Newton IV; rate and forward from parity. Plus: where Newton fails and Brent takes over, the smile from OTM options with implied vs guessed inputs |
| `05_second_order_greeks.ipynb` | S5 | Gamma by bump-and-revalue; vanna and volga. Plus: the U-shaped finite-difference error, gamma, theta and charm in the last week |
| `06_numerical_methods.ipynb` | S6 | The CRR tree's backward step (American put); antithetic Monte Carlo. Plus: tree convergence, the early-exercise premium, MC delta with and without common random numbers |
| `07_vol_surface.ipynb` | S7 | Raw SVI; calendar-arbitrage check; interpolation in total variance. Plus: an SVI fit that recovers the true parameters, a butterfly-arbitrage check, SABR smiles |
| `08_portfolio_and_hedging.ipynb` | S8 | Dollar delta and vega with multipliers; the delta–gamma–vega approximation; the hedge in shares. Plus: a spot × vol grid, delta-hedging P&L vs realized vol, hedge frequency and costs |

`p6lib.py` holds the reference implementations the checks compare against, the synthetic data and the chart style.
`p.check` compares text and objects **exactly**, and floats and arrays with a tolerance (also inside lists, tuples and
dicts).

## Setup

```bash
cd notebooks/part06
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

All data is generated with fixed seeds, so every notebook runs offline. py_vollib is not needed.

## What the notebooks show

* **Futures rolls.** Each roll adds a contango gap of about 30 points to an unadjusted continuous series. Over the
  sample, those gaps overstate the P&L of holding the front contract by 218 points.
* **Parity.** Put–call parity recovers the rate and the forward within 0.012% and 0.01 points. With those inputs, the
  OTM smile matches the true one within 0.06 vol points.
* **Units.** A call with a raw vega of 68 shows as 0.68 on a broker screen. Comparing the two without converting looks
  like a 100× bug.
* **Implied vol.** Plain Newton from σ = 0.3 fails on a 3-day, far out-of-the-money put. Brent's method recovers the
  true 60%.
* **Finite differences.** The error of a finite-difference gamma is U-shaped in the bump size, with the minimum near
  h ≈ 0.01. A bump of 0.1% of spot is accurate to 4e-7.
* **Early exercise.** The binomial tree gives an early-exercise premium of 0.52 on a one-year ATM American put.
  Antithetic variates cut Monte Carlo error by about 30% for the same number of payoffs.
* **Monte Carlo Greeks.** A bumped Monte Carlo delta is useless with independent random numbers and accurate with
  common ones.
* **Volatility surface.** An SVI fit to 30-day OTM quotes recovers all five true parameters closely. A smile with
  over-steep wings fails the butterfly check.
* **Crash risk.** For a book short OTM puts, the delta–gamma–vega estimate of a −20% / +25-vol crash is −$118k. Full
  revaluation gives −$195k.
* **Delta hedging.** A delta-hedged option earns on average what realized volatility pays over implied. Hedging 120
  times instead of 5 cuts the standard deviation of that P&L from 0.84 to 0.17. At 5 bp a trade, it also triples the
  cost.

## Instructor material

- `solutions/`: the same notebooks with the answers filled in. **Remove this folder (or keep it on a private branch)
  before sharing the repository with learners.**
- `tools/build_notebooks.py`: the single source for starter and solution notebooks (it also contains the answers).
  Edit content there and rebuild with `python tools/build_notebooks.py`.

## Tests

```bash
python -m pytest -q tests     # 24 tests: solutions pass every check (strict mode), starters run with blanks,
                              # starter and solution notebooks differ only in the exercise cells
```
