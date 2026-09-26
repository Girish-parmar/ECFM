# Part 6 labs — Futures & Options Engineering

Auto-graded exercises for [Part 6](../../docs/lessons/PART_06_FUTURES_OPTIONS_ENGINEERING.md) (program weeks 21–22, milestone **M3a**).
Everything runs offline. Pricing and Greeks are checked against **vollib / py_vollib** values stored in [`golden/`](golden/),
so learners do not need vollib. The clinics use a recorded, synthetic SPY-like chain in [`data/`](data/). Its prices come from a
known SVI surface, and its IB-style `modelGreeks` are in display units.

Fill in each `raise NotImplementedError("✍️ Your turn ...")` and run the tests until they are green.

**Units (lesson plan, Section 4).** The library works in raw units: T in years (ACT/365), σ as a decimal, vega per 1.00 of σ,
theta per year (−∂V/∂T) and rho per 1.00 of r. Brokers and py_vollib show vega per vol point, theta per day and rho per 1%.
The labs test both, and `to_display` converts between them.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week21_futures_chains/](week21_futures_chains/) | S1–S2 | Third Fridays, quarterly expiries, contract codes, calendar and volume-crossover roll rules, continuous series (unadjusted, difference and ratio back-adjusted, each tested for what it preserves), fair value and implied carry, OCC symbols, IB and Alpaca chains in one schema, liquidity filter, expiry and strike selection (ATM on the forward, by delta, ±1 expected move) | `python -m pytest week21_futures_chains` |
| [week21_pricing_iv/](week21_pricing_iv/) | S3–S4 | BSM and Black-76 prices and first-order Greeks (golden vs vollib after unit conversion), put–call parity, implied forward (dividend-agnostic), no-arbitrage bounds, IV by Newton with a Brent fallback, vectorized chain IV | `python -m pytest week21_pricing_iv` |
| [week22_greeks_numerics/](week22_greeks_numerics/) | S5–S6 | Eight second-order Greeks verified by finite differences, generic bump-and-revalue, CRR American tree (early-exercise premium), antithetic Monte Carlo with standard error, MC delta with and without common random numbers, Crank–Nicolson with Rannacher start-up and a cell-averaged payoff (second-order convergence) | `python -m pytest week22_greeks_numerics` |
| [week22_surface_portfolio/](week22_surface_portfolio/) | S7–S8 | Raw SVI fit (recovers known parameters), Durrleman butterfly check, calendar-arbitrage check, total-variance interpolation, Hagan SABR, dollar Greeks with multipliers (Black-76 for futures options), beta-weighted book report, full-revaluation spot × vol grid vs the delta–gamma–vega approximation, delta-hedging simulator with costs | `python -m pytest week22_surface_portfolio` |
| [clinic_w1_chain_vs_broker/](clinic_w1_chain_vs_broker/) | Clinic W1 | Our IV and Greeks vs the broker's on a recorded chain: large differences with guessed inputs, then the rate and forward recovered from put–call parity make them vanish on out-of-the-money options | `python compare.py` |
| [clinic_w2_risk_report/](clinic_w2_risk_report/) | Clinic W2 | The M3a risk report as Markdown: SVI surface snapshot (ATM IV, 25-delta risk reversal, arbitrage checks), dollar Greeks, beta-weighted delta, spot × vol grid and the worst scenario | `python risk_report.py` |

Later labs use earlier ones: the week 22 labs import your week 21 pricing module, clinic W1 uses both week 21 labs, and
clinic W2 uses week 22 and clinic W1.

## Setup

```bash
cd labs/part06
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week21_pricing_iv        # one lab
python -m pytest                          # everything
```

Run all commands from `labs/part06` (its `conftest.py` makes the imports work).

## Things the labs make you notice

* **"Our Greeks differ from the broker's" is usually about inputs.** In clinic W1, guessing r = 4% and no dividend gives IV
  differences of up to 5–10 vol points. Recovering r and the forward from put–call parity (by regressing C − P on K) brings
  out-of-the-money IVs within 1e-4.
* **Deep in-the-money options have (almost) no time value.** Their IV is ill-conditioned, so the smile is read from OTM options.
* **The delta–gamma approximation fails in both directions for large moves.** For a short OTM put it understates a −20% crash
  loss by more than 3×. For a short ATM straddle it overstates the loss. Always revalue fully.
* **Monte Carlo Greeks need common random numbers.** Bumping with independent seeds gives a delta that is pure noise.
* **Numerical PDE details matter.** Where the strike sits on the grid decides whether Crank–Nicolson converges cleanly.
  Cell-averaging the payoff fixes it.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P6_SOLUTIONS=1 python -m pytest` (51 tests pass). On the blank starters every failure is a
  `NotImplementedError`; the two clinic W2 tests that share a fixture report errors rather than failures.
* `tools/make_data.py` regenerates `golden/vollib.npz` (needs `pip install vollib`) and `data/spy_chain.csv`. It uses the
  solution code and asserts that the generating surface has no calendar arbitrage.
* **Before sharing with learners, remove `solutions/` and `tools/`.** Keep `golden/` and `data/`. The sync test then skips itself.
* The chain is synthetic. For the real clinic, record a live SPY or SPX chain during US market hours (lesson plan, Section 10)
  and run the same functions on it.
