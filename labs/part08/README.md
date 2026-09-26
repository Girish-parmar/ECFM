# Part 8 labs — Backtesting, Risk & Portfolio

Auto-graded exercises for [Part 8](../../docs/lessons/PART_08_BACKTESTING_RISK_PORTFOLIO.md) (program weeks 25–30,
milestones **M4** and **M5a**). Everything runs offline on synthetic data from `common.py`: a regime market for the
strategies, and strategy return streams with known Sharpes and correlations for the portfolio labs.

Fill in each `raise NotImplementedError("✍️ Your turn ...")` and run the tests until they are green.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week25_engine/](week25_engine/) | S1–S4 | Config hashing and a DuckDB research log, event queue (fills before bars before timers, deterministic ties), portfolio ledger with futures multipliers, fill models (next-open market with slippage, limit through, stop at the worse of stop and gap, participation cap), IB fixed commission, square-root impact, and an event-driven backtester that reconciles with the Part 7 first-look evaluator | `python -m pytest week25_engine` |
| [week26_analysis/](week26_analysis/) | S5–S8 | Point-in-time universes (survivorship), drawdown and duration, tear sheet (Sharpe, Sortino, Calmar, tail ratio), monthly table, trades with MAE/MFE, trade statistics, Sharpe standard error, PSR, minimum track record, stationary bootstrap, drawdown distribution, capacity curve, implementation shortfall | `python -m pytest week26_analysis` |
| [week27_optimization/](week27_optimization/) | S9–S12 | Grid and random search, plateau scoring (broad plateaus beat sharp peaks), rolling and anchored walk-forward, walk-forward optimization with a stitched OOS curve and efficiency, purged k-fold with embargo, CPCV splits and path counts | `python -m pytest week27_optimization` |
| [week28_overfitting/](week28_overfitting/) | S13–S16 | Expected maximum Sharpe of N trials, the Deflated Sharpe Ratio, PBO by CSCV (noise ≈ 0.5, real skill low), robustness scorecard (parameter ±20%, one-bar delay, 2× costs, without the best days, both halves), the validation gate and its dossier | `python -m pytest week28_overfitting` |
| [week29_risk_sizing/](week29_risk_sizing/) | S17–S20 | Risk engine (chain of rules with reasons and a rejection log: notional, gross leverage, single-name and sector concentration, ADV participation, daily loss), historical, parametric and component VaR, sizing (risk per trade, vol target, Turtle units), risk of ruin, Kelly (discrete, continuous, multi-asset) and a Kelly simulation with estimation error | `python -m pytest week29_risk_sizing` |
| [week30_portfolio/](week30_portfolio/) | S21–S24 | Inverse volatility, diversification ratio, minimum variance (closed form and long-only with cvxpy), Ledoit–Wolf shrinkage, max Sharpe, risk contributions, risk parity (Spinu), HRP, Black–Litterman, minimum-CVaR linear program, walk-forward allocation with turnover | `python -m pytest week30_portfolio` |
| [clinic_w1_parity/](clinic_w1_parity/) | Clinic W1 | Three Part 7 strategies on the engine: parity report vs the first look, and Sharpe as costs rise | `python parity.py` |
| [clinic_w4_validation/](clinic_w4_validation/) | Clinic W4 | Full validation dossiers: every configuration logged, walk-forward, PBO, DSR against all trials, robustness, gate. A "noise miner" must be rejected | `python validation.py` |
| [clinic_w6_allocators/](clinic_w6_allocators/) | Clinic W6 | Walk-forward comparison of six allocators and a recommendation rule fixed in advance | `python allocators.py` |

Later labs use earlier ones: week 28 uses the week 26 PSR, and the clinics use the weeks they follow.

## Setup

```bash
cd labs/part08
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week25_engine
python -m pytest                          # everything
```

Run all commands from `labs/part08` (its `conftest.py` makes the imports work).

## Things the labs make you notice

* **Ten years is short.** The synthetic strategies have known annual Sharpes of 0.3–0.8. Over ten simulated years, their measured Sharpes range from 0.05 to 0.9.
* **Most strategies should fail the gate.** In clinic W4, TSMOM, the SMA crossover and the noise miner all fail on the synthetic market. The noise miner even shows a lucky out-of-sample Sharpe of 0.47. PBO (0.73) and robustness (29% of tests passed) catch it.
* **Costs eat thin edges first.** In clinic W1, RSI(2)'s Sharpe drops from 0.09 to about 0 at one unit of costs, while TSMOM keeps about 80%.
* **Kelly is an upper bound.** With μ estimated from five years of data, 2× Kelly ends with a median wealth of about 0.1. ½ Kelly grows the most, and full Kelly's median drawdown is worse than −80%.
* **Simple often wins out of sample.** In clinic W6, 1/N has the best out-of-sample Sharpe and the lowest turnover, as DeMiguel et al. (2009) found. The risk-based allocators buy lower volatility.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P8_SOLUTIONS=1 python -m pytest` (50 tests pass). On the blank starters, every failure is a
  `NotImplementedError`; tests that share a fixture report errors instead.
* **Before sharing with learners, remove `solutions/` and `tools/`.** The sync test then skips itself.
* The IB commission follows the lesson plan's figures ($0.005/share, $1 minimum, 1% cap). Check the current schedule.
