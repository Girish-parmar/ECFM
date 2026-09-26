# Part 7 labs — Strategy Library

Auto-graded exercises for [Part 7](../../docs/lessons/PART_07_STRATEGY_LIBRARY.md) (program weeks 23–24, milestone **M3b**).
Everything runs offline. The linear labs use a synthetic market whose trend and volatility **regimes are known**
(`common.regime_market`). The option labs use the recorded, synthetic SPY chain from Part 6, in [`data/`](data/).

**No strategy here is presented as profitable.** Each one is a hypothesis with a first-look evaluation. The honest
backtesting, validation and sizing come in Part 8. The synthetic regimes are tuned to realistic strength, with Sharpe
ratios of about 1–2.5 at best inside the right regime and negative in the wrong one.

Fill in each `raise NotImplementedError("✍️ Your turn ...")` (or `pass  # ✍️ Your turn`) and run the tests until they are green.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week23_framework_momentum/](week23_framework_momentum/) | S1–S2 | `Strategy` base class (Template Method, declared parameter ranges), registry, YAML config loader ([`configs/`](configs/)), runner whose context only shows past bars, next-bar first-look evaluator, the **mandatory next-bar execution test**, TSMOM, Donchian breakout (vectorized and live-style must agree), 12-1 cross-sectional momentum, dual momentum | `python -m pytest week23_framework_momentum` |
| [week23_linear_groups/](week23_linear_groups/) | S3–S4 | RSI(2) with trend filter and time stop, IBS, z-score reversion, Bollinger fade gated by ADX, opening-range breakout on intraday bars, volatility-target and VIX-regime overlays, Kalman level, Hurst exponent, rolling pairs trade (no look-ahead in the hedge ratio), calendar-based turn-of-month, overnight vs intraday split | `python -m pytest week23_linear_groups` |
| [week24_option_builder/](week24_option_builder/) | S5–S6 | Option strategy Builder (value, payoff, net Greeks, breakevens, max P/L, POP; reproduces the lesson plan's iron condor), defined-risk check, templates (iron condor by delta, verticals, collar, calendar), IV rank and percentile, structure selector, combo pricing (net mid vs natural), IB `BAG` and Alpaca multi-leg orders | `python -m pytest week24_option_builder` |
| [week24_vol_hedging/](week24_vol_hedging/) | S7–S8 | Implied move and the 0.8·S·σ·√T rule, event study, research-only VRP vs a tradable VRP signal (with a truncation test), futures hedge sizing, zero-cost collar strike, hedging study through a crash (none / puts / collar / futures) with cost vs drawdown | `python -m pytest week24_vol_hedging` |
| [clinic_w1_regime_map/](clinic_w1_regime_map/) | Clinic W1 | The regime map: 7 strategies (one per group) × 4 regimes, with true labels and with labels estimated from the data | `python regime_map.py` |
| [clinic_w2_option_playbook/](clinic_w2_option_playbook/) | Clinic W2 | Option playbook from the chain: bull call spread, 16-delta iron condor and collar, analyzed with Greeks and a spot × vol grid, with combo orders at the net mid | `python playbook.py` |

`common.py` holds complete reference indicators (Part 5), BSM pricing and IV (Part 6), the regime market and the chain loader,
so these labs do not depend on your earlier code. The clinics use this part's weekly labs.

## Setup

```bash
cd labs/part07
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week23_framework_momentum
python -m pytest                          # everything
```

Run all commands from `labs/part07` (its `conftest.py` makes the imports work).

## Things the labs make you notice

* **Same-bar fills are fantasy.** The same "today was up" signal has a Sharpe above 10 when filled on its own bar and
  an ordinary one when filled at the next open.
* **Regimes are easy to see afterwards and hard to see at the time.** With the true labels, the regime map is clean:
  trend followers win in trends, and faders win in ranges. With labels estimated from the data (ADX, realized vol), the
  picture blurs. The trend labels are only about 63% right in the demo.
* **Look-ahead can hide in a calendar.** Counting rows back from the end of a month flags the last *available* day of an
  unfinished month. The turn-of-month mask must come from the business-day calendar.
* **Win rate is not risk.** The playbook's iron condor has a 78% POP and a max loss 8× its credit.
* **A hedge that expires before the crash is no hedge.** Rolling 3-month puts only at expiry left a gap four days into
  the crash, so the drawdown was no better than unhedged. Rolling monthly cut it from −40% to −28%. In a rising market,
  every hedge costs return.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P7_SOLUTIONS=1 python -m pytest` (42 tests pass). On the blank starters, failures are
  `NotImplementedError`, except the registry test, which explains that `register()` does not store strategies yet.
* **Before sharing with learners, remove `solutions/` and `tools/`.** Keep `data/` and `configs/`. The sync test then skips itself.
* The playbook's combo orders are built offline (`ib_async` objects and an Alpaca payload dict). Check your `alpaca-py`
  version's multi-leg order support before paper-trading them through the Part 4 OMS.
