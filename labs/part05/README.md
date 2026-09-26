# Part 5 labs — Analytics Library: Patterns, Indicators & Sentiment

Auto-graded exercises for [Part 5](../../docs/lessons/PART_05_ANALYTICS_LIBRARY.md) (program weeks 17–20, milestone **M2**).
Everything runs offline on reproducible synthetic bars (`common.synthetic_ohlcv`). For graded research, swap in your
cached Part 4 Parquet bars (same column names).

Fill in each `raise NotImplementedError("✍️ Your turn ...")` (or `pass  # ✍️ Your turn`) and run the tests until they are green.
The library conventions from the lesson plan are tested everywhere: NumPy in and out, output length = input length,
NaN only during the declared look-back, and **no look-ahead**.

| Folder | Sessions | What you build | Run |
|---|---|---|---|
| [week17_core/](week17_core/) | S1–S4 | Indicator registry and Markdown catalog, helpers, SMA/EMA/RSI/ATR/MACD/Bollinger/Stochastic/Williams %R/OBV **matching TA-Lib to 1e-8** (golden files, including TA-Lib's MACD and Stochastic alignment), streaming EMA/RSI/rolling-max/Bollinger (rolling Welford) proven equal to the vectorized versions with Hypothesis, candlestick anatomy, doji, engulfing, hammer with trend context, morning/evening star | `python -m pytest week17_core` |
| [week18_groups/](week18_groups/) | S5–S8 | Next-open forward returns, permutation edge test, Benjamini–Hochberg (checked against SciPy), KAMA and ADX/DMI (golden), SuperTrend, five volatility estimators checked on simulated Brownian bars (Yang–Zhang handles overnight gaps), CCI/ROC/MFI/A-D (golden), session and anchored VWAP, volume profile, higher-timeframe alignment without look-ahead | `python -m pytest week18_groups` |
| [week19_patterns/](week19_patterns/) | S9–S12 | Classic/Fibonacci/Camarilla/Woodie pivots, opening range (known only once complete), KDE support/resistance, ZigZag and fractals with **confirmation indices**, double tops/bottoms, head & shoulders, actions (crossovers with tie rules, gaps, inside bars, NR-n, new highs, `bars_since`), `Condition` with `within(n)` and `confirm(n)` | `python -m pytest week19_patterns` |
| [week20_sentiment_tail/](week20_sentiment_tail/) | S13–S15 | VIX percentile and term structure, regime labels with hysteresis, McClellan oscillator, % above MA, COT publication-time join, clause-aware lexicon scoring, news de-duplication, decay-weighted news index, Hill and EVT (GPD) VaR/ES, Kupiec test, scenario P&L | `python -m pytest week20_sentiment_tail` |
| [clinic_w1_audit/](clinic_w1_audit/) | Clinic W1 | Generic property audit (length, warm-up, no look-ahead, scale invariance) run over the registry, catching a "hall of shame" of buggy indicators | `python audit.py` |
| [clinic_w2_edge_study/](clinic_w2_edge_study/) | Clinic W2 | Edge table of candlestick patterns across a universe with BH-FDR, where honest patterns show nothing and a pattern that peeks two bars ahead looks spectacular | `python edge_study.py` |
| [clinic_w3_scanner/](clinic_w3_scanner/) | Clinic W3 | Chart-pattern scanner with breakouts, plus a truncation audit that passes for the honest scanner and catches the pivot-index bug | `python scanner.py` |
| [clinic_w4_composite/](clinic_w4_composite/) | Clinic W4 | Daily sentiment composite (trailing z-scores, sign-aligned), forward returns by quintile, stress report over historical one-day index moves and hypothetical shocks | `python composite.py` |

The weeks build on each other: week 18 uses your week 17 `sma`/`atr`, and each clinic uses that week's lab code.

## Setup

```bash
cd labs/part05
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python -m pytest week17_core              # one lab
python -m pytest                          # everything
```

Run all commands from `labs/part05` (its `conftest.py` makes the imports work). TA-Lib is **not** needed by learners:
the reference values are stored in [`golden/`](golden/).

## Things the labs make you notice

* **Seeding decides the first values.** TA-Lib seeds an EMA with the SMA of the first *n* values; `pandas.ewm(adjust=False)`
  seeds with the first value. The two differ at the start and agree later (a test shows both).
* **Floating-point equivalence is relative to scale.** The streaming Bollinger test allows 1e-7 of the largest price. A running
  sum of squares loses precision after a large value leaves the window; rolling Welford keeps more of it.
* **FDR is a rate, not a guarantee.** In the full W2 demo (20 symbols, horizons 1 and 5), BH at 10% makes 43 discoveries.
  Forty are the look-ahead cheat and three are false engulfing discoveries, which is within the 10% the method allows.
* **Most classic patterns show no edge** once you test them honestly. The skill is the testing method.

## For instructors

* `solutions/` holds the complete answers. The learner files are **generated** from them: run `python tools/make_starters.py`
  after editing a solution; `test_starters_in_sync.py` fails if you forget.
* Grade against the solutions: `P5_SOLUTIONS=1 python -m pytest` (71 tests pass). On the blank starters, the clinic W2
  tests report errors rather than failures because their shared fixture calls `edge_table`.
* `tools/make_golden.py` regenerates `golden/*.npz` with TA-Lib (`pip install TA-Lib`). Run it when the TA-Lib version changes.
* **Before sharing with learners, remove `solutions/` and `tools/`.** Keep `golden/`. The sync test then skips itself.
* The historical shock table in clinic W4 lists approximate S&P 500 one-day moves; check them against your own data
  before using them in a graded report.
