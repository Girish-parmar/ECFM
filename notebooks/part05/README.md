# Part 5 guided notebooks — Analytics Library: Patterns, Indicators & Sentiment

Guided notebooks for Part 5 ([lesson plan](../../docs/lessons/PART_05_ANALYTICS_LIBRARY.md)). The setup, data and plotting
code is written for you; learners fill in short **✍️ Your turn** cells, replacing each `...`. Each exercise ends with
`p.check(...)`, which prints ✅ or ❌. If an answer isn't right yet, the notebook continues with the reference value, so
later cells still run.

All data is synthetic with a known structure, so learners always know whether an "edge" is real or luck:
- random-walk bars with no planted edge;
- bars with a known volatility;
- intraday bars with U-shaped volume;
- weekly COT releases with publication times;
- fat-tailed returns.

These notebooks are the **exploratory** companion to the auto-graded labs in [`labs/part05/`](../../labs/part05/). Here
you *see* why a rule exists: a seed that differs for 200 bars, a detector that buys the bottom it can't know yet, an
hourly value fifty-five minutes early. In the labs you build the tested version and match TA-Lib to 1e-8 with golden
files.

| Notebook | Sessions | Exercises |
|---|---|---|
| `01_core_indicators.ipynb` | S1–S2 | EMA with TA-Lib's SMA seed; Wilder's RSI smoothing; True Range. Plus: the indicator registry and catalog, the pandas seeding gap, the length / warm-up / look-ahead contract catching a centered average |
| `02_streaming_indicators.ipynb` | S3 | Streaming SMA, streaming EMA, rolling max with a monotonic deque. Plus: streaming vs recompute speed, a Hypothesis property test (and the floating-point case it found), `update` vs `peek` |
| `03_candles_and_edge.ipynb` | S4–S5 | Candle anatomy with the flat-bar guard; bullish engulfing; next-open forward returns; Benjamini–Hochberg. Plus: hammer context, a permutation test, 600 coin-flip tests vs one look-ahead pattern |
| `04_volatility_and_trend.ipynb` | S6–S7 | Parkinson and Garman–Klass volatility; Kaufman's efficiency ratio. Plus: estimator precision on bars with a known σ, KAMA vs EMA in trend and chop |
| `05_vwap_and_timeframes.ipynb` | S8 | Session VWAP; point of control; higher-timeframe alignment without look-ahead. Plus: anchored VWAP, a volume profile, the default-`resample` leak in a toy rule |
| `06_levels_and_swings.ipynb` | S9–S11 | Floor pivots from yesterday's bar; fractal swings with `known_from`. Plus: the swing-bar vs confirmation-bar trap for ZigZag and fractals, a truncation test |
| `07_actions_and_conditions.ipynb` | S12 | Crossover with tie and NaN rules; `bars_since`; `within(n)`. Plus: composing an entry rule with `Condition` and watching signals thin out |
| `08_sentiment_and_tails.ipynb` | S13–S15 | A point-in-time COT join; a lexicon scorer with negation; the Hill tail index. Plus: a decaying news index, normal vs historical vs EVT VaR with Kupiec tests |

`p5lib.py` holds the reference implementations the checks compare against, the synthetic data and the chart style.
`p.check` compares text and objects **exactly**, and floats and arrays with a tolerance (also inside lists and dicts).

## Setup

```bash
cd notebooks/part05
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
jupyter lab
```

All data is generated with fixed seeds, so every notebook runs offline. TA-Lib is not needed.

## What the notebooks show

* **Seeding.** A TA-Lib-seeded EMA(20) and `pandas.ewm(adjust=False)` differ by more than 1e-8 until bar 199.
* **Contract checks.** A centered moving average passes the length check but fails both the warm-up check and the
  truncation (no look-ahead) check.
* **Streaming speed.** Recomputing an EMA over 3,000 bars on every bar is about 1,200× slower than updating it in
  streaming form.
* **Property testing.** Hypothesis found that after a spike leaves the window, a streaming standard deviation keeps an
  error of 1e-5 where the true value is 0. That error is floating point, not a logic bug, so the test's tolerance must
  scale with the size of the data.
* **Updating on ticks.** Calling `update` on every intrabar tick gives an EMA of 97.78 instead of the backtested 96.68.
* **Multiple testing.** Of 600 coin-flip pattern tests, 29 pass at 5%. After Benjamini–Hochberg none pass, but a
  pattern that peeks at the next two opens passes on all 20 symbols.
* **Volatility estimators.** On bars with a known σ, Parkinson is about 5× and Garman–Klass about 8× as efficient as
  close-to-close.
* **Higher timeframes.** The default `resample` hands 5-minute bars an hourly close up to 55 minutes early. That turns
  a toy rule into a +6 bp vs −5 bp "edge" on random data.
* **Swing points.** Buying ZigZag swing lows at the swing bar shows +2.2% per trade; buying at the confirmation bar
  shows −1.9%.
* **Point-in-time data.** Joining COT on its report date leaks unpublished data into 59% of bars.
* **Tail risk.** At 99.9%, normal VaR is exceeded 15 times where 2.5 were expected (Kupiec p < 0.0001). Historical and
  EVT VaR pass.

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
