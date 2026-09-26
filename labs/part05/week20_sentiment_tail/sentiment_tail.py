"""Week 20 — Market sentiment, news sentiment and tail risk (Part 5, S13–S15).

Everything is offline: VIX/breadth/COT/news inputs are small arrays or DataFrames built in the tests. With real data,
load VIX and VIX3M from IB or FRED, COT from the CFTC public reporting API, and news from Alpaca (see the lesson plan).
Fill in every block marked "Your turn", then run:  python -m pytest week20_sentiment_tail
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import chi2, genpareto


# ------------------------------------------------------------------ S13 market sentiment
def rolling_percentile(x, n: int = 252) -> np.ndarray:
    """Fraction of the last n values (including today) that are <= today's value; NaN for the first n−1 bars.
    (VIX percentile: 0.95 means today's VIX is higher than 95% of the last year.)"""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def term_structure_stress(vix, vix3m) -> tuple[np.ndarray, np.ndarray]:
    """(ratio, inverted): ratio = VIX / VIX3M; inverted = ratio > 1 (short-dated fear above 3-month: stress)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def regime_with_hysteresis(score, enter: float = 1.0, exit: float = 0.5) -> np.ndarray:
    """Label each bar +1 risk-on, 0 neutral, −1 risk-off from a sentiment score (e.g. a z-scored composite where HIGH
    = calm/bullish). Start neutral. From neutral: go −1 when score <= −enter, +1 when score >= enter.
    Leave −1 (to neutral) only when score > −exit; leave +1 only when score < exit (a leave can go straight to the
    opposite regime if that threshold is also crossed). A NaN score keeps the previous label. int array.
    Hysteresis (exit < enter) stops the label flipping every day around one threshold."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def mcclellan(advances, declines) -> np.ndarray:
    """McClellan oscillator = EMA_10%(net) − EMA_5%(net), net = advances − declines, where EMA_α starts at the first
    net value and then e[t] = e[t−1] + α (net[t] − e[t−1]) (α = 0.10 and 0.05)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def pct_above_ma(closes: pd.DataFrame, n: int = 50) -> pd.Series:
    """Breadth: share of symbols (columns) whose close is above their own n-bar SMA, per row. Symbols without an SMA
    yet (warm-up or NaN) are left out of the denominator; NaN when no symbol qualifies."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def cot_available_at(report_date: pd.Series) -> pd.Series:
    """CFTC COT data is 'as of' Tuesday and published Friday 15:30 New York time. From tz-naive Tuesday dates return
    the UTC timestamp when each report became AVAILABLE: date + 3 days, 15:30 America/New_York, converted to UTC."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def pit_join(bars: pd.DataFrame, releases: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Point-in-time join: `bars` has a UTC column 'ts', `releases` a UTC column 'available_at'. Attach to each bar the
    latest release available AT OR BEFORE the bar time (merge_asof, backward). Return bars (sorted by ts) plus
    value_cols. Never join on the reference date: that leaks data published later."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------- S14 news sentiment
NEGATORS = {"not", "no", "never", "without", "nor"}
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
CLAUSE = re.compile(r"[,;:.!?]")


def lexicon_score(text: str, positive: set[str], negative: set[str], window: int = 2) -> float:
    """Dictionary sentiment. Split the lowercased text into clauses with CLAUSE, then tokenize each clause with TOKEN.
    Each positive word counts +1, negative −1, but the sign flips when one of the previous `window` tokens OF THE SAME
    CLAUSE is in NEGATORS ('not good' is negative; in 'not good, lawsuit looms' the 'not' does not reach 'lawsuit').
    score = (Σ signs) / (number of sentiment words), in [−1, 1]; 0.0 when there is none."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def dedupe_news(news: pd.DataFrame) -> pd.DataFrame:
    """The same story arrives from many sources. Normalize each headline (lowercase, drop everything that is not a
    letter, digit or space, collapse whitespace) and keep only the EARLIEST row per (symbol, normalized headline).
    Input columns: ts (UTC), symbol, headline. Return sorted by ts with a fresh index."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def decay_index(event_ts, event_score, bar_ts, tau_hours: float = 24.0) -> np.ndarray:
    """Sentiment at each bar time: S = Σ score_i × exp(−age_i / τ) over news published AT OR BEFORE the bar
    (age in hours). No look-ahead: a story published after the bar never counts. Inputs are datetime64 arrays."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------ S15 tail risk
def hill(returns, k: int = 100) -> float:
    """Hill estimator of the tail index ξ of LOSSES from the k largest losses:
    ξ = mean(ln x_(1..k)) − ln x_(k+1), with x the losses (−returns) sorted descending."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def evt_var_es(returns, q: float = 0.99, threshold_q: float = 0.95) -> dict:
    """Peaks-over-threshold: losses = −returns, u = quantile(losses, threshold_q), excesses = losses − u above u.
    Fit a GPD with genpareto.fit(excesses, floc=0) → (ξ, loc, β). With n losses and n_u excesses:
      VaR_q = u + β/ξ × ((n/n_u × (1−q))^(−ξ) − 1);   ES_q = (VaR_q + β − ξ u) / (1 − ξ)   (valid for 0 < ξ < 1)
    Return {"xi", "beta", "u", "VaR", "ES"} as floats (VaR and ES as positive loss numbers)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def kupiec_pof(exceptions: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test of a VaR model: x exceptions in n days at expected rate p.
    LR = −2 ln[(1−p)^(n−x) p^x] + 2 ln[(1−x/n)^(n−x) (x/n)^x]  (use 0·ln 0 = 0), p-value = chi2(1).sf(LR).
    Return (LR, p_value). A small p-value rejects the VaR model."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def scenario_pnl(positions: dict[str, tuple[float, float]], equity_shock: float) -> dict[str, float]:
    """Stress P&L of positions {symbol: (market value, beta to the index)} for an index move `equity_shock`
    (−0.10 = −10%): pnl = value × beta × shock per symbol, plus 'TOTAL'."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
