"""Week 21 (S1–S2) — Futures library (roll calendar, continuous series, carry) and option-chain / strike library.

Signals may use ADJUSTED continuous series; P&L and orders always use the ACTUAL contract.
Chains from IB (`reqTickers` → modelGreeks) and Alpaca (option snapshots) are normalized into ONE schema:
    expiry (date) · strike · cp (+1/−1) · bid · ask · mid · iv · delta · source
Fill in every block marked "Your turn", then run:  python -m pytest week21_futures_chains
"""
from __future__ import annotations

import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

MONTH_CODES = "FGHJKMNQUVXZ"
CHAIN_COLUMNS = ["expiry", "strike", "cp", "bid", "ask", "mid", "iv", "delta", "source"]


# ------------------------------------------------------------------------ S1 roll calendar
def third_friday(year: int, month: int) -> date:
    """Third Friday of the month (ES/NQ quarterly expiry, standard monthly equity options)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def quarterly_expiries(start: date, end: date) -> list[date]:
    """Third Fridays of Mar/Jun/Sep/Dec (codes H M U Z) with start <= expiry <= end, ascending."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def contract_code(root: str, expiry: date) -> str:
    """'ES' + month code + last digit of the year, e.g. ESZ6 for December 2026."""
    return f"{root}{MONTH_CODES[expiry.month - 1]}{expiry.year % 10}"


def roll_date(expiry: date, business_days_before: int = 8) -> date:
    """Go back `business_days_before` weekdays (Mon–Fri; holidays ignored) from the expiry."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def active_contract(day: date, expiries: list[date], business_days_before: int = 8) -> date:
    """Expiry of the contract to hold on `day` under a calendar rule: the first expiry whose roll date is AFTER `day`
    (on the roll date itself you already hold the next contract). ValueError if none."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def volume_roll_dates(volume: pd.DataFrame) -> list[pd.Timestamp]:
    """Volume-crossover roll rule. `volume` has one column per contract (ordered by expiry) and a date index.
    For each consecutive pair (i, i+1): the first date where volume[i+1] > volume[i]. Return those dates in order."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------- S1 continuous series
def continuous(prices: pd.DataFrame, rolls: list, method: str = "difference") -> pd.Series:
    """Stitch contracts (columns, ordered by expiry) into one series. rolls[i] is the date from which contract i+1 is
    used instead of contract i (len(rolls) == ncols − 1).
      'none'        unadjusted: raw prices of the held contract (jumps at rolls)
      'difference'  back-adjusted: at each roll add gap = new[roll] − old[roll] to ALL earlier history
      'ratio'       back-adjusted: multiply all earlier history by new[roll] / old[roll]
    The LAST contract's prices are never changed. Unknown method: ValueError."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def fair_value(S, r, q, T):
    """Futures fair value F = S e^{(r−q)T}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def implied_carry(f_near, f_far, t_near, t_far):
    """Annualized carry (r − q) implied by two futures on the same underlying: ln(F_far/F_near) / (t_far − t_near)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------- S2 option symbology
OCC = re.compile(r"^(?P<root>[A-Z.]{1,6})(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def occ_symbol(root: str, expiry: date, cp: str, strike: float) -> str:
    """OCC symbol: root + YYMMDD + C/P + strike × 1000 as 8 digits, e.g. AAPL261218C00200000."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def parse_occ(symbol: str) -> dict:
    """Inverse of occ_symbol: {"root", "expiry" (date), "cp" (+1/−1), "strike" (float)}; ValueError if malformed."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------ S2 chain normalization
def chain_from_ib(rows: list[dict]) -> pd.DataFrame:
    """IB tickers flattened to dicts with: lastTradeDateOrContractMonth 'YYYYMMDD', strike, right 'C'/'P', bid, ask,
    impliedVol, delta (modelGreeks; may be None). → CHAIN_COLUMNS with source 'ib'. IB sends −1 for a missing bid/ask:
    turn negative bid/ask into NaN. mid = (bid+ask)/2. Missing Greeks → NaN."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def chain_from_alpaca(snapshots: dict[str, dict]) -> pd.DataFrame:
    """Alpaca option snapshots {occ_symbol: {"latest_quote": {"bid_price", "ask_price"}, "implied_volatility",
    "greeks": {"delta", ...} or None}} → CHAIN_COLUMNS with source 'alpaca' (use parse_occ). Sorted by expiry,
    strike, cp."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ---------------------------------------------------------------------- S2 selection
def liquidity_filter(chain: pd.DataFrame, max_spread_pct: float = 0.10, min_oi: int = 0) -> pd.DataFrame:
    """Keep rows with bid > 0, ask >= bid, (ask − bid)/mid <= max_spread_pct and, if the chain has an 'oi' column,
    oi >= min_oi. Returns a copy with a new column spread_pct."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def select_expiry(expiries: list[date], today: date, dte_min: int, dte_max: int, prefer: str = "monthly") -> date:
    """Choose an expiry with dte_min <= (expiry − today).days <= dte_max. prefer='monthly': a third-Friday expiry if
    any qualifies (the nearest one), else the nearest qualifying expiry; prefer='nearest': the nearest qualifying one.
    ValueError if none qualifies."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def atm_strike(strikes, forward: float) -> float:
    """Listed strike nearest to the FORWARD (not spot); ties go to the lower strike."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def strike_by_delta(chain: pd.DataFrame, target: float) -> pd.Series:
    """Row of `chain` (one expiry, one cp) whose delta is closest to target (e.g. −0.25 for a 25-delta put)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def expected_move_strikes(strikes, S: float, sigma: float, T: float) -> tuple[float, float]:
    """Listed strikes nearest to S·(1 − σ√T) and S·(1 + σ√T) (±1 expected move)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
