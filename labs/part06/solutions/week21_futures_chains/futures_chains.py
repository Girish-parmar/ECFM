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
    # >>> SOLUTION
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)
    # <<< SOLUTION


def quarterly_expiries(start: date, end: date) -> list[date]:
    """Third Fridays of Mar/Jun/Sep/Dec (codes H M U Z) with start <= expiry <= end, ascending."""
    # >>> SOLUTION
    out = []
    for y in range(start.year, end.year + 1):
        for m in (3, 6, 9, 12):
            d = third_friday(y, m)
            if start <= d <= end:
                out.append(d)
    return out
    # <<< SOLUTION


def contract_code(root: str, expiry: date) -> str:
    """'ES' + month code + last digit of the year, e.g. ESZ6 for December 2026."""
    return f"{root}{MONTH_CODES[expiry.month - 1]}{expiry.year % 10}"


def roll_date(expiry: date, business_days_before: int = 8) -> date:
    """Go back `business_days_before` weekdays (Mon–Fri; holidays ignored) from the expiry."""
    # >>> SOLUTION
    d, left = expiry, business_days_before
    while left:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            left -= 1
    return d
    # <<< SOLUTION


def active_contract(day: date, expiries: list[date], business_days_before: int = 8) -> date:
    """Expiry of the contract to hold on `day` under a calendar rule: the first expiry whose roll date is AFTER `day`
    (on the roll date itself you already hold the next contract). ValueError if none."""
    # >>> SOLUTION
    for e in sorted(expiries):
        if day < roll_date(e, business_days_before):
            return e
    raise ValueError(f"no contract to hold on {day}")
    # <<< SOLUTION


def volume_roll_dates(volume: pd.DataFrame) -> list[pd.Timestamp]:
    """Volume-crossover roll rule. `volume` has one column per contract (ordered by expiry) and a date index.
    For each consecutive pair (i, i+1): the first date where volume[i+1] > volume[i]. Return those dates in order."""
    # >>> SOLUTION
    cols = list(volume.columns)
    out = []
    for a, b in zip(cols, cols[1:]):
        cross = volume.index[volume[b] > volume[a]]
        if len(cross):
            out.append(cross[0])
    return out
    # <<< SOLUTION


# ------------------------------------------------------------------- S1 continuous series
def continuous(prices: pd.DataFrame, rolls: list, method: str = "difference") -> pd.Series:
    """Stitch contracts (columns, ordered by expiry) into one series. rolls[i] is the date from which contract i+1 is
    used instead of contract i (len(rolls) == ncols − 1).
      'none'        unadjusted: raw prices of the held contract (jumps at rolls)
      'difference'  back-adjusted: at each roll add gap = new[roll] − old[roll] to ALL earlier history
      'ratio'       back-adjusted: multiply all earlier history by new[roll] / old[roll]
    The LAST contract's prices are never changed. Unknown method: ValueError."""
    # >>> SOLUTION
    if method not in ("none", "difference", "ratio"):
        raise ValueError(f"unknown method {method!r}")
    cols = list(prices.columns)
    held = np.searchsorted(pd.DatetimeIndex(rolls).to_numpy(), prices.index.to_numpy(), side="right")
    out = prices.to_numpy()[np.arange(len(prices)), held].astype(float)
    for i, d in enumerate(rolls):
        old, new = prices.at[d, cols[i]], prices.at[d, cols[i + 1]]
        before = held <= i
        if method == "difference":
            out[before] += new - old
        elif method == "ratio":
            out[before] *= new / old
    return pd.Series(out, index=prices.index, name=method)
    # <<< SOLUTION


def fair_value(S, r, q, T):
    """Futures fair value F = S e^{(r−q)T}."""
    # >>> SOLUTION
    return S * np.exp((r - q) * T)
    # <<< SOLUTION


def implied_carry(f_near, f_far, t_near, t_far):
    """Annualized carry (r − q) implied by two futures on the same underlying: ln(F_far/F_near) / (t_far − t_near)."""
    # >>> SOLUTION
    return np.log(f_far / f_near) / (t_far - t_near)
    # <<< SOLUTION


# --------------------------------------------------------------------- S2 option symbology
OCC = re.compile(r"^(?P<root>[A-Z.]{1,6})(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def occ_symbol(root: str, expiry: date, cp: str, strike: float) -> str:
    """OCC symbol: root + YYMMDD + C/P + strike × 1000 as 8 digits, e.g. AAPL261218C00200000."""
    # >>> SOLUTION
    return f"{root}{expiry:%y%m%d}{cp}{int(round(strike * 1000)):08d}"
    # <<< SOLUTION


def parse_occ(symbol: str) -> dict:
    """Inverse of occ_symbol: {"root", "expiry" (date), "cp" (+1/−1), "strike" (float)}; ValueError if malformed."""
    # >>> SOLUTION
    m = OCC.match(symbol)
    if not m:
        raise ValueError(f"not an OCC symbol: {symbol!r}")
    ymd = m["ymd"]
    return {"root": m["root"], "expiry": date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:])),
            "cp": 1 if m["cp"] == "C" else -1, "strike": int(m["strike"]) / 1000}
    # <<< SOLUTION


# ------------------------------------------------------------------ S2 chain normalization
def chain_from_ib(rows: list[dict]) -> pd.DataFrame:
    """IB tickers flattened to dicts with: lastTradeDateOrContractMonth 'YYYYMMDD', strike, right 'C'/'P', bid, ask,
    impliedVol, delta (modelGreeks; may be None). → CHAIN_COLUMNS with source 'ib'. IB sends −1 for a missing bid/ask:
    turn negative bid/ask into NaN. mid = (bid+ask)/2. Missing Greeks → NaN."""
    # >>> SOLUTION
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "expiry": pd.to_datetime(df["lastTradeDateOrContractMonth"], format="%Y%m%d").dt.date,
        "strike": df["strike"].astype(float),
        "cp": np.where(df["right"] == "C", 1, -1),
        "bid": df["bid"].astype(float).where(df["bid"].astype(float) >= 0),
        "ask": df["ask"].astype(float).where(df["ask"].astype(float) >= 0),
        "iv": pd.to_numeric(df["impliedVol"], errors="coerce"),
        "delta": pd.to_numeric(df["delta"], errors="coerce"),
    })
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["source"] = "ib"
    return out[CHAIN_COLUMNS]
    # <<< SOLUTION


def chain_from_alpaca(snapshots: dict[str, dict]) -> pd.DataFrame:
    """Alpaca option snapshots {occ_symbol: {"latest_quote": {"bid_price", "ask_price"}, "implied_volatility",
    "greeks": {"delta", ...} or None}} → CHAIN_COLUMNS with source 'alpaca' (use parse_occ). Sorted by expiry,
    strike, cp."""
    # >>> SOLUTION
    rows = []
    for sym, s in snapshots.items():
        p = parse_occ(sym)
        q = s.get("latest_quote") or {}
        g = s.get("greeks") or {}
        bid, ask = q.get("bid_price", np.nan), q.get("ask_price", np.nan)
        rows.append({"expiry": p["expiry"], "strike": p["strike"], "cp": p["cp"], "bid": bid, "ask": ask,
                     "mid": (bid + ask) / 2, "iv": s.get("implied_volatility", np.nan) or np.nan,
                     "delta": g.get("delta", np.nan), "source": "alpaca"})
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS).sort_values(["expiry", "strike", "cp"], ignore_index=True)
    # <<< SOLUTION


# ---------------------------------------------------------------------- S2 selection
def liquidity_filter(chain: pd.DataFrame, max_spread_pct: float = 0.10, min_oi: int = 0) -> pd.DataFrame:
    """Keep rows with bid > 0, ask >= bid, (ask − bid)/mid <= max_spread_pct and, if the chain has an 'oi' column,
    oi >= min_oi. Returns a copy with a new column spread_pct."""
    # >>> SOLUTION
    c = chain.copy()
    c["spread_pct"] = (c["ask"] - c["bid"]) / c["mid"]
    keep = (c["bid"] > 0) & (c["ask"] >= c["bid"]) & (c["spread_pct"] <= max_spread_pct)
    if "oi" in c.columns:
        keep &= c["oi"] >= min_oi
    return c[keep]
    # <<< SOLUTION


def select_expiry(expiries: list[date], today: date, dte_min: int, dte_max: int, prefer: str = "monthly") -> date:
    """Choose an expiry with dte_min <= (expiry − today).days <= dte_max. prefer='monthly': a third-Friday expiry if
    any qualifies (the nearest one), else the nearest qualifying expiry; prefer='nearest': the nearest qualifying one.
    ValueError if none qualifies."""
    # >>> SOLUTION
    ok = sorted(e for e in expiries if dte_min <= (e - today).days <= dte_max)
    if not ok:
        raise ValueError("no expiry in the DTE window")
    if prefer == "monthly":
        monthly = [e for e in ok if e == third_friday(e.year, e.month)]
        if monthly:
            return monthly[0]
    return ok[0]
    # <<< SOLUTION


def atm_strike(strikes, forward: float) -> float:
    """Listed strike nearest to the FORWARD (not spot); ties go to the lower strike."""
    # >>> SOLUTION
    k = np.sort(np.asarray(strikes, dtype=float))
    return float(k[np.argmin(np.abs(k - forward))])
    # <<< SOLUTION


def strike_by_delta(chain: pd.DataFrame, target: float) -> pd.Series:
    """Row of `chain` (one expiry, one cp) whose delta is closest to target (e.g. −0.25 for a 25-delta put)."""
    # >>> SOLUTION
    return chain.loc[(chain["delta"] - target).abs().idxmin()]
    # <<< SOLUTION


def expected_move_strikes(strikes, S: float, sigma: float, T: float) -> tuple[float, float]:
    """Listed strikes nearest to S·(1 − σ√T) and S·(1 + σ√T) (±1 expected move)."""
    # >>> SOLUTION
    k = np.sort(np.asarray(strikes, dtype=float))
    move = S * sigma * np.sqrt(T)
    return float(k[np.argmin(np.abs(k - (S - move)))]), float(k[np.argmin(np.abs(k - (S + move)))])
    # <<< SOLUTION
