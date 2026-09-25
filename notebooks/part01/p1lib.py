"""Helper functions for the Part 1 lab notebooks (MFAAT, Month 1).

Learners do not need to edit this file in Part 1: the notebooks call these functions and
learners change only the inputs marked "Change me". From Part 3 onward you will write code
like this yourself.

Offline mode: if the environment variable P1_FIXTURES points to a folder of CSV/JSON files,
data is read from there instead of the internet (used by the test suite and when you are
working without a connection).
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# --------------------------------------------------------------------------------------
# Plot style (course palette, thin lines, recessive grid)
# --------------------------------------------------------------------------------------
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL_SHADE = "#d9d8d2"      # recessions / highlighted periods (neutral grey, not a series colour)


def use_course_style() -> None:
    """Apply the course chart style to matplotlib."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=PALETTE),
        "figure.figsize": (9, 4.5),
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#52514e",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "text.color": "#0b0b0b",
        "legend.frameon": False,
    })


def shade_periods(ax, flag: pd.Series, label: str = "Recession") -> None:
    """Shade the periods where a 0/1 series equals 1 (e.g. NBER recessions)."""
    flag = flag.fillna(0).astype(int)
    starts = flag.index[(flag.diff() == 1) | ((flag == 1) & (flag.index == flag.index[0]))]
    ends = flag.index[flag.diff() == -1]
    for i, s in enumerate(starts):
        e = ends[ends > s][0] if (ends > s).any() else flag.index[-1]
        ax.axvspan(s, e, color=NEUTRAL_SHADE, alpha=1.0, lw=0, zorder=0, label=label if i == 0 else None)


# --------------------------------------------------------------------------------------
# Data: FRED (latest values) and ALFRED (historical vintages, point-in-time)
# --------------------------------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
CACHE_DIR = Path(__file__).parent / "data"


def _fixtures() -> Path | None:
    p = os.environ.get("P1_FIXTURES")
    return Path(p) if p else None


def fred_series(series_id: str, start: str | None = None, max_age_hours: float = 24) -> pd.Series:
    """Latest-vintage series from FRED (no API key needed).

    WARNING: these are today's (revised) values, NOT what was known at the time.
    Use `alfred_vintages` for point-in-time history.
    """
    fx = _fixtures()
    if fx is not None:
        df = pd.read_csv(fx / f"{series_id}.csv", index_col=0, parse_dates=True)
    else:
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / f"{series_id}.csv"
        fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600
        if fresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = pd.read_csv(FRED_CSV.format(series=series_id), index_col=0, parse_dates=True, na_values=".")
            df.to_csv(path)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(series_id)
    s.index.name = "date"
    return s.loc[start:] if start else s


def _parse_rt(x: pd.Series) -> pd.Series:
    # FRED uses 9999-12-31 for "still current", which is outside pandas' date range.
    return pd.to_datetime(x.replace("9999-12-31", "2262-04-01"))


def alfred_vintages(series_id: str, observation_start: str = "2015-01-01",
                    api_key: str | None = None) -> pd.DataFrame:
    """Every published vintage of a series from the FRED/ALFRED API.

    Needs a free API key (https://fred.stlouisfed.org/docs/api/api_key.html) in the
    environment variable FRED_API_KEY or passed as `api_key`.

    Returns columns: date (observation period), value, realtime_start, realtime_end.
    A row means: "from realtime_start to realtime_end, the published value for `date` was `value`".
    """
    fx = _fixtures()
    if fx is not None:
        df = pd.read_csv(fx / f"{series_id}_vintages.csv", dtype=str)
    else:
        import requests

        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise RuntimeError("Set FRED_API_KEY (free key from fred.stlouisfed.org) to download vintages.")
        params = {"series_id": series_id, "api_key": key, "file_type": "json",
                  "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
                  "observation_start": observation_start}
        resp = requests.get(FRED_API, params=params, timeout=60)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["observations"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "value": pd.to_numeric(df["value"], errors="coerce"),
        "realtime_start": _parse_rt(df["realtime_start"]),
        "realtime_end": _parse_rt(df["realtime_end"]),
    })
    return out.sort_values(["date", "realtime_start"]).reset_index(drop=True)


def first_release(vintages: pd.DataFrame) -> pd.DataFrame:
    """The first published value of each observation and the date it was published."""
    first = vintages.sort_values("realtime_start").groupby("date", as_index=True).first()
    return first[["value", "realtime_start"]].rename(columns={"value": "first_value",
                                                             "realtime_start": "published"})


def as_of(vintages: pd.DataFrame, when) -> pd.Series:
    """The series exactly as it could be seen on `when` (point-in-time)."""
    when = pd.Timestamp(when)
    v = vintages[(vintages["realtime_start"] <= when) & (vintages["realtime_end"] >= when)]
    return v.set_index("date")["value"].sort_index()


def revisions(vintages: pd.DataFrame, latest: pd.Series | None = None) -> pd.DataFrame:
    """First release vs latest value for each observation, and the total revision."""
    fr = first_release(vintages)
    if latest is None:
        latest = vintages.sort_values("realtime_start").groupby("date")["value"].last()
    out = fr.join(latest.rename("latest_value"), how="inner")
    out["revision"] = out["latest_value"] - out["first_value"]
    return out


# --------------------------------------------------------------------------------------
# Macro helpers
# --------------------------------------------------------------------------------------
def yoy_pct(s: pd.Series, periods: int = 12) -> pd.Series:
    """Year-over-year % change (12 periods for monthly data)."""
    return s.pct_change(periods, fill_method=None) * 100


def regime(growth: pd.Series, inflation: pd.Series, lookback: int = 3) -> pd.DataFrame:
    """Four-quadrant regime from the DIRECTION of growth and inflation over `lookback` periods.

    growth, inflation: e.g. YoY % of industrial production and CPI (monthly).
    """
    df = pd.concat({"growth": growth, "inflation": inflation}, axis=1).dropna()
    df["growth_up"] = df["growth"].diff(lookback) > 0
    df["inflation_up"] = df["inflation"].diff(lookback) > 0
    names = {(True, False): "Goldilocks (growth up, inflation down)",
             (True, True): "Reflation (growth up, inflation up)",
             (False, True): "Stagflation (growth down, inflation up)",
             (False, False): "Disinflationary slowdown (growth down, inflation down)"}
    df["regime"] = [names[(g, i)] for g, i in zip(df["growth_up"], df["inflation_up"])]
    return df.iloc[lookback:]


def bond_return_from_yield(yield_pct: pd.Series, duration: float = 8.5) -> pd.Series:
    """Approximate daily return of a 10-year Treasury from yield changes: r ≈ -duration × Δy."""
    return -duration * yield_pct.diff() / 100


def fed_funds_post_meeting_rate(futures_price: float, current_rate: float,
                                meeting_day: int, days_in_month: int) -> float:
    """Fed funds futures settle on 100 - the MONTHLY AVERAGE effective rate.
    Solve for the rate after an FOMC meeting on `meeting_day` (new rate from the next day)."""
    avg = 100 - futures_price
    return (avg * days_in_month - current_rate * meeting_day) / (days_in_month - meeting_day)


def move_probability(post_rate: float, current_rate: float, step: float = 0.25) -> float:
    """Implied probability of a `step` move (positive = hike, negative = cut)."""
    return (post_rate - current_rate) / step


# --------------------------------------------------------------------------------------
# Futures
# --------------------------------------------------------------------------------------
def futures_fair_value(spot: float, rate: float, div_yield: float, days: int) -> float:
    """Cost of carry for an equity index future: F = S·exp((r - q)·T), T in years (ACT/365)."""
    return spot * np.exp((rate - div_yield) * days / 365)


def margin_ledger(settle_prices: list[float], entry_price: float, contracts: int, multiplier: float,
                  initial_margin: float, maintenance_margin: float) -> pd.DataFrame:
    """Daily variation-margin ledger for a futures position (contracts > 0 long, < 0 short).

    If the balance falls below maintenance margin, a margin call brings it back to INITIAL margin.
    """
    rows, balance, prev = [], initial_margin * abs(contracts), entry_price
    for day, px in enumerate(settle_prices, start=1):
        pnl = (px - prev) * contracts * multiplier
        balance += pnl
        call = 0.0
        if balance < maintenance_margin * abs(contracts):
            call = initial_margin * abs(contracts) - balance
            balance += call
        rows.append({"day": day, "settle": px, "daily_pnl": pnl, "margin_call": call, "balance": balance})
        prev = px
    return pd.DataFrame(rows).set_index("day")


# --------------------------------------------------------------------------------------
# Options: payoffs, Black–Scholes–Merton, parity, implied volatility
# --------------------------------------------------------------------------------------
def payoff_at_expiry(legs: list[dict], prices: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
    """P&L at expiry of a list of legs over a grid of underlying prices.

    Each leg: {"type": "call" | "put" | "stock", "qty": +long/-short, "strike": K (options),
               "premium": price paid per unit (options) or entry price (stock)}.
    """
    prices = np.asarray(prices, dtype=float)
    total = np.zeros_like(prices)
    for leg in legs:
        q = leg["qty"]
        if leg["type"] == "call":
            total += q * (np.maximum(prices - leg["strike"], 0) - leg["premium"])
        elif leg["type"] == "put":
            total += q * (np.maximum(leg["strike"] - prices, 0) - leg["premium"])
        elif leg["type"] == "stock":
            total += q * (prices - leg["premium"])
        else:
            raise ValueError(f"unknown leg type {leg['type']}")
    return total * multiplier


def breakevens(prices: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Prices where the P&L crosses zero (linear interpolation on the grid)."""
    out = []
    for i in range(1, len(prices)):
        a, b = pnl[i - 1], pnl[i]
        if a == 0:
            out.append(float(prices[i - 1]))
        elif a * b < 0:
            out.append(float(prices[i - 1] + (prices[i] - prices[i - 1]) * (-a) / (b - a)))
    return out


def bsm(spot: float, strike: float, years: float, rate: float, div: float, vol: float) -> dict:
    """Black–Scholes–Merton prices and Greeks (European options, continuous dividend yield).

    Same formulas and units as the Excel sheet in S15: vega per 1 vol point, theta per calendar day.
    """
    sq = np.sqrt(years)
    d1 = (np.log(spot / strike) + (rate - div + vol**2 / 2) * years) / (vol * sq)
    d2 = d1 - vol * sq
    dq, dr = np.exp(-div * years), np.exp(-rate * years)
    call = spot * dq * norm.cdf(d1) - strike * dr * norm.cdf(d2)
    put = strike * dr * norm.cdf(-d2) - spot * dq * norm.cdf(-d1)
    return {
        "d1": d1, "d2": d2, "call": call, "put": put,
        "delta_call": dq * norm.cdf(d1), "delta_put": dq * (norm.cdf(d1) - 1),
        "gamma": dq * norm.pdf(d1) / (spot * vol * sq),
        "vega_per_point": spot * dq * norm.pdf(d1) * sq / 100,
        "theta_call_per_day": (-spot * dq * norm.pdf(d1) * vol / (2 * sq)
                               - rate * strike * dr * norm.cdf(d2) + div * spot * dq * norm.cdf(d1)) / 365,
        "parity_error": call - put - (spot * dq - strike * dr),
    }


def implied_vol(price: float, spot: float, strike: float, years: float, rate: float, div: float,
                kind: str = "call") -> float:
    """Volatility that makes the BSM price equal `price` (Brent's method, like Excel's Goal Seek)."""
    key = "call" if kind == "call" else "put"
    return brentq(lambda v: bsm(spot, strike, years, rate, div, v)[key] - price, 1e-4, 5.0)


def occ_symbol(root: str, expiry: date, right: str, strike: float) -> str:
    """OCC option symbol, e.g. occ_symbol('SPY', date(2026, 12, 18), 'C', 600) -> 'SPY261218C00600000'."""
    return f"{root}{expiry:%y%m%d}{right}{int(round(strike * 1000)):08d}"


# --------------------------------------------------------------------------------------
# Microstructure: order book, spreads
# --------------------------------------------------------------------------------------
def synthetic_book(mid: float = 100.0, tick: float = 0.01, levels: int = 10, top_size: int = 800,
                   growth: float = 1.35, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A made-up limit order book (bids, asks) for practice: 1-tick spread, depth growing away from the top."""
    rng = np.random.default_rng(seed)
    sizes = (top_size * growth ** np.arange(levels) * rng.uniform(0.7, 1.3, levels)).round(-2).astype(int)
    half = tick / 2
    asks = pd.DataFrame({"price": np.round(mid + half + tick * np.arange(levels), 4), "size": sizes})
    bids = pd.DataFrame({"price": np.round(mid - half - tick * np.arange(levels), 4), "size": sizes})
    return bids, asks


def walk_the_book(asks: pd.DataFrame, qty: int, mid: float) -> dict:
    """Average fill price and cost (vs mid, in bps) of a market BUY order of `qty` shares."""
    remaining, cost = qty, 0.0
    for px, sz in zip(asks["price"], asks["size"]):
        take = min(remaining, sz)
        cost += take * px
        remaining -= take
        if remaining == 0:
            break
    if remaining > 0:
        raise ValueError("order larger than the visible book")
    avg = cost / qty
    return {"qty": qty, "avg_price": avg, "cost_bps": (avg - mid) / mid * 1e4}


def simulate_trades_quotes(n_minutes: int = 390, base_spread: float = 0.02, open_close_mult: float = 3.0,
                           daily_vol: float = 0.012, start: str = "2025-03-03 14:30", seed: int = 0):
    """SYNTHETIC one-day quotes (1/min) and trades for practice. Spreads are wider at the open and
    close (a U-shape, as in real data). Replace with the course data files for the graded lab."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    x = np.linspace(-1, 1, n_minutes)
    spread = base_spread * (1 + (open_close_mult - 1) * x**2)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, daily_vol / np.sqrt(n_minutes), n_minutes)))
    quotes = pd.DataFrame({"ts": ts, "bid": mid - spread / 2, "ask": mid + spread / 2})
    t_idx = np.sort(rng.choice(n_minutes, size=n_minutes * 3, replace=True))
    side = rng.choice([1, -1], size=len(t_idx))
    trade_ts = ts[t_idx] + pd.to_timedelta(rng.uniform(1, 59, len(t_idx)), unit="s")
    price = np.where(side > 0, quotes["ask"].to_numpy()[t_idx], quotes["bid"].to_numpy()[t_idx])
    trades = pd.DataFrame({"ts": trade_ts, "price": price, "size": rng.integers(1, 10, len(t_idx)) * 100,
                           "side": side})
    return quotes, trades


def spread_measures(trades: pd.DataFrame, quotes: pd.DataFrame, realized_after: str = "5min") -> pd.DataFrame:
    """Quoted, effective and realized spreads (bps) for each trade, using the last quote BEFORE the trade.

    effective = 2 × side × (price − mid) / mid;  realized = 2 × side × (price − mid_later) / mid.
    `side` = +1 buyer-initiated, −1 seller-initiated (if missing, use the quote rule: price vs mid).
    """
    q = quotes.sort_values("ts").assign(mid=lambda d: (d["bid"] + d["ask"]) / 2)
    q["ts"] = q["ts"].dt.as_unit("ns")                      # same time resolution on both sides
    t = trades.sort_values("ts").copy()
    t["ts"] = t["ts"].dt.as_unit("ns")
    t = pd.merge_asof(t, q, on="ts", direction="backward")
    if "side" not in t:
        t["side"] = np.sign(t["price"] - t["mid"]).replace(0, np.nan)
    later = q[["ts", "mid"]].rename(columns={"mid": "mid_later", "ts": "ts_later"})
    t["ts_later"] = t["ts"] + pd.Timedelta(realized_after)
    t = pd.merge_asof(t.sort_values("ts_later"), later, left_on="ts_later", right_on="ts_later",
                      direction="backward").sort_values("ts")
    t["quoted_bps"] = (t["ask"] - t["bid"]) / t["mid"] * 1e4
    t["effective_bps"] = 2 * t["side"] * (t["price"] - t["mid"]) / t["mid"] * 1e4
    t["realized_bps"] = 2 * t["side"] * (t["price"] - t["mid_later"]) / t["mid"] * 1e4
    t["impact_bps"] = t["effective_bps"] - t["realized_bps"]
    return t


def roll_spread(prices: np.ndarray) -> float:
    """Roll (1984) estimator: spread ≈ 2·sqrt(−cov(Δp_t, Δp_{t−1})). NaN if the covariance is positive."""
    dp = np.diff(np.asarray(prices, dtype=float))
    c = np.cov(dp[1:], dp[:-1])[0, 1]
    return float(2 * np.sqrt(-c)) if c < 0 else float("nan")


# --------------------------------------------------------------------------------------
# Research log
# --------------------------------------------------------------------------------------
def log_research(entry: dict, path: str | Path = "research_log.csv") -> pd.DataFrame:
    """Append one entry (source, series, notes, ...) with a timestamp to the research log CSV."""
    path = Path(path)
    row = {"logged_at": datetime.now().isoformat(timespec="seconds"), **entry}
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    return df
