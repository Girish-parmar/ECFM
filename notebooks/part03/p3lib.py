"""Helper functions for the Part 3 guided notebooks (MFAAT, Months 2–3: Python engineering for trading).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

All data here is SYNTHETIC (generated with a fixed seed), so every notebook runs offline.
The graded, test-driven versions of these exercises are in labs/part03/.
"""
from __future__ import annotations

import os
import time as _time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P3_STRICT") == "1"      # tests: a failed check raises instead of continuing
NY, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")


# --------------------------------------------------------------------------------------
# Plot style and the exercise checker
# --------------------------------------------------------------------------------------
def use_course_style() -> None:
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=PALETTE), "figure.figsize": (9, 4.5),
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#52514e", "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "legend.frameon": False,
    })


def _numeric(x) -> bool:
    return isinstance(x, (int, float, np.number, np.ndarray)) and not isinstance(x, bool)


def _same(got, expected, rtol: float, atol: float) -> bool:
    if isinstance(expected, Decimal):
        return isinstance(got, Decimal) and got == expected            # exact, and a float is not money
    if isinstance(expected, (pd.DataFrame, pd.Series)):
        test = pd.testing.assert_frame_equal if isinstance(expected, pd.DataFrame) else pd.testing.assert_series_equal
        try:
            test(got, expected, check_exact=False, rtol=rtol, atol=atol, check_dtype=False, check_names=False,
                 check_freq=False)
            return True
        except (AssertionError, TypeError):
            return False
    if isinstance(expected, (list, tuple)) and expected and all(_numeric(v) for v in expected):
        expected = np.asarray(expected, dtype=float)
    if _numeric(expected):
        g, e = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
        return g.shape == e.shape and np.allclose(g, e, rtol=rtol, atol=atol, equal_nan=True)
    if isinstance(expected, (list, tuple)):
        return list(got) == list(expected)
    return got == expected


def check(name: str, got, expected, rtol: float = 1e-6, atol: float = 1e-9):
    """Compare your answer with the reference: exact for Decimals, text, dates and objects; with a tolerance for
    floats and arrays. Returns your value if correct, else the reference (so the notebook keeps running)."""
    try:
        if got is Ellipsis or (isinstance(got, tuple) and Ellipsis in got):
            raise ValueError("not done yet")
        ok = bool(_same(got, expected, rtol, atol))
    except Exception:  # noqa: BLE001 - any failure means "not correct yet"
        ok = False
    if ok:
        print(f"✅ {name}: correct")
        return got
    msg = f"❌ {name}: not matching the reference yet"
    if STRICT:
        raise AssertionError(msg)
    print(msg + " — continuing with the reference answer so the rest of the notebook runs.")
    return expected


def timeit(fn, *args, repeat: int = 5, number: int = 1) -> float:
    """Best-of-`repeat` wall time in seconds of fn(*args), run `number` times per repeat."""
    best = float("inf")
    for _ in range(repeat):
        t0 = _time.perf_counter()
        for _ in range(number):
            fn(*args)
        best = min(best, (_time.perf_counter() - t0) / number)
    return best


# --------------------------------------------------------------------------------------
# 01 · money and time
# --------------------------------------------------------------------------------------
def round_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    """Exact decimal rounding to the instrument's tick size (half up)."""
    return (price / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick


def ib_fixed_commission(shares: int, price: Decimal) -> Decimal:
    """IB fixed pricing (lesson-plan figures): $0.005/share, min $1.00, max 1% of trade value; to the cent."""
    raw = Decimal("0.005") * shares
    cap = Decimal("0.01") * shares * price
    return min(max(raw, Decimal("1.00")), cap).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def market_open_utc(d: date) -> datetime:
    """09:30 New York time on date d, in UTC."""
    return datetime.combine(d, time(9, 30), tzinfo=NY).astimezone(UTC)


def rolling_mean_deque(values, n: int) -> list:
    """Mean of the last n values after each new value (None until n values are in)."""
    win, out = deque(maxlen=n), []
    for v in values:
        win.append(v)
        out.append(sum(win) / n if len(win) == n else None)
    return out


# --------------------------------------------------------------------------------------
# 02 · functions, collections, files
# --------------------------------------------------------------------------------------
def simple_returns(prices: list[float]) -> list[float]:
    return [b / a - 1 for a, b in zip(prices, prices[1:])]


def max_drawdown(prices: list[float]) -> float:
    peak, mdd = prices[0], 0.0
    for p_ in prices:
        peak = max(peak, p_)
        mdd = min(mdd, p_ / peak - 1)
    return mdd


def route(msg: dict) -> str:
    """Route a broker message by its shape (match statement in the notebook)."""
    match msg:
        case {"type": "fill", "qty": q} if q > 0:
            return "book_fill"
        case {"type": "fill"}:
            return "reject_bad_fill"
        case {"type": "cancel" | "reject"}:
            return "release_order"
        case {"type": "heartbeat"}:
            return "ignore"
        case _:
            return "log_unknown"


TRADE_CSV = """time,symbol,side,qty,price
2025-03-07 09:31:02,SPY,BUY,100,512.10
2025-03-07 09:45:10,SPY,SELL,100,513.35
2025-03-07 10:02:55,QQQ,BUY,50,440.00
2025-03-07 10:05:00,QQQ,SELL,fifty,441.10
2025-03-07 10:30:12,AAPL,BUY,10,
2025-03-07 11:00:00,QQQ,SELL,50,441.60
2025-03-07 11:15:40,MSFT,HOLD,10,400.00
2025-03-10 09:35:00,SPY,BUY,200,509.80
"""


def parse_trades(text: str) -> tuple[list[dict], list[tuple[int, str]]]:
    """Parse the CSV text: valid rows as dicts (qty int, price Decimal, time datetime); errors as
    (line number counting the header as 1, message)."""
    import csv
    import io
    from decimal import InvalidOperation

    rows, errors = [], []
    for i, r in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        try:
            if r["side"] not in ("BUY", "SELL"):
                raise ValueError(f"bad side {r['side']!r}")
            qty = int(r["qty"])
            if not r["price"]:
                raise ValueError("missing price")
            price = Decimal(r["price"])
            rows.append({"time": datetime.fromisoformat(r["time"]), "symbol": r["symbol"], "side": r["side"],
                         "qty": qty, "price": price})
        except (ValueError, InvalidOperation) as e:
            errors.append((i, str(e)))
    return rows, errors


def realized_pnl(rows: list[dict]) -> dict[str, Decimal]:
    """Average-cost realized P&L per symbol for long-only round trips (enough for the demo data)."""
    pos, cost, pnl = {}, {}, {}
    for r in rows:
        s, q, px = r["symbol"], r["qty"], r["price"]
        if r["side"] == "BUY":
            cost[s] = cost.get(s, Decimal(0)) + q * px
            pos[s] = pos.get(s, 0) + q
        else:
            avg = cost[s] / pos[s]
            pnl[s] = pnl.get(s, Decimal(0)) + q * (px - avg)
            cost[s] -= q * avg
            pos[s] -= q
    return pnl


# --------------------------------------------------------------------------------------
# 03 · generators
# --------------------------------------------------------------------------------------
def tick_stream(n: int = 5_000, seed: int = 0, start: str = "2025-03-07 09:30"):
    """A GENERATOR of (timestamp, price, size) ticks, one every 0.5–3 s."""
    rng = np.random.default_rng(seed)
    t, px = pd.Timestamp(start), 100.0
    for _ in range(n):
        t += pd.Timedelta(seconds=float(rng.uniform(0.5, 3.0)))
        px = round(px + float(rng.normal(0, 0.02)), 2)
        yield t, px, int(rng.integers(1, 10)) * 100


def bars_from_ticks(ticks, minutes: int = 1):
    """Group a tick stream into OHLCV bars, yielding each bar when the next one starts."""
    cur = None
    for t, px, sz in ticks:
        start = t.floor(f"{minutes}min")
        if cur is not None and start != cur["start"]:
            yield cur
            cur = None
        if cur is None:
            cur = {"start": start, "open": px, "high": px, "low": px, "close": px, "volume": 0}
        cur["high"], cur["low"] = max(cur["high"], px), min(cur["low"], px)
        cur["close"] = px
        cur["volume"] += sz
    if cur is not None:
        yield cur


# --------------------------------------------------------------------------------------
# 05 · performance
# --------------------------------------------------------------------------------------
def prices_array(n: int = 1_000_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def drawdown_vectorized(x: np.ndarray) -> np.ndarray:
    return x / np.maximum.accumulate(x) - 1


def ema_loop(x, alpha: float) -> np.ndarray:
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


# --------------------------------------------------------------------------------------
# 06 · OOP
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Money:
    """Reference Money: Decimal amount + currency; adding different currencies is an error."""
    amount: Decimal
    currency: str = "USD"

    def __add__(self, other: "Money") -> "Money":
        if other.currency != self.currency:
            raise ValueError(f"cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)


ORDER_TRANSITIONS = {"NEW": {"SUBMITTED", "REJECTED"}, "SUBMITTED": {"PARTIAL", "FILLED", "CANCELLED", "REJECTED"},
                     "PARTIAL": {"PARTIAL", "FILLED", "CANCELLED"}, "FILLED": set(), "CANCELLED": set(),
                     "REJECTED": set()}


def replay_order(events: list[str]) -> str:
    state = "NEW"
    for e in events:
        if e not in ORDER_TRANSITIONS[state]:
            raise ValueError(f"{state} -> {e} is not allowed")
        state = e
    return state


# --------------------------------------------------------------------------------------
# 07 · pandas & polars
# --------------------------------------------------------------------------------------
def trades_and_quotes(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One hour of synthetic quotes (bid/ask updates) and trades, with UTC timestamps."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-03-07 14:30", tz="UTC")
    qt = start + pd.to_timedelta(np.sort(rng.uniform(0, 3600, 4000)), unit="s")
    mid = 100 + np.cumsum(rng.normal(0, 0.005, 4000))
    half = rng.choice([0.01, 0.02], 4000)
    quotes = pd.DataFrame({"time": qt, "bid": (mid - half).round(2), "ask": (mid + half).round(2)})
    tt = start + pd.to_timedelta(np.sort(rng.uniform(0, 3600, 600)), unit="s")
    idx = np.searchsorted(qt, tt, side="right") - 1
    side = rng.choice([1, -1], 600)
    px = np.where(side > 0, quotes["ask"].to_numpy()[idx.clip(0)], quotes["bid"].to_numpy()[idx.clip(0)])
    trades = pd.DataFrame({"time": tt, "price": px, "size": rng.integers(1, 20, 600) * 100})
    return trades, quotes


def with_quotes_asof(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Each trade with the LAST quote at or before it (never a later one)."""
    return pd.merge_asof(trades.sort_values("time"), quotes.sort_values("time"), on="time", direction="backward")


def ohlcv(trades: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    g = trades.set_index("time").resample(rule)
    out = g["price"].ohlc()
    out["volume"] = g["size"].sum()
    return out.dropna()


# --------------------------------------------------------------------------------------
# 08 · data validation
# --------------------------------------------------------------------------------------
def bad_bars(df: pd.DataFrame) -> pd.Series:
    """Rows violating OHLC logic: high < max(open, close), low > min(open, close), non-positive prices or volume."""
    return ((df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
            | (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["volume"] < 0))
