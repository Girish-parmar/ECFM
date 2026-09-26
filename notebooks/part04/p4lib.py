"""Helper functions for the Part 4 guided notebooks (MFAAT, Month 4: broker connectivity & trading infrastructure).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

Nothing here talks to a broker. Account rows, bars, ticks, error codes and order events are SYNTHETIC, shaped
like what `ib_async` and `alpaca-py` return, and generated with fixed seeds, so every notebook runs offline.
The graded, test-driven versions (with the real SDK objects) are in labs/part04/.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P4_STRICT") == "1"      # tests: a failed check raises instead of continuing
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
        except (AssertionError, TypeError, AttributeError):
            return False
    if isinstance(expected, dict):
        return (isinstance(got, dict) and set(got) == set(expected)
                and all(_same(got[k], v, rtol, atol) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)) and expected and all(_numeric(v) for v in expected):
        expected = np.asarray(expected, dtype=float)
    if _numeric(expected):
        g, e = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
        return g.shape == e.shape and np.allclose(g, e, rtol=rtol, atol=atol, equal_nan=True)
    if isinstance(expected, (list, tuple)):
        got = list(got)
        return len(got) == len(expected) and all(_same(a, b, rtol, atol) for a, b in zip(got, expected))
    return type(got) is type(expected) and got == expected if isinstance(expected, bool) else got == expected


def check(name: str, got, expected, rtol: float = 1e-6, atol: float = 1e-9):
    """Compare your answer with the reference: exact for Decimals, text, dates and objects (recursively inside
    lists and dicts); with a tolerance for floats and arrays. Returns your value if correct, else the reference
    (so the notebook keeps running)."""
    try:
        if got is Ellipsis or (isinstance(got, (tuple, list)) and Ellipsis in got):
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


def attempt(fn, *args, **kwargs):
    """Run fn; if it raises (for example because a blank `...` is still in it), return Ellipsis instead, so
    p.check reports "not done yet" and the notebook keeps going."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - an unfinished exercise may fail in any way
        return Ellipsis


# --------------------------------------------------------------------------------------
# 01 · environment, config, secrets
# --------------------------------------------------------------------------------------
IB_PORTS = {("gateway", "paper"): 4002, ("gateway", "live"): 4001, ("tws", "paper"): 7497, ("tws", "live"): 7496}


def ib_port(app: str, mode: str) -> int:
    """IB API port for app 'gateway' | 'tws' and mode 'paper' | 'live'."""
    return IB_PORTS[(app.lower(), mode.lower())]


def port_problem(env: str, port: int) -> str | None:
    """None if the port fits the environment, else a message. A paper (or backtest) environment must never
    reach a live port; a live environment on a paper port is a configuration mistake too."""
    live, paper = {4001, 7496}, {4002, 7497}
    if port not in live | paper:
        return "unknown port"
    if env in ("paper", "backtest") and port in live:
        return "live port in a paper environment"
    if env == "live" and port in paper:
        return "paper port in a live environment"
    return None


ALPACA_KEY_ID = re.compile(r"\b(?:PK|AK)[A-Z0-9]{18}\b")
ASSIGNMENT = re.compile(r"""(?i)\b\w*(?:secret|password|passwd|api_?key|token)\w*\s*[:=]\s*["']?([^\s"'#]{8,})""")


def find_secrets(text: str) -> list[tuple[int, str]]:
    """(line number from 1, kind) for each leaking line: 'alpaca_key_id' if ALPACA_KEY_ID matches, else
    'assignment' if ASSIGNMENT matches and the value is not a placeholder (starts with '<', '$' or '{', or is
    'changeme')."""
    hits = []
    for n, line in enumerate(text.splitlines(), start=1):
        if ALPACA_KEY_ID.search(line):
            hits.append((n, "alpaca_key_id"))
            continue
        m = ASSIGNMENT.search(line)
        if m and not (m.group(1)[0] in "<${" or m.group(1).lower() == "changeme"):
            hits.append((n, "assignment"))
    return hits


# built by concatenation so the repository itself holds no key-shaped literal on one line
_FAKE_KEY = "PK" + "TESTONLY" + "0000000000"
SAMPLE_FILES = {
    ".env.example": "QF_ENV=paper\nQF_IB_PORT=4002\nQF_ALPACA_KEY=<your-paper-key>\nQF_ALPACA_SECRET=changeme\n",
    "notebooks/scratch.py": ("import alpaca\n"
                             f"client = TradingClient('{_FAKE_KEY}', SECRET)\n"
                             "api_key = 'abc'\n"
                             "timeout = 30\n"),
    "config/settings.yaml": "broker: alpaca\nalpaca_secret: 9f8e7d6c5b4a39281706f5e4d3c2b1a0\nretries: 3\n",
    "README.md": "Set `QF_ALPACA_SECRET=${ALPACA_SECRET}` in your shell.\nNever paste token: values into notebooks.\n",
}


def settings_from_env(env: dict):
    """A pydantic settings object from QF_-prefixed variables (what pydantic-settings does for you with .env)."""
    from pydantic import BaseModel, SecretStr

    class Settings(BaseModel):
        env: str = "paper"
        ib_host: str = "127.0.0.1"
        ib_port: int = 4002
        ib_client_id: int = 1
        alpaca_key: SecretStr
        alpaca_secret: SecretStr
        alpaca_paper: bool = True

    return Settings(**{k[3:].lower(): v for k, v in env.items() if k.startswith("QF_")})


# --------------------------------------------------------------------------------------
# 02 · contracts and accounts
# --------------------------------------------------------------------------------------
MONTH_CODES = "FGHJKMNQUVXZ"                      # Jan..Dec


@dataclass(frozen=True)
class Instrument:
    """symbol: 'AAPL' (equity), 'ES' (future root), 'EURUSD' (FX pair), 'BTC' (crypto base).
    expiry: 'YYYYMM' for futures. exchange: IB listing exchange ('NASDAQ', 'CME', ...)."""
    symbol: str
    asset_class: str                               # EQUITY | FUTURE | FX | CRYPTO
    currency: str = "USD"
    exchange: str = ""
    expiry: str = ""


INSTRUMENTS = [Instrument("AAPL", "EQUITY", exchange="NASDAQ"), Instrument("ES", "FUTURE", exchange="CME", expiry="202612"),
               Instrument("CL", "FUTURE", exchange="NYMEX", expiry="202603"), Instrument("EURUSD", "FX", currency="USD"),
               Instrument("USDJPY", "FX", currency="JPY"), Instrument("BTC", "CRYPTO")]


def canonical_symbol(inst: Instrument) -> str:
    """'AAPL'; future root + month code + last digit of the year ('ESZ6'); FX 'EUR.USD'; crypto 'BTC/USD'."""
    match inst.asset_class:
        case "EQUITY":
            return inst.symbol
        case "FUTURE":
            return inst.symbol + MONTH_CODES[int(inst.expiry[4:6]) - 1] + inst.expiry[3]
        case "FX":
            return f"{inst.symbol[:3]}.{inst.symbol[3:]}"
        case "CRYPTO":
            return f"{inst.symbol}/{inst.currency}"
    raise ValueError(inst.asset_class)


def ib_contract_fields(inst: Instrument) -> dict:
    """The fields of the ib_async Contract for this instrument (Stock / Future / Forex / Crypto)."""
    match inst.asset_class:
        case "EQUITY":
            return {"secType": "STK", "symbol": inst.symbol, "exchange": "SMART", "primaryExchange": inst.exchange,
                    "currency": inst.currency}
        case "FUTURE":
            return {"secType": "FUT", "symbol": inst.symbol, "lastTradeDateOrContractMonth": inst.expiry,
                    "exchange": inst.exchange, "currency": inst.currency}
        case "FX":
            return {"secType": "CASH", "symbol": inst.symbol[:3], "exchange": "IDEALPRO", "currency": inst.symbol[3:]}
        case "CRYPTO":
            return {"secType": "CRYPTO", "symbol": inst.symbol, "exchange": "PAXOS", "currency": inst.currency}
    raise ValueError(inst.asset_class)


SUMMARY_TAGS = ("NetLiquidation", "TotalCashValue", "BuyingPower", "AvailableFunds", "MaintMarginReq")


def ib_account_rows() -> list[dict]:
    """Rows shaped like ib_async AccountValue(account, tag, value, currency, modelCode): strings, several
    currencies, a BASE line per tag and tags we don't need."""
    rows = []
    usd = {"NetLiquidation": "1003127.44", "TotalCashValue": "912450.10", "BuyingPower": "4012509.76",
           "AvailableFunds": "1003127.44", "MaintMarginReq": "0.00", "GrossPositionValue": "90677.34",
           "Cushion": "1", "AccountType": "INDIVIDUAL"}
    for tag, v in usd.items():
        rows.append({"account": "DU1234567", "tag": tag, "value": v, "currency": "USD" if tag != "AccountType" else ""})
        if tag in SUMMARY_TAGS:
            rows.append({"account": "DU1234567", "tag": tag, "value": v, "currency": "BASE"})
    rows += [{"account": "DU1234567", "tag": "TotalCashValue", "value": "1520.00", "currency": "EUR"},
             {"account": "DU1234567", "tag": "NetLiquidation", "value": "1520.00", "currency": "EUR"}]
    return rows


def ib_account_summary(rows: list[dict], currency: str = "USD") -> dict[str, Decimal]:
    """{tag: Decimal(value)} for the SUMMARY_TAGS rows in `currency`."""
    return {r["tag"]: Decimal(r["value"]) for r in rows if r["tag"] in SUMMARY_TAGS and r["currency"] == currency}


ALPACA_ACCOUNTS = {
    "healthy": {"status": "ACTIVE", "equity": "100412.55", "cash": "62110.00", "buying_power": "200825.10",
                "pattern_day_trader": False, "trading_blocked": False, "account_blocked": False},
    "small_pdt": {"status": "ACTIVE", "equity": "18950.00", "cash": "18950.00", "buying_power": "0",
                  "pattern_day_trader": True, "trading_blocked": False, "account_blocked": False},
    "blocked": {"status": "ACCOUNT_UPDATED", "equity": "50000", "cash": "50000", "buying_power": "100000",
                "pattern_day_trader": False, "trading_blocked": True, "account_blocked": False},
}


def alpaca_account_warnings(acct: dict) -> list[str]:
    """Warnings, in this order: 'status <status>' if not ACTIVE, 'account blocked', 'trading blocked',
    'PDT with equity below $25,000', 'no buying power' (<= 0)."""
    out = []
    if acct["status"] != "ACTIVE":
        out.append(f"status {acct['status']}")
    if acct["account_blocked"]:
        out.append("account blocked")
    if acct["trading_blocked"]:
        out.append("trading blocked")
    if acct["pattern_day_trader"] and Decimal(acct["equity"]) < 25000:
        out.append("PDT with equity below $25,000")
    if Decimal(acct["buying_power"]) <= 0:
        out.append("no buying power")
    return out


# --------------------------------------------------------------------------------------
# 03 · historical data, pacing, the canonical schema
# --------------------------------------------------------------------------------------
CANON = ["ts", "symbol", "open", "high", "low", "close", "volume", "source"]


def ib_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    """Walk back from `end` in chunks of `chunk_days` (IB requests are 'endDateTime + durationStr');
    the last chunk is clipped at `start`. Newest chunk first."""
    out, cur = [], end
    while cur > start:
        s = max(start, cur - timedelta(days=chunk_days))
        out.append((s, cur))
        cur = s
    return out


def paced_times(desired: list[float], max_n: int, window: float) -> list[float]:
    """Actual send times (seconds) for requests wanted at `desired` (sorted) so that no `window`-second
    interval holds more than `max_n` sends. Requests keep their order."""
    sent: list[float] = []
    for t in desired:
        t = max(t, sent[-1]) if sent else t
        if len(sent) >= max_n and t < sent[-max_n] + window:
            t = sent[-max_n] + window
        sent.append(t)
    return sent


def paced_multi(desired: list[float], rules: list[tuple[int, float]]) -> list[float]:
    """paced_times for several (max_n, window) rules at once (IB: 6 per 2 s AND 60 per 10 min)."""
    sent: list[float] = []
    for t in desired:
        t = max(t, sent[-1]) if sent else t
        moved = True
        while moved:
            moved = False
            for n, w in rules:
                if len(sent) >= n and t < sent[-n] + w:
                    t, moved = sent[-n] + w, True
        sent.append(t)
    return sent


def _prices(n: int, seed: int, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))


def raw_bars(day: str = "2025-03-07", minutes: int = 390, seed: int = 4):
    """The same 5-minute session as IB and Alpaca would return it.
    IB (ib_async util.df of BarData): 'date' as naive New York local time, OHLC, volume, average, barCount.
    Alpaca (BarSet.df): MultiIndex (symbol, timestamp in UTC), OHLC, volume from the IEX feed (a few % of
    consolidated volume), trade_count, vwap."""
    n = minutes // 5
    close = _prices(n, seed, 512.0).round(2)
    rng = np.random.default_rng(seed + 1)
    open_ = np.r_[close[0] - 0.05, close[:-1]].round(2)
    high = (np.maximum(open_, close) + rng.uniform(0, 0.25, n)).round(2)
    low = (np.minimum(open_, close) - rng.uniform(0, 0.25, n)).round(2)
    u = np.linspace(-1, 1, n)
    vol = (rng.gamma(4.0, 1.0, n) * (1 + 2.5 * u ** 2) * 60_000).round()   # U-shaped intraday volume
    local = pd.date_range(f"{day} 09:30", periods=n, freq="5min")           # naive NY local
    ib = pd.DataFrame({"date": local, "open": open_, "high": high, "low": low, "close": close, "volume": vol,
                       "average": ((high + low + close) / 3).round(4), "barCount": (vol / 90).round().astype(int)})
    iex_share = rng.uniform(0.02, 0.035, n)
    ts = local.tz_localize(NY).tz_convert(UTC)
    alp = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": (vol * iex_share).round(),
                        "trade_count": (vol * iex_share / 40).round(), "vwap": ((high + low + close) / 3).round(4)},
                       index=pd.MultiIndex.from_arrays([["SPY"] * n, ts], names=["symbol", "timestamp"]))
    return ib, alp


def normalize_ib(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """IB bars → CANON: 'date' (New York local) localized and converted to UTC as 'ts'; source 'ib'."""
    out = df.rename(columns={"date": "ts"})
    out["ts"] = pd.to_datetime(out["ts"]).dt.tz_localize(NY).dt.tz_convert(UTC)
    out["symbol"], out["source"] = symbol, "ib"
    out["volume"] = out["volume"].astype(float)
    return out[CANON].sort_values("ts").reset_index(drop=True)


def normalize_alpaca(df: pd.DataFrame, feed: str = "iex") -> pd.DataFrame:
    """Alpaca bars → CANON: index levels to columns, 'timestamp' → 'ts' (already UTC); source 'alpaca_<feed>'."""
    out = df.reset_index().rename(columns={"timestamp": "ts"})
    out["source"] = f"alpaca_{feed}"
    out["volume"] = out["volume"].astype(float)
    return out[CANON].sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 04 · live streams, the bar builder, the read-through cache
# --------------------------------------------------------------------------------------
def tick_stream(seconds: int = 300, seed: int = 7) -> list[tuple[float, float, int]]:
    """(t seconds since the open, price, size): busy spells and quiet spells (no ticks for up to ~40 s)."""
    rng = np.random.default_rng(seed)
    t, out, px = 0.0, [], 100.0
    while t < seconds:
        busy = (t // 60) % 2 == 0
        t += rng.exponential(0.4 if busy else 9.0)
        if t >= seconds:
            break
        px = round(px + rng.normal(0, 0.02), 2)
        out.append((round(t, 3), px, int(rng.integers(1, 10)) * 100))
    return out


class BarBuilder:
    """Time bars from ticks. A bar covers [start, start + seconds). on_tick returns the bars it completes
    (a tick past the current bar's end closes it); on_timer(now) closes the current bar once `now` has passed its
    end, so a bar is published on time even when no tick follows it. Intervals without ticks give no bar."""

    def __init__(self, seconds: int = 5):
        self.seconds = seconds
        self.cur: dict | None = None

    def on_tick(self, ts: float, price: float, size: int) -> list[dict]:
        out = []
        if self.cur is not None and ts >= self.cur["start"] + self.seconds:
            out.append(self.cur)
            self.cur = None
        if self.cur is None:
            start = ts // self.seconds * self.seconds
            self.cur = {"start": start, "open": price, "high": price, "low": price, "close": price, "volume": size}
        else:
            self.cur.update(high=max(self.cur["high"], price), low=min(self.cur["low"], price), close=price)
            self.cur["volume"] += size
        return out

    def on_timer(self, now: float) -> list[dict]:
        if self.cur is not None and now >= self.cur["start"] + self.seconds:
            bar, self.cur = self.cur, None
            return [bar]
        return []


def run_builder(builder, ticks, timer_every: float | None = 1.0, end: float = 300.0):
    """Feed ticks (and, if timer_every, a timer event every timer_every seconds) in time order.
    Returns (bars, publish_delays): delay = publish time − bar end."""
    events = [(t, 0, (t, px, sz)) for t, px, sz in ticks]
    if timer_every:
        events += [(float(k), 1, None) for k in np.arange(timer_every, end + timer_every, timer_every)]
    bars, delays = [], []
    for t, kind, payload in sorted(events, key=lambda e: (e[0], e[1])):
        done = builder.on_tick(*payload) if kind == 0 else builder.on_timer(t)
        for b in done:
            bars.append(b)
            delays.append(t - (b["start"] + builder.seconds))
    return bars, delays


def missing_ranges(have, start, end):
    """Sub-ranges of [start, end) not covered by the (possibly overlapping, unsorted) ranges in `have`."""
    out, cur = [], start
    for s, e in sorted(have):
        if e <= cur:
            continue
        if s > cur:
            out.append((cur, min(s, end)))
        cur = max(cur, e)
        if cur >= end:
            break
    if cur < end:
        out.append((cur, end))
    return [(a, b) for a, b in out if a < b]


class BarCache:
    """Read-through cache over a `fetch(start, end)` function: only the missing ranges are downloaded."""

    def __init__(self, fetch):
        self.fetch, self.have, self.downloads = fetch, [], 0

    def get(self, start, end):
        for a, b in missing_ranges(self.have, start, end):
            self.fetch(a, b)
            self.downloads += 1
            self.have.append((a, b))
        return (start, end)


# --------------------------------------------------------------------------------------
# 05 · connections: back-off, circuit breaker, IB codes
# --------------------------------------------------------------------------------------
def backoff_delays(attempts: int, base: float, cap: float, rng: random.Random) -> list[float]:
    """'Full jitter': attempt i waits rng.uniform(0, min(cap, base * 2**i)), i = 0 .. attempts-1."""
    return [rng.uniform(0, min(cap, base * 2 ** i)) for i in range(attempts)]


class CircuitBreaker:
    """closed → (threshold consecutive failures) → open → (cooldown seconds) → half_open → one trial call:
    success closes it, failure opens it again."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0):
        self.threshold, self.cooldown = threshold, cooldown
        self.state, self.failures, self.opened_at = "closed", 0, 0.0

    def allow(self, now: float) -> bool:
        if self.state == "open":
            if now - self.opened_at >= self.cooldown:
                self.state = "half_open"
                return True
            return False
        return True

    def record(self, ok: bool, now: float) -> None:
        if ok:
            self.state, self.failures = "closed", 0
            return
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.threshold:
            self.state, self.opened_at = "open", now


def breaker_trace(breaker, outcomes: list[tuple[float, bool]]) -> list[tuple[float, str, str]]:
    """For each (time, would_succeed): (time, 'call' | 'blocked', state after)."""
    out = []
    for t, ok in outcomes:
        if breaker.allow(t):
            breaker.record(ok, t)
            out.append((t, "call", breaker.state))
        else:
            out.append((t, "blocked", breaker.state))
    return out


def herd(n_clients: int = 500, capacity: int = 50, jitter: bool = True, base: float = 1.0, cap: float = 32.0,
         seed: int = 0, horizon: int = 240):
    """A gateway restarts at t=0 and accepts at most `capacity` connections per second; every client retries with
    exponential back-off, with or without full jitter. Returns (attempts per second, second the last client got in
    or None)."""
    rng = random.Random(seed)
    nxt = [0.0] * n_clients                       # everyone notices the drop at once
    tries, done_at = [0] * n_clients, [None] * n_clients
    per_sec = np.zeros(horizon)
    for sec in range(horizon):
        idx = [i for i in range(n_clients) if done_at[i] is None and sec <= nxt[i] < sec + 1]
        per_sec[sec] = len(idx)
        rng.shuffle(idx)
        for i in idx[:capacity]:
            done_at[i] = sec
        for i in idx[capacity:]:
            tries[i] += 1
            d = min(cap, base * 2 ** tries[i])
            nxt[i] = sec + 1 + (rng.uniform(0, d) if jitter else d)
    last = None if any(d is None for d in done_at) else max(done_at)
    return per_sec, last


def classify_ib_code(code: int) -> str:
    match code:
        case 2104 | 2106 | 2107 | 2108 | 2158:
            return "info"
        case 1100 | 1101 | 1102 | 2110 | 504:
            return "connectivity"
        case 162 | 420:
            return "pacing"
        case 326:
            return "client_id_in_use"
        case 103 | 110 | 201:
            return "order_reject"
        case _:
            return "unknown"


def ib_error_log(n: int = 400, seed: int = 3) -> list[int]:
    rng = np.random.default_rng(seed)
    codes = [2104, 2106, 2158, 2107, 1100, 1102, 2110, 504, 162, 420, 326, 201, 110, 103, 399, 10167]
    w = np.array([30, 30, 20, 5, 3, 3, 2, 1, 6, 2, 1, 3, 2, 1, 2, 4], dtype=float)
    return [int(c) for c in rng.choice(codes, size=n, p=w / w.sum())]


# --------------------------------------------------------------------------------------
# 06 · order mapping
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Order:
    symbol: str
    side: str                                      # BUY | SELL
    qty: Decimal
    type: str                                      # MKT | LMT | STP | STP_LMT | TRAIL
    limit: Decimal | None = None
    stop: Decimal | None = None
    trail_pct: Decimal | None = None
    tif: str = "DAY"                               # DAY | GTC | IOC | FOK
    outside_rth: bool = False
    client_id: str = "qf-000"


SAMPLE_ORDERS = [
    Order("AAPL", "BUY", Decimal("100"), "MKT", client_id="qf-001"),
    Order("AAPL", "SELL", Decimal("50"), "LMT", limit=Decimal("231.55"), tif="GTC", client_id="qf-002"),
    Order("SPY", "SELL", Decimal("20"), "STP", stop=Decimal("505.00"), client_id="qf-003"),
    Order("QQQ", "BUY", Decimal("10"), "STP_LMT", limit=Decimal("441.20"), stop=Decimal("441.00"), client_id="qf-004"),
    Order("MSFT", "SELL", Decimal("15"), "TRAIL", trail_pct=Decimal("1.5"), client_id="qf-005"),
    Order("NVDA", "BUY", Decimal("5"), "LMT", limit=Decimal("118.40"), outside_rth=True, client_id="qf-006"),
]


def round_limit(price: Decimal, tick: Decimal, side: str) -> Decimal:
    """Round a limit price onto the tick grid in the trader's favour: a BUY rounds down, a SELL rounds up
    (never pay more, or sell for less, than asked)."""
    rounding = ROUND_FLOOR if side == "BUY" else ROUND_CEILING
    return (price / tick).quantize(Decimal("1"), rounding=rounding) * tick


IB_TYPES = {"MKT": "MKT", "LMT": "LMT", "STP": "STP", "STP_LMT": "STP LMT", "TRAIL": "TRAIL"}
ALPACA_TYPES = {"MKT": "market", "LMT": "limit", "STP": "stop", "STP_LMT": "stop_limit", "TRAIL": "trailing_stop"}


def to_ib_fields(o: Order) -> dict:
    """ib_async Order fields; None values are left out."""
    d = {"action": o.side, "totalQuantity": o.qty, "orderType": IB_TYPES[o.type], "lmtPrice": o.limit,
         "auxPrice": o.stop, "trailingPercent": o.trail_pct, "tif": o.tif, "outsideRth": o.outside_rth,
         "orderRef": o.client_id}
    return {k: v for k, v in d.items() if v is not None}


def to_alpaca_fields(o: Order) -> dict:
    """alpaca-py order request fields; None values are left out. Alpaca only accepts extended hours on
    DAY limit orders."""
    if o.outside_rth and not (o.type == "LMT" and o.tif == "DAY"):
        raise ValueError("Alpaca extended hours needs a DAY limit order")
    d = {"symbol": o.symbol, "qty": o.qty, "side": o.side.lower(), "type": ALPACA_TYPES[o.type],
         "time_in_force": o.tif.lower(), "limit_price": o.limit, "stop_price": o.stop, "trail_percent": o.trail_pct,
         "extended_hours": o.outside_rth, "client_order_id": o.client_id}
    return {k: v for k, v in d.items() if v is not None}


IB_STATUS = {"ApiPending": "PENDING_NEW", "PendingSubmit": "PENDING_NEW", "PreSubmitted": "ACCEPTED",
             "Submitted": "ACCEPTED", "PendingCancel": "PENDING_CANCEL", "ApiCancelled": "CANCELLED",
             "Cancelled": "CANCELLED", "Filled": "FILLED", "Inactive": "REJECTED"}
ALPACA_STATUS = {"pending_new": "PENDING_NEW", "new": "ACCEPTED", "accepted": "ACCEPTED",
                 "partially_filled": "PARTIALLY_FILLED", "filled": "FILLED", "pending_cancel": "PENDING_CANCEL",
                 "canceled": "CANCELLED", "expired": "EXPIRED", "rejected": "REJECTED", "done_for_day": "EXPIRED"}


def map_ib_status(status: str, filled: float) -> str:
    """IB has no 'partially filled' status: Submitted/PreSubmitted with filled > 0 means PARTIALLY_FILLED."""
    s = IB_STATUS[status]
    return "PARTIALLY_FILLED" if s == "ACCEPTED" and filled > 0 else s


# --------------------------------------------------------------------------------------
# 07 · order lifecycle, positions, reconciliation
# --------------------------------------------------------------------------------------
TRANSITIONS = {
    "PENDING_NEW": {"ACCEPTED", "REJECTED", "PARTIALLY_FILLED", "FILLED", "PENDING_CANCEL", "CANCELLED"},
    "ACCEPTED": {"PARTIALLY_FILLED", "FILLED", "PENDING_CANCEL", "CANCELLED", "EXPIRED"},
    "PARTIALLY_FILLED": {"PARTIALLY_FILLED", "FILLED", "PENDING_CANCEL", "CANCELLED", "EXPIRED"},
    "PENDING_CANCEL": {"CANCELLED", "FILLED", "PARTIALLY_FILLED"},
    "FILLED": set(), "CANCELLED": set(), "REJECTED": set(), "EXPIRED": set(),
}


def order_events() -> list[dict]:
    """A morning of broker updates, already mapped to canonical states, in ARRIVAL order: a duplicated fill,
    a fill before its order's ACCEPTED, a late ACCEPTED after FILLED, a cancel that races a fill."""
    D = Decimal
    return [
        {"order_id": "qf-101", "type": "status", "status": "ACCEPTED"},
        {"order_id": "qf-101", "type": "fill", "exec_id": "e1", "qty": D("40"), "price": D("512.10")},
        {"order_id": "qf-101", "type": "status", "status": "PARTIALLY_FILLED"},
        {"order_id": "qf-101", "type": "fill", "exec_id": "e1", "qty": D("40"), "price": D("512.10")},   # duplicate
        {"order_id": "qf-101", "type": "fill", "exec_id": "e2", "qty": D("60"), "price": D("512.14")},
        {"order_id": "qf-101", "type": "status", "status": "FILLED"},
        {"order_id": "qf-102", "type": "fill", "exec_id": "e3", "qty": D("25"), "price": D("440.05")},   # fill first
        {"order_id": "qf-102", "type": "status", "status": "FILLED"},
        {"order_id": "qf-102", "type": "status", "status": "ACCEPTED"},                                  # late
        {"order_id": "qf-103", "type": "status", "status": "ACCEPTED"},
        {"order_id": "qf-103", "type": "status", "status": "PENDING_CANCEL"},
        {"order_id": "qf-103", "type": "fill", "exec_id": "e4", "qty": D("10"), "price": D("231.60")},   # race
        {"order_id": "qf-103", "type": "status", "status": "FILLED"},
        {"order_id": "qf-103", "type": "status", "status": "CANCELLED"},                                 # too late
        {"order_id": "qf-104", "type": "status", "status": "REJECTED"},
        {"order_id": "qf-104", "type": "status", "status": "ACCEPTED"},                                  # after reject
        {"order_id": "qf-105", "type": "status", "status": "ACCEPTED"},
        {"order_id": "qf-105", "type": "fill", "exec_id": "e5", "qty": D("30"), "price": D("118.40")},
        {"order_id": "qf-105", "type": "fill", "exec_id": "e5", "qty": D("30"), "price": D("118.40")},   # duplicate
        {"order_id": "qf-105", "type": "status", "status": "PARTIALLY_FILLED"},
    ]


def apply_events(events: list[dict]) -> dict[str, dict]:
    """{order_id: {"state", "filled", "avg_price"}}: every order starts PENDING_NEW; a status is applied only
    if TRANSITIONS allows it from the current state; a fill counts once per exec_id; avg_price is the
    fill-weighted price quantized to 0.0001 (None with no fills)."""
    book, seen = {}, set()
    for ev in events:
        o = book.setdefault(ev["order_id"], {"state": "PENDING_NEW", "filled": Decimal(0), "notional": Decimal(0)})
        if ev["type"] == "fill":
            if ev["exec_id"] in seen:
                continue
            seen.add(ev["exec_id"])
            o["filled"] += ev["qty"]
            o["notional"] += ev["qty"] * ev["price"]
        elif ev["status"] in TRANSITIONS[o["state"]]:
            o["state"] = ev["status"]
    return {k: {"state": o["state"], "filled": o["filled"],
                "avg_price": (o["notional"] / o["filled"]).quantize(Decimal("0.0001")) if o["filled"] else None}
            for k, o in book.items()}


def positions_from_fills(fills: list[dict]) -> dict[str, Decimal]:
    """Net quantity per symbol from executions (BUY +, SELL −); symbols that net to zero are left out."""
    pos: dict[str, Decimal] = {}
    for f in fills:
        pos[f["symbol"]] = pos.get(f["symbol"], Decimal(0)) + (f["qty"] if f["side"] == "BUY" else -f["qty"])
    return {s: q for s, q in pos.items() if q != 0}


def day_fills() -> list[dict]:
    D = Decimal
    return [{"symbol": "SPY", "side": "BUY", "qty": D("100")}, {"symbol": "QQQ", "side": "BUY", "qty": D("25")},
            {"symbol": "AAPL", "side": "SELL", "qty": D("10")}, {"symbol": "SPY", "side": "SELL", "qty": D("40")},
            {"symbol": "QQQ", "side": "SELL", "qty": D("25")}, {"symbol": "NVDA", "side": "BUY", "qty": D("30")},
            {"symbol": "SPY", "side": "BUY", "qty": D("15")}]


def reconcile(ours: dict[str, Decimal], broker: dict[str, Decimal], prefix: str = "qf-") -> dict[str, list[str]]:
    """Open orders by client order id → remaining qty, ours vs the broker's.
    missing_at_broker: ours, not at the broker. orphans: at the broker with OUR prefix but not in our book
    (e.g. sent before a crash). qty_mismatch: on both sides with different qty. foreign: at the broker without our
    prefix (someone else's, never touch). All sorted."""
    return {"missing_at_broker": sorted(set(ours) - set(broker)),
            "orphans": sorted(k for k in broker if k.startswith(prefix) and k not in ours),
            "qty_mismatch": sorted(k for k in set(ours) & set(broker) if ours[k] != broker[k]),
            "foreign": sorted(k for k in broker if not k.startswith(prefix))}


# --------------------------------------------------------------------------------------
# 08 · safety and multi-asset
# --------------------------------------------------------------------------------------
FUTURES = {"ES": (Decimal("0.25"), Decimal("50")), "MES": (Decimal("0.25"), Decimal("5")),
           "NQ": (Decimal("0.25"), Decimal("20")), "CL": (Decimal("0.01"), Decimal("1000")),
           "GC": (Decimal("0.10"), Decimal("100")), "ZN": (Decimal("0.015625"), Decimal("1000"))}


def futures_pnl(entry: Decimal, exit: Decimal, qty: Decimal, multiplier: Decimal) -> Decimal:
    """P&L of a futures trade in USD; qty is signed (+ long, − short)."""
    return (exit - entry) * qty * multiplier


def pip_value_usd(pair: str, units: Decimal, usd_per_quote: Decimal) -> Decimal:
    """One pip (0.01 for JPY-quoted pairs, else 0.0001) on `units` of the base currency, in USD."""
    pip = Decimal("0.01") if pair.endswith("JPY") else Decimal("0.0001")
    return pip * units * usd_per_quote


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def roll_date(year: int, month: int, business_days_before: int = 8) -> date:
    """Roll `business_days_before` weekdays before the third-Friday expiry (holidays ignored)."""
    d, n = third_friday(year, month), 0
    while n < business_days_before:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


@dataclass
class Context:
    killed: bool = False
    last: dict = field(default_factory=lambda: {"SPY": Decimal("512.00"), "AAPL": Decimal("231.50"),
                                                "QQQ": Decimal("440.00")})
    positions: dict = field(default_factory=lambda: {"SPY": Decimal("300"), "AAPL": Decimal("-50")})
    max_notional: Decimal = Decimal("150000")
    max_position: Decimal = Decimal("500")
    band: Decimal = Decimal("0.05")
    buying_power: Decimal = Decimal("80000")


def pretrade_check(order: dict, ctx: Context) -> str | None:
    """First failing check or None, in order: 'kill switch', 'no market data', 'price outside band'
    (|price/last − 1| > band), 'order too large' (qty·price > max_notional), 'position limit'
    (|position after| > max_position), 'buying power' (a BUY costing more than buying_power)."""
    if ctx.killed:
        return "kill switch"
    last = ctx.last.get(order["symbol"])
    if last is None:
        return "no market data"
    if abs(order["price"] / last - 1) > ctx.band:
        return "price outside band"
    notional = order["qty"] * order["price"]
    if notional > ctx.max_notional:
        return "order too large"
    signed = order["qty"] if order["side"] == "BUY" else -order["qty"]
    if abs(ctx.positions.get(order["symbol"], Decimal(0)) + signed) > ctx.max_position:
        return "position limit"
    if order["side"] == "BUY" and notional > ctx.buying_power:
        return "buying power"
    return None


def random_orders(n: int = 1000, seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    syms = ["SPY", "AAPL", "QQQ", "TSLA"]
    last = Context().last
    out = []
    for _ in range(n):
        s = syms[int(rng.choice(4, p=[0.4, 0.3, 0.25, 0.05]))]
        ref = last.get(s, Decimal("250"))
        px = (ref * Decimal(str(round(float(rng.normal(1, 0.03)), 4)))).quantize(Decimal("0.01"))
        out.append({"symbol": s, "side": "BUY" if rng.random() < 0.55 else "SELL",
                    "qty": Decimal(int(rng.choice([10, 50, 100, 200, 300, 400]))), "price": px})
    return out


def flatten_plan(open_orders: list[str], positions: dict[str, Decimal]) -> list[tuple]:
    """Kill-switch plan: first ('cancel', id) for every open order (sorted), then ('market', symbol, side, qty)
    closing every non-zero position (sorted by symbol): SELL a long, BUY a short, qty always positive."""
    plan: list[tuple] = [("cancel", oid) for oid in sorted(open_orders)]
    plan += [("market", s, "SELL" if q > 0 else "BUY", abs(q)) for s, q in sorted(positions.items()) if q != 0]
    return plan


class KillSwitch:
    """Sticky: tripping writes a state file, so a restarted process is still stopped until someone resets it
    with the confirmation phrase. Every trip and reset is appended to an audit list."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.audit: list[str] = []

    @property
    def tripped(self) -> bool:
        return self.path.exists() and json.loads(self.path.read_text()).get("tripped", False)

    def trip(self, reason: str) -> None:
        if not self.tripped:
            self.path.write_text(json.dumps({"tripped": True, "reason": reason}))
            self.audit.append(f"trip: {reason}")

    def reset(self, confirm: str) -> bool:
        if confirm != "I have flattened and reviewed":
            self.audit.append("reset refused")
            return False
        self.path.write_text(json.dumps({"tripped": False}))
        self.audit.append("reset")
        return True


class AsyncSimBroker:
    """Slow broker for the time-to-flat drill: every request takes `latency` seconds."""

    def __init__(self, name: str, n_orders: int, positions: dict, latency: float = 0.05):
        self.name, self.latency = name, latency
        self.open = [f"{name}-{i}" for i in range(n_orders)]
        self.positions = dict(positions)

    async def cancel(self, oid: str) -> None:
        await asyncio.sleep(self.latency)
        self.open.remove(oid)

    async def close(self, sym: str) -> None:
        await asyncio.sleep(self.latency)
        self.positions[sym] = Decimal(0)


# ---- a tiny SimBroker and the broker contract tests (S16)
class SimBroker:
    """In-memory broker: idempotent on client order id, market orders fill at `last`, limits rest."""

    def __init__(self, last: dict[str, Decimal]):
        self.last, self.orders, self.positions = dict(last), {}, {}

    def submit(self, o: Order) -> str:
        if o.qty <= 0:
            raise ValueError("qty must be positive")
        if o.client_id in self.orders:
            return o.client_id                                      # idempotent: same id, same order
        self.orders[o.client_id] = {"order": o, "state": "ACCEPTED"}
        if o.type == "MKT":
            signed = o.qty if o.side == "BUY" else -o.qty
            self.positions[o.symbol] = self.positions.get(o.symbol, Decimal(0)) + signed
            self.orders[o.client_id]["state"] = "FILLED"
        return o.client_id

    def cancel(self, client_id: str) -> None:
        if client_id not in self.orders:
            raise KeyError(client_id)
        if self.orders[client_id]["state"] == "ACCEPTED":
            self.orders[client_id]["state"] = "CANCELLED"

    def open_orders(self) -> list[str]:
        return sorted(k for k, v in self.orders.items() if v["state"] == "ACCEPTED")


class SloppyBroker(SimBroker):
    """Looks fine in a demo: not idempotent (a retry doubles the position) and accepts qty 0."""

    def submit(self, o: Order) -> str:
        cid = o.client_id if o.client_id not in self.orders else f"{o.client_id}-{len(self.orders)}"
        self.orders[cid] = {"order": o, "state": "ACCEPTED"}
        if o.type == "MKT":
            signed = o.qty if o.side == "BUY" else -o.qty
            self.positions[o.symbol] = self.positions.get(o.symbol, Decimal(0)) + signed
            self.orders[cid]["state"] = "FILLED"
        return cid


def contract_tests(make_broker) -> dict[str, bool]:
    """The same tests run against every adapter (IB, Alpaca, Sim). Each gets a fresh broker."""
    D, last = Decimal, {"SPY": Decimal("512.00")}
    results = {}

    def run(name, fn):
        try:
            fn(make_broker(last))
            results[name] = True
        except Exception:  # noqa: BLE001 - any exception is a failed contract
            results[name] = False

    def market_fill_moves_position(b):
        b.submit(Order("SPY", "BUY", D("10"), "MKT", client_id="qf-a"))
        assert b.positions["SPY"] == 10

    def retry_is_idempotent(b):
        o = Order("SPY", "BUY", D("10"), "MKT", client_id="qf-b")
        b.submit(o)
        b.submit(o)                                                  # a timeout → the same order again
        assert b.positions["SPY"] == 10

    def limit_rests_and_cancels(b):
        b.submit(Order("SPY", "BUY", D("5"), "LMT", limit=D("500"), client_id="qf-c"))
        assert b.open_orders() == ["qf-c"]
        b.cancel("qf-c")
        assert b.open_orders() == []

    def rejects_zero_qty(b):
        try:
            b.submit(Order("SPY", "BUY", D("0"), "MKT", client_id="qf-d"))
        except ValueError:
            return
        raise AssertionError("accepted qty 0")

    def cancel_unknown_raises(b):
        try:
            b.cancel("qf-nope")
        except KeyError:
            return
        raise AssertionError("silently ignored an unknown id")

    for f in (market_fill_moves_position, retry_is_idempotent, limit_rests_and_cancels, rejects_zero_qty,
              cancel_unknown_raises):
        run(f.__name__, f)
    return results


def half_up(price: Decimal, tick: Decimal) -> Decimal:
    return (price / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick
