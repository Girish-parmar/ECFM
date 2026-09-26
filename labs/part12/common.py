"""Shared, complete data for the Part 12 labs (nothing to fill in here).

Part 12 finishes and operates the platform. The labs exercise each operating discipline offline, on synthetic data
with known answers: a toy package with planted layer violations, bars and universe snapshots, a quote path for
execution, a trade journal with a planted edge, latency samples, and a replayable trading day with injected faults.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = ["apps", "monitoring", "research", "strategy", "execution", "lib", "data", "domain", "core"]


# ------------------------------------------------------------------------------ S1 a toy platform
TOY_MODULES = {
    "core/__init__.py": "", "core/events.py": "import dataclasses\n",
    "domain/__init__.py": "", "domain/orders.py": "from quantforge.core import events\n",
    "data/__init__.py": "", "data/bars.py": "import pandas as pd\nfrom quantforge.domain.orders import Order\n",
    "lib/__init__.py": "", "lib/indicators.py": "import numpy as np\nfrom quantforge.data import bars\n",
    "execution/__init__.py": "", "execution/oms.py": "from quantforge.lib import indicators\nfrom quantforge.domain import orders\n",
    "strategy/__init__.py": "", "strategy/momentum.py": "from quantforge.lib.indicators import ema\n",
    "research/__init__.py": "", "research/backtest.py": "from quantforge.strategy import momentum\n",
    "monitoring/__init__.py": "", "monitoring/audit.py": "import hashlib\nfrom quantforge.core.events import Event\n",
    "apps/__init__.py": "", "apps/api.py": "from quantforge.monitoring import audit\nfrom quantforge.execution import oms\n",
    "brokers/__init__.py": "", "brokers/ib.py": "from quantforge.domain.orders import Order\n",
}
VIOLATIONS = {
    "strategy/shortcut.py": "from quantforge.brokers.ib import submit\n",         # strategy talks to a broker
    "lib/report.py": "import quantforge.apps.api\n",                                # lower layer imports a higher one
    "core/util.py": "def f():\n    from quantforge.execution import oms\n",          # hidden inside a function
}


def make_toy_package(root: Path, violations: bool = True) -> Path:
    """Write a small `quantforge` package under root (optionally with three planted violations); return its path."""
    pkg = Path(root) / "quantforge"
    files = {**TOY_MODULES, **(VIOLATIONS if violations else {})}
    (pkg).mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for rel, src in files.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return pkg


# --------------------------------------------------------------------------- S3–S4 bars & universe
def ohlcv(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Daily bars: open, high, low, close, volume (a trending random walk)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, n) + 0.002 * np.sin(np.arange(n) / 60)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.lognormal(np.log(1e6), 0.3, n).round()},
                        index=pd.bdate_range("2017-01-02", periods=n))


SECTORS = ["tech", "health", "energy", "financials", "industrials", "etf"]


def universe_snapshot(n: int = 300, date="2025-06-02", seed: int = 0) -> pd.DataFrame:
    """One day's point-in-time universe: symbol, date, price, adv_usd (average daily dollar volume), spread_bps,
    market_cap_bn, sector, shortable, option_oi (open interest), iv_rank (0–100)."""
    rng = np.random.default_rng(seed)
    sector = rng.choice(SECTORS, n, p=[0.25, 0.15, 0.1, 0.15, 0.15, 0.2])
    adv = np.exp(rng.normal(np.log(3e7), 1.6, n))
    return pd.DataFrame({
        "symbol": [f"X{i:03d}" for i in range(n)], "date": pd.Timestamp(date),
        "price": np.round(np.exp(rng.normal(np.log(60), 0.9, n)), 2), "adv_usd": adv,
        "spread_bps": np.round(np.clip(40 / np.sqrt(adv / 1e6) + rng.exponential(1, n), 0.5, None), 2),
        "market_cap_bn": np.round(adv / 1e7 * rng.lognormal(0, 0.5, n), 2), "sector": sector,
        "shortable": rng.random(n) > 0.1, "option_oi": np.round(adv / 2e3 * rng.lognormal(0, 1, n)).astype(int),
        "iv_rank": np.round(rng.uniform(0, 100, n), 1)})


def next_snapshot(snap: pd.DataFrame, seed: int = 1, noise: float = 0.15) -> pd.DataFrame:
    """The next day's universe: the same names with ADV, spread and IV rank moved by random noise (given)."""
    rng = np.random.default_rng(seed)
    out = snap.copy()
    out["date"] = snap["date"] + pd.tseries.offsets.BDay(1)
    out["adv_usd"] = snap["adv_usd"] * np.exp(rng.normal(0, noise, len(snap)))
    out["spread_bps"] = np.round(snap["spread_bps"] * np.exp(rng.normal(0, noise, len(snap))), 2)
    out["iv_rank"] = np.clip(snap["iv_rank"] + rng.normal(0, 10, len(snap)), 0, 100).round(1)
    return out


def latency_samples(n: int = 300, spike_at: int | None = 200, seed: int = 0) -> np.ndarray:
    """Order-ack latencies in seconds (lognormal around 40 ms) with one 400 ms spike (lesson plan S11)."""
    rng = np.random.default_rng(seed)
    x = rng.lognormal(np.log(0.04), 0.25, n)
    if spike_at is not None:
        x[spike_at] = 0.4
    return x


# ------------------------------------------------------------------------------ S5–S6 execution
def quote_path(n: int = 3600, tick: float = 0.01, seed: int = 0, p_same: float = 0.8, p_with: float = 0.3,
               p_against: float = 0.05) -> pd.DataFrame:
    """One second per row: bid, ask (spread 2–5 ticks) and the last trade (at the bid or the ask). ORDER FLOW IS
    PERSISTENT AND INFORMED: the next trade is on the same side with probability p_same, and the mid ticks WITH the
    flow with probability p_with (against it p_against). So a passive buy tends to fill when sellers are active and
    the price is falling, and goes unfilled when the price runs away. Index = seconds."""
    rng = np.random.default_rng(seed)
    at_ask = np.empty(n, dtype=bool)
    at_ask[0] = True
    for t in range(1, n):
        at_ask[t] = at_ask[t - 1] if rng.random() < p_same else not at_ask[t - 1]
    u = rng.random(n)
    step = np.where(u < p_with, 1, np.where(u < p_with + p_against, -1, 0))
    step = np.where(at_ask, step, -step)
    mid = 10_000 + np.r_[0, np.cumsum(step[:-1])]
    half = rng.integers(1, 3, n)
    bid, ask = (mid - half) * tick, (mid + half + rng.integers(0, 2, n)) * tick
    return pd.DataFrame({"bid": np.round(bid, 2), "ask": np.round(ask, 2),
                         "trade": np.round(np.where(at_ask, ask, bid), 2)})


def intraday_session(seed: int = 0) -> pd.DataFrame:
    """A 390-minute session: price (random walk from 100) and volume with the usual U shape (heavy at the open and
    the close). Index = minute timestamps."""
    rng = np.random.default_rng(seed)
    m = np.arange(390)
    shape = 1 + 2.5 * np.exp(-m / 30) + 2.0 * np.exp(-(389 - m) / 25)
    volume = np.round(shape * 20_000 * rng.lognormal(0, 0.2, 390))
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 390)))
    return pd.DataFrame({"price": price, "volume": volume},
                        index=pd.date_range("2025-06-02 09:30", periods=390, freq="min"))


# ------------------------------------------------------------------------------ S8 the journal
def trade_journal(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A trade journal with a PLANTED edge: "breakout" trades earn +0.6R on average in the trend regime and −0.3R in
    chop; "pullback" trades earn +0.1R everywhere. Columns: trade_id, entry_time, setup_tag, regime, side, qty,
    planned_qty, entry, stop, exit, exit_reason. About 3% of trades have no stop, some are oversized or outside
    the allowed hours (rule violations)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tag = str(rng.choice(["breakout", "pullback"]))
        regime = str(rng.choice(["trend", "chop"]))
        mu = {("breakout", "trend"): 0.6, ("breakout", "chop"): -0.3}.get((tag, regime), 0.1)
        r = float(np.clip(rng.normal(mu, 1.0), -1.2, 5))
        side = int(rng.choice([1, -1]))
        entry = round(float(rng.uniform(20, 200)), 2)
        risk = round(entry * float(rng.uniform(0.01, 0.03)), 2)
        stop = round(entry - side * risk, 2)
        exit_ = round(entry + side * r * risk, 2)
        minute = int(rng.integers(0, 390)) if rng.random() > 0.04 else int(rng.integers(390, 420))
        planned = int(rng.integers(1, 5)) * 100
        qty = planned * (2 if rng.random() < 0.03 else 1)
        rows.append({"trade_id": i, "entry_time": pd.Timestamp("2025-06-02 09:30") + pd.Timedelta(days=i // 8,
                     minutes=minute), "setup_tag": tag, "regime": regime, "side": side, "qty": qty,
                     "planned_qty": planned, "entry": entry, "stop": np.nan if rng.random() < 0.03 else stop,
                     "exit": exit_, "exit_reason": "stop" if r <= -1 else "target" if r >= 2 else "time"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- clinic W3: a day with faults
FAULTS = {"bad_tick": 7_000, "stale_data": 12_000, "latency_spike": 150, "runaway_order_rate": 300,
          "duplicate_fill": "F0042"}


def trading_day(seed: int = 0, faults: bool = True) -> dict:
    """A replayable session with INJECTED faults (positions in FAULTS; the truth for grading):
    ticks: DataFrame ts (seconds from the open), price — one tick per second, except a 30-second silence starting at
    FAULTS["stale_data"] and a bad print (price × 1.08) at FAULTS["bad_tick"];
    acks: order-ack latencies (s), one 400 ms spike at FAULTS["latency_spike"];
    order_counts: orders per minute (390 minutes), a runaway burst of 60 at minute FAULTS["runaway_order_rate"];
    fills: list of {fill_id, symbol, qty}, one fill delivered twice (FAULTS["duplicate_fill"]);
    broker_positions: the broker's (true) positions."""
    rng = np.random.default_rng(seed)
    n = 23_400
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.0002, n)))
    ts = np.arange(n)
    if faults:
        price[FAULTS["bad_tick"]] *= 1.08
        keep = (ts < FAULTS["stale_data"]) | (ts >= FAULTS["stale_data"] + 30)
        ts, price = ts[keep], price[keep]
    acks = rng.lognormal(np.log(0.04), 0.25, 400)
    counts = rng.poisson(3, 390).astype(float)
    if faults:
        acks[FAULTS["latency_spike"]] = 0.4
        counts[FAULTS["runaway_order_rate"]] = 60
    fills, pos = [], {}
    for i in range(80):
        sym = str(rng.choice(["SPY", "QQQ", "IWM"]))
        qty = int(rng.choice([-100, 100]))
        fills.append({"fill_id": f"F{i:04d}", "symbol": sym, "qty": qty})
        pos[sym] = pos.get(sym, 0) + qty
    if faults:
        dup = next(f for f in fills if f["fill_id"] == FAULTS["duplicate_fill"])
        fills.append(dict(dup))
    return {"ticks": pd.DataFrame({"ts": ts, "price": price}), "acks": acks, "order_counts": counts,
            "fills": fills, "broker_positions": {k: v for k, v in pos.items() if v != 0}}


# ------------------------------------------------------------------------ clinic W5: a track record
def track_record(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest daily returns (1000 days) and a 4-week live record (20 days) for three strategies with known daily
    mean / vol: momentum (0.05% / 1.0%), pairs (0.04% / 0.5%), options (0.06% / 0.8%). LIVE, momentum and pairs
    behave like their backtest; the options strategy has STOPPED WORKING and bleeds (−0.8% a day)."""
    rng = np.random.default_rng(seed)
    spec = {"momentum": (0.0005, 0.010), "pairs": (0.0004, 0.005), "options": (0.0006, 0.008)}
    bt = pd.DataFrame({k: rng.normal(m, s, 1000) for k, (m, s) in spec.items()})
    live = pd.DataFrame({k: rng.normal(-0.008 if k == "options" else m, s, 20) for k, (m, s) in spec.items()},
                        index=pd.bdate_range("2025-11-03", periods=20))
    return bt, live
