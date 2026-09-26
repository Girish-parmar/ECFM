"""Week 42 (S5–S8) — Execution and trade management: spread-aware limit prices and a chase schedule (simulated on a
quote path), TWAP / VWAP / POV schedules with implementation shortfall, an option position manager, and the trade
journal with R-multiples, expectancy tables and rule-violation checks.

Rule of the week: every order has a cost; measure it (TCA) and feed it back into the backtest cost model.
Fill in every block marked "Your turn", then run:  python -m pytest week42_execution
"""
from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_UP, Decimal

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------ S5 spread-aware orders
def limit_price(side: str, bid: float, ask: float, tick: float, aggressiveness: float = 0.0) -> float:
    """Lesson plan S5: BUY at bid + a·(ask − bid), SELL at ask − a·(ask − bid); round to the tick with Decimal in the
    direction that never overpays (BUY down, SELL up)."""
    # >>> SOLUTION
    px = bid + aggressiveness * (ask - bid) if side == "BUY" else ask - aggressiveness * (ask - bid)
    q = Decimal(str(tick))
    steps = (Decimal(str(px)) / q).to_integral_value(rounding=ROUND_DOWN if side == "BUY" else ROUND_UP)
    return float(steps * q)
    # <<< SOLUTION


def simulate_chase(side: str, quotes: pd.DataFrame, start: int, schedule=(0.0, 0.33, 0.67, 1.0), wait: int = 5,
                   tick: float = 0.01, markout: int = 30) -> dict:
    """Offline chase on a quote path (one row per second). For each aggressiveness in the schedule: price with
    limit_price from the quote at the current second, then watch the NEXT `wait` seconds; a BUY fills at its limit
    in the first second where ask <= limit (marketable) or the last trade <= limit (a seller hit us); SELL
    mirrored. Unfilled after the schedule → cancel. sign = +1 BUY, −1 SELL. Return {"filled", "price" (NaN if not),
    "seconds" (start → fill, or time spent), "aggr" (level that filled, NaN), "cost_bps": sign·(price − arrival
    mid)/arrival mid·1e4, "markout_bps": sign·(mid `markout` seconds after the fill − price)/price·1e4 (what the
    fill is worth soon after: negative = adverse selection); both NaN if unfilled, "all_in_bps": cost_bps if filled,
    else the cost of completing by CROSSING at the last quote watched (BUY at its ask, SELL at its bid) vs the
    arrival mid — non-fill is not free}."""
    # >>> SOLUTION
    sign = 1 if side == "BUY" else -1
    mid = (quotes["bid"] + quotes["ask"]) / 2
    mid0 = mid.iloc[start]
    t = start
    for aggr in schedule:
        q = quotes.iloc[t]
        px = limit_price(side, q["bid"], q["ask"], tick, aggr)
        for s in range(t + 1, min(t + 1 + wait, len(quotes))):
            r = quotes.iloc[s]
            hit = (r["ask"] <= px or r["trade"] <= px) if side == "BUY" else (r["bid"] >= px or r["trade"] >= px)
            if hit:
                later = mid.iloc[min(s + markout, len(quotes) - 1)]
                return {"filled": True, "price": px, "seconds": s - start, "aggr": aggr,
                        "cost_bps": sign * (px - mid0) / mid0 * 1e4, "markout_bps": sign * (later - px) / px * 1e4,
                        "all_in_bps": sign * (px - mid0) / mid0 * 1e4}
        t = min(t + wait, len(quotes) - 1)
    r = quotes.iloc[t]
    cross = r["ask"] if side == "BUY" else r["bid"]
    return {"filled": False, "price": np.nan, "seconds": t - start, "aggr": np.nan, "cost_bps": np.nan,
            "markout_bps": np.nan, "all_in_bps": sign * (cross - mid0) / mid0 * 1e4}
    # <<< SOLUTION


def chase_report(quotes: pd.DataFrame, schedules: dict[str, tuple], n_orders: int = 200, seed: int = 0,
                 wait: int = 5) -> pd.DataFrame:
    """Run simulate_chase for n_orders random (side, start) pairs (rng = default_rng(seed): sides uniform in
    BUY/SELL, starts in [0, len − 100)) for every schedule. Per schedule: fill_rate, avg_seconds, avg_cost_bps and
    avg_markout_bps (filled orders only), avg_all_in_bps (all orders). Index = schedule name."""
    # >>> SOLUTION
    rng = np.random.default_rng(seed)
    orders = [(str(rng.choice(["BUY", "SELL"])), int(rng.integers(0, len(quotes) - 100))) for _ in range(n_orders)]
    rows = {}
    for name, sched in schedules.items():
        res = pd.DataFrame([simulate_chase(s, quotes, t, sched, wait) for s, t in orders])
        f = res[res["filled"]]
        rows[name] = {"fill_rate": float(res["filled"].mean()), "avg_seconds": float(f["seconds"].mean()),
                      "avg_cost_bps": float(f["cost_bps"].mean()), "avg_markout_bps": float(f["markout_bps"].mean()),
                      "avg_all_in_bps": float(res["all_in_bps"].mean())}
    return pd.DataFrame(rows).T
    # <<< SOLUTION


# --------------------------------------------------------------------------- S6 algos & TCA
def twap_schedule(qty: int, start, end, slices: int) -> pd.Series:
    """Lesson plan S6: equal integer slices at `slices` times from start (end excluded); the remainder goes to the
    first slices, one share each."""
    # >>> SOLUTION
    times = pd.date_range(start, end, periods=slices + 1)[:-1]
    base = qty // slices
    return pd.Series([base + (1 if i < qty - base * slices else 0) for i in range(slices)], index=times)
    # <<< SOLUTION


def vwap_schedule(qty: int, volume_profile: pd.Series) -> pd.Series:
    """Lesson plan S6: integer sizes proportional to the volume profile, remainder by largest fractional parts."""
    # >>> SOLUTION
    raw = volume_profile / volume_profile.sum() * qty
    sizes = np.floor(raw).astype(int)
    sizes.iloc[np.argsort(-(raw - sizes).to_numpy(), kind="stable")[: int(qty - sizes.sum())]] += 1
    return sizes
    # <<< SOLUTION


def pov_child_qty(market_volume_since_last: float, participation: float, remaining: int) -> int:
    """Lesson plan S6."""
    # >>> SOLUTION
    return int(min(remaining, np.floor(participation * market_volume_since_last)))
    # <<< SOLUTION


def implementation_shortfall_bps(side: str, decision_px: float, fills: list[tuple[float, float]],
                                 fees: float = 0.0) -> float:
    """Lesson plan S6: cost vs the decision price in bps (positive = cost)."""
    # >>> SOLUTION
    qty = sum(q for _, q in fills)
    avg = sum(p * q for p, q in fills) / qty
    sign = 1 if side == "BUY" else -1
    return 1e4 * (sign * (avg - decision_px) * qty + fees) / (decision_px * qty)
    # <<< SOLUTION


def execute_schedule(side: str, schedule: pd.Series, session: pd.DataFrame, eta: float = 0.02) -> list[tuple]:
    """Fill each child at its minute's price moved AGAINST us by temporary impact η·(child / minute volume)·price
    (a child that is a big share of the minute's volume pays more). Zero-size children are skipped. Returns fills
    [(price, qty)]."""
    # >>> SOLUTION
    sign = 1 if side == "BUY" else -1
    fills = []
    for ts, q in schedule.items():
        if q <= 0:
            continue
        row = session.loc[ts]
        fills.append((row["price"] * (1 + sign * eta * q / row["volume"]), float(q)))
    return fills
    # <<< SOLUTION


# --------------------------------------------------------------------- S7 option position manager
class OptionPositionManager:
    """Management rules for a short-premium option position, checked each bar in this PRIORITY order:
    stop (cost to close >= credit·(1 + stop_mult)) → take_profit (cost to close <= credit·(1 − tp_frac)) →
    expiry (dte == 0) → ex_dividend (a short call is ITM and days_to_exdiv <= 1: early-assignment risk) →
    adjust (any |short delta| > max_delta) → roll (dte <= roll_dte). Returns (action, reason) or None."""

    def __init__(self, tp_frac: float = 0.5, stop_mult: float = 2.0, roll_dte: int = 21, max_delta: float = 0.30):
        self.tp, self.stop, self.roll_dte, self.max_delta = tp_frac, stop_mult, roll_dte, max_delta

    def evaluate(self, pos: dict):
        """pos: credit, value (cost to close now), dte, short_deltas (list), short_call_itm (bool),
        days_to_exdiv (int or None)."""
        # >>> SOLUTION
        if pos["value"] >= pos["credit"] * (1 + self.stop):
            return "close", "stop"
        if pos["value"] <= pos["credit"] * (1 - self.tp):
            return "close", "take_profit"
        if pos["dte"] == 0:
            return "close", "expiry"
        dx = pos.get("days_to_exdiv")
        if pos.get("short_call_itm") and dx is not None and dx <= 1:
            return "close", "ex_dividend"
        if any(abs(d) > self.max_delta for d in pos["short_deltas"]):
            return "adjust", "short_delta"
        if pos["dte"] <= self.roll_dte:
            return "roll", "dte"
        return None
        # <<< SOLUTION


# ---------------------------------------------------------------------------------- S8 journal
def r_multiples(journal: pd.DataFrame) -> pd.Series:
    """R = side·(exit − entry) / |entry − stop|: P&L in units of the PLANNED risk; NaN when there is no stop."""
    # >>> SOLUTION
    return journal["side"] * (journal["exit"] - journal["entry"]) / (journal["entry"] - journal["stop"]).abs()
    # <<< SOLUTION


def journal_stats(journal: pd.DataFrame, by="setup_tag") -> pd.DataFrame:
    """Lesson plan S8 on the r_multiple column (rows with NaN R dropped): trades, win_rate, avg_win_R, avg_loss_R,
    expectancy_R per group, sorted by expectancy descending."""
    # >>> SOLUTION
    j = journal.dropna(subset=["r_multiple"])
    g = j.groupby(by)["r_multiple"]
    return pd.DataFrame({"trades": g.size(), "win_rate": g.apply(lambda r: (r > 0).mean()),
                         "avg_win_R": g.apply(lambda r: r[r > 0].mean()),
                         "avg_loss_R": g.apply(lambda r: r[r <= 0].mean()),
                         "expectancy_R": g.mean()}).sort_values("expectancy_R", ascending=False)
    # <<< SOLUTION


def rule_violations(journal: pd.DataFrame, open_="09:30", close="16:00") -> pd.DataFrame:
    """One row per violation (columns trade_id, rule): "no_stop" (stop missing), "outside_hours" (entry time not in
    [open, close)), "oversized" (qty > planned_qty). Sorted by trade_id, then rule."""
    # >>> SOLUTION
    t = journal["entry_time"].dt.strftime("%H:%M")
    checks = {"no_stop": journal["stop"].isna(), "outside_hours": ~((t >= open_) & (t < close)),
              "oversized": journal["qty"] > journal["planned_qty"]}
    rows = [{"trade_id": tid, "rule": rule} for rule, mask in checks.items() for tid in journal.loc[mask, "trade_id"]]
    return pd.DataFrame(rows, columns=["trade_id", "rule"]).sort_values(["trade_id", "rule"], ignore_index=True)
    # <<< SOLUTION
