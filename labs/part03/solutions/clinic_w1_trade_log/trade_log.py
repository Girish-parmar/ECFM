"""Clinic W1 — Trade-log parser → P&L report (Part 3, S1–S4).

Input CSV columns: ts, tz, symbol, side, qty, price, fee
  ts   'YYYY-MM-DD HH:MM:SS' in time zone `tz` (e.g. America/New_York, UTC)
  side BUY or SELL; qty > 0; price > 0; fee >= 0 (empty fee = 0)

Tasks: parse and validate every line (collect errors instead of crashing), normalize times to UTC,
compute realized P&L per symbol and New York trading date with FIFO matching, write a JSON report.
Run:  python trade_log.py sample_fills.csv report.json      Test:  python -m pytest clinic_w1_trade_log
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

NY, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
FIELDS = ["ts", "tz", "symbol", "side", "qty", "price", "fee"]


def parse_line(row: dict) -> dict:
    """Validate one CSV row and return {"ts": aware UTC datetime, "symbol", "qty" (signed Decimal:
    + BUY, − SELL), "price": Decimal, "fee": Decimal}. Raise ValueError with a short reason on bad input:
    'bad timestamp', 'unknown time zone', 'bad side', 'bad qty', 'bad price', 'bad fee'."""
    # >>> SOLUTION
    try:
        tz = ZoneInfo(row["tz"])
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        raise ValueError("unknown time zone") from None
    try:
        ts = datetime.strptime(row["ts"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).astimezone(UTC)
    except (ValueError, TypeError):
        raise ValueError("bad timestamp") from None
    side = (row.get("side") or "").strip().upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("bad side")

    def dec(field, allow_zero=False, default=None):
        raw = (row.get(field) or "").strip()
        if raw == "" and default is not None:
            return default
        try:
            v = Decimal(raw)
        except InvalidOperation:
            raise ValueError(f"bad {field}") from None
        if v < 0 or (v == 0 and not allow_zero) or not v.is_finite():
            raise ValueError(f"bad {field}")
        return v

    qty, price, fee = dec("qty"), dec("price"), dec("fee", allow_zero=True, default=Decimal("0"))
    return {"ts": ts, "symbol": row["symbol"].strip().upper(), "qty": qty if side == "BUY" else -qty,
            "price": price, "fee": fee}
    # <<< SOLUTION


def parse_fills(path: str | Path) -> tuple[list[dict], list[str]]:
    """Parse the file. Return (fills sorted by ts, errors) where each error is 'line N: reason'
    (N = line number in the file, header is line 1). Never raise for a bad row."""
    # >>> SOLUTION
    fills, errors = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for n, row in enumerate(reader, start=2):
            try:
                fills.append(parse_line(row))
            except ValueError as e:
                errors.append(f"line {n}: {e}")
    return sorted(fills, key=lambda x: x["ts"]), errors
    # <<< SOLUTION


def realized_pnl(fills: list[dict]) -> dict[tuple[str, str], Decimal]:
    """FIFO realized P&L (minus fees) per (symbol, New York date 'YYYY-MM-DD' of the closing fill).
    Fees are charged on the date of the fill that paid them. Days with fees but no closes still appear."""
    # >>> SOLUTION
    lots: dict[str, deque] = defaultdict(deque)
    pnl: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    for f in sorted(fills, key=lambda x: x["ts"]):
        key = (f["symbol"], f["ts"].astimezone(NY).date().isoformat())
        pnl[key] -= f["fee"]
        book, q = lots[f["symbol"]], f["qty"]
        while q != 0 and book and (book[0][0] > 0) != (q > 0):
            lot_q, lot_p = book[0]
            close = min(abs(q), abs(lot_q))
            s = 1 if lot_q > 0 else -1
            pnl[key] += (f["price"] - lot_p) * close * s
            lot_q -= s * close
            q += s * close
            if lot_q == 0:
                book.popleft()
            else:
                book[0] = (lot_q, lot_p)
        if q != 0:
            book.append((q, f["price"]))
    return dict(pnl)
    # <<< SOLUTION


def write_report(fills: list[dict], errors: list[str], path: str | Path) -> dict:
    """Write JSON: {"fills": n, "errors": [...], "pnl": {"SYMBOL": {"YYYY-MM-DD": "123.45"}}} with P&L
    as strings (exact decimals). Return the same dict."""
    # >>> SOLUTION
    out: dict = {"fills": len(fills), "errors": errors, "pnl": {}}
    for (sym, day), v in sorted(realized_pnl(fills).items()):
        out["pnl"].setdefault(sym, {})[day] = str(v)
    Path(path).write_text(json.dumps(out, indent=2))
    return out
    # <<< SOLUTION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trade log -> P&L report")
    ap.add_argument("fills_csv")
    ap.add_argument("report_json")
    args = ap.parse_args(argv)
    fills, errors = parse_fills(args.fills_csv)
    report = write_report(fills, errors, args.report_json)
    print(f"{report['fills']} fills, {len(errors)} errors -> {args.report_json}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
