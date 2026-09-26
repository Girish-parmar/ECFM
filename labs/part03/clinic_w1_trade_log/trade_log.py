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
    raise NotImplementedError("✍️ Your turn: see the docstring")


def parse_fills(path: str | Path) -> tuple[list[dict], list[str]]:
    """Parse the file. Return (fills sorted by ts, errors) where each error is 'line N: reason'
    (N = line number in the file, header is line 1). Never raise for a bad row."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def realized_pnl(fills: list[dict]) -> dict[tuple[str, str], Decimal]:
    """FIFO realized P&L (minus fees) per (symbol, New York date 'YYYY-MM-DD' of the closing fill).
    Fees are charged on the date of the fill that paid them. Days with fees but no closes still appear."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def write_report(fills: list[dict], errors: list[str], path: str | Path) -> dict:
    """Write JSON: {"fills": n, "errors": [...], "pnl": {"SYMBOL": {"YYYY-MM-DD": "123.45"}}} with P&L
    as strings (exact decimals). Return the same dict."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


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
