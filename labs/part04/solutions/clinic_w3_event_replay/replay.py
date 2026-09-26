"""Clinic W3 — Replay a bracket order's event log through the order state machine (Part 4, S11–S12).

The same bracket (entry BUY 10 SPY LMT 500, take-profit SELL LMT 502, stop-loss SELL STP 499) was run on
IB paper and Alpaca paper. Each broker's raw events were saved as JSON lines, with the quirks you meet in
real logs: duplicated executions after a reconnect, a status that arrives before its execution, a late 'new'.
Normalize both logs, replay them through your week 15 OrderTracker, and prove the final states agree.

Run:  python replay.py bracket_orders.json events_ib.jsonl events_alpaca.jsonl
Test: python -m pytest clinic_w3_event_replay          (needs week15_orders done)
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # labs/part04 (or solutions/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))             # labs/part04 when run from solutions/
from _loader import load                                                  # noqa: E402
from common import AssetClass, Instrument, OrderRequest, OrdType, Side  # noqa: E402

od = load("week15_orders", "orders")


def load_orders(path: str | Path) -> list[OrderRequest]:
    """Read the bracket legs from JSON (given)."""
    out = []
    for o in json.loads(Path(path).read_text()):
        out.append(OrderRequest(
            Instrument(o["symbol"], AssetClass.EQUITY), Side(o["side"]), Decimal(o["qty"]), OrdType(o["type"]),
            limit_price=Decimal(o["limit_price"]) if o.get("limit_price") else None,
            stop_price=Decimal(o["stop_price"]) if o.get("stop_price") else None,
            client_order_id=o["client_order_id"]))
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalize_event(raw: dict) -> dict:
    """Turn one raw broker event into ONE canonical record:
      {"coid", "kind": "status", "state": OrderState}                              or
      {"coid", "kind": "fill", "exec_id", "qty": Decimal, "price": Decimal}
    IB:     type 'orderStatus'  -> status via od.map_ib_status(status, filled); coid = orderRef
            type 'execDetails'  -> fill (execId, shares, price); use Decimal(str(x)) for floats
    Alpaca: event 'fill' or 'partial_fill' -> fill (execution_id, qty, price); any other event -> status via
            od.map_alpaca_status(event); coid = order.client_order_id
    Unknown broker or IB type -> ValueError."""
    # >>> SOLUTION
    if raw.get("broker") == "ib":
        if raw["type"] == "orderStatus":
            return {"coid": raw["orderRef"], "kind": "status", "state": od.map_ib_status(raw["status"], raw["filled"])}
        if raw["type"] == "execDetails":
            return {"coid": raw["orderRef"], "kind": "fill", "exec_id": raw["execId"],
                    "qty": Decimal(str(raw["shares"])), "price": Decimal(str(raw["price"]))}
        raise ValueError(f"unknown IB event type {raw['type']!r}")
    if raw.get("broker") == "alpaca":
        coid = raw["order"]["client_order_id"]
        if raw["event"] in ("fill", "partial_fill"):
            return {"coid": coid, "kind": "fill", "exec_id": raw["execution_id"],
                    "qty": Decimal(raw["qty"]), "price": Decimal(raw["price"])}
        return {"coid": coid, "kind": "status", "state": od.map_alpaca_status(raw["event"])}
    raise ValueError(f"unknown broker {raw.get('broker')!r}")
    # <<< SOLUTION


def replay(orders: list[OrderRequest], raw_events: list[dict]):
    """Create an od.OrderTracker, track every order, then feed every normalized event in log order
    (status -> on_status, fill -> on_fill). Return the tracker."""
    # >>> SOLUTION
    tr = od.OrderTracker()
    for o in orders:
        tr.track(o)
    for raw in raw_events:
        e = normalize_event(raw)
        if e["kind"] == "status":
            tr.on_status(e["coid"], e["state"])
        else:
            tr.on_fill(e["coid"], e["exec_id"], e["qty"], e["price"])
    return tr
    # <<< SOLUTION


def summary(tracker) -> dict:
    """JSON-ready summary: {"orders": {coid: {"state": name, "filled": str, "avg_price": str or None}},
    "positions": {symbol: str}, "anomalies": [list(a) for each anomaly]}. Decimal values as str,
    avg_price quantized to 0.0001."""
    # >>> SOLUTION
    orders, symbols = {}, set()
    for coid, t in tracker.orders.items():
        symbols.add(t.req.instrument.symbol)
        avg = t.avg_price
        orders[coid] = {"state": t.state.value, "filled": str(t.filled),
                        "avg_price": str(avg.quantize(Decimal("0.0001"))) if avg is not None else None}
    return {"orders": orders, "positions": {s: str(tracker.position(s)) for s in sorted(symbols)},
            "anomalies": [list(a) for a in tracker.anomalies]}
    # <<< SOLUTION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay broker event logs and compare final states")
    ap.add_argument("orders")
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args(argv)
    orders = load_orders(args.orders)
    results = {Path(p).name: summary(replay(orders, read_jsonl(p))) for p in args.logs}
    print(json.dumps(results, indent=2))
    finals = [{k: (v["state"], v["filled"]) for k, v in r["orders"].items()} for r in results.values()]
    same = all(f == finals[0] for f in finals)
    print("final states match" if same else "FINAL STATES DIFFER")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
