"""Clinic W1 — connect to IB (Gateway/TWS) and Alpaca paper accounts, print both side by side, disconnect.

Needs: IB Gateway or TWS running in PAPER mode with the API enabled, and Alpaca paper keys in the environment
(or labs/part04/.env, never committed):  QF_ALPACA_KEY=...  QF_ALPACA_SECRET=...  QF_IB_PORT=4002
Run from labs/part04:  python paper/connect_both.py   [--skip-ib | --skip-alpaca]
Uses YOUR week 13 functions (P4_SOLUTIONS=1 to use the reference ones).
"""
import argparse
import asyncio

import _setup  # noqa: F401  (import path + settings)
from _loader import load

fd = load("week13_foundations", "foundations")


async def ib_snapshot(s) -> dict:
    from ib_async import IB
    fd.check_port_matches_env(s.env, s.ib_port)                          # refuse a live port in paper mode
    ib = IB()
    ib.errorEvent += lambda req_id, code, msg, contract: print(f"  IB {code}: {msg}")
    await ib.connectAsync(s.ib_host, s.ib_port, clientId=s.ib_client_id, timeout=10)
    try:
        values = await ib.accountSummaryAsync()
        return {"accounts": ib.managedAccounts(), **{k: str(v) for k, v in fd.ib_account_summary(values).items()}}
    finally:
        ib.disconnect()


def alpaca_snapshot(s) -> dict:
    from alpaca.trading.client import TradingClient
    if s.alpaca_key is None or s.alpaca_secret is None:
        raise SystemExit("set QF_ALPACA_KEY and QF_ALPACA_SECRET (paper keys)")
    tc = TradingClient(s.alpaca_key.get_secret_value(), s.alpaca_secret.get_secret_value(), paper=True)
    acct = tc.get_account().model_dump()
    clock = tc.get_clock()
    return {"equity": acct["equity"], "buying_power": acct["buying_power"], "cash": acct["cash"],
            "market": "open" if clock.is_open else f"opens {clock.next_open}",
            "warnings": fd.alpaca_account_warnings(acct) or "none"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ib", action="store_true")
    ap.add_argument("--skip-alpaca", action="store_true")
    args = ap.parse_args()
    s = _setup.PaperSettings()
    if s.env != "paper":
        raise SystemExit("Part 4 is paper-only: set QF_ENV=paper")
    rows = {}
    if not args.skip_ib:
        rows["IB"] = asyncio.run(ib_snapshot(s))
    if not args.skip_alpaca:
        rows["Alpaca"] = alpaca_snapshot(s)
    for broker, data in rows.items():
        print(f"\n== {broker} (paper) ==")
        for k, v in data.items():
            print(f"  {k:<16} {v}")


if __name__ == "__main__":
    main()
