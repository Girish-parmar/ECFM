"""Week 13 — Environment & broker foundations (Part 4, S1–S4).

Everything here runs offline: the IB contract objects come from `ib_async` but no connection is made.
Fill in every block marked "Your turn", then run:  python -m pytest week13_foundations
"""
from __future__ import annotations

import re
from decimal import Decimal

from ib_async import Contract, Crypto, Forex, Future, Stock

from common import AssetClass, Instrument, UnsupportedInstrument

# ---------------------------------------------------------------------------- S1–S2 ports
PORTS = {("gateway", "live"): 4001, ("gateway", "paper"): 4002, ("tws", "live"): 7496, ("tws", "paper"): 7497}


def ib_port(app: str, mode: str) -> int:
    """Port for app 'gateway' or 'tws' in mode 'paper' or 'live' (case-insensitive). ValueError otherwise."""
    # >>> SOLUTION
    key = (app.lower(), mode.lower())
    if key not in PORTS:
        raise ValueError(f"unknown app/mode {app!r}/{mode!r}")
    return PORTS[key]
    # <<< SOLUTION


def check_port_matches_env(env: str, port: int) -> None:
    """Safety check run at startup: in env 'paper' a LIVE port (4001/7496) raises RuntimeError, and in env 'live'
    a PAPER port raises RuntimeError (you meant to go live but are still on paper). Unknown ports: ValueError."""
    # >>> SOLUTION
    modes = {p: m for (_, m), p in PORTS.items()}
    if port not in modes:
        raise ValueError(f"unknown IB port {port}")
    if modes[port] != env:
        raise RuntimeError(f"env={env!r} but port {port} is a {modes[port]} port")
    # <<< SOLUTION


# ------------------------------------------------------------------------- S1 secrets
ALPACA_KEY_ID = re.compile(r"\b(?:PK|AK)[A-Z0-9]{18}\b")
ASSIGNMENT = re.compile(r"""(?i)\b\w*(?:secret|password|passwd|api_?key|token)\w*\s*[:=]\s*["']?([^\s"'#]{8,})""")


def find_secrets(text: str) -> list[tuple[int, str]]:
    """Return (line number starting at 1, kind) for every line that looks like it leaks a credential:
    kind 'alpaca_key_id' if ALPACA_KEY_ID matches, else 'assignment' if ASSIGNMENT matches.
    Lines whose value is a placeholder are fine: skip a match whose value starts with '<', '$' or '{'
    or is 'changeme'. Used by the pre-commit hook of week 13 and by 01_env_check."""
    # >>> SOLUTION
    hits = []
    for n, line in enumerate(text.splitlines(), start=1):
        if ALPACA_KEY_ID.search(line):
            hits.append((n, "alpaca_key_id"))
            continue
        m = ASSIGNMENT.search(line)
        if m and not (m.group(1)[0] in "<${" or m.group(1).lower() == "changeme"):
            hits.append((n, "assignment"))
    return hits
    # <<< SOLUTION


# -------------------------------------------------------------------- S3 IB contracts
MONTH_CODES = "FGHJKMNQUVXZ"                      # Jan..Dec


def canonical_symbol(inst: Instrument) -> str:
    """Canonical symbol used in the bar schema: equity 'AAPL', future 'ESZ6' (root + month code + last
    digit of the year, from expiry 'YYYYMM'), FX 'EUR.USD', crypto 'BTC/USD'."""
    # >>> SOLUTION
    match inst.asset_class:
        case AssetClass.EQUITY:
            return inst.symbol
        case AssetClass.FUTURE:
            year, month = inst.expiry[:4], int(inst.expiry[4:6])
            return f"{inst.symbol}{MONTH_CODES[month - 1]}{year[-1]}"
        case AssetClass.FX:
            return f"{inst.symbol[:3]}.{inst.symbol[3:]}"
        case AssetClass.CRYPTO:
            return f"{inst.symbol}/{inst.currency}"
    raise UnsupportedInstrument(inst.asset_class)
    # <<< SOLUTION


def to_ib_contract(inst: Instrument) -> Contract:
    """Map an Instrument to an unqualified ib_async contract:
    EQUITY  -> Stock(symbol, 'SMART', currency, primaryExchange=inst.exchange)
    FUTURE  -> Future(symbol, expiry, inst.exchange, currency=currency)   (expiry 'YYYYMM', required)
    FX      -> Forex(pair)  with pair = inst.symbol, e.g. 'EURUSD'
    CRYPTO  -> Crypto(symbol, 'PAXOS', currency)
    anything else, or a FUTURE without expiry -> UnsupportedInstrument."""
    # >>> SOLUTION
    match inst.asset_class:
        case AssetClass.EQUITY:
            return Stock(inst.symbol, "SMART", inst.currency, primaryExchange=inst.exchange)
        case AssetClass.FUTURE if inst.expiry:
            return Future(inst.symbol, inst.expiry, inst.exchange, currency=inst.currency)
        case AssetClass.FX:
            return Forex(inst.symbol)
        case AssetClass.CRYPTO:
            return Crypto(inst.symbol, "PAXOS", inst.currency)
    raise UnsupportedInstrument(f"{inst.asset_class} {inst.symbol}")
    # <<< SOLUTION


# ------------------------------------------------------------------- S3–S4 account state
IB_TAGS = ("NetLiquidation", "TotalCashValue", "BuyingPower", "MaintMarginReq")


def ib_account_summary(values, currency: str = "USD") -> dict[str, Decimal]:
    """`values` is what `ib.accountSummary()` returns: objects with .tag, .value (str) and .currency.
    Keep the tags in IB_TAGS whose currency equals `currency`; return {tag: Decimal(value)}."""
    # >>> SOLUTION
    return {v.tag: Decimal(v.value) for v in values if v.tag in IB_TAGS and v.currency == currency}
    # <<< SOLUTION


def alpaca_account_warnings(acct: dict) -> list[str]:
    """`acct` holds Alpaca account fields as strings/bools (as in the REST JSON). Return warnings, in this order:
    'trading blocked' if trading_blocked or account_blocked is true;
    'pattern day trader' if pattern_day_trader is true;
    'day trades: N' if daytrade_count >= 3;
    'low equity' if Decimal(equity) < 25000 and daytrade_count >= 3.
    (Check your broker's current day-trading rules; the thresholds here are an exercise.)"""
    # >>> SOLUTION
    out = []
    if acct.get("trading_blocked") or acct.get("account_blocked"):
        out.append("trading blocked")
    if acct.get("pattern_day_trader"):
        out.append("pattern day trader")
    n = int(acct.get("daytrade_count", 0))
    if n >= 3:
        out.append(f"day trades: {n}")
        if Decimal(acct.get("equity", "0")) < 25000:
            out.append("low equity")
    return out
    # <<< SOLUTION
