"""Clinic W2 — download 5-minute bars from IB and Alpaca, normalize both to the canonical schema, save Parquet.

Uses YOUR week 13/14 functions: ib_chunks + PacingGuard (IB pacing), normalize_ib, normalize_alpaca.
Run from labs/part04 (same set-up as connect_both.py):
    python paper/download_bars.py --symbols SPY AAPL MSFT --days 30 --out data/part04
Output: <out>/source=<ib|alpaca_iex>/symbol=<SYM>/bars.parquet. Then compare the two sources for Clinic W2.
"""
import argparse
import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import _setup  # noqa: F401
from _loader import load
from common import AssetClass, Instrument

fd = load("week13_foundations", "foundations")
md = load("week14_data", "market_data")


async def ib_bars(s, symbols, start, end) -> pd.DataFrame:
    from ib_async import IB, util
    fd.check_port_matches_env(s.env, s.ib_port)
    ib = IB()
    await ib.connectAsync(s.ib_host, s.ib_port, clientId=s.ib_client_id, timeout=10)
    guard = md.PacingGuard(time.monotonic)
    frames = []
    try:
        for sym in symbols:
            [contract] = await ib.qualifyContractsAsync(fd.to_ib_contract(Instrument(sym, AssetClass.EQUITY)))
            for chunk_end, duration in md.ib_chunks(start, end, chunk_days=10):
                key = (sym, "5 mins", "TRADES", chunk_end.isoformat())
                wait = guard.wait(key)
                if wait:
                    print(f"  pacing: waiting {wait:.1f}s")
                    await asyncio.sleep(wait)
                guard.record(key)
                bars = await ib.reqHistoricalDataAsync(contract, endDateTime=chunk_end, durationStr=duration,
                                                       barSizeSetting="5 mins", whatToShow="TRADES",
                                                       useRTH=True, formatDate=2)
                if bars:
                    frames.append(md.normalize_ib(util.df(bars), sym, "5m"))
            print(f"IB {sym}: done")
    finally:
        ib.disconnect()
    out = pd.concat(frames) if frames else pd.DataFrame()
    return out.drop_duplicates(["symbol", "ts"]).sort_values(["symbol", "ts"]) if len(out) else out


def alpaca_bars(s, symbols, start, end) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    dc = StockHistoricalDataClient(s.alpaca_key.get_secret_value(), s.alpaca_secret.get_secret_value())
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                           start=start, end=end, feed=DataFeed.IEX, adjustment=Adjustment.ALL)
    return md.normalize_alpaca(dc.get_stock_bars(req).df, "5m", feed="iex")


def save(df: pd.DataFrame, out: Path) -> None:
    for (source, sym), part in df.groupby(["source", "symbol"]):
        d = out / f"source={source}" / f"symbol={sym.replace('/', '-')}"
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / "bars.parquet", index=False)
        print(f"saved {len(part):>6} bars -> {d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["SPY", "AAPL"])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("data/part04"))
    ap.add_argument("--skip-ib", action="store_true")
    ap.add_argument("--skip-alpaca", action="store_true")
    args = ap.parse_args()
    s = _setup.PaperSettings()
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=20)   # recent SIP data is paid; IEX is fine
    start = end - timedelta(days=args.days)
    if not args.skip_ib:
        save(asyncio.run(ib_bars(s, args.symbols, start, end)), args.out)
    if not args.skip_alpaca:
        save(alpaca_bars(s, args.symbols, start, end), args.out)


if __name__ == "__main__":
    main()
