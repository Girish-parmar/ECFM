import json
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from _loader import load

tl = load("clinic_w1_trade_log", "trade_log")
SAMPLE = Path(__file__).parent / "sample_fills.csv"


def test_parse_line_normalizes_to_utc_and_signs_qty():
    f = tl.parse_line({"ts": "2025-07-15 09:30:00", "tz": "America/New_York", "symbol": "spy",
                       "side": "sell", "qty": "10", "price": "500.25", "fee": ""})
    assert f == {"ts": datetime(2025, 7, 15, 13, 30, tzinfo=ZoneInfo("UTC")), "symbol": "SPY",
                 "qty": D("-10"), "price": D("500.25"), "fee": D("0")}


@pytest.mark.parametrize("field,value,reason", [("side", "HOLD", "bad side"), ("tz", "Mars/Olympus", "unknown time zone"),
                                                ("qty", "-1", "bad qty"), ("price", "abc", "bad price"),
                                                ("ts", "2025-03-04 25:61:00", "bad timestamp"), ("fee", "-2", "bad fee")])
def test_parse_line_errors(field, value, reason):
    row = {"ts": "2025-03-04 10:00:00", "tz": "UTC", "symbol": "X", "side": "BUY", "qty": "1", "price": "1", "fee": "0"}
    row[field] = value
    with pytest.raises(ValueError, match=reason):
        tl.parse_line(row)


def test_parse_fills_collects_errors():
    fills, errors = tl.parse_fills(SAMPLE)
    assert len(fills) == 8 and len(errors) == 5
    assert errors[0] == "line 9: bad side" and errors[-1] == "line 13: bad timestamp"
    assert fills == sorted(fills, key=lambda f: f["ts"])


def test_realized_pnl_fifo_by_ny_date():
    fills, _ = tl.parse_fills(SAMPLE)
    pnl = tl.realized_pnl(fills)
    # SPY 03-03: close 100 @580.10 and 50 @581.30 at 583 -> 290 + 85 = 375, fees 3 -> 372
    assert pnl[("SPY", "2025-03-03")] == D("372.00")
    # AAPL: short 50 @240, covered @238.50 -> +75, fee 1 (second fill has no fee)
    assert pnl[("AAPL", "2025-03-03")] == D("74.00")
    # SPY 03-04: sell 100 closes remaining 50 @581.30 at 579 (-115), opens short 50 @579; buy 50 @578 covers (+50); fees 2
    assert pnl[("SPY", "2025-03-04")] == D("-67.00")
    # MSFT bought 01:30 UTC on 03-04 = 20:30 NY on 03-03: fee only, dated by New York
    assert pnl[("MSFT", "2025-03-03")] == D("-1.00")


def test_report_and_cli(tmp_path):
    out = tmp_path / "report.json"
    code = tl.main([str(SAMPLE), str(out)])
    rep = json.loads(out.read_text())
    assert code == 1 and rep["fills"] == 8 and len(rep["errors"]) == 5
    assert rep["pnl"]["SPY"]["2025-03-03"] == "372.00"
