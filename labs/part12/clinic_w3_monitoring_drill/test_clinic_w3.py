import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import FAULTS, trading_day

dr = load("clinic_w3_monitoring_drill", "drill")
DAY = trading_day()


def test_detectors_by_hand():
    ticks = pd.DataFrame({"ts": [0, 1, 2, 10, 11], "price": [1.0] * 5})
    assert dr.stale_gaps(ticks) == [2]
    assert dr.stream_flags([1.0] * 60 + [1.0, 50.0, 1.0]) == [61]
    pos, dups = dr.book_fills([{"fill_id": "a", "symbol": "SPY", "qty": 100}, {"fill_id": "a", "symbol": "SPY", "qty": 100},
                               {"fill_id": "b", "symbol": "QQQ", "qty": 50}, {"fill_id": "c", "symbol": "QQQ", "qty": -50}])
    assert pos == {"SPY": 100} and dups == ["a"]


def test_every_injected_fault_is_detected():
    assert dr.detect(DAY) == [("bad_tick", FAULTS["bad_tick"]), ("stale_data", FAULTS["stale_data"] - 1),
                              ("latency_spike", FAULTS["latency_spike"]),
                              ("runaway_order_rate", FAULTS["runaway_order_rate"]),
                              ("duplicate_fill", FAULTS["duplicate_fill"])]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_clean_day_raises_no_alarm(seed):
    assert dr.detect(trading_day(seed, faults=False)) == []


def test_raw_latency_z_scores_give_false_alarms():
    clean = trading_day(0, faults=False)
    assert dr.stream_flags(clean["acks"]) != [] and dr.stream_flags(np.log(clean["acks"])) == []   # skew → use logs


def test_controller_response_and_idempotent_booking():
    out = dr.run_drill(DAY)
    assert out["actions"] == {"bad_tick": "pause_strategy", "stale_data": "pause_strategy", "latency_spike": "alert",
                              "runaway_order_rate": "trip_kill_switch", "duplicate_fill": "pause_all_and_reconcile"}
    assert out["killed"] and out["paused"] == ["mom", "options", "pairs"] and out["audit_ok"]
    assert out["breaks"] == []                                                          # the duplicate was not booked
    naive = {}
    for f in DAY["fills"]:                                                              # booking without fill ids
        naive[f["symbol"]] = naive.get(f["symbol"], 0) + f["qty"]
    dup = next(f for f in DAY["fills"] if f["fill_id"] == FAULTS["duplicate_fill"])
    breaks = dr.mo.reconcile_positions({k: v for k, v in naive.items() if v}, DAY["broker_positions"])
    assert [b["symbol"] for b in breaks] == [dup["symbol"]] and breaks[0]["diff"] == -dup["qty"]
