import time

import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import LAYERS, latency_samples, make_toy_package, next_snapshot, ohlcv, universe_snapshot

it = load("week41_integration", "integration")
CONFIG = """
name: ema_cross_rsi_filter
universe: {screener: liquid_us_etfs}
timeframe: 1d
entry:
  all:
    - {fn: crossover, args: [{ind: ema, params: {n: 20}}, {ind: ema, params: {n: 50}}]}
    - not: {fn: gt, args: [{ind: rsi, params: {n: 14}}, 70]}
exit: {stop: {type: atr, n: 14, mult: 2.0}, target: {type: r_multiple, r: 3}, max_bars: 20}
sizing: {type: fixed_risk, risk_frac: 0.005}
"""


# ----------------------------------------------------------------------------------- S1
def test_module_imports_sees_everything(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import os, numpy as np\nfrom quantforge.lib import ema, sma\nfrom . import sibling\n"
                 "def g():\n    import quantforge.brokers.ib\n")
    assert it.module_imports(f) == {"os", "numpy", "quantforge.lib", "quantforge.lib.ema", "quantforge.lib.sma",
                                    "quantforge.brokers.ib"}                            # relative import skipped


def test_architecture_rules_find_the_planted_violations(tmp_path):
    graph = it.import_graph(make_toy_package(tmp_path))
    assert "quantforge.apps.api" in graph and "quantforge.core" in graph and "quantforge.core.__init__" not in graph
    assert graph["quantforge.research.backtest"] == {"quantforge.strategy", "quantforge.strategy.momentum"}
    assert it.layer_of("quantforge.strategy.momentum", LAYERS) == 3 and it.layer_of("quantforge.brokers.ib", LAYERS) is None
    layers = it.check_layers(graph, LAYERS)
    assert {imp for imp, _ in layers} == {"quantforge.core.util", "quantforge.lib.report"}   # incl. the hidden one
    forbidden = it.check_forbidden(graph, "quantforge.strategy", ["quantforge.brokers", "quantforge.execution"])
    assert {imp for imp, _ in forbidden} == {"quantforge.strategy.shortcut"}
    clean = it.import_graph(make_toy_package(tmp_path / "clean", violations=False))
    assert it.check_layers(clean, LAYERS) == [] and it.check_forbidden(clean, "quantforge.strategy", ["quantforge.brokers"]) == []


# ----------------------------------------------------------------------------------- S2
def test_latency_histogram():
    h = it.LatencyHistogram()
    for x in (0.002, 0.002, 0.02, 0.2):
        h.observe(x)
    assert h.counts == [0, 2, 2, 3, 3, 3, 4] and h.n == 4 and h.total == pytest.approx(0.224)
    assert (h.quantile(0.5), h.quantile(0.75), h.quantile(0.99)) == (0.005, 0.025, float("inf"))
    live = it.LatencyHistogram()
    for x in latency_samples(spike_at=None):
        live.observe(x)
    assert live.quantile(0.5) == 0.05 and live.quantile(0.99) == 0.1                   # p99 < 100 ms budget
    with live.time():
        time.sleep(0.002)
    assert live.n == 301


def test_ring_buffer_and_streaming_ema():
    rb = it.RingBuffer(3)
    for x in (1, 2):
        rb.append(x)
    np.testing.assert_array_equal(rb.values(), [1, 2])
    for x in (3, 4, 5):
        rb.append(x)
    np.testing.assert_array_equal(rb.values(), [3, 4, 5])                              # oldest first
    x = ohlcv(300)["close"]
    s = it.StreamingEMA(20)
    np.testing.assert_allclose([s.update(v) for v in x], x.ewm(span=20, adjust=False).mean())


def test_blocking_the_event_loop():
    blocked = it.event_loop_lag(lambda: time.sleep(0.2), in_executor=False)
    offloaded = it.event_loop_lag(lambda: time.sleep(0.2), in_executor=True)
    assert blocked > 0.15 and offloaded < 0.05


# ----------------------------------------------------------------------------------- S3
def test_config_strategy_matches_hand_code():
    bars = ohlcv()
    cfg, signal, h = it.load_strategy(CONFIG)
    hand = it.crossover(it.ema(bars["close"], 20), it.ema(bars["close"], 50)) & ~(it.rsi(bars["close"], 14) > 70)
    s = signal(bars)
    assert s.dtype == bool and s.sum() > 0 and (s == hand).all()                     # lesson plan: identical
    assert h == it.config_hash(cfg) and len(h) == 12
    cfg2 = {**cfg, "sizing": {"type": "fixed_risk", "risk_frac": 0.01}}
    assert it.config_hash(cfg2) != h and it.config_hash(dict(reversed(list(cfg.items())))) == h


def test_config_validation_rejects_before_running():
    bad = {"name": "x", "exit": {}, "entry": {"any": [
        {"fn": "gt", "args": [{"ind": "macd"}, 0]},
        {"fn": "crossover", "args": [{"ind": "ema", "params": {"n": 1000}}, {"ind": "sma", "params": {"len": 5}}]},
        {"not": {"fn": "moon", "args": ["close", 1]}}]}}
    assert it.validate_config(bad) == ["missing sizing", "unknown indicator macd", "ema.n=1000 outside [2, 500]",
                                       "unknown parameter sma.len", "unknown action moon"]
    with pytest.raises(ValueError, match="ema.n=1000"):
        it.load_strategy(CONFIG.replace("n: 20", "n: 1000"))
    assert it.validate_config(it.yaml.safe_load(CONFIG)) == []


# ----------------------------------------------------------------------------------- S4
def test_screens():
    snap = universe_snapshot()
    df = pd.DataFrame({"symbol": list("abcd"), "adv_usd": [5e7, 2e8, 1e6, 2e8], "price": [50, 5, 30, 80],
                       "spread_bps": [2, 1, 20, 3], "sector": ["etf", "tech", "etf", "etf"],
                       "option_oi": [10, 50_000, 0, 40_000], "shortable": [True, True, False, False]})
    assert it.screen(df, [it.min_adv(4e7), it.price_between(10, 100)]) == ["d", "a"]   # by ADV, descending
    assert it.screen(df, [it.sector_in(["etf"]), it.max_spread(5)]) == ["d", "a"]
    assert it.screen(df, [it.min_option_oi(30_000), it.shortable()]) == ["b"]
    assert it.screen(df, [], limit=2) == ["b", "d"]
    liquid = [it.min_adv(5e7), it.max_spread(5), it.price_between(10, 1000)]
    a, b = it.screen(snap, liquid), it.screen(next_snapshot(snap), liquid)
    assert len(a) > 30 and 0 < it.universe_turnover(a, b) < 0.25                       # screens are not free
    assert it.universe_turnover(["a", "b"], ["b", "c", "d", "e"]) == 0.75 and it.universe_turnover(["a"], []) == 0


def test_universe_store_is_point_in_time():
    store = it.UniverseStore()
    cfg = {"filters": ["min_adv 5e7"]}
    store.save("2025-06-02", "liquid", cfg, ["A", "B"])
    store.save("2025-06-04", "liquid", cfg, ["A", "C"])
    assert store.get("2025-06-03", "liquid")["symbols"] == ["A", "B"]
    assert store.get("2025-06-05", "liquid") == {"config_hash": it.config_hash(cfg), "symbols": ["A", "C"]}
    assert store.get("2025-06-01", "liquid") is None and store.get("2025-06-05", "other") is None
