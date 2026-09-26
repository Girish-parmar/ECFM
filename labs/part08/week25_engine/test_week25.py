import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import quick_eval_pnl, regime_market, tsmom_signal

en = load("week25_engine", "engine")
BARS = regime_market(n_blocks=8, seed=3)


def test_config_hash_and_research_log(tmp_path):
    a = en.config_hash({"lookback": 120, "vol_n": 20})
    assert a == en.config_hash({"vol_n": 20, "lookback": 120}) and len(a) == 12
    assert a != en.config_hash({"lookback": 121, "vol_n": 20})
    log = en.ResearchLog(str(tmp_path / "log.duckdb"))
    assert log.record("tsmom", {"lookback": 120}, "v1", 0.8, -0.2, 1000) == 1
    assert log.record("tsmom", {"lookback": 140}, "v1", 0.5, -0.25, 1000) == 2
    assert log.record("rsi2", {"entry": 10}, "v1", 0.3, -0.1, 1000) == 3
    t = log.trials("tsmom")
    assert list(t["run_id"]) == [1, 2] and list(t["sharpe"]) == [0.8, 0.5]
    assert len(log.trials()) == 3 and t["params"].iloc[0] == '{"lookback": 120}'
    again = en.ResearchLog(str(tmp_path / "log.duckdb"))          # persisted on disk
    log.con.close()
    assert len(again.trials()) == 3


def test_event_queue_ordering():
    q = en.EventQueue()
    ts = pd.Timestamp("2026-01-05 14:30", tz="UTC")
    q.push(ts, 2, "timer")
    q.push(ts, 1, "bar", "A")
    q.push(ts - pd.Timedelta("1min"), 2, "timer-early")
    q.push(ts, 0, "fill")
    q.push(ts, 1, "bar", "B")
    got = []
    while q:
        e = q.pop()
        got.append((e.kind, e.data))
    assert got == [("timer-early", None), ("fill", None), ("bar", "A"), ("bar", "B"), ("timer", None)]


def test_portfolio_ledger_with_futures_multiplier():
    pf = en.Portfolio(100_000, {"ES": 50})
    pf.on_fill("SPY", 100, 500.0, fee=1.0)
    pf.on_fill("ES", 1, 5000.0, fee=2.25)
    pf.mark("SPY", 510.0)
    pf.mark("ES", 4990.0)
    assert pf.cash == pytest.approx(100_000 - 50_000 - 1 - 250_000 - 2.25)
    assert pf.equity == pytest.approx(100_000 - 3.25 + 100 * 10 - 50 * 10) and pf.fees == pytest.approx(3.25)


def test_fill_models():
    bar = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0}
    assert en.market_fill_price(10, 100.0, 5) == pytest.approx(100.05) and en.market_fill_price(-10, 100.0, 5) == pytest.approx(99.95)
    assert en.limit_fill_price(10, 99.0, bar) == 99.0 and en.limit_fill_price(10, 98.0, bar) is None     # touch only
    assert en.limit_fill_price(10, 101.0, bar) == 100.0                                                  # better open
    assert en.limit_fill_price(-10, 101.5, bar) == 101.5 and en.limit_fill_price(-10, 102.0, bar) is None
    gap_down = {"open": 95.0, "high": 96.0, "low": 94.0, "close": 95.5}
    assert en.stop_fill_price(-10, 97.0, gap_down) == 95.0                  # the gap makes it worse than the stop
    assert en.stop_fill_price(-10, 98.5, bar) == 98.5 and en.stop_fill_price(-10, 97.0, bar) is None
    assert en.stop_fill_price(10, 103.0, bar) is None
    assert en.stop_fill_price(10, 101.0, bar) == 101.0
    assert en.participation_cap(-50_000, 200_000, 0.10) == -20_000 and en.participation_cap(500, 200_000) == 500


def test_costs():
    assert en.ib_fixed_commission(100, 50.0) == 1.0                         # minimum
    assert en.ib_fixed_commission(1000, 50.0) == 5.0
    assert en.ib_fixed_commission(1000, 0.10) == pytest.approx(1.0)          # capped at 1% of value
    assert en.ib_fixed_commission(0, 10.0) == 0.0
    assert en.sqrt_impact_bps(10_000, 1_000_000, 0.02) == pytest.approx(20.0)   # lesson-plan example
    assert en.sqrt_impact_bps(40_000, 1_000_000, 0.02) == pytest.approx(40.0)   # 4× size -> 2× cost


def test_engine_fills_next_bar_and_is_deterministic():
    decided = np.zeros(len(BARS))
    decided[10] = 1.0
    decided[11:20] = 1.0
    out = en.run_backtest(BARS, decided)
    assert (out["position"].iloc[:11] == 0).all() and out["position"].iloc[11] > 0      # filled at bar 11's open
    assert out["position"].iloc[20] > 0 and out["position"].iloc[21] == 0
    pd.testing.assert_frame_equal(out, en.run_backtest(BARS, decided))


def test_parity_with_the_first_look_evaluator():
    decided = tsmom_signal(BARS["close"].to_numpy(), 120, 20)
    out = en.run_backtest(BARS, decided)
    eng = out["equity_open"].pct_change().shift(-1).fillna(0).to_numpy()   # open-to-open, like quick_eval
    ql = quick_eval_pnl(decided, BARS["open"].to_numpy(), cost_bps=0)
    assert np.corrcoef(eng[130:-2], ql[130:-2])[0, 1] > 0.99               # differences: share rounding, weight drift
    assert abs(np.prod(1 + eng) - np.prod(1 + ql)) < 0.05


def test_costs_and_participation_reduce_the_result():
    decided = tsmom_signal(BARS["close"].to_numpy(), 120, 20)
    free = en.run_backtest(BARS, decided)["equity"].iloc[-1]
    costly = en.run_backtest(BARS, decided, slippage_bps=10, commission=en.ib_fixed_commission)
    assert costly["equity"].iloc[-1] < free and costly["fees"].iloc[-1] > 0
    thin = BARS.assign(volume=500.0)                                        # tiny volume: fills are rationed
    capped = en.run_backtest(thin, decided, max_participation=0.1)
    assert capped["position"].abs().max() < en.run_backtest(BARS, decided)["position"].abs().max()
