import numpy as np
import pandas as pd
import pytest
from _loader import load
from common import CONFIGS, regime_market

fw = load("week23_framework_momentum", "framework")
MKT = regime_market(n_blocks=8, seed=1)


# --------------------------------------------------------------------------------- framework
def test_registry_and_parameter_validation():
    assert "tsmom" in fw.REGISTRY, "✍️ register() does not store strategies yet"
    assert fw.REGISTRY["tsmom"] is fw.TSMOMStrategy and fw.TSMOMStrategy.name == "tsmom"
    assert "donchian" in fw.REGISTRY
    with pytest.raises(ValueError):
        fw.register("tsmom")(type("Dup", (fw.Strategy,), {"on_bar": lambda self: None}))
    ctx = fw.StrategyContext(MKT)
    s = fw.TSMOMStrategy(ctx, lookback=60)
    assert s.p == {"lookback": 60, "vol_n": 60, "target_vol": 0.10, "max_lev": 2.0}
    with pytest.raises(ValueError):
        fw.TSMOMStrategy(ctx, lookbak=60)                     # typo -> undeclared parameter
    with pytest.raises(ValueError):
        fw.TSMOMStrategy(ctx, target_vol=5.0)                 # outside the declared range


def test_load_config(tmp_path):
    cfg = fw.load_config(CONFIGS / "tsmom_etf.yaml")
    assert cfg["strategy"] == "tsmom" and len(cfg["universe"]) == 10 and cfg["broker"] == "ib"
    (tmp_path / "a.yaml").write_text("strategy: tsmom\nuniverse: [SPY]\ntimeframe: 1d\n")
    assert fw.load_config(tmp_path / "a.yaml") == {"strategy": "tsmom", "universe": ["SPY"], "timeframe": "1d",
                                                  "params": {}, "broker": "sim"}
    for bad in ["strategy: nope\nuniverse: [SPY]\ntimeframe: 1d\n",
                "strategy: tsmom\nuniverse: []\ntimeframe: 1d\n",
                "strategy: tsmom\nuniverse: [SPY]\n",
                "strategy: tsmom\nuniverse: [SPY]\ntimeframe: 1d\nleverage: 3\n",
                "strategy: tsmom\nuniverse: [SPY]\ntimeframe: 1d\nbroker: robinhood\n"]:
        (tmp_path / "b.yaml").write_text(bad)
        with pytest.raises(ValueError):
            fw.load_config(tmp_path / "b.yaml")


class Peeker(fw.Strategy):
    """Tries to read tomorrow's close; the context must not allow it."""
    name = "peeker"

    def on_bar(self):
        h = self.ctx.history()
        assert h.index[-1] == self.ctx.bars.index[self.ctx.t] and len(h) == self.ctx.t + 1
        self.target(1.0 if len(h) % 3 == 0 else 0.0, "every third bar")


class Silent(fw.Strategy):
    name = "silent"

    def on_bar(self):
        if self.ctx.t == 5:
            self.target(0.5, "once")
            self.target(-1.0, "changed my mind")               # the last intent of a bar wins


def test_runner_and_intents():
    d = fw.run(Peeker, MKT.iloc[:10])
    np.testing.assert_array_equal(d, [0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    d = fw.run(Silent, MKT.iloc[:10])
    np.testing.assert_array_equal(d, [0, 0, 0, 0, 0, -1, -1, -1, -1, -1])
    ctx = fw.StrategyContext(MKT, "SPY")
    s = Silent(ctx)
    ctx.t = 5
    s.on_bar()
    assert [(i.t, i.symbol, i.weight, i.strategy) for i in ctx.intents] == [(5, "SPY", 0.5, "silent"),
                                                                            (5, "SPY", -1.0, "silent")]


def test_quick_eval_fills_at_the_next_open():
    open_ = np.array([100.0, 101, 99, 102, 103])
    r = fw.quick_eval(np.array([1.0, 1, 0, 0, 0]), open_, cost_bps=0)
    # decided at close 0 -> held from open 1 to open 2, then open 2 -> open 3
    np.testing.assert_allclose(r["pnl"], [0, 99 / 101 - 1, 102 / 99 - 1, 0, 0])
    assert r["exposure"] == pytest.approx(0.4) and r["max_dd"] < 0
    costly = fw.quick_eval(np.array([1.0, 1, 0, 0, 0]), open_, cost_bps=10)
    assert costly["pnl"][1] == pytest.approx(99 / 101 - 1 - 0.001)
    assert np.isnan(fw.quick_eval(np.zeros(5), open_)["sharpe"])


def test_next_bar_execution_is_mandatory():
    """The M3b gate: the same signal looks miraculous when filled on its own bar and ordinary when filled next bar."""
    class TodayUp(fw.Strategy):
        name = "today_up"

        def on_bar(self):
            b = self.ctx.history(1).iloc[-1]
            self.target(1.0 if b["close"] > b["open"] else -1.0, "today was up")
    d = fw.run(TodayUp, MKT)
    o, c = MKT["open"].to_numpy(), MKT["close"].to_numpy()
    same_bar = d * (c / o - 1)                               # "filled" at the open of the bar that created it
    same_sharpe = same_bar.mean() / same_bar.std() * np.sqrt(252)
    honest = fw.quick_eval(d, o)["sharpe"]
    assert same_sharpe > 10 and honest < same_sharpe / 4


# ---------------------------------------------------------------------------------- momentum
def test_tsmom_values():
    close = MKT["close"].to_numpy()
    w = fw.tsmom(close, lookback=60, vol_n=20, target_vol=0.1)
    assert np.isnan(w[:60]).all() and np.isfinite(w[60:]).all() and np.abs(w[60:]).max() <= 2.0
    t = 300
    r = np.diff(np.log(close))
    vol = np.std(r[t - 20:t], ddof=1) * np.sqrt(252)
    assert w[t] == pytest.approx(np.clip(np.sign(close[t] / close[t - 60] - 1) * 0.1 / vol, -2, 2))


def test_tsmom_strategy_matches_the_function():
    d = fw.run(fw.TSMOMStrategy, MKT, lookback=60, vol_n=20)
    ref = np.nan_to_num(fw.tsmom(MKT["close"].to_numpy(), 60, 20, 0.10, max_lev=2.0))
    np.testing.assert_allclose(d[61:], ref[61:], rtol=1e-10)


def test_donchian_function_and_live_strategy_agree():
    h, l, c = (MKT[k].to_numpy() for k in ("high", "low", "close"))
    pos = fw.donchian_breakout(h, l, c, 20, 10)
    assert set(np.unique(pos)) <= {-1, 0, 1} and (pos[:20] == 0).all() and (pos != 0).mean() > 0.3
    t = int(np.flatnonzero((pos[1:] == 1) & (pos[:-1] == 0))[0]) + 1
    assert c[t] > h[t - 20:t].max()                                        # a long entry is a real breakout
    np.testing.assert_array_equal(fw.run(fw.DonchianStrategy, MKT, entry_n=20, exit_n=10), pos)
    trend = fw.quick_eval(pos, MKT["open"].to_numpy())
    assert np.isfinite(trend["sharpe"])


def test_cross_sectional_and_dual_momentum():
    idx = pd.date_range("2020-01-31", periods=15, freq="ME")
    growth = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    closes = pd.DataFrame(np.exp(np.outer(np.arange(15), growth)), index=idx, columns=[f"S{i}" for i in range(10)])
    w = fw.cross_sectional_momentum(closes, top_q=0.2)
    assert (w.iloc[:12].sum(axis=1) == 0).all()
    assert w.iloc[12].to_dict() == {**{f"S{i}": 0.0 for i in range(8)}, "S8": 0.5, "S9": 0.5}
    closes = closes.iloc[:, [0, 3, 9]].set_axis(["BIL", "A", "B"], axis=1)
    pick = fw.dual_momentum(closes, "BIL", lookback=3)
    assert pick.iloc[:3].isna().all() and (pick.iloc[3:] == "B").all()
    falling = closes.assign(A=closes["A"].iloc[::-1].to_numpy(), B=closes["B"].iloc[::-1].to_numpy())
    assert (fw.dual_momentum(falling, "BIL", 3).iloc[3:] == "BIL").all()   # both lose to cash -> hold cash
