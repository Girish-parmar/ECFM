import numpy as np
import pytest
from _loader import load
from common import quick_eval_pnl, regime_market, tsmom_signal

op = load("week27_optimization", "optimization")
BARS = regime_market(n_blocks=20, seed=11)


def test_grid_and_random_search():
    f = lambda a, b: -(a - 3) ** 2 - (b - 1) ** 2                       # noqa: E731
    g = op.grid_search(f, {"a": [1, 2, 3, 4], "b": [0, 1, 2]})
    assert list(g.columns) == ["a", "b", "score"] and len(g) == 12
    assert g.iloc[g["score"].idxmax()][["a", "b"]].tolist() == [3, 1]
    assert g.iloc[0][["a", "b"]].tolist() == [1, 0] and g.iloc[1][["a", "b"]].tolist() == [1, 1]
    r = op.random_search(f, {"a": [1, 2, 3, 4], "b": [0, 1, 2]}, 20, seed=1)
    assert len(r) == 20 and set(r["a"]) <= {1, 2, 3, 4}
    assert r.equals(op.random_search(f, {"a": [1, 2, 3, 4], "b": [0, 1, 2]}, 20, seed=1))


def test_plateau_prefers_broad_regions():
    xs, ys = list(range(5)), list(range(5))
    rows = []
    for x in xs:
        for y in ys:
            broad = 1.0 if (x <= 1 and y <= 1) else 0.0
            spike = 2.0 if (x, y) == (4, 4) else 0.0
            rows.append({"x": x, "y": y, "score": broad + spike})
    import pandas as pd
    res = pd.DataFrame(rows)
    assert res.loc[res["score"].idxmax(), ["x", "y"]].tolist() == [4, 4]    # the raw peak is an isolated spike
    sm = op.plateau_scores(res, "x", "y", radius=1)
    best = np.unravel_index(np.argmax(sm.to_numpy()), sm.shape)
    assert (sm.index[best[0]], sm.columns[best[1]]) == (0, 0) and sm.loc[4, 4] == pytest.approx(0.5)


def test_walk_forward_splits():
    s = list(op.walk_forward(100, 40, 20))
    assert [(tr[0], tr[-1], te[0], te[-1]) for tr, te in s] == [(0, 39, 40, 59), (20, 59, 60, 79), (40, 79, 80, 99)]
    a = list(op.walk_forward(100, 40, 20, anchored=True))
    assert [tr[0] for tr, _ in a] == [0, 0, 0] and len(a[-1][0]) == 80
    assert list(op.walk_forward(50, 40, 20)) == []


def test_walk_forward_optimization_is_out_of_sample():
    close, open_ = BARS["close"].to_numpy(), BARS["open"].to_numpy()

    def pnl(lookback, vol_n):
        return quick_eval_pnl(tsmom_signal(close, lookback, vol_n), open_)
    res = op.walk_forward_optimize(pnl, {"lookback": [20, 60, 120, 250], "vol_n": [20, 60]}, len(close), 500, 250)
    n_win = len(list(op.walk_forward(len(close), 500, 250)))
    assert len(res["params"]) == len(res["oos_sharpe"]) == n_win and res["oos"].size == n_win * 250
    tr, te = next(op.walk_forward(len(close), 500, 250))
    np.testing.assert_allclose(res["oos"][:250], pnl(**res["params"][0])[te])            # OOS = the chosen params on the test slice
    assert np.mean(res["is_sharpe"]) > np.mean(res["oos_sharpe"])         # in-sample always flatters
    assert res["wfe"] == pytest.approx(np.mean(res["oos_sharpe"]) / np.mean(res["is_sharpe"]))


def test_purged_kfold():
    folds = list(op.purged_kfold(100, k=5, label_horizon=5, embargo=0.03))
    assert len(folds) == 5
    train, test = folds[2]                                                # test = 40..59
    assert test[0] == 40 and test[-1] == 59
    assert 34 in train and 35 not in train                                # 35's label reaches 40
    assert 62 not in train and 63 in train                                # embargo of 3 bars after 59
    for train, test in folds:
        assert not set(train) & set(test)


def test_cpcv():
    splits = op.cpcv_splits(120, n_groups=6, k_test=2, label_horizon=3, embargo=0.0)
    assert len(splits) == 15 and splits[0][1] == (0, 1)
    train, groups, test = splits[4]                                       # groups (0, 5)
    assert groups == (0, 5) and len(test) == 40 and 20 in train and 97 not in train and 96 in train
    assert op.cpcv_n_paths(6, 2) == 5 and op.cpcv_n_paths(10, 2) == 9
    counts = np.zeros(6)
    for _, g, _ in splits:
        counts[list(g)] += 1
    assert (counts == 5).all()                                            # every group tested C(5,1) = 5 times
