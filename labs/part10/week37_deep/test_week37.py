import numpy as np
import pytest
import torch
from _loader import load
from common import ar1

dp = load("week37_deep", "deep")
L = 20
X_AR = ar1(3000, 0.6)


def test_window_dataset():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(100.0, 110.0)
    ds = dp.WindowDataset(X, y, lookback=4)
    assert len(ds) == 7
    w, t = ds[2]
    assert w.shape == (4, 2) and w.dtype == torch.float32 and t.item() == 105.0      # target of the LAST row
    np.testing.assert_allclose(w.mean(0), 0, atol=1e-6)                             # normalized with its own stats
    ds1 = dp.WindowDataset(np.arange(10.0), y, 3)                                   # 1-D input is one feature
    assert ds1[0][0].shape == (3, 1)
    raw = dp.WindowDataset(X, y, 4, normalize=False)
    np.testing.assert_allclose(raw[2][0], X[2:6])


def test_time_splits_leave_gaps():
    tr, va, te = dp.time_splits(100, lookback=5, val_frac=0.1, test_frac=0.1)
    np.testing.assert_array_equal(te, np.arange(90, 100))
    np.testing.assert_array_equal(va, np.arange(75, 85))
    np.testing.assert_array_equal(tr, np.arange(0, 70))
    assert va[0] - tr[-1] > 5 and te[0] - va[-1] > 5                                # no shared bars across splits


@pytest.fixture(scope="module")
def ar_setup():
    ds = dp.WindowDataset(X_AR[:-1], X_AR[1:], L)
    return ds, dp.time_splits(len(ds), L)


def test_lstm_learns_ar1_but_not_better_than_the_baseline(ar_setup):
    ds, sp = ar_setup
    dp.set_seed(0)
    model = dp.LSTMRegressor(1, hidden=16)
    tr, va, te = dp.loaders(ds, sp)
    best = dp.train(model, tr, va, epochs=30, patience=4)
    p, y = dp.predict(model, te)
    assert best < 1.1 and len(p) == len(sp[2])
    corr = np.corrcoef(p, y)[0, 1]
    last_value = X_AR[:-1][sp[2] + L - 1]                                           # "tomorrow ∝ today"
    baseline = np.corrcoef(last_value, y)[0, 1]
    assert 0.45 < corr < baseline + 0.02 and baseline == pytest.approx(0.6, abs=0.05)
    pv, yv = dp.predict(model, va)
    lo, hi = dp.conformal_interval(yv - pv, p, alpha=0.1)
    assert 0.85 < dp.coverage(lo, hi, y) < 0.95                                      # ≈ 90% as promised


def test_mlp_and_tcn_shapes_and_causality():
    x = torch.randn(8, L, 3)
    assert dp.MLP(3, L)(x).shape == (8,) and dp.LSTMRegressor(3)(x).shape == (8,)
    tcn = dp.TCNRegressor(3, channels=8, levels=3)
    assert tcn(x).shape == (8,)
    conv = dp.CausalConv1d(3, 4, kernel=3, dilation=2)
    assert conv(x.transpose(1, 2)).shape == (8, 4, L)
    x2 = x.clone()
    x2[:, 12:] = torch.randn(8, L - 12, 3)                                          # change the future only
    with torch.no_grad():
        a, b = tcn.features(x), tcn.features(x2)
    torch.testing.assert_close(a[:, :, :12], b[:, :, :12])                          # the past output is unchanged
    assert not torch.allclose(a[:, :, 12:], b[:, :, 12:])


def test_pinball_loss():
    assert dp.pinball_loss(torch.zeros(2), torch.tensor([1.0, -1.0]), 0.9).item() == pytest.approx(0.5)
    y = torch.tensor(np.random.default_rng(0).normal(size=4000), dtype=torch.float32)
    grid = torch.linspace(-3, 3, 601)
    losses = torch.stack([dp.pinball_loss(g.expand_as(y), y, 0.95) for g in grid])
    assert grid[losses.argmin()].item() == pytest.approx(np.quantile(y.numpy(), 0.95), abs=0.02)


def test_conformal_by_hand():
    lo, hi = dp.conformal_interval(np.arange(1.0, 20.0), np.array([0.0, 10.0]), alpha=0.1)
    np.testing.assert_allclose(hi, [18.0, 28.0])                                     # k = ceil(20·0.9) = 18th smallest
    np.testing.assert_allclose(lo, [-18.0, -8.0])
    assert dp.conformal_interval(-np.arange(1.0, 10.0), np.zeros(1), alpha=0.1)[1][0] == 9.0
    assert dp.coverage(np.zeros(4), np.ones(4), [0.5, 2, -1, 1]) == 0.5


def test_mc_dropout():
    dp.set_seed(0)
    x = torch.randn(5, L, 1)
    mean, std = dp.mc_dropout(dp.MLP(1, L, dropout=0.5), x, n=30)
    assert mean.shape == (5,) and (std > 0).all()
    model = dp.MLP(1, L, dropout=0.0)
    _, std0 = dp.mc_dropout(model, x)
    np.testing.assert_allclose(std0, 0, atol=1e-6)
    assert not model.training                                                        # back in eval mode


def test_autoencoder_flags_anomalies():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(1000, 2))
    X = F @ rng.normal(size=(2, 10)) + 0.1 * rng.normal(size=(1000, 10))            # 10 assets, 2 factors
    model, mu, sd = dp.fit_autoencoder(X[:800])
    new = X[800:].copy()
    odd = [5, 50, 150]
    new[odd] += rng.choice([-3, 3], size=(3, 10))                                   # break the factor structure
    s = dp.anomaly_scores(model, new, mu, sd)
    assert set(np.argsort(s)[-3:]) == set(odd)
