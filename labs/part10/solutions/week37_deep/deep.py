"""Week 37 (S17–S20) — Deep learning for financial time series in PyTorch: leakage-free window datasets, splits
with gaps, a training loop with early stopping, MLP / LSTM / causal TCN, pinball loss, split-conformal intervals,
MC dropout and an autoencoder anomaly score.

Every model is compared with a simple baseline on the SAME windows. Small models, CPU only.
Fill in every block marked "Your turn", then run:  python -m pytest week37_deep
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


def set_seed(seed: int = 0) -> None:
    """Reproducibility (given): Python, NumPy and PyTorch seeds, one CPU thread."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


# -------------------------------------------------------------------------- S17 data & training
class WindowDataset(Dataset):
    """Lesson plan S17: item i = (window X[i .. i+L−1] normalized with ITS OWN column means and stds (+1e-8),
    target y[i+L−1]); y must already be the FUTURE target aligned to the window's last row. float32 tensors;
    window shape (L, n_features). normalize=False returns the raw window (scale X with TRAINING statistics
    yourself): window normalization erases the level, which is the signal when forecasting volatility."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int, normalize: bool = True):
        # >>> SOLUTION
        X = np.asarray(X, dtype=np.float32)
        self.X = X.reshape(len(X), -1)
        self.y, self.L, self.normalize = np.asarray(y, dtype=np.float32), lookback, normalize
        # <<< SOLUTION

    def __len__(self):
        # >>> SOLUTION
        return len(self.X) - self.L + 1
        # <<< SOLUTION

    def __getitem__(self, i):
        # >>> SOLUTION
        w = self.X[i: i + self.L]
        if self.normalize:
            w = (w - w.mean(0)) / (w.std(0) + 1e-8)
        return torch.from_numpy(w), torch.tensor(self.y[i + self.L - 1])
        # <<< SOLUTION


def time_splits(n_windows: int, lookback: int, val_frac: float = 0.15, test_frac: float = 0.15
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-ordered window positions: test = the last round(test_frac·n) windows, validation = the round(val_frac·n)
    before a GAP of `lookback` windows, train = everything before another gap of `lookback`. The gaps make sure no
    bar (and no target) is shared across splits."""
    # >>> SOLUTION
    n_te, n_va = round(test_frac * n_windows), round(val_frac * n_windows)
    te = np.arange(n_windows - n_te, n_windows)
    va = np.arange(te[0] - lookback - n_va, te[0] - lookback)
    tr = np.arange(0, va[0] - lookback)
    return tr, va, te
    # <<< SOLUTION


def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, epochs: int = 50, lr: float = 1e-3,
          patience: int = 5, loss_fn: nn.Module | None = None) -> float:
    """Lesson plan S17: AdamW(lr, weight_decay=1e-4); per epoch: train with gradient clipping (norm 1.0), then the
    mean validation loss; keep the best state (improvement > 1e-6), stop after `patience` epochs without one;
    restore the best state; return the best validation loss. loss_fn defaults to MSE."""
    # >>> SOLUTION
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = loss_fn or nn.MSELoss()
    best, bad, best_state = np.inf, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(model(xb), yb).item() for xb, yb in val_dl]))
        if val < best - 1e-6:
            best, bad, best_state = val, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best
    # <<< SOLUTION


def predict(model: nn.Module, dl: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, targets) over a loader, in eval mode without gradients."""
    # >>> SOLUTION
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            ps.append(model(xb).numpy())
            ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)
    # <<< SOLUTION


def loaders(ds: Dataset, splits, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train loader shuffled (windows are self-contained, so shuffling INSIDE the training period is fine), the
    validation and test loaders in time order (given)."""
    tr, va, te = splits
    return (DataLoader(Subset(ds, tr), batch_size=batch_size, shuffle=True),
            DataLoader(Subset(ds, va), batch_size=256), DataLoader(Subset(ds, te), batch_size=256))


# ----------------------------------------------------------------------------- S17–S18 models
class MLP(nn.Module):
    """Flatten the (L, F) window → Linear(L·F, hidden) → ReLU → Dropout → Linear(hidden, 1); output shape (batch,)."""

    def __init__(self, n_features: int, lookback: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        # >>> SOLUTION
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_features * lookback, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))
        # <<< SOLUTION

    def forward(self, x):
        # >>> SOLUTION
        return self.net(x).squeeze(-1)
        # <<< SOLUTION


class LSTMRegressor(nn.Module):
    """Lesson plan S18: nn.LSTM(batch_first=True) → Dropout → Linear on the LAST time step's output; shape (batch,)."""

    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        # >>> SOLUTION
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        # <<< SOLUTION

    def forward(self, x):
        # >>> SOLUTION
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)
        # <<< SOLUTION


class CausalConv1d(nn.Module):
    """Conv1d over time (input (batch, channels, time)) padded on the LEFT only by (kernel − 1)·dilation, so the output
    at t sees inputs <= t only (no future). Output length = input length."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        # >>> SOLUTION
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        # <<< SOLUTION

    def forward(self, x):
        # >>> SOLUTION
        return self.conv(nn.functional.pad(x, (self.pad, 0)))
        # <<< SOLUTION


class TCNRegressor(nn.Module):
    """Stack of `levels` CausalConv1d (dilations 1, 2, 4, …) with ReLU, then Linear on the last time step. Input
    (batch, time, features) like the LSTM; transpose to (batch, features, time) inside. Output (batch,)."""

    def __init__(self, n_features: int, channels: int = 16, levels: int = 3, kernel: int = 3):
        super().__init__()
        # >>> SOLUTION
        layers, c = [], n_features
        for k in range(levels):
            layers += [CausalConv1d(c, channels, kernel, 2 ** k), nn.ReLU()]
            c = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)
        # <<< SOLUTION

    def features(self, x):
        """The conv stack's output, (batch, channels, time) — used by the causality test."""
        # >>> SOLUTION
        return self.tcn(x.transpose(1, 2))
        # <<< SOLUTION

    def forward(self, x):
        # >>> SOLUTION
        return self.head(self.features(x)[:, :, -1]).squeeze(-1)
        # <<< SOLUTION


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile loss: mean of max(q·(y − pred), (q − 1)·(y − pred)). Minimized by the q-quantile."""
    # >>> SOLUTION
    e = y - pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))
    # <<< SOLUTION


# ------------------------------------------------------------------------------- S18 uncertainty
def conformal_interval(cal_residuals, pred, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Split conformal (lesson plan S18): q = the k-th smallest |calibration residual| with k = ceil((n+1)(1−alpha))
    (capped at n) — np.quantile at k/n with method="inverted_cdf" gives exactly that; interval pred ± q."""
    # >>> SOLUTION
    r = np.abs(np.asarray(cal_residuals, dtype=float))
    n = len(r)
    q = np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="inverted_cdf")
    pred = np.asarray(pred, dtype=float)
    return pred - q, pred + q
    # <<< SOLUTION


def coverage(lo, hi, y) -> float:
    """Share of y inside [lo, hi]."""
    # >>> SOLUTION
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))
    # <<< SOLUTION


def mc_dropout(model: nn.Module, x: torch.Tensor, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo dropout: keep dropout ON (model.train()), n stochastic forward passes without gradients;
    return (mean, std) per sample; put the model back in eval mode."""
    # >>> SOLUTION
    model.train()
    with torch.no_grad():
        draws = torch.stack([model(x) for _ in range(n)]).numpy()
    model.eval()
    return draws.mean(0), draws.std(0)
    # <<< SOLUTION


# ------------------------------------------------------------------------------- S19 autoencoder
class AutoEncoder(nn.Module):
    """n_inputs → Linear → ReLU → Linear(bottleneck) → Linear → ReLU → Linear(n_inputs)."""

    def __init__(self, n_inputs: int, hidden: int = 16, bottleneck: int = 3):
        super().__init__()
        # >>> SOLUTION
        self.enc = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, n_inputs))
        # <<< SOLUTION

    def forward(self, x):
        # >>> SOLUTION
        return self.dec(self.enc(x))
        # <<< SOLUTION


def fit_autoencoder(X: np.ndarray, epochs: int = 200, lr: float = 1e-2, seed: int = 0, bottleneck: int = 3
                    ) -> tuple[AutoEncoder, np.ndarray, np.ndarray]:
    """Standardize X with ITS column means/stds, full-batch Adam on the MSE reconstruction loss. Return (model,
    mean, std) — score new data with the SAME mean and std."""
    # >>> SOLUTION
    set_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32)
    model = AutoEncoder(X.shape[1], bottleneck=bottleneck)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Z), Z)
        loss.backward()
        opt.step()
    return model, mu, sd
    # <<< SOLUTION


def anomaly_scores(model: AutoEncoder, X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per row: mean squared reconstruction error of the standardized row."""
    # >>> SOLUTION
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return ((model(Z) - Z) ** 2).mean(1).numpy()
    # <<< SOLUTION
