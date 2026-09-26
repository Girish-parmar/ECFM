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
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def __len__(self):
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def __getitem__(self, i):
        raise NotImplementedError("✍️ Your turn: see the docstring")


def time_splits(n_windows: int, lookback: int, val_frac: float = 0.15, test_frac: float = 0.15
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-ordered window positions: test = the last round(test_frac·n) windows, validation = the round(val_frac·n)
    before a GAP of `lookback` windows, train = everything before another gap of `lookback`. The gaps make sure no
    bar (and no target) is shared across splits."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, epochs: int = 50, lr: float = 1e-3,
          patience: int = 5, loss_fn: nn.Module | None = None) -> float:
    """Lesson plan S17: AdamW(lr, weight_decay=1e-4); per epoch: train with gradient clipping (norm 1.0), then the
    mean validation loss; keep the best state (improvement > 1e-6), stop after `patience` epochs without one;
    restore the best state; return the best validation loss. loss_fn defaults to MSE."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def predict(model: nn.Module, dl: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, targets) over a loader, in eval mode without gradients."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


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
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def forward(self, x):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class LSTMRegressor(nn.Module):
    """Lesson plan S18: nn.LSTM(batch_first=True) → Dropout → Linear on the LAST time step's output; shape (batch,)."""

    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def forward(self, x):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class CausalConv1d(nn.Module):
    """Conv1d over time (input (batch, channels, time)) padded on the LEFT only by (kernel − 1)·dilation, so the output
    at t sees inputs <= t only (no future). Output length = input length."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def forward(self, x):
        raise NotImplementedError("✍️ Your turn: see the docstring")


class TCNRegressor(nn.Module):
    """Stack of `levels` CausalConv1d (dilations 1, 2, 4, …) with ReLU, then Linear on the last time step. Input
    (batch, time, features) like the LSTM; transpose to (batch, features, time) inside. Output (batch,)."""

    def __init__(self, n_features: int, channels: int = 16, levels: int = 3, kernel: int = 3):
        super().__init__()
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def features(self, x):
        """The conv stack's output, (batch, channels, time) — used by the causality test."""
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def forward(self, x):
        raise NotImplementedError("✍️ Your turn: see the docstring")


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile loss: mean of max(q·(y − pred), (q − 1)·(y − pred)). Minimized by the q-quantile."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------------- S18 uncertainty
def conformal_interval(cal_residuals, pred, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Split conformal (lesson plan S18): q = the k-th smallest |calibration residual| with k = ceil((n+1)(1−alpha))
    (capped at n) — np.quantile at k/n with method="inverted_cdf" gives exactly that; interval pred ± q."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def coverage(lo, hi, y) -> float:
    """Share of y inside [lo, hi]."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def mc_dropout(model: nn.Module, x: torch.Tensor, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo dropout: keep dropout ON (model.train()), n stochastic forward passes without gradients;
    return (mean, std) per sample; put the model back in eval mode."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------------- S19 autoencoder
class AutoEncoder(nn.Module):
    """n_inputs → Linear → ReLU → Linear(bottleneck) → Linear → ReLU → Linear(n_inputs)."""

    def __init__(self, n_inputs: int, hidden: int = 16, bottleneck: int = 3):
        super().__init__()
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def forward(self, x):
        raise NotImplementedError("✍️ Your turn: see the docstring")


def fit_autoencoder(X: np.ndarray, epochs: int = 200, lr: float = 1e-2, seed: int = 0, bottleneck: int = 3
                    ) -> tuple[AutoEncoder, np.ndarray, np.ndarray]:
    """Standardize X with ITS column means/stds, full-batch Adam on the MSE reconstruction loss. Return (model,
    mean, std) — score new data with the SAME mean and std."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def anomaly_scores(model: AutoEncoder, X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per row: mean squared reconstruction error of the standardized row."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
