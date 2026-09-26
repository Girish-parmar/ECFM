"""Clinic W5 — Deep learning vs LightGBM vs a linear baseline, walk-forward, with seed variance and conformal
intervals (Part 10, weeks 36–37).

Target: tomorrow's |return| on a GARCH market (volatility is forecastable). Same data, same folds, same metric for
every model. The deep model must justify its extra cost — here, it has to beat HAR.
Run:  python dl_vs_gbm.py       Test:  python -m pytest clinic_w5_dl   (needs week36_strategies and week37_deep)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import garch_returns                                                     # noqa: E402

st = load("week36_strategies", "strategies")
dp = load("week37_deep", "deep")
L = 22


def folds(n: int, first_test: int, n_folds: int) -> list[tuple[int, int]]:
    """Expanding walk-forward (given): fold k trains on [0, start_k) and tests on [start_k, start_k + size)."""
    size = (n - first_test) // n_folds
    return [(first_test + k * size, first_test + (k + 1) * size) for k in range(n_folds)]


def har_and_gbm(har: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """st.vol_forecasts trained on har rows [:start], kept for rows [start, end). Columns har, lgbm, target."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def lstm_forecast(absret: np.ndarray, start: int, end: int, seed: int = 0, epochs: int = 25) -> dict:
    """LSTM on RAW windows of |r| (dp.WindowDataset(normalize=False)) scaled by the TRAINING mean of |r|.
    Window i ends at row i+L−1 and predicts |r| of the next row. Training windows: those whose target row is < start;
    the last 15% of them (after a gap of L windows) are the validation set for early stopping. Test windows: target
    rows in [start, end). dp.set_seed(seed); LSTMRegressor(1, hidden=16); dp.train(epochs, patience=4).
    Return {"pred": test predictions (unscaled), "target", "cal_resid": validation residuals (unscaled)}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def compare(ret: pd.Series, true_vol: pd.Series, first_test: int = 2000, n_folds: int = 2, seeds=(0, 1, 2)
            ) -> tuple[pd.DataFrame, dict]:
    """For each fold [start, end) of TARGET days: HAR and LightGBM (har_and_gbm on st.har_features(|ret|): its row at
    position p is day p + 21 and forecasts day p + 22, so use positions start − 22 .. end − 22) and an LSTM per seed. Per model: mse (vs the |r| target, mean over folds),
    corr_true (correlation with the TRUE next-day vol, mean over folds). LSTM rows: "lstm" = mean over seeds,
    plus column seed_std (std over seeds of the fold-mean mse; 0 for the others).
    Also return {"coverage": mean over folds and seeds of the 90% split-conformal coverage of the LSTM}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


if __name__ == "__main__":
    g = garch_returns(3000)
    table, extra = compare(g["ret"], g["vol"])
    print("next-day |return| forecasts, walk-forward (2 folds, 3 LSTM seeds):\n", (table * [1e6, 1, 1e6]).round(3)
          .rename(columns={"mse": "mse×1e6", "seed_std": "seed_std×1e6"}).to_string())
    print(f"\nLSTM 90% conformal coverage: {extra['coverage']:.3f}")
    best = table["mse"].idxmin()
    print(f"lowest error: {best}. Adopt the deep model only if it beats the simple one by more than its seed noise.")
