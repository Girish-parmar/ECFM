"""Week 38 (S21–S24) — Reinforcement learning: value iteration, tabular Q-learning (one path vs many paths), a
Gymnasium trading environment with correct timing and costs, the differential Sharpe ratio, a tabular agent trained
on the environment, and optimal execution: TWAP vs Almgren–Chriss in an execution environment.

The action at t is rewarded with what happens AFTER t. Always show the agent works on data with known structure
before trying real data, and always compare it with a simple rule.
Fill in every block marked "Your turn", then run:  python -m pytest week38_rl
"""
from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ------------------------------------------------------------------------------- S21 foundations
def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 0.9, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """P: (A, S, S) transition probabilities, R: (S, A) expected rewards. Iterate
    V ← max_a [R[:, a] + γ P[a] V] until max |ΔV| < tol. Return (V, greedy policy as action indices)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def bucket(x: float, edges: np.ndarray) -> int:
    """State index of a continuous value: np.searchsorted(edges, x) (given)."""
    return int(np.searchsorted(edges, x))


def q_learning_paths(paths: list[np.ndarray], edges: np.ndarray, epochs: int = 5, alpha: float = 0.05,
                     eps: float = 0.1, seed: int = 0) -> np.ndarray:
    """Tabular Q-learning on price paths (a contextual bandit: gamma = 0, no costs). State = bucket(x_t, edges);
    actions 0, 1, 2 = positions −1, 0, +1; reward = position · (x_{t+1} − x_t). ε-greedy (rng = default_rng(seed));
    Q[s, a] += alpha (reward − Q[s, a]). Loop epochs × paths × time steps. Return Q, shape (len(edges)+1, 3)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def greedy_reward(Q: np.ndarray, paths: list[np.ndarray], edges: np.ndarray) -> float:
    """Mean reward per step of the greedy policy (argmax Q) on the given paths."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------------- S22 trading env
class TradingEnv(gym.Env):
    """Lesson plan S22. Single asset; action 0/1/2 = target position −1/0/+1; reward = new position × r[t] (the
    return AFTER the decision) − cost·|position change|. Observation = the last `lookback` returns / scale, then the
    current position (float32). `scale` defaults to the std of the returns GIVEN (for a test environment, pass the
    TRAINING scale: using the test period's std would be a small look-ahead). The episode starts at t = lookback
    (or a random start if episode_len is set: rng from reset(seed)) and ends when the data (or episode_len) runs out."""

    metadata = {"render_modes": []}

    def __init__(self, returns: np.ndarray, lookback: int = 10, cost: float = 0.0005, scale: float | None = None,
                 episode_len: int | None = None):
        super().__init__()
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def _obs(self):
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def step(self, action):
        raise NotImplementedError("✍️ Your turn: see the docstring")


def run_policy(env: TradingEnv, policy: Callable[[np.ndarray], int], seed: int | None = None) -> float:
    """Play one full episode with policy(obs) → action; return the total reward."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def differential_sharpe(returns, eta: float = 0.01) -> np.ndarray:
    """Moody & Saffell (2001): A_t = A + η(R_t − A), B_t = B + η(R_t² − B) (start A = B = 0);
    D_t = (B_{t−1}ΔA_t − ½A_{t−1}ΔB_t) / (B_{t−1} − A_{t−1}²)^{3/2}, with ΔA_t = R_t − A_{t−1}, ΔB_t = R_t² − B_{t−1};
    D_t = 0 while B_{t−1} − A_{t−1}² <= 0. A reward that is the marginal contribution to a running Sharpe ratio."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def sign_state(obs: np.ndarray) -> int:
    """Tabular state from a TradingEnv observation (given): (last return > 0) × 3 + current position + 1 → 0..5."""
    return int(obs[-2] > 0) * 3 + int(round(obs[-1])) + 1


def q_learning_env(env: TradingEnv, n_states: int, state_fn: Callable[[np.ndarray], int] = sign_state,
                   episodes: int = 200, alpha: float = 0.05, gamma: float = 0.9, eps: float = 0.1, seed: int = 0
                   ) -> np.ndarray:
    """Q-learning on the environment: each episode env.reset(seed=seed + episode); ε-greedy (rng default_rng(seed));
    target = r + γ max Q[s'] (just r when terminated). Return Q (n_states × 3)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------------- S23 execution
def twap_schedule(X: float, N: int) -> np.ndarray:
    """Sell X in N equal slices: the N trade sizes."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def almgren_chriss_schedule(X: float, N: int, sigma: float, eta: float, gamma: float, lam: float) -> np.ndarray:
    """Almgren–Chriss (2000), τ = 1: η̃ = η − γ/2; κ from cosh κ = 1 + λσ²/(2η̃); holdings
    x_j = X sinh(κ(N − j))/sinh(κN), j = 0..N; trades n_j = x_{j−1} − x_j (j = 1..N). λ → 0 gives TWAP."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def cost_moments(trades: np.ndarray, sigma: float, eta: float, gamma: float) -> tuple[float, float]:
    """Expected implementation shortfall and its variance for a sell schedule (τ = 1, linear impact, no fixed cost):
    E = ½γX² + η̃ Σ n_k², Var = σ² Σ x_k² over the holdings AFTER each trade except the last (x_1..x_{N−1})."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


class ExecutionEnv(gym.Env):
    """Sell X shares in N steps. Observation = [inventory / X, steps left / N]. Action (Box [0, 1]) = fraction of
    the REMAINING inventory to sell now (everything on the last step). Price: S_k = S_{k−1} + σ ξ_k − γ n_k; the
    fill price of n_k is S_{k−1} − η n_k. Reward = n_k (fill − S_0): minus the shortfall of this slice (so the
    episode's total reward = −implementation shortfall)."""

    metadata = {"render_modes": []}

    def __init__(self, X: float = 1000, N: int = 20, sigma: float = 0.3, eta: float = 0.01, gamma: float = 0.001,
                 s0: float = 100.0):
        super().__init__()
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def _obs(self):
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raise NotImplementedError("✍️ Your turn: see the docstring")

    def step(self, action):
        raise NotImplementedError("✍️ Your turn: see the docstring")


def schedule_policy(trades: np.ndarray) -> Callable[[np.ndarray, int], float]:
    """Turn a trade schedule into an ExecutionEnv policy (given): at step k sell trades[k] / remaining."""
    remaining = np.r_[trades[::-1].cumsum()[::-1]]

    def policy(obs, k):
        return float(trades[k] / remaining[k]) if remaining[k] > 0 else 1.0
    return policy


def simulate_execution(env: ExecutionEnv, policy: Callable[[np.ndarray, int], float], n_paths: int = 500,
                       seed: int = 0) -> np.ndarray:
    """Implementation shortfall (= −total reward) of `policy(obs, k)` on n_paths episodes, reset(seed=seed + i)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
