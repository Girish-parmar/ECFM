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
    # >>> SOLUTION
    V = np.zeros(P.shape[1])
    while True:
        Q = R + gamma * np.einsum("ast,t->sa", P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new
    # <<< SOLUTION


def bucket(x: float, edges: np.ndarray) -> int:
    """State index of a continuous value: np.searchsorted(edges, x) (given)."""
    return int(np.searchsorted(edges, x))


def q_learning_paths(paths: list[np.ndarray], edges: np.ndarray, epochs: int = 5, alpha: float = 0.05,
                     eps: float = 0.1, seed: int = 0) -> np.ndarray:
    """Tabular Q-learning on price paths (a contextual bandit: gamma = 0, no costs). State = bucket(x_t, edges);
    actions 0, 1, 2 = positions −1, 0, +1; reward = position · (x_{t+1} − x_t). ε-greedy (rng = default_rng(seed));
    Q[s, a] += alpha (reward − Q[s, a]). Loop epochs × paths × time steps. Return Q, shape (len(edges)+1, 3)."""
    # >>> SOLUTION
    rng = np.random.default_rng(seed)
    Q = np.zeros((len(edges) + 1, 3))
    for _ in range(epochs):
        for x in paths:
            for t in range(len(x) - 1):
                s = bucket(x[t], edges)
                a = int(rng.integers(3)) if rng.random() < eps else int(Q[s].argmax())
                reward = (a - 1) * (x[t + 1] - x[t])
                Q[s, a] += alpha * (reward - Q[s, a])
    return Q
    # <<< SOLUTION


def greedy_reward(Q: np.ndarray, paths: list[np.ndarray], edges: np.ndarray) -> float:
    """Mean reward per step of the greedy policy (argmax Q) on the given paths."""
    # >>> SOLUTION
    total, n = 0.0, 0
    for x in paths:
        for t in range(len(x) - 1):
            total += (Q[bucket(x[t], edges)].argmax() - 1) * (x[t + 1] - x[t])
            n += 1
    return total / n
    # <<< SOLUTION


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
        # >>> SOLUTION
        self.r = np.asarray(returns, dtype=np.float32)
        self.L, self.cost, self.episode_len = lookback, cost, episode_len
        self.scale = float(np.std(self.r)) if scale is None else float(scale)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(lookback + 1,), dtype=np.float32)
        # <<< SOLUTION

    def _obs(self):
        # >>> SOLUTION
        return np.append(self.r[self.t - self.L: self.t] / (self.scale + 1e-8), self.pos).astype(np.float32)
        # <<< SOLUTION

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # >>> SOLUTION
        if self.episode_len is None:
            self.t = self.L
            self.end = len(self.r)
        else:
            self.t = int(self.np_random.integers(self.L, len(self.r) - self.episode_len + 1))
            self.end = self.t + self.episode_len
        self.pos = 0.0
        return self._obs(), {}
        # <<< SOLUTION

    def step(self, action):
        # >>> SOLUTION
        new_pos = float(action) - 1.0
        reward = new_pos * float(self.r[self.t]) - self.cost * abs(new_pos - self.pos)
        self.pos, self.t = new_pos, self.t + 1
        terminated = self.t >= self.end
        obs = self._obs() if not terminated else np.zeros(self.L + 1, dtype=np.float32)
        return obs, float(reward), terminated, False, {}
        # <<< SOLUTION


def run_policy(env: TradingEnv, policy: Callable[[np.ndarray], int], seed: int | None = None) -> float:
    """Play one full episode with policy(obs) → action; return the total reward."""
    # >>> SOLUTION
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, r, done, trunc, _ = env.step(policy(obs))
        done = done or trunc
        total += r
    return total
    # <<< SOLUTION


def differential_sharpe(returns, eta: float = 0.01) -> np.ndarray:
    """Moody & Saffell (2001): A_t = A + η(R_t − A), B_t = B + η(R_t² − B) (start A = B = 0);
    D_t = (B_{t−1}ΔA_t − ½A_{t−1}ΔB_t) / (B_{t−1} − A_{t−1}²)^{3/2}, with ΔA_t = R_t − A_{t−1}, ΔB_t = R_t² − B_{t−1};
    D_t = 0 while B_{t−1} − A_{t−1}² <= 0. A reward that is the marginal contribution to a running Sharpe ratio."""
    # >>> SOLUTION
    A = B = 0.0
    out = []
    for R in np.asarray(returns, dtype=float):
        dA, dB = R - A, R * R - B
        den = B - A * A
        out.append((B * dA - 0.5 * A * dB) / den ** 1.5 if den > 0 else 0.0)
        A, B = A + eta * dA, B + eta * dB
    return np.array(out)
    # <<< SOLUTION


def sign_state(obs: np.ndarray) -> int:
    """Tabular state from a TradingEnv observation (given): (last return > 0) × 3 + current position + 1 → 0..5."""
    return int(obs[-2] > 0) * 3 + int(round(obs[-1])) + 1


def q_learning_env(env: TradingEnv, n_states: int, state_fn: Callable[[np.ndarray], int] = sign_state,
                   episodes: int = 200, alpha: float = 0.05, gamma: float = 0.9, eps: float = 0.1, seed: int = 0
                   ) -> np.ndarray:
    """Q-learning on the environment: each episode env.reset(seed=seed + episode); ε-greedy (rng default_rng(seed));
    target = r + γ max Q[s'] (just r when terminated). Return Q (n_states × 3)."""
    # >>> SOLUTION
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, 3))
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        s, done = state_fn(obs), False
        while not done:
            a = int(rng.integers(3)) if rng.random() < eps else int(Q[s].argmax())
            obs, r, done, _, _ = env.step(a)
            s2 = state_fn(obs)
            target = r if done else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
    return Q
    # <<< SOLUTION


# --------------------------------------------------------------------------- S23 execution
def twap_schedule(X: float, N: int) -> np.ndarray:
    """Sell X in N equal slices: the N trade sizes."""
    # >>> SOLUTION
    return np.full(N, X / N)
    # <<< SOLUTION


def almgren_chriss_schedule(X: float, N: int, sigma: float, eta: float, gamma: float, lam: float) -> np.ndarray:
    """Almgren–Chriss (2000), τ = 1: η̃ = η − γ/2; κ from cosh κ = 1 + λσ²/(2η̃); holdings
    x_j = X sinh(κ(N − j))/sinh(κN), j = 0..N; trades n_j = x_{j−1} − x_j (j = 1..N). λ → 0 gives TWAP."""
    # >>> SOLUTION
    eta_t = eta - gamma / 2
    kappa = np.arccosh(1 + lam * sigma ** 2 / (2 * eta_t))
    if kappa < 1e-12:
        return twap_schedule(X, N)
    j = np.arange(N + 1)
    x = X * np.sinh(kappa * (N - j)) / np.sinh(kappa * N)
    return -np.diff(x)
    # <<< SOLUTION


def cost_moments(trades: np.ndarray, sigma: float, eta: float, gamma: float) -> tuple[float, float]:
    """Expected implementation shortfall and its variance for a sell schedule (τ = 1, linear impact, no fixed cost):
    E = ½γX² + η̃ Σ n_k², Var = σ² Σ x_k² over the holdings AFTER each trade except the last (x_1..x_{N−1})."""
    # >>> SOLUTION
    n = np.asarray(trades, dtype=float)
    X = n.sum()
    x = X - np.cumsum(n)
    return float(0.5 * gamma * X ** 2 + (eta - gamma / 2) * np.sum(n ** 2)), float(sigma ** 2 * np.sum(x[:-1] ** 2))
    # <<< SOLUTION


class ExecutionEnv(gym.Env):
    """Sell X shares in N steps. Observation = [inventory / X, steps left / N]. Action (Box [0, 1]) = fraction of
    the REMAINING inventory to sell now (everything on the last step). Price: S_k = S_{k−1} + σ ξ_k − γ n_k; the
    fill price of n_k is S_{k−1} − η n_k. Reward = n_k (fill − S_0): minus the shortfall of this slice (so the
    episode's total reward = −implementation shortfall)."""

    metadata = {"render_modes": []}

    def __init__(self, X: float = 1000, N: int = 20, sigma: float = 0.3, eta: float = 0.01, gamma: float = 0.001,
                 s0: float = 100.0):
        super().__init__()
        # >>> SOLUTION
        self.X, self.N, self.sigma, self.eta, self.gamma, self.s0 = X, N, sigma, eta, gamma, s0
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        # <<< SOLUTION

    def _obs(self):
        # >>> SOLUTION
        return np.array([self.inv / self.X, (self.N - self.k) / self.N], dtype=np.float32)
        # <<< SOLUTION

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # >>> SOLUTION
        self.inv, self.k, self.S = float(self.X), 0, self.s0
        return self._obs(), {}
        # <<< SOLUTION

    def step(self, action):
        # >>> SOLUTION
        frac = float(np.clip(np.asarray(action).ravel()[0], 0.0, 1.0))
        n = self.inv if self.k == self.N - 1 else frac * self.inv
        fill = self.S - self.eta * n
        reward = n * (fill - self.s0)
        self.S += self.sigma * self.np_random.standard_normal() - self.gamma * n
        self.inv -= n
        self.k += 1
        return self._obs(), float(reward), self.k >= self.N, False, {"n": n}
        # <<< SOLUTION


def schedule_policy(trades: np.ndarray) -> Callable[[np.ndarray, int], float]:
    """Turn a trade schedule into an ExecutionEnv policy (given): at step k sell trades[k] / remaining."""
    remaining = np.r_[trades[::-1].cumsum()[::-1]]

    def policy(obs, k):
        return float(trades[k] / remaining[k]) if remaining[k] > 0 else 1.0
    return policy


def simulate_execution(env: ExecutionEnv, policy: Callable[[np.ndarray, int], float], n_paths: int = 500,
                       seed: int = 0) -> np.ndarray:
    """Implementation shortfall (= −total reward) of `policy(obs, k)` on n_paths episodes, reset(seed=seed + i)."""
    # >>> SOLUTION
    out = []
    for i in range(n_paths):
        obs, _ = env.reset(seed=seed + i)
        total, done, k = 0.0, False, 0
        while not done:
            obs, r, done, _, _ = env.step(np.array([policy(obs, k)], dtype=np.float32))
            total += r
            k += 1
        out.append(-total)
    return np.array(out)
    # <<< SOLUTION
