"""Clinic W6 — RL agent defense: robustness across seeds and regimes, comparison with simple rules, and sensitivity
to the cost assumption (Part 10, week 38).

Regimes are AR(1) return markets with φ = +0.3 (momentum), 0 (noise) and −0.3 (reversal). A credible agent learns
momentum, reversal and "stay out" on its own, across every seed, and never beats the best simple rule by luck.
Run:  python rl_defense.py [--ppo]     Test:  python -m pytest clinic_w6_rl    (needs week38_rl)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _loader import load                                                             # noqa: E402
from common import ar1                                                               # noqa: E402

rl = load("week38_rl", "rl")
COST = 0.0002
RULES = {"long_only": lambda o: 2, "flat": lambda o: 1, "momentum": lambda o: 2 if o[-2] > 0 else 0,
         "reversal": lambda o: 0 if o[-2] > 0 else 2}


def regime_returns(phi: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """4000 AR(1) daily returns (σ = 1%), train = first 3000, test = last 1000 (given)."""
    r = ar1(4000, phi, 0.01, seed=seed)
    return r[:3000], r[3000:]


def train_agent(train: np.ndarray, seed: int, cost: float = COST, episodes: int = 300) -> np.ndarray:
    """rl.q_learning_env on rl.TradingEnv(train, cost=cost, episode_len=250) with 6 sign states; return Q."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def evaluate(policy, test: np.ndarray, train: np.ndarray, cost: float = COST) -> float:
    """Total reward of policy(obs) on the whole TEST series, observations scaled with the TRAINING std."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def greedy(Q: np.ndarray):
    """Greedy policy of a sign-state Q table (given)."""
    return lambda o: int(Q[rl.sign_state(o)].argmax())


def robustness_table(phis=(0.3, 0.0, -0.3), seeds=range(5), cost: float = COST) -> pd.DataFrame:
    """Per regime φ (one data set per regime, seed 0; the agent's seed varies): agent_mean, agent_std, agent_min
    over the seeds, the test reward of every rule in RULES, and best_rule (the name of the best rule)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def cost_sensitivity(phi: float = 0.3, train_cost: float = 0.0, test_cost: float = 0.003, seeds=range(5)) -> dict:
    """Train with an optimistic cost assumption and with the real one; evaluate BOTH at the real cost (test_cost).
    Return {"optimistic": mean test reward, "realistic": mean test reward, "turnover_optimistic",
    "turnover_realistic": mean number of position changes on the test series}."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def ppo_agent(train: np.ndarray, test: np.ndarray, steps: int = 30_000, seed: int = 0) -> float:
    """Optional (Stable-Baselines3, given): PPO on the same environment; test reward. Slow on CPU for many seeds."""
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", rl.TradingEnv(train, cost=COST, episode_len=250), seed=seed, verbose=0).learn(steps)
    return evaluate(lambda o: int(model.predict(o, deterministic=True)[0]), test, train)


if __name__ == "__main__":
    table = robustness_table()
    print("test reward (sum of daily P&L), 5 agent seeds per regime:\n", table.to_string())
    print("\ncost assumption (φ = 0.3, real cost 30 bp):", {k: round(v, 3) for k, v in cost_sensitivity().items()})
    if "--ppo" in sys.argv:
        tr, te = regime_returns(0.3)
        print("\nPPO (30k steps) on φ = 0.3:", round(ppo_agent(tr, te), 3))
