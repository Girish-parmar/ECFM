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
    # >>> SOLUTION
    return rl.q_learning_env(rl.TradingEnv(train, cost=cost, episode_len=250), 6, episodes=episodes, seed=seed)
    # <<< SOLUTION


def evaluate(policy, test: np.ndarray, train: np.ndarray, cost: float = COST) -> float:
    """Total reward of policy(obs) on the whole TEST series, observations scaled with the TRAINING std."""
    # >>> SOLUTION
    return rl.run_policy(rl.TradingEnv(test, cost=cost, scale=float(np.std(train))), policy)
    # <<< SOLUTION


def greedy(Q: np.ndarray):
    """Greedy policy of a sign-state Q table (given)."""
    return lambda o: int(Q[rl.sign_state(o)].argmax())


def robustness_table(phis=(0.3, 0.0, -0.3), seeds=range(5), cost: float = COST) -> pd.DataFrame:
    """Per regime φ (one data set per regime, seed 0; the agent's seed varies): agent_mean, agent_std, agent_min
    over the seeds, the test reward of every rule in RULES, and best_rule (the name of the best rule)."""
    # >>> SOLUTION
    rows = {}
    for phi in phis:
        tr, te = regime_returns(phi)
        agent = [evaluate(greedy(train_agent(tr, s, cost)), te, tr, cost) for s in seeds]
        rules = {k: evaluate(p, te, tr, cost) for k, p in RULES.items()}
        rows[phi] = {"agent_mean": float(np.mean(agent)), "agent_std": float(np.std(agent, ddof=1)),
                     "agent_min": float(np.min(agent)), **rules, "best_rule": max(rules, key=rules.get)}
    return pd.DataFrame(rows).T
    # <<< SOLUTION


def cost_sensitivity(phi: float = 0.3, train_cost: float = 0.0, test_cost: float = 0.003, seeds=range(5)) -> dict:
    """Train with an optimistic cost assumption and with the real one; evaluate BOTH at the real cost (test_cost).
    Return {"optimistic": mean test reward, "realistic": mean test reward, "turnover_optimistic",
    "turnover_realistic": mean number of position changes on the test series}."""
    # >>> SOLUTION
    tr, te = regime_returns(phi)
    out = {}
    for name, c in (("optimistic", train_cost), ("realistic", test_cost)):
        rewards, turns = [], []
        for s in seeds:
            pol = greedy(train_agent(tr, s, c))
            rewards.append(evaluate(pol, te, tr, test_cost))
            env = rl.TradingEnv(te, cost=test_cost, scale=float(np.std(tr)))
            obs, _ = env.reset()
            done, pos = False, [0.0]
            while not done:
                obs, _, done, _, _ = env.step(pol(obs))
                pos.append(env.pos)
            turns.append(int(np.count_nonzero(np.diff(pos))))
        out[name], out[f"turnover_{name}"] = float(np.mean(rewards)), float(np.mean(turns))
    return out
    # <<< SOLUTION


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
