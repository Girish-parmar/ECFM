import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from _loader import load
from common import ar1

rl = load("week38_rl", "rl")
EDGES = np.linspace(-4, 4, 17)
R = ar1(4000, 0.3, 0.01, seed=0)                                                   # momentum at lag 1
TRAIN, TEST = R[:3000], R[3000:]


def ou_path(n, seed, theta=0.2):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] * (1 - theta) + rng.normal()
    return x


def check(env):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                            # unbounded Box: expected
        check_env(env, skip_render_check=True)


def test_value_iteration_by_hand():
    P = np.array([np.eye(2), [[0, 1], [1, 0]]])                                    # stay, move
    Rw = np.array([[0.0, 0.0], [1.0, 0.0]])                                        # staying in state 1 pays 1
    V, pi = rl.value_iteration(P, Rw, gamma=0.9)
    np.testing.assert_allclose(V, [9.0, 10.0], atol=1e-8)
    np.testing.assert_array_equal(pi, [1, 0])


def test_q_learning_one_path_overfits_many_paths_generalize():
    one, many = [ou_path(100, 0)], [ou_path(100, s) for s in range(10, 210)]
    test = [ou_path(100, s) for s in range(1000, 1300)]
    q1 = rl.q_learning_paths(one, EDGES, epochs=200)
    qm = rl.q_learning_paths(many, EDGES, epochs=1)
    assert q1.shape == (18, 3)
    in_sample, oos_one, oos_many = (rl.greedy_reward(q1, one, EDGES), rl.greedy_reward(q1, test, EDGES),
                                    rl.greedy_reward(qm, test, EDGES))
    assert in_sample > 2 * oos_one and oos_many > oos_one                          # memorized one history
    assert qm.argmax(1)[0] == 2 and qm.argmax(1)[-1] == 0                          # buy low, sell high


def test_trading_env_timing_and_costs():
    r = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    env = rl.TradingEnv(r, lookback=2, cost=0.001, scale=1.0)
    check(rl.TradingEnv(TRAIN))
    obs, _ = env.reset()
    np.testing.assert_allclose(obs, [0.01, 0.02, 0.0])
    obs, rew, done, trunc, _ = env.step(2)                                         # long: earns r[2], AFTER deciding
    assert rew == pytest.approx(-0.011) and not done and not trunc
    np.testing.assert_allclose(obs, [0.02, -0.01, 1.0])
    _, rew, _, _, _ = env.step(0)                                                  # flip to short: cost × 2
    assert rew == pytest.approx(-0.032)
    _, rew, done, _, _ = env.step(1)
    assert rew == pytest.approx(-0.001) and done
    assert rl.TradingEnv(TEST).scale == pytest.approx(TEST.std(), rel=1e-5)
    ep = rl.TradingEnv(TRAIN, episode_len=50)
    a, _ = ep.reset(seed=3)
    b, _ = ep.reset(seed=3)
    np.testing.assert_array_equal(a, b)
    assert sum(1 for _ in iter(lambda: ep.step(1)[2], True)) == 49                  # 50 steps in all


def test_rules_and_hindsight_on_known_structure():
    env = rl.TradingEnv(TEST, cost=0.0002, scale=TRAIN.std())
    long_only = rl.run_policy(env, lambda o: 2)
    momentum = rl.run_policy(env, lambda o: 2 if o[-2] > 0 else 0)
    hindsight = np.abs(TEST[10:]).sum()
    assert long_only < momentum < hindsight


def test_q_agent_learns_momentum():
    Q = rl.q_learning_env(rl.TradingEnv(TRAIN, cost=0.0002, episode_len=250), 6, episodes=300)
    policy = Q.argmax(1)
    np.testing.assert_array_equal(policy[:3], 0)                                   # last return down → short
    np.testing.assert_array_equal(policy[3:], 2)                                   # last return up → long
    env = rl.TradingEnv(TEST, cost=0.0002, scale=TRAIN.std())
    agent = rl.run_policy(env, lambda o: int(Q[rl.sign_state(o)].argmax()))
    assert agent > rl.run_policy(env, lambda o: 2)


def test_differential_sharpe():
    d = rl.differential_sharpe([0.01, 0.02], eta=0.01)
    A, B = 0.0001, 1e-6
    expected = (B * (0.02 - A) - 0.5 * A * (0.0004 - B)) / (B - A * A) ** 1.5
    np.testing.assert_allclose(d, [0.0, expected])
    r = np.random.default_rng(0).normal(0.001, 0.01, 2000)
    d = rl.differential_sharpe(r)
    assert np.corrcoef(d[100:], r[100:])[0, 1] > 0.9                                # a good day raises the Sharpe
    assert abs(d[100:].mean()) < 0.1                                               # it is a CHANGE: ≈ 0 on average


def test_almgren_chriss_vs_twap():
    X, N, sig, eta, gam, lam = 1000, 20, 0.3, 0.01, 0.001, 1e-3
    tw = rl.twap_schedule(X, N)
    ac = rl.almgren_chriss_schedule(X, N, sig, eta, gam, lam)
    assert len(ac) == N and ac.sum() == pytest.approx(X) and (np.diff(ac) < 0).all()   # front-loaded
    np.testing.assert_allclose(rl.almgren_chriss_schedule(X, N, sig, eta, gam, 1e-12), tw, rtol=1e-4)
    e_tw, v_tw = rl.cost_moments(tw, sig, eta, gam)
    e_ac, v_ac = rl.cost_moments(ac, sig, eta, gam)
    assert e_tw == pytest.approx(0.5 * gam * X ** 2 + (eta - gam / 2) * N * 50 ** 2)
    assert e_tw < e_ac and v_ac < v_tw                                             # pay a little for less risk
    assert e_ac + lam * v_ac < e_tw + lam * v_tw
    rng = np.random.default_rng(0)
    for _ in range(10):                                                            # no nearby schedule does better
        d = rng.normal(0, 1, N)
        d -= d.mean()
        e, v = rl.cost_moments(ac + d, sig, eta, gam)
        assert e + lam * v >= e_ac + lam * v_ac - 1e-9


def test_execution_env():
    env = rl.ExecutionEnv()
    check(env)
    tw = rl.twap_schedule(1000, 20)
    sim = rl.simulate_execution(env, rl.schedule_policy(tw), n_paths=400)
    e, v = rl.cost_moments(tw, 0.3, 0.01, 0.001)
    assert sim.mean() == pytest.approx(e, abs=3 * np.sqrt(v / 400)) and sim.var() == pytest.approx(v, rel=0.2)
    wait = rl.simulate_execution(env, lambda obs, k: 0.0, n_paths=50)             # dump everything at the end
    assert wait.mean() > 5 * e
    obs, _ = env.reset(seed=0)
    for k in range(20):
        obs, _, done, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert done and obs[0] == 0.0 and info["n"] == 1000
