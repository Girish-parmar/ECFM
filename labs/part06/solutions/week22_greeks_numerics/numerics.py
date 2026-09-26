"""Week 22 (S5–S6) — Second-order Greeks, bump-and-revalue, binomial trees, Monte Carlo, Crank–Nicolson.

Always verify closed-form Greeks with finite differences. Charm, color and veta follow theta's convention: the change
AS TIME PASSES (−∂/∂T), per year. Uses your week 21 bsm_price / greeks / d1d2.
Fill in every block marked "Your turn", then run:  python -m pytest week22_greeks_numerics
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm

from _loader import load

pr = load("week21_pricing_iv", "pricing")
bsm_price, d1d2 = pr.bsm_price, pr.d1d2
N, n = norm.cdf, norm.pdf


# -------------------------------------------------------------------- S5 second-order Greeks
def second_order(S, K, T, r, q, sigma, cp) -> dict[str, np.ndarray]:
    """vanna ∂Δ/∂σ · volga ∂vega/∂σ · charm −∂Δ/∂T · speed ∂Γ/∂S · zomma ∂Γ/∂σ · color −∂Γ/∂T · veta −∂vega/∂T ·
    ultima ∂volga/∂σ (closed forms from the lesson plan, S5; vega = S e^{−qT} n(d1) √T, gamma as in week 21)."""
    # >>> SOLUTION
    d1, d2 = d1d2(S, K, T, r, q, sigma)
    sqT, dq = np.sqrt(T), np.exp(-q * T)
    pdf = n(d1)
    gamma = dq * pdf / (S * sigma * sqT)
    vega = S * dq * pdf * sqT
    g = {"vanna": -dq * pdf * d2 / sigma, "volga": vega * d1 * d2 / sigma}
    g["charm"] = cp * q * dq * N(cp * d1) - dq * pdf * (2 * (r - q) * T - d2 * sigma * sqT) / (2 * T * sigma * sqT)
    g["speed"] = -gamma / S * (d1 / (sigma * sqT) + 1)
    g["zomma"] = gamma * (d1 * d2 - 1) / sigma
    g["color"] = dq * pdf / (2 * S * T * sigma * sqT) * (
        2 * q * T + 1 + (2 * (r - q) * T - d2 * sigma * sqT) / (sigma * sqT) * d1)
    g["veta"] = S * dq * pdf * sqT * (q + (r - q) * d1 / (sigma * sqT) - (1 + d1 * d2) / (2 * T))
    g["ultima"] = -vega / sigma ** 2 * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)
    return g
    # <<< SOLUTION


def greeks_fd(pricer, S, K, T, r, q, sigma, cp, hs: float = 1e-3, hv: float = 1e-4, ht: float = 1e-5,
              hr: float = 1e-5) -> dict[str, float]:
    """Bump-and-revalue Greeks for ANY pricer(S, K, T, r, q, sigma, cp), central differences with relative spot bump
    dS = hs·S:  delta, gamma, vega, theta (= −∂V/∂T), rho, vanna (∂²V/∂S∂σ), volga (∂²V/∂σ²)."""
    # >>> SOLUTION
    def V(s=S, t=T, rr=r, v=sigma):
        return pricer(s, K, t, rr, q, v, cp)
    dS = hs * S
    return {
        "delta": (V(s=S + dS) - V(s=S - dS)) / (2 * dS),
        "gamma": (V(s=S + dS) - 2 * V() + V(s=S - dS)) / dS ** 2,
        "vega": (V(v=sigma + hv) - V(v=sigma - hv)) / (2 * hv),
        "theta": -(V(t=T + ht) - V(t=T - ht)) / (2 * ht),
        "rho": (V(rr=r + hr) - V(rr=r - hr)) / (2 * hr),
        "vanna": (V(s=S + dS, v=sigma + hv) - V(s=S + dS, v=sigma - hv)
                  - V(s=S - dS, v=sigma + hv) + V(s=S - dS, v=sigma - hv)) / (4 * dS * hv),
        "volga": (V(v=sigma + hv) - 2 * V() + V(v=sigma - hv)) / hv ** 2,
    }
    # <<< SOLUTION


# ------------------------------------------------------------------------- S6 binomial tree
def crr_price(S, K, T, r, q, sigma, cp, steps: int = 500, american: bool = True) -> float:
    """Cox–Ross–Rubinstein tree: u = e^{σ√dt}, d = 1/u, p = (e^{(r−q)dt} − d)/(u − d), discount e^{−r dt}.
    Roll back from the terminal payoff; if american, take max(continuation, exercise value) at every node."""
    # >>> SOLUTION
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    ST = S * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    V = np.maximum(cp * (ST - K), 0.0)
    for _ in range(steps):
        ST = ST[:-1] / u
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            V = np.maximum(V, cp * (ST - K))
    return float(V[0])
    # <<< SOLUTION


# ---------------------------------------------------------------------------- S6 Monte Carlo
def mc_european(S, K, T, r, q, sigma, cp, n_paths: int = 200_000, seed: int = 0) -> tuple[float, float]:
    """Risk-neutral GBM terminal prices with ANTITHETIC variates: draw n_paths//2 normals z, use z and −z.
    Return (price, standard error), where the SE uses the n_paths//2 antithetic PAIR averages (they are the i.i.d.
    units): std(pairs, ddof=1)/√(#pairs)."""
    # >>> SOLUTION
    m = n_paths // 2
    z = np.random.default_rng(seed).standard_normal(m)
    z = np.concatenate([z, -z])
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    pay = np.exp(-r * T) * np.maximum(cp * (ST - K), 0.0)
    pairs = 0.5 * (pay[:m] + pay[m:])
    return float(pay.mean()), float(pairs.std(ddof=1) / np.sqrt(m))
    # <<< SOLUTION


def mc_delta(S, K, T, r, q, sigma, cp, h: float = 0.01, n_paths: int = 50_000, seed: int = 0,
             common_random_numbers: bool = True) -> float:
    """Bump-and-revalue MC delta (V(S+h) − V(S−h)) / 2h with mc_european. With common random numbers both
    revaluations use the SAME seed; without, the down-bump uses seed + 1 (to show how noisy that is)."""
    # >>> SOLUTION
    up = mc_european(S + h, K, T, r, q, sigma, cp, n_paths, seed)[0]
    dn = mc_european(S - h, K, T, r, q, sigma, cp, n_paths, seed if common_random_numbers else seed + 1)[0]
    return (up - dn) / (2 * h)
    # <<< SOLUTION


# --------------------------------------------------------------------- S6 finite differences
def crank_nicolson_call(S, K, T, r, q, sigma, n_s: int = 200, n_t: int = 200, s_max_mult: float = 4.0) -> float:
    """European call by Crank–Nicolson on the BSM PDE in S, with RANNACHER start-up (the first 2 time steps are
    replaced by 2 implicit Euler steps of dt/2) to damp the kink of the payoff at K. The initial payoff is
    CELL-AVERAGED: node S_j gets the mean of max(s − K, 0) over [S_j − dS/2, S_j + dS/2], which restores clean
    second-order convergence when K falls between nodes.
    Grid: S_j = j·dS, j = 0..n_s, with dS chosen so that S is exactly a node (j0 = round(n_s·S/(s_max_mult·K)),
    dS = S/j0, S_max = n_s·dS); τ = time to expiry, 0 → T in n_t steps.
    Boundaries: V(0) = 0, V(S_max) = S_max e^{−qτ} − K e^{−rτ}. Return V at node j0."""
    # >>> SOLUTION
    j0 = max(1, round(n_s * S / (s_max_mult * K)))           # put S exactly on node j0
    dS = S / j0
    s_max = n_s * dS
    Sg = np.arange(n_s + 1) * dS
    h = dS / 2                                                  # cell-averaged payoff: smooths the kink at K
    V = np.where(K <= Sg - h, Sg - K, np.where(K >= Sg + h, 0.0, (Sg + h - K) ** 2 / (2 * dS)))
    j = np.arange(1, n_s)
    a = 0.5 * (sigma ** 2 * j ** 2 - (r - q) * j)          # coefficient of V_{j-1}
    b = -(sigma ** 2 * j ** 2 + r)                            # coefficient of V_j
    c = 0.5 * (sigma ** 2 * j ** 2 + (r - q) * j)          # coefficient of V_{j+1}

    def step(V, tau_new, dt, theta):
        """theta = 1 implicit Euler, 0.5 Crank–Nicolson."""
        rhs = V[1:-1] + (1 - theta) * dt * (a * V[:-2] + b * V[1:-1] + c * V[2:])
        upper = s_max * np.exp(-q * tau_new) - K * np.exp(-r * tau_new)
        rhs[-1] += theta * dt * c[-1] * upper
        ab = np.zeros((3, n_s - 1))
        ab[0, 1:] = -theta * dt * c[:-1]
        ab[1] = 1 - theta * dt * b
        ab[2, :-1] = -theta * dt * a[1:]
        out = np.empty_like(V)
        out[0], out[-1] = 0.0, upper
        out[1:-1] = solve_banded((1, 1), ab, rhs)
        return out

    dt = T / n_t
    tau = 0.0
    for _ in range(2):                                         # Rannacher: 2 implicit half steps
        tau += dt / 2
        V = step(V, tau, dt / 2, 1.0)
    for _ in range(n_t - 1):
        tau += dt
        V = step(V, tau, dt, 0.5)
    return float(V[j0])
    # <<< SOLUTION
