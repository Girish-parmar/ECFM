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
    raise NotImplementedError("✍️ Your turn: see the docstring")


def greeks_fd(pricer, S, K, T, r, q, sigma, cp, hs: float = 1e-3, hv: float = 1e-4, ht: float = 1e-5,
              hr: float = 1e-5) -> dict[str, float]:
    """Bump-and-revalue Greeks for ANY pricer(S, K, T, r, q, sigma, cp), central differences with relative spot bump
    dS = hs·S:  delta, gamma, vega, theta (= −∂V/∂T), rho, vanna (∂²V/∂S∂σ), volga (∂²V/∂σ²)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ------------------------------------------------------------------------- S6 binomial tree
def crr_price(S, K, T, r, q, sigma, cp, steps: int = 500, american: bool = True) -> float:
    """Cox–Ross–Rubinstein tree: u = e^{σ√dt}, d = 1/u, p = (e^{(r−q)dt} − d)/(u − d), discount e^{−r dt}.
    Roll back from the terminal payoff; if american, take max(continuation, exercise value) at every node."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# ---------------------------------------------------------------------------- S6 Monte Carlo
def mc_european(S, K, T, r, q, sigma, cp, n_paths: int = 200_000, seed: int = 0) -> tuple[float, float]:
    """Risk-neutral GBM terminal prices with ANTITHETIC variates: draw n_paths//2 normals z, use z and −z.
    Return (price, standard error), where the SE uses the n_paths//2 antithetic PAIR averages (they are the i.i.d.
    units): std(pairs, ddof=1)/√(#pairs)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


def mc_delta(S, K, T, r, q, sigma, cp, h: float = 0.01, n_paths: int = 50_000, seed: int = 0,
             common_random_numbers: bool = True) -> float:
    """Bump-and-revalue MC delta (V(S+h) − V(S−h)) / 2h with mc_european. With common random numbers both
    revaluations use the SAME seed; without, the down-bump uses seed + 1 (to show how noisy that is)."""
    raise NotImplementedError("✍️ Your turn: see the docstring")


# --------------------------------------------------------------------- S6 finite differences
def crank_nicolson_call(S, K, T, r, q, sigma, n_s: int = 200, n_t: int = 200, s_max_mult: float = 4.0) -> float:
    """European call by Crank–Nicolson on the BSM PDE in S, with RANNACHER start-up (the first 2 time steps are
    replaced by 2 implicit Euler steps of dt/2) to damp the kink of the payoff at K. The initial payoff is
    CELL-AVERAGED: node S_j gets the mean of max(s − K, 0) over [S_j − dS/2, S_j + dS/2], which restores clean
    second-order convergence when K falls between nodes.
    Grid: S_j = j·dS, j = 0..n_s, with dS chosen so that S is exactly a node (j0 = round(n_s·S/(s_max_mult·K)),
    dS = S/j0, S_max = n_s·dS); τ = time to expiry, 0 → T in n_t steps.
    Boundaries: V(0) = 0, V(S_max) = S_max e^{−qτ} − K e^{−rτ}. Return V at node j0."""
    raise NotImplementedError("✍️ Your turn: see the docstring")
