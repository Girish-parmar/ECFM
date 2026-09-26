"""Helper functions for the Part 6 guided notebooks (MFAAT, Month 6: futures & options engineering).

The notebooks give you the setup, data and plotting code; you write the short cells marked "✍️ Your turn".
Each exercise ends with `p.check(...)`, which compares your answer with the reference implementation in this
file. If it does not match yet, the notebook carries on with the reference value so later cells still run.

All data is SYNTHETIC with known parameters (a futures strip with a known carry, an option chain priced from a known SVI
surface), so every notebook runs offline and every estimate can be compared with the truth. The graded versions (checked
against py_vollib golden values) are in labs/part06/.

Units (lesson plan, Section 4): T in years (ACT/365), σ as a decimal, vega per 1.00 of σ, theta per year (−∂V/∂T), rho
per 1.00 of r, continuous compounding, cp = +1 call / −1 put. Brokers display vega per vol point, theta per day.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRICT = os.environ.get("P6_STRICT") == "1"      # tests: a failed check raises instead of continuing
N, n = norm.cdf, norm.pdf


# --------------------------------------------------------------------------------------
# Plot style and the exercise checker
# --------------------------------------------------------------------------------------
def use_course_style() -> None:
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=PALETTE), "figure.figsize": (9, 4.5),
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#52514e", "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "legend.frameon": False,
    })


def _numeric(x) -> bool:
    return isinstance(x, (int, float, np.number, np.ndarray)) and not isinstance(x, bool)


def _same(got, expected, rtol: float, atol: float) -> bool:
    if isinstance(expected, Decimal):
        return isinstance(got, Decimal) and got == expected            # exact, and a float is not money
    if isinstance(expected, (pd.DataFrame, pd.Series)):
        test = pd.testing.assert_frame_equal if isinstance(expected, pd.DataFrame) else pd.testing.assert_series_equal
        try:
            test(got, expected, check_exact=False, rtol=rtol, atol=atol, check_dtype=False, check_names=False,
                 check_freq=False)
            return True
        except (AssertionError, TypeError, AttributeError):
            return False
    if isinstance(expected, dict):
        return (isinstance(got, dict) and set(got) == set(expected)
                and all(_same(got[k], v, rtol, atol) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)) and expected and all(_numeric(v) and np.ndim(v) == 0 for v in expected):
        expected = np.asarray(expected, dtype=float)
    if _numeric(expected):
        g, e = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
        return g.shape == e.shape and np.allclose(g, e, rtol=rtol, atol=atol, equal_nan=True)
    if isinstance(expected, (list, tuple)):
        got = list(got)
        return len(got) == len(expected) and all(_same(a, b, rtol, atol) for a, b in zip(got, expected))
    return type(got) is type(expected) and got == expected if isinstance(expected, bool) else got == expected


def check(name: str, got, expected, rtol: float = 1e-6, atol: float = 1e-9):
    """Compare your answer with the reference: exact for Decimals, text, dates and objects (recursively inside
    lists and dicts); with a tolerance for floats and arrays. Returns your value if correct, else the reference
    (so the notebook keeps running)."""
    try:
        if got is Ellipsis or (isinstance(got, (tuple, list)) and any(g is Ellipsis for g in got)):
            raise ValueError("not done yet")
        ok = bool(_same(got, expected, rtol, atol))
    except Exception:  # noqa: BLE001 - any failure means "not correct yet"
        ok = False
    if ok:
        print(f"✅ {name}: correct")
        return got
    msg = f"❌ {name}: not matching the reference yet"
    if STRICT:
        raise AssertionError(msg)
    print(msg + " — continuing with the reference answer so the rest of the notebook runs.")
    return expected


def attempt(fn, *args, **kwargs):
    """Run fn; if it raises (for example because a blank `...` is still in it), return Ellipsis instead, so
    p.check reports "not done yet" and the notebook keeps going."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - an unfinished exercise may fail in any way
        return Ellipsis


# --------------------------------------------------------------------------------------
# 01 · futures: expiries, fair value, continuous series
# --------------------------------------------------------------------------------------
MONTH_CODES = "FGHJKMNQUVXZ"


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def roll_date(expiry: date, business_days_before: int = 8) -> date:
    d, k = expiry, 0
    while k < business_days_before:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            k += 1
    return d


def fair_value(S, r, q, T):
    """F = S·e^{(r−q)T}."""
    return S * np.exp((r - q) * T)


def implied_carry(f_near, f_far, t_near, t_far):
    """Annualized carry (r − q) implied by two futures: ln(F_far/F_near)/(t_far − t_near)."""
    return np.log(f_far / f_near) / (t_far - t_near)


def futures_panel(seed: int = 1, carry: float = 0.03):
    """Daily closes of quarterly index futures (columns 'ESH4' … in expiry order, NaN outside each contract's life),
    the spot index, the expiries and the roll dates (8 business days before each expiry; from a roll date on, the
    next contract is held). Futures = spot·e^{carry·T} + a little noise: a market in contango."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", "2025-11-28")
    spot = 4800 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(days))))
    exps = [third_friday(y, m) for y in (2024, 2025) for m in (3, 6, 9, 12)]
    cols, data = [], {}
    for e in exps:
        code = f"ES{MONTH_CODES[e.month - 1]}{str(e.year)[-1]}"
        T = np.array([(e - d.date()).days / 365 for d in days])
        live = (T >= 0) & (T <= 0.55)
        px = spot * np.exp(carry * T) + rng.normal(0, 0.8, len(days))
        data[code] = np.where(live, np.round(px * 4) / 4, np.nan)
        cols.append(code)
    prices = pd.DataFrame(data, index=days)
    rolls = [pd.Timestamp(roll_date(e)) for e in exps[:-1]]
    return prices, pd.Series(spot, index=days, name="spot"), exps, rolls


def continuous(prices: pd.DataFrame, rolls: list, method: str = "difference") -> pd.Series:
    """Stitch contracts (columns in expiry order) into one series; from rolls[i] on, contract i+1 is held.
    'none': raw prices (jumps at rolls). 'difference': add new − old at each roll to ALL earlier history.
    'ratio': multiply all earlier history by new/old. The last contract's prices are never changed."""
    cols = list(prices.columns)
    held = np.searchsorted(pd.DatetimeIndex(rolls).to_numpy(), prices.index.to_numpy(), side="right")
    out = prices.to_numpy()[np.arange(len(prices)), held].astype(float)
    for i, d in enumerate(rolls):
        old, new = prices.at[d, cols[i]], prices.at[d, cols[i + 1]]
        before = held <= i
        if method == "difference":
            out[before] += new - old
        elif method == "ratio":
            out[before] *= new / old
    return pd.Series(out, index=prices.index, name=method)


# --------------------------------------------------------------------------------------
# 02 · option chains and strike selection
# --------------------------------------------------------------------------------------
def occ_symbol(root: str, expiry: date, cp: str, strike: float) -> str:
    """root + YYMMDD + C/P + strike × 1000 as 8 digits, e.g. AAPL261218C00200000."""
    return f"{root}{expiry:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


def parse_occ(symbol: str) -> dict:
    """{"root", "expiry", "cp", "strike"} from an OCC symbol: the last 15 characters are YYMMDD, C/P and 8 digits."""
    root, ymd, cp, k = symbol[:-15], symbol[-15:-9], symbol[-9], symbol[-8:]
    return {"root": root, "expiry": date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:])), "cp": cp,
            "strike": int(k) / 1000}


def liquidity_filter(chain: pd.DataFrame, max_spread_pct: float = 0.10, min_oi: int = 100) -> pd.DataFrame:
    """Keep rows with bid > 0, ask >= bid, (ask − bid)/mid <= max_spread_pct and oi >= min_oi."""
    spread = (chain["ask"] - chain["bid"]) / chain["mid"]
    keep = (chain["bid"] > 0) & (chain["ask"] >= chain["bid"]) & (spread <= max_spread_pct) & (chain["oi"] >= min_oi)
    return chain[keep]


def atm_strike(strikes, forward: float) -> float:
    """Listed strike nearest to the FORWARD; ties go to the lower strike."""
    k = np.sort(np.asarray(strikes, dtype=float))
    return float(k[np.argmin(np.abs(k - forward))])


def expected_move_strikes(strikes, S: float, sigma: float, T: float) -> tuple[float, float]:
    """Listed strikes nearest to S·(1 − σ√T) and S·(1 + σ√T)."""
    k = np.sort(np.asarray(strikes, dtype=float))
    m = S * sigma * np.sqrt(T)
    return float(k[np.argmin(np.abs(k - (S - m)))]), float(k[np.argmin(np.abs(k - (S + m)))])


# --------------------------------------------------------------------------------------
# 03 · BSM, Black-76 and first-order Greeks
# --------------------------------------------------------------------------------------
def d1d2(S, K, T, r, q, sigma):
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sq)
    return d1, d1 - sigma * sq


def bsm_price(S, K, T, r, q, sigma, cp):
    """cp·(S e^{−qT} N(cp·d1) − K e^{−rT} N(cp·d2))."""
    d1, d2 = d1d2(S, K, T, r, q, sigma)
    return cp * (S * np.exp(-q * T) * N(cp * d1) - K * np.exp(-r * T) * N(cp * d2))


def black76_price(F, K, T, r, sigma, cp):
    """Options on futures: BSM with S = F and q = r."""
    return bsm_price(F, K, T, r, r, sigma, cp)


def greeks(S, K, T, r, q, sigma, cp) -> dict:
    """RAW units: delta, gamma, vega (per 1.00 σ), theta (per year, −∂V/∂T), rho (per 1.00 r)."""
    d1, d2 = d1d2(S, K, T, r, q, sigma)
    sq, dq, dr = np.sqrt(T), np.exp(-q * T), np.exp(-r * T)
    pdf = n(d1)
    return {"delta": cp * dq * N(cp * d1), "gamma": dq * pdf / (S * sigma * sq), "vega": S * dq * pdf * sq,
            "theta": -S * dq * pdf * sigma / (2 * sq) - cp * r * K * dr * N(cp * d2) + cp * q * S * dq * N(cp * d1),
            "rho": cp * K * T * dr * N(cp * d2)}


def to_display(g: dict) -> dict:
    """vega per vol point (÷100), theta per calendar day (÷365), rho per 1% (÷100); a NEW dict."""
    out = dict(g)
    out["vega"], out["theta"], out["rho"] = g["vega"] / 100, g["theta"] / 365, g["rho"] / 100
    return out


def parity_gap(call, put, S, K, T, r, q):
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))


# --------------------------------------------------------------------------------------
# 04 · implied volatility and the implied forward
# --------------------------------------------------------------------------------------
def svi_total_var(k, a, b, rho, m, s):
    """Raw SVI total variance w(k) = a + b(ρ(k − m) + √((k − m)² + s²)), k = ln(K/F)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


TRUE_SVI = (0.0012, 0.035, -0.70, 0.03, 0.08)     # one-month equity-index skew


def synthetic_chain(S: float = 600.0, T: float = 30 / 365, r: float = 0.045, q: float = 0.013, seed: int = 0,
                    svi=TRUE_SVI, width: float = 5.0) -> pd.DataFrame:
    """Calls and puts on one expiry, priced from a KNOWN SVI smile around the forward, with bid/ask spreads that widen
    in the wings, zero bids far out, and open interest. Columns: strike, cp, bid, ask, mid, oi, iv_true."""
    rng = np.random.default_rng(seed)
    F = fair_value(S, r, q, T)
    K = np.arange(np.round(S * 0.80 / width) * width, S * 1.20 + width, width)
    iv = np.sqrt(svi_total_var(np.log(K / F), *svi) / T)
    rows = []
    for cp in (1, -1):
        mid = bsm_price(S, K, T, r, q, iv, cp)
        tick = np.where(mid < 3, 0.01, 0.05)
        half = np.maximum(0.01, 0.015 * mid + 0.02 * np.abs(np.log(K / F)) * 40) / 2
        bid = np.round((mid - half) / tick) * tick
        ask = np.round((mid + half) / tick) * tick
        bid = np.where(mid < 0.08, 0.0, bid)
        oi = (rng.gamma(2, 1, K.size) * 3000 * np.exp(-30 * np.log(K / F) ** 2)).round().astype(int)
        rows.append(pd.DataFrame({"strike": K, "cp": cp, "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "oi": oi,
                                  "iv_true": iv, "model_mid": mid}))
    return pd.concat(rows, ignore_index=True)


def implied_rate_forward(strikes, calls, puts, T: float) -> tuple[float, float]:
    """Regress C − P = a + b·K (parity: C − P = DF·F − DF·K). DF = −b, r = −ln(DF)/T, F = a/DF. Returns (r, F)."""
    b, a = np.polyfit(np.asarray(strikes, float), np.asarray(calls, float) - np.asarray(puts, float), 1)
    df_ = -b
    return float(-np.log(df_) / T), float(a / df_)


def price_bounds(S, K, T, r, q, cp) -> tuple[float, float]:
    fs, fk = S * np.exp(-q * T), K * np.exp(-r * T)
    return max(cp * (fs - fk), 0.0), (fs if cp == 1 else fk)


def newton_iv(price, S, K, T, r, q, cp, sigma0: float = 0.3, steps: int = 20, tol: float = 1e-10) -> float:
    """Plain Newton on σ from sigma0: σ ← σ − (model − price)/vega. NaN if it does not converge within `steps`,
    vega < 1e-8, or σ leaves (1e-4, 5)."""
    sigma = sigma0
    for _ in range(steps):
        diff = bsm_price(S, K, T, r, q, sigma, cp) - price
        if abs(diff) < tol:
            return float(sigma)
        vega = greeks(S, K, T, r, q, sigma, cp)["vega"]
        if vega < 1e-8:
            return np.nan
        sigma -= diff / vega
        if not 1e-4 < sigma < 5:
            return np.nan
    return np.nan


def implied_vol(price, S, K, T, r, q, cp, tol: float = 1e-10) -> float:
    """Newton first, Brent's bracketing method as the fallback; NaN outside the no-arbitrage bounds."""
    lo, hi = price_bounds(S, K, T, r, q, cp)
    if not lo < price < hi:
        return np.nan
    s = newton_iv(price, S, K, T, r, q, cp, tol=tol)
    if np.isfinite(s):
        return s
    return float(brentq(lambda v: bsm_price(S, K, T, r, q, v, cp) - price, 1e-4, 5.0, xtol=tol))


# --------------------------------------------------------------------------------------
# 05 · second-order Greeks and bump-and-revalue
# --------------------------------------------------------------------------------------
def second_order(S, K, T, r, q, sigma, cp) -> dict:
    """vanna = ∂Δ/∂σ = −e^{−qT} n(d1) d2/σ;  volga = ∂vega/∂σ = vega·d1·d2/σ;
    charm = −∂Δ/∂T (change of delta as time passes, per year)."""
    d1, d2 = d1d2(S, K, T, r, q, sigma)
    dq, sq = np.exp(-q * T), np.sqrt(T)
    vega = S * dq * n(d1) * sq
    charm = cp * q * dq * N(cp * d1) - dq * n(d1) * (2 * (r - q) * T - d2 * sigma * sq) / (2 * T * sigma * sq)
    return {"vanna": -dq * n(d1) * d2 / sigma, "volga": vega * d1 * d2 / sigma, "charm": charm}


def bump(pricer, args: dict, name: str, h: float, order: int = 1):
    """Central finite difference of pricer(**args) in argument `name`: first derivative (V₊ − V₋)/(2h), or
    second derivative (V₊ − 2V + V₋)/h²."""
    up, dn = dict(args), dict(args)
    up[name] = args[name] + h
    dn[name] = args[name] - h
    if order == 1:
        return (pricer(**up) - pricer(**dn)) / (2 * h)
    return (pricer(**up) - 2 * pricer(**args) + pricer(**dn)) / h ** 2


# --------------------------------------------------------------------------------------
# 06 · numerical methods
# --------------------------------------------------------------------------------------
def crr_price(S, K, T, r, q, sigma, cp, steps: int = 500, american: bool = True) -> float:
    """Cox–Ross–Rubinstein tree: u = e^{σ√dt}, d = 1/u, p = (e^{(r−q)dt} − d)/(u − d); backward induction, taking
    max(continuation, exercise) at every node when american."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    v = np.maximum(cp * (S * u ** j * d ** (steps - j) - K), 0.0)
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        v = disc * (p * v[1:] + (1 - p) * v[:-1])
        if american:
            v = np.maximum(v, cp * (S * u ** j * d ** (i - j) - K))
    return float(v[0])


def mc_european(S, K, T, r, q, sigma, cp, n_paths: int = 100_000, seed: int = 0, antithetic: bool = True):
    """Monte Carlo price and its standard error. With antithetic, use z and −z and average each pair FIRST (the pairs
    are the independent samples)."""
    rng = np.random.default_rng(seed)
    drift, vol = (r - q - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        pay = 0.5 * (np.maximum(cp * (S * np.exp(drift + vol * z) - K), 0)
                     + np.maximum(cp * (S * np.exp(drift - vol * z) - K), 0))
    else:
        z = rng.standard_normal(n_paths)
        pay = np.maximum(cp * (S * np.exp(drift + vol * z) - K), 0)
    x = disc * pay
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def mc_delta(S, K, T, r, q, sigma, cp, h: float = 0.5, n_paths: int = 20_000, seed: int = 0, common: bool = True):
    """Bump-and-revalue delta by Monte Carlo, with the SAME random numbers for S+h and S−h (common) or different ones."""
    up = mc_european(S + h, K, T, r, q, sigma, cp, n_paths, seed, antithetic=False)[0]
    dn = mc_european(S - h, K, T, r, q, sigma, cp, n_paths, seed if common else seed + 1, antithetic=False)[0]
    return (up - dn) / (2 * h)


# --------------------------------------------------------------------------------------
# 07 · the volatility surface
# --------------------------------------------------------------------------------------
def fit_svi(k, iv, T, x0=(0.001, 0.05, -0.5, 0.0, 0.1)) -> np.ndarray:
    """Least-squares raw SVI fit to total variance, with bounds that keep it sensible (b >= 0, |ρ| < 1, s > 0)."""
    w = np.asarray(iv) ** 2 * T
    res = least_squares(lambda x: svi_total_var(k, *x) - w, x0,
                        bounds=([-0.05, 0.0, -0.999, -0.5, 1e-4], [0.05, 1.0, 0.999, 0.5, 1.0]))
    return res.x


def butterfly_g(k, a, b, rho, m, s):
    """Durrleman's density condition g(k) = (1 − k w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2; g < 0 anywhere means
    butterfly arbitrage (a negative risk-neutral density)."""
    x = k - m
    root = np.sqrt(x ** 2 + s ** 2)
    w = a + b * (rho * x + root)
    w1 = b * (rho + x / root)
    w2 = b * s ** 2 / root ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def calendar_violations(k_grid, params_by_T: dict) -> list[tuple[float, float, float]]:
    """(T_short, T_long, k) wherever total variance DECREASES from one expiry to the next (calendar arbitrage)."""
    Ts = sorted(params_by_T)
    out = []
    for t1, t2 in zip(Ts, Ts[1:]):
        w1, w2 = svi_total_var(k_grid, *params_by_T[t1]), svi_total_var(k_grid, *params_by_T[t2])
        out += [(t1, t2, float(k)) for k, a, b in zip(k_grid, w1, w2) if b < a - 1e-12]
    return out


def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) lognormal SABR implied volatility."""
    F, K = np.asarray(F, float), np.asarray(K, float)
    one_b = 1 - beta
    fk = (F * K) ** (one_b / 2)
    logfk = np.log(F / K)
    corr = 1 + (one_b ** 2 / 24 * alpha ** 2 / fk ** 2 + rho * beta * nu * alpha / (4 * fk)
                + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
    z = nu / alpha * fk * logfk
    x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    zx = np.where(np.abs(z) < 1e-8, 1.0, z / np.where(np.abs(x) < 1e-12, 1.0, x))
    denom = fk * (1 + one_b ** 2 / 24 * logfk ** 2 + one_b ** 4 / 1920 * logfk ** 4)
    return alpha / denom * zx * corr


# --------------------------------------------------------------------------------------
# 08 · portfolio Greeks, scenarios, hedging
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Position:
    """qty contracts (negative = short). kind 'option' (on the underlying S) or 'stock'. multiplier from contract
    details (100 for equity options)."""
    name: str
    kind: str
    qty: float
    multiplier: float = 100.0
    K: float = 0.0
    T: float = 0.0
    iv: float = 0.0
    cp: int = 1

    def value(self, S, r=0.045, q=0.013, dvol=0.0, dT=0.0):
        if self.kind == "stock":
            return self.qty * self.multiplier * S
        return self.qty * self.multiplier * bsm_price(S, self.K, max(self.T - dT, 1e-8), r, q, self.iv + dvol, self.cp)


def dollar_greeks(p: Position, S: float, r=0.045, q=0.013) -> dict:
    """Book-level Greeks: $delta = Δ·qty·mult·S (P&L of a 100% move, linearized), $gamma = ½Γ·qty·mult·S²·1%² (P&L of
    a 1% move from gamma), vega per vol point ×qty×mult, theta per day ×qty×mult."""
    if p.kind == "stock":
        return {"$delta": p.qty * p.multiplier * S, "$gamma(1%)": 0.0, "vega/pt": 0.0, "theta/day": 0.0}
    g = to_display(greeks(S, p.K, p.T, r, q, p.iv, p.cp))
    m = p.qty * p.multiplier
    return {"$delta": float(g["delta"] * m * S), "$gamma(1%)": float(0.5 * g["gamma"] * m * (0.01 * S) ** 2),
            "vega/pt": float(g["vega"] * m), "theta/day": float(g["theta"] * m)}


def full_reval(book: list[Position], S: float, dS: float, dvol: float, r=0.045, q=0.013) -> float:
    return float(sum(p.value(S * (1 + dS), r, q, dvol) - p.value(S, r, q) for p in book))


def taylor_pnl(book: list[Position], S: float, dS: float, dvol: float, r=0.045, q=0.013) -> float:
    """Δ·ΔS + ½Γ·ΔS² + vega·Δσ, summed over the book with quantities and multipliers (raw units)."""
    tot = 0.0
    for p in book:
        m = p.qty * p.multiplier
        if p.kind == "stock":
            tot += m * S * dS
            continue
        g = greeks(S, p.K, p.T, r, q, p.iv, p.cp)
        x = S * dS
        tot += m * (g["delta"] * x + 0.5 * g["gamma"] * x ** 2 + g["vega"] * dvol)
    return float(tot)


def delta_hedge_pnl(S0=100.0, K=100.0, T=30 / 365, r=0.0, iv=0.20, rv=0.20, n_paths: int = 2000, steps: int = 30,
                    cost_bps: float = 0.0, seed: int = 0) -> np.ndarray:
    """P&L of BUYING one option at implied vol `iv` and delta-hedging it `steps` times until expiry while the stock
    moves with realized vol `rv`. Hedging costs cost_bps of traded notional."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    S = np.full(n_paths, S0)
    opt0 = bsm_price(S0, K, T, r, 0.0, iv, 1)
    delta = greeks(S0, K, T, r, 0.0, iv, 1)["delta"] * np.ones(n_paths)
    cash = -opt0 + delta * S0 - cost_bps * 1e-4 * np.abs(delta) * S0
    for i in range(1, steps + 1):
        S = S * np.exp(-0.5 * rv ** 2 * dt + rv * np.sqrt(dt) * rng.standard_normal(n_paths))
        cash *= np.exp(r * dt)
        tau = T - i * dt
        new = greeks(S, K, tau, r, 0.0, iv, 1)["delta"] if tau > 1e-12 else (S > K).astype(float)
        cash += (new - delta) * S - cost_bps * 1e-4 * np.abs(new - delta) * S
        delta = new
    return np.maximum(S - K, 0) - delta * S + cash
