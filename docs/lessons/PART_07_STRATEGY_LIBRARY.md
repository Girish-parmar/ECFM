# Part 7 — Strategy Library: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 2, **Month 6, second half** (program weeks 23–24), right after Part 6 |
| Format | 8 sessions × **120 min** (4 per week) + 2 lab clinics × 120 min + self-study (~7 hrs/week) |
| Total effort | ~16 hrs live + 4 hrs clinic + 14 hrs self-study ≈ **34 hours** |
| Instruments | US equities & ETFs (IB, Alpaca), CME index futures (IB), SPY/SPX/QQQ options (IB, Alpaca) |
| Platform milestone | **M3b**: `strategy/base.py`, `strategy/registry.py`, YAML strategy configs, `strategy/library/*` (linear), `options/strategies/*` (option builder + templates) |
| Covers original items | "Advance with Python" 17–28 · Platform roadmap 27 (strategy creator foundation), 33–39 (option strategy creators) |

**Scope note:** Part 7 teaches *what* each strategy is, *why* it might work, and *how* to code it correctly. Signals are checked with a simple first-look evaluator. Rigorous backtesting, optimization, walk-forward validation and risk sizing are Part 8. **No strategy in this library is presented as profitable**; each is a hypothesis to be tested.

---

## 1. Learning Objectives

By the end of Part 7 the learner will be able to:

1. **Specify** any strategy with one standard template: hypothesis → regime → signal → entry → exit → sizing → costs → failure modes → expected metrics.
2. **Implement** strategies on one `Strategy` base class (Template Method) that runs unchanged in research, backtest and live, configured by YAML.
3. **Build** at least one strategy from each of the 7 linear groups: directional momentum, range-bound, mean reversion, either-way, volatility, mathematical, statistical.
4. **Build** multi-leg option strategies with a Builder, and compute payoff, breakevens, max profit/loss, probability of profit and net Greeks.
5. **Choose** option structures from the view (direction, volatility, time) and the volatility regime (IV rank, term structure, skew).
6. **Design** portfolio hedges with options and futures, and quantify cost vs drawdown protection.
7. **Diagnose** in which market regime each strategy should work or fail, and show it with data.

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| 1.6 F&O strategies (payoff diagrams) | S5–S8 |
| 2.3 Time series (stationarity, Hurst, Kalman, cointegration) | S3, S4 |
| 3.4 LLD (Template Method, Builder, Registry, Specification) | S1, S5 |
| Part 4 OMS & kill switch (orders, brackets, combo orders) | S1, S5 |
| Part 5 indicators, patterns, actions, `Condition` | S1–S4 |
| Part 6 pricing, Greeks, surface, strike selection, portfolio Greeks | S5–S8 |

---

## 3. Two-Week Overview

| Week | Theme | Sessions | Clinic Lab | Platform Output |
|---|---|---|---|---|
| **W1** | Framework + linear strategy groups | S1 Framework & spec template · S2 Directional momentum · S3 Mean reversion & range-bound · S4 Either-way, volatility, mathematical, statistical | Regime map: 7 strategies × 4 market regimes | `strategy/base.py`, `strategy/registry.py`, `strategy/library/*.py`, `research/quick_eval.py` |
| **W2** | Option strategy groups + hedging | S5 Option strategy builder · S6 Directional & mean-reversion options · S7 Range-bound, either-way & volatility options · S8 Hedging & M3b release | Option playbook: pick, build, analyze and paper-trade 3 structures | `options/strategies/builder.py`, templates, `portfolio/hedging.py`, **M3b release** |

---

## 4. The Strategy Specification Template (used for every strategy)

| Section | Question it answers |
|---|---|
| 1. Hypothesis | What inefficiency or risk premium is being harvested? Who is on the other side, and why do they accept it? |
| 2. Regime | Trending / ranging, high / low volatility, risk-on / risk-off: where should it work, where should it fail? |
| 3. Universe & data | Instruments, bar size, history, point-in-time requirements |
| 4. Signal | Exact formulas and parameters (with ranges for the Part 8 optimizer) |
| 5. Entry | Order type, timing (always the bar *after* the signal), filters |
| 6. Exit | Target, stop, trailing, time exit, signal reversal |
| 7. Sizing | Placeholder rule (fixed risk or volatility target); refined in Part 8 |
| 8. Costs | Commission, spread, slippage, borrow, option leg costs |
| 9. Failure modes | Known ways it loses (crowding, regime change, gap risk, short-vol blow-ups) |
| 10. Expected metrics | Expected trade frequency, win rate, payoff ratio, holding period, capacity |

Every strategy ships as: spec (Markdown) + class (Python) + config (YAML) + first-look evaluation (notebook).

---

## 5. Strategy Catalog (target for M3b)

**7A — Linear strategy groups**

| Group | Strategies in the library | Core idea / reference |
|---|---|---|
| Directional momentum | Time-series momentum (vol-targeted), Donchian/Turtle breakout, dual MA trend, dual momentum, cross-sectional 12-1 momentum | Trends persist (Moskowitz, Ooi & Pedersen 2012; Jegadeesh & Titman 1993; Antonacci 2014) |
| Range-bound | Bollinger fade with ADX/Choppiness filter, RSI band, S/R fade (Part 5 KDE levels), grid (educational, with risk warning) | Prices oscillate inside ranges when trend strength is low |
| Mean reversion | Z-score reversion, RSI(2) with trend filter, IBS, short-term (1-week) reversal, gap fade, intraday VWAP reversion | Liquidity provision: short-term overreaction is paid for |
| Either-way | Opening-range breakout (ORB), volatility-squeeze breakout, event breakout (earnings/FOMC) | Volatility expansion after compression; direction decided by the break |
| Volatility | Volatility targeting overlay, ATR channel, VIX-regime switching (VIX/VIX3M), volatility breakout | Risk-adjusting exposure improves risk-adjusted returns (Moreira & Muir 2017) |
| Mathematical | Kalman-filter trend, Hurst-regime switch (trend vs revert), Fourier/wavelet cycle, entropy filter | Model-based signal extraction from noise |
| Statistical | Pairs trading (cointegration), sector-ETF baskets, turn-of-month seasonality, overnight vs intraday returns | Statistical regularities; full stat-arb in Part 9 |

**7B — Option strategy groups**

| Group | Structures in the library | View | Prefer when |
|---|---|---|---|
| Directional momentum | Long call/put, bull call / bear put debit spreads, call/put back spreads, risk reversal | Direction, often with vol expansion | IV rank low; strong trend signal |
| Range-bound | Iron condor, iron butterfly, defined-risk short strangle, calendar | No big move; time decay | IV rank high; low trend strength |
| Mean reversion | Short put / call credit spreads at extremes, ratio spreads, skew trades | Price returns toward the mean | Oversold/overbought signal + rich IV |
| Hedging | Protective put, collar, put spread, put-spread collar, VIX calls, tail ladders, futures overlay | Protect a portfolio | Always sized as insurance, not as a trade |
| Either-way | Long straddle/strangle, long iron condor, event straddle | Big move, direction unknown | Implied move < expected realized move |
| Volatility *(added)* | Variance-risk-premium harvesting, term-structure calendars, gamma scalping, dispersion (intro) | Trade vol itself | IV vs forecast RV gap; term-structure shape |

---

## 6. Session-by-Session Plan

> Each 120-min session: **25 min theory → 60 min live coding → 25 min guided lab → 10 min wrap-up and homework.**
> Notebooks live in `notebooks/part07/`; promoted code lives in `quantforge/strategy/` and `quantforge/options/strategies/`. Offline, auto-graded exercises for every session and both clinics are in [`labs/part07/`](../../labs/part07/).

### Week 1 — Framework & Linear Strategy Groups

#### S1 · Strategy Framework, Spec Template & First-Look Evaluator

| Block | Content |
|---|---|
| Theory | **One strategy class, three runtimes** (research, backtest, live): strategies emit *order intents* and never call a broker directly; the OMS and risk engine (Parts 4, 8) decide what is actually sent. **Template Method:** the base class owns the lifecycle (`on_start`, `on_bar`, `on_fill`, `on_stop`), subclasses fill in the logic. **Registry + YAML config** so new strategies need no core edits. **Regime taxonomy** used all Part: trend strength (ADX, Hurst) × volatility level (VIX percentile or realized vol). **Timing rule:** a signal computed at bar *t*'s close is acted on at bar *t+1*'s open. The first-look evaluator: fast, honest (next-bar fills, costs), deliberately simple; the real backtester is Part 8. |
| Live coding | `Strategy` base class, registry, config loader, `quick_eval` (below). |
| Lab | Port a Part 5 `Condition`-based rule into a registered strategy with a YAML config; evaluate it on 10 ETFs. |
| Homework | Write the full spec (Section 4) for one strategy of your choice. |

```python
# quantforge/strategy/base.py
from abc import ABC, abstractmethod

class Strategy(ABC):
    """Template Method: the engine calls these hooks; subclasses implement the logic."""
    name: str = "base"
    params: dict = {}

    def __init__(self, ctx: "StrategyContext", **params):
        self.ctx = ctx                                  # data access, clock, order-intent sink, logger
        self.p = {**self.params, **params}

    def on_start(self) -> None: ...
    @abstractmethod
    def on_bar(self, bar: "Bar") -> None: ...
    def on_fill(self, fill: "Fill") -> None: ...
    def on_stop(self) -> None: ...

    def target(self, instrument, weight: float, reason: str) -> None:
        """Emit an order intent. Risk engine + OMS decide what is actually sent."""
        self.ctx.intents.submit(instrument, weight, strategy=self.name, reason=reason)
```

```yaml
# configs/strategies/tsmom_etf.yaml
strategy: tsmom
universe: [SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, UUP]
timeframe: 1d
params: {lookback: 252, vol_n: 60, target_vol: 0.10, max_lev: 2.0}
broker: ib          # ib | alpaca | sim
```

```python
# quantforge/research/quick_eval.py
def quick_eval(signal, open_, cost_bps=2.0, periods_per_year=252):
    """First-look evaluator (the real backtester comes in Part 8).
    signal[t] = target position decided at the CLOSE of bar t (-1..+1),
    filled at the OPEN of bar t+1 and held until the next open."""
    pos = np.nan_to_num(np.roll(signal, 1)); pos[0] = 0.0                # act on the next bar
    ret = np.zeros_like(open_); ret[:-1] = open_[1:] / open_[:-1] - 1    # open-to-open returns
    turnover = np.abs(np.diff(pos, prepend=0.0))
    pnl = pos * ret - turnover * cost_bps / 1e4
    equity = np.cumprod(1 + pnl)
    dd = equity / np.maximum.accumulate(equity) - 1
    sharpe = pnl.mean() / pnl.std() * np.sqrt(periods_per_year) if pnl.std() > 0 else np.nan
    return {"cagr": equity[-1] ** (periods_per_year / len(pnl)) - 1, "sharpe": sharpe,
            "max_dd": dd.min(), "exposure": np.mean(pos != 0),
            "turnover": turnover.sum() / len(pnl) * periods_per_year}
```

#### S2 · Directional Momentum Group

| Block | Content |
|---|---|
| Theory | Evidence and explanations for momentum: under-reaction, herding, risk premia, slow-moving capital. **Time-series momentum (TSMOM):** position = sign of past 12-month return, scaled to a target volatility (volatility scaling is most of the benefit). **Breakout systems:** Donchian channels, the Turtle rules (N = ATR-based unit sizing, 20/55-day breakouts, 2N stop). **Dual MA trend.** **Dual momentum:** relative (best asset) + absolute (above cash). **Cross-sectional momentum:** rank stocks by 12-month return skipping the last month (12-1), long top decile; crash risk after market rebounds (2009). Failure modes: choppy markets, sharp reversals, crowding. |
| Live coding | TSMOM (below), Donchian breakout, 12-1 cross-sectional ranking on a 500-stock universe. |
| Lab | TSMOM on 10 ETFs (config from S1): with and without volatility scaling; compare Sharpe and max drawdown. |
| Homework | Turtle system with ATR unit sizing and pyramiding; write its spec. |

```python
def tsmom(close, lookback=252, vol_n=60, target_vol=0.10, periods_per_year=252, max_lev=2.0):
    """Time-series momentum: sign of past return, scaled to a target volatility."""
    past = np.full_like(close, np.nan)
    past[lookback:] = close[lookback:] / close[:-lookback] - 1
    r = np.full_like(close, np.nan)
    r[1:] = np.diff(np.log(close))
    vol = pd.Series(r).rolling(vol_n).std().to_numpy() * np.sqrt(periods_per_year)
    return np.clip(np.sign(past) * target_vol / vol, -max_lev, max_lev)

def cross_sectional_momentum(closes: pd.DataFrame, top_q=0.1):
    """closes: dates x symbols (month-end). Rank on 12-1 momentum; equal-weight top decile."""
    mom = closes.shift(1) / closes.shift(12) - 1          # skip the most recent month
    rank = mom.rank(axis=1, pct=True)
    w = (rank >= 1 - top_q).astype(float)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)
```

#### S3 · Mean Reversion & Range-Bound Groups

| Block | Content |
|---|---|
| Theory | Why short-term reversion exists: liquidity provision, overreaction, inventory effects in market making. It often works at short horizons (days) and in index ETFs; momentum works at longer horizons (months). **Strategies:** z-score of price vs moving average, **RSI(2)** with a 200-day trend filter, **IBS** (internal bar strength `(C − L)/(H − L)`), 1-week reversal, gap fade, intraday **VWAP reversion**. **Range-bound:** Bollinger fade only when ADX is low or Choppiness is high; S/R fades using Part 5 KDE levels. **Grid trading:** taught as a cautionary example (unbounded inventory in trends). Failure modes: "catching a falling knife", regime shifts, gap risk, stops placed too tight. |
| Live coding | RSI(2) reversion (below), IBS, regime filter from ADX. |
| Lab | Compare RSI(2) with and without the trend filter on SPY, QQQ, IWM; show trade-level win rate vs payoff ratio. |
| Homework | Intraday VWAP reversion on 5-minute bars with a time-of-day filter; spec + first-look evaluation. |

```python
def rsi2_reversion(close, entry=10, exit_=70, trend_n=200):
    """Connors-style: long when RSI(2) < entry in an uptrend (close > SMA200); exit when RSI(2) > exit_."""
    r, trend = rsi(close, 2), close > sma(close, trend_n)
    pos = np.zeros_like(close)
    for t in range(1, close.size):
        if pos[t - 1] == 0 and trend[t] and r[t] < entry:
            pos[t] = 1.0
        elif pos[t - 1] == 1 and r[t] > exit_:
            pos[t] = 0.0
        else:
            pos[t] = pos[t - 1]
    return pos

def ibs(high, low, close):
    """Internal bar strength in [0, 1]; low values tend to precede short-term bounces in index ETFs."""
    rng = high - low
    return np.divide(close - low, rng, out=np.full_like(close, 0.5), where=rng > 0)
```

#### S4 · Either-Way, Volatility, Mathematical & Statistical Groups

| Block | Content |
|---|---|
| Theory | **Either-way:** opening-range breakout (first 15/30 min high/low, stop at the other side, exit at close), volatility-squeeze breakout (Bollinger inside Keltner), event breakouts. **Volatility:** volatility-targeting overlay (scale any strategy to constant risk), VIX/VIX3M regime switch (reduce equity exposure when the term structure inverts). **Mathematical:** local-level **Kalman filter** trend (the ratio `q/r` controls smoothness), Hurst-exponent regime switch between trend and reversion rules, cycle extraction. **Statistical:** pairs trading via cointegration (preview of Part 9), turn-of-month seasonality, overnight vs intraday return decomposition. |
| Live coding | ORB on 5-minute SPY bars, Kalman trend (below), VIX-regime overlay. |
| Lab | Regime table: for 7 strategies (one per group), first-look Sharpe in each of 4 regimes (trend/range × high/low vol). |
| Homework | Pairs trade on two cointegrated ETFs (e.g. an index and its equal-weight version): hedge ratio from OLS, z-score entry/exit; spec it. |

```python
def kalman_trend(price, q=1e-5, r=1e-2):
    """Local-level Kalman filter. Larger q/r = faster, noisier level. Signal = sign of the level's slope."""
    x, p = price[0], 1.0
    level = np.empty_like(price)
    for t, z in enumerate(price):
        p += q                     # predict
        k = p / (p + r)            # Kalman gain
        x += k * (z - x)           # update with the new observation
        p *= (1 - k)
        level[t] = x
    return level

def opening_range_breakout(bars: pd.DataFrame, minutes=30):
    """bars: one session of 5-min bars (UTC index, open time). Returns +1/-1 entry on the first break."""
    session_open = bars.index[0]
    rng = bars[bars.index < session_open + pd.Timedelta(minutes=minutes)]
    hi, lo = rng["high"].max(), rng["low"].min()
    after = bars[bars.index >= session_open + pd.Timedelta(minutes=minutes)]
    for ts, b in after.iterrows():
        if b["close"] > hi:
            return ts, +1, lo            # entry time, direction, stop
        if b["close"] < lo:
            return ts, -1, hi
    return None
```

**Clinic W1 (120 min):** the regime map. Each learner presents one strategy: spec, code, first-look metrics, and the regime table showing where it works and where it fails.

---

### Week 2 — Option Strategy Groups & Hedging

#### S5 · Option Strategy Builder (items 33, 30)

| Block | Content |
|---|---|
| Theory | Any option strategy is a list of legs (call/put/underlying, strike, expiry, quantity). **Builder pattern** to construct them; templates as factory functions (`iron_condor(chain, short_delta=0.16, wing=5)`). **Analysis:** payoff at expiry, P&L today and at future dates (BSM with the Part 6 engine), breakevens, max profit/loss, **probability of profit** under a lognormal distribution (a model estimate, not a promise), net Greeks, margin/buying power (IB `whatIfOrder`). **Execution:** multi-leg orders as a single combo so legs cannot fill separately: IB `Contract(secType="BAG")` with `ComboLeg`s; Alpaca multi-leg options orders (`OrderClass.MLEG`; confirm your `alpaca-py` version supports it). Price combos at the net mid and improve in small steps. |
| Live coding | `OptionStrategy` builder (below); payoff and "P&L today / at half-life / at expiry" plot. |
| Lab | Build and analyze 6 structures on SPY (long call, bull call spread, iron condor, long straddle, calendar, collar); submit one as a combo order on paper. |
| Homework | Template factories that select strikes from a live chain by delta (Part 6 `strike_by_delta`). |

```python
from dataclasses import dataclass, field
from scipy.stats import norm

@dataclass(frozen=True)
class Leg:
    cp: int          # +1 call, -1 put, 0 underlying
    K: float
    T: float         # years to expiry
    qty: int         # + long / - short, in contracts (underlying: 1 = one multiplier of shares)
    iv: float = 0.2

@dataclass
class OptionStrategy:
    name: str
    legs: list[Leg] = field(default_factory=list)
    multiplier: int = 100

    def add(self, cp, K, T, qty, iv=0.2):               # Builder: chainable
        self.legs.append(Leg(cp, K, T, qty, iv))
        return self

    def value(self, S, r=0.04, q=0.0, dt=0.0, dvol=0.0):
        """Mark-to-model value after `dt` years pass and IV shifts by `dvol`."""
        v = 0.0
        for L in self.legs:
            if L.cp == 0:
                v += L.qty * S
                continue
            tau = L.T - dt
            v += L.qty * (bsm_price(S, L.K, tau, r, q, L.iv + dvol, L.cp) if tau > 1e-9
                          else np.maximum(L.cp * (S - L.K), 0.0))
        return v * self.multiplier

    def payoff_at_expiry(self, S):
        return self.value(S, dt=min(L.T for L in self.legs if L.cp != 0))

    def greeks(self, S, r=0.04, q=0.0):
        tot = dict.fromkeys(("delta", "gamma", "vega", "theta"), 0.0)
        for L in self.legs:
            if L.cp == 0:
                tot["delta"] += L.qty
                continue
            g = bsm_greeks(S, L.K, L.T, r, q, L.iv, L.cp)
            for k in tot:
                tot[k] += L.qty * g[k]
        return {k: v * self.multiplier for k, v in tot.items()}

    def analyze(self, S, r=0.04, q=0.0, sigma=None, grid=None):
        """Breakevens, max profit/loss (within the price grid) and lognormal probability of profit."""
        grid = np.linspace(0.5 * S, 1.5 * S, 2001) if grid is None else grid
        cost = self.value(S, r, q)
        pnl = self.payoff_at_expiry(grid) - cost
        sign = np.sign(pnl)
        T0 = min(L.T for L in self.legs if L.cp != 0)
        sigma = sigma or np.mean([L.iv for L in self.legs if L.cp != 0])
        z = (np.log(grid / S) - (r - q - 0.5 * sigma**2) * T0) / (sigma * np.sqrt(T0))
        density = norm.pdf(z) / (grid * sigma * np.sqrt(T0))
        return {"cost": cost, "breakevens": grid[1:][sign[1:] != sign[:-1]].round(2),
                "max_profit": pnl.max(), "max_loss": pnl.min(),
                "pop": np.trapezoid(density * (pnl > 0), grid)}

T = 30 / 365
condor = (OptionStrategy("iron condor")
          .add(-1, 90, T, +1, 0.24).add(-1, 95, T, -1, 0.22)
          .add(+1, 105, T, -1, 0.19).add(+1, 110, T, +1, 0.18))
condor.analyze(S=100)   # credit ≈ $105, breakevens ≈ 94 / 106, max loss ≈ -$395, POP ≈ 69%
```

#### S6 · Directional & Mean-Reversion Option Strategies (items 34, 35)

| Block | Content |
|---|---|
| Theory | Matching structure to view: **direction × volatility × time.** **IV rank / IV percentile** decides whether to buy premium (low IV rank: long options, debit spreads, back spreads) or sell it (high IV rank: credit spreads). Bullish: long call, bull call spread, bull put (credit) spread, call back spread, risk reversal (short put + long call). Bearish: mirror images. **Mean-reversion options:** credit spreads placed at statistical extremes (Part 5 signals: RSI(2), Bollinger %B), ratio spreads (with their tail risk). Strike and DTE choices (delta-based), skew effects on pricing. **Management rules as testable hypotheses:** take profit at 50% of max credit, stop at 2× credit, close at 21 DTE; these are popular rules, not proven ones; Part 8 tests them. |
| Live coding | Signal → structure mapper: a directional signal plus IV rank selects and builds the structure from a live chain. |
| Lab | For 3 historical signal dates on QQQ, build the chosen structures from recorded chain snapshots and track their P&L to expiry. |
| Homework | Compare a long call vs a bull call spread vs long shares for the same signal: P&L, capital used, sensitivity to IV change. |

#### S7 · Range-Bound, Either-Way & Volatility Option Strategies (items 36, 37, 38)

| Block | Content |
|---|---|
| Theory | **Range-bound:** iron condor (16-delta shorts, wings for defined risk), iron butterfly, defined-risk strangles, calendars (long vega, short gamma). **Either-way:** long straddle/strangle, long iron condor; **event trades:** compare the implied earnings move (ATM straddle ≈ 0.8 × S × σ × √T, or straddle price / S) with historical realized earnings moves. **Volatility group:** the **variance risk premium** (implied vol has historically exceeded later realized vol on average, which is why selling options is rewarded most of the time and punished severely in crashes); term-structure calendars; **gamma scalping** (long options + delta hedging profits when realized > implied, from Part 6 S8); dispersion (index vs components, overview). **Sentiment & parameter-driven selection (item 38):** VIX regime, skew, term structure and news sentiment (Part 5) as structure selectors. Short-volatility failure modes: Feb 2018, Mar 2020. |
| Live coding | VRP study (below-style): SPX 30-day IV vs subsequent 30-day realized vol; implied earnings move vs realized for 20 stocks. |
| Lab | Iron condor vs long strangle on SPY over recorded snapshots in low-, mid- and high-VIX regimes; P&L distributions and worst cases. |
| Homework | A structure selector: inputs (direction signal, IV rank, term-structure slope, event flag) → recommended template + parameters, with documented rules. |

```python
def implied_move(straddle_price, spot):
    """Approximate expected absolute move to expiry implied by the ATM straddle."""
    return straddle_price / spot

def variance_risk_premium(iv_30d: pd.Series, close: pd.Series, window=21):
    """Implied vol today minus the realized vol over the NEXT `window` trading days (research only:
    uses future data by design, so it must never be used as a trading signal)."""
    rv_future = np.log(close).diff().rolling(window).std().shift(-window) * np.sqrt(252)
    return iv_30d - rv_future
```

#### S8 · Hedging Strategies & M3b Release (item 39)

| Block | Content |
|---|---|
| Theory | Hedging is insurance: judge it by **cost vs drawdown reduction**, not by P&L on its own. **Option hedges:** protective puts (expensive), collars (finance the put with a call), put spreads, put-spread collars, VIX calls (convex but with basis risk), tail ladders. **Futures overlay:** hedge beta exposure with index futures; contracts = hedge ratio × β × portfolio value / (futures price × multiplier); micro futures (MES) for small accounts. Hedge timing: always-on vs signal-based (Part 5 sentiment/tail indicators). Delta-hedging an options book with the underlying or futures. Roll cost and theta budget (e.g. ≤ 1–2% of portfolio per year). |
| Live coding | Futures hedge calculator (below); collar builder; hedged vs unhedged portfolio through the Part 5 crash library. |
| Lab | For a sample $1M equity portfolio (β = 1.1): compare no hedge, rolling 3-month 10% OTM puts, collars and a 50% futures overlay through 2018, 2020 and 2022. |
| Homework | Final assessment (Section 9). |

```python
def futures_hedge_contracts(portfolio_value, beta, fut_price, multiplier, hedge_ratio=1.0):
    """Number of index futures to SELL to neutralize `hedge_ratio` of the portfolio's beta exposure."""
    return round(hedge_ratio * beta * portfolio_value / (fut_price * multiplier))

futures_hedge_contracts(1_000_000, beta=1.1, fut_price=6000, multiplier=50)   # -> 4 ES contracts
```

**Clinic W2:** option playbook. Each learner picks 3 structures (one each for direction, range and hedge), builds them from a live chain, analyzes them (payoff, POP, Greeks, scenario grid) and places them as combo orders on a paper account.

---

## 7. Notebook Map (`notebooks/part07/`)

> Offline **guided** versions of these eight notebooks, with self-checking exercises on synthetic data with known regimes, are in [`notebooks/part07/`](../../notebooks/part07/) (instructor solutions in `notebooks/part07/solutions/`), one per session: `01_framework_quick_eval`, `02_momentum`, `03_mean_reversion`, `04_vol_math_stat`, `05_option_builder`, `06_directional_options`, `07_range_event_vol`, `08_hedging`.

| Notebook | Session | Promoted to |
|---|---|---|
| `01_framework_quick_eval.ipynb` | S1 | `strategy/base.py`, `strategy/registry.py`, `research/quick_eval.py` |
| `02_momentum.ipynb` | S2 | `strategy/library/momentum.py` |
| `03_mean_reversion_range.ipynb` | S3 | `strategy/library/mean_reversion.py`, `strategy/library/range_bound.py` |
| `04_either_vol_math_stat.ipynb` | S4 | `strategy/library/{either_way,volatility,mathematical,statistical}.py` |
| `05_option_builder.ipynb` | S5 | `options/strategies/builder.py`, `options/strategies/templates.py` |
| `06_directional_options.ipynb` | S6 | `options/strategies/{bullish,bearish}.py`, `options/strategies/selector.py` |
| `07_range_event_vol_options.ipynb` | S7 | `options/strategies/{sideways,either_way,sentiment}.py` |
| `08_hedging.ipynb` | S8 | `options/strategies/hedging.py`, `portfolio/hedging.py` |

---

## 8. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Acting on the signal bar's close | Unrealistically good results | Next-bar-open rule in `quick_eval` and in the base class |
| 2 | Strategy calls the broker directly | Bypasses risk engine and kill switch | Strategies only emit intents; code review rule |
| 3 | Parameters hard-coded | Cannot optimize or audit | YAML config + declared parameter ranges |
| 4 | Treating first-look results as validation | Overfit strategies reach live trading | First-look ≠ backtest; Part 8 gate required |
| 5 | Momentum without volatility scaling | Huge drawdowns in volatile periods | Volatility-target overlay |
| 6 | Mean reversion without a regime filter | Buying into crashes | Trend/volatility filters; max holding period |
| 7 | Selling premium sized by POP | Rare, catastrophic losses | Size by max loss / stress loss, not win rate |
| 8 | Legging into spreads | Unintended naked exposure | Combo (BAG / multi-leg) orders only |
| 9 | Ignoring bid-ask on 4-leg structures | Costs eat the whole edge | Model each leg's half-spread; limit orders at net mid |
| 10 | Undefined-risk short options in the library | Unbounded loss | Library templates default to defined risk |
| 11 | Using future data in research features (e.g. VRP) as signals | Look-ahead | Clearly separated research-only functions; truncation tests |
| 12 | Hedge judged by its own P&L | Hedges removed right before crashes | Evaluate portfolio-level drawdown and cost budget |
| 13 | Early assignment ignored on short American options | Surprise stock positions | Check ITM short legs before ex-dividend dates |
| 14 | Pin risk at expiry | Unexpected positions after expiry | Close or roll short legs before expiry day |

---

## 9. Assessment — Platform Milestone M3b

**Task:** ship `quantforge.strategy` and `quantforge.options.strategies` v0.4 plus a strategy book.

1. Framework: `Strategy` base, registry, YAML configs; strategies emit intents only.
2. **Linear library:** ≥ 1 strategy per group (7 total), each with spec, code, config, first-look evaluation and a regime table.
3. **Option library:** builder + ≥ 10 templates across the 6 option groups, each with payoff, breakevens, POP, Greeks and a scenario grid.
4. **Structure selector** mapping (signal, IV rank, term structure, event flag) → template.
5. **Hedging study** for the sample portfolio across 3 crash periods.
6. Paper-trade evidence: at least one linear strategy running on paper via the Part 4 OMS, and one combo option order filled on paper.

| Criterion | Points |
|---|---|
| Framework design (Template Method, registry, config, intent-only) | 15 |
| Linear strategies: correctness, specs, regime analysis | 25 |
| Option builder & templates: analysis correctness, defined risk | 20 |
| Structure selector & volatility-regime logic | 10 |
| Hedging study (cost vs drawdown, honest conclusions) | 15 |
| Paper-trading evidence via the platform | 5 |
| Code quality, tests, documentation | 10 |
| **Total** | **100** |

Pass mark: 70, **and** every strategy must pass the next-bar-execution test (no same-bar fills) (mandatory).

---

## 10. Further Reading

| Type | Reference |
|---|---|
| Paper | Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). "Time Series Momentum." *JFE*, 104(2). |
| Paper | Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*, 48(1). |
| Paper | Daniel, K. & Moskowitz, T. (2016). "Momentum Crashes." *JFE*, 122(2). |
| Paper | Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios." *Journal of Finance*, 72(4). |
| Paper | Carr, P. & Wu, L. (2009). "Variance Risk Premiums." *RFS*, 22(3). |
| Paper | Gatev, E., Goetzmann, W. & Rouwenhorst, K. G. (2006). "Pairs Trading." *RFS*, 19(3). |
| Book | Antonacci, G. (2014). *Dual Momentum Investing*. McGraw-Hill. |
| Book | Clenow, A. (2013). *Following the Trend*. Wiley. |
| Book | Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley. |
| Book | Connors, L. & Alvarez, C. (2009). *Short Term Trading Strategies That Work*. TradingMarkets. |
| Book | Faith, C. (2007). *Way of the Turtle*. McGraw-Hill. |
| Book | Sinclair, E. (2020). *Positional Option Trading*. Wiley. |
| Book | Natenberg, S. (2014). *Option Volatility and Pricing* (2nd ed.). McGraw-Hill. |
| Docs | IB TWS API: combo (`BAG`) orders, `whatIfOrder`; Alpaca options trading (multi-leg orders) |

---

## 11. Instructor Notes

- Repeat often: every strategy here is a **hypothesis**. Published anomalies decay after publication; costs and capacity matter more than the idea.
- Use recorded option chain snapshots (Parquet) for S6–S7 labs so everyone works on the same data and labs do not depend on market hours.
- Short-premium structures are always introduced together with their worst historical week. Show the Feb 2018 and Mar 2020 P&L before the win rates.
- The first-look evaluator is intentionally minimal; resist adding features to it. Its limits motivate Part 8.
- No live-money trading in Part 7. Paper only, through the Part 4 OMS with the kill switch armed.
