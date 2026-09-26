# Part 12 — Complete Trading Platform: LLD Capstone & Production: Detailed Lesson Plan

| Item | Detail |
|---|---|
| Program | Master in Financial Analysis and Algorithmic Trading (MFAAT) |
| Placement | Term 4, **Months 11–12** (program weeks 41–48) |
| Format | Weeks 41–45: 20 sessions × 90 min (4 per week) + 5 lab clinics × 120 min. Weeks 45–48: **4-week live/paper track record** with weekly review sessions (S21–S23) and the capstone defense (S24) |
| Total effort | ~36 hrs live + 10 hrs clinic + ~60 hrs capstone build & operations ≈ **106 hours** |
| Brokers | IB (TWS/IB Gateway) and Alpaca: paper for everyone; small-size live for learners who pass the go-live review |
| Platform milestones | **M8** (end of week 43): strategy creator, screeners, spread-aware OMS & execution algos, option strategy integration, trade journal, monitoring department, dashboards · **M9** (end of week 45): containerized deployment, CI/CD, anomaly detection, runbooks, go-live review · **Capstone** (week 48) |
| Covers original items | "Create Complete Trading Platform" items 1–54 (integration of everything built since Part 4; items first built here: 2, 7, 27–29, 31–32, 51–54), design in [03_PLATFORM_LLD_ROADMAP.md](../03_PLATFORM_LLD_ROADMAP.md) |

**Where this fits:** learners have been building `quantforge` since Month 3. This Part completes it, hardens it for production, and runs it for four weeks under real operating discipline. The deliverable is a platform a professional could trust: every order passes the risk engine, every action is audited, every failure has a runbook, and results reconcile between backtest, paper and live.

---

## 1. Learning Objectives

By the end of Part 12 the learner will be able to:

1. **Audit** the platform against the LLD (layers, interfaces, patterns, tests) and close the gaps.
2. **Engineer for performance:** profile, set and meet a latency budget, manage memory, and keep the event loop responsive.
3. **Create strategies from configuration** using library functions (no new code for common strategies), including option strategies, and select instruments with screeners.
4. **Execute well:** spread-aware limit placement, re-pricing, TWAP/VWAP/POV algorithms, and implementation-shortfall measurement.
5. **Run a monitoring department:** observability (logs, metrics, dashboards), reconciliation, hash-chained audit, approval workflows, anomaly detection with automatic pause, and reports.
6. **Deploy and operate:** Docker Compose stack, CI/CD with backtest regression tests, secrets, backups, disaster recovery, runbooks and incident response.
7. **Go live safely:** paper → shadow → small live with capital-ramp rules, daily operating procedures, and a reconciled 4-week track record.
8. **Present** a complete, defensible trading system at the capstone defense.

---

## 2. Prerequisites

| From | Needed for |
|---|---|
| Part 4 brokers, OMS, kill switch, connection management | W1–W2, W4–W5 |
| Part 5 indicator registry, `Condition`, streaming indicators | S3, S4 |
| Part 6–7 options engine, strategy framework, option builder | S3, S7 |
| Part 8 backtester, validation gate, risk engine, sizing, portfolio | All weeks |
| Part 9–10 stat-arb, ML models, drift monitoring | S4, S11, capstone |
| Part 11 audit log for AI, n8n workflows, signed webhooks | S10, S12, S19 |
| 3.7 Software craft (Docker, CI, testing) | W4 |

---

## 3. Eight-Week Overview

| Week | Theme | Sessions | Clinic / Activity | Platform Output |
|---|---|---|---|---|
| **W1** (41) | Integration & performance | S1 Platform audit vs LLD · S2 Performance engineering · S3 Strategy creator · S4 Screeners | Gap-closing sprint; latency budget met | `strategy/creator.py`, `screener/`, `core/metrics.py` |
| **W2** (42) | Execution & trade management | S5 Spread-aware OMS · S6 Execution algorithms · S7 Option strategy integration · S8 Trade journal | Execution-quality report on paper | `execution/spread_aware.py`, `execution/algos.py`, `options/strategies/manager.py`, `analytics/journal_analysis.py` |
| **W3** (43) | Monitoring department | S9 Observability · S10 Reconciliation, audit & approvals · S11 Anomaly detection & controller · S12 Reports & dashboards | Grafana + Streamlit dashboards live; **M8 review** | `monitoring/*`, `apps/dashboard.py` |
| **W4** (44) | Production engineering | S13 Deployment with Docker Compose · S14 CI/CD & release process · S15 Reliability, DR & chaos · S16 Security & operational risk | Full stack deployed on a VPS; recovery drill | `deploy/`, `.github/workflows/`, `runbooks/` |
| **W5** (45) | Go-live | S17 Rollout & capital ramp · S18 Daily operations · S19 Incident response drills · S20 Capstone kickoff | **M9 go-live review**; track record starts | `ops/checklists/`, capstone plan |
| **W6** (46) | Track record week 1–2 | S21 Weekly review: performance attribution & reconciliation | Operating the platform | Weekly report |
| **W7** (47) | Track record week 3 | S22 Weekly review: execution quality, risk, incidents | Operating the platform | Weekly report |
| **W8** (48) | Track record week 4 & defense | S23 Final review & career track · S24 Capstone defense (demo day) | Defense | Capstone dossier |

---

## 4. Session-by-Session Plan

> Sessions S1–S20: **15 min recap/theory → 45 min live coding → 20 min guided lab → 10 min wrap-up and homework.** S21–S24 are review and presentation sessions.
> Offline, auto-graded exercises for weeks 41–45 and clinics W2, W3 and W5 (track record and capstone checks) are in [`labs/part12/`](../../labs/part12/).

### Week 1 — Integration & Performance

#### S1 · Platform Audit Against the LLD (item 1)

| Block | Content |
|---|---|
| Theory | Walk the LLD layer by layer (core → domain → services → library → trading → research → monitoring → apps). For each package: does it exist, does it follow its interface, is it tested, is it used by backtest *and* live? **Architecture tests:** dependency rules (e.g. `lib/` must not import `brokers/`; strategies must not import the OMS), enforced in CI with `import-linter`. Technical-debt register and prioritization for the capstone. |
| Live coding | `import-linter` contracts for the layer rules; coverage report per package; a generated architecture diagram from the package graph. |
| Lab | Each learner produces a gap list against the LLD and a 4-week plan to close it. |
| Homework | Close the top 5 gaps. |

```ini
# .importlinter — layer rules checked in CI
[importlinter]
root_package = quantforge

[importlinter:contract:layers]
name = Platform layers
type = layers
layers =
    quantforge.apps
    quantforge.monitoring
    quantforge.research
    quantforge.strategy
    quantforge.execution
    quantforge.lib
    quantforge.data
    quantforge.domain
    quantforge.core

[importlinter:contract:strategies-no-broker]
name = Strategies only emit intents (no direct broker or OMS access)
type = forbidden
source_modules = quantforge.strategy
forbidden_modules =
    quantforge.brokers
    quantforge.execution
```

#### S2 · Performance Engineering (item 2)

| Block | Content |
|---|---|
| Theory | **Latency budget** for the internal path (bar/tick in → indicators → strategy → risk → OMS → adapter submit), e.g. p99 < 50 ms; broker round-trip measured separately. **Profiling:** `cProfile`/`py-spy` (sampling, production-safe), `line_profiler`, `memray` for memory. **Event-loop health:** never block the loop (move CPU work to process pools, use `numba` kernels, streaming indicators), `uvloop`, measure loop lag. **Memory:** ring buffers instead of growing DataFrames, bounded caches, object reuse, avoiding pandas in hot paths. **Boosting:** vectorize, `numba`, Polars for batch, pre-computed features. Measure before optimizing; keep a benchmark suite. |
| Live coding | Latency instrumentation with Prometheus histograms (below); `py-spy` flame graph of the live loop; fix the top hotspot. |
| Lab | Meet the latency budget with 20 strategies × 50 symbols on 1-minute bars in replay mode. |
| Homework | Memory profile a full trading day replay; remove the largest allocation source. |

```python
from prometheus_client import Counter, Gauge, Histogram

ORDERS = Counter("qf_orders_total", "Orders sent", ["strategy", "broker"])
ACK_LATENCY = Histogram("qf_order_ack_seconds", "Submit-to-broker-ack latency", ["broker"],
                        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PIPELINE_LATENCY = Histogram("qf_signal_to_submit_seconds", "Bar received -> order submitted (internal)",
                             buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1))
EQUITY = Gauge("qf_equity_usd", "Account equity", ["account"])
LOOP_LAG = Gauge("qf_event_loop_lag_seconds", "asyncio scheduling delay")

with PIPELINE_LATENCY.time():          # wraps the internal hot path
    intents = engine.on_bar(bar)
```

#### S3 · Strategy Creator from Library Functions (items 27, 54)

| Block | Content |
|---|---|
| Theory | Most strategies are compositions of library pieces: universe + entry conditions + exit rules (targets, stops, time) + sizing + risk profile. **Declarative strategies:** YAML parsed into `Condition` trees (Part 5) and exit/sizing objects (Parts 7–8), validated against the registry (unknown function, wrong parameter, parameter outside declared range → rejected before running). **Robust by construction (item 54):** every created strategy automatically gets next-bar execution, costs, the risk engine, the validation gate and a journal. Versioning: config hash stored with every trade. Custom Python strategies remain possible for what configuration cannot express. |
| Live coding | `build_condition` (below) + schema validation; one config file running unchanged in backtest, paper and live. |
| Lab | Recreate 3 Part 7 strategies purely from YAML; prove identical signals to their hand-coded versions. |
| Homework | Exit-rule and sizing sections of the config (target/stop/trailing/time exit; fixed-risk or volatility target). |

```yaml
# configs/strategies/ema_cross_rsi_filter.yaml
name: ema_cross_rsi_filter
universe: {screener: liquid_us_etfs}
timeframe: 1d
entry:
  all:
    - {fn: crossover, args: [{ind: ema, params: {n: 20}}, {ind: ema, params: {n: 50}}]}
    - not: {fn: gt, args: [{ind: rsi, params: {n: 14}}, 70]}
exit: {stop: {type: atr, n: 14, mult: 2.0}, target: {type: r_multiple, r: 3}, max_bars: 20}
sizing: {type: fixed_risk, risk_frac: 0.005}
```

```python
def build_condition(spec, registry):
    """Config dict -> function(data) -> boolean array.
    Supports {"all": [...]}, {"any": [...]}, {"not": {...}} and {"fn": name, "args": [...]}."""
    if "all" in spec:
        parts = [build_condition(s, registry) for s in spec["all"]]
        return lambda d: np.logical_and.reduce([p(d) for p in parts])
    if "any" in spec:
        parts = [build_condition(s, registry) for s in spec["any"]]
        return lambda d: np.logical_or.reduce([p(d) for p in parts])
    if "not" in spec:
        inner = build_condition(spec["not"], registry)
        return lambda d: ~inner(d)
    fn = registry["actions"][spec["fn"]]                    # KeyError -> config rejected
    args = [build_series(a, registry) for a in spec["args"]]
    return lambda d: fn(*[a(d) for a in args])

def build_series(spec, registry):
    if isinstance(spec, (int, float)):
        return lambda d: spec
    if isinstance(spec, str):
        return lambda d: d[spec]                            # raw column, e.g. "close"
    ind = registry["indicators"][spec["ind"]]
    return lambda d: ind(d[spec.get("input", "close")], **spec.get("params", {}))
```

> Verified: the YAML above, built through `build_condition`, produces exactly the same signals as the hand-written `crossover(ema20, ema50) & ~(rsi14 > 70)` on 2,000 bars.

#### S4 · Instrument Selection & Screeners (items 28, 29)

| Block | Content |
|---|---|
| Theory | **Universe screens:** liquidity (ADV, dollar volume), price, spread, market cap, shortability, options liquidity, sector. **Advanced screens:** chart patterns (Part 5 scanner), factor ranks (Part 9), ML scores (Part 10), event calendars (Part 11), volatility regime (IV rank). Point-in-time screening for backtests (Part 8 universe). Scheduling: nightly screens via the scheduler or n8n; results stored and versioned so a live universe can be reproduced. |
| Live coding | Screener pipeline as composable filters; nightly job; universe snapshot table. |
| Lab | Build 3 screens (liquid ETFs, options-liquid large caps, pairs candidates) and feed them into strategy configs. |
| Homework | Screen-stability report: how much does each universe change day to day (turnover cost)? |

**Clinic W1:** gap-closing sprint and latency benchmark sign-off.

---

### Week 2 — Execution & Trade Management

#### S5 · Spread-Aware Order Management (item 53)

| Block | Content |
|---|---|
| Theory | Market orders pay the full spread plus impact; passive limits save the spread but may not fill (and fill more when price moves against you: adverse selection). **Spread-aware placement:** choose a price between joining the bid (passive) and crossing (aggressive) by an *aggressiveness* parameter, round to the tick in the direction that never overpays, then **re-price on a timer** (chase) up to a limit; cancel if the signal decays. Rules by context: wide spread or low urgency → passive; urgent exits and stops → aggressive; avoid trading the first minutes after the open and around auctions unless intended. **Queue position** intuition; odd lots; midpoint pegs (IB `PEG MID`) where available. Measure fill rate, time to fill and price improvement vs mid. |
| Live coding | `limit_price` and the chase loop (below) inside the OMS; cancel-replace with idempotent client IDs (Part 4). |
| Lab | 1 week of paper orders at aggressiveness 0 / 0.5 / 1.0: fill rate, time to fill, cost vs arrival mid. |
| Homework | Urgency-aware policy: map signal type (entry, exit, stop, rebalance) to a chase schedule. |

```python
from decimal import Decimal, ROUND_DOWN, ROUND_UP

def limit_price(side, bid, ask, tick, aggressiveness=0.0):
    """aggressiveness 0 = join own side (passive), 0.5 = mid, 1 = cross the spread.
    Rounded to the tick in the direction that never overpays."""
    px = bid + aggressiveness * (ask - bid) if side == "BUY" else ask - aggressiveness * (ask - bid)
    q = Decimal(str(tick))
    steps = (Decimal(str(px)) / q).to_integral_value(rounding=ROUND_DOWN if side == "BUY" else ROUND_UP)
    return float(steps * q)

async def place_with_chase(oms, order, quotes, schedule=(0.0, 0.33, 0.67, 1.0), wait_s=5.0):
    """Start passive; every `wait_s` seconds without a full fill, re-price more aggressively."""
    for aggr in schedule:
        q = quotes.latest(order.instrument)
        px = limit_price(order.side, q.bid, q.ask, order.instrument.tick, aggr)
        await oms.replace_or_submit(order, limit=px)          # same client_order_id lineage
        if await oms.wait_filled(order, timeout=wait_s):
            return True
    await oms.cancel(order)                                    # give up; the strategy decides what next
    return False
```

#### S6 · Execution Algorithms & Transaction-Cost Analysis

| Block | Content |
|---|---|
| Theory | For orders that are large relative to liquidity: **TWAP** (equal slices over time), **VWAP** (slices following the historical intraday volume curve), **POV** (a fixed share of live volume), **iceberg** (show a small part). Randomize slice timing and size slightly to avoid being predictable. Broker-native algos (IB Adaptive, VWAP) vs our own. **Transaction-cost analysis (TCA):** **implementation shortfall** (vs decision price), vs arrival mid, vs interval VWAP; feed results back into the Part 8 cost model. Almgren–Chriss as the benchmark from Part 10 S23. |
| Live coding | Schedule generators and shortfall (below); algo runner using S5 child-order placement. |
| Lab | Execute the same parent order on paper with TWAP, VWAP and a single child; compare shortfall. |
| Homework | Update the backtester's cost model from the learner's own paper TCA. |

```python
def twap_schedule(qty, start, end, slices):
    times = pd.date_range(start, end, periods=slices + 1)[:-1]
    base = qty // slices
    return pd.Series([base + (1 if i < qty - base * slices else 0) for i in range(slices)], index=times)

def vwap_schedule(qty, volume_profile: pd.Series):
    """Split qty in proportion to a historical intraday volume profile (index = time bucket)."""
    raw = volume_profile / volume_profile.sum() * qty
    sizes = np.floor(raw).astype(int)
    sizes.iloc[np.argsort(-(raw - sizes).to_numpy())[: int(qty - sizes.sum())]] += 1   # largest remainders
    return sizes

def pov_child_qty(market_volume_since_last, participation, remaining):
    return int(min(remaining, np.floor(participation * market_volume_since_last)))

def implementation_shortfall_bps(side, decision_px, fills, fees=0.0):
    """Cost vs the decision price in bps (positive = cost). fills: [(price, qty), ...]."""
    qty = sum(q for _, q in fills)
    avg = sum(p * q for p, q in fills) / qty
    sign = 1 if side == "BUY" else -1
    return 1e4 * (sign * (avg - decision_px) * qty + fees) / (decision_px * qty)

implementation_shortfall_bps("BUY", 100.0, [(100.02, 500), (100.05, 500)], fees=5.0)   # -> 4.0 bps
```

#### S7 · Option Strategy Integration (items 30, 33–41)

| Block | Content |
|---|---|
| Theory | From Part 7 templates to a running options book: config-driven option strategies (template, underlying screen, DTE window, delta targets, width, max risk), **combo execution** with S5 re-pricing at the net mid, **position management rules** (take profit, stop, roll at N DTE, adjust when a short strike's delta exceeds a threshold, close before ex-dividend or expiry to avoid assignment and pin risk), portfolio Greeks limits in the risk engine (Part 6 S8 / Part 8 S17), and hedging overlays (Part 7 S8). Expiration-day procedures. |
| Live coding | `OptionPositionManager` evaluating management rules each bar and emitting close/roll intents; combo re-pricing. |
| Lab | Run a paper iron-condor program on SPY for the week with automated management; review every action in the journal. |
| Homework | Assignment and early-exercise handling test (simulated assignment event). |

#### S8 · Trade Journal & Journal Analytics (items 26, 31, 32)

| Block | Content |
|---|---|
| Theory | An **automatic journal**: every trade records strategy/config hash, signal values at entry, market context (regime, VIX, spread), planned stop/target (so R-multiples are defined), fills and slippage, exit reason, MAE/MFE, and free-text notes. **Analytics:** expectancy by setup tag, regime, time of day, exit reason; slippage vs model; rule violations. **Journal-driven optimization (item 32):** use journal evidence to propose parameter or rule changes, which then go back through the Part 8 gate (never tuned live). |
| Live coding | Journal writer hooked to fills; `journal_stats` (below); weekly journal report. |
| Lab | Analyze the S5–S7 paper trades: where is expectancy positive and where negative? |
| Homework | Rule-violation detector (trades outside allowed hours, size above plan, missing stop). |

```python
def journal_stats(journal: pd.DataFrame, by="setup_tag") -> pd.DataFrame:
    """Expectancy in R-multiples (P&L / planned risk) by any journal dimension."""
    g = journal.groupby(by)["r_multiple"]
    return pd.DataFrame({"trades": g.size(),
                         "win_rate": g.apply(lambda r: (r > 0).mean()),
                         "avg_win_R": g.apply(lambda r: r[r > 0].mean()),
                         "avg_loss_R": g.apply(lambda r: r[r <= 0].mean()),
                         "expectancy_R": g.mean()}).sort_values("expectancy_R", ascending=False)
```

**Clinic W2:** execution-quality report: fill rates, shortfall by algo and aggressiveness, option-combo fills, and journal analytics.

---

### Week 3 — The Monitoring Department (item 7)

#### S9 · Observability: Logs, Metrics, Dashboards

| Block | Content |
|---|---|
| Theory | Three signals: **structured logs** (JSON with correlation IDs from signal → intent → order → fill), **metrics** (Prometheus: orders, rejects, fills, latency histograms, equity, exposure, drawdown, data freshness, loop lag, broker connection state), **traces** for the order path. **Grafana** dashboards: trading (P&L, exposure, positions), execution (fills, slippage, latency), system (CPU, memory, loop lag, connections), data (staleness, gaps). **Alert rules** with severities, routed through n8n (Part 11). |
| Live coding | Metrics exporter across the platform; Grafana dashboards as code (provisioned JSON); alert rules. |
| Lab | Deliberately cause: stale data, broker disconnect, reject storm. Each must appear on a dashboard and fire the right alert. |
| Homework | SLOs for the platform (e.g. data freshness < 5 s during market hours 99.9% of the time) and alerts on SLO burn. |

#### S10 · Reconciliation, Audit & Approvals

| Block | Content |
|---|---|
| Theory | **Reconciliation** (club, find, check, verify): internal vs broker orders, positions, cash and fills, continuously (e.g. every 60 s) and at end of day; breaks are alerts, broker state wins. **Audit trail:** append-only, **hash-chained** records of every intent, risk decision, order, fill, config change, AI suggestion and manual action; replayable. **Approve / reject workflow:** new strategies, parameter changes, limit changes and large orders require approval (four-eyes for teams), with reason codes, recorded in the audit log. |
| Live coding | `AuditLog` (below) persisted to PostgreSQL; reconciliation job; approval queue API used by the dashboard and n8n. |
| Lab | Tamper test: modify one audit record in the database and show `verify()` pinpoints it. Reconciliation break drill: place a manual order in TWS and watch the platform detect it. |
| Homework | Daily signed audit digest (last hash) sent to email via n8n, so history cannot be rewritten unnoticed. |

```python
import hashlib, json, time

class AuditLog:
    """Append-only, hash-chained audit trail: altering any past record breaks every later hash."""
    def __init__(self):
        self.records, self._last = [], "0" * 64

    def append(self, event: str, **data) -> str:
        rec = {"ts": time.time(), "event": event, "data": data, "prev": self._last}
        rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
        self.records.append(rec)
        self._last = rec["hash"]
        return rec["hash"]

    def verify(self) -> int:
        """Index of the first broken record, or -1 if the chain is intact."""
        prev = "0" * 64
        for i, rec in enumerate(self.records):
            body = {k: v for k, v in rec.items() if k != "hash"}
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
            if rec["prev"] != prev or digest != rec["hash"]:
                return i
            prev = rec["hash"]
        return -1
```

#### S11 · Anomaly Detection & the Controller (item 52)

| Block | Content |
|---|---|
| Theory | What to watch: **data anomalies** (price jumps vs recent volatility, zero/negative prices, stale quotes, crossed markets), **execution anomalies** (latency spikes, reject bursts, slippage far above model), **behavioural anomalies** (order rate, position size, turnover vs the strategy's own history), **P&L anomalies**, and **fat-tail errors**: a P&L move far beyond what the risk model allows is more often a data/position/fill error than a real event, so pause and check. Methods: streaming robust z-scores (median/MAD), CUSUM for drift, Part 10 autoencoder score, Part 10 drift monitors for ML strategies. **Controller:** maps anomaly severity to actions: log → alert → pause strategy → trip kill switch; everything audited; resume only with approval. |
| Live coding | `RobustZ` detector and fat-tail check (below); controller policy table; integration with the risk engine and kill switch. |
| Lab | Replay a day with injected faults (bad tick, duplicated fill, latency spike, runaway order loop); each must be caught and handled by the right action. |
| Homework | Tune thresholds from 4 weeks of paper data to target < 1 false alarm per day. |

```python
class RobustZ:
    """Streaming robust z-score (median / MAD) over a rolling window."""
    def __init__(self, window=500, threshold=6.0, min_obs=50):
        self.buf, self.window, self.threshold, self.min_obs = [], window, threshold, min_obs

    def update(self, x: float) -> tuple[float, bool]:
        z = 0.0
        if len(self.buf) >= self.min_obs:
            arr = np.asarray(self.buf)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826 + 1e-12
            z = (x - med) / mad
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        return z, abs(z) > self.threshold

def fat_tail_error(realized_pnl, expected_sigma, k=5.0):
    """P&L move too large for the risk model: pause and verify data, positions and fills."""
    return abs(realized_pnl) > k * expected_sigma

CONTROLLER_POLICY = {             # anomaly -> action (all actions audited)
    "stale_data":         "pause_strategy",
    "latency_spike":      "alert",
    "reject_burst":       "pause_strategy",
    "slippage_outlier":   "alert",
    "fat_tail_pnl":       "pause_all_and_reconcile",
    "runaway_order_rate": "trip_kill_switch",
}
```

> Verified: on 300 simulated order-ack latencies plus one 400 ms spike, `RobustZ` raises exactly one flag, on the spike.

#### S12 · Reports & Dashboards (item 51)

| Block | Content |
|---|---|
| Theory | Audiences and cadences: **real-time** (Grafana for operations), **daily** (P&L, risk, exposures, fills, incidents; via n8n), **weekly** (strategy performance, attribution, journal findings, reconciliation breaks), **monthly** (performance vs backtest expectations, DSR update, capacity, decisions: scale/keep/retire). **Streamlit dashboard** for research/strategy views: equity curves, live vs backtest overlay, approvals queue, journal browser. Performance attribution by strategy, asset and factor (Part 9 factor model). |
| Live coding | Streamlit app pages; report generators reusing Part 8 `performance.py`. |
| Lab | Produce the first daily and weekly reports from paper data; review them as a desk would. |
| Homework | "Live vs backtest" tracking chart with a statistical band (expected range from the backtest distribution). |

**Clinic W3 — M8 review:** strategy creator, screeners, spread-aware OMS, algos, option manager, journal, monitoring department and dashboards demonstrated on paper.

---

### Week 4 — Production Engineering

#### S13 · Deployment with Docker Compose

| Block | Content |
|---|---|
| Theory | Service layout: IB Gateway (with IBC for automated login and daily restarts), trading engine, API, scheduler, PostgreSQL/TimescaleDB, Redis, n8n, Prometheus, Grafana. One container per concern; health checks; restart policies; volumes for state; pinned image versions. Hosting: a VPS in a US East region (close to broker servers), time sync (NTP/chrony), resource sizing. Secrets via environment files outside Git or Docker secrets. Environments: `paper`, `live` as separate stacks with separate credentials. |
| Live coding | `docker-compose.yml` (below); health endpoints; first deploy to a VPS. |
| Lab | Deploy the full stack; restart any single service and confirm the platform recovers and reconciles. |
| Homework | Nightly database backups with a tested restore. |

```yaml
# deploy/docker-compose.yml (paper stack; the live stack uses separate env files and volumes)
services:
  ib-gateway:
    image: ${IB_GATEWAY_IMAGE}            # maintained IB Gateway + IBC image, pinned by digest
    env_file: ./secrets/ib.env            # credentials never in Git
    environment: {TRADING_MODE: paper}
    restart: unless-stopped
  db:
    image: timescale/timescaledb:latest-pg16   # pin an exact tag in production
    volumes: [dbdata:/var/lib/postgresql/data]
    env_file: ./secrets/db.env
    healthcheck: {test: ["CMD-SHELL", "pg_isready -U quantforge"], interval: 10s, retries: 5}
  redis:
    image: redis:7
  engine:
    build: ..
    command: python -m quantforge.apps.cli run --config configs/portfolio.yaml --env paper
    env_file: ./secrets/engine.env
    depends_on:
      db: {condition: service_healthy}
      redis: {condition: service_started}
      ib-gateway: {condition: service_started}
    restart: unless-stopped
  api:
    build: ..
    command: uvicorn quantforge.apps.api:app --host 0.0.0.0 --port 8000
    env_file: ./secrets/engine.env
    depends_on: [db]
  n8n:
    image: n8nio/n8n
    volumes: [n8ndata:/home/node/.n8n]
  prometheus:
    image: prom/prometheus
    volumes: [./prometheus.yml:/etc/prometheus/prometheus.yml:ro]
  grafana:
    image: grafana/grafana
    volumes: [./grafana:/etc/grafana/provisioning:ro]
volumes: {dbdata: {}, n8ndata: {}}
```

#### S14 · CI/CD & the Release Process

| Block | Content |
|---|---|
| Theory | Pipeline: lint (`ruff`), types (`mypy --strict`), unit and contract tests (Sim adapter), architecture rules (S1), **backtest regression tests** (a fixed dataset and config must reproduce stored results within tolerance, so refactors cannot silently change strategy behaviour), AI evaluation suite (Part 11), image build. **Release process:** semantic versions, changelog, deploy to paper first, promote to live only after a paper soak period; feature flags to switch strategies on/off without a deploy; rollback procedure. |
| Live coding | GitHub Actions workflow (below); regression-test fixtures. |
| Lab | Introduce a subtle bug (off-by-one in an indicator) and watch the regression test catch it. |
| Homework | Release checklist and changelog template. |

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run mypy --strict quantforge
      - run: uv run lint-imports                       # architecture rules (S1)
      - run: uv run pytest -q --cov=quantforge --cov-fail-under=85 -m "not paper"
      - run: uv run pytest -q tests/regression          # backtest results must match stored baselines
  image:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t quantforge:${{ github.sha }} .
```

#### S15 · Reliability, Disaster Recovery & Chaos Testing

| Block | Content |
|---|---|
| Theory | Failure modes and recovery targets (RTO/RPO): VPS down, database loss, broker outage, gateway stuck at login, network partition, clock drift, disk full. **Startup recovery:** load state, reconcile with brokers, resume only after reconciliation passes. **Backups** tested by restore. **Chaos tests:** kill containers, drop the network, fill the disk, skew the clock, in paper. **Runbooks:** one page per failure: symptoms, checks, actions, who to notify. |
| Live coding | Chaos script (random container kills during a replay) and recovery assertions. |
| Lab | Recovery drill: destroy the engine container mid-session; recover to a reconciled state within the RTO. |
| Homework | Complete runbooks for the 10 most likely incidents. |

#### S16 · Security & Operational Risk

| Block | Content |
|---|---|
| Theory | Least privilege: separate broker API users/keys for paper and live, read-only keys where possible, trading permissions limited to the instruments actually traded; 2FA on broker accounts; IP allow-lists; no inbound ports except via a reverse proxy with authentication; secret rotation; dependency and image scanning; audit of who changed what. Operational-risk controls: max order size and notional at broker level (IB precautionary settings) as a second line behind the platform's risk engine; daily loss limits; the kill switch reachable from a phone. **Go-live checklist** preview. |
| Live coding | Security checklist automation (secret scan, dependency audit, open-port check); broker-side precautionary limits configured. |
| Lab | Threat-model the platform (what can go wrong, how likely, impact, control); fix the top 3 risks. |
| Homework | Complete the go-live checklist evidence pack. |

**Clinic W4:** full stack deployed on a VPS; recovery drill passed; security checklist complete.

---

### Week 5 — Go-Live

#### S17 · Rollout Plan & Capital Ramp

| Block | Content |
|---|---|
| Theory | Stages: **paper** (plumbing), **shadow** (live data, orders logged not sent; compare to paper), **small live** (minimum size), **scaled live**. **Promotion criteria** per stage (days without incidents, reconciliation clean, slippage within model, performance inside the backtest's expected band). **Capital-ramp rules** (e.g. increase size only after N weeks inside the band; cut size automatically after a drawdown beyond the plan). **Demotion rules** (performance outside the band, drift alarms, repeated incidents → back to paper). Written risk budget per strategy. |
| Live coding | Stage and ramp rules as configuration enforced by the risk engine. |
| Lab | Each learner writes and defends their rollout plan. |
| Homework | Prepare the go-live review. |

#### S18 · Daily Operations

| Block | Content |
|---|---|
| Theory | **Pre-market checklist:** gateway connected and authenticated, data fresh, reconciliation clean, kill switch armed and tested, screens updated, calendar reviewed (events, holidays, early closes), risk limits loaded, disk and backups OK. **Intraday:** dashboards, alert handling, no ad-hoc parameter changes. **End of day:** reconciliation, journal review, daily report, backup confirmation, incident log. Weekly: IB re-authentication, dependency updates in paper first. Holidays and half-days from the exchange calendar. |
| Live coding | Automated pre-market checklist (a script whose failures block trading) triggered by n8n. |
| Lab | Run a full operating day on paper by the checklist. |
| Homework | Personal operating handbook. |

#### S19 · Incident Response Drills

| Block | Content |
|---|---|
| Theory | Incident lifecycle: detect → contain (pause/kill) → diagnose → recover → reconcile → post-mortem (blameless, with action items). Severity levels and response times. **Drills:** broker disconnect during open positions, runaway strategy (order loop), bad price tick, partial fill on one leg of a pair or combo, database outage, VPS reboot at the open. |
| Live coding | Incident log template and post-mortem generator from the audit trail. |
| Lab | Timed drills on paper; each learner leads one incident. |
| Homework | Post-mortem for the drill they led. |

#### S20 · Capstone Kickoff & M9 Go-Live Review

| Block | Content |
|---|---|
| Theory | Capstone requirements (Section 7). **Go-live review panel:** validation-gate evidence for each strategy (Part 8), rollout plan, risk budgets, runbooks, security checklist, drill results. Outcome per learner: approved for small live, or paper-only track record. |
| Lab | Go-live reviews; track record starts at the next session open. |

**Clinic W5:** go-live review panel; final fixes; track-record start.

---

### Weeks 6–8 — Track Record, Reviews & Defense

| Session | Content |
|---|---|
| **S21 · Weekly review 1** (week 46) | Performance vs backtest band, attribution by strategy/factor, reconciliation breaks, execution TCA, alerts and incidents, journal findings; decisions made (scale/hold/demote) with reasons in the audit log. |
| **S22 · Weekly review 2** (week 47) | Same agenda plus a deep dive into live-vs-backtest differences (fills, costs, timing, data) and what was changed in the cost model; peer review of another learner's platform. |
| **S23 · Final review & career track** (week 48) | 4-week results; lessons learned; portfolio of work for careers (GitHub repo, capstone dossier, research papers); quant-developer interview topics; how to continue: scaling capital, new markets, team workflows. |
| **S24 · Capstone defense (demo day)** | 30-minute presentation + live demo + 20-minute Q&A with the panel (Section 7). |

---

## 5. Notebook & Artifact Map

| Artifact | Session | Location |
|---|---|---|
| Architecture rules, gap register | S1 | `.importlinter`, `docs/gaps.md` |
| Latency benchmark & metrics | S2 | `core/metrics.py`, `benchmarks/` |
| Strategy creator & configs | S3 | `strategy/creator.py`, `configs/strategies/` |
| Screeners | S4 | `screener/`, `configs/screens/` |
| Spread-aware OMS | S5 | `execution/spread_aware.py` |
| Execution algos & TCA | S6 | `execution/algos.py`, `analytics/tca.py` |
| Option position manager | S7 | `options/strategies/manager.py` |
| Trade journal & analytics | S8 | `lib/journal.py`, `analytics/journal_analysis.py` |
| Observability | S9 | `monitoring/metrics.py`, `deploy/grafana/`, `deploy/alerts.yml` |
| Reconciliation, audit, approvals | S10 | `monitoring/audit.py`, `monitoring/reconcile.py`, `monitoring/approvals.py` |
| Anomaly detection & controller | S11 | `monitoring/anomaly.py`, `monitoring/controller.py` |
| Reports & dashboard | S12 | `monitoring/reports.py`, `apps/dashboard.py` |
| Deployment | S13 | `deploy/docker-compose.yml`, `Dockerfile` |
| CI/CD | S14 | `.github/workflows/ci.yml`, `tests/regression/` |
| Runbooks & chaos tests | S15 | `runbooks/`, `tests/chaos/` |
| Security & go-live checklist | S16–S18 | `ops/checklists/` |
| Incident log & post-mortems | S19 | `ops/incidents/` |

---

## 6. Common Mistakes & How to Catch Them

| # | Mistake | Symptom | Detection / Fix |
|---|---|---|---|
| 1 | Strategies importing broker code | Risk engine bypassed | Architecture rules in CI (S1) |
| 2 | Optimizing without measuring | Effort spent in the wrong place | Profiling and latency histograms first |
| 3 | Blocking the event loop | Delayed fills, missed updates | Loop-lag metric; process pools; streaming indicators |
| 4 | Market orders by default | Paying the full spread on every trade | Spread-aware placement with chase schedules |
| 5 | No TCA | Backtest cost model drifts from reality | Shortfall per order; feed back into costs |
| 6 | Journal without planned risk | R-multiples undefined, analysis impossible | Stop and target recorded at entry, mandatory fields |
| 7 | Logs without correlation IDs | Incidents impossible to reconstruct | One ID from signal to fill |
| 8 | Mutable audit records | History can be rewritten | Hash chain + daily digest |
| 9 | Alerts nobody acts on | Alert fatigue, real alerts missed | Severity levels, tuned thresholds, runbook per alert |
| 10 | Restart without reconciliation | Duplicate or orphan orders | Startup gate: reconcile before trading |
| 11 | Untested backups | Data loss discovered during an incident | Scheduled restore tests |
| 12 | Same credentials for paper and live | Live orders by accident | Separate stacks, users, keys, env files |
| 13 | Parameter changes during the session | Untracked behaviour changes | Changes only via approvals + next-session deploy |
| 14 | Scaling after a lucky week | Oversized losses when luck turns | Ramp rules tied to the backtest band and track-record length |
| 15 | No post-mortems | The same incident repeats | Blameless post-mortem with action items for every incident |

---

## 7. Capstone Assessment

**Deliverables:**
1. **Platform:** the complete `quantforge` repository: all LLD layers, tests (≥ 85% coverage), architecture rules, CI green, Docker deployment, dashboards, runbooks.
2. **Strategy portfolio:** ≥ 3 strategies from at least 2 families (e.g. momentum, stat-arb, options, ML), each with a Part 8 validation dossier; portfolio allocation with risk budgets.
3. **Track record:** 4 weeks of paper or small-size live trading through the platform, with daily reports, weekly reviews, a reconciled trade log, TCA and incident log.
4. **Live vs backtest reconciliation:** results against the backtest's expected band, with every significant difference explained.
5. **Operations evidence:** drill results, recovery test, security checklist, audit-chain verification.
6. **Defense:** presentation, live demo (including a kill-switch trip and recovery) and Q&A.

| Criterion | Points |
|---|---|
| Architecture & code quality (LLD conformance, tests, CI, architecture rules) | 15 |
| Strategy research quality (validation dossiers, portfolio construction) | 15 |
| Execution quality (spread-aware OMS, algos, TCA) | 10 |
| Risk management in operation (limits, sizing, controller, kill switch) | 15 |
| Monitoring department (observability, reconciliation, audit, anomaly detection, reports) | 15 |
| Production readiness (deployment, recovery, security, runbooks) | 10 |
| Track record & live-vs-backtest reconciliation (honesty over returns) | 10 |
| Defense: clarity, demo, answers | 10 |
| **Total** | **100** |

**Graduation requirements** (all mandatory, in addition to ≥ 70 points): kill switch demonstrated live at the defense; no order bypassed the risk engine during the track record (verified from the audit chain); reconciliation clean at every end of day (or every break explained and resolved); DSR of each strategy computed from the full research log.

**Note on returns:** the capstone does **not** grade profit. Four weeks is too short to judge performance. It grades whether the system behaves as designed, whether results stay within the expected range, and whether the learner can explain every difference.

---

## 8. Further Reading

| Type | Reference |
|---|---|
| Book | Percival, H. & Gregory, B. (2020). *Architecture Patterns with Python*. O'Reilly. |
| Book | Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly. |
| Book | Beyer, B. et al. (2016). *Site Reliability Engineering*. O'Reilly (free online). |
| Book | Nygard, M. (2018). *Release It!* (2nd ed.). Pragmatic Bookshelf. |
| Book | Johnson, B. (2010). *Algorithmic Trading and DMA*. 4Myeloma Press. (execution, order types, algos) |
| Book | Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management*. Academic Press. (TCA) |
| Book | Harris, L. (2003). *Trading and Exchanges*. Oxford University Press. |
| Paper | Perold, A. (1988). "The Implementation Shortfall: Paper versus Reality." *Journal of Portfolio Management*, 14(3). |
| Paper | Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3. |
| Case study | SEC (2013). Administrative proceeding on Knight Capital Americas LLC (deployment failure, Aug 1 2012). |
| Docs | Docker Compose, GitHub Actions, Prometheus & Grafana, `import-linter`, `py-spy`, `memray`, Streamlit; IB TWS API (order types, precautionary settings), IBC; Alpaca API |

---

## 9. Instructor Notes

- Hold the go-live review to a high bar: approval for small live is earned by evidence (gate, drills, runbooks), not by returns.
- Keep a shared "incident wall" across learners during the track record; the post-mortems are some of the most valuable material in the course.
- Encourage paper-only track records for anyone unsure; the grade does not depend on going live.
- Live capital is the learner's own and at their own risk; sizes stay at the minimum allowed by the rollout plan during the course.
- Demo day: invite practitioners as panelists; ask each learner to trip the kill switch live and show the audit chain entry it created.
