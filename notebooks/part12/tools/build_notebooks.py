"""Build the Part 12 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part12:  python tools/build_notebooks.py
Cells are ("md", text), ("code", code) or ("ex", starter_code, solution_code).
Edit the content here and rebuild, so starter and solution notebooks never drift apart.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
KERNEL = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}

SETUP = """import sys
from pathlib import Path
for d in (Path.cwd(), Path.cwd().parent):       # p12lib.py is in notebooks/part12/
    sys.path.insert(0, str(d))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p12lib as p

p.use_course_style()"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 12 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_12_TRADING_PLATFORM_CAPSTONE.md) · graded labs in [`labs/part12/`](../../labs/part12/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.
Everything runs offline on synthetic data with known answers (planted violations, injected faults, a strategy that stops working): no broker, Docker or network is needed.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
_MI_HEAD = """import ast

def module_imports(path):
    tree = ast.parse(Path(path).read_text())
    out = set()
    for node in ast.walk(tree):                                      # every node, also inside functions
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
"""
_MI_TAIL = """    return out

import tempfile
pkg = p.make_toy_package(Path(tempfile.mkdtemp()))                   # a small `quantforge` with 3 planted violations
files = sorted(pkg.rglob("*.py"))
mine = {str(f.relative_to(pkg)): p.attempt(lambda f=f: sorted(module_imports(f))) for f in files}
mine = p.check("module_imports", mine, {str(f.relative_to(pkg)): sorted(p.module_imports(f)) for f in files})
{k: v for k, v in mine.items() if v}"""

_Q_HEAD = """class MyLatencyHistogram(p.LatencyHistogram):
    def quantile(self, q):
"""
_Q_TAIL = """
latencies = p.latency_samples()                                      # order-ack latencies in seconds, one 400 ms spike
hists = [MyLatencyHistogram(), p.LatencyHistogram()]
for h in hists:
    for x in latencies:
        h.observe(x)
qs = (0.5, 0.9, 0.99, 1.0)
mine = p.check("LatencyHistogram.quantile", [p.attempt(hists[0].quantile, q) for q in qs], [hists[1].quantile(q) for q in qs])
pd.DataFrame({"histogram (bucket upper bound)": mine, "exact percentile": np.quantile(latencies, qs)}, index=[f"p{int(q * 100)}" for q in qs])"""

_EMA_CHECK = """
x = np.cumsum(np.random.default_rng(0).normal(size=500))
live = MyStreamingEMA(20)
mine = p.check("StreamingEMA", p.attempt(lambda: np.array([live.update(v) for v in x])),
               pd.Series(x).ewm(span=20, adjust=False).mean().to_numpy())
print(f"live EMA vs pandas: max difference {np.max(np.abs(mine - pd.Series(x).ewm(span=20, adjust=False).mean().to_numpy())):.1e}")"""

NB["01_architecture_latency"] = [
    header("01", "Architecture rules and latency", "S1–S2 (Platform audit against the LLD · Performance engineering)",
           "1. Extract every import of a module from its syntax tree, including imports hidden in functions.\n"
           "2. Enforce the layered architecture and a forbidden dependency from the import graph.\n"
           "3. Read latency percentiles from a Prometheus-style histogram.\n"
           "4. See what blocking the event loop costs, and keep indicators O(1) per tick."),
    ("code", SETUP),
    ("md", "## 1. Let the machine enforce the architecture\n\n"
           "The platform's layers, top to bottom: `apps → monitoring → research → strategy → execution → lib → data → domain → "
           "core` (`p.LAYERS`). A module may import its own layer or lower ones only, and strategies must never talk to a broker "
           "directly. `p.make_toy_package` writes a small `quantforge` package with three planted violations. Step one: every "
           "module a file imports, found by walking the whole syntax tree. For `from a.b import c` (absolute, `level == 0`) record "
           "both `a.b` and `a.b.c`."),
    ("md", YOUR_TURN),
    ("ex", _MI_HEAD + """            ...                                                      # ✍️ add node.module and module.name for each name
""" + _MI_TAIL,
     _MI_HEAD + """            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
""" + _MI_TAIL),
    ("code", """graph = p.import_graph(pkg)
print("layer violations:", p.check_layers(graph, p.LAYERS))
print("strategy → broker:", p.check_forbidden(graph, "quantforge.strategy", ["quantforge.brokers"]))
header_grep = [str(f.relative_to(pkg)) for f in files
               if any(line.startswith(("import quantforge", "from quantforge")) and any(f"quantforge.{x}" in line for x in ("apps", "execution", "brokers"))
                      for line in f.read_text().splitlines())]
print("a grep for top-level import lines finds only:", [f for f in header_grep if f.split("/")[0] in ("lib", "core", "strategy")])"""),
    ("md", "All three violations, including `core/util.py`, whose import of the execution layer hides inside a function where a "
           "grep of top-level import lines never looks. In the platform this runs in CI (`import-linter`), so a violation fails "
           "the build instead of waiting for a reviewer to notice.\n\n"
           "## 2. Measure latency like Prometheus\n\n"
           "A histogram keeps **cumulative** counts per bucket (upper bounds 1, 5, 10, 25, 50, 100 ms, +∞). A quantile is read as "
           "the upper bound of the first bucket whose cumulative count reaches `q·n`: cheap, mergeable across processes, and "
           "conservative. `self.bounds`, `self.counts` (cumulative) and `self.n` are given."),
    ("md", YOUR_TURN),
    ("ex", _Q_HEAD + """        return ...                                                   # ✍️
""" + _Q_TAIL,
     _Q_HEAD + """        return next(b for b, c in zip(self.bounds, self.counts) if c >= q * self.n)
""" + _Q_TAIL),
    ("md", "The histogram overstates each percentile up to the next bucket bound: choose buckets around your SLO (here 50 ms) so "
           "the answer that matters is precise.\n\n"
           "## 3. Never block the event loop\n\n"
           "The live engine runs on one asyncio loop: market data, orders and heartbeats share it. `p.event_loop_lag` runs a 10 ms "
           "ticker while a 200 ms computation runs either **on** the loop or in a thread pool, and reports the worst delay. "
           "(Jupyter already runs a loop, so the measurement runs in its own thread.)"),
    ("code", """from concurrent.futures import ThreadPoolExecutor

work = lambda: time.sleep(0.2)                                       # noqa: E731  (stands in for a 200 ms computation)
with ThreadPoolExecutor(1) as pool:
    on_loop = pool.submit(p.event_loop_lag, work, False).result()
    in_executor = pool.submit(p.event_loop_lag, work, True).result()
print(f"worst delay of the 10 ms ticker: {on_loop * 1e3:.0f} ms with the work on the loop, {in_executor * 1e3:.1f} ms with it in an executor")""".replace("time.sleep", "__import__('time').sleep")),
    ("md", "Everything else waits for the blocking call. In the hot path, also avoid work that grows with history: an EMA can be "
           "updated in O(1) per tick, `value += α·(x − value)` with `α = 2/(n + 1)`, starting from the first value. It must equal "
           "pandas' `ewm(span=n, adjust=False)` (train/serve parity, as in Part 10)."),
    ("md", YOUR_TURN),
    ("ex", """class MyStreamingEMA:
    def __init__(self, n):
        self.alpha, self.value = 2 / (n + 1), None

    def update(self, x):
        self.value = ...                                             # ✍️ the first value, then the O(1) update
        return self.value
""" + _EMA_CHECK,
     """class MyStreamingEMA:
    def __init__(self, n):
        self.alpha, self.value = 2 / (n + 1), None

    def update(self, x):
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return self.value
""" + _EMA_CHECK),
    ("md", "## Wrap-up\n\n"
           "* Architecture rules are checked from the import graph, in CI, over the whole syntax tree.\n"
           "* Latency is measured as histograms and read as percentiles; buckets sit around the SLO.\n"
           "* Nothing CPU-heavy on the event loop; O(1) streaming indicators in the hot path.\n"
           "* Graded version: `labs/part12/week41_integration` (`module_imports`, `import_graph`, `check_layers`, "
           "`LatencyHistogram`, `RingBuffer`, `StreamingEMA`, `event_loop_lag`)."),
]

# ---------------------------------------------------------------------------------------------- 02
_BC_HEAD = """def build_condition(spec, registry=p.REGISTRY):
    if "all" in spec:
        parts = [build_condition(s, registry) for s in spec["all"]]
"""
_BC_TAIL = """    if "not" in spec:
        inner = build_condition(spec["not"], registry)
        return lambda d: ~inner(d)
    fn = registry["actions"][spec["fn"]]
    args = [p.build_series(a, registry) for a in spec["args"]]
    return lambda d: fn(*[a(d) for a in args])

CONFIG = \"\"\"
name: ema_cross_rsi
entry:
  all:
    - fn: crossover
      args: [{ind: ema, params: {n: 20}}, {ind: ema, params: {n: 50}}]
    - not: {fn: gt, args: [{ind: rsi, params: {n: 14}}, 70]}
exit: {fn: lt, args: [close, {ind: sma, params: {n: 20}}]}
sizing: {risk_per_trade: 0.005}
\"\"\"
import yaml

cfg = yaml.safe_load(CONFIG)
bars = p.ohlcv()
specs = [cfg["entry"], cfg["exit"], {"any": [cfg["exit"], {"fn": "gt", "args": ["close", 150]}]}]
mine = [p.attempt(lambda s=s: build_condition(s)(bars)) for s in specs]
mine = p.check("build_condition", mine, [p.build_condition(s)(bars) for s in specs])
print(f"entry signals: {int(mine[0].sum())} of {len(bars)} days")"""

_VC_HEAD = """def walk_series(s, errors, registry=p.REGISTRY):
    if not isinstance(s, dict):
        return                                                       # a column name or a constant
    ind = s.get("ind")
    if ind not in registry["indicators"]:
        errors.append(f"unknown indicator {ind}")
        return
    ranges = registry["params"].get(ind, {})
    for name, v in s.get("params", {}).items():
"""
_VC_TAIL = """
series = [{"ind": "ema", "params": {"n": 20}}, {"ind": "ema", "params": {"n": 1000}}, {"ind": "ema", "params": {"window": 5}},
          {"ind": "macd"}, "close", 70, {"ind": "rsi", "params": {"n": 1}}]
mine = []
for s in series:
    errs = []
    mine.append(p.attempt(lambda: walk_series(s, errs) or errs))
bad = yaml.safe_load(CONFIG)
bad["entry"]["all"][0]["args"][1]["params"]["n"] = 1000
bad["entry"]["all"][1] = {"fn": "less", "args": [1, 2]}
del bad["sizing"]
reference = []
for s in series:
    cfg_one = {"name": "x", "exit": {}, "sizing": {}, "entry": {"fn": "gt", "args": [s, 0]}}
    reference.append(p.validate_config(cfg_one))
mine = p.check("walk_series", mine, reference)
print("errors in a broken config:", p.validate_config(bad))
try:
    p.load_strategy(yaml.safe_dump(bad))
except ValueError as err:
    print("load_strategy refuses it:", err)"""

_SCR_CHECK = """
snap = p.universe_snapshot()                                         # one day's point-in-time universe (300 names)
filters = [p.min_adv(2e7), p.price_between(10, 500), p.max_spread(10), p.shortable()]
cases = [(snap, filters), (snap, filters, 20), (snap, [p.sector_in(["tech"]), p.min_option_oi(5000)], 10), (snap, [])]
mine = [p.attempt(screen, *c) for c in cases]
mine = p.check("screen", mine, [p.screen(*c) for c in cases])
print(f"{len(snap)} names → {len(mine[0])} pass the liquid, shortable screen;  top 5 by dollar volume: {mine[1][:5]}")"""

NB["02_strategy_creator_screeners"] = [
    header("02", "The strategy creator and screeners", "S3–S4 (Strategy creator from library functions · Instrument selection & screeners)",
           "1. Turn a YAML strategy config into a signal function from registered indicators and actions.\n"
           "2. Reject a bad config before it runs, with every error listed.\n"
           "3. Screen a point-in-time universe with composable filters.\n"
           "4. Measure universe turnover, and store screen results so a live universe can be reproduced."),
    ("code", SETUP),
    ("md", "## 1. Strategies as data\n\n"
           "A strategy is a YAML config, not code: conditions are trees of `all`, `any`, `not` and leaf actions (`crossover`, "
           "`gt`, `lt`) over series (registered indicators, columns or constants). Build it recursively: `all` → element-wise "
           "AND of its parts (`np.logical_and.reduce`), `any` → OR (`np.logical_or.reduce`), `not` → negation. Each node returns "
           "a function of the data."),
    ("md", YOUR_TURN),
    ("ex", _BC_HEAD + """        return lambda d: ...                                         # ✍️ AND of the parts
    if "any" in spec:
        parts = [build_condition(s, registry) for s in spec["any"]]
        return lambda d: ...                                         # ✍️ OR of the parts
""" + _BC_TAIL,
     _BC_HEAD + """        return lambda d: np.logical_and.reduce([f(d) for f in parts])
    if "any" in spec:
        parts = [build_condition(s, registry) for s in spec["any"]]
        return lambda d: np.logical_or.reduce([f(d) for f in parts])
""" + _BC_TAIL),
    ("md", "## 2. Validate before running\n\n"
           "A config is checked against the registry before any backtest or live run: unknown actions and indicators, unknown "
           "parameters, and parameters outside their allowed range (`p.REGISTRY[\"params\"]`, e.g. `ema.n` in [2, 500]). The part "
           "that checks one series: for each parameter, report `\"unknown parameter <ind>.<name>\"` if it has no range, else "
           "`\"<ind>.<name>=<v> outside [lo, hi]\"` if it is out of range."),
    ("md", YOUR_TURN),
    ("ex", _VC_HEAD + """        ...                                                          # ✍️ the two checks
""" + _VC_TAIL,
     _VC_HEAD + """        if name not in ranges:
            errors.append(f"unknown parameter {ind}.{name}")
        elif not ranges[name][0] <= v <= ranges[name][1]:
            errors.append(f"{ind}.{name}={v} outside [{ranges[name][0]}, {ranges[name][1]}]")
""" + _VC_TAIL),
    ("md", "A valid config also gets a **hash** (`p.config_hash`: the first 12 hex digits of the sha256 of the sorted JSON), "
           "stored with every trade, so any fill can be traced to the exact parameters that produced it:"),
    ("code", """cfg, entry, h = p.load_strategy(CONFIG)
signal = entry(bars)
print(f"config hash {h}; {int(signal.sum())} entries")
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(bars.index, bars["close"], lw=0.8)
ax.scatter(bars.index[signal], bars["close"][signal], color="#1baf7a", s=20, zorder=3, label="entry")
ax.legend(); ax.set_title(f"{cfg['name']} (config {h})"); plt.show()"""),
    ("md", "## 3. Screeners\n\n"
           "A screen is a list of filter factories (`p.min_adv`, `p.price_between`, `p.max_spread`, `p.sector_in`, "
           "`p.min_option_oi`, `p.shortable`), each mapping a snapshot to a boolean Series. Keep the symbols that pass **all** "
           "filters (`np.logical_and.reduce`, all True when there are none), sort by `adv_usd` descending then `symbol`, and return "
           "at most `limit`."),
    ("md", YOUR_TURN),
    ("ex", """def screen(df, filters, limit=None):
    mask = np.logical_and.reduce([f(df) for f in filters]) if filters else np.ones(len(df), bool)
    out = ...                                                        # ✍️ the passing symbols, sorted
    return out if limit is None else out[:limit]
""" + _SCR_CHECK,
     """def screen(df, filters, limit=None):
    mask = np.logical_and.reduce([f(df) for f in filters]) if filters else np.ones(len(df), bool)
    out = df[mask].sort_values(["adv_usd", "symbol"], ascending=[False, True])["symbol"].tolist()
    return out if limit is None else out[:limit]
""" + _SCR_CHECK),
    ("md", "## 4. Turnover and reproducibility\n\n"
           "Every name that enters the universe is a trade. **Universe turnover** is the share of today's universe that wasn't in "
           "yesterday's. How much does day-to-day noise in liquidity and spreads move it?"),
    ("code", """rows = {}
for noise in (0.05, 0.15, 0.30):
    tomorrow = p.next_snapshot(snap, noise=noise)
    rows[f"noise {noise:.0%}"] = {"names today": len(p.screen(snap, filters)), "names tomorrow": len(p.screen(tomorrow, filters)),
                                   "turnover": p.universe_turnover(p.screen(snap, filters), p.screen(tomorrow, filters))}
display(pd.DataFrame(rows).T.round(3))
store = p.UniverseStore()
store.save("2025-06-02", "liquid_shortable", {"filters": "adv>=2e7, 10<=px<=500, spread<=10, shortable"}, p.screen(snap, filters))
print("as of 2025-06-05 the live universe came from:", {k: (v if k != "symbols" else f"{len(v)} symbols") for k, v in store.get("2025-06-05", "liquid_shortable").items()})
print("as of 2025-06-01:", store.get("2025-06-01", "liquid_shortable"))"""),
    ("md", "Turnover grows with the noise at the filter edges: names near a threshold flip in and out. Hysteresis (enter above "
           "one level, leave below a lower one) or ranking on smoothed values reduces it. The store answers \"what was the "
           "universe on that date, and from which screen config?\" point-in-time, like the feature store of Part 10.\n\n"
           "## Wrap-up\n\n"
           "* Strategies are validated data: registered building blocks, trees, parameter ranges, a config hash on every trade.\n"
           "* Screens compose small filters; measure the universe turnover they cause.\n"
           "* Store screen results point-in-time, with the config that produced them.\n"
           "* Graded version: `labs/part12/week41_integration` (`build_condition`, `validate_config`, `load_strategy`, the "
           "filters, `screen`, `universe_turnover`, `UniverseStore`)."),
]

# ---------------------------------------------------------------------------------------------- 03
_LP_HEAD = """from decimal import ROUND_DOWN, ROUND_UP, Decimal

def limit_price(side, bid, ask, tick, aggressiveness=0.0):
    px = bid + aggressiveness * (ask - bid) if side == "BUY" else ask - aggressiveness * (ask - bid)
    q = Decimal(str(tick))
"""
_LP_TAIL = """    return float(steps * q)

cases = [("BUY", 100.00, 100.05, 0.01, 0.0), ("BUY", 100.00, 100.05, 0.01, 0.33), ("SELL", 100.00, 100.05, 0.01, 0.33),
         ("BUY", 100.00, 100.05, 0.01, 1.0), ("SELL", 4.10, 4.15, 0.05, 0.5), ("BUY", 4.10, 4.15, 0.05, 0.5)]
mine = [p.attempt(limit_price, *c) for c in cases]
mine = p.check("limit_price", mine, [p.limit_price(*c) for c in cases])
pd.DataFrame(cases, columns=["side", "bid", "ask", "tick", "aggressiveness"]).assign(limit=mine)"""

_IS_CHECK = """
cases = [("BUY", 100.0, [(100.10, 300), (100.20, 700)]), ("SELL", 100.0, [(99.90, 1000)], 5.0), ("BUY", 50.0, [(49.95, 100)])]
mine = [p.attempt(implementation_shortfall_bps, *c) for c in cases]
mine = p.check("implementation_shortfall_bps", mine, [p.implementation_shortfall_bps(*c) for c in cases])
print(np.round(mine, 2))"""

NB["03_execution_tca"] = [
    header("03", "Spread-aware orders, execution algorithms and TCA", "S5–S6 (Spread-aware order management · Execution algorithms & TCA)",
           "1. Price a limit order inside the spread and round it to the tick in the safe direction.\n"
           "2. Simulate passive, mid and crossing orders on a quote path with informed flow, counting non-fills.\n"
           "3. Measure implementation shortfall against the decision price.\n"
           "4. Compare a single order with TWAP, VWAP and POV on a 60,000-share parent order."),
    ("code", SETUP),
    ("md", "## 1. Limit prices on the tick\n\n"
           "BUY at `bid + a·(ask − bid)`, SELL at `ask − a·(ask − bid)`, where `a` is the aggressiveness (0 passive, 1 cross). "
           "Round to the tick with `Decimal`, in the direction that never overpays: a BUY rounds **down** (`ROUND_DOWN`), a SELL "
           "rounds **up** (`ROUND_UP`). `(Decimal(str(px)) / q).to_integral_value(rounding=...)` gives the number of ticks."),
    ("md", YOUR_TURN),
    ("ex", _LP_HEAD + """    steps = ...                                                      # ✍️ whole ticks, rounded the safe way
""" + _LP_TAIL,
     _LP_HEAD + """    steps = (Decimal(str(px)) / q).to_integral_value(rounding=ROUND_DOWN if side == "BUY" else ROUND_UP)
""" + _LP_TAIL),
    ("md", "## 2. Passive orders look cheap until you count non-fills\n\n"
           "`p.quote_path` has **informed** order flow: a passive buy tends to fill when sellers are pushing the price down, and "
           "goes unfilled when the price runs away. `p.chase_report` sends 150 random orders per path under each schedule of "
           "aggressiveness levels (5 seconds each) and measures the fill rate, the cost of filled orders, the **markout** (the "
           "mid 30 s after the fill vs the fill price: negative = adverse selection), and the **all-in** cost, which charges an "
           "unfilled order the cost of crossing at the end. Averaged over three quote paths:"),
    ("code", """schedules = {"passive": (0.0,), "mid": (0.5,), "cross": (1.0,), "chase passive → cross": (0.0, 0.33, 0.67, 1.0)}
sweep = sum(p.chase_report(p.quote_path(seed=s), schedules, 150, seed=s) for s in range(3)) / 3
sweep.round(2)"""),
    ("md", "Passive orders look cheap when they fill (about −1.7 bps: they earn part of the spread), but the markout shows about "
           "half of that saving gone 30 seconds later: they fill when the price is moving against them (it keeps falling after "
           "a passive buy fills). They fill only about two times in three, and completing the rest costs the spread and then "
           "some. Measured all-in, the chase schedule is the cheapest and crossing at once the most expensive. Your backtest's "
           "cost model should use these measured numbers.\n\n"
           "## 3. Implementation shortfall\n\n"
           "For a parent order, TCA compares the average fill price with the **decision** price: "
           "`IS (bps) = 1e4 · (sign·(avg − decision)·qty + fees) / (decision·qty)`, with `sign = +1` for BUY, −1 for SELL, "
           "`qty` the total filled and `avg` the quantity-weighted average price. Positive = cost."),
    ("md", YOUR_TURN),
    ("ex", """def implementation_shortfall_bps(side, decision_px, fills, fees=0.0):
    qty = sum(q for _, q in fills)
    avg = ...                                                        # ✍️ quantity-weighted average fill price
    sign = 1 if side == "BUY" else -1
    return ...                                                       # ✍️
""" + _IS_CHECK,
     """def implementation_shortfall_bps(side, decision_px, fills, fees=0.0):
    qty = sum(q for _, q in fills)
    avg = sum(px * q for px, q in fills) / qty
    sign = 1 if side == "BUY" else -1
    return 1e4 * (sign * (avg - decision_px) * qty + fees) / (decision_px * qty)
""" + _IS_CHECK),
    ("md", "## 4. Slice big orders\n\n"
           "Buy 60,000 shares in a session whose volume is heavy at the open and close. Each child fill pays a temporary impact "
           "proportional to its share of that minute's volume (`p.execute_schedule`). Compare: everything in the first minute; "
           "TWAP (13 equal slices); VWAP (sized by the **previous** session's 30-minute volume profile, known before the open); "
           "POV (10% of the previous minute's volume until done). Five sessions:"),
    ("code", """def pov_schedule(qty, session, participation=0.1):
    remaining, out, vols = qty, [], session["volume"].to_numpy()
    for i in range(len(session)):
        child = 0 if i == 0 else p.pov_child_qty(vols[i - 1], participation, remaining)
        remaining -= child
        out.append(child)
    return pd.Series(out, index=session.index)

qty, results = 60_000, {k: [] for k in ("single", "TWAP", "VWAP", "POV 10%")}
for s in range(1, 6):
    sess, prev = p.intraday_session(s), p.intraday_session(s - 1)
    start, end = sess.index[0], sess.index[-1] + pd.Timedelta(minutes=1)
    profile = prev["volume"].resample("30min").sum()
    profile.index = pd.date_range(start, periods=len(profile), freq="30min")
    schedules = {"single": pd.Series([qty], index=[start]), "TWAP": p.twap_schedule(qty, start, end, 13),
                 "VWAP": p.vwap_schedule(qty, profile), "POV 10%": pov_schedule(qty, sess)}
    for name, sched in schedules.items():
        results[name].append(p.implementation_shortfall_bps("BUY", sess["price"].iloc[0], p.execute_schedule("BUY", sched, sess)))
pd.DataFrame({"mean IS (bps)": {k: np.mean(v) for k, v in results.items()}, "std (bps)": {k: np.std(v, ddof=1) for k, v in results.items()}}).round(1)"""),
    ("md", "One big order pays enormous impact. TWAP and VWAP cut it by more than two thirds, but they stay exposed to the "
           "price drifting all day, so their results vary a lot. POV finishes early in the heavy morning volume: less drift "
           "exposure, lower and steadier cost here. The right algo depends on urgency and on how much the price can move while "
           "you wait.\n\n"
           "## Wrap-up\n\n"
           "* Price inside the spread, round the safe way, and measure all-in cost including non-fills.\n"
           "* Informed flow makes passive fills adverse; markouts reveal it.\n"
           "* TCA against the decision price; feed the measured costs back into the backtest.\n"
           "* Graded version: `labs/part12/week42_execution` (`limit_price`, `simulate_chase`, `chase_report`, the schedules, "
           "`implementation_shortfall_bps`) and Clinic W2 (the execution-quality report)."),
]

# ---------------------------------------------------------------------------------------------- 04
_OPM_HEAD = """class MyOptionPositionManager(p.OptionPositionManager):
    def evaluate(self, pos):
        if pos["value"] >= pos["credit"] * (1 + self.stop):
            return "close", "stop"
        if pos["value"] <= pos["credit"] * (1 - self.tp):
            return "close", "take_profit"
        if pos["dte"] == 0:
            return "close", "expiry"
"""
_OPM_TAIL = """        if pos["dte"] <= self.roll_dte:
            return "roll", "dte"
        return None

base = {"credit": 2.0, "value": 1.6, "dte": 30, "short_deltas": [0.20, -0.18], "short_call_itm": False, "days_to_exdiv": None}
scenarios = {"nothing to do": base, "loss of 2× the credit": {**base, "value": 6.1}, "half the credit captured": {**base, "value": 0.9},
             "short delta 0.35": {**base, "short_deltas": [0.35, -0.10]}, "21 days left": {**base, "dte": 21},
             "short call ITM, ex-div tomorrow": {**base, "short_call_itm": True, "days_to_exdiv": 1, "short_deltas": [0.6, -0.1]},
             "ITM call, ex-div tomorrow AND big loss": {**base, "value": 6.5, "short_call_itm": True, "days_to_exdiv": 1}}
mine = {k: p.attempt(MyOptionPositionManager().evaluate, v) for k, v in scenarios.items()}
mine = p.check("OptionPositionManager.evaluate", mine, {k: p.OptionPositionManager().evaluate(v) for k, v in scenarios.items()})
pd.Series({k: v if v else "—" for k, v in mine.items()}, name="(action, reason)").to_frame()"""

_R_CHECK = """
journal = p.trade_journal()                                          # 400 trades with a PLANTED edge
mine = p.check("r_multiples", p.attempt(r_multiples, journal), p.r_multiples(journal))
journal = journal.assign(r_multiple=mine)
print(f"{int(mine.isna().sum())} trades have no stop, so no R;  mean R of the rest: {mine.mean():+.3f}")"""

NB["04_options_journal"] = [
    header("04", "Option position management and the trade journal", "S7–S8 (Option strategy integration · Trade journal & journal analytics)",
           "1. Manage a short-premium option position with prioritized rules: stop, take profit, expiry, ex-dividend, delta, roll.\n"
           "2. Measure every trade in R-multiples.\n"
           "3. Find the edge that only shows up by setup and regime.\n"
           "4. Catch rule violations in the journal."),
    ("code", SETUP),
    ("md", "## 1. Rules, in priority order\n\n"
           "A short-premium position (credit received, cost to close now = `value`) is checked every bar. The **order** matters: "
           "a position that is both at its stop and near ex-dividend must be closed as a stop. After stop, take profit and expiry "
           "(given): **ex-dividend** (a short call is in the money and ex-dividend is at most 1 day away: early-assignment risk → "
           "close), then **adjust** (any |short delta| above `self.max_delta`), then **roll** (days to expiry ≤ `self.roll_dte`)."),
    ("md", YOUR_TURN),
    ("ex", _OPM_HEAD + """        ...                                                          # ✍️ ex-dividend, then the delta adjustment
""" + _OPM_TAIL,
     _OPM_HEAD + """        dx = pos.get("days_to_exdiv")
        if pos.get("short_call_itm") and dx is not None and dx <= 1:
            return "close", "ex_dividend"
        if any(abs(d) > self.max_delta for d in pos["short_deltas"]):
            return "adjust", "short_delta"
""" + _OPM_TAIL),
    ("md", "## 2. R-multiples\n\n"
           "Measure every trade in units of the risk you **planned**: `R = side·(exit − entry) / |entry − stop|`. A trade without a "
           "stop has no defined risk, so its R is NaN (and it is a rule violation)."),
    ("md", YOUR_TURN),
    ("ex", """def r_multiples(journal):
    return ...                                                       # ✍️
""" + _R_CHECK,
     """def r_multiples(journal):
    return journal["side"] * (journal["exit"] - journal["entry"]) / (journal["entry"] - journal["stop"]).abs()
""" + _R_CHECK),
    ("md", "## 3. Where is the edge?\n\n"
           "`p.journal_stats` gives trades, win rate, average win and loss in R, and **expectancy** (mean R) per group. By setup "
           "alone, then by setup and regime:"),
    ("code", """display(p.journal_stats(journal).round(3))
p.journal_stats(journal, ["setup_tag", "regime"]).round(3)"""),
    ("md", "By setup the two look alike. Split by regime, breakouts earn a lot in trends and lose in chop, while pullbacks earn a "
           "little everywhere. The action: take breakouts only in the trend regime (a filter the regime model of Part 10 can "
           "supply), and journal the regime at entry so the split is possible.\n\n"
           "## 4. Rule violations\n\n"
           "The journal is also a compliance record: trades without a stop, entries outside market hours, positions larger than "
           "planned (`p.rule_violations`):"),
    ("code", """violations = p.rule_violations(journal)
display(violations["rule"].value_counts().to_frame("trades"))
violations.head()"""),
    ("md", "## Wrap-up\n\n"
           "* Option management rules are prioritized; test combinations, not single conditions.\n"
           "* R-multiples make trades comparable; no stop, no R.\n"
           "* Edges hide in segments: journal the setup and the regime.\n"
           "* Graded version: `labs/part12/week42_execution` (`OptionPositionManager`, `r_multiples`, `journal_stats`, "
           "`rule_violations`)."),
]

# ---------------------------------------------------------------------------------------------- 05
_SLO_CHECK = """
lat = pd.Series(p.latency_samples())
cases = [(lat, 0.1), (lat, 0.06, 0.99), (lat, 0.05, 0.9), (pd.Series([0.01, 0.02, 0.2, 0.03]), 0.1, 0.75)]
mine = [p.attempt(slo_report, *c) for c in cases]
mine = p.check("slo_report", mine, [p.slo_report(*c) for c in cases])
pd.DataFrame(mine, index=["ack ≤ 100 ms, 99.9%", "ack ≤ 60 ms, 99%", "ack ≤ 50 ms, 90%", "toy"]).round(3)"""

_VER_HEAD = """class MyAuditLog(p.AuditLog):
    def verify(self):
        prev = "0" * 64
        for i, rec in enumerate(self.records):
            body = {k: v for k, v in rec.items() if k != "hash"}
"""
_VER_TAIL = """            prev = rec["hash"]
        return -1

def filled_log(cls):
    clock = iter(range(1000))
    log = cls(now=lambda: next(clock))
    for i in range(6):
        log.append("order", id=f"O{i}", qty=100)
        log.append("fill", order_id=f"O{i}", qty=100, price=100 + i)
    return log

intact, edited, relinked = filled_log(MyAuditLog), filled_log(MyAuditLog), filled_log(MyAuditLog)
edited.records[5]["data"]["price"] = 99                              # someone "fixes" a fill price
relinked.records[5]["data"]["price"] = 99
relinked.records[5]["hash"] = p._digest({k: v for k, v in relinked.records[5].items() if k != "hash"})   # and re-hashes it
mine = p.check("AuditLog.verify", [p.attempt(log.verify) for log in (intact, edited, relinked)],
               [p.AuditLog.verify(log) for log in (intact, edited, relinked)])
print(f"intact chain: {mine[0]};  edited record 5: breaks at {mine[1]};  edited and re-hashed: breaks at {mine[2]} (the NEXT record's link)")"""

_REC_CHECK = """
cases = [({"SPY": 300, "QQQ": -100}, {"SPY": 300, "QQQ": -100}), ({"SPY": 300, "QQQ": -100}, {"SPY": 200, "QQQ": -100, "IWM": 50}),
         ({"SPY": 100}, {})]
mine = [p.attempt(reconcile_positions, *c) for c in cases]
mine = p.check("reconcile_positions", mine, [p.reconcile_positions(*c) for c in cases])
pd.DataFrame(mine[1])"""

NB["05_observability_audit"] = [
    header("05", "Observability, the audit chain, reconciliation and approvals",
           "S9–S10 (Observability: logs, metrics, dashboards · Reconciliation, audit & approvals)",
           "1. Log with correlation ids and expose Prometheus metrics; evaluate alert rules and SLOs.\n"
           "2. Make history tamper-evident with a hash-chained audit log.\n"
           "3. Reconcile positions and orders with the broker, which is always right.\n"
           "4. Require a second person for risky changes (four eyes)."),
    ("code", SETUP),
    ("md", "## 1. Logs, metrics, alerts\n\n"
           "Rule of the week: if it isn't logged, measured and reconciled, it didn't happen. Structured JSON logs carry a "
           "**correlation id** that follows one idea from signal to fill; metrics are counters, gauges and histograms that "
           "Prometheus scrapes:"),
    ("code", """from prometheus_client import CollectorRegistry

clock = iter(range(100))
logger = p.StructuredLogger(now=lambda: next(clock))
for msg, fields in (("signal", {"strategy": "mom", "symbol": "SPY", "side": "BUY"}), ("intent", {"qty": 100}),
                    ("order", {"order_id": "O17", "limit": 512.31}), ("fill", {"order_id": "O17", "price": 512.30})):
    logger.log("info", msg, cid="c-7f3a", **fields)
logger.log("info", "heartbeat", cid="c-0000")
print("the story of one trade:", [r["msg"] for r in logger.trace("c-7f3a")])

registry = CollectorRegistry()
metrics = p.make_metrics(registry)
metrics["orders"].labels("mom", "ib").inc(3)
metrics["rejects"].labels("ib").inc()
for x in p.latency_samples()[:50]:
    metrics["latency"].observe(x)
metrics["staleness"].labels("polygon").set(2.4)
print("\\n".join(line for line in p.exposition(registry).splitlines() if line.startswith(("qf_orders_total{", "qf_rejects_total{", "qf_signal_to_submit_seconds_count", "qf_data_staleness"))))

rules = [("data stale", lambda s: s["staleness_s"] > 5, "critical"), ("reject rate high", lambda s: s["reject_rate"] > 0.05, "warning"),
         ("latency p99 high", lambda s: s["p99_s"] > 0.1, "warning"), ("disk low", lambda s: s["disk_free"] < 0.15, "info")]
print("\\nalerts firing:", p.evaluate_alerts({"staleness_s": 8, "reject_rate": 0.25, "p99_s": 0.08, "disk_free": 0.1}, rules))"""),
    ("md", "An **SLO** says \"the value stays under a threshold for a target share of the time\". Report the compliance (share "
           "of samples at or under the threshold), whether the target is met, and how much of the **error budget** is used: the "
           "share of bad samples divided by the allowed share `1 − target` (1.0 = the whole budget spent)."),
    ("md", YOUR_TURN),
    ("ex", """def slo_report(values, threshold, target=0.999):
    ok = ...                                                         # ✍️ share of samples at or under the threshold
    return {"compliance": ok, "met": ok >= target, "budget_used": ...}   # ✍️
""" + _SLO_CHECK,
     """def slo_report(values, threshold, target=0.999):
    ok = float((values <= threshold).mean())
    return {"compliance": ok, "met": ok >= target, "budget_used": (1 - ok) / (1 - target)}
""" + _SLO_CHECK),
    ("md", "One 400 ms spike in 300 acks already spends more than the whole 99.9% budget: tight SLOs make rare problems visible.\n\n"
           "## 2. A hash-chained audit log\n\n"
           "Every record stores the previous record's hash (`prev`) and its own `hash` = sha256 of the record without its hash "
           "(`p._digest(body)`). To verify, walk the chain: a record is broken if its `prev` isn't the previous hash, or its hash "
           "doesn't match its body. Return the index of the first broken record, −1 if intact."),
    ("md", YOUR_TURN),
    ("ex", _VER_HEAD + """            if ...:                                                  # ✍️ broken link or wrong hash
                return i
""" + _VER_TAIL,
     _VER_HEAD + """            if rec["prev"] != prev or p._digest(body) != rec["hash"]:
                return i
""" + _VER_TAIL),
    ("md", "Editing a record breaks its hash; re-hashing it breaks the next record's link, and so on to the end. Sending the "
           "latest hash out every day (the `head()`) means even rewriting the whole tail can be detected.\n\n"
           "## 3. Reconciliation\n\n"
           "Compare our positions with the broker's over the union of symbols (missing = 0) and list every break: `{symbol, "
           "internal, broker, diff = broker − internal}`, sorted by symbol. The **broker** is the source of truth."),
    ("md", YOUR_TURN),
    ("ex", """def reconcile_positions(internal, broker, tol=1e-9):
    out = []
    for s in sorted(set(internal) | set(broker)):
        a, b = internal.get(s, 0.0), broker.get(s, 0.0)
        ...                                                          # ✍️ record a break if they differ
    return out
""" + _REC_CHECK,
     """def reconcile_positions(internal, broker, tol=1e-9):
    out = []
    for s in sorted(set(internal) | set(broker)):
        a, b = internal.get(s, 0.0), broker.get(s, 0.0)
        if abs(a - b) > tol:
            out.append({"symbol": s, "internal": a, "broker": b, "diff": b - a})
    return out
""" + _REC_CHECK),
    ("md", "## 4. Four eyes\n\n"
           "New strategies, parameter and limit changes, resuming a paused strategy and large orders need a second person. "
           "`p.ApprovalQueue` audits every step and refuses self-approval:"),
    ("code", """audit = p.AuditLog()
approvals = p.ApprovalQueue(audit)
aid = approvals.request("limit_change", {"max_gross": 1.5}, requester="alice", reason="new pairs book")
try:
    approvals.decide(aid, approver="alice", approve=True, reason_code="ok")
except ValueError as err:
    print("alice approving her own request:", err)
approvals.decide(aid, approver="bob", approve=True, reason_code="reviewed_backtest")
print(approvals.items[aid]["status"], "| audit:", [r["event"] for r in audit.records], "| chain intact:", audit.verify() == -1)
print("order of 5,000 × $120 needs a human:", p.needs_approval({"qty": 5000, "price": 120.0}, max_notional=250_000))"""),
    ("md", "## Wrap-up\n\n"
           "* Correlation ids tell the story of a trade; metrics feed alerts and SLOs with explicit error budgets.\n"
           "* The audit log is append-only and hash-chained; publish its head daily.\n"
           "* Reconcile with the broker; the broker wins; every break is explained.\n"
           "* Four eyes on anything that changes risk.\n"
           "* Graded version: `labs/part12/week43_monitoring` (`StructuredLogger`, `make_metrics`, `evaluate_alerts`, "
           "`slo_report`, `AuditLog`, `reconcile_positions`, `reconcile_orders`, `ApprovalQueue`)."),
]

# ---------------------------------------------------------------------------------------------- 06
_RZ_HEAD = """class MyRobustZ:
    def __init__(self, window=500, threshold=6.0, min_obs=50):
        self.buf, self.window, self.threshold, self.min_obs = [], window, threshold, min_obs

    def update(self, x):
        z = 0.0
        if len(self.buf) >= self.min_obs:                            # only the values BEFORE x, and only once warm
            arr = np.asarray(self.buf)
"""
_RZ_TAIL = """            z = (x - med) / mad
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        return z, abs(z) > self.threshold

def run(detector, xs):
    out = [detector.update(float(x)) for x in xs]
    return np.array([z for z, _ in out]), [i for i, (_, flag) in enumerate(out) if flag]

latencies = p.latency_samples()                                      # 300 order-ack latencies, one 400 ms spike at 200
z_raw, flags = p.check("RobustZ", p.attempt(run, MyRobustZ(), latencies), run(p.RobustZ(), latencies))
print(f"flagged: {flags}; z of the spike: {z_raw[200]:.1f}")"""

_BF_HEAD = """def book_fills(fills):
    seen, dups, pos = set(), [], {}
    for f in fills:
"""
_BF_TAIL = """        seen.add(f["fill_id"])
        pos[f["symbol"]] = pos.get(f["symbol"], 0) + f["qty"]
    return {k: v for k, v in pos.items() if v != 0}, dups

booked, dups = p.check("book_fills", p.attempt(book_fills, day["fills"]), p.book_fills(day["fills"]))
print("delivered twice:", dups, "| breaks after idempotent booking:", p.reconcile_positions(booked, day["broker_positions"]))"""

_BAND_HEAD = """def backtest_band(daily_returns, horizon=20, n_boot=2000, q=(0.05, 0.95), seed=0):
    r = np.asarray(daily_returns, dtype=float)
    rng = np.random.default_rng(seed)
"""
_BAND_TAIL = """    lo, hi = np.quantile(paths, q, axis=0)
    return pd.DataFrame({"lower": lo, "upper": hi}, index=range(1, horizon + 1))

bt, live = p.track_record()                                          # 1,000 backtest days and 20 live days per strategy
band = p.check("backtest_band", p.attempt(backtest_band, bt["options"]), p.backtest_band(bt["options"]))
band.iloc[[0, 4, 9, 19]]"""

NB["06_anomaly_controller_reports"] = [
    header("06", "Anomaly detection, the controller and reports", "S11–S12 (Anomaly detection & the controller · Reports & dashboards)",
           "1. Build a streaming robust z-score, and see why latency is scored on the log scale.\n"
           "2. Catch a slow drift that no single observation reveals, with CUSUM.\n"
           "3. Replay a trading day with injected faults: detect each one, book fills idempotently, let the controller act.\n"
           "4. Write the daily report, and judge a live record against the backtest's expected band."),
    ("code", SETUP),
    ("md", "## 1. A streaming robust z-score\n\n"
           "Mean and standard deviation are dragged around by the very outliers you are hunting. The **median** and the **MAD** "
           "(median absolute deviation, × 1.4826 so it matches σ for normal data) are not. The score of a new value uses only the "
           "values before it (the last `window`), and nothing is flagged until `min_obs` values have been seen."),
    ("md", YOUR_TURN),
    ("ex", _RZ_HEAD + """            med = ...                                                # ✍️ the median of the window
            mad = ...                                                # ✍️ the MAD × 1.4826, plus 1e-12 against a zero spread
""" + _RZ_TAIL,
     _RZ_HEAD + """            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826 + 1e-12
""" + _RZ_TAIL),
    ("md", "One flag, on the spike. Latencies are right-skewed, though, so the normal tail itself scores high on the raw scale. "
           "Compare the largest score on 20 clean days (no spike) with the log scale:"),
    ("code", """clean = pd.DataFrame([{"raw": np.abs(run(p.RobustZ(), x)[0]).max(), "log": np.abs(run(p.RobustZ(), np.log(x))[0]).max()}
                      for x in (p.latency_samples(spike_at=None, seed=s) for s in range(1, 21))])
z_log, flags_log = run(p.RobustZ(), np.log(latencies))
print(f"largest |z| on 20 clean days: raw up to {clean['raw'].max():.1f}, log up to {clean['log'].max():.1f} (the alarm is at 6)")
print(f"the spike: z = {z_raw[200]:.1f} raw, {z_log[200]:.1f} log; flagged on the log scale: {flags_log}")
print("fat-tail check, a −$12,000 day when the risk model says σ = $2,000:", p.fat_tail_error(-12_000, 2_000))"""),
    ("md", "On the raw scale an ordinary day already comes within a whisker of the alarm; on the log scale the clean days stay far "
           "below it and the spike still stands out. Score latency on the log scale. The fat-tail check is the P&L version of the "
           "same idea: a 6σ loss is more often a bad price, a missing position or a duplicated fill than a real market move, so the "
           "controller pauses everything and reconciles before anyone trusts the number.\n\n"
           "## 2. Slow drift: CUSUM\n\n"
           "A strategy's slippage creeping up by 0.8σ never produces one extreme value. CUSUM adds up the small excesses "
           "(`s⁺ = max(0, s⁺ + z − k)`, and the mirror image for drops) and alarms when the sum passes `h`, then starts again:"),
    ("code", """rng = np.random.default_rng(1)
x = rng.normal(0, 1, 400)
x[200:] += 0.8                                                       # from t = 200 the mean is 0.8 σ higher
cusum, rz = p.Cusum(mean=0, sd=1), p.RobustZ()
alarms = [t for t, v in enumerate(x) if cusum.update(v)]
flags = [t for t, v in enumerate(x) if rz.update(v)[1]]
print("RobustZ flags:", flags, "| CUSUM alarms:", alarms)

fig, ax = plt.subplots()
ax.plot(x, lw=0.7, color="#8a8984", label="observations")
ax.plot(pd.Series(x).rolling(20).mean().to_numpy(), label="20-point mean")
for t in alarms:
    ax.axvline(t, color=p.PALETTE[7], lw=0.8)
ax.axvline(200, color="k", ls="--", lw=1, label="the drift starts")
ax.set(title="CUSUM alarms (red lines) on a 0.8σ drift", xlabel="observation")
ax.legend(loc="upper left")
plt.show()"""),
    ("md", "The z-score never fires; CUSUM fires a few observations after the drift starts and keeps firing while it lasts. Each "
           "detector has a job: z-scores for spikes, CUSUM for drift.\n\n"
           "## 3. The monitoring drill\n\n"
           "`p.trading_day()` replays a session with five injected faults. `p.FAULTS` is the answer key. Each detector looks for "
           "one kind of fault:"),
    ("code", """day = p.trading_day()
print("injected (the answer key):", p.FAULTS)
dets = ([("bad_tick", t) for t in p.bad_ticks(day["ticks"])] + [("stale_data", t) for t in p.stale_gaps(day["ticks"])]
        + [("latency_spike", i) for i in p.stream_flags(np.log(day["acks"]))]
        + [("runaway_order_rate", i) for i in p.stream_flags(day["order_counts"])])
print("detected:", dets)

naive = {}
for f in day["fills"]:                                               # book every fill message as it arrives
    naive[f["symbol"]] = naive.get(f["symbol"], 0) + f["qty"]
print("naive booking vs the broker:", p.reconcile_positions(naive, day["broker_positions"]))"""),
    ("md", "Four faults caught where they were injected (the stale feed is reported at its last tick, 11,999, just before the "
           "30-second silence). The fifth hides in the fills: after a reconnect the broker re-sent "
           "one fill, and booking it twice leaves our position different from the broker's. Book **idempotently**: each "
           "`fill_id` once, and report the ids that arrived more than once."),
    ("md", YOUR_TURN),
    ("ex", _BF_HEAD + """        ...                                                          # ✍️ a fill_id seen before: record it and skip it
""" + _BF_TAIL,
     _BF_HEAD + """        if f["fill_id"] in seen:
            dups.append(f["fill_id"])
            continue
""" + _BF_TAIL),
    ("md", "Now the **controller**: a policy table maps each anomaly to an action (alert, pause the strategy, pause everything and "
           "reconcile, trip the kill switch), and every decision is audited. The drill adds two anomalies to the lesson-plan "
           "policy:"),
    ("code", """POLICY = {**p.CONTROLLER_POLICY, "bad_tick": "pause_strategy", "duplicate_fill": "pause_all_and_reconcile"}
audit = p.AuditLog()
approvals = p.ApprovalQueue(audit)
ctl = p.Controller(["mom", "pairs", "options"], audit, approvals, POLICY)
for anomaly, where in dets + [("duplicate_fill", d) for d in dups]:
    print(f"{anomaly:>20} at {str(where):>6} → {ctl.handle(anomaly, 'mom')}")
print("\\nkilled:", ctl.killed, "| paused:", sorted(ctl.paused), "| needs reconcile:", ctl.needs_reconcile,
      "| audit chain intact:", audit.verify() == -1)
aid = approvals.request("resume", {"strategy": "pairs"}, requester="alice", reason="false alarm")
approvals.decide(aid, approver="bob", approve=True, reason_code="checked")
print("resume pairs with an approved request:", ctl.resume("pairs", aid), "(nothing resumes while the kill switch is tripped)")

quiet = p.trading_day(faults=False)
print("a clean day:", p.bad_ticks(quiet["ticks"]) + p.stale_gaps(quiet["ticks"]) + p.stream_flags(np.log(quiet["acks"]))
      + p.stream_flags(quiet["order_counts"]), "detections,", p.book_fills(quiet["fills"])[1], "duplicates")"""),
    ("md", "Every fault detected, handled and audited, and a clean day raises nothing: a monitor that cries wolf gets muted, "
           "which is worse than no monitor. (The homework tunes thresholds on four weeks of paper data to stay under one false "
           "alarm a day.)\n\n"
           "## 4. Reports and the backtest band\n\n"
           "The **daily report** (sent by n8n after the close) covers P&L, return, drawdown, the number of fills and slippage "
           "against the arrival mid:"),
    ("code", """rng = np.random.default_rng(0)
fills = pd.DataFrame({"side": rng.choice([1, -1], 12), "qty": rng.integers(1, 6, 12) * 100,
                      "arrival_mid": 100 + rng.normal(0, 1, 12).cumsum()})
fills["price"] = fills["arrival_mid"] * (1 + fills["side"] * rng.normal(3, 2, 12) / 1e4)   # about 3 bps paid on each fill
equity = pd.Series([100_000, 101_200, 100_700, 99_900, 100_450.0])
{k: round(v, 4) for k, v in p.daily_report(fills, equity).items()}"""),
    ("md", "The monthly question is harder: *is the live record what the backtest promised?* Live results are short and noisy, so "
           "compare them with a **band**: resample the backtest's daily returns into many 20-day paths (with replacement), add "
           "each path up, and take the 5% and 95% quantiles for every day."),
    ("md", YOUR_TURN),
    ("ex", _BAND_HEAD + """    paths = ...                                                      # ✍️ (n_boot, horizon) resampled returns, summed along each path
""" + _BAND_TAIL,
     _BAND_HEAD + """    paths = rng.choice(r, size=(n_boot, horizon), replace=True).cumsum(axis=1)
""" + _BAND_TAIL),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, s in zip(axes, live.columns):
    b = p.backtest_band(bt[s], horizon=len(live))
    ax.fill_between(b.index, b["lower"], b["upper"], alpha=0.25, label="backtest 5–95%")
    ax.plot(b.index, live[s].cumsum().to_numpy(), marker="o", ms=3, label="live")
    ax.set(title=f"{s}: inside on {p.inside_band(live[s], b).mean():.0%} of days", xlabel="live day")
axes[0].set_ylabel("cumulative return")
axes[0].legend(loc="lower left")
plt.show()
p.attribution(live * 1_000_000).round(2)                            # daily P&L on $1m per strategy"""),
    ("md", "Momentum stays inside its band. Pairs runs *above* its band for a while: better than promised, which calls for a "
           "look (a data or accounting error can flatter too) rather than a demotion. The options strategy falls below its band "
           "within a week and keeps falling: something changed (the synthetic record has it bleeding 0.8% a day). Attribution "
           "needs the same care: with a losing total, the shares flip sign (options is almost twice the net loss, and the other "
           "two offset part of it), so read the totals first. A band turns \"it's just a bad "
           "week\" into a testable statement, and notebook 10 turns it into weekly scale / hold / demote decisions.\n\n"
           "## Wrap-up\n\n"
           "* Robust z-scores for spikes (latency on the log scale), CUSUM for drift, a fat-tail check on P&L.\n"
           "* Book fills idempotently; the controller maps anomalies to audited actions; resuming needs an approval.\n"
           "* A clean day must stay quiet.\n"
           "* Judge live results against a band built from the backtest, not against a single number.\n"
           "* Graded versions: `labs/part12/week43_monitoring` (`RobustZ`, `fat_tail_error`, `Cusum`, `Controller`, "
           "`daily_report`, `backtest_band`, `inside_band`, `attribution`) and the drill in "
           "`labs/part12/clinic_w3_monitoring_drill`."),
]

# ---------------------------------------------------------------------------------------------- 07
_LESSON_COMPOSE = '''LESSON_PLAN_COMPOSE = """
services:
  ib-gateway:
    image: ${IB_GATEWAY_IMAGE}
    env_file: ./secrets/ib.env
    environment: {TRADING_MODE: paper}
    restart: unless-stopped
  db:
    image: timescale/timescaledb:latest-pg16
    volumes: [dbdata:/var/lib/postgresql/data]
    env_file: ./secrets/db.env
    healthcheck: {test: ["CMD-SHELL", "pg_isready -U quantforge"], interval: 10s, retries: 5}
  redis:
    image: redis:7
  engine:
    build: ..
    env_file: ./secrets/engine.env
    depends_on:
      db: {condition: service_healthy}
    restart: unless-stopped
  api:
    build: ..
    env_file: ./secrets/engine.env
    depends_on: [db]
  n8n:
    image: n8nio/n8n
  prometheus:
    image: prom/prometheus
  grafana:
    image: grafana/grafana
"""'''

_CI_HEAD = """def check_ci(workflow):
"""
_CI_TAIL = """    return sorted(k for k, needle in p.REQUIRED_CI.items() if needle not in runs)

ci_text = (p.DEPLOY_DIR / "ci.yml").read_text()
ci = yaml.safe_load(ci_text)
no_regression = copy.deepcopy(ci)
no_regression["jobs"]["test"]["steps"] = [s for s in ci["jobs"]["test"]["steps"] if "regression" not in s.get("run", "")]
loosened = yaml.safe_load(ci_text.replace("mypy --strict", "mypy").replace(" --cov-fail-under=85", ""))
cases = {"committed ci.yml": ci, "regression step deleted": no_regression, "loosened": loosened}
mine = p.check("check_ci", {k: p.attempt(check_ci, w) for k, w in cases.items()}, {k: p.check_ci(w) for k, w in cases.items()})
mine"""

_REG_HEAD = """def regression_check(result, baseline, rtol=1e-9):
    bad = []
    for k in sorted(set(result) | set(baseline)):
"""
_REG_TAIL = """            bad.append(k)
    return bad

mine = p.check("regression_check", {k: p.attempt(regression_check, r, baseline) for k, r in results.items()},
               {k: p.regression_check(r, baseline) for k, r in results.items()})
pd.DataFrame(results).T.assign(changed=[", ".join(v) or "—" for v in mine.values()])"""

NB["07_deploy_ci_regression"] = [
    header("07", "Deployment, CI and backtest regression tests", "S13–S14 (Deployment with Docker Compose · CI/CD & the release process)",
           "1. Check a docker-compose file against production rules: pinned images, no secrets inline, restart policies, a "
           "database health check, one exposed port.\n"
           "2. Check that the CI workflow runs every required gate.\n"
           "3. Freeze a backtest's results as a regression baseline, and watch it catch bugs that look like improvements."),
    ("code", SETUP),
    ("md", "## 1. The compose file is code, so review it with code\n\n"
           "The paper stack runs one container per concern: IB Gateway, the engine, the API, PostgreSQL/TimescaleDB, Redis, n8n, "
           "Prometheus, Grafana, and a reverse proxy in front. `p.check_compose` encodes the S13 rules: images pinned to an exact "
           "version (never `latest`), secrets only through env files or `${...}` references, restart policies on the critical "
           "services, a health check on the database and dependents that wait for it, and no published ports except the "
           "proxy's. The committed `deploy/docker-compose.yml` passes. The lesson plan's teaching version does not, and neither "
           "does a 2 a.m. \"quick fix\":"),
    ("code", """import copy
import yaml

""" + _LESSON_COMPOSE + """

committed = yaml.safe_load((p.DEPLOY_DIR / "docker-compose.yml").read_text())
print("deploy/docker-compose.yml:", p.check_compose(committed) or "passes")
print("\\nthe lesson plan's teaching version:")
for problem in p.check_compose(yaml.safe_load(LESSON_PLAN_COMPOSE)):
    print("   ", problem)
hotfix = copy.deepcopy(committed)
hotfix["services"]["db"]["environment"] = {"POSTGRES_PASSWORD": "changeme123"}   # "just to get it running"
hotfix["services"]["db"]["ports"] = ["5432:5432"]                                  # "so I can connect from my laptop"
hotfix["services"]["engine"]["image"] = "quantforge:latest"
print("\\nthe quick fix:", p.check_compose(hotfix))"""),
    ("md", "Each finding is an incident waiting to happen: an unpinned image changes under you on the next pull, an inline "
           "password ends up in Git, a database port open to the internet gets scanned within hours.\n\n"
           "## 2. The CI gates\n\n"
           "The S14 pipeline must run lint (`ruff check`), strict types (`mypy --strict`), the architecture rules "
           "(`lint-imports`), tests with a coverage floor (`--cov-fail-under`) and the backtest regression tests "
           "(`tests/regression`). `p.REQUIRED_CI` maps each gate to the text that proves it runs. Collect the `run` command of "
           "every step of every job, and report the gates whose text is missing."),
    ("md", YOUR_TURN),
    ("ex", _CI_HEAD + """    runs = ...                                                       # ✍️ every step's "run" (default "") of every job, joined into one string
""" + _CI_TAIL,
     _CI_HEAD + """    runs = " \\n".join(step.get("run", "") for job in workflow.get("jobs", {}).values() for step in job.get("steps", []))
""" + _CI_TAIL),
    ("md", "Gates erode quietly: someone drops `--strict` to get a release out, and nobody notices until the type errors pile up. "
           "A check like this can run on the workflow file itself.\n\n"
           "## 3. Backtest regression tests\n\n"
           "Unit tests check small pieces; a **regression test** checks that behaviour didn't change: a frozen dataset and config "
           "must reproduce the stored results. Here the fixture is an SMA(10/40) crossover on `p.ohlcv()`, and the baseline is its "
           "total return, number of trades and exposure. Then five changes to the SMA: a harmless refactor and four bugs."),
    ("code", """import json

close = p.ohlcv()["close"].to_numpy()                               # the frozen fixture dataset
baseline = p.fixture_backtest(close)
print("stored baseline (tests/regression/baseline.json):", json.dumps(baseline))

def sma_cumsum(x, n):                                                # the same maths, computed from a running sum
    c = np.r_[0.0, np.cumsum(x)]
    return np.r_[np.full(n - 1, np.nan), (c[n:] - c[:-n]) / n]

variants = {"refactor: SMA from a running sum": sma_cumsum,
            "bug: window n − 1": lambda x, n: p.sma(x, n - 1),
            "bug: rolling(min_periods=1)": lambda x, n: pd.Series(x).rolling(n, min_periods=1).mean().to_numpy(),
            "bug: one-bar look-ahead": lambda x, n: np.r_[p.sma(x, n)[1:], np.nan]}
results = {name: p.fixture_backtest(close, sma_fn=f) for name, f in variants.items()}
results["bug: a key dropped from the report"] = {k: v for k, v in baseline.items() if k != "exposure"}"""),
    ("md", "Compare a result with the baseline: list (sorted) every key that is missing on either side, or whose values differ "
           "beyond `np.isclose(result, baseline, rtol=rtol, atol=1e-12)`. An empty list means behaviour is unchanged."),
    ("md", YOUR_TURN),
    ("ex", _REG_HEAD + """        if ...:                                                      # ✍️ missing on either side, or not close
""" + _REG_TAIL,
     _REG_HEAD + """        if k not in result or k not in baseline or not np.isclose(result[k], baseline[k], rtol=rtol, atol=1e-12):
""" + _REG_TAIL),
    ("md", "The refactor passes: its differences are floating-point noise. Every bug is caught, and the dangerous one is the "
           "look-ahead: it *raises* the total return and keeps the same number of trades, so in a code review it looks like an "
           "improvement. A changed baseline is never re-frozen silently: the pull request must explain why the numbers moved.\n\n"
           "**Release process:** semantic versions and a changelog; deploy to paper first; promote to live only after a paper "
           "soak; feature flags to switch a strategy off without a deploy; a written rollback procedure (the previous image tag "
           "is one line in the compose file, which is one more reason to pin it).\n\n"
           "## Wrap-up\n\n"
           "* The deployment and the pipeline are files, so they can be checked automatically.\n"
           "* Pinned images, secrets outside Git, restart policies, health checks, one exposed port.\n"
           "* Regression baselines catch behaviour changes that unit tests and reviewers miss.\n"
           "* Graded version: `labs/part12/week44_production` (`check_compose`, `check_ci`, `fixture_backtest`, "
           "`regression_check`)."),
]

# ---------------------------------------------------------------------------------------------- 08
_RB_HEAD = """def rebuild(events, state=None):
    state = state or {"positions": {}, "cash": 0.0, "last_seq": -1}
    for f in events:
"""
_RB_TAIL = """    return state

cases = {"from scratch": (journal,), "snapshot at 25, replay from 20": (journal[20:], p.rebuild(journal[:26])),
         "journal delivered twice": (journal + journal,)}
mine = p.check("rebuild", {k: p.attempt(rebuild, *a) for k, a in cases.items()}, {k: p.rebuild(*a) for k, a in cases.items()})
mine["from scratch"]"""

_SS_HEAD = """import re

def scan_secrets(files):
    out = []
    for name, text in files.items():
        for i, line in enumerate(text.splitlines(), start=1):
"""
_SS_TAIL = """    return sorted(out)

FILES = {"quantforge/config.py": 'BROKER_HOST = "127.0.0.1"\\npassword = "Tr4d1ng!2025"\\n',
         "deploy/secrets.example.env": "DB_PASSWORD=${DB_PASSWORD}\\nANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}\\n",
         "research/scratch.py": 'client = anthropic.Anthropic(api_key="' + "sk-ant-" + "api03-" + "x" * 24 + '")\\n',
         "quantforge/apps/api.py": 'token = request.headers["X-Token"]\\npassword = os.environ["API_PASSWORD"]\\n'}
p.check("scan_secrets", p.attempt(scan_secrets, FILES), p.scan_secrets(FILES))"""

_PC_HEAD = """def precautionary_check(order, last_price, limits):
    out = []
    if abs(order["qty"]) > limits["max_qty"]:
        out.append("size")
"""
_PC_TAIL = """    return out

LIMITS = {"max_qty": 2_000, "max_notional": 250_000, "band_pct": 0.03}
orders = {"normal": {"qty": 300, "price": 512.10}, "large but plausible": {"qty": 600, "price": 512.10},
          "fat finger: 30,000 shares": {"qty": 30_000, "price": 512.10}, "price typo: 51.21": {"qty": 300, "price": 51.21},
          "both": {"qty": 30_000, "price": 5121.0}}
mine = p.check("precautionary_check", {k: p.attempt(precautionary_check, o, 512.0, LIMITS) for k, o in orders.items()},
               {k: p.precautionary_check(o, 512.0, LIMITS) for k, o in orders.items()})
mine"""

NB["08_recovery_security"] = [
    header("08", "Recovery, backups and security", "S15–S16 (Reliability, disaster recovery & chaos testing · Security & operational risk)",
           "1. Rebuild the platform's state from its fill journal, idempotently, on top of an atomic snapshot.\n"
           "2. Crash the engine at many points and recover; trade again only after a clean reconciliation.\n"
           "3. Verify a backup's checksum before restoring it.\n"
           "4. Scan for secrets, rank threats, and add broker-side limits as a second line of defense."),
    ("code", SETUP),
    ("md", "## 1. State is a function of events\n\n"
           "The engine's state (positions, cash) is never edited in place: it is rebuilt from the **fill journal**, one fill at "
           "a time (`p.apply_fill`). Each fill has a sequence number, and the state remembers the last one applied, so replaying "
           "a fill that the state already holds must do nothing. That makes recovery safe: replay generously, and duplicates "
           "cannot double-count."),
    ("code", """rng = np.random.default_rng(0)
journal = [{"seq": i, "symbol": str(rng.choice(["SPY", "QQQ", "IWM"])), "qty": int(rng.choice([-100, 100])),
            "price": round(float(rng.uniform(95, 105)), 2)} for i in range(40)]
journal[:3]"""),
    ("md", YOUR_TURN),
    ("ex", _RB_HEAD + """        ...                                                          # ✍️ apply (p.apply_fill) only fills newer than state["last_seq"]
""" + _RB_TAIL,
     _RB_HEAD + """        if f["seq"] > state["last_seq"]:
            state = p.apply_fill(state, f)
""" + _RB_TAIL),
    ("md", "## 2. Crash and recover\n\n"
           "The engine writes each fill to the journal first, and snapshots its state every 5 fills with `p.save_snapshot` "
           "(write to a temporary file, then rename: a crash mid-write never leaves half a snapshot). `p.recover` loads the "
           "snapshot, replays the journal on top, and reconciles with the broker; trading resumes only if nothing breaks. Kill "
           "the engine at ten different points:"),
    ("code", """import tempfile

work = Path(tempfile.mkdtemp())
rows = []
for crash in range(3, 40, 4):                                       # the engine dies right after fill `crash`
    snap = work / f"state_{crash}.json"
    last_snap = crash // 5 * 5 - 1                                   # snapshots after seq 4, 9, 14, …
    if last_snap >= 0:
        p.save_snapshot(p.rebuild(journal[: last_snap + 1]), snap)
    written = journal[: crash + 1]                                   # the journal survived the crash
    broker = p.rebuild(written)["positions"]                         # the broker's truth
    out = p.recover(snap, written, broker)
    rows.append({"crash after seq": crash, "snapshot at seq": last_snap if last_snap >= 0 else "none",
                 "state correct": out["state"] == p.rebuild(written), "may trade": out["may_trade"]})
pd.DataFrame(rows).set_index("crash after seq").T"""),
    ("md", "Correct at every crash point, including the one before the first snapshot. Now someone places a manual order in TWS "
           "while the engine is down:"),
    ("code", """broker = dict(p.rebuild(journal)["positions"])
broker["IWM"] = broker.get("IWM", 0) + 100                           # a manual buy, placed outside the platform
out = p.recover(work / "state_39.json", journal, broker)
print("breaks:", out["breaks"], "| may trade:", out["may_trade"])
print("open orders:", p.reconcile_orders({"O101", "O102"}, {"O102", "TWS-5531"}))"""),
    ("md", "The platform refuses to trade until a human explains the break. That is the rule: the broker is the truth, and an "
           "unexplained difference means our model of the world is wrong.\n\n"
           "**Backups** are only real if they restore. Store the checksum apart from the file, and verify it *before* restoring:"),
    ("code", """tables = {"positions": p.rebuild(journal)["positions"], "orders": [{"id": "O101", "status": "filled"}], "date": "2025-06-02"}
blob, digest = p.backup(tables)
print(f"backup: {len(blob)} bytes, sha256 {digest[:16]}…; restored equal: {p.restore(blob, digest) == tables}")
corrupt = blob[:10] + bytes([blob[10] ^ 1]) + blob[11:]              # one flipped bit on the disk
try:
    p.restore(corrupt, digest)
except ValueError as err:
    print("restoring the damaged copy:", err)"""),
    ("md", "Recovery targets: the **RTO** (how long until trading again) and the **RPO** (how much data you may lose). The journal "
           "plus snapshots keep the RPO at zero fills; the drill measures the RTO. Chaos tests (kill a container, drop the network, "
           "fill the disk, skew the clock) run in paper, and every failure gets a one-page runbook.\n\n"
           "## 3. Security\n\n"
           "Secrets in code end up in Git history, backups and screenshots. A scanner runs in CI over every file, line by line: "
           "report `(file, line number from 1, rule name)` for every `p.SECRET_RULES` pattern that matches a line, sorted."),
    ("md", YOUR_TURN),
    ("ex", _SS_HEAD + """            ...                                                      # ✍️ for each rule, pattern in p.SECRET_RULES: record a match
""" + _SS_TAIL,
     _SS_HEAD + """            for rule, pat in p.SECRET_RULES.items():
                if re.search(pat, line):
                    out.append((name, i, rule))
""" + _SS_TAIL),
    ("md", "The hard-coded password and the pasted API key are found; `${...}` references and values read from the environment "
           "are not. A found key is **rotated**, not just deleted: it is still in the history.\n\n"
           "A **threat model** lists what can go wrong, how likely (1–5) and how bad (1–5); risk = likelihood × impact, and the top "
           "three get fixed first:"),
    ("code", """threats = pd.DataFrame([("bad data feed", 4, 4, "RobustZ on ticks, a second feed to cross-check"),
                        ("runaway strategy", 3, 5, "order-rate limit, kill switch"),
                        ("stolen API key", 2, 5, "IP allow-list, 2FA, rotation, separate paper/live keys"),
                        ("VPS down at the open", 3, 3, "restart policies, runbook, broker-side stops"),
                        ("gateway stuck at login", 4, 2, "IBC daily restart, pre-market check"),
                        ("clock drift", 2, 2, "chrony, alert on skew")],
                       columns=["threat", "likelihood", "impact", "control"])
p.rank_threats(threats)"""),
    ("md", "The platform's risk engine is the first line of defense. The second is at the broker (IB's precautionary settings): "
           "limits on order size, notional and distance from the last price, which hold even if the platform has a bug. Return "
           "the failed checks in the order `size`, `notional`, `price_band` (limit price more than `band_pct` from the last "
           "price)."),
    ("md", YOUR_TURN),
    ("ex", _PC_HEAD + """    ...                                                              # ✍️ "notional" (|qty| · price), then "price_band"
""" + _PC_TAIL,
     _PC_HEAD + """    if abs(order["qty"]) * order["price"] > limits["max_notional"]:
        out.append("notional")
    if abs(order["price"] / last_price - 1) > limits["band_pct"]:
        out.append("price_band")
""" + _PC_TAIL),
    ("md", "## Wrap-up\n\n"
           "* State is rebuilt from the journal; replay is idempotent; snapshots are atomic.\n"
           "* Recovery ends with a reconciliation, and trading waits for a clean one.\n"
           "* A backup is verified before it is restored, and restores are tested.\n"
           "* No secrets in code; rank threats by likelihood × impact; broker-side limits behind the risk engine.\n"
           "* Graded version: `labs/part12/week44_production` (`apply_fill`, `rebuild`, `save_snapshot`, `recover`, `backup`, "
           "`restore`, `scan_secrets`, `rank_threats`, `precautionary_check`)."),
]

# ---------------------------------------------------------------------------------------------- 09
_CR_HEAD = """PLAN = {"min": 0.1, "max": 1.0, "step": 0.1, "weeks_per_step": 2, "max_dd": -0.03}

def capital_ramp(fraction, weeks_in_band, drawdown, plan):
    if drawdown < plan["max_dd"]:
"""
_CR_TAIL = """        return min(fraction + plan["step"], plan["max"])
    return fraction

cases = [(0.1, 2, -0.01), (0.1, 3, -0.01), (0.5, 4, -0.02), (0.4, 6, -0.05), (0.15, 0, -0.04), (1.0, 8, 0.0), (0.2, 0, 0.0)]
mine = p.check("capital_ramp", [p.attempt(capital_ramp, *c, PLAN) for c in cases], [p.capital_ramp(*c, PLAN) for c in cases])
pd.DataFrame(cases, columns=["fraction", "weeks in band", "drawdown"]).assign(next_week=mine)"""

_ST_HEAD = """def session_times(day):
    d = pd.Timestamp(day).normalize()
"""
_ST_TAIL = """        return None
    close = pd.Timedelta(hours=13) if d in p.EARLY_CLOSE else pd.Timedelta(hours=16)
    return d + pd.Timedelta(hours=9, minutes=30), d + close

days = [*pd.date_range("2025-07-02", "2025-07-07"), *pd.date_range("2025-11-26", "2025-11-28"), pd.Timestamp("2025-12-24")]
mine = p.check("session_times", {str(d.date()): p.attempt(session_times, d) for d in days},
               {str(d.date()): p.session_times(d) for d in days})
pd.DataFrame({day: {"open": t[0].time() if t else None, "close": t[1].time() if t else None} for day, t in mine.items()}).T"""

_PM_HEAD = """def premarket_check(status):
    res = {k: bool(f(status)) for k, f in p.PREMARKET.items()}
    failed = {k for k, ok in res.items() if not ok}
"""
_PM_TAIL = """
good = {"gateway_connected": True, "data_age_s": 1.2, "open_breaks": 0, "kill_switch_tested_today": True,
        "risk_limits_loaded": True, "last_backup_age_h": 20, "disk_free_pct": 40}
cases = {"a normal morning": good,
         "after a long weekend": {**good, "kill_switch_tested_today": False, "last_backup_age_h": 50},
         "feed down, disk filling": {**good, "data_age_s": 300, "disk_free_pct": 9}}
mine = p.check("premarket_check", {k: p.attempt(premarket_check, s) for k, s in cases.items()},
               {k: p.premarket_check(s) for k, s in cases.items()})
pd.DataFrame({k: {c: v[c] for c in ("blocking_failures", "warnings", "may_trade")} for k, v in mine.items()}).T"""

NB["09_rollout_operations"] = [
    header("09", "Rollout, capital ramp and daily operations", "S17–S18 (Rollout plan & capital ramp · Daily operations)",
           "1. Move a strategy through paper → shadow → small live → scaled live on evidence, and demote it on evidence.\n"
           "2. Ramp capital by written rules, and cut it automatically after a drawdown.\n"
           "3. Know the trading calendar: holidays and early closes.\n"
           "4. Automate the pre-market checklist, so a failed blocking check stops trading."),
    ("code", SETUP),
    ("md", "## 1. Stages, promotion and demotion\n\n"
           "**Paper** tests the plumbing, **shadow** runs on live data and logs orders without sending them, **small live** trades "
           "minimum size, **scaled live** the planned size. `p.rollout_decision` applies written rules:\n\n"
           "* **Demote** one stage if fewer than half of the days were inside the backtest band, a drift alarm fired, or there "
           "were 3 or more incidents.\n"
           "* **Promote** one stage only if *every* criterion holds: enough days at this stage, no sev1 incident, no open "
           "reconciliation break, slippage at most 1.5× the model, and at least 80% of the days inside the band.\n"
           "* Otherwise **hold**, and say which criteria failed."),
    ("code", """base = dict(days=12, sev1_incidents=0, incidents=0, open_breaks=0, slippage_ratio=1.1, inside_band=0.9, drift_alarm=False)
cases = {"paper, 12 clean days": ("paper", base),
         "shadow, 12 clean days": ("shadow", base),
         "small live, 12 days": ("small_live", base),
         "small live, 25 days, a sev1 and 1.8× slippage": ("small_live", {**base, "days": 25, "sev1_incidents": 1, "slippage_ratio": 1.8}),
         "small live, 25 days, 40% inside the band": ("small_live", {**base, "days": 25, "inside_band": 0.4}),
         "scaled live, a drift alarm": ("scaled_live", {**base, "days": 60, "drift_alarm": True})}
pd.DataFrame({k: dict(zip(["action", "new stage", "reasons"], p.rollout_decision(*c))) for k, c in cases.items()}).T"""),
    ("md", "Rules written before going live stop the two classic mistakes: scaling up after a lucky week, and \"giving it one more "
           "week\" while a broken strategy bleeds.\n\n"
           "## 2. The capital ramp\n\n"
           "Size follows a plan too (`PLAN`). A drawdown worse than `max_dd` halves the capital fraction, but not below `min`. "
           "Otherwise, every full `weeks_per_step` weeks inside the band add `step`, up to `max`: increase only when "
           "`weeks_in_band` is a positive multiple of `weeks_per_step`. Anything else: no change."),
    ("md", YOUR_TURN),
    ("ex", _CR_HEAD + """        return ...                                                   # ✍️ halve, but not below plan["min"]
    if ...:                                                          # ✍️ a positive multiple of plan["weeks_per_step"]
""" + _CR_TAIL,
     _CR_HEAD + """        return max(fraction / 2, plan["min"])
    if weeks_in_band > 0 and weeks_in_band % plan["weeks_per_step"] == 0:
""" + _CR_TAIL),
    ("code", """weeks = pd.DataFrame({"in_band": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      "drawdown": [0, 0, -0.01, 0, 0, -0.01, -0.045, -0.02, -0.01, 0, 0, 0, 0, -0.01, 0, 0]}, index=range(1, 17))
frac, streak, path = 0.1, 0, []
for w, row in weeks.iterrows():
    streak = streak + 1 if row["in_band"] else 0
    frac = p.capital_ramp(frac, streak, row["drawdown"], PLAN)
    path.append(frac)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.step(weeks.index, path, where="post")
ax.axvline(7, color=p.PALETTE[7], ls="--", lw=1, label="week 7: outside the band, drawdown −4.5%")
ax.set(title="Capital fraction under the ramp rules", xlabel="week", ylabel="share of planned capital")
ax.legend(loc="upper left")
plt.show()
print("fraction by week:", [round(f, 2) for f in path])"""),
    ("md", "Slow up, fast down: two weeks in the band for each step up; one bad week halves the size and resets the count.\n\n"
           "## 3. The trading calendar\n\n"
           "The session is 09:30–16:00 New York time, or 09:30–13:00 on early-close days (`p.EARLY_CLOSE`); there is no session "
           "on weekends and exchange holidays (`p.HOLIDAYS`). A strategy that assumes 16:00 on the day after Thanksgiving "
           "sends its closing orders three hours after the market closed."),
    ("md", YOUR_TURN),
    ("ex", _ST_HEAD + """    if ...:                                                          # ✍️ a weekend (weekday() >= 5) or a holiday
""" + _ST_TAIL,
     _ST_HEAD + """    if d.weekday() >= 5 or d in p.HOLIDAYS:
""" + _ST_TAIL),
    ("md", "## 4. The pre-market checklist\n\n"
           "Every morning, before the open, n8n triggers a script that runs every check in `p.PREMARKET`. Failing any check in "
           "`p.BLOCKING` (gateway connected, data fresh, reconciliation clean, kill switch tested today, risk limits loaded) blocks "
           "trading; the others (backup age, free disk) are warnings. Return the results, the sorted blocking failures, the "
           "sorted warnings, and whether trading may start."),
    ("md", YOUR_TURN),
    ("ex", _PM_HEAD + """    return {"results": res, "blocking_failures": ..., "warnings": ..., "may_trade": ...}   # ✍️
""" + _PM_TAIL,
     _PM_HEAD + """    return {"results": res, "blocking_failures": sorted(failed & p.BLOCKING), "warnings": sorted(failed - p.BLOCKING),
            "may_trade": not (failed & p.BLOCKING)}
""" + _PM_TAIL),
    ("md", "An untested kill switch is not a kill switch, so that check blocks; an old backup is serious but does not make the "
           "day's trading unsafe, so it warns. **Intraday:** watch the dashboards, handle alerts, no ad-hoc parameter changes. "
           "**End of day:** reconcile, review the journal, send the daily report, confirm the backup, log incidents.\n\n"
           "## Wrap-up\n\n"
           "* Stages with written promotion and demotion rules, applied the same way every time.\n"
           "* Capital ramps slowly and is cut automatically.\n"
           "* The calendar knows holidays and early closes.\n"
           "* The pre-market script's blocking failures stop trading; warnings get fixed the same day.\n"
           "* Graded version: `labs/part12/week45_golive` (`rollout_decision`, `capital_ramp`, `session_times`, "
           "`premarket_check`)."),
]

# ---------------------------------------------------------------------------------------------- 10
_INC_HEAD = """class MyIncident(p.Incident):
    def advance(self, phase, at):
        i = p.PHASES.index(self.phase)
"""
_INC_TAIL = """            raise ValueError(f"{self.phase} cannot go to {phase}")
        at = pd.Timestamp(at)
        if at < self.times[self.phase]:
            raise ValueError("time goes forward")
        self.times[phase] = at

def drill(cls):
    inc = cls("runaway order loop in mom", "sev1", "2025-11-12 10:02")
    log = []
    for phase, at in (("contained", "10:05"), ("recovered", "10:20"), ("diagnosed", "10:31"), ("recovered", "10:48"),
                      ("reconciled", "11:15"), ("closed", "09:00"), ("closed", "16:30")):
        try:
            inc.advance(phase, f"2025-11-12 {at}")
            log.append(f"{at} {phase}")
        except ValueError as err:
            log.append(f"{at} refused: {err}")
    return log, inc.sla_breaches()

log, breaches = p.check("Incident.advance", p.attempt(drill, MyIncident), drill(p.Incident))
print("\\n".join(log))
print("SLA breaches:", breaches)"""

_WD_HEAD = """def weekly_decisions(inside, incidents):
    out = {}
    for s in inside.columns:
        share, inc = inside[s].mean(), incidents.get(s, 0)
"""
_WD_TAIL = """    return out

incidents = {3: {"pairs": 1}}                                        # week 3: one incident on pairs (a partial fill on one leg)
weeks = {w: (inside.iloc[(w - 1) * 5: w * 5], incidents.get(w, {})) for w in range(1, 5)}
mine = p.check("weekly_decisions", {w: p.attempt(weekly_decisions, *a) for w, a in weeks.items()},
               {w: p.weekly_decisions(*a) for w, a in weeks.items()})
decisions = pd.DataFrame(mine).T.rename_axis("week")
decisions"""

_OW_HEAD = """def orders_without_risk_approval(records):
    approved, bad = set(), []
    for r in records:
        if r["event"] == "risk_decision" and r["data"].get("approved"):
            approved.add(r["data"]["order_id"])
"""
_OW_TAIL = """            bad.append(r["data"]["id"])
    return bad

def audit_trail(n_orders=30, skip=(), late=(), rejected=()):
    log = p.AuditLog()
    for i in range(n_orders):
        oid = f"O{i:03d}"
        if oid in late:                                              # approved only AFTER it was sent
            log.append("order", id=oid)
            log.append("risk_decision", order_id=oid, approved=True)
            continue
        if oid not in skip:
            log.append("risk_decision", order_id=oid, approved=oid not in rejected)
        log.append("order", id=oid)
    log.append("killswitch_trip", reason="defense demo")
    return log.records

trails = {"clean": audit_trail(), "one order skipped the risk engine": audit_trail(skip={"O017"}),
          "approved after sending": audit_trail(late={"O023"}), "rejected but sent anyway": audit_trail(rejected={"O008"})}
p.check("orders_without_risk_approval", {k: p.attempt(orders_without_risk_approval, r) for k, r in trails.items()},
        {k: p.orders_without_risk_approval(r) for k, r in trails.items()})"""

NB["10_incidents_track_record"] = [
    header("10", "Incidents, the go-live review and the track record",
           "S19–S20 (Incident response drills · Capstone kickoff & go-live review) and weeks 6–8 (S21–S24: weekly reviews, "
           "final review, defense)",
           "1. Run an incident through its lifecycle and check the response-time SLAs.\n"
           "2. Build a post-mortem timeline from the audit trail.\n"
           "3. Hold the go-live review on evidence.\n"
           "4. Review a 4-week live record every week against each strategy's backtest band: scale, hold or demote, and ramp "
           "capital.\n"
           "5. Check the graduation requirements from the audit chain."),
    ("code", SETUP),
    ("md", "## 1. The incident lifecycle\n\n"
           "Detect → contain (pause or kill) → diagnose → recover → reconcile → close, in that order and forward in time. Each "
           "severity has response targets (`p.SLA_MINUTES`): a sev1 must be contained within 5 minutes and reconciled within 60. "
           "Allow only the **next** phase; refuse anything else."),
    ("md", YOUR_TURN),
    ("ex", _INC_HEAD + """        if ...:                                                      # ✍️ there is no next phase, or `phase` is not it
""" + _INC_TAIL,
     _INC_HEAD + """        if i + 1 >= len(p.PHASES) or p.PHASES[i + 1] != phase:
""" + _INC_TAIL),
    ("md", "Recovering before diagnosing is refused, and so is a time earlier than the previous phase (09:00 typed for 16:30). "
           "The team contained "
           "the loop in 3 minutes, but reconciling took 73 minutes against a 60-minute target: that is an action item for the "
           "post-mortem.\n\n"
           "## 2. The post-mortem timeline\n\n"
           "A **blameless** post-mortem starts from facts, and the audit chain has them. `p.postmortem_timeline` cuts the records "
           "of the incident window, in time order:"),
    ("code", """clock = iter(pd.Timestamp("2025-11-12 09:58") + pd.to_timedelta([0, 220, 261, 262, 263, 540, 1380, 2760, 4380, 9000], unit="s"))
audit = p.AuditLog(now=lambda: next(clock))
for event, data in [("order", {"id": "O900"}), ("alert", {"anomaly": "runaway_order_rate", "rate": 60}),
                    ("controller_action", {"action": "trip_kill_switch"}), ("killswitch_trip", {"by": "controller"}),
                    ("orders_cancelled", {"n": 57}), ("diagnosis", {"cause": "retry loop on a rejected order"}),
                    ("fix_deployed", {"version": "1.4.1"}), ("reconciled", {"breaks": 0}),
                    ("approval_granted", {"kind": "resume", "approver": "bob"}), ("daily_report", {})]:
    audit.append(event, **data)
timeline = p.postmortem_timeline(audit.records, "2025-11-12 10:00", "2025-11-12 11:30")
print("audit chain intact:", audit.verify() == -1)
pd.DataFrame(timeline, columns=["time", "event"])"""),
    ("md", "The report around this timeline: summary, impact (orders, P&L, time without trading), root cause, what went well, what "
           "didn't, and action items with owners, such as a retry limit, and a check that a rejected order is never resent "
           "unchanged. Blameless means it asks *how the system let this happen*, not *who did it*.\n\n"
           "## 3. The go-live review\n\n"
           "Approval for small live is based on evidence only: every strategy passed the Part 8 validation gate (DSR ≥ 0.95, PBO "
           "< 0.5), at least 10 runbooks, every drill passed, the security checklist complete, a written rollout plan:"),
    ("code", """evidence = {"strategies": {"momentum": {"dsr": 0.97, "pbo": 0.2}, "pairs": {"dsr": 0.96, "pbo": 0.3}, "options": {"dsr": 0.95, "pbo": 0.4}},
            "runbooks": 10, "drills": {"disconnect": True, "runaway": True, "bad_tick": True},
            "security_complete": True, "rollout_plan": True}
print("the evidence pack:", p.go_live_review(evidence))
weak = {**evidence, "strategies": {**evidence["strategies"], "options": {"dsr": 0.91, "pbo": 0.55}}, "runbooks": 7,
        "drills": {**evidence["drills"], "db_outage": False}}
print("a weaker pack:     ", p.go_live_review(weak))"""),
    ("md", "## 4. Four weeks live: the weekly reviews\n\n"
           "The track record starts. `p.track_record()` has three strategies; each week, the review compares each strategy's "
           "cumulative live return with its backtest band. Only falling **below** the band counts against a strategy "
           "(`lower_only=True`): beating it calls for an investigation, not a demotion."),
    ("code", """bt, live = p.track_record()
inside = pd.DataFrame({s: p.inside_band(live[s], p.backtest_band(bt[s], horizon=len(live)), lower_only=True).to_numpy()
                       for s in live.columns}, index=live.index)
print("share of live days inside the band:", inside.mean().round(2).to_dict())"""),
    ("md", "Each weekly review decides per strategy, from that week's 5 days and incidents: **demote** if inside on fewer than "
           "half of the days or 2+ incidents; **scale** if inside every day with no incident; otherwise **hold**."),
    ("md", YOUR_TURN),
    ("ex", _WD_HEAD + """        out[s] = ...                                                 # ✍️ "demote", "scale" or "hold"
""" + _WD_TAIL,
     _WD_HEAD + """        out[s] = "demote" if share < 0.5 or inc >= 2 else "scale" if share == 1.0 and inc == 0 else "hold"
""" + _WD_TAIL),
    ("md", "The capital follows the decisions and the S17 ramp: a demotion sends a strategy back to paper (capital 0, and it "
           "stays there until it re-qualifies); otherwise `p.capital_ramp` with the number of consecutive \"scale\" weeks and the "
           "drawdown so far:"),
    ("code", """PLAN = {"min": 0.1, "max": 1.0, "step": 0.1, "weeks_per_step": 2, "max_dd": -0.03}
capital = {}
for s in decisions.columns:
    frac, streak, path = 0.1, 0, []
    for w, dec in decisions[s].items():
        cum = pd.concat([pd.Series([0.0]), live[s].iloc[: w * 5].cumsum()], ignore_index=True)
        drawdown = float((cum - cum.cummax()).iloc[-1])
        if dec == "demote" or frac == 0:
            frac = 0.0
        else:
            streak = streak + 1 if dec == "scale" else 0
            frac = p.capital_ramp(frac, streak, drawdown, PLAN)
        path.append(frac)
    capital[s] = path
pd.DataFrame(capital, index=decisions.index).rename_axis("capital fraction after the review of week")"""),
    ("md", "The options strategy is demoted at the week-2 review, while it still runs at the 10% starting size, so the lesson "
           "is cheap. Momentum and pairs earn their steps up slowly; one day below the band (momentum, week 2) or one incident "
           "(pairs, week 3) is enough to hold and restart the count. Every decision, with its reasons, goes into the audit "
           "log.\n\n"
           "## 5. Graduation\n\n"
           "Four requirements are checked from the evidence, not from slides: the kill switch was demonstrated, **no order "
           "bypassed the risk engine**, every end-of-day reconciliation break is explained, and every strategy has a DSR. The "
           "second comes from the audit chain: list the ids of `order` records that were not preceded by an approved "
           "`risk_decision` for that order."),
    ("md", YOUR_TURN),
    ("ex", _OW_HEAD + """        elif ...:                                                    # ✍️ an order whose id was not approved before it
""" + _OW_TAIL,
     _OW_HEAD + """        elif r["event"] == "order" and r["data"]["id"] not in approved:
""" + _OW_TAIL),
    ("code", """dsr = {"momentum": 0.97, "pairs": 0.96, "options": 0.95}
print("the capstone:", p.graduation(trails["clean"], {"2025-11-07": 0, "2025-11-14": 1}, {"2025-11-14"}, dsr))
print("a weaker one:", p.graduation(trails["one order skipped the risk engine"][:-1], {"2025-11-07": 0, "2025-11-14": 1, "2025-11-21": 2},
                                    {"2025-11-14"}, {**dsr, "options": None}))"""),
    ("md", "## Wrap-up: the capstone\n\n"
           "* Incidents follow a lifecycle with SLAs; post-mortems are blameless and built from the audit trail.\n"
           "* Go-live, weekly decisions, capital and graduation are all decided on evidence, with rules written in advance.\n"
           "* **S21–S22 weekly reviews:** performance against the band, attribution, reconciliation breaks, TCA, alerts and "
           "incidents, journal findings, and the decisions with their reasons.\n"
           "* **S23–S24:** the 4-week results, lessons learned, and the defense: a 30-minute presentation, a live demo (including "
           "the kill switch) and questions from the panel.\n"
           "* Graded versions: `labs/part12/week45_golive` (`Incident`, `postmortem_timeline`, `go_live_review`, "
           "`weekly_decisions`, `orders_without_risk_approval`, `graduation`) and the dossier in "
           "`labs/part12/clinic_w5_track_record`."),
]


def build():
    (ROOT / "solutions").mkdir(exist_ok=True)
    for name, cells in NB.items():
        for kind in ("starter", "solution"):
            n = nbf.v4.new_notebook()
            n.metadata.update(KERNEL)
            out = []
            if kind == "solution":
                out.append(nbf.v4.new_markdown_cell("> **INSTRUCTOR SOLUTIONS** — do not share with learners before the session."))
            for c in cells:
                if c[0] == "md":
                    out.append(nbf.v4.new_markdown_cell(c[1]))
                elif c[0] == "code":
                    out.append(nbf.v4.new_code_cell(c[1]))
                else:
                    cell = nbf.v4.new_code_cell(c[1] if kind == "starter" else c[2])
                    cell.metadata["tags"] = ["exercise"]
                    out.append(cell)
            n.cells = out
            path = ROOT / (f"{name}.ipynb" if kind == "starter" else f"solutions/{name}_solution.ipynb")
            nbf.write(n, path)
    print(f"built {len(NB)} starter + {len(NB)} solution notebooks")


if __name__ == "__main__":
    build()
