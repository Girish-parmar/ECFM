"""Build the Part 4 guided notebooks (starter versions) and the instructor solutions.

Run from notebooks/part04:  python tools/build_notebooks.py
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
for d in (Path.cwd(), Path.cwd().parent):       # p4lib.py is in notebooks/part04/
    sys.path.insert(0, str(d))
from decimal import Decimal
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import p4lib as p

p.use_course_style()"""

YOUR_TURN = "✍️ **Your turn** — replace each `...` and run the cell. `p.check` tells you if you are right."


def header(num, title, sessions, goals):
    return ("md", f"""# Part 4 · Notebook {num} — {title}

**Sessions:** {sessions} · [Lesson plan](../../docs/lessons/PART_04_BROKER_CONNECTIVITY.md) · graded labs in [`labs/part04/`](../../labs/part04/)

**You will:**
{goals}

How these notebooks work: the setup, data and plotting code is written for you. Cells marked **✍️ Your turn** need a few lines from you.
If your answer does not match yet, the notebook continues with the reference answer so nothing else breaks.
Nothing here connects to a broker: the account rows, bars, ticks and order events are synthetic, shaped like what `ib_async` and `alpaca-py` return.""")


NB = {}

# ---------------------------------------------------------------------------------------------- 01
NB["01_env_config_secrets"] = [
    header("01", "Environment, config and secrets", "S1 (Reproducible environment) · S2 (IB architecture)",
           "1. Load settings from environment variables and keep secrets out of printouts.\n"
           "2. Map IB Gateway / TWS ports and refuse a live port in a paper environment.\n"
           "3. Write the secret scanner a pre-commit hook runs before every commit.\n"
           "4. See why every process needs its own IB `clientId`."),
    ("code", SETUP),
    ("md", "## 1. Settings come from the environment, never from the code\n\n"
           "In the project, `pydantic-settings` reads `QF_*` variables from `.env`. Here we pass a dict to the same kind of model. "
           "`SecretStr` keeps the key out of `print`, logs and tracebacks; you ask for it explicitly with `.get_secret_value()`."),
    ("code", """env = {"QF_ENV": "paper", "QF_IB_PORT": "4002", "QF_IB_CLIENT_ID": "11",
       "QF_ALPACA_KEY": "paper-key-id", "QF_ALPACA_SECRET": "paper-secret-value", "HOME": "/home/me"}
settings = p.settings_from_env(env)
print(settings)                                   # the secrets print as '**********'
print(repr(settings.alpaca_secret))
print("ib_port is an int:", type(settings.ib_port).__name__, settings.ib_port)
print("the real value, only when you ask:", settings.alpaca_secret.get_secret_value()[:6] + "…")"""),
    ("md", "## 2. IB ports: which door are you knocking on?\n\n"
           "| App | Paper | Live |\n|---|---|---|\n| IB Gateway | 4002 | 4001 |\n| TWS | 7497 | 7496 |\n\n"
           "The port is the only thing between a notebook experiment and a real-money order. Know them by heart."),
    ("md", YOUR_TURN),
    ("ex", """PORTS = {("gateway", "paper"): ..., ("gateway", "live"): ...,   # ✍️ fill in the four ports
         ("tws", "paper"): ..., ("tws", "live"): ...}

def ib_port(app: str, mode: str) -> int:
    return PORTS[(app, mode)]

combos = [(a, m) for a in ("gateway", "tws") for m in ("paper", "live")]
mine = [ib_port(a, m) for a, m in combos]
mine = p.check("IB ports", mine, [p.ib_port(a, m) for a, m in combos])
dict(zip(combos, mine))""",
     """PORTS = {("gateway", "paper"): 4002, ("gateway", "live"): 4001,
         ("tws", "paper"): 7497, ("tws", "live"): 7496}

def ib_port(app: str, mode: str) -> int:
    return PORTS[(app, mode)]

combos = [(a, m) for a in ("gateway", "tws") for m in ("paper", "live")]
mine = [ib_port(a, m) for a, m in combos]
mine = p.check("IB ports", mine, [p.ib_port(a, m) for a, m in combos])
dict(zip(combos, mine))"""),
    ("md", "Now the guard that runs at startup. A paper (or backtest) environment must **never** reach a live port. "
           "A live environment on a paper port is also a mistake: you'd think you were trading and you're not."),
    ("md", YOUR_TURN),
    ("ex", """def port_problem(env: str, port: int) -> str | None:
    live, paper = {4001, 7496}, {4002, 7497}
    if port not in live | paper:
        return "unknown port"
    # ✍️ return "live port in a paper environment" for env paper/backtest on a live port,
    # ✍️ and "paper port in a live environment" for env live on a paper port
    ...
    return None

cases = [("paper", 4002), ("paper", 4001), ("backtest", 7496), ("live", 7497), ("live", 4001), ("paper", 5000)]
mine = [port_problem(e, port) for e, port in cases]
mine = p.check("port guard", mine, [p.port_problem(e, port) for e, port in cases])
list(zip(cases, mine))""",
     """def port_problem(env: str, port: int) -> str | None:
    live, paper = {4001, 7496}, {4002, 7497}
    if port not in live | paper:
        return "unknown port"
    if env in ("paper", "backtest") and port in live:
        return "live port in a paper environment"
    if env == "live" and port in paper:
        return "paper port in a live environment"
    return None

cases = [("paper", 4002), ("paper", 4001), ("backtest", 7496), ("live", 7497), ("live", 4001), ("paper", 5000)]
mine = [port_problem(e, port) for e, port in cases]
mine = p.check("port guard", mine, [p.port_problem(e, port) for e, port in cases])
list(zip(cases, mine))"""),
    ("code", """def connect(settings):
    problem = p.port_problem(settings.env, settings.ib_port)
    if problem:
        raise RuntimeError(f"refusing to connect: {problem}")
    return f"would connect to {settings.ib_host}:{settings.ib_port} as clientId {settings.ib_client_id}"

print(connect(settings))
try:
    connect(p.settings_from_env({**env, "QF_IB_PORT": "4001"}))
except RuntimeError as e:
    print("🛑", e)"""),
    ("md", "## 3. A secret scanner for pre-commit\n\n"
           "Leaked keys are the most common real incident in retail algo trading. The scanner flags two things:\n"
           "* an **Alpaca key id**: `PK` or `AK` followed by 18 capitals/digits (`p.ALPACA_KEY_ID`);\n"
           "* an **assignment** such as `secret = ...` / `api_key: ...` with a value of 8+ characters (`p.ASSIGNMENT`), "
           "unless the value is a placeholder: it starts with `<`, `$` or `{`, or is `changeme`.\n\n"
           "The regex group 1 of `ASSIGNMENT` is the value."),
    ("md", YOUR_TURN),
    ("ex", """def find_secrets(text: str) -> list[tuple[int, str]]:
    hits = []
    for n, line in enumerate(text.splitlines(), start=1):
        if ...:                                   # ✍️ p.ALPACA_KEY_ID found anywhere on the line
            hits.append((n, "alpaca_key_id"))
            continue
        m = p.ASSIGNMENT.search(line)
        if m and ...:                             # ✍️ ...and the value m.group(1) is not a placeholder
            hits.append((n, "assignment"))
    return hits

for name, text in p.SAMPLE_FILES.items():
    print(f"--- {name}")
    print(text)
mine = {name: find_secrets(text) for name, text in p.SAMPLE_FILES.items()}
mine = p.check("find_secrets", mine, {name: p.find_secrets(text) for name, text in p.SAMPLE_FILES.items()})
mine""",
     """def find_secrets(text: str) -> list[tuple[int, str]]:
    hits = []
    for n, line in enumerate(text.splitlines(), start=1):
        if p.ALPACA_KEY_ID.search(line):
            hits.append((n, "alpaca_key_id"))
            continue
        m = p.ASSIGNMENT.search(line)
        if m and not (m.group(1)[0] in "<${" or m.group(1).lower() == "changeme"):
            hits.append((n, "assignment"))
    return hits

for name, text in p.SAMPLE_FILES.items():
    print(f"--- {name}")
    print(text)
mine = {name: find_secrets(text) for name, text in p.SAMPLE_FILES.items()}
mine = p.check("find_secrets", mine, {name: p.find_secrets(text) for name, text in p.SAMPLE_FILES.items()})
mine"""),
    ("md", "Notice what the scanner **misses**: `api_key = 'abc'` is too short to flag, and a key split over two lines passes. "
           "Regex scanners catch the common mistake; they are not a guarantee. The real defence is that keys only ever live in `.env` "
           "(git-ignored) or a secret store."),
    ("md", "## 4. One `clientId` per process\n\n"
           "IB allows several API connections to one Gateway, but each needs a different `clientId`. "
           "A second connection with an id already in use gets **error 326** and is dropped, often mid-session. "
           "Allocate ids in config, per process, and assert at startup."),
    ("code", """from collections import Counter

processes = {"strategy_momentum": 11, "downloader": 12, "monitor": 13, "notebook": 11}
clash = {cid: [name for name, c in processes.items() if c == cid]
         for cid, n in Counter(processes.values()).items() if n > 1}
print("clientId clashes:", clash or "none")
print("→ the second of", clash.get(11), "to connect gets error 326 and loses its connection")"""),
    ("md", "## Wrap-up\n\n"
           "* Settings from the environment, secrets as `SecretStr`, and `.env` in `.gitignore`.\n"
           "* The port is the paper/live switch: guard it at startup.\n"
           "* A pre-commit secret scan catches the classic leak before it reaches Git history.\n"
           "* Next: the graded version is `labs/part04/week13_foundations` (`ib_port`, `check_port_matches_env`, `find_secrets`)."),
]

# ---------------------------------------------------------------------------------------------- 02
NB["02_contracts_and_accounts"] = [
    header("02", "Contracts, symbols and accounts", "S3 (`ib_async` core) · S4 (Alpaca core)",
           "1. Give every instrument one canonical symbol, whichever broker it came from.\n"
           "2. See what an IB contract needs for stocks, futures, FX and crypto.\n"
           "3. Turn IB's string account rows into exact `Decimal` numbers.\n"
           "4. Read an Alpaca account for the warnings that should stop a strategy from starting."),
    ("code", SETUP),
    ("md", "## 1. One symbol per instrument\n\n"
           "Each broker names things its own way (`EUR.USD` vs `EUR/USD`, `ESZ6` vs `ES 202612`). "
           "Our data store and order book use one **canonical** symbol:\n\n"
           "| Asset class | Example | Rule |\n|---|---|---|\n"
           "| Equity | `AAPL` | the ticker |\n"
           "| Future | `ESZ6` | root + month code + last digit of the year (expiry `YYYYMM`) |\n"
           "| FX | `EUR.USD` | base `.` quote |\n"
           "| Crypto | `BTC/USD` | base `/` currency |\n\n"
           "Futures month codes, January to December: `F G H J K M N Q U V X Z` (in `p.MONTH_CODES`)."),
    ("code", """pd.DataFrame([vars(i) for i in p.INSTRUMENTS])"""),
    ("md", YOUR_TURN),
    ("ex", """def canonical_symbol(inst: p.Instrument) -> str:
    match inst.asset_class:
        case "EQUITY":
            return inst.symbol
        case "FUTURE":
            return ...                            # ✍️ root + p.MONTH_CODES[month - 1] + last digit of the year
        case "FX":
            return f"{inst.symbol[:3]}.{inst.symbol[3:]}"
        case "CRYPTO":
            return ...                            # ✍️ base "/" currency
    raise ValueError(inst.asset_class)

mine = [canonical_symbol(i) for i in p.INSTRUMENTS]
mine = p.check("canonical symbols", mine, [p.canonical_symbol(i) for i in p.INSTRUMENTS])
mine""",
     """def canonical_symbol(inst: p.Instrument) -> str:
    match inst.asset_class:
        case "EQUITY":
            return inst.symbol
        case "FUTURE":
            return inst.symbol + p.MONTH_CODES[int(inst.expiry[4:6]) - 1] + inst.expiry[3]
        case "FX":
            return f"{inst.symbol[:3]}.{inst.symbol[3:]}"
        case "CRYPTO":
            return f"{inst.symbol}/{inst.currency}"
    raise ValueError(inst.asset_class)

mine = [canonical_symbol(i) for i in p.INSTRUMENTS]
mine = p.check("canonical symbols", mine, [p.canonical_symbol(i) for i in p.INSTRUMENTS])
mine"""),
    ("md", "## 2. What IB needs to identify a contract\n\n"
           "`ib_async` builds these with `Stock(...)`, `Future(...)`, `Forex(...)`, `Crypto(...)`. The fields that trip people up:\n"
           "* stocks route via `SMART` but need `primaryExchange` to be unambiguous;\n"
           "* futures need the contract month, and you **trade** a specific month (`ContFuture` is for data only);\n"
           "* FX is `CASH` on `IDEALPRO`, with the *quote* currency as `currency`."),
    ("code", """pd.DataFrame({p.canonical_symbol(i): p.ib_contract_fields(i) for i in p.INSTRUMENTS}).T.fillna("")"""),
    ("md", "## 3. IB account values are strings, in several currencies\n\n"
           "`ib.accountValues()` returns rows like `AccountValue(account, tag, value, currency, modelCode)`. "
           "Every value is a **string**, the same tag appears once per currency plus a `BASE` line, and many tags aren't numbers at all."),
    ("code", """rows = p.ib_account_rows()
pd.DataFrame(rows)"""),
    ("md", YOUR_TURN + "\n\nKeep the rows whose tag is in `p.SUMMARY_TAGS` **and** whose currency matches; convert the value with `Decimal`."),
    ("ex", """def account_summary(rows: list[dict], currency: str = "USD") -> dict[str, Decimal]:
    return ...                                    # ✍️ {tag: Decimal(value)} for the summary tags in `currency`

mine = account_summary(rows)
mine = p.check("account summary", mine, p.ib_account_summary(rows))
mine""",
     """def account_summary(rows: list[dict], currency: str = "USD") -> dict[str, Decimal]:
    return {r["tag"]: Decimal(r["value"]) for r in rows if r["tag"] in p.SUMMARY_TAGS and r["currency"] == currency}

mine = account_summary(rows)
mine = p.check("account summary", mine, p.ib_account_summary(rows))
mine"""),
    ("code", """naive = {r["tag"]: float(r["value"]) for r in rows if r["tag"] in p.SUMMARY_TAGS}   # no currency filter
print("NetLiquidation, naive:", naive["NetLiquidation"], " ← the EUR row came last and overwrote the USD value")
print("NetLiquidation, right:", p.ib_account_summary(rows)["NetLiquidation"])"""),
    ("md", "## 4. Alpaca: read the account before you trade\n\n"
           "`TradingClient.get_account()` returns the account with flags that must stop a strategy at startup:\n"
           "* `status` other than `ACTIVE`, `account_blocked`, `trading_blocked`;\n"
           "* the **pattern day trader** rule: flagged as PDT with equity under $25,000 means no more day trades;\n"
           "* buying power at or below zero.\n\n"
           "Return the warnings **in that order**: `'status <status>'`, `'account blocked'`, `'trading blocked'`, "
           "`'PDT with equity below $25,000'`, `'no buying power'`."),
    ("md", YOUR_TURN),
    ("ex", """def account_warnings(acct: dict) -> list[str]:
    out = []
    if acct["status"] != "ACTIVE":
        out.append(f"status {acct['status']}")
    if acct["account_blocked"]:
        out.append("account blocked")
    if acct["trading_blocked"]:
        out.append("trading blocked")
    if ...:                                       # ✍️ PDT flag and equity (a string!) below 25,000
        out.append("PDT with equity below $25,000")
    if ...:                                       # ✍️ buying power at or below zero
        out.append("no buying power")
    return out

mine = {name: account_warnings(a) for name, a in p.ALPACA_ACCOUNTS.items()}
mine = p.check("Alpaca account warnings", mine, {name: p.alpaca_account_warnings(a) for name, a in p.ALPACA_ACCOUNTS.items()})
mine""",
     """def account_warnings(acct: dict) -> list[str]:
    out = []
    if acct["status"] != "ACTIVE":
        out.append(f"status {acct['status']}")
    if acct["account_blocked"]:
        out.append("account blocked")
    if acct["trading_blocked"]:
        out.append("trading blocked")
    if acct["pattern_day_trader"] and Decimal(acct["equity"]) < 25000:
        out.append("PDT with equity below $25,000")
    if Decimal(acct["buying_power"]) <= 0:
        out.append("no buying power")
    return out

mine = {name: account_warnings(a) for name, a in p.ALPACA_ACCOUNTS.items()}
mine = p.check("Alpaca account warnings", mine, {name: p.alpaca_account_warnings(a) for name, a in p.ALPACA_ACCOUNTS.items()})
mine"""),
    ("md", "## 5. Both accounts in one table\n\nThis is Exercise 1 of the lesson plan, on synthetic data. In Clinic W1 you run the same table against your real paper accounts with `labs/part04/paper/connect_both.py`."),
    ("code", """ib = p.ib_account_summary(rows)
alp = p.ALPACA_ACCOUNTS["healthy"]
table = pd.DataFrame({
    "IB (DU…567)": {"equity": ib["NetLiquidation"], "cash": ib["TotalCashValue"], "buying power": ib["BuyingPower"]},
    "Alpaca": {"equity": Decimal(alp["equity"]), "cash": Decimal(alp["cash"]), "buying power": Decimal(alp["buying_power"])},
})
table.map(lambda d: f"${d:,.2f}")"""),
    ("md", "## Wrap-up\n\n"
           "* Canonical symbols let data and orders from both brokers meet in one place.\n"
           "* Account numbers arrive as strings: parse them to `Decimal`, and filter by currency.\n"
           "* Check account flags **before** the first order, not after the first rejection.\n"
           "* Graded version: `labs/part04/week13_foundations` (`canonical_symbol`, `to_ib_contract` with real `ib_async` objects, `ib_account_summary`, `alpaca_account_warnings`)."),
]

# ---------------------------------------------------------------------------------------------- 03
NB["03_historical_data"] = [
    header("03", "Historical data, pacing and one schema", "S5 (IB historical data) · S6 (Alpaca historical data & the canonical schema)",
           "1. Split a long IB download into requests that walk back in time.\n"
           "2. Pace requests so IB never bans you (error 162), and see what pacing costs.\n"
           "3. Normalize IB and Alpaca bars into one canonical, UTC schema.\n"
           "4. See the two classic data bugs: naive time zones and IEX vs consolidated volume."),
    ("code", SETUP + "\nfrom datetime import datetime"),
    ("md", "## 1. Walk back in chunks\n\n"
           "An IB historical request is *\"`durationStr` of bars ending at `endDateTime`\"*, and each bar size has a maximum duration. "
           "A long history is therefore many requests, **walking back** from the end. The last chunk is clipped at the start date."),
    ("md", YOUR_TURN),
    ("ex", """from datetime import timedelta

def ib_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    out, cur = [], end
    while cur > start:
        s = ...                                   # ✍️ chunk_days before cur, but never before start
        out.append((s, cur))
        cur = s
    return out

args = (datetime(2025, 1, 1), datetime(2025, 3, 7), 30)
mine = p.attempt(ib_chunks, *args)
mine = p.check("ib_chunks", mine, p.ib_chunks(*args))
[(a.date().isoformat(), b.date().isoformat()) for a, b in mine]""",
     """from datetime import timedelta

def ib_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    out, cur = [], end
    while cur > start:
        s = max(start, cur - timedelta(days=chunk_days))
        out.append((s, cur))
        cur = s
    return out

args = (datetime(2025, 1, 1), datetime(2025, 3, 7), 30)
mine = p.attempt(ib_chunks, *args)
mine = p.check("ib_chunks", mine, p.ib_chunks(*args))
[(a.date().isoformat(), b.date().isoformat()) for a, b in mine]"""),
    ("md", "## 2. Pacing: how fast may you ask?\n\n"
           "IB's historical-data pacing rules (lesson plan S5): **no more than 6 requests in any 2 seconds**, **no more than 60 in any 10 minutes**, "
           "and no identical request within 15 seconds. Break them and you get **error 162** and a temporary ban.\n\n"
           "A pacer takes the times you *want* to send requests and returns the times you *may*: a request waits while the last `max_n` sends "
           "are all inside the window, until the oldest of them leaves it (`sent[-max_n] + window`)."),
    ("md", YOUR_TURN),
    ("ex", """def paced_times(desired: list[float], max_n: int, window: float) -> list[float]:
    sent = []
    for t in desired:
        t = max(t, sent[-1]) if sent else t       # keep the order
        # ✍️ if at least max_n were sent and t is inside the window of the max_n-th last send, wait for it to leave
        ...
        sent.append(t)
    return sent

burst = [0.0] * 20 + [5.0] * 5
mine = paced_times(burst, 6, 2.0)
mine = p.check("paced_times", mine, p.paced_times(burst, 6, 2.0))
mine""",
     """def paced_times(desired: list[float], max_n: int, window: float) -> list[float]:
    sent = []
    for t in desired:
        t = max(t, sent[-1]) if sent else t       # keep the order
        if len(sent) >= max_n and t < sent[-max_n] + window:
            t = sent[-max_n] + window
        sent.append(t)
    return sent

burst = [0.0] * 20 + [5.0] * 5
mine = paced_times(burst, 6, 2.0)
mine = p.check("paced_times", mine, p.paced_times(burst, 6, 2.0))
mine"""),
    ("md", "What does pacing cost? Say you download two years of 5-minute bars for 10 symbols in 1-week chunks: 1,040 requests, all wanted *now*."),
    ("code", """want = [0.0] * 1040
fast = p.paced_multi(want, [(6, 2.0)])
both = p.paced_multi(want, [(6, 2.0), (60, 600.0)])
fig, ax = plt.subplots()
ax.plot(np.array(fast) / 60, np.arange(1, 1041), label="6 per 2 s only")
ax.plot(np.array(both) / 60, np.arange(1, 1041), label="6 per 2 s and 60 per 10 min")
ax.set(xlabel="minutes after start", ylabel="requests sent", title="Pacing turns a burst into a staircase")
ax.legend(); plt.show()
print(f"6/2s only: {fast[-1] / 60:.1f} min.   With the 10-minute rule: {both[-1] / 3600:.1f} hours.")
print("→ download once into a local store and read from it (notebook 04); never re-download in a backtest loop.")"""),
    ("md", "## 3. Two brokers, one schema\n\n"
           "Every bar in our store has the columns `p.CANON` = `ts, symbol, open, high, low, close, volume, source`, with **`ts` a tz-aware UTC timestamp**.\n\n"
           "* IB (`util.df(bars)`) gives a `date` column in **exchange local time, without a time zone**.\n"
           "* Alpaca (`BarSet.df`) gives a MultiIndex `(symbol, timestamp)` already in UTC, with extra columns."),
    ("code", """ib_raw, alp_raw = p.raw_bars()
display(ib_raw.head(3))
display(alp_raw.head(3))"""),
    ("md", YOUR_TURN + "\n\nFor IB: localize the naive times to New York (`p.NY`) and convert to UTC (`.dt.tz_localize(...).dt.tz_convert(...)`)."),
    ("ex", """def normalize_ib(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.rename(columns={"date": "ts"})
    out["ts"] = ...                               # ✍️ naive New York local time → tz-aware UTC
    out["symbol"], out["source"] = symbol, "ib"
    out["volume"] = out["volume"].astype(float)
    return out[p.CANON].sort_values("ts").reset_index(drop=True)

mine = p.attempt(normalize_ib, ib_raw, "SPY")
mine = p.check("normalize_ib", mine, p.normalize_ib(ib_raw, "SPY"))
mine.head()""",
     """def normalize_ib(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.rename(columns={"date": "ts"})
    out["ts"] = pd.to_datetime(out["ts"]).dt.tz_localize(p.NY).dt.tz_convert(p.UTC)
    out["symbol"], out["source"] = symbol, "ib"
    out["volume"] = out["volume"].astype(float)
    return out[p.CANON].sort_values("ts").reset_index(drop=True)

mine = p.attempt(normalize_ib, ib_raw, "SPY")
mine = p.check("normalize_ib", mine, p.normalize_ib(ib_raw, "SPY"))
mine.head()"""),
    ("md", YOUR_TURN + "\n\nFor Alpaca: move the index levels into columns (`reset_index`) and rename `timestamp` to `ts`."),
    ("ex", """def normalize_alpaca(df: pd.DataFrame, feed: str = "iex") -> pd.DataFrame:
    out = ...                                     # ✍️ index levels → columns, timestamp → ts
    out["source"] = f"alpaca_{feed}"
    out["volume"] = out["volume"].astype(float)
    return out[p.CANON].sort_values("ts").reset_index(drop=True)

mine = p.attempt(normalize_alpaca, alp_raw)
mine = p.check("normalize_alpaca", mine, p.normalize_alpaca(alp_raw))
mine.head()""",
     """def normalize_alpaca(df: pd.DataFrame, feed: str = "iex") -> pd.DataFrame:
    out = df.reset_index().rename(columns={"timestamp": "ts"})
    out["source"] = f"alpaca_{feed}"
    out["volume"] = out["volume"].astype(float)
    return out[p.CANON].sort_values("ts").reset_index(drop=True)

mine = p.attempt(normalize_alpaca, alp_raw)
mine = p.check("normalize_alpaca", mine, p.normalize_alpaca(alp_raw))
mine.head()"""),
    ("md", "## 4. Two classic data bugs\n\n**Bug 1: treating naive exchange time as UTC.** The bars then sit 5 hours early (4 in summer). "
           "Worse than not joining at all, some of them *do* join, to the wrong bar."),
    ("code", """ib_ok, alp_ok = p.normalize_ib(ib_raw, "SPY"), p.normalize_alpaca(alp_raw)
ib_bad = ib_ok.assign(ts=ib_raw["date"].dt.tz_localize("UTC"))            # the bug
for name, left in [("naive-as-UTC", ib_bad), ("localized", ib_ok)]:
    joined = left.merge(alp_ok, on="ts", suffixes=("_ib", "_alp"))
    wrong = (joined["close_ib"] != joined["close_alp"]).sum()
    print(f"{name:13s}: {len(joined):3d} of {len(alp_ok)} bars join, {wrong} of them to a different bar")"""),
    ("md", "**Bug 2: mixing feeds.** Alpaca's free IEX feed sees only the trades on IEX, a few percent of consolidated volume. "
           "Prices agree, volumes don't. Keep `source` on every bar and never mix feeds inside one strategy."),
    ("code", """share = alp_ok["volume"] / ib_ok["volume"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ib_ok["ts"], ib_ok["close"], label="IB")
axes[0].plot(alp_ok["ts"], alp_ok["close"], "--", label="Alpaca IEX")
axes[0].set(title="Close: the same", ylabel="price"); axes[0].legend(); axes[0].tick_params(axis="x", rotation=30)
axes[1].plot(ib_ok["ts"], share * 100, color=p.PALETTE[1])
axes[1].set(title="Volume: IEX as % of consolidated", ylabel="%"); axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
print(f"IEX sees on average {share.mean():.1%} of the volume")"""),
    ("md", "## Wrap-up\n\n"
           "* Long histories are many requests walking back in time; pacing makes them slow, so download once and store.\n"
           "* One schema, UTC everywhere, `source` on every bar.\n"
           "* Graded version: `labs/part04/week14_data` (`ib_chunks`, `PacingGuard` with all three rules, `normalize_ib`, `normalize_alpaca`)."),
]

# ---------------------------------------------------------------------------------------------- 04
NB["04_live_bars_and_cache"] = [
    header("04", "Live streams, bar building and the cache", "S7 (Live streaming data) · S8 (The unified `DataHandler`)",
           "1. Build time bars from a stream of ticks.\n"
           "2. See why a bar builder needs a timer, not just ticks.\n"
           "3. Work out which date ranges a read-through cache still has to download.\n"
           "4. Spot a stale feed before a strategy trades on it."),
    ("code", SETUP),
    ("md", "## 1. A tick stream with quiet spells\n\n"
           "Real streams are bursty: busy minutes with a tick every few hundred milliseconds, then quiet spells with nothing for tens of seconds."),
    ("code", """ticks = p.tick_stream()
t, px = np.array([x[0] for x in ticks]), np.array([x[1] for x in ticks])
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(t, px, ".", ms=3)
ax.set(xlabel="seconds since the open", ylabel="price", title=f"{len(ticks)} ticks in 5 minutes")
plt.show()
gaps = np.diff(t)
print(f"median gap {np.median(gaps):.2f} s, longest gap {gaps.max():.1f} s")"""),
    ("md", "## 2. Ticks → 5-second bars\n\n"
           "A bar covers `[start, start + seconds)`, where `start` is `ts` rounded **down** to a multiple of `seconds`. "
           "`on_tick` closes the current bar when a tick arrives past its end, then starts or updates a bar. "
           "`on_timer(now)` (already written) closes the current bar once `now` has passed its end, even if no tick arrives."),
    ("md", YOUR_TURN),
    ("ex", """class MyBarBuilder(p.BarBuilder):
    def on_tick(self, ts: float, price: float, size: int) -> list[dict]:
        out = []
        if self.cur is not None and ts >= self.cur["start"] + self.seconds:
            out.append(self.cur)
            self.cur = None
        if self.cur is None:
            start = ...                           # ✍️ ts rounded down to a multiple of self.seconds
            self.cur = {"start": start, "open": price, "high": price, "low": price, "close": price, "volume": size}
        else:
            ...                                   # ✍️ update high, low and close, and add size to volume
        return out

mine = p.attempt(lambda: p.run_builder(MyBarBuilder(5), ticks, timer_every=1.0)[0])
ref, _ = p.run_builder(p.BarBuilder(5), ticks, timer_every=1.0)
mine = p.check("bar builder", mine, ref)
pd.DataFrame(mine).head()""",
     """class MyBarBuilder(p.BarBuilder):
    def on_tick(self, ts: float, price: float, size: int) -> list[dict]:
        out = []
        if self.cur is not None and ts >= self.cur["start"] + self.seconds:
            out.append(self.cur)
            self.cur = None
        if self.cur is None:
            start = ts // self.seconds * self.seconds
            self.cur = {"start": start, "open": price, "high": price, "low": price, "close": price, "volume": size}
        else:
            self.cur["high"] = max(self.cur["high"], price)
            self.cur["low"] = min(self.cur["low"], price)
            self.cur["close"] = price
            self.cur["volume"] += size
        return out

mine = p.attempt(lambda: p.run_builder(MyBarBuilder(5), ticks, timer_every=1.0)[0])
ref, _ = p.run_builder(p.BarBuilder(5), ticks, timer_every=1.0)
mine = p.check("bar builder", mine, ref)
pd.DataFrame(mine).head()"""),
    ("md", "## 3. Why the timer matters\n\n"
           "Without a timer, a bar is only published when the **next** tick arrives. In a quiet spell your strategy sees the bar late, "
           "and the last bar before a long silence may never arrive at all."),
    ("code", """with_timer, d1 = p.run_builder(p.BarBuilder(5), ticks, timer_every=1.0)
ticks_only, d2 = p.run_builder(p.BarBuilder(5), ticks, timer_every=None)
fig, ax = plt.subplots()
ax.hist([d1, d2], bins=np.arange(0, 32, 1), label=["with a 1 s timer", "ticks only"])
ax.set(xlabel="seconds between the bar's end and its publication", ylabel="bars", title="Bar publication delay")
ax.legend(); plt.show()
print(f"with timer: {len(with_timer)} bars, max delay {max(d1):.1f} s")
print(f"ticks only: {len(ticks_only)} bars, max delay {max(d2):.1f} s, and the last bar was never published")"""),
    ("md", "## 4. A read-through cache\n\n"
           "`DataHandler.get_bars(symbol, start, end)` should read from the local store and download **only the missing ranges**. "
           "`have` is a list of `(start, end)` ranges already stored, possibly overlapping and unsorted. Return the sub-ranges of `[start, end)` "
           "that no stored range covers, in order."),
    ("md", YOUR_TURN),
    ("ex", """def missing_ranges(have, start, end):
    out, cur = [], start                          # cur: everything before cur is covered
    for s, e in sorted(have):
        if e <= cur:
            continue                              # entirely behind us
        if s > cur:
            ...                                   # ✍️ a gap from cur to s (but not past end)
        cur = ...                                 # ✍️ now covered up to here
        if cur >= end:
            break
    if cur < end:
        out.append((cur, end))
    return out

cases = [([(5, 10), (8, 15), (20, 25)], 0, 30), ([], 0, 10), ([(0, 100)], 10, 20), ([(12, 18), (0, 4)], 2, 15)]
mine = [p.attempt(missing_ranges, *c) for c in cases]
mine = p.check("missing_ranges", mine, [p.missing_ranges(*c) for c in cases])
mine""",
     """def missing_ranges(have, start, end):
    out, cur = [], start                          # cur: everything before cur is covered
    for s, e in sorted(have):
        if e <= cur:
            continue                              # entirely behind us
        if s > cur:
            out.append((cur, min(s, end)))
        cur = max(cur, e)
        if cur >= end:
            break
    if cur < end:
        out.append((cur, end))
    return out

cases = [([(5, 10), (8, 15), (20, 25)], 0, 30), ([], 0, 10), ([(0, 100)], 10, 20), ([(12, 18), (0, 4)], 2, 15)]
mine = [p.attempt(missing_ranges, *c) for c in cases]
mine = p.check("missing_ranges", mine, [p.missing_ranges(*c) for c in cases])
mine"""),
    ("md", "Now a research day: 200 requests for random windows of days in a 2-year range. How many downloads does the cache make?"),
    ("code", """downloads = []
cache = p.BarCache(lambda a, b: downloads.append((a, b)))
rng = np.random.default_rng(1)
for _ in range(200):
    a = int(rng.integers(0, 700)); cache.get(a, a + int(rng.integers(5, 60)))
print(f"200 requests → {cache.downloads} downloads, {sum(b - a for a, b in downloads)} days fetched in total")
print("without the cache: 200 downloads, and every one of them paced by the broker")"""),
    ("md", "## 5. A stale feed\n\n"
           "A connected socket is not a live feed. Track the age of the last tick; if it exceeds a threshold during market hours, "
           "stop trading on that symbol and alert. Here, a 10-second threshold during our quiet spells:"),
    ("code", """now = np.arange(0, 300)
last_tick = np.array([t[t <= s].max() if (t <= s).any() else np.nan for s in now])
age = now - last_tick
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(now, age); ax.axhline(10, color=p.PALETTE[7], ls="--", label="stale threshold")
ax.set(xlabel="seconds since the open", ylabel="age of last tick (s)", title="Feed staleness"); ax.legend(); plt.show()
print(f"stale for {(age > 10).sum()} of 300 seconds; the quiet spells are exactly where a stale check matters")"""),
    ("md", "## Wrap-up\n\n"
           "* Bars close on time only if something besides ticks (a timer) closes them.\n"
           "* A read-through cache downloads only what is missing: fast research and fewer pacing waits.\n"
           "* Watch tick age, not just the connection state.\n"
           "* Graded version: `labs/part04/week14_data` (`BarBuilder` with timer flush, `missing_ranges`, `BarCache`)."),
]

# ---------------------------------------------------------------------------------------------- 05
NB["05_connections"] = [
    header("05", "Connections that survive the real world", "S9 (Connection management)",
           "1. Compute reconnect delays with exponential back-off and full jitter.\n"
           "2. Watch 500 clients hammer a restarted gateway, with and without jitter.\n"
           "3. Write the circuit breaker that stops you retrying a broker that is down.\n"
           "4. Sort IB's message codes into what to ignore, what to wait out and what to stop for."),
    ("code", SETUP + "\nimport random"),
    ("md", "## 1. Back-off with full jitter\n\n"
           "After a disconnect, retry after a delay that doubles each attempt, capped: `min(cap, base · 2**i)` for attempt `i = 0, 1, 2, …`. "
           "**Full jitter** waits a random time between 0 and that ceiling: `rng.uniform(0, min(cap, base * 2**i))`. "
           "Call `rng.uniform` once per attempt, in order, so the result is reproducible for a given seed."),
    ("md", YOUR_TURN),
    ("ex", """def backoff_delays(attempts: int, base: float, cap: float, rng: random.Random) -> list[float]:
    return [... for i in range(attempts)]        # ✍️ full jitter for attempt i

mine = backoff_delays(8, 1.0, 30.0, random.Random(42))
mine = p.check("backoff_delays", mine, p.backoff_delays(8, 1.0, 30.0, random.Random(42)))
np.round(mine, 2)""",
     """def backoff_delays(attempts: int, base: float, cap: float, rng: random.Random) -> list[float]:
    return [rng.uniform(0, min(cap, base * 2 ** i)) for i in range(attempts)]

mine = backoff_delays(8, 1.0, 30.0, random.Random(42))
mine = p.check("backoff_delays", mine, p.backoff_delays(8, 1.0, 30.0, random.Random(42)))
np.round(mine, 2)"""),
    ("md", "## 2. Why jitter: the thundering herd\n\n"
           "The IB Gateway restarts every night. Suppose 500 processes (yours, or a platform's) lose the connection at once and the gateway "
           "accepts at most 50 new connections per second. Without jitter, the failures stay **in lockstep**: every retry wave hits at the same instant."),
    ("code", """plain, t_plain = p.herd(jitter=False)
jit, t_jit = p.herd(jitter=True)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(np.arange(60) - 0.2, plain[:60], width=0.4, label=f"no jitter (all in after {t_plain} s)")
ax.bar(np.arange(60) + 0.2, jit[:60], width=0.4, label=f"full jitter (all in after {t_jit} s)")
ax.axhline(50, color=p.PALETTE[7], ls="--", lw=1, label="gateway capacity")
ax.set(xlabel="seconds after the restart", ylabel="connection attempts", title="500 clients reconnecting")
ax.legend(); plt.show()
print(f"attempts in total: no jitter {plain.sum():.0f}, full jitter {jit.sum():.0f}")"""),
    ("md", "## 3. The circuit breaker\n\n"
           "Back-off handles a blip. When a broker is really down, stop calling it for a while:\n\n"
           "* **closed**: calls go through; `threshold` consecutive failures open the breaker;\n"
           "* **open**: calls are blocked until `cooldown` seconds have passed since it opened;\n"
           "* **half_open**: one trial call is allowed; success closes the breaker, failure opens it again.\n\n"
           "`record(ok, now)` is written for you. Write `allow(now)`: when open, go to half_open and allow the call once the cooldown has passed, "
           "else block it."),
    ("md", YOUR_TURN),
    ("ex", """class MyBreaker(p.CircuitBreaker):
    def allow(self, now: float) -> bool:
        if self.state == "open":
            ...                                   # ✍️ cooldown passed → state "half_open", return True; else return False
        return True

outage = [(0, True), (5, False), (6, False), (7, False), (8, True), (20, True), (37, False), (40, True), (70, True), (71, True)]
mine = p.breaker_trace(MyBreaker(threshold=3, cooldown=30), outage)
mine = p.check("circuit breaker", mine, p.breaker_trace(p.CircuitBreaker(threshold=3, cooldown=30), outage))
pd.DataFrame(mine, columns=["t", "call", "state after"])""",
     """class MyBreaker(p.CircuitBreaker):
    def allow(self, now: float) -> bool:
        if self.state == "open":
            if now - self.opened_at >= self.cooldown:
                self.state = "half_open"
                return True
            return False
        return True

outage = [(0, True), (5, False), (6, False), (7, False), (8, True), (20, True), (37, False), (40, True), (70, True), (71, True)]
mine = p.breaker_trace(MyBreaker(threshold=3, cooldown=30), outage)
mine = p.check("circuit breaker", mine, p.breaker_trace(p.CircuitBreaker(threshold=3, cooldown=30), outage))
pd.DataFrame(mine, columns=["t", "call", "state after"])"""),
    ("md", "The same breaker over a 10-minute outage, with a call attempted every 5 seconds. Count how many calls actually hit the dead broker."),
    ("code", """times = np.arange(0, 900, 5)
up = (times < 120) | (times >= 720)                     # down from 2:00 to 12:00
trace = p.breaker_trace(p.CircuitBreaker(threshold=3, cooldown=60), list(zip(times, up)))
level = {"closed": 0, "half_open": 1, "open": 2}
fig, ax = plt.subplots(figsize=(10, 3))
ax.step(times / 60, [level[s] for _, _, s in trace], where="post")
ax.axvspan(2, 12, color=p.PALETTE[7], alpha=0.1, label="broker down")
ax.set(yticks=[0, 1, 2], yticklabels=list(level), xlabel="minutes", title="Circuit breaker state"); ax.legend(); plt.show()
calls_while_down = sum(1 for (t, c, _), u in zip(trace, up) if c == "call" and not u)
print(f"calls to the dead broker: {calls_while_down} with the breaker, {(~up).sum()} without it")"""),
    ("md", "## 4. IB message codes\n\n"
           "IB sends informational messages, connectivity warnings and real errors through the **same** error callback. "
           "Treating them all as errors floods your alerts; ignoring them all misses a lost connection.\n\n"
           "| Category | Codes | Action |\n|---|---|---|\n"
           "| info | 2104, 2106, 2107, 2108, 2158 | log at debug (\"market data farm OK\") |\n"
           "| connectivity | 1100, 1101, 1102, 2110, 504 | pause trading, reconnect, resubscribe |\n"
           "| pacing | 162, 420 | slow down (notebook 03) |\n"
           "| client_id_in_use | 326 | fatal: exit and fix the config (notebook 01) |\n"
           "| order_reject | 103, 110, 201 | release the order, alert |\n"
           "| unknown | anything else | log at warning and review |"),
    ("md", YOUR_TURN),
    ("ex", """CATEGORIES = {
    "info": {2104, 2106, 2107, 2108, 2158},
    "connectivity": {1100, 1101, 1102, 2110, 504},
    "pacing": ...,                                # ✍️ the pacing codes
    "client_id_in_use": ...,                      # ✍️ the duplicate client id code
    "order_reject": ...,                          # ✍️ the three order reject codes
}

def classify(code: int) -> str:
    for category, codes in CATEGORIES.items():
        if code in codes:
            return category
    return "unknown"

log = p.ib_error_log()
mine = [p.attempt(classify, c) for c in log]
mine = p.check("classify IB codes", mine, [p.classify_ib_code(c) for c in log])
pd.Series(mine).value_counts()""",
     """CATEGORIES = {
    "info": {2104, 2106, 2107, 2108, 2158},
    "connectivity": {1100, 1101, 1102, 2110, 504},
    "pacing": {162, 420},
    "client_id_in_use": {326},
    "order_reject": {103, 110, 201},
}

def classify(code: int) -> str:
    for category, codes in CATEGORIES.items():
        if code in codes:
            return category
    return "unknown"

log = p.ib_error_log()
mine = [p.attempt(classify, c) for c in log]
mine = p.check("classify IB codes", mine, [p.classify_ib_code(c) for c in log])
pd.Series(mine).value_counts()"""),
    ("md", "Three quarters of what arrives through the error callback is harmless information. That's why a monitor that alerts on every message gets muted by week two."),
    ("md", "## Wrap-up\n\n"
           "* Exponential back-off with **full jitter** for reconnects; a circuit breaker for real outages.\n"
           "* Classify every broker message and give each category one action.\n"
           "* Graded version: `labs/part04/week15_orders` (`backoff_delays`, `CircuitBreaker` with a fake clock, `classify_ib_code`)."),
]

# ---------------------------------------------------------------------------------------------- 06
NB["06_order_mapping"] = [
    header("06", "Orders: one model, two brokers", "S10 (Order types & cross-broker mapping)",
           "1. Round a limit price onto the tick grid without giving money away.\n"
           "2. Map one canonical order to IB's and Alpaca's fields.\n"
           "3. Map both brokers' order statuses to one set of states.\n"
           "4. Use a property test to show the mapping never changes an order's meaning."),
    ("code", SETUP + "\nfrom decimal import ROUND_CEILING, ROUND_FLOOR"),
    ("md", "## 1. Rounding a limit price\n\n"
           "A price off the tick grid is rejected (IB error 110). But which way to round? Half-up rounding sometimes moves a **buy** limit *up*: "
           "you'd pay more than you decided to. Round in your favour: a BUY rounds **down**, a SELL rounds **up**."),
    ("md", YOUR_TURN),
    ("ex", """def round_limit(price: Decimal, tick: Decimal, side: str) -> Decimal:
    rounding = ...                                # ✍️ ROUND_FLOOR for a BUY, ROUND_CEILING for a SELL
    return (price / tick).quantize(Decimal("1"), rounding=rounding) * tick

cases = [(Decimal("101.237"), Decimal("0.05"), "BUY"), (Decimal("101.237"), Decimal("0.05"), "SELL"),
         (Decimal("4512.30"), Decimal("0.25"), "BUY"), (Decimal("4512.30"), Decimal("0.25"), "SELL"),
         (Decimal("1.08765"), Decimal("0.00005"), "BUY"), (Decimal("231.55"), Decimal("0.01"), "SELL")]
mine = [p.attempt(round_limit, *c) for c in cases]
mine = p.check("round_limit", mine, [p.round_limit(*c) for c in cases])
mine""",
     """def round_limit(price: Decimal, tick: Decimal, side: str) -> Decimal:
    rounding = ROUND_FLOOR if side == "BUY" else ROUND_CEILING
    return (price / tick).quantize(Decimal("1"), rounding=rounding) * tick

cases = [(Decimal("101.237"), Decimal("0.05"), "BUY"), (Decimal("101.237"), Decimal("0.05"), "SELL"),
         (Decimal("4512.30"), Decimal("0.25"), "BUY"), (Decimal("4512.30"), Decimal("0.25"), "SELL"),
         (Decimal("1.08765"), Decimal("0.00005"), "BUY"), (Decimal("231.55"), Decimal("0.01"), "SELL")]
mine = [p.attempt(round_limit, *c) for c in cases]
mine = p.check("round_limit", mine, [p.round_limit(*c) for c in cases])
mine"""),
    ("code", """rng = np.random.default_rng(0)
raw = [Decimal(str(round(x, 3))) for x in rng.uniform(4000, 5000, 10_000)]    # ES quotes, tick 0.25
tick = Decimal("0.25")
worse = sum(p.half_up(x, tick) > x for x in raw)
print(f"half-up rounding raises a BUY limit above the intended price in {worse:,} of 10,000 cases")
print(f"side-aware rounding: {sum(p.round_limit(x, tick, 'BUY') > x for x in raw)} cases")"""),
    ("md", "## 2. One order, two brokers\n\n"
           "Strategies create one canonical `p.Order`; the adapters translate it. The field names and spellings differ:\n\n"
           "| Canonical | IB (`ib_async.Order`) | Alpaca (`alpaca-py` request) |\n|---|---|---|\n"
           "| side `BUY` | `action=\"BUY\"` | `side=\"buy\"` |\n"
           "| qty | `totalQuantity` | `qty` |\n"
           "| type `STP_LMT` | `orderType=\"STP LMT\"` | `type=\"stop_limit\"` |\n"
           "| limit | `lmtPrice` | `limit_price` |\n"
           "| stop | `auxPrice` | `stop_price` |\n"
           "| trail % | `trailingPercent` | `trail_percent` |\n"
           "| outside RTH | `outsideRth` | `extended_hours` (DAY limit orders only) |\n"
           "| client id | `orderRef` | `client_order_id` |"),
    ("code", """pd.DataFrame([vars(o) for o in p.SAMPLE_ORDERS]).fillna("")"""),
    ("md", YOUR_TURN + "\n\nFields that are `None` are left out of the result (the dict comprehension at the end does that)."),
    ("ex", """IB_TYPES = {"MKT": "MKT", "LMT": "LMT", "STP": "STP", "STP_LMT": ..., "TRAIL": "TRAIL"}   # ✍️ IB's stop-limit spelling

def to_ib_fields(o: p.Order) -> dict:
    d = {"action": o.side, "totalQuantity": o.qty, "orderType": IB_TYPES[o.type], "lmtPrice": o.limit,
         "auxPrice": ..., "trailingPercent": o.trail_pct, "tif": o.tif,   # ✍️ auxPrice carries the stop price
         "outsideRth": o.outside_rth, "orderRef": o.client_id}
    return {k: v for k, v in d.items() if v is not None}

mine = [to_ib_fields(o) for o in p.SAMPLE_ORDERS]
mine = p.check("to_ib_fields", mine, [p.to_ib_fields(o) for o in p.SAMPLE_ORDERS])
pd.DataFrame(mine).fillna("")""",
     """IB_TYPES = {"MKT": "MKT", "LMT": "LMT", "STP": "STP", "STP_LMT": "STP LMT", "TRAIL": "TRAIL"}

def to_ib_fields(o: p.Order) -> dict:
    d = {"action": o.side, "totalQuantity": o.qty, "orderType": IB_TYPES[o.type], "lmtPrice": o.limit,
         "auxPrice": o.stop, "trailingPercent": o.trail_pct, "tif": o.tif,
         "outsideRth": o.outside_rth, "orderRef": o.client_id}
    return {k: v for k, v in d.items() if v is not None}

mine = [to_ib_fields(o) for o in p.SAMPLE_ORDERS]
mine = p.check("to_ib_fields", mine, [p.to_ib_fields(o) for o in p.SAMPLE_ORDERS])
pd.DataFrame(mine).fillna("")"""),
    ("md", "The Alpaca side is written for you (`p.to_alpaca_fields`). It also enforces a broker rule: extended hours only for DAY limit orders."),
    ("code", """display(pd.DataFrame([p.to_alpaca_fields(o) for o in p.SAMPLE_ORDERS]).fillna(""))
gtc_night = p.Order("NVDA", "BUY", Decimal("5"), "LMT", limit=Decimal("118.40"), tif="GTC", outside_rth=True)
try:
    p.to_alpaca_fields(gtc_night)
except ValueError as e:
    print("🛑 refused before it reached the broker:", e)"""),
    ("md", "## 3. Statuses into one state machine\n\n"
           "Both brokers report order status, with different words. We map them to one set of states (`PENDING_NEW`, `ACCEPTED`, "
           "`PARTIALLY_FILLED`, `FILLED`, `PENDING_CANCEL`, `CANCELLED`, `REJECTED`, `EXPIRED`).\n\n"
           "IB has one quirk: there is **no partial-fill status**. A `Submitted` order with `filled > 0` is partially filled. "
           "`p.IB_STATUS` maps the words; you add the quirk."),
    ("code", """pd.DataFrame({"IB status": list(p.IB_STATUS), "canonical": list(p.IB_STATUS.values())})"""),
    ("md", YOUR_TURN),
    ("ex", """def map_ib_status(status: str, filled: float) -> str:
    s = p.IB_STATUS[status]
    return ...                                    # ✍️ an ACCEPTED order with filled > 0 is PARTIALLY_FILLED

cases = [("PendingSubmit", 0), ("PreSubmitted", 0), ("Submitted", 0), ("Submitted", 40), ("Filled", 100),
         ("PendingCancel", 40), ("Cancelled", 40), ("Inactive", 0)]
mine = [map_ib_status(s, f) for s, f in cases]
mine = p.check("map_ib_status", mine, [p.map_ib_status(s, f) for s, f in cases])
list(zip(cases, mine))""",
     """def map_ib_status(status: str, filled: float) -> str:
    s = p.IB_STATUS[status]
    return "PARTIALLY_FILLED" if s == "ACCEPTED" and filled > 0 else s

cases = [("PendingSubmit", 0), ("PreSubmitted", 0), ("Submitted", 0), ("Submitted", 40), ("Filled", 100),
         ("PendingCancel", 40), ("Cancelled", 40), ("Inactive", 0)]
mine = [map_ib_status(s, f) for s, f in cases]
mine = p.check("map_ib_status", mine, [p.map_ib_status(s, f) for s, f in cases])
list(zip(cases, mine))"""),
    ("md", "## 4. A property test: the mapping never changes meaning\n\n"
           "Example tests check the cases you thought of. A **property test** (Hypothesis) generates hundreds of orders and checks a rule that "
           "must always hold: both brokers get the same side, quantity and prices, and a rounded limit is never worse than the one asked for."),
    ("code", """from hypothesis import given, settings, strategies as st

prices = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("5000"), places=4)
orders = st.builds(p.Order, symbol=st.sampled_from(["SPY", "AAPL"]), side=st.sampled_from(["BUY", "SELL"]),
                   qty=st.integers(1, 10_000).map(Decimal), type=st.just("LMT"), limit=prices)

@settings(max_examples=300, deadline=None)
@given(orders, st.sampled_from([Decimal("0.01"), Decimal("0.05"), Decimal("0.25")]))
def test_mapping_keeps_meaning(o, tick):
    ib, alp = p.to_ib_fields(o), p.to_alpaca_fields(o)
    assert ib["action"].lower() == alp["side"] and ib["totalQuantity"] == alp["qty"] and ib["lmtPrice"] == alp["limit_price"]
    r = p.round_limit(o.limit, tick, o.side)
    assert (r <= o.limit) if o.side == "BUY" else (r >= o.limit)      # never worse than asked
    assert r % tick == 0                                              # on the grid

test_mapping_keeps_meaning()
print("✔ 300 random orders: mapping and rounding keep their meaning")"""),
    ("md", "## Wrap-up\n\n"
           "* Round prices in the trader's favour, in `Decimal`, before the order leaves.\n"
           "* One canonical order and state set; the adapters translate both ways.\n"
           "* Property tests find the mapping bug you didn't think to write an example for.\n"
           "* Graded version: `labs/part04/week15_orders` (with real `ib_async` / `alpaca-py` objects and a Hypothesis test)."),
]

# ---------------------------------------------------------------------------------------------- 07
NB["07_order_lifecycle"] = [
    header("07", "Order lifecycle, positions and reconciliation", "S11 (Order state machine & update streams) · S12 (Bracket, OCO, reconciliation)",
           "1. Apply a messy stream of order updates (duplicates, out of order, races) to a state machine.\n"
           "2. Build positions from executions, not from order statuses.\n"
           "3. Reconcile your open orders against the broker's after a crash.\n"
           "4. Watch a bracket order's OCO children cancel each other."),
    ("code", SETUP),
    ("md", "## 1. Update streams are messy\n\n"
           "Order updates arrive over a network, from different broker subsystems. In one morning you can see: the same fill twice, a fill "
           "**before** the order's `ACCEPTED`, an `ACCEPTED` that arrives after `FILLED`, and a cancel that loses the race to a fill. "
           "Here are the events in **arrival order** (already mapped to canonical states)."),
    ("code", """events = p.order_events()
pd.DataFrame(events).fillna("")"""),
    ("md", "The naive way: the last status wins and every fill message adds up."),
    ("code", """naive = {}
for ev in events:
    o = naive.setdefault(ev["order_id"], {"state": "PENDING_NEW", "filled": Decimal(0)})
    if ev["type"] == "fill":
        o["filled"] += ev["qty"]
    else:
        o["state"] = ev["status"]
pd.DataFrame(naive).T"""),
    ("md", "Every order is wrong: qf-101 shows 140 shares filled of 100, qf-102 is `ACCEPTED` although it is filled, qf-103 looks `CANCELLED` "
           "with 10 shares filled, and the rejected qf-104 looks live.\n\n"
           "The fix: every order starts at `PENDING_NEW`; apply a status **only if `p.TRANSITIONS` allows it** from the current state "
           "(terminal states allow nothing); count each fill **once per `exec_id`**."),
    ("code", """pd.Series({k: ", ".join(sorted(v)) or "— terminal —" for k, v in p.TRANSITIONS.items()}, name="may move to").to_frame()"""),
    ("md", YOUR_TURN),
    ("ex", """def apply_events(events: list[dict]) -> dict[str, dict]:
    book, seen = {}, set()
    for ev in events:
        o = book.setdefault(ev["order_id"], {"state": "PENDING_NEW", "filled": Decimal(0), "notional": Decimal(0)})
        if ev["type"] == "fill":
            # ✍️ skip an exec_id already seen; otherwise remember it, add qty to filled and qty*price to notional
            ...
        elif ...:                                 # ✍️ only if p.TRANSITIONS allows o["state"] → ev["status"]
            o["state"] = ev["status"]
    return {k: {"state": o["state"], "filled": o["filled"],
                "avg_price": (o["notional"] / o["filled"]).quantize(Decimal("0.0001")) if o["filled"] else None}
            for k, o in book.items()}

mine = apply_events(events)
mine = p.check("apply_events", mine, p.apply_events(events))
pd.DataFrame(mine).T""",
     """def apply_events(events: list[dict]) -> dict[str, dict]:
    book, seen = {}, set()
    for ev in events:
        o = book.setdefault(ev["order_id"], {"state": "PENDING_NEW", "filled": Decimal(0), "notional": Decimal(0)})
        if ev["type"] == "fill":
            if ev["exec_id"] in seen:
                continue
            seen.add(ev["exec_id"])
            o["filled"] += ev["qty"]
            o["notional"] += ev["qty"] * ev["price"]
        elif ev["status"] in p.TRANSITIONS[o["state"]]:
            o["state"] = ev["status"]
    return {k: {"state": o["state"], "filled": o["filled"],
                "avg_price": (o["notional"] / o["filled"]).quantize(Decimal("0.0001")) if o["filled"] else None}
            for k, o in book.items()}

mine = apply_events(events)
mine = p.check("apply_events", mine, p.apply_events(events))
pd.DataFrame(mine).T"""),
    ("md", "Look at qf-103: we asked to cancel, but the fill won the race. `PENDING_CANCEL → FILLED` is a legal transition and the late "
           "`CANCELLED` is ignored. A strategy that assumed \"I cancelled, so I'm flat\" would now hold 10 shares it doesn't know about."),
    ("md", "## 2. Positions come from executions\n\n"
           "Order status says what happened to an *order*. Your *position* is the sum of executions: BUY adds, SELL subtracts. "
           "Leave out symbols that net to zero."),
    ("code", """fills = p.day_fills()
pd.DataFrame(fills)"""),
    ("md", YOUR_TURN),
    ("ex", """def positions_from_fills(fills: list[dict]) -> dict[str, Decimal]:
    pos = {}
    for f in fills:
        pos[f["symbol"]] = ...                    # ✍️ previous position (0 if none) + qty for a BUY, − qty for a SELL
    return {s: q for s, q in pos.items() if q != 0}

mine = p.attempt(positions_from_fills, fills)
mine = p.check("positions_from_fills", mine, p.positions_from_fills(fills))
mine""",
     """def positions_from_fills(fills: list[dict]) -> dict[str, Decimal]:
    pos = {}
    for f in fills:
        pos[f["symbol"]] = pos.get(f["symbol"], Decimal(0)) + (f["qty"] if f["side"] == "BUY" else -f["qty"])
    return {s: q for s, q in pos.items() if q != 0}

mine = p.attempt(positions_from_fills, fills)
mine = p.check("positions_from_fills", mine, p.positions_from_fills(fills))
mine"""),
    ("md", "## 3. Reconciliation after a crash\n\n"
           "Your process crashed at 10:14 and restarted. Compare **your** open orders (client order id → remaining qty) with the **broker's**:\n\n"
           "* `missing_at_broker`: in your book, not at the broker (filled or cancelled while you were down: check executions);\n"
           "* `orphans`: at the broker with **your prefix** `qf-` but not in your book (sent just before the crash): cancel or adopt;\n"
           "* `qty_mismatch`: on both sides with a different remaining qty (a fill you missed);\n"
           "* `foreign`: at the broker without your prefix (placed by hand or by another system): **never touch**.\n\n"
           "Every list sorted."),
    ("code", """D = Decimal
ours = {"qf-201": D("100"), "qf-202": D("50"), "qf-203": D("20"), "qf-204": D("10")}
broker = {"qf-201": D("100"), "qf-202": D("30"), "qf-205": D("15"), "qf-206": D("5"), "manual-77": D("200"), "TWS-3": D("1")}"""),
    ("md", YOUR_TURN),
    ("ex", """def reconcile(ours: dict, broker: dict, prefix: str = "qf-") -> dict[str, list[str]]:
    return {"missing_at_broker": sorted(set(ours) - set(broker)),
            "orphans": ...,                       # ✍️ broker ids with our prefix that we don't know
            "qty_mismatch": ...,                  # ✍️ ids on both sides with different qty
            "foreign": sorted(k for k in broker if not k.startswith(prefix))}

mine = reconcile(ours, broker)
mine = p.check("reconcile", mine, p.reconcile(ours, broker))
mine""",
     """def reconcile(ours: dict, broker: dict, prefix: str = "qf-") -> dict[str, list[str]]:
    return {"missing_at_broker": sorted(set(ours) - set(broker)),
            "orphans": sorted(k for k in broker if k.startswith(prefix) and k not in ours),
            "qty_mismatch": sorted(k for k in set(ours) & set(broker) if ours[k] != broker[k]),
            "foreign": sorted(k for k in broker if not k.startswith(prefix))}

mine = reconcile(ours, broker)
mine = p.check("reconcile", mine, p.reconcile(ours, broker))
mine"""),
    ("md", "## 4. A bracket order and its OCO children\n\n"
           "A bracket is an entry with two exit children: a take-profit limit and a stop-loss. The children are **one-cancels-other**: "
           "when one fills, the broker cancels the other. Both brokers support this natively (IB: `ib.bracketOrder`; Alpaca: `order_class=\"bracket\"`). "
           "Doing it client-side means a crash between the fill and your cancel leaves you with an unintended position."),
    ("code", """rng = np.random.default_rng(5)
path = 100 + np.cumsum(rng.normal(0, 0.12, 400))
entry, tp, sl = 100.0, 101.5, 99.0
state, log = "entry working", []
for i, px in enumerate(path):
    if state == "entry working" and px <= entry:
        state = "in position"; log.append((i, "entry filled at 100.00, TP and SL now working"))
    elif state == "in position" and (px >= tp or px <= sl):
        hit, other = ("take-profit", "stop-loss") if px >= tp else ("stop-loss", "take-profit")
        state = "flat"; log.append((i, f"{hit} filled at {px:.2f}; broker cancels the {other} (OCO)")); break
fig, ax = plt.subplots()
ax.plot(path[: log[-1][0] + 20])
for y, c, name in [(tp, p.PALETTE[2], "take-profit"), (entry, p.PALETTE[0], "entry"), (sl, p.PALETTE[7], "stop-loss")]:
    ax.axhline(y, color=c, ls="--", lw=1, label=name)
for i, _ in log:
    ax.plot(i, path[i], "o", color="black")
ax.set(xlabel="tick", ylabel="price", title="Bracket order"); ax.legend(); plt.show()
for i, msg in log:
    print(f"tick {i:3d}: {msg}")"""),
    ("md", "## Wrap-up\n\n"
           "* A state machine with legal transitions, and fills counted once per execution id, turn a messy stream into the truth.\n"
           "* Positions come from executions; reconcile against the broker at startup and on a timer.\n"
           "* Use broker-native brackets and OCO so exits survive your crash.\n"
           "* Graded version: `labs/part04/week15_orders` (`OrderTracker`, `reconcile`) and the Clinic W3 event replay."),
]

# ---------------------------------------------------------------------------------------------- 08
NB["08_safety_multiasset"] = [
    header("08", "Kill switch, pre-trade checks and multi-asset", "S13 (Kill switch) · S14 (Futures, FX & crypto) · S15 (Pre-trade checks) · S16 (Contract tests)",
           "1. Compute futures P&L and FX pip values exactly.\n"
           "2. Find a futures expiry and its roll date.\n"
           "3. Write the pre-trade check chain every order passes.\n"
           "4. Plan a kill-switch flatten, make it sticky, and time it.\n"
           "5. Run the same contract tests against two broker adapters."),
    ("code", SETUP + "\nfrom datetime import date, timedelta\nimport asyncio, tempfile, time"),
    ("md", "## 1. Futures: the multiplier is the risk\n\n"
           "A futures P&L is `(exit − entry) × qty × multiplier`, with `qty` signed (+ long, − short). `p.FUTURES` holds `(tick, multiplier)`."),
    ("code", """pd.DataFrame({k: {"tick": t, "multiplier": m, "tick value $": t * m} for k, (t, m) in p.FUTURES.items()}).T"""),
    ("md", YOUR_TURN),
    ("ex", """def futures_pnl(entry: Decimal, exit: Decimal, qty: Decimal, multiplier: Decimal) -> Decimal:
    return ...                                    # ✍️

D = Decimal
trades = [("ES", D("5012.25"), D("5020.00"), D("2")), ("NQ", D("18010.50"), D("17950.25"), D("1")),
          ("CL", D("71.42"), D("70.95"), D("-3")), ("MES", D("5012.25"), D("5020.00"), D("2")),
          ("GC", D("2350.1"), D("2361.4"), D("-1"))]
mine = [futures_pnl(e, x, q, p.FUTURES[s][1]) for s, e, x, q in trades]
mine = p.check("futures P&L", mine, [p.futures_pnl(e, x, q, p.FUTURES[s][1]) for s, e, x, q in trades])
dict(zip([t[0] for t in trades], mine))""",
     """def futures_pnl(entry: Decimal, exit: Decimal, qty: Decimal, multiplier: Decimal) -> Decimal:
    return (exit - entry) * qty * multiplier

D = Decimal
trades = [("ES", D("5012.25"), D("5020.00"), D("2")), ("NQ", D("18010.50"), D("17950.25"), D("1")),
          ("CL", D("71.42"), D("70.95"), D("-3")), ("MES", D("5012.25"), D("5020.00"), D("2")),
          ("GC", D("2350.1"), D("2361.4"), D("-1"))]
mine = [futures_pnl(e, x, q, p.FUTURES[s][1]) for s, e, x, q in trades]
mine = p.check("futures P&L", mine, [p.futures_pnl(e, x, q, p.FUTURES[s][1]) for s, e, x, q in trades])
dict(zip([t[0] for t in trades], mine))"""),
    ("md", "The same 7.75-point move is $775 on 2 ES and $77.50 on 2 MES. Micro contracts exist so a small account can size correctly. "
           "FX works the same way through the pip value: one pip (0.0001, or 0.01 for JPY pairs) × units × the USD value of the quote currency."),
    ("code", """fx = [("EURUSD", D("100000"), D("1")), ("USDJPY", D("100000"), D("1") / D("151.20")), ("GBPUSD", D("10000"), D("1"))]
{pair: f"${p.pip_value_usd(pair, units, usd).quantize(D('0.01'))} per pip" for pair, units, usd in fx}"""),
    ("md", "## 2. Expiry and roll\n\n"
           "Equity index futures expire on the **third Friday** of the contract month. Volume moves to the next contract about a week "
           "earlier, so we roll a fixed number of business days before expiry (8 here, holidays ignored).\n\n"
           "`date.weekday()` is 0 for Monday … 4 for Friday."),
    ("md", YOUR_TURN),
    ("ex", """def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    days_to_friday = ...                          # ✍️ days from `first` to the month's first Friday (0 if it is one)
    return first + timedelta(days=days_to_friday + 14)

months = [(2026, 3), (2026, 6), (2026, 9), (2026, 12), (2027, 1), (2025, 8)]
mine = [p.attempt(third_friday, y, m) for y, m in months]
mine = p.check("third_friday", mine, [p.third_friday(y, m) for y, m in months])
mine""",
     """def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_friday + 14)

months = [(2026, 3), (2026, 6), (2026, 9), (2026, 12), (2027, 1), (2025, 8)]
mine = [p.attempt(third_friday, y, m) for y, m in months]
mine = p.check("third_friday", mine, [p.third_friday(y, m) for y, m in months])
mine"""),
    ("code", """pd.DataFrame([{"contract": f"ES{p.MONTH_CODES[m - 1]}{str(y)[-1]}", "roll on": p.roll_date(y, m), "expires": p.third_friday(y, m)}
              for y, m in [(2026, 3), (2026, 6), (2026, 9), (2026, 12)]])"""),
    ("md", "## 3. The pre-trade check chain\n\n"
           "Every order passes the same checks before it reaches an adapter. Return the **first** failure, or `None`:\n\n"
           "1. `'kill switch'` if `ctx.killed`;\n"
           "2. `'no market data'` if the symbol has no `ctx.last` price;\n"
           "3. `'price outside band'` if `|price / last − 1| > ctx.band` (a fat-finger guard);\n"
           "4. `'order too large'` if `qty × price > ctx.max_notional`;\n"
           "5. `'position limit'` if `|position + signed qty| > ctx.max_position` (signed: + for BUY, − for SELL);\n"
           "6. `'buying power'` if a BUY costs more than `ctx.buying_power`."),
    ("code", """ctx = p.Context()
print(ctx)"""),
    ("md", YOUR_TURN),
    ("ex", """def pretrade_check(order: dict, ctx: p.Context) -> str | None:
    if ctx.killed:
        return "kill switch"
    last = ctx.last.get(order["symbol"])
    if last is None:
        return "no market data"
    if ...:                                       # ✍️ price more than ctx.band away from last, relatively
        return "price outside band"
    notional = order["qty"] * order["price"]
    if notional > ctx.max_notional:
        return "order too large"
    signed = order["qty"] if order["side"] == "BUY" else -order["qty"]
    if ...:                                       # ✍️ |current position (0 if none) + signed| above ctx.max_position
        return "position limit"
    if order["side"] == "BUY" and notional > ctx.buying_power:
        return "buying power"
    return None

orders = p.random_orders()
mine = [pretrade_check(o, ctx) for o in orders]
mine = p.check("pretrade_check", mine, [p.pretrade_check(o, ctx) for o in orders])
counts = pd.Series(["passed" if r is None else r for r in mine]).value_counts()
counts.plot.barh(title="1,000 orders through the pre-trade checks"); plt.gca().invert_yaxis(); plt.show()""",
     """def pretrade_check(order: dict, ctx: p.Context) -> str | None:
    if ctx.killed:
        return "kill switch"
    last = ctx.last.get(order["symbol"])
    if last is None:
        return "no market data"
    if abs(order["price"] / last - 1) > ctx.band:
        return "price outside band"
    notional = order["qty"] * order["price"]
    if notional > ctx.max_notional:
        return "order too large"
    signed = order["qty"] if order["side"] == "BUY" else -order["qty"]
    if abs(ctx.positions.get(order["symbol"], Decimal(0)) + signed) > ctx.max_position:
        return "position limit"
    if order["side"] == "BUY" and notional > ctx.buying_power:
        return "buying power"
    return None

orders = p.random_orders()
mine = [pretrade_check(o, ctx) for o in orders]
mine = p.check("pretrade_check", mine, [p.pretrade_check(o, ctx) for o in orders])
counts = pd.Series(["passed" if r is None else r for r in mine]).value_counts()
counts.plot.barh(title="1,000 orders through the pre-trade checks"); plt.gca().invert_yaxis(); plt.show()"""),
    ("md", "## 4. The kill switch\n\n"
           "When it trips: **cancel every open order first** (so nothing new fills), **then** close every position with a market order. "
           "Return the plan: `('cancel', id)` for each open order (sorted), then `('market', symbol, side, qty)` for each non-zero position "
           "(sorted by symbol): SELL a long, BUY a short, qty always positive."),
    ("md", YOUR_TURN),
    ("ex", """def flatten_plan(open_orders: list[str], positions: dict[str, Decimal]) -> list[tuple]:
    plan = [("cancel", oid) for oid in sorted(open_orders)]
    plan += ...                                   # ✍️ ("market", symbol, side, abs(qty)) for each non-zero position
    return plan

open_orders = ["qf-310", "qf-302", "qf-305"]
positions = {"SPY": D("75"), "AAPL": D("-10"), "QQQ": D("0"), "NVDA": D("30")}
mine = p.attempt(flatten_plan, open_orders, positions)
mine = p.check("flatten_plan", mine, p.flatten_plan(open_orders, positions))
mine""",
     """def flatten_plan(open_orders: list[str], positions: dict[str, Decimal]) -> list[tuple]:
    plan = [("cancel", oid) for oid in sorted(open_orders)]
    plan += [("market", s, "SELL" if q > 0 else "BUY", abs(q)) for s, q in sorted(positions.items()) if q != 0]
    return plan

open_orders = ["qf-310", "qf-302", "qf-305"]
positions = {"SPY": D("75"), "AAPL": D("-10"), "QQQ": D("0"), "NVDA": D("30")}
mine = p.attempt(flatten_plan, open_orders, positions)
mine = p.check("flatten_plan", mine, p.flatten_plan(open_orders, positions))
mine"""),
    ("md", "**Sticky.** A kill switch that forgets it was tripped when the process restarts is not a kill switch. "
           "`p.KillSwitch` writes its state to a file; a new process reads it and stays stopped until a human resets it with the confirmation phrase."),
    ("code", """state_file = Path(tempfile.mkdtemp()) / "killswitch.json"
ks = p.KillSwitch(state_file)
ks.trip("daily loss limit hit: -2.1%")
print("tripped:", ks.tripped)
restarted = p.KillSwitch(state_file)                    # a brand-new process after a crash
print("after restart, still tripped:", restarted.tripped)
print("reset with the wrong phrase:", restarted.reset("ok"))
print("reset with the right phrase:", restarted.reset("I have flattened and reviewed"), "→ tripped:", restarted.tripped)
print("audit:", ks.audit + restarted.audit)"""),
    ("md", "**Fast.** The lesson plan's target is flat on both brokers in under 5 seconds. With 20 open orders and 8 positions per broker "
           "and 100 ms per request, doing it one request at a time is too slow; `asyncio.gather` sends them concurrently. "
           "(Cancels still finish before the closes start.)"),
    ("code", """def brokers():
    pos = {f"S{i}": D("10") for i in range(8)}
    return [p.AsyncSimBroker("ib", 20, pos, latency=0.1), p.AsyncSimBroker("alpaca", 20, pos, latency=0.1)]

async def flatten_sequential(bs):
    for b in bs:
        for oid in list(b.open):
            await b.cancel(oid)
        for s in list(b.positions):
            await b.close(s)

async def flatten_concurrent(bs):
    await asyncio.gather(*(b.cancel(oid) for b in bs for oid in list(b.open)))
    await asyncio.gather(*(b.close(s) for b in bs for s in list(b.positions)))

for name, fn in [("one at a time", flatten_sequential), ("concurrent", flatten_concurrent)]:
    bs = brokers(); t0 = time.perf_counter()
    await fn(bs)
    flat = all(not b.open and not any(b.positions.values()) for b in bs)
    took = time.perf_counter() - t0
    print(f"{name:14s}: {took:5.2f} s, flat = {flat}, {'within' if took < 5 else 'MISSES'} the 5 s target")"""),
    ("md", "## 5. Contract tests\n\n"
           "Every adapter (IB, Alpaca, the simulator) must behave the same way to the code above it. A **contract test suite** is one set "
           "of tests run against every adapter. `SloppyBroker` works in a quick demo, but the suite finds two problems that would cost money."),
    ("code", """pd.DataFrame({"SimBroker": p.contract_tests(p.SimBroker), "SloppyBroker": p.contract_tests(p.SloppyBroker)}).replace({True: "✔ pass", False: "✘ FAIL"})"""),
    ("md", "`retry_is_idempotent` is the one that matters most: after a timeout you resend the **same** client order id, and a correct "
           "adapter returns the existing order instead of buying twice (common mistake #3 in the lesson plan)."),
    ("md", "## Wrap-up\n\n"
           "* Know each contract's multiplier and expiry; roll before the volume leaves.\n"
           "* One pre-trade chain in front of every adapter; the kill switch is check number one.\n"
           "* Kill switch: cancel, then close; sticky across restarts; fast enough to matter.\n"
           "* Contract tests keep every adapter honest.\n"
           "* Graded version: `labs/part04/week16_safety` (`SimBroker` against the full contract suite, sticky `KillSwitch`) and the Clinic W4 kill drill."),
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
