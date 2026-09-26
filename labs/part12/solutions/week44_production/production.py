"""Week 44 (S13–S16) — Production engineering: checks for the Docker Compose stack and the CI workflow, backtest
regression tests that catch silent behaviour changes, event-sourced state with snapshots and crash recovery behind
a reconciliation gate, verified backups, secret scanning, threat-model ranking and broker-side precautionary
limits.

Rule of the week: a failure you have not rehearsed is a failure you will handle badly.
Fill in every block marked "Your turn", then run:  python -m pytest week44_production
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

CRITICAL = ("engine", "api", "ib-gateway", "db")
SECRET_VAR = re.compile(r"PASSWORD|SECRET|TOKEN|API_KEY|PASS$", re.I)


# ------------------------------------------------------------------------------ S13 deployment
def check_compose(spec: dict) -> list[str]:
    """Problems in a parsed docker-compose file ([] = passes), in this order per service (services in file order):
    "unpinned image: <svc>" — an image with no ":tag", or a tag containing "latest" (a digest "@sha256:" is pinned);
    "inline secret: <svc>.<VAR>" — an `environment` entry whose NAME matches SECRET_VAR and whose value is not a
    "${...}" reference (environment may be a mapping or a list of "KEY=value"); "no restart policy: <svc>" for CRITICAL services; "no healthcheck: db";
    "exposed port: <svc>" — any service except "proxy" publishing ports; "unhealthy dependency: <svc>" — a service
    that depends on db without condition service_healthy (a list-style depends_on counts as unconditional)."""
    # >>> SOLUTION
    problems = []
    for name, svc in spec.get("services", {}).items():
        image = svc.get("image")
        if image and "@sha256:" not in image:
            tag = image.rsplit(":", 1)[1] if ":" in image.split("/")[-1] else ""
            if not tag or "latest" in tag:
                problems.append(f"unpinned image: {name}")
        env = svc.get("environment") or {}
        if isinstance(env, list):                                                     # ["KEY=value", …] form
            env = dict(e.split("=", 1) if "=" in e else (e, "") for e in env)
        for var, val in env.items():
            if SECRET_VAR.search(str(var)) and not str(val).startswith("${"):
                problems.append(f"inline secret: {name}.{var}")
        if name in CRITICAL and "restart" not in svc:
            problems.append(f"no restart policy: {name}")
        if name == "db" and "healthcheck" not in svc:
            problems.append("no healthcheck: db")
        if svc.get("ports") and name != "proxy":
            problems.append(f"exposed port: {name}")
        dep = svc.get("depends_on", {})
        if "db" in dep and (isinstance(dep, list) or dep["db"].get("condition") != "service_healthy"):
            problems.append(f"unhealthy dependency: {name}")
    return problems
    # <<< SOLUTION


REQUIRED_CI = {"lint": "ruff check", "types": "mypy --strict", "architecture": "lint-imports",
               "coverage": "--cov-fail-under", "regression": "tests/regression"}


def check_ci(workflow: dict) -> list[str]:
    """Names of REQUIRED_CI gates missing from the "run" commands of ANY job's steps (sorted)."""
    # >>> SOLUTION
    runs = " \n".join(step.get("run", "") for job in workflow.get("jobs", {}).values() for step in job.get("steps", []))
    return sorted(k for k, needle in REQUIRED_CI.items() if needle not in runs)
    # <<< SOLUTION


# ------------------------------------------------------------------------------ S14 regression tests
def sma(x: np.ndarray, n: int) -> np.ndarray:
    """The CORRECT simple moving average (given): mean of the last n values including today, NaN before."""
    return pd.Series(x).rolling(n).mean().to_numpy()


def fixture_backtest(close: np.ndarray, sma_fn=sma, fast: int = 10, slow: int = 40, cost_bps: float = 5.0) -> dict:
    """The regression fixture: long when sma_fn(fast) > sma_fn(slow) (decided at the close, held the NEXT day),
    else flat; costs on position changes. Return {"total_return" (sum of daily net returns, rounded to 10 decimals),
    "n_trades" (position changes), "exposure" (share of days long)}."""
    # >>> SOLUTION
    r = np.r_[0.0, np.diff(close) / close[:-1]]
    pos = np.nan_to_num((sma_fn(close, fast) > sma_fn(close, slow)).astype(float))
    held = np.r_[0.0, pos[:-1]]
    changes = np.abs(np.diff(np.r_[0.0, held]))
    net = held * r - changes * cost_bps / 1e4
    return {"total_return": round(float(net.sum()), 10), "n_trades": int(changes.sum()), "exposure": float(held.mean())}
    # <<< SOLUTION


def regression_check(result: dict, baseline: dict, rtol: float = 1e-9) -> list[str]:
    """Keys whose value differs from the stored baseline (numbers: beyond rtol relative, 1e-12 absolute; keys
    missing on either side also differ). Sorted. [] = behaviour unchanged."""
    # >>> SOLUTION
    bad = []
    for k in sorted(set(result) | set(baseline)):
        if k not in result or k not in baseline:
            bad.append(k)
        elif not np.isclose(result[k], baseline[k], rtol=rtol, atol=1e-12):
            bad.append(k)
    return bad
    # <<< SOLUTION


# ------------------------------------------------------------------------ S15 reliability & recovery
def apply_fill(state: dict, fill: dict) -> dict:
    """New state after a fill {symbol, qty (signed), price}: positions[symbol] += qty (removed when 0),
    cash −= qty·price, and last_seq = fill["seq"]. Does not modify the input."""
    # >>> SOLUTION
    pos = dict(state["positions"])
    pos[fill["symbol"]] = pos.get(fill["symbol"], 0) + fill["qty"]
    if pos[fill["symbol"]] == 0:
        del pos[fill["symbol"]]
    return {"positions": pos, "cash": state["cash"] - fill["qty"] * fill["price"], "last_seq": fill["seq"]}
    # <<< SOLUTION


def rebuild(events: list[dict], state: dict | None = None) -> dict:
    """Replay fills (skipping seq <= the state's last_seq: idempotent) onto a state (default: no positions, cash 0,
    last_seq −1)."""
    # >>> SOLUTION
    state = state or {"positions": {}, "cash": 0.0, "last_seq": -1}
    for f in events:
        if f["seq"] > state["last_seq"]:
            state = apply_fill(state, f)
    return state
    # <<< SOLUTION


def save_snapshot(state: dict, path: Path) -> None:
    """Write the state atomically: to "<path>.tmp" first, then rename over path (a crash mid-write never leaves a
    half-written snapshot)."""
    # >>> SOLUTION
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    tmp.replace(path)
    # <<< SOLUTION


def recover(snapshot: Path, journal: list[dict], broker_positions: dict[str, float]) -> dict:
    """Startup recovery: load the snapshot (if the file exists, else an empty state), replay the fill journal on
    top (rebuild skips what the snapshot already holds), reconcile with the broker. Return {"state", "breaks" (list
    of {symbol, internal, broker}), "may_trade": no breaks} — trading resumes only after a clean reconciliation."""
    # >>> SOLUTION
    snapshot = Path(snapshot)
    state = json.loads(snapshot.read_text()) if snapshot.exists() else None
    state = rebuild(journal, state)
    breaks = [{"symbol": s, "internal": state["positions"].get(s, 0), "broker": broker_positions.get(s, 0)}
              for s in sorted(set(state["positions"]) | set(broker_positions))
              if state["positions"].get(s, 0) != broker_positions.get(s, 0)]
    return {"state": state, "breaks": breaks, "may_trade": not breaks}
    # <<< SOLUTION


def backup(tables: dict) -> tuple[bytes, str]:
    """Compressed JSON dump (zlib) and its sha256 hex digest (store them apart)."""
    # >>> SOLUTION
    blob = zlib.compress(json.dumps(tables, sort_keys=True, default=str).encode())
    return blob, hashlib.sha256(blob).hexdigest()
    # <<< SOLUTION


def restore(blob: bytes, digest: str) -> dict:
    """Verify the checksum BEFORE restoring (ValueError "corrupt backup" if it differs), then decompress."""
    # >>> SOLUTION
    if hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("corrupt backup")
    return json.loads(zlib.decompress(blob))
    # <<< SOLUTION


# ------------------------------------------------------------------------ S16 security & op-risk
SECRET_RULES = {"anthropic_key": r"sk-ant-[A-Za-z0-9_-]{16,}", "aws_access_key": r"AKIA[0-9A-Z]{16}",
                "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                "hardcoded_password": r"(?i)password\s*[=:]\s*['\"][^'\"$]{4,}['\"]"}


def scan_secrets(files: dict[str, str]) -> list[tuple[str, int, str]]:
    """(file, 1-based line number, rule name) for every SECRET_RULES match, sorted."""
    # >>> SOLUTION
    out = []
    for name, text in files.items():
        for i, line in enumerate(text.splitlines(), start=1):
            for rule, pat in SECRET_RULES.items():
                if re.search(pat, line):
                    out.append((name, i, rule))
    return sorted(out)
    # <<< SOLUTION


def rank_threats(threats: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """threats: threat, likelihood (1–5), impact (1–5), control. Add risk = likelihood × impact; return the `top`
    rows by risk (ties: higher impact first, then threat name)."""
    # >>> SOLUTION
    t = threats.assign(risk=threats["likelihood"] * threats["impact"])
    return t.sort_values(["risk", "impact", "threat"], ascending=[False, False, True]).head(top).reset_index(drop=True)
    # <<< SOLUTION


def precautionary_check(order: dict, last_price: float, limits: dict) -> list[str]:
    """Broker-side precautionary limits, a second line behind the platform's risk engine: "size" (|qty| >
    max_qty), "notional" (|qty|·price > max_notional), "price_band" (limit price more than band_pct away from the
    last price). Returns the failed checks in that order."""
    # >>> SOLUTION
    out = []
    if abs(order["qty"]) > limits["max_qty"]:
        out.append("size")
    if abs(order["qty"]) * order["price"] > limits["max_notional"]:
        out.append("notional")
    if abs(order["price"] / last_price - 1) > limits["band_pct"]:
        out.append("price_band")
    return out
    # <<< SOLUTION
