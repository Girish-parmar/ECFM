import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from _loader import load
from common import ohlcv

pr = load("week44_production", "production")
DEPLOY = Path(__file__).parent / "deploy"
COMPOSE = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text())
CI = yaml.safe_load((DEPLOY / "ci.yml").read_text())


# ----------------------------------------------------------------------------------- S13–S14
def test_the_committed_stack_and_ci_pass():
    assert pr.check_compose(COMPOSE) == []
    assert pr.check_ci(CI) == []


def test_compose_problems_are_caught():
    bad = copy.deepcopy(COMPOSE)
    s = bad["services"]
    s["db"]["image"] = "timescale/timescaledb:latest-pg16"                            # the lesson's placeholder
    s["redis"]["image"] = "redis"
    s["grafana"]["image"] = "grafana/grafana@sha256:" + "a" * 64                       # a digest IS pinned
    s["engine"]["environment"] = {"IB_PASSWORD": "hunter2", "DB_TOKEN": "${DB_TOKEN}", "LOG_LEVEL": "info"}
    s["api"]["environment"] = ["API_KEY=abc123", "MODE=paper"]
    del s["engine"]["restart"]
    del s["db"]["healthcheck"]
    s["api"]["ports"] = ["8000:8000"]
    s["api"]["depends_on"] = ["db"]
    assert pr.check_compose(bad) == ["unpinned image: db", "no healthcheck: db", "unpinned image: redis",
                                     "inline secret: engine.IB_PASSWORD", "no restart policy: engine",
                                     "inline secret: api.API_KEY", "exposed port: api", "unhealthy dependency: api"]


def test_ci_gates():
    ci = copy.deepcopy(CI)
    ci["jobs"]["test"]["steps"] = [st for st in ci["jobs"]["test"]["steps"]
                                   if "mypy" not in st.get("run", "") and "regression" not in st.get("run", "")]
    assert pr.check_ci(ci) == ["regression", "types"]
    assert pr.check_ci({"jobs": {}}) == sorted(pr.REQUIRED_CI)


def test_regression_test_catches_an_off_by_one():
    close = ohlcv(1500)["close"].to_numpy()
    baseline = pr.fixture_backtest(close)
    assert baseline["n_trades"] > 10 and 0 < baseline["exposure"] < 1
    assert pr.regression_check(pr.fixture_backtest(close), baseline) == []            # a refactor that changes nothing
    off_by_one = lambda x, n: pd.Series(x).rolling(n + 1).mean().to_numpy()        # noqa: E731
    looks_ahead = lambda x, n: pd.Series(x).rolling(n).mean().shift(-1).to_numpy()  # noqa: E731
    assert pr.regression_check(pr.fixture_backtest(close, off_by_one), baseline) != []
    assert pr.regression_check(pr.fixture_backtest(close, looks_ahead), baseline) != []
    assert pr.regression_check({"a": 1.0}, {"a": 1.0, "b": 2}) == ["b"]


# ----------------------------------------------------------------------------------- S15
FILLS = [{"seq": i, "symbol": s, "qty": q, "price": p} for i, (s, q, p) in enumerate(
    [("SPY", 100, 500.0), ("QQQ", -50, 400.0), ("SPY", -100, 505.0), ("IWM", 30, 200.0), ("QQQ", 20, 395.0)])]


def test_event_sourced_state():
    st = pr.rebuild(FILLS)
    assert st["positions"] == {"QQQ": -30, "IWM": 30} and st["last_seq"] == 4          # SPY closed: removed
    assert st["cash"] == pytest.approx(-100 * 500 + 50 * 400 + 100 * 505 - 30 * 200 - 20 * 395)
    assert pr.rebuild(FILLS + FILLS[:3]) == st                                          # replays are idempotent
    s0 = {"positions": {}, "cash": 0.0, "last_seq": -1}
    assert pr.apply_fill(s0, FILLS[0]) != s0 and s0["positions"] == {}                  # no mutation


def test_crash_recovery_behind_a_reconciliation_gate(tmp_path):
    snap = tmp_path / "state.json"
    truth = pr.rebuild(FILLS)
    for crash_after in range(len(FILLS) + 1):                                           # crash anywhere
        snap.unlink(missing_ok=True)
        if crash_after >= 2:
            pr.save_snapshot(pr.rebuild(FILLS[:2]), snap)                               # the last snapshot
        out = pr.recover(snap, FILLS, truth["positions"])
        assert out["state"] == truth and out["may_trade"] and out["breaks"] == []
    assert not (tmp_path / "state.json.tmp").exists()
    manual = {**truth["positions"], "TLT": 10}                                          # someone traded in TWS
    out = pr.recover(snap, FILLS, manual)
    assert not out["may_trade"] and out["breaks"] == [{"symbol": "TLT", "internal": 0, "broker": 10}]


def test_backups_are_verified_before_restore():
    tables = {"positions": {"SPY": 100}, "fills": FILLS}
    blob, digest = pr.backup(tables)
    assert pr.restore(blob, digest) == json.loads(json.dumps(tables))
    corrupted = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(ValueError, match="corrupt"):
        pr.restore(corrupted, digest)


# ----------------------------------------------------------------------------------- S16
def test_secret_scan():
    files = {"config.py": "DB = 'postgres://db'\npassword = 'hunter22'\n",
             "notes.md": "key: sk-ant-api03-ABCDEFGHIJKLMNOPQR\n",
             "ok.py": "password = os.environ['PW']\nkey = '${ANTHROPIC_API_KEY}'\n",
             "id_rsa": "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n"}
    assert pr.scan_secrets(files) == [("config.py", 2, "hardcoded_password"), ("id_rsa", 1, "private_key"),
                                      ("notes.md", 1, "anthropic_key")]


def test_threat_model_and_precautionary_limits():
    t = pd.DataFrame({"threat": ["stolen API key", "runaway strategy", "VPS outage", "fat-finger size"],
                      "likelihood": [2, 3, 3, 2], "impact": [5, 5, 3, 4], "control": ["", "", "", ""]})
    top = pr.rank_threats(t)
    assert top["threat"].tolist() == ["runaway strategy", "stolen API key", "VPS outage"]
    assert top["risk"].tolist() == [15, 10, 9]
    limits = {"max_qty": 1000, "max_notional": 250_000, "band_pct": 0.03}
    assert pr.precautionary_check({"qty": 500, "price": 100}, 100, limits) == []
    assert pr.precautionary_check({"qty": -5000, "price": 110}, 100, limits) == ["size", "notional", "price_band"]
    assert pr.precautionary_check({"qty": 600, "price": 500}, 499, limits) == ["notional"]
