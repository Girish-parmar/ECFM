import io
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from quantforge.core import errors
from quantforge.core.clock import LiveClock, SimClock
from quantforge.core.config import Settings
from quantforge.core.logging import JsonFormatter, new_correlation_id


def test_settings_from_env_hide_secrets(monkeypatch):
    monkeypatch.setenv("QF_IB_PORT", "4001")
    monkeypatch.setenv("QF_ALPACA_KEY", "super-secret")
    s = Settings()
    assert s.ib_port == 4001 and "super-secret" not in repr(s)
    assert s.alpaca_key.get_secret_value() == "super-secret"


def test_live_with_paper_port_is_refused(monkeypatch):
    monkeypatch.setenv("QF_ENV", "live")
    with pytest.raises(errors.ConfigError):
        Settings().require_live_confirmation()


def test_json_logging_with_correlation_id():
    buf = io.StringIO()
    h = logging.StreamHandler(buf); h.setFormatter(JsonFormatter())
    log = logging.getLogger("t"); log.handlers = [h]; log.setLevel(logging.INFO); log.propagate = False
    cid = new_correlation_id()
    log.info("order sent", extra={"fields": {"symbol": "SPY", "qty": 10}})
    rec = json.loads(buf.getvalue())
    assert rec["cid"] == cid and rec["symbol"] == "SPY" and rec["msg"] == "order sent"


def test_error_hierarchy():
    assert issubclass(errors.DataStale, errors.DataError) and issubclass(errors.RiskRejected, errors.TradingError)


def test_clocks():
    assert LiveClock().now().tzinfo is not None
    t0 = datetime(2025, 3, 3, 14, 30, tzinfo=timezone.utc)
    c = SimClock(t0)
    c.advance(timedelta(minutes=5))
    assert c.now() == t0 + timedelta(minutes=5)
    with pytest.raises(ValueError):
        c.advance_to(t0)
    with pytest.raises(ValueError):
        SimClock(datetime(2025, 1, 1))
