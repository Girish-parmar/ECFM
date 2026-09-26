"""Typed configuration from environment variables / .env (prefix QF_). Secrets never appear in repr or logs."""
from __future__ import annotations

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QF_", env_file=".env", extra="ignore")

    env: Literal["paper", "live", "backtest"] = "paper"
    log_level: str = "INFO"
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002                    # Gateway paper 4002 / live 4001
    ib_client_id: int = 1
    alpaca_key: SecretStr = SecretStr("")
    alpaca_secret: SecretStr = SecretStr("")
    database_url: SecretStr = SecretStr("postgresql://localhost/quantforge")
    data_dir: str = "data"
    max_order_notional: float = 50_000.0

    def require_live_confirmation(self) -> None:
        """Call before any live trading: refuses to run live with paper-style ports by mistake."""
        from quantforge.core.errors import ConfigError

        if self.env == "live" and self.ib_port in (4002, 7497):
            raise ConfigError("env=live but ib_port is a PAPER port")
