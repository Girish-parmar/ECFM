"""Shared set-up for the paper-account scripts: import path + settings from environment / .env (given)."""
import sys
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # labs/part04


class PaperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="QF_", extra="ignore")

    env: str = "paper"
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 11
    alpaca_key: SecretStr | None = None
    alpaca_secret: SecretStr | None = None
