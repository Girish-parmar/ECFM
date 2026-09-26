"""Platform exception hierarchy: catch precisely, never swallow errors in trading code."""


class TradingError(Exception):
    """Base class for all platform errors."""


class ConfigError(TradingError):
    """Invalid or missing configuration."""


class DataError(TradingError):
    """Invalid market data."""


class DataStale(DataError):
    """Market data older than allowed."""


class BrokerError(TradingError):
    """Broker rejected or failed a request."""


class RiskRejected(TradingError):
    """Risk engine rejected an order."""


class CurrencyMismatch(TradingError):
    """Arithmetic between different currencies."""
