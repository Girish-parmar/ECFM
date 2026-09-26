# ADR 0001: Decimal for prices, quantities and money

- **Status:** accepted
- **Context:** binary floats cannot represent most decimal prices exactly (0.1 + 0.2 != 0.3), which breaks tick rounding, P&L reconciliation and equality checks.
- **Decision:** `decimal.Decimal` (constructed from strings) for prices, quantities, fees and money; floats only for analytics (returns, statistics).
- **Consequences:** slower arithmetic (irrelevant at our order rates); conversions at the analytics boundary must be explicit.

Write ADR 0002 (FIFO accounting) and 0003 (asyncio) yourself.
