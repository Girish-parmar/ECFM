# domain/ (M0)

Move your Week 9 code here, split into modules:
- `instrument.py` — `AssetClass`, `Instrument`, `Future` (+ `Option`, `FXPair`, `Crypto` for M0)
- `market.py` — `Bar`, `Tick`, `Quote` value objects with validation
- `order.py` — order request + your Week 10 `OrderStateMachine`
- `position.py` — FIFO `Position`, `Fill`; `Money`
Keep the Week 9 property test (P&L reconciles with cash) in `tests/property/`.
