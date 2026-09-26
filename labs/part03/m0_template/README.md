# quantforge — M0 repository template

Start your own `quantforge` repository from this folder in week 12 (Part 3, S21–S24):

```bash
cp -r labs/part03/m0_template ~/quantforge && cd ~/quantforge && git init
uv sync && uv run pytest -q          # the provided core tests pass out of the box
uv run pre-commit install
```

**Provided (working):** `core/config.py` (typed settings, secrets hidden), `core/logging.py` (JSON logs with correlation
ids), `core/errors.py` (exception hierarchy), `core/clock.py` (live and simulated clocks), `pyproject.toml`
(ruff, mypy strict, pytest), pre-commit (ruff, gitleaks, nbstripout), CI workflow, ADR template.

**You add (M0 checklist, see the Part 3 lesson plan, Section 7):**
- [ ] `core/events.py` — your Week 10 `EventBus`
- [ ] `domain/` — your Week 9 domain model + Week 10 order state machine (see `domain/README.md`)
- [ ] `data/` — your Week 11 data functions, a `BarRepository`, `sql/schema.sql`
- [ ] `stats/`, `analytics/metrics.py` — promoted from Part 2
- [ ] tests: ≥ 90% coverage on `core/` and `domain/`, including the P&L property test
- [ ] `mypy --strict` and `ruff` clean; CI green
- [ ] `docs/uml/` diagrams and ADRs 0002–0003
