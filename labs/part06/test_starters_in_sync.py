"""Meta test: the learner starters must match the instructor solutions with the answers removed.

Skipped automatically once `solutions/` has been removed from the learner copy of the labs.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOLUTIONS = ROOT / "solutions"
pytestmark = pytest.mark.skipif(not SOLUTIONS.exists(), reason="learner copy: no solutions/")


def _make_starters():
    spec = importlib.util.spec_from_file_location("make_starters", ROOT / "tools" / "make_starters.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_starters_are_up_to_date():
    assert _make_starters().main(check=True) == 0


def test_no_solution_leaks_into_starters():
    for sol in SOLUTIONS.rglob("*.py"):
        starter = ROOT / sol.relative_to(SOLUTIONS)
        assert "# >>> SOLUTION" not in starter.read_text(), starter
