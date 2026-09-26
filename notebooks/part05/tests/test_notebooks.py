"""Part 5 notebook tests (run from notebooks/part05:  python -m pytest -q tests).

- Solution notebooks run with P5_STRICT=1: every p.check must pass.
- Starter notebooks run with blanks (`...`): they must still execute end to end (checks fall back).
- Starter and solution notebooks must be identical except for the exercise cells.
"""
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parents[1]
STARTERS = sorted(HERE.glob("[0-9][0-9]_*.ipynb"))
SOLUTIONS = sorted((HERE / "solutions").glob("*_solution.ipynb"))


def run(path: Path, tmp_path: Path, monkeypatch, strict: bool):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(HERE / "p5lib.py", work)
    shutil.copy(path, work)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    if strict:
        monkeypatch.setenv("P5_STRICT", "1")
    else:
        monkeypatch.delenv("P5_STRICT", raising=False)
    nb = nbformat.read(work / path.name, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
    return nb


def outputs_text(nb) -> str:
    return "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))


@pytest.mark.parametrize("path", SOLUTIONS, ids=[p.stem for p in SOLUTIONS])
def test_solution_passes_all_checks(path, tmp_path, monkeypatch):
    nb = run(path, tmp_path, monkeypatch, strict=True)
    text = outputs_text(nb)
    assert "❌" not in text and text.count("✅") >= 1


@pytest.mark.parametrize("path", STARTERS, ids=[p.stem for p in STARTERS])
def test_starter_runs_with_blanks(path, tmp_path, monkeypatch):
    nb = run(path, tmp_path, monkeypatch, strict=False)
    assert "❌" in outputs_text(nb)              # blanks are reported, and the notebook still finished


@pytest.mark.parametrize("path", STARTERS, ids=[p.stem for p in STARTERS])
def test_starter_matches_solution_outside_exercises(path):
    s = nbformat.read(path, as_version=4)
    sol = nbformat.read(HERE / "solutions" / f"{path.stem}_solution.ipynb", as_version=4)
    sol_cells = sol.cells[1:]                     # skip the instructor banner
    assert len(s.cells) == len(sol_cells)
    for a, b in zip(s.cells, sol_cells):
        if "exercise" in a.metadata.get("tags", []):
            assert "..." in a.source and "..." not in b.source.replace("# ✍️", "")
        else:
            assert a.source == b.source
