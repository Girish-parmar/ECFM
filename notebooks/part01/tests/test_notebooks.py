"""Execute every Part 1 notebook end to end with SYNTHETIC offline fixtures (no internet needed).

Run from notebooks/part01:  python -m pytest -q tests/test_notebooks.py
"""
import os
import shutil
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from make_fixtures import make  # noqa: E402

NOTEBOOKS = sorted(HERE.glob("[0-9][0-9]_*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_runs(path, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(HERE / "p1lib.py", work)
    shutil.copy(path, work)
    monkeypatch.setenv("P1_FIXTURES", str(make(tmp_path / "fixtures")))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(work / path.name, as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors
    assert (work / "research_log.csv").exists()
