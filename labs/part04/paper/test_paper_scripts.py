"""The paper scripts need real accounts, so CI only checks that they import (all broker calls are inside functions)."""
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("name", ["connect_both", "download_bars"])
def test_script_imports(name, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    spec = importlib.util.spec_from_file_location(f"paper_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except NotImplementedError:
        pytest.skip("week 13/14 starters not finished yet")
    assert callable(mod.main)
    sys.modules.pop(f"paper_{name}", None)
