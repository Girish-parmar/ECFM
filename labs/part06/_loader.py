"""Import a lab module: your version by default, the instructor solution when P6_SOLUTIONS=1."""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(folder: str, module: str):
    base = ROOT / "solutions" if os.environ.get("P6_SOLUTIONS") == "1" else ROOT
    path = base / folder / f"{module}.py"
    name = f"p6_{'sol' if base != ROOT else 'lab'}_{folder}_{module}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod            # needed for dataclasses and pickling
    spec.loader.exec_module(mod)
    return mod
