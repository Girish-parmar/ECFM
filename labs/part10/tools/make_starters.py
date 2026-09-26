"""Generate the learner (starter) files from the instructor solutions.

Every block between `# >>> SOLUTION` and `# <<< SOLUTION` is replaced by
`raise NotImplementedError(...)`, or by `pass` when the marker is `# >>> SOLUTION (pass)`
(used where the object must still construct, e.g. validation in __post_init__).
Non-Python files (tests, data) are not touched: tests live next to the starters.

Run from labs/part10:   python tools/make_starters.py          (writes files)
                        python tools/make_starters.py --check  (exit 1 if a starter is out of date)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS = ROOT / "solutions"
BLOCK = re.compile(r"^(?P<indent>[ \t]*)# >>> SOLUTION(?P<mode> \(pass\))?\n.*?^[ \t]*# <<< SOLUTION\n", re.S | re.M)


def to_starter(source: str) -> str:
    def repl(m):
        ind = m.group("indent")
        if m.group("mode"):
            return f"{ind}pass  # ✍️ Your turn: see the docstring\n"
        return f'{ind}raise NotImplementedError("✍️ Your turn: see the docstring")\n'
    return BLOCK.sub(repl, source)


def main(check: bool = False) -> int:
    stale = []
    for sol in sorted(SOLUTIONS.rglob("*.py")):
        rel = sol.relative_to(SOLUTIONS)
        target = ROOT / rel
        text = to_starter(sol.read_text())
        if check:
            if not target.exists() or target.read_text() != text:
                stale.append(str(rel))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    if check and stale:
        print("Out of date starters (run tools/make_starters.py):", *stale, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(check="--check" in sys.argv))
