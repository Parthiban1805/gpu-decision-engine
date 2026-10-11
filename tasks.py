"""Cross-platform task runner for GDE (replaces a Makefile).

Usage: python tasks.py <target> [extra args passed to the underlying tool]

Uses only the standard library so it runs with any Python >= 3.8; every target
except `setup` executes inside the project venv (.venv).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
SUPPORTED_PY = ((3, 11), (3, 12))


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[str | os.PathLike[str]]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def require_venv() -> Path:
    py = venv_python()
    if not py.exists():
        sys.exit("No venv found. Run `python tasks.py setup` first.")
    return py


def find_base_python() -> list[str]:
    """Return a command for a supported (3.11/3.12) interpreter to build the venv."""
    if sys.version_info[:2] in SUPPORTED_PY:
        return [sys.executable]
    if os.name == "nt" and shutil.which("py"):
        for major, minor in reversed(SUPPORTED_PY):
            probe = subprocess.run(["py", f"-{major}.{minor}", "-c", "pass"], capture_output=True)
            if probe.returncode == 0:
                return ["py", f"-{major}.{minor}"]
    for major, minor in reversed(SUPPORTED_PY):
        exe = shutil.which(f"python{major}.{minor}")
        if exe:
            return [exe]
    sys.exit("Python 3.11 or 3.12 is required to create the venv; none was found.")


def task_setup(args: list[str]) -> None:
    """Create .venv with Python 3.11/3.12 and install pinned deps + the gde package."""
    if not venv_python().exists():
        run([*find_base_python(), "-m", "venv", VENV])
    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])
    run([py, "-m", "pip", "install", "--no-deps", "-e", ROOT])


def task_lock(args: list[str]) -> None:
    """Re-resolve requirements.in and rewrite the fully pinned requirements.txt."""
    py = require_venv()
    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.in"])
    frozen = subprocess.run(
        [str(py), "-m", "pip", "freeze", "--exclude-editable"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    header = "# Fully pinned lock file. Regenerate with: python tasks.py lock\n"
    (ROOT / "requirements.txt").write_text(header + frozen, encoding="utf-8")
    print("wrote requirements.txt")


def task_hwcheck(args: list[str]) -> None:
    """Run the hardware/environment check and write docs/env.md."""
    run([require_venv(), ROOT / "scripts" / "hw_check.py", *args])


def task_test(args: list[str]) -> None:
    """CI-style check: interpreter version, dependency consistency, pytest."""
    py = require_venv()
    run([py, "-c",
         "import sys; v = sys.version_info[:2]; "
         f"assert v in {SUPPORTED_PY!r}, f'venv Python {{v}} not supported'; "
         "print('venv python', sys.version.split()[0])"])
    run([py, "-m", "pip", "check"])
    run([py, "-m", "pytest", *args])


def not_implemented(phase: str) -> Callable[[list[str]], None]:
    def task(args: list[str]) -> None:
        sys.exit(f"Not implemented yet (arrives in phase {phase} of PLAN.md).")
    task.__doc__ = f"(phase {phase}, not implemented yet)"
    return task


TASKS: dict[str, Callable[[list[str]], None]] = {
    "setup": task_setup,
    "lock": task_lock,
    "hwcheck": task_hwcheck,
    "test": task_test,
    "data": not_implemented("P1"),
    "sim-smoke": not_implemented("P3"),
    "baselines": not_implemented("P4"),
    "label": not_implemented("P6"),
    "train-v1": not_implemented("P7"),
    "bench": not_implemented("P8"),
    "train-v2": not_implemented("P9"),
    "api": not_implemented("P12"),
}


def main(argv: list[str]) -> None:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        for name, fn in TASKS.items():
            print(f"  {name:<10} {(fn.__doc__ or '').strip()}")
        return
    name, rest = argv[0], argv[1:]
    if name not in TASKS:
        sys.exit(f"Unknown target {name!r}. Known: {', '.join(TASKS)}")
    TASKS[name](rest)


if __name__ == "__main__":
    main(sys.argv[1:])
