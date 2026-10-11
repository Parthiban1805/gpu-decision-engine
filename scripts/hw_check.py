"""Hardware/environment check (PLAN.md §0 rule 8). Writes docs/env.md.

Runs `nvidia-smi` and the torch CUDA probe from the plan, and records the
interpreter and pinned environment so the result can be re-run and compared.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_PROBE = 'import torch; print(torch.__version__); print(torch.cuda.is_available())'


def sh(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return r.returncode, (r.stdout + r.stderr).strip()


def gpu_section() -> tuple[list[str], bool]:
    if not shutil.which("nvidia-smi"):
        return ["`nvidia-smi` not found on PATH: no NVIDIA GPU/driver visible."], False
    code, query = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"])
    _, full = sh(["nvidia-smi"])
    lines = ["| GPU | Memory | Driver |", "|---|---|---|"]
    found = code == 0 and bool(query)
    if found:
        for row in query.splitlines():
            lines.append("| " + " | ".join(p.strip() for p in row.split(",")) + " |")
    else:
        lines = [f"`nvidia-smi --query-gpu` failed (exit {code}): {query}"]
    lines += ["", "Raw `nvidia-smi`:", "", "```text", full, "```"]
    return lines, found


def torch_section() -> list[str]:
    code, out = sh([sys.executable, "-c", TORCH_PROBE])
    cmd = f'`python -c "{TORCH_PROBE}"`'
    if code != 0:
        last = out.splitlines()[-1] if out else "no output"
        return [f"{cmd} failed: `{last}`",
                "",
                "torch is not part of the P0 environment. It is added when a phase needs it "
                "(MLP set-encoder in P7, Qwen scorer in P9); re-run this check then."]
    version, cuda = (out.splitlines() + ["?", "?"])[:2]
    return [f"{cmd}", "", f"- torch version: `{version}`", f"- `torch.cuda.is_available()`: `{cuda}`"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "env.md")
    args = ap.parse_args()

    gpu_lines, has_gpu = gpu_section()
    _, pip_freeze = sh([sys.executable, "-m", "pip", "freeze", "--exclude-editable"])
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    doc = [
        "# Environment and hardware check",
        "",
        f"Generated {now} by `python tasks.py hwcheck` (`scripts/hw_check.py`). Re-run to refresh.",
        "",
        "## Host",
        "",
        f"- OS: {platform.platform()}",
        f"- Machine: {platform.machine()}, processor: {platform.processor() or 'unknown'}",
        f"- Python (venv): {sys.version.split()[0]} at `{Path(sys.executable).relative_to(ROOT) if Path(sys.executable).is_relative_to(ROOT) else sys.executable}`",
        "",
        "## GPU",
        "",
        *gpu_lines,
        "",
        "## PyTorch CUDA probe",
        "",
        *torch_section(),
        "",
        "## Implication for the phase plan",
        "",
        "Phases P0–P8 are CPU-only and do not depend on this result.",
        "",
        ("An NVIDIA GPU is visible, so P9+ (Qwen scorer) is not blocked by absence of a GPU. "
         "Whether its memory is enough for the chosen backbone (frozen and LoRA) must be "
         "measured at P9, not assumed."
         if has_gpu else
         "No NVIDIA GPU is visible: stop at P8 and use Colab/cloud GPU for P9+ (PLAN.md §0 rule 8)."),
        "",
        "## Pinned environment (`pip freeze`)",
        "",
        "Installed from `requirements.txt` via `python tasks.py setup`.",
        "",
        "```text",
        pip_freeze,
        "```",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(doc), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)} (gpu_visible={has_gpu})")


if __name__ == "__main__":
    main()
