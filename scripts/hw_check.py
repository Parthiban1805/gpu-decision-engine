"""Hardware/environment check (PLAN.md §0 rule 8). Writes docs/env.md.

Runs `nvidia-smi` and the torch CUDA probe, renders docs/vram_probe.json (if present),
and records the interpreter and pinned environment so the result can be re-run and compared.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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


def torch_section() -> tuple[list[str], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        return [f"torch is not importable: `{exc}`. Run `python tasks.py setup`."], {}
    info: dict[str, Any] = {
        "torch version": torch.__version__,
        "CUDA version (torch build)": torch.version.cuda,
        "torch.cuda.is_available()": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        major, minor = torch.cuda.get_device_capability(0)
        info |= {
            "GPU name": torch.cuda.get_device_name(0),
            "compute capability": f"{major}.{minor}",
            "bf16 supported": torch.cuda.is_bf16_supported(),
            "cuDNN": torch.backends.cudnn.version(),
            "total VRAM (MiB)": round(total / 2**20),
            "free VRAM (MiB)": round(free / 2**20),
        }
    from gde.runtime import resolve_device
    info["configs/runtime.yaml device resolves to"] = str(resolve_device())
    return [f"- {k}: `{v}`" for k, v in info.items()], info


def vram_probe_section() -> list[str]:
    path = ROOT / "docs" / "vram_probe.json"
    if not path.exists():
        return ["Not run yet. Run `python tasks.py vram-probe`."]
    out: list[str] = []
    for model_id, r in json.loads(path.read_text(encoding="utf-8")).items():
        out += [
            f"### `{model_id}` @ `{r['revision'][:12]}`",
            "",
            f"Measured {r['generated']} with `{r['command']}`: {r['architecture']}, "
            f"{r['params'] / 1e6:.0f}M params, {r['dtype']}, seq_len {r['seq_len']}, "
            f"attention `{r['attn_implementation']}`, torch {r['torch']} / CUDA {r['cuda']}.",
            f"Weights on GPU: {r['weights_mib']} MiB. Already in use before loading "
            f"(CUDA context + other processes): {r['in_use_before_load_mib']} MiB of {r['total_vram_mib']} MiB.",
            "",
            "| Run | Batch | Status | Peak allocated (MiB) | Peak reserved (MiB) | Est. total in use (MiB) | Time (s) |",
            "|---|---|---|---|---|---|---|",
        ]
        runs = [("frozen forward", r["frozen_forward"]), ("LoRA step", r["lora_step"]["by_batch"])]
        for label, by_batch in runs:
            for bs, m in by_batch.items():
                if m["status"] == "OOM":
                    out.append(f"| {label} | {bs} | **OOM** | – | – | – | – |")
                else:
                    status = m["status"] if m["status"] == "fits" else f"**{m['status']}**"
                    out.append(f"| {label} | {bs} | {status} | {m['peak_allocated_mib']} | "
                               f"{m['peak_reserved_mib']} | {m['est_total_in_use_mib']} | {m['seconds']} |")
        cfg = r["lora_step"]["config"]
        out += ["", "LoRA config: " + ", ".join(f"{k}={v}" for k, v in cfg.items()), ""]
    out += [
        "\"Fits\" = peak reserved + memory in use before loading ≤ dedicated VRAM. On Windows the "
        "driver can spill into shared system RAM instead of raising OOM, so a run that completes "
        "can still exceed dedicated VRAM (and becomes much slower).",
        "",
        "Times are single runs and the first run of each section includes CUDA warm-up; they are "
        "not latency measurements (P9 measures p50/p95 latency properly). The same LoRA settings "
        "give different trainable-parameter counts per model because only layers with standard "
        "q/k/v/o attention projections receive adapters.",
    ]
    return out


def implication_section(cuda_ok: bool) -> list[str]:
    out = [
        "Simulator, baselines and labeling (P3–P6) are CPU-only and parallelized across cores; "
        "they do not depend on the GPU. MLP training (P7) and the Qwen scorer (P9+) use the GPU "
        "via `configs/runtime.yaml` and fall back to CPU.",
        "",
    ]
    if not cuda_ok:
        return out + ["torch cannot use CUDA here: stop at P8 and use Colab/cloud GPU for P9+ "
                      "(PLAN.md §0 rule 8)."]
    path = ROOT / "docs" / "vram_probe.json"
    if not path.exists():
        return out + ["CUDA works. Qwen memory fit is not measured yet (`python tasks.py vram-probe`)."]
    for model_id, r in json.loads(path.read_text(encoding="utf-8")).items():
        def fitting(by_batch: dict[str, Any]) -> list[str]:
            return [bs for bs, m in by_batch.items() if m["status"] == "fits"]
        fwd = fitting(r["frozen_forward"])
        lora = fitting(r["lora_step"]["by_batch"])
        fwd_txt = f"fits at batch {', '.join(fwd)}" if fwd else "does not fit at any tested batch size"
        lora_txt = (f"fits in this GPU's {r['total_vram_mib'] / 1024:.0f} GB at batch {', '.join(lora)}" if lora
                    else "**does not fit in this GPU's dedicated VRAM at any tested batch size**")
        out.append(f"- `{model_id}` ({r['dtype']}, seq_len {r['seq_len']}): frozen forward {fwd_txt}; "
                   f"one LoRA step with gradient checkpointing {lora_txt}.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "env.md")
    args = ap.parse_args()

    gpu_lines, has_gpu = gpu_section()
    torch_lines, torch_info = torch_section()
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
        "## PyTorch",
        "",
        *torch_lines,
        "",
        "## Qwen VRAM probe (`scripts/vram_probe.py`)",
        "",
        *vram_probe_section(),
        "",
        "## Implication for the phase plan",
        "",
        *implication_section(bool(torch_info.get("torch.cuda.is_available()"))),
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
