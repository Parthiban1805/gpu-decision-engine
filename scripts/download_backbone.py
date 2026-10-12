"""Download the pinned Qwen backbone for this machine's GPU and verify it there.

Downloads exactly the model id + revision from configs/backbone.yaml into
data/models/<name>/ (gitignored), records SHA-256 checksums and provenance
(SHA256SUMS + SOURCES.md, which *are* committed), then verifies the snapshot on
the compute device from configs/runtime.yaml: load the backbone in bf16 (no LM
head, as the V2 scorer will use it), run one forward pass, and report measured
VRAM. Re-running is idempotent (existing files are kept); to re-fetch from
scratch, delete the model directory.

    python tasks.py download-qwen
    python scripts/download_backbone.py --no-verify   # download + checksums only
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from huggingface_hub import list_repo_files, snapshot_download

from gde import paths
from gde.models.backbone import backbone_local_dir, load_backbone_config
from gde.runtime import resolve_device

MIB = 1024**2
# Redundant weight formats: skipped when the repo also ships safetensors.
ALWAYS_IGNORE = ["*.gguf", "*.pth", "*.onnx", "original/*", "*.msgpack", "*.h5"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_license(snapshot: Path) -> str:
    for candidate in sorted(snapshot.glob("LICENSE*")) + sorted(snapshot.glob("license*")):
        if candidate.is_file():
            head = candidate.read_text(encoding="utf-8", errors="replace")[:400]
            for name in ("Apache License", "MIT License", "GNU Affero General Public License"):
                if name.lower() in head.lower():
                    return f"{name.split(' License')[0]} (see `{candidate.name}`)"
            return f"non-standard; see `{candidate.name}`"
    return "no LICENSE file found in the repository"


def verify_on_device(snapshot: Path, dtype_name: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    report: dict[str, Any] = {"device": str(device), "dtype": dtype_name}

    tok = AutoTokenizer.from_pretrained(snapshot)
    enc = tok(["state: 8 gpus, 3 free | candidates: RUN:job_a RUN:job_b WAIT"],
              return_tensors="pt", padding="max_length", max_length=64,
              truncation=True).to(device)  # type: ignore[union-attr]

    model = AutoModel.from_pretrained(snapshot, dtype=dtype, attn_implementation="sdpa").to(device)
    model.eval()
    report["params"] = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        report["weights_mib"] = round(torch.cuda.memory_allocated(device) / MIB, 1)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(**enc).last_hidden_state
        float(out.sum())  # force a device sync
    report["forward_s"] = round(time.perf_counter() - start, 3)
    report["output_shape"] = list(out.shape)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        report["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved(device) / MIB, 1)
        report["total_vram_mib"], report["free_vram_after_mib"] = (round(total / MIB, 1),
                                                                   round(free / MIB, 1))
    report["status"] = "ok"
    return report


def main() -> int:
    cfg = load_backbone_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=cfg["model_id"])
    ap.add_argument("--revision", default=cfg["revision"],
                    help="full commit hash; defaults to the pinned revision in backbone.yaml")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default=None,
                    help="verification dtype (default: bf16 if the GPU supports it)")
    ap.add_argument("--no-verify", action="store_true", help="skip the on-device forward pass")
    args = ap.parse_args()

    out_dir = backbone_local_dir(args.model)
    print(f"model: {args.model} @ {args.revision}")
    print(f"target: {out_dir.relative_to(paths.REPO_ROOT)}")

    repo_files = list_repo_files(args.model, revision=args.revision)
    ignore = list(ALWAYS_IGNORE)
    if any(f.endswith(".safetensors") for f in repo_files):
        ignore.append("*.bin")
    snapshot = Path(snapshot_download(repo_id=args.model, revision=args.revision,
                                      local_dir=out_dir, ignore_patterns=ignore))
    print(f"snapshot ready ({len(repo_files)} repo files listed; "
          f"ignored: {', '.join(p for p in ignore if p not in ALWAYS_IGNORE) or 'none'})")

    files = sorted(p for p in snapshot.rglob("*")
                   if p.is_file() and ".cache" not in p.parts
                   and p.name not in {"SOURCES.md", "SHA256SUMS"})
    digests = {p.name: sha256_file(p) for p in files}
    (snapshot / "SHA256SUMS").write_text(
        "".join(f"{digests[p.name]}  {p.name}\n" for p in files), encoding="utf-8")
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"checksums: {len(files)} files, {total_bytes / 1e9:.2f} GB -> SHA256SUMS")

    verify: dict[str, Any] = {"status": "skipped (--no-verify)"}
    if not args.no_verify:
        import torch

        if args.dtype is None:
            args.dtype = ("bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                          else "fp16")
        print(f"verifying on {resolve_device()} ({args.dtype}) ...")
        verify = verify_on_device(snapshot, args.dtype)
        print(json.dumps(verify, indent=2))

    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Qwen backbone snapshot", "",
        f"- Model: [{args.model}](https://huggingface.co/{args.model}) "
        f"(revision `{args.revision}`, pinned in `configs/backbone.yaml`)",
        f"- Retrieved: {now} by `python tasks.py download-qwen`",
        f"- License: {detect_license(snapshot)}",
        f"- Files: {len(files)}, {total_bytes / 1e9:.2f} GB; SHA-256 in `SHA256SUMS` "
        "(recomputed on every run). Weights are NOT committed; this provenance file is.",
        f"- Chosen for: this machine's GPU — measured fit recorded in `docs/env.md` "
        "(VRAM probe); pinned in `configs/backbone.yaml`.", "",
        "## On-device verification", "",
        "```json", json.dumps(verify, indent=2), "```", "",
        "## Files", "",
        "| File | Bytes | SHA-256 (first 12) |", "|---|---|---|",
    ]
    for p in files:
        lines.append(f"| `{p.name}` | {p.stat().st_size} | `{digests[p.name][:12]}` |")
    (snapshot / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {snapshot / 'SOURCES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
