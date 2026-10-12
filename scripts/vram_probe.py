"""Measure peak VRAM of a small Qwen backbone on this GPU. Writes docs/vram_probe.json.

(a) frozen forward pass at several batch sizes, (b) one LoRA training step with gradient
checkpointing. The backbone is loaded without the LM head (AutoModel): the V2 scorer reads
hidden states at candidate-marker positions and never computes vocabulary logits.

"Fits" means the run's peak reserved memory plus memory already in use before loading
(CUDA context, other processes) stays within dedicated VRAM. On Windows (WDDM) the driver
can silently spill into shared system RAM instead of raising OOM, so absence of an OOM
error alone is not treated as fitting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from gde import paths
from gde.models.backbone import resolve_backbone_source
from gde.runtime import resolve_device

MIB = 1024**2
N_CANDIDATES = 17  # K=16 jobs + WAIT (PLAN.md §4.3)


def mib(n: int) -> float:
    return round(n / MIB, 1)


def measure(fn, device: torch.device, used_before_load: int, total: int) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        fn()
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "OOM", "error": str(exc).splitlines()[0]}
    elapsed = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    need = used_before_load + peak_reserved
    return {
        "status": "fits" if need <= total else "exceeds dedicated VRAM",
        "peak_allocated_mib": mib(peak_alloc),
        "peak_reserved_mib": mib(peak_reserved),
        "est_total_in_use_mib": mib(need),
        "seconds": round(elapsed, 3),
    }


def main() -> None:
    backbone_cfg = yaml.safe_load((paths.CONFIGS / "backbone.yaml").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=backbone_cfg["model_id"])
    ap.add_argument("--revision", default=None,
                    help="default: pinned hash for the configured model, else the repo's main branch")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default=None, help="default: bf16 if supported")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--lora-batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--out", type=Path, default=paths.DOCS / "vram_probe.json")
    args = ap.parse_args()
    if args.revision is None:
        args.revision = (backbone_cfg["revision"] if args.model == backbone_cfg["model_id"]
                         else "main")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModel

    device = resolve_device()
    if device.type != "cuda":
        raise SystemExit("vram_probe needs CUDA (configs/runtime.yaml device resolved to cpu).")
    if args.dtype is None:
        args.dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    torch.manual_seed(0)
    torch.zeros(1, device=device)
    free, total = torch.cuda.mem_get_info(device)
    used_before_load = total - free

    source, load_revision = resolve_backbone_source(args.model, args.revision)
    config = AutoConfig.from_pretrained(source, revision=load_revision)
    model = AutoModel.from_pretrained(
        source, revision=load_revision, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    hidden = model.config.get_text_config().hidden_size
    vocab = model.config.get_text_config().vocab_size
    weights_mib = mib(torch.cuda.memory_allocated(device))
    n_params = sum(p.numel() for p in model.parameters())

    def batch(bs: int) -> dict[str, torch.Tensor]:
        ids = torch.randint(0, vocab, (bs, args.seq_len), device=device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    frozen: dict[str, Any] = {}
    for bs in args.batches:
        inputs = batch(bs)

        def fwd(inputs=inputs) -> None:
            with torch.inference_mode():
                model(**inputs).last_hidden_state.sum().item()

        frozen[str(bs)] = measure(fwd, device, used_before_load, total)
        del inputs

    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_cfg)
    for p in peft_model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 1)).to(device)
    trainable = [p for p in peft_model.parameters() if p.requires_grad] + list(head.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    marker_pos = torch.linspace(0, args.seq_len - 1, N_CANDIDATES, device=device).long()

    lora: dict[str, Any] = {}
    for bs in args.lora_batches:
        inputs = batch(bs)
        target = torch.randint(0, N_CANDIDATES, (bs,), device=device)

        def step(inputs=inputs, target=target) -> None:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                h = peft_model(**inputs).last_hidden_state[:, marker_pos, :]
                logits = head(h.float()).squeeze(-1)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            opt.step()
            loss.item()

        lora[str(bs)] = measure(step, device, used_before_load, total)
        del inputs, target

    result = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python tasks.py vram-probe",
        "model_id": args.model,
        "revision": args.revision,
        "loaded_from": source,
        "architecture": (config.architectures or ["?"])[0],
        "params": n_params,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "attn_implementation": "sdpa",
        "gpu": torch.cuda.get_device_name(device),
        "total_vram_mib": mib(total),
        "in_use_before_load_mib": mib(used_before_load),
        "weights_mib": weights_mib,
        "frozen_forward": frozen,
        "lora_step": {
            "config": {"r": args.lora_r, "alpha": 2 * args.lora_r, "targets": "q,k,v,o_proj",
                       "gradient_checkpointing": True, "optimizer": "AdamW",
                       "trainable_params": n_trainable, "adapter_dtype": "fp32",
                       "autocast": args.dtype, "head": f"MLP {hidden}->256->1 over {N_CANDIDATES} marker positions"},
            "by_batch": lora,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    existing[args.model] = result
    args.out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
