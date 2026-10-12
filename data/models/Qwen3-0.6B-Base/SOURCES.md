# Qwen backbone snapshot

- Model: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, pinned in `configs/backbone.yaml`)
- Retrieved: 2026-10-12T16:39:12+05:30 by `python tasks.py download-qwen`
- License: Apache (see `LICENSE`)
- Files: 10, 1.20 GB; SHA-256 in `SHA256SUMS` (recomputed on every run). Weights are NOT committed; this provenance file is.
- Chosen for: this machine's GPU — measured fit recorded in `docs/env.md` (VRAM probe); pinned in `configs/backbone.yaml`.

## On-device verification

```json
{
  "device": "cuda",
  "dtype": "bf16",
  "params": 596049920,
  "weights_mib": 1136.9,
  "forward_s": 0.261,
  "output_shape": [
    1,
    64,
    1024
  ],
  "peak_reserved_mib": 1168.0,
  "total_vram_mib": 12287.5,
  "free_vram_after_mib": 10045.0,
  "status": "ok"
}
```

## Files

| File | Bytes | SHA-256 (first 12) |
|---|---|---|
| `.gitattributes` | 1519 | `11ad7efa2497` |
| `config.json` | 727 | `504a6b58c427` |
| `generation_config.json` | 138 | `8c970692323e` |
| `LICENSE` | 11343 | `832dd9e00a68` |
| `merges.txt` | 1671853 | `8831e4f1a044` |
| `model.safetensors` | 1192135096 | `cd2a512003e2` |
| `README.md` | 2973 | `910d9be25c64` |
| `tokenizer.json` | 7031645 | `c0382117ea32` |
| `tokenizer_config.json` | 9678 | `3c04ed3ca964` |
| `vocab.json` | 2776833 | `ca10d7e9fb3e` |
