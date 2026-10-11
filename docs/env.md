# Environment and hardware check

Generated 2026-10-11T19:57:29+05:30 by `python tasks.py hwcheck` (`scripts/hw_check.py`). Re-run to refresh.

## Host

- OS: Windows-10-10.0.26200-SP0
- Machine: AMD64, processor: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD
- Python (venv): 3.11.3 at `.venv\Scripts\python.exe`

## GPU

| GPU | Memory | Driver |
|---|---|---|
| NVIDIA GeForce RTX 3050 Laptop GPU | 4096 MiB | 592.00 |

Raw `nvidia-smi`:

```text
Sun Oct 11 19:57:24 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 592.00                 Driver Version: 592.00         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   60C    P8              6W /   75W |     104MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           30436    C+G   ...al\Programs\cursor\Cursor.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

## PyTorch

- torch version: `2.14.1+cu130`
- CUDA version (torch build): `13.0`
- torch.cuda.is_available(): `True`
- GPU name: `NVIDIA GeForce RTX 3050 Laptop GPU`
- compute capability: `8.6`
- bf16 supported: `True`
- cuDNN: `92400`
- total VRAM (MiB): `4096`
- free VRAM (MiB): `3300`
- configs/runtime.yaml device resolves to: `cuda`

## Qwen VRAM probe (`scripts/vram_probe.py`)

### `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c1`

Measured 2026-10-11T19:46:54+05:30 with `python tasks.py vram-probe`: Qwen3ForCausalLM, 596M params, bf16, seq_len 512, attention `sdpa`, torch 2.14.1+cu130 / CUDA 13.0.
Weights on GPU: 1136.9 MiB. Already in use before loading (CUDA context + other processes): 807.8 MiB of 4095.5 MiB.

| Run | Batch | Status | Peak allocated (MiB) | Peak reserved (MiB) | Est. total in use (MiB) | Time (s) |
|---|---|---|---|---|---|---|
| frozen forward | 1 | fits | 1215.3 | 1242.0 | 2049.8 | 2.05 |
| frozen forward | 4 | fits | 1421.3 | 1452.0 | 2259.8 | 0.54 |
| frozen forward | 8 | fits | 1697.3 | 1740.0 | 2547.8 | 1.055 |
| LoRA step | 1 | fits | 1247.8 | 1294.0 | 2101.8 | 7.095 |
| LoRA step | 4 | fits | 1493.0 | 1590.0 | 2397.8 | 1.956 |
| LoRA step | 8 | fits | 1775.8 | 1898.0 | 2705.8 | 2.86 |

LoRA config: r=16, alpha=32, targets=q,k,v,o_proj, gradient_checkpointing=True, optimizer=AdamW, trainable_params=4850177, adapter_dtype=fp32, autocast=bf16, head=MLP 1024->256->1 over 17 marker positions

### `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2ee41`

Measured 2026-10-11T19:56:45+05:30 with `python tasks.py vram-probe`: Qwen3_5ForConditionalGeneration, 853M params, bf16, seq_len 512, attention `sdpa`, torch 2.14.1+cu130 / CUDA 13.0.
Weights on GPU: 1656.1 MiB. Already in use before loading (CUDA context + other processes): 807.8 MiB of 4095.5 MiB.

| Run | Batch | Status | Peak allocated (MiB) | Peak reserved (MiB) | Est. total in use (MiB) | Time (s) |
|---|---|---|---|---|---|---|
| frozen forward | 1 | fits | 1748.2 | 1812.0 | 2619.8 | 1.491 |
| frozen forward | 4 | fits | 1994.7 | 2084.0 | 2891.8 | 0.863 |
| frozen forward | 8 | fits | 2323.3 | 2466.0 | 3273.8 | 1.612 |
| LoRA step | 1 | fits | 1828.6 | 1884.0 | 2691.8 | 2.553 |
| LoRA step | 4 | fits | 2277.3 | 2368.0 | 3175.8 | 2.79 |
| LoRA step | 8 | fits | 2863.1 | 2966.0 | 3773.8 | 4.811 |

LoRA config: r=16, alpha=32, targets=q,k,v,o_proj, gradient_checkpointing=True, optimizer=AdamW, trainable_params=1344001, adapter_dtype=fp32, autocast=bf16, head=MLP 1024->256->1 over 17 marker positions

"Fits" = peak reserved + memory in use before loading ≤ dedicated VRAM. On Windows the driver can spill into shared system RAM instead of raising OOM, so a run that completes can still exceed dedicated VRAM (and becomes much slower).

Times are single runs and the first run of each section includes CUDA warm-up; they are not latency measurements (P9 measures p50/p95 latency properly). The same LoRA settings give different trainable-parameter counts per model because only layers with standard q/k/v/o attention projections receive adapters.

## Implication for the phase plan

Simulator, baselines and labeling (P3–P6) are CPU-only and parallelized across cores; they do not depend on the GPU. MLP training (P7) and the Qwen scorer (P9+) use the GPU via `configs/runtime.yaml` and fall back to CPU.

- `Qwen/Qwen3-0.6B-Base` (bf16, seq_len 512): frozen forward fits at batch 1, 4, 8; one LoRA step with gradient checkpointing fits in 4 GB at batch 1, 4, 8.
- `Qwen/Qwen3.5-0.8B-Base` (bf16, seq_len 512): frozen forward fits at batch 1, 4, 8; one LoRA step with gradient checkpointing fits in 4 GB at batch 1, 4, 8.

## Pinned environment (`pip freeze`)

Installed from `requirements.txt` via `python tasks.py setup`.

```text
accelerate==1.15.0
annotated-doc==0.0.5
anyio==4.15.1
click==8.5.0
colorama==0.4.6
filelock==4.1.0
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.13.1
httpx2==2.13.1
huggingface_hub==2.2.0
hypothesis==6.169.0
idna==3.20
iniconfig==2.3.1
Jinja2==3.1.6
markdown-it-py==4.2.0
MarkupSafe==3.0.4
mdurl==0.1.2
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
packaging==26.3
peft==0.21.2
pluggy==1.6.0
psutil==7.2.2
Pygments==2.21.0
pytest==9.1.1
PyYAML==6.0.3
regex==2026.9.29
rich==15.0.0
safetensors==0.8.0
shellingham==1.5.4
sortedcontainers==2.4.0
sympy==1.14.0
tokenizers==0.23.3
torch==2.14.1+cu130
tqdm==4.70.1
transformers==5.19.0
truststore==0.10.4
typer==0.27.3
typing_extensions==4.16.0
```
