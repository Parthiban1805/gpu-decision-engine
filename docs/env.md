# Environment and hardware check

Generated 2026-10-12T16:39:15+05:30 by `python tasks.py hwcheck` (`scripts/hw_check.py`). Re-run to refresh.

## Host

- OS: Windows-10-10.0.26200-SP0
- Machine: AMD64, processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- Python (venv): 3.11.5 at `.venv\Scripts\python.exe`

## GPU

| GPU | Memory | Driver |
|---|---|---|
| NVIDIA GeForce RTX 3060 | 12288 MiB | 591.86 |

Raw `nvidia-smi`:

```text
Mon Oct 12 16:39:13 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.86                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060      WDDM  |   00000000:04:00.0 Off |                  N/A |
| 53%   48C    P2             51W /  170W |    1124MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5248    C+G   ...em32\ApplicationFrameHost.exe      N/A      |
|    0   N/A  N/A            6680    C+G   ...e\dotnet\PAD.Console.Host.exe      N/A      |
|    0   N/A  N/A            8644    C+G   ...t\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A           10608    C+G   ...2txyewy\CrossDeviceResume.exe      N/A      |
|    0   N/A  N/A           10652    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           11516    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           12344    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           12492    C+G   ...5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A           12656    C+G   ..._cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A           12768    C+G   ...y\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A           15048    C+G   ...xyewy\ShellExperienceHost.exe      N/A      |
|    0   N/A  N/A           15936    C+G   ....0.4258.53\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           17416    C+G   ...ntrolPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A           19332    C+G   ...desk\Autodesk AdSSO\AdSSO.exe      N/A      |
|    0   N/A  N/A           19908    C+G   ...launcher\AdskAccessUIHost.exe      N/A      |
|    0   N/A  N/A           20996    C+G   ...lare WARP\Cloudflare WARP.exe      N/A      |
|    0   N/A  N/A           22740    C+G   ....0.4258.53\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           26228    C+G   ....0.4258.62\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           27724    C+G   ...8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A           27768    C+G   ...ocal\Programs\ZCode\ZCode.exe      N/A      |
|    0   N/A  N/A           30148    C+G   ...yb3d8bbwe\WindowsTerminal.exe      N/A      |
|    0   N/A  N/A           33628    C+G   ...indows\System32\ShellHost.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

## PyTorch

- torch version: `2.14.1+cu130`
- CUDA version (torch build): `13.0`
- torch.cuda.is_available(): `True`
- GPU name: `NVIDIA GeForce RTX 3060`
- compute capability: `8.6`
- bf16 supported: `True`
- cuDNN: `92400`
- total VRAM (MiB): `12288`
- free VRAM (MiB): `11245`
- configs/runtime.yaml device resolves to: `cuda`

## Qwen VRAM probe (`scripts/vram_probe.py`)

### `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c1`

Measured 2026-10-12T16:38:15+05:30 with `python tasks.py vram-probe`: Qwen3ForCausalLM, 596M params, bf16, seq_len 512, attention `sdpa`, torch 2.14.1+cu130 / CUDA 13.0.
Weights on GPU: 1136.9 MiB. Already in use before loading (CUDA context + other processes): 1060.5 MiB of 12287.5 MiB.

| Run | Batch | Status | Peak allocated (MiB) | Peak reserved (MiB) | Est. total in use (MiB) | Time (s) |
|---|---|---|---|---|---|---|
| frozen forward | 1 | fits | 1215.3 | 1242.0 | 2302.5 | 0.226 |
| frozen forward | 4 | fits | 1421.3 | 1452.0 | 2512.5 | 0.153 |
| frozen forward | 8 | fits | 1697.3 | 1740.0 | 2800.5 | 0.292 |
| LoRA step | 1 | fits | 1247.8 | 1294.0 | 2354.5 | 0.495 |
| LoRA step | 4 | fits | 1493.0 | 1590.0 | 2650.5 | 0.669 |
| LoRA step | 8 | fits | 1775.8 | 1898.0 | 2958.5 | 1.274 |

LoRA config: r=16, alpha=32, targets=q,k,v,o_proj, gradient_checkpointing=True, optimizer=AdamW, trainable_params=4850177, adapter_dtype=fp32, autocast=bf16, head=MLP 1024->256->1 over 17 marker positions

"Fits" = peak reserved + memory in use before loading ≤ dedicated VRAM. On Windows the driver can spill into shared system RAM instead of raising OOM, so a run that completes can still exceed dedicated VRAM (and becomes much slower).

Times are single runs and the first run of each section includes CUDA warm-up; they are not latency measurements (P9 measures p50/p95 latency properly). The same LoRA settings give different trainable-parameter counts per model because only layers with standard q/k/v/o attention projections receive adapters.

## Implication for the phase plan

Simulator, baselines and labeling (P3–P6) are CPU-only and parallelized across cores; they do not depend on the GPU. MLP training (P7) and the Qwen scorer (P9+) use the GPU via `configs/runtime.yaml` and fall back to CPU.

- `Qwen/Qwen3-0.6B-Base` (bf16, seq_len 512): frozen forward fits at batch 1, 4, 8; one LoRA step with gradient checkpointing fits in this GPU's 12 GB at batch 1, 4, 8.

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
pandas==3.0.6
peft==0.21.2
pluggy==1.6.0
psutil==7.2.2
pyarrow==26.0.0
Pygments==2.21.0
pytest==9.1.1
python-dateutil==2.9.0.post0
PyYAML==6.0.3
regex==2026.9.29
rich==15.0.0
safetensors==0.8.0
shellingham==1.5.4
six==1.17.0
sortedcontainers==2.4.0
sympy==1.14.0
tokenizers==0.23.3
torch @ file:///E:/gpu-decision-engine/gpu-decision-engine/.wheels/torch-2.14.1%2Bcu130-cp311-cp311-win_amd64.whl#sha256=e98711737150b5a6451cbbfdb425abd5d1bea77f4a1a5d29c48d5ec41cd54949
tqdm==4.70.1
transformers==5.19.0
truststore==0.10.4
typer==0.27.3
typing_extensions==4.16.0
tzdata==2026.5
```
