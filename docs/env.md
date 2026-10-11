# Environment and hardware check

Generated 2026-10-11T19:11:39+05:30 by `python tasks.py hwcheck` (`scripts/hw_check.py`). Re-run to refresh.

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
Sun Oct 11 19:11:38 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 592.00                 Driver Version: 592.00         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   60C    P5              7W /   75W |      86MiB /   4096MiB |     38%      Default |
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

## PyTorch CUDA probe

`python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"` failed: `ModuleNotFoundError: No module named 'torch'`

torch is not part of the P0 environment. It is added when a phase needs it (MLP set-encoder in P7, Qwen scorer in P9); re-run this check then.

## Implication for the phase plan

Phases P0–P8 are CPU-only and do not depend on this result.

An NVIDIA GPU is visible, so P9+ (Qwen scorer) is not blocked by absence of a GPU. Whether its memory is enough for the chosen backbone (frozen and LoRA) must be measured at P9, not assumed.

## Pinned environment (`pip freeze`)

Installed from `requirements.txt` via `python tasks.py setup`.

```text
colorama==0.4.6
hypothesis==6.169.0
iniconfig==2.3.1
packaging==26.3
pluggy==1.6.0
Pygments==2.21.0
pytest==9.1.1
PyYAML==6.0.3
sortedcontainers==2.4.0
```
