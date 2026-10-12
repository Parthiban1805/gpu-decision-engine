# P3 simulator smoke — FIFO on the trace replayer

Generated 2026-10-12T19:15:11 by `python tasks.py sim-smoke` (config `configs/sim_p3.yaml`, seed 30001).

Cluster: `generic_24gb` × 8 (24 GB each), power budget 3.0 kW. Trace: `data/processed/alibaba2020.parquet`, window 86400 s, target rho 0.8, max_jobs 5000, gpu_type ignore.

Window scaling: window start 2107480, 8266 jobs in window, dropped 636 ({'too_many_gpus': 622, 'vram_too_big': 14, 'nonpositive_duration': 0}), 468 fractional-above-1 jobs ceiled to whole GPUs, W = 49.1M GPU-s over 5000 sampled jobs -> **T_sim = 88.7 sim-days** (arrival scale factor 0.0113; durations unscaled, deadline slack preserved in seconds). Whale jobs dominate W, so the 24 h trace window maps to a long, sparse simulated arrival process.

## Runs (FIFO, strict head-of-line)

| est_mode | ov.sh | jobs tr/ov | rho realized tr/ov/tot | util(AW) | wait p50(h)/p95(d) tr | wait p50(h)/p95(d) ov | SLO miss | DL miss | makespan d | events |
|---|---|---|---|---|---|---|---|---|---|---|
| stored | 0.00 | 5000/0 | 0.561/-/0.800 | 0.561 | 506.6h/37.1d | 0.0h/0.0d | - | 0.953 | 119.8d | 25000 |
| stored | 0.10 | 5000/500 | 0.561/0.000/0.800 | 0.561 | 507.0h/37.1d | 371.0h/36.8d | 0.932 | 0.953 | 119.8d | 27500 |
| stored | 0.25 | 5000/1250 | 0.560/0.000/0.800 | 0.560 | 508.0h/37.1d | 507.3h/36.4d | 0.934 | 0.953 | 119.8d | 31250 |
| oracle | 0.00 | 5000/0 | 0.561/-/0.800 | 0.561 | 506.6h/37.1d | 0.0h/0.0d | - | 0.953 | 119.8d | 25000 |
| oracle | 0.10 | 5000/500 | 0.560/0.000/0.800 | 0.560 | 507.9h/37.1d | 549.6h/37.0d | 0.938 | 0.953 | 119.9d | 27500 |
| oracle | 0.25 | 5000/1250 | 0.560/0.000/0.800 | 0.560 | 507.5h/37.1d | 520.3h/37.0d | 0.936 | 0.953 | 119.8d | 31250 |
| noisy | 0.00 | 5000/0 | 0.561/-/0.800 | 0.561 | 506.6h/37.1d | 0.0h/0.0d | - | 0.953 | 119.8d | 25000 |
| noisy | 0.10 | 5000/500 | 0.562/0.000/0.800 | 0.562 | 503.8h/36.9d | 503.7h/36.3d | 0.934 | 0.953 | 119.7d | 27500 |
| noisy | 0.25 | 5000/1250 | 0.561/0.000/0.800 | 0.561 | 504.5h/37.0d | 547.4h/36.8d | 0.936 | 0.953 | 119.8d | 31250 |

AW = arrival window (utilization/load measured there, amendment C); makespan covers the post-arrival drain. Overlay load is reported separately from trace load (overlay jobs are 5-30 s on a 0.1-0.5 share, so their load share is small by construction at this window scale).

## Reading of the P3 baseline (honest, un-tuned)

Strict FIFO cannot skip a blocked head, and the trace mix contains large multi-GPU jobs: at target rho=0.8 the cluster idles ~30% of the arrival window (utilization ~0.56 vs offered 0.8) while the queue grows, waits reach weeks, deadline misses ~95% and overlay SLO misses ~93%. This is the head-of-line blocking pathology that EASY backfilling (P4) exists to relieve and is the number to beat; nothing here was tuned to produce it.

## Gate re-checks at smoke level

- PASS — determinism (same-seed digest)
- PASS — fork reproduces original trajectory
- PASS — no VRAM/share overcommit at any event (all runs)
- PASS — GPU-hours <= capacity x time (all runs)
- PASS — no guard trips / deadlocks (all runs)

## Performance (amendment D)

- Throughput (largest run): **4,865 events/s** (31250 events / 6.4 s, CPU-only)
- snapshot+fork at queue length 10: **140 µs**
- snapshot+fork at queue length 100: **235 µs**
- snapshot+fork at queue length 1000: **1220 µs**
- candidate (view) build at queue length 1000: **149 µs** (view_pending_cap 64)
- max-decisions-per-timestep guard: 1000 (no trip in any run: PASS)

## Synthetic fields used (disclosure)

- Deadlines/SLOs and priorities: **synthetic** for all trace jobs (no public trace has them; PLAN §2.3 rules, unchanged).
- VRAM: `derived` (fractional-share jobs) or `synthetic` (whole-GPU jobs, seeded Beta) — as produced by P2, unchanged.
- `job_type` is `derived` for trace jobs; `batch_inference` is a **short-small proxy** (duration <= 600 s AND num_gpus <= 0.5 heuristic, not tuned — P2b decision).
- Overlay jobs are **fully synthetic** (duration, share, priority, SLO, VRAM; every field tagged `synthetic`, group `overlay`).
- `runtime_est_s`: `stored` = P2 derived estimate (MdAPE 45.3% on alibaba2020); `oracle`/`noisy(sigma=0.25)` are simulation-time ablations tagged `synthetic`. Helios runtime estimates remain weak by decision (not used in this smoke).
- compute_factor assumed 1.0 when targeting rho; realized load above is the measured value.
