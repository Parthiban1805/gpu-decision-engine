# P2 normalization: cleaning report

Generated 2026-10-12T17:10:30+05:30 by `python tasks.py normalize` (`src/gde/ingestion/normalize.py`).

## Synthetic / derived fields used (PLAN.md §2.2 disclosure)

- `deadline_ts`: **synthetic** for every dataset (no trace has deadlines/SLOs). Training slack ~ LogUniform(2, 10), batch_inference ~ LogUniform(1.5, 4) around the ACTUAL duration (label-side field by design).
- `vram_gb`: **derived** for fractional-GPU jobs (share x per-type VRAM constant), **synthetic** (seeded Beta) for whole-GPU jobs.
- `priority`: **synthetic** default 5 for alibaba2020/helios (no class field); **derived** from `qos` for alibaba2023.
- `job_type`: **derived** heuristic on runtime_est_s and num_gpus (no trace label).
- `runtime_est_s`: **derived** expanding log-mean of PRIOR jobs only (user_group+job_type -> job_type -> global; first job gets a 3600 s constant).
- GPU VRAM capacity constants are ASSUMPTIONS: T4=16GB, P100=16GB, V100=32GB, V100M16=16GB, V100M32=32GB, A10=24GB, MISC=16GB, G2=16GB, G3=16GB, default 16.0GB.
- Seeds: alibaba2020=20200001, alibaba2023=20230001, helios_earth=20210101, helios_saturn=20210102, helios_uranus=20210103, helios_venus=20210104.

## alibaba2020

alibaba2020: job rows read 1055501, task rows 1261050, tag rows 1055032
alibaba2020: jobs without any task row: 0 (duration from job table)

Drops (pipeline order):

| Reason | Rows |
|---|---|
| status_not_finished (Running/Waiting, censored) | 66591 |
| no_submit_time | 0 |
| no_gpu_request | 18558 |
| nonpositive_or_missing_duration | 230892 |

- rows out: **739460**; submit span: 68.5 days
- duration_s p50/p95: 623.0 / 24251.0 s; num_gpus p50/p95: 1.0 / 12.5
- job_type: training 88.4%, batch_inference 11.6% (derived; online_inference never comes from these traces)
- final_status: pass 96.7%, failed 3.3%

- schema validation: **PASS** -> `data\processed\alibaba2020.parquet`

## alibaba2023

alibaba2023: pod rows read 8152 (gpuspec33 list with GPU-type spec)

Drops (pipeline order):

| Reason | Rows |
|---|---|
| phase_not_finished (Running/Pending, censored snapshot) | 6090 |
| no_submit_time | 0 |
| no_scheduled_time | 0 |
| no_gpu_request | 8 |
| nonpositive_or_missing_duration | 0 |
| unknown_qos | 0 |

- rows out: **2054**; submit span: 40.1 days
- duration_s p50/p95: 234.5 / 5800.5 s; num_gpus p50/p95: 0.81 / 1.0
- job_type: training 97.7%, batch_inference 2.3% (derived; online_inference never comes from these traces)
- final_status: failed 91.0%, pass 9.0%

- schema validation: **PASS** -> `data\processed\alibaba2023.parquet`

## helios_earth

helios_earth: rows read 872886 from cluster_log.csv

Drops (pipeline order):

| Reason | Rows |
|---|---|
| state_not_finished (SUSPENDED/RUNNING) | 0 |
| cpu_job (gpu_num == 0) | 445738 |
| no_submit_time | 0 |
| nonpositive_or_missing_duration | 0 |

- rows out: **427148**; submit span: 191.3 days
- duration_s p50/p95: 234.0 / 14895.6 s; num_gpus p50/p95: 1.0 / 8.0
- job_type: training 100.0% (derived; online_inference never comes from these traces)
- final_status: pass 73.5%, killed 13.6%, failed 12.9%

- schema validation: **PASS** -> `data\processed\helios_earth.parquet`

## helios_saturn

helios_saturn: rows read 1753078 from cluster_log.csv

Drops (pipeline order):

| Reason | Rows |
|---|---|
| state_not_finished (SUSPENDED/RUNNING) | 0 |
| cpu_job (gpu_num == 0) | 1054182 |
| no_submit_time | 0 |
| nonpositive_or_missing_duration | 0 |

- rows out: **698896**; submit span: 193.4 days
- duration_s p50/p95: 124.0 / 13075.0 s; num_gpus p50/p95: 1.0 / 16.0
- job_type: training 100.0% (derived; online_inference never comes from these traces)
- final_status: pass 58.2%, killed 27.2%, failed 14.5%

- schema validation: **PASS** -> `data\processed\helios_saturn.parquet`

## helios_uranus

helios_uranus: rows read 490309 from cluster_log.csv

Drops (pipeline order):

| Reason | Rows |
|---|---|
| state_not_finished (SUSPENDED/RUNNING) | 78 |
| cpu_job (gpu_num == 0) | 161168 |
| no_submit_time | 0 |
| nonpositive_or_missing_duration | 0 |

- rows out: **329063**; submit span: 189.6 days
- duration_s p50/p95: 280.0 / 38682.8 s; num_gpus p50/p95: 1.0 / 16.0
- job_type: training 100.0% (derived; online_inference never comes from these traces)
- final_status: pass 60.8%, killed 20.1%, failed 19.2%

- schema validation: **PASS** -> `data\processed\helios_uranus.parquet`

## helios_venus

helios_venus: rows read 246708 from cluster_log.csv

Drops (pipeline order):

| Reason | Rows |
|---|---|
| state_not_finished (SUSPENDED/RUNNING) | 0 |
| cpu_job (gpu_num == 0) | 121405 |
| no_submit_time | 0 |
| nonpositive_or_missing_duration | 0 |

- rows out: **125303**; submit span: 193.8 days
- duration_s p50/p95: 204.0 / 58054.4 s; num_gpus p50/p95: 1.0 / 24.0
- job_type: training 100.0% (derived; online_inference never comes from these traces)
- final_status: pass 52.6%, killed 27.2%, failed 20.2%

- schema validation: **PASS** -> `data\processed\helios_venus.parquet`

