# Data dictionary (raw traces)

Generated 2026-10-11T20:46:40+05:30 by `scripts/data_dictionary.py` (run by `python tasks.py data`) from the files under `data/raw/`. Column names are read from the files (or the trace's `.header` files) and checked against each dataset's README. Acquisition status: `data/raw/DOWNLOADS.md`.

## alibaba2020

Notes from the README (relevant to normalization in P2):

- Time: `start_time`/`end_time` are seconds with a constant offset removed (README); in UTC+8 they keep the original time of day and weekday but fake dates.
- `pai_job_table.start_time` is job **submission**; `pai_task_table.start_time` is task launch. README: the gap between job.start_time and the earliest task.start_time is the wait (scheduling latency). **Measured in P1 EDA: that gap is 0 s for all 732,355 Terminated jobs** (p99 and max both 0), so the trace carries no usable queueing delay; it gives arrivals, requests and runtimes, and waits must come from the simulator.
- `inst_id` should be treated as the job id (README); joins to `pai_group_tag_table.inst_id`.
- `plan_gpu` is a percentage per instance (50.0 = half a GPU); job GPU request = Σ plan_gpu × inst_num / 100. `plan_cpu` is percent of a core (600 = 6 cores); `plan_mem` is GB.
- `gpu_type` in the task table is the type **assigned** (outcome-ish); `gpu_type_spec` in the group-tag table is the type **requested** (empty = any) — use the latter as a decision-time feature.
- `status` values per README: Running, Terminated, Failed, Waiting; only Terminated is successful.
- No explicit training/inference column (PLAN.md §2.2 assumed Alibaba has one; it does not in these tables). Proxies: `task_name` (tensorflow 49%, worker 22%, ps 15%, PyTorchWorker 9%, evaluator 1.4%, ...) and `group_tag.workload` (tagged for ~10% of jobs: bert, ctr, nmt, ...). Any job_type mapping is `derived`.
- No VRAM requirement column in the downloaded tables. `pai_sensor_table` (not downloaded by default, 388 MB) has measured `max_gpu_wrk_mem` per instance — usage, an outcome, not a request.

### `pai_group_tag_table.csv`

- Path: `data/raw/alibaba2020/pai_group_tag_table.csv` — 1,055,032 rows; column names from `pai_group_tag_table.header` (CSV has no header row).
- README check: Columns match the README.

| Column | dtype (first 100,000 rows) | non-null % | example |
|---|---|---|---|
| `inst_id` | str | 100.0 | `c1bdfdda2af739f74be81079cfe176800093c...` |
| `user` | str | 100.0 | `58540f191766` |
| `gpu_type_spec` | str | 2.6 | `V100` |
| `group` | str | 100.0 | `ba4e3be6472ded359f4b1ca53d9bd53e` |
| `workload` | str | 7.6 | `bert` |

### `pai_job_table.csv`

- Path: `data/raw/alibaba2020/pai_job_table.csv` — 1,055,501 rows; column names from `pai_job_table.header` (CSV has no header row).
- README check: Columns match the README.

| Column | dtype (first 100,000 rows) | non-null % | example |
|---|---|---|---|
| `job_name` | str | 100.0 | `87a79c6373f7439d82f32a1c` |
| `inst_id` | str | 100.0 | `38e2d7187762a9241cc1cd5732dbe52a5d8b3...` |
| `user` | str | 100.0 | `74238accb90b` |
| `status` | str | 100.0 | `Running` |
| `start_time` | float64 | 99.5 | `1053513.0` |
| `end_time` | float64 | 78.4 | `1140852.0` |

### `pai_machine_spec.csv`

- Path: `data/raw/alibaba2020/pai_machine_spec.csv` — 1,897 rows; column names from `pai_machine_spec.header` (CSV has no header row).
- README check: Columns match the README.

| Column | dtype (first 1,897 rows) | non-null % | example |
|---|---|---|---|
| `machine` | str | 100.0 | `7399a758eb02bae1a3621236` |
| `gpu_type` | str | 100.0 | `CPU` |
| `cap_cpu` | int64 | 100.0 | `96` |
| `cap_mem` | int64 | 100.0 | `512` |
| `cap_gpu` | int64 | 100.0 | `0` |

### `pai_task_table.csv`

- Path: `data/raw/alibaba2020/pai_task_table.csv` — 1,261,050 rows; column names from `pai_task_table.header` (CSV has no header row).
- README check: Columns match the README.

| Column | dtype (first 100,000 rows) | non-null % | example |
|---|---|---|---|
| `job_name` | str | 100.0 | `c936346f45eccd34bf748541` |
| `task_name` | str | 100.0 | `tensorflow` |
| `inst_num` | float64 | 100.0 | `1.0` |
| `status` | str | 100.0 | `Terminated` |
| `start_time` | float64 | 99.7 | `2693235.0` |
| `end_time` | float64 | 75.6 | `2695847.0` |
| `plan_cpu` | float64 | 98.4 | `600.0` |
| `plan_mem` | float64 | 98.4 | `29.296875` |
| `plan_gpu` | float64 | 80.3 | `50.0` |
| `gpu_type` | str | 80.9 | `MISC` |

## alibaba2023

Notes from the README (relevant to normalization in P2):

- Times (`creation_time`, `scheduled_time`, `deletion_time`) are seconds (README).
- `gpu_milli` is the GPU share for GPU-sharing pods (num_gpu == 1), in milli-GPU (README).
- `gpu_spec` lists allowed GPU types separated by `|`; missing = unconstrained. `openb_pod_list_gpuspec33.csv` is the default list augmented with GPU-type constraints (README); `openb_node_list_gpu_node.csv` is the GPU-node subset of `openb_node_list_all_node.csv`.
- No train/inference field; `qos` (LS / Burstable / BE) is the closest proxy. No license file upstream.

### `openb_node_list_all_node.csv`

- Path: `data/raw/alibaba2023/csv/openb_node_list_all_node.csv` — 1,523 rows; column names from the CSV header row.
- README check: Columns match the README.

| Column | dtype (first 1,523 rows) | non-null % | example |
|---|---|---|---|
| `sn` | str | 100.0 | `openb-node-0000` |
| `cpu_milli` | int64 | 100.0 | `32000` |
| `memory_mib` | int64 | 100.0 | `262144` |
| `gpu` | int64 | 100.0 | `0` |
| `model` | str | 79.6 | `P100` |

### `openb_node_list_gpu_node.csv`

- Path: `data/raw/alibaba2023/csv/openb_node_list_gpu_node.csv` — 1,213 rows; column names from the CSV header row.
- README check: Columns match the README.

| Column | dtype (first 1,213 rows) | non-null % | example |
|---|---|---|---|
| `sn` | str | 100.0 | `openb-node-0000` |
| `cpu_milli` | int64 | 100.0 | `64000` |
| `memory_mib` | int64 | 100.0 | `262144` |
| `gpu` | int64 | 100.0 | `2` |
| `model` | str | 100.0 | `P100` |

### `openb_pod_list_default.csv`

- Path: `data/raw/alibaba2023/csv/openb_pod_list_default.csv` — 8,152 rows; column names from the CSV header row.
- README check: Columns match the README.

| Column | dtype (first 8,152 rows) | non-null % | example |
|---|---|---|---|
| `name` | str | 100.0 | `openb-pod-0000` |
| `cpu_milli` | int64 | 100.0 | `12000` |
| `memory_mib` | int64 | 100.0 | `16384` |
| `num_gpu` | int64 | 100.0 | `1` |
| `gpu_milli` | int64 | 100.0 | `1000` |
| `gpu_spec` | float64 | 0.0 | `` |
| `qos` | str | 100.0 | `LS` |
| `pod_phase` | str | 100.0 | `Running` |
| `creation_time` | int64 | 100.0 | `0` |
| `deletion_time` | int64 | 100.0 | `12537496` |
| `scheduled_time` | float64 | 89.0 | `0.0` |

### `openb_pod_list_gpuspec33.csv`

- Path: `data/raw/alibaba2023/csv/openb_pod_list_gpuspec33.csv` — 8,152 rows; column names from the CSV header row.
- README check: Columns match the README.

| Column | dtype (first 8,152 rows) | non-null % | example |
|---|---|---|---|
| `name` | str | 100.0 | `openb-pod-0000` |
| `cpu_milli` | int64 | 100.0 | `12000` |
| `memory_mib` | int64 | 100.0 | `16384` |
| `num_gpu` | int64 | 100.0 | `1` |
| `gpu_milli` | int64 | 100.0 | `1000` |
| `gpu_spec` | str | 29.3 | `V100M16|V100M32` |
| `qos` | str | 100.0 | `LS` |
| `pod_phase` | str | 100.0 | `Running` |
| `creation_time` | int64 | 100.0 | `0` |
| `deletion_time` | int64 | 100.0 | `12537496` |
| `scheduled_time` | float64 | 89.0 | `0.0` |

## helios

Notes from the README (relevant to normalization in P2):

- One `cluster_log.csv` + `cluster_gpu_number.csv` per cluster (Earth, Saturn, Uranus, Venus).
- `submit_time`/`start_time`/`end_time` are datetime strings; `duration` = end − start and `queue` = start − submit, both seconds (README notes 3 and 4).
- `state`: COMPLETED, CANCELLED, FAILED, TIMEOUT, NODE_FAIL (README treats the last two as failed).
- No job-type, GPU-type, memory or VRAM fields. `gpu_num` = 0 means a CPU job.
- `cluster_gpu_number.csv`: daily GPU count per virtual cluster (`vc*` columns) plus `total`.

### `Earth/cluster_log.csv`

- Path: `data/raw/helios/data/Earth/cluster_log.csv` — 872,886 rows; column names from the CSV header row.
- README check: Columns match the README.

| Column | dtype (first 100,000 rows) | non-null % | example |
|---|---|---|---|
| `job_id` | int64 | 100.0 | `1000001` |
| `user` | str | 100.0 | `usBbF` |
| `vc` | str | 100.0 | `vc3sl` |
| `gpu_num` | int64 | 100.0 | `0` |
| `cpu_num` | int64 | 100.0 | `1` |
| `node_num` | int64 | 100.0 | `1` |
| `state` | str | 100.0 | `FAILED` |
| `submit_time` | str | 100.0 | `2020-05-15 18:12:19` |
| `start_time` | str | 100.0 | `2020-05-15 18:12:19` |
| `end_time` | str | 100.0 | `2020-05-15 18:12:20` |
| `duration` | int64 | 100.0 | `1` |
| `queue` | int64 | 100.0 | `0` |

### `Earth/cluster_gpu_number.csv`

- Path: `data/raw/helios/data/Earth/cluster_gpu_number.csv` — 181 rows; column names from the CSV header row.
- README check: Columns match the README. (25 VC columns; first 3 shown.)

| Column | dtype (first 181 rows) | non-null % | example |
|---|---|---|---|
| `date` | str | 100.0 | `2020-04-01` |
| `vchbv` | int64 | 100.0 | `112` |
| `vc4om` | int64 | 100.0 | `0` |
| `vcVP5` | int64 | 100.0 | `0` |
| `total` | int64 | 100.0 | `784` |

### `Saturn/cluster_log.csv`

- Path: `data/raw/helios/data/Saturn/cluster_log.csv` — 1,753,078 rows; column names from the CSV header row.
- README check: Columns match the README.

| Column | dtype (first 100,000 rows) | non-null % | example |
|---|---|---|---|
| `job_id` | int64 | 100.0 | `2921826` |
| `user` | str | 100.0 | `uYlPU` |
| `vc` | str | 100.0 | `vcSE7` |
| `gpu_num` | int64 | 100.0 | `1` |
| `cpu_num` | int64 | 100.0 | `1` |
| `node_num` | int64 | 100.0 | `1` |
| `state` | str | 100.0 | `TIMEOUT` |
| `submit_time` | str | 100.0 | `2020-03-18 14:21:22` |
| `start_time` | str | 100.0 | `2020-03-18 14:21:22` |
| `end_time` | str | 100.0 | `2020-04-01 14:21:45` |
| `duration` | int64 | 100.0 | `1209623` |
| `queue` | int64 | 100.0 | `0` |

### `Saturn/cluster_gpu_number.csv`

- Path: `data/raw/helios/data/Saturn/cluster_gpu_number.csv` — 181 rows; column names from the CSV header row.
- README check: Columns match the README. (28 VC columns; first 3 shown.)

| Column | dtype (first 181 rows) | non-null % | example |
|---|---|---|---|
| `date` | str | 100.0 | `2020-04-01` |
| `vczIT` | int64 | 100.0 | `152` |
| `vcdoX` | int64 | 100.0 | `144` |
| `vcOIr` | int64 | 100.0 | `192` |
| `total` | int64 | 100.0 | `2072` |

### `Uranus/cluster_log.csv`

- Path: `data/raw/helios/data/Uranus/cluster_log.csv` — 490,309 rows; column names from the CSV header row.
- README check: Columns match the README.

| Column | dtype (first 100,000 rows) | non-null % | example |
|---|---|---|---|
| `job_id` | int64 | 100.0 | `1380066` |
| `user` | str | 100.0 | `uVCWx` |
| `vc` | str | 100.0 | `vcrV2` |
| `gpu_num` | int64 | 100.0 | `1` |
| `cpu_num` | int64 | 100.0 | `1` |
| `node_num` | int64 | 100.0 | `1` |
| `state` | str | 100.0 | `TIMEOUT` |
| `submit_time` | str | 100.0 | `2020-03-22 10:38:56` |
| `start_time` | str | 100.0 | `2020-03-22 10:38:56` |
| `end_time` | str | 100.0 | `2020-04-05 10:39:21` |
| `duration` | int64 | 100.0 | `1209625` |
| `queue` | int64 | 100.0 | `0` |

### `Uranus/cluster_gpu_number.csv`

- Path: `data/raw/helios/data/Uranus/cluster_gpu_number.csv` — 181 rows; column names from the CSV header row.
- README check: Columns match the README. (25 VC columns; first 3 shown.)

| Column | dtype (first 181 rows) | non-null % | example |
|---|---|---|---|
| `date` | str | 100.0 | `2020-04-01` |
| `vcVgR` | int64 | 100.0 | `240` |
| `vcrV2` | int64 | 100.0 | `288` |
| `vcW23` | int64 | 100.0 | `136` |
| `total` | int64 | 100.0 | `2136` |

### `Venus/cluster_log.csv`

- Path: `data/raw/helios/data/Venus/cluster_log.csv` — 246,708 rows; column names from the CSV header row.
- README check: Columns match the README.

| Column | dtype (first 100,000 rows) | non-null % | example |
|---|---|---|---|
| `job_id` | int64 | 100.0 | `363167` |
| `user` | str | 100.0 | `u1Eem` |
| `vc` | str | 100.0 | `vccJW` |
| `gpu_num` | int64 | 100.0 | `32` |
| `cpu_num` | int64 | 100.0 | `160` |
| `node_num` | int64 | 100.0 | `4` |
| `state` | str | 100.0 | `TIMEOUT` |
| `submit_time` | str | 100.0 | `2020-03-18 04:01:39` |
| `start_time` | str | 100.0 | `2020-03-18 04:01:39` |
| `end_time` | str | 100.0 | `2020-04-01 04:01:54` |
| `duration` | int64 | 100.0 | `1209615` |
| `queue` | int64 | 100.0 | `0` |

### `Venus/cluster_gpu_number.csv`

- Path: `data/raw/helios/data/Venus/cluster_gpu_number.csv` — 181 rows; column names from the CSV header row.
- README check: Columns match the README. (25 VC columns; first 3 shown.)

| Column | dtype (first 181 rows) | non-null % | example |
|---|---|---|---|
| `date` | str | 100.0 | `2020-04-01` |
| `vcEwI` | int64 | 100.0 | `72` |
| `vcWoR` | int64 | 100.0 | `64` |
| `vcHvQ` | int64 | 100.0 | `64` |
| `total` | int64 | 100.0 | `968` |

## philly

Not downloaded (awaiting user approval; ~0.98 GB compressed). Schema per its README: `cluster_job_log` (JSON: status, vc, jobid, attempts[start_time, end_time, detail[ip, gpus]], submitted_time, user), `cluster_machine_list` (machineId, number of GPUs, single GPU mem), plus per-minute GPU/CPU/memory utilization CSVs. Column names will be verified against the files when downloaded.
