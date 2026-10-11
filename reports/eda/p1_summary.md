# P1 EDA summary (raw traces)

Generated 2026-10-11T20:43:53+05:30 by `python tasks.py eda` (`scripts/eda.py`) from files under `data/raw/`. All values are computed from measured trace fields; nothing synthetic is used here.

## alibaba2020 (cluster-trace-gpu-v2020, PAI)

- Jobs: **1,055,501** (rows in pai_job_table; job_name unique: True); tasks: 1,261,050; users: 1,433; span ≈ 68.9 days.

Job status (README: only 'Terminated' is successful) (n=1,055,501):

| value | count | share |
|---|---|---|
| Terminated | 732,355 | 69.4% |
| Failed | 256,555 | 24.3% |
| Running | 62,928 | 6.0% |
| Waiting | 3,663 | 0.3% |

Jobs with no task rows: 574.

Runtime of Terminated jobs = max(task.end_time) - min(task.start_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 732,355 | 5,382.0 | 43.0 | 144.0 | 663.0 | 3,871.0 | 12,848.0 | 68,883.8 | 626,384.0 |

Wait of Terminated jobs = min(task.start_time) - job.start_time (submit) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 732,355 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

GPU request per job = Σ plan_gpu × inst_num / 100 over tasks (GPUs):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 1,055,501 | 2.6 | 0.1 | 0.2 | 1.0 | 1.0 | 5.0 | 32.0 | 1,000.0 |

GPU request per job, binned (n=1,055,501):

| value | count | share |
|---|---|---|
| (0,1) fractional | 420,545 | 39.8% |
| 1 | 373,921 | 35.4% |
| (2,8] | 100,159 | 9.5% |
| >8 | 75,279 | 7.1% |
| (1,2] | 48,000 | 4.5% |
| 0 | 37,597 | 3.6% |

Per-instance plan_gpu (%) of GPU tasks (n=1,037,085):

| value | count | share |
|---|---|---|
| 100 | 442,591 | 42.7% |
| (0,25] | 425,078 | 41.0% |
| (25,50] | 137,955 | 13.3% |
| >100 | 30,349 | 2.9% |
| (50,100) | 1,112 | 0.1% |

Task gpu_type (assigned GPU type) (n=1,261,050):

| value | count | share |
|---|---|---|
| MISC | 696,281 | 55.2% |
| T4 | 226,923 | 18.0% |
| <missing> | 217,738 | 17.3% |
| P100 | 72,732 | 5.8% |
| V100 | 28,797 | 2.3% |
| V100M32 | 18,579 | 1.5% |

Task role / task_name (job-type proxy; no explicit train/inference field) (n=1,261,050):

| value | count | share |
|---|---|---|
| tensorflow | 621,415 | 49.3% |
| worker | 275,785 | 21.9% |
| ps | 183,283 | 14.5% |
| PyTorchWorker | 110,784 | 8.8% |
| xComputeWorker | 27,402 | 2.2% |
| evaluator | 17,210 | 1.4% |
| TensorboardTask | 10,681 | 0.8% |
| ReduceTask | 4,136 | 0.3% |
| DecoderWorker | 4,136 | 0.3% |
| JupyterTask | 2,066 | 0.2% |
| TVMTuneMain | 1,158 | 0.1% |
| OpenmpiTracker | 745 | 0.1% |
| (12 others) | 2,249 | 0.2% |

group_tag.workload (README: ~9% of instances tagged) (n=1,055,032):

| value | count | share |
|---|---|---|
| <missing> | 952,234 | 90.3% |
| bert | 54,887 | 5.2% |
| ctr | 27,083 | 2.6% |
| nmt | 10,363 | 1.0% |
| inception | 5,440 | 0.5% |
| graphlearn | 3,754 | 0.4% |
| resnet | 541 | 0.1% |
| xlnet | 415 | 0.0% |
| rl | 249 | 0.0% |
| vgg | 66 | 0.0% |

group_tag.gpu_type_spec (requested GPU type; empty = any) (n=1,055,032):

| value | count | share |
|---|---|---|
| <missing> | 1,034,399 | 98.0% |
| P100 | 7,653 | 0.7% |
| V100M32 | 6,112 | 0.6% |
| V100 | 5,750 | 0.5% |
| T4 | 1,099 | 0.1% |
| MISC | 19 | 0.0% |

Machines by gpu_type (pai_machine_spec):

| gpu_type | machines | GPUs |
|---|---|---|
| CPU | 83 | 0 |
| MISC | 280 | 2,240 |
| P100 | 798 | 1,596 |
| T4 | 497 | 994 |
| V100 | 104 | 832 |
| V100M32 | 135 | 1,080 |
| **total** | 1,897 | 6,742 |

## helios (SenseTime HeliosData)

Job type: **no job-type field** in `cluster_log.csv` (README schema); reported as N/A.

| cluster | jobs | GPU jobs | GPU-job share | median GPU-job duration (s) | GPUs (median daily total) | span (days) |
|---|---|---|---|---|---|---|
| Earth | 872,886 | 427,148 | 48.9% | 234 | 1,120 | 191 |
| Saturn | 1,753,078 | 698,896 | 39.9% | 124 | 2,072 | 193 |
| Uranus | 490,309 | 329,117 | 67.1% | 280 | 2,112 | 190 |
| Venus | 246,708 | 125,303 | 50.8% | 204 | 968 | 194 |

### Earth

Job state (all jobs) (n=872,886):

| value | count | share |
|---|---|---|
| COMPLETED | 708,616 | 81.2% |
| FAILED | 103,297 | 11.8% |
| CANCELLED | 60,330 | 6.9% |
| NODE_FAIL | 470 | 0.1% |
| TIMEOUT | 173 | 0.0% |

Duration of GPU jobs (README: end_time - start_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 427,148 | 5,130.6 | 70.0 | 158.0 | 234.0 | 813.0 | 7,760.0 | 91,793.9 | 1,217,376.0 |

Queue time of GPU jobs (README: start_time - submit_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 427,148 | 319.5 | 0.0 | 0.0 | 0.0 | 0.0 | 119.0 | 4,086.1 | 983,125.0 |

gpu_num per job (all jobs) (n=872,886):

| value | count | share |
|---|---|---|
| 0 (CPU job) | 445,738 | 51.1% |
| 1 | 380,873 | 43.6% |
| 5-8 | 23,214 | 2.7% |
| 2-4 | 12,425 | 1.4% |
| 9-16 | 6,761 | 0.8% |
| >16 | 3,875 | 0.4% |

### Saturn

Job state (all jobs) (n=1,753,078):

| value | count | share |
|---|---|---|
| COMPLETED | 1,401,406 | 79.9% |
| CANCELLED | 210,673 | 12.0% |
| FAILED | 137,690 | 7.9% |
| NODE_FAIL | 1,925 | 0.1% |
| TIMEOUT | 1,384 | 0.1% |

Duration of GPU jobs (README: end_time - start_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 698,896 | 5,252.9 | 4.0 | 32.0 | 124.0 | 884.2 | 5,826.0 | 114,921.6 | 1,212,397.0 |

Queue time of GPU jobs (README: start_time - submit_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 698,896 | 611.6 | 0.0 | 0.0 | 0.0 | 0.0 | 65.0 | 16,585.1 | 2,834,199.0 |

gpu_num per job (all jobs) (n=1,753,078):

| value | count | share |
|---|---|---|
| 0 (CPU job) | 1,054,182 | 60.1% |
| 1 | 477,012 | 27.2% |
| 5-8 | 108,853 | 6.2% |
| 2-4 | 77,615 | 4.4% |
| >16 | 17,783 | 1.0% |
| 9-16 | 17,633 | 1.0% |

### Uranus

Job state (all jobs) (n=490,309):

| value | count | share |
|---|---|---|
| COMPLETED | 327,440 | 66.8% |
| CANCELLED | 84,840 | 17.3% |
| FAILED | 75,330 | 15.4% |
| NODE_FAIL | 1,968 | 0.4% |
| TIMEOUT | 653 | 0.1% |
| SUSPENDED | 76 | 0.0% |
| RUNNING | 2 | 0.0% |

Duration of GPU jobs (README: end_time - start_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 329,117 | 9,163.7 | 22.0 | 72.0 | 280.0 | 1,286.0 | 10,687.0 | 184,725.8 | 4,320,009.0 |

Queue time of GPU jobs (README: start_time - submit_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 329,117 | 1,949.2 | 0.0 | 0.0 | 0.0 | 0.0 | 3,043.0 | 41,714.7 | 1,889,875.0 |

gpu_num per job (all jobs) (n=490,309):

| value | count | share |
|---|---|---|
| 1 | 218,086 | 44.5% |
| 0 (CPU job) | 161,192 | 32.9% |
| 5-8 | 61,115 | 12.5% |
| 2-4 | 24,612 | 5.0% |
| 9-16 | 18,285 | 3.7% |
| >16 | 7,019 | 1.4% |

### Venus

Job state (all jobs) (n=246,708):

| value | count | share |
|---|---|---|
| COMPLETED | 170,320 | 69.0% |
| CANCELLED | 45,883 | 18.6% |
| FAILED | 30,103 | 12.2% |
| NODE_FAIL | 338 | 0.1% |
| TIMEOUT | 64 | 0.0% |

Duration of GPU jobs (README: end_time - start_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 125,303 | 13,040.6 | 18.0 | 72.0 | 204.0 | 1,285.0 | 18,200.6 | 280,455.8 | 1,964,673.0 |

Queue time of GPU jobs (README: start_time - submit_time) (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 125,303 | 1,253.3 | 0.0 | 0.0 | 0.0 | 0.0 | 23.0 | 34,615.0 | 565,003.0 |

gpu_num per job (all jobs) (n=246,708):

| value | count | share |
|---|---|---|
| 0 (CPU job) | 121,405 | 49.2% |
| 1 | 64,954 | 26.3% |
| 5-8 | 31,544 | 12.8% |
| 2-4 | 16,245 | 6.6% |
| >16 | 7,339 | 3.0% |
| 9-16 | 5,221 | 2.1% |

## alibaba2023 (cluster-trace-gpu-v2023, openb)

- `openb_pod_list_default.csv`: **8,152** pods (tasks); `openb_node_list_all_node.csv`: 1,523 nodes, 6,212 GPUs.
- Job type: no train/inference field; `qos` is the closest proxy (README: Burstable, BE, LS).

qos (n=8,152):

| value | count | share |
|---|---|---|
| LS | 4,647 | 57.0% |
| BE | 3,398 | 41.7% |
| Burstable | 100 | 1.2% |
| Guaranteed | 7 | 0.1% |

pod_phase (n=8,152):

| value | count | share |
|---|---|---|
| Running | 5,193 | 63.7% |
| Failed | 1,870 | 22.9% |
| Pending | 897 | 11.0% |
| Succeeded | 192 | 2.4% |

deletion_time - scheduled_time of Succeeded pods (s):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 192 | 14,131.0 | 103.3 | 1,119.0 | 3,846.5 | 4,131.0 | 4,643.2 | 147,942.8 | 1,332,357.0 |

num_gpu per pod (n=8,152):

| value | count | share |
|---|---|---|
| 1 | 6,989 | 85.7% |
| 0 | 1,088 | 13.3% |
| 8 | 44 | 0.5% |
| 2 | 16 | 0.2% |
| 4 | 15 | 0.2% |

GPU-sharing pods (num_gpu==1, gpu_milli<1000): 3,078 (43.6% of GPU pods).

gpu_milli of GPU-sharing pods (milli-GPU):

| n | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 3,078 | 562.6 | 258.0 | 460.0 | 470.0 | 810.0 | 810.0 | 810.0 | 810.0 |

gpu_spec (required GPU types; missing = unconstrained) (n=8,152):

| value | count | share |
|---|---|---|
| <missing> | 8,152 | 100.0% |

Node GPU model (all nodes) (n=1,523):

| value | count | share |
|---|---|---|
| G2 | 549 | 36.0% |
| T4 | 404 | 26.5% |
| <missing> | 310 | 20.4% |
| P100 | 134 | 8.8% |
| V100M16 | 55 | 3.6% |
| G3 | 39 | 2.6% |
| V100M32 | 30 | 2.0% |
| A10 | 2 | 0.1% |

