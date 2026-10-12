# P2b data-quality follow-up (runtime estimates, job_type, provenance)

Generated 2026-10-12T17:25:46+05:30 by `python scripts/p2b_quality_report.py` from `data/processed/*.parquet`.

`runtime_est_s` is the stored (second-pass) estimate: expanding geometric mean of PRIOR jobs' durations, grouped by (user_group, job_type) -> (job_type) -> global, cold-start constant 3600 s. Actual `duration_s` is an outcome, never an input.

## alibaba2020

- jobs: **739,460**
- runtime_est_s quality: MdAPE **45.3%**, Spearman **0.674**, within factor-2 of actual **60.0%** (overestimate >2x: 22.0%, underestimate <0.5x: 17.9%)
- job_type: `training` 88.4% (653,551), `batch_inference` 11.6% (85,909)
- vram_gb_src: `derived` 387,445, `synthetic` 352,015
- job_type_src: `derived` 739,460
- priority_src: `synthetic` 739,460
- deadline_src: `synthetic` 739,460
- runtime_est_src: `derived` 739,460

## alibaba2023

- jobs: **2,054**
- runtime_est_s quality: MdAPE **80.7%**, Spearman **0.100**, within factor-2 of actual **39.3%** (overestimate >2x: 29.7%, underestimate <0.5x: 31.0%)
- job_type: `training` 97.7% (2,006), `batch_inference` 2.3% (48)
- vram_gb_src: `derived` 1,161, `synthetic` 893
- job_type_src: `derived` 2,054
- priority_src: `derived` 2,054
- deadline_src: `synthetic` 2,054
- runtime_est_src: `derived` 2,054

## helios_earth

- jobs: **427,148**
- runtime_est_s quality: MdAPE **46.2%**, Spearman **0.325**, within factor-2 of actual **57.7%** (overestimate >2x: 23.9%, underestimate <0.5x: 18.4%)
- job_type: `training` 100.0% (427,148)
- vram_gb_src: `synthetic` 427,148
- job_type_src: `derived` 427,148
- priority_src: `synthetic` 427,148
- deadline_src: `synthetic` 427,148
- runtime_est_src: `derived` 427,148

## helios_saturn

- jobs: **698,896**
- runtime_est_s quality: MdAPE **92.4%**, Spearman **0.232**, within factor-2 of actual **26.2%** (overestimate >2x: 37.6%, underestimate <0.5x: 36.2%)
- job_type: `training` 100.0% (698,896)
- vram_gb_src: `synthetic` 698,896
- job_type_src: `derived` 698,896
- priority_src: `synthetic` 698,896
- deadline_src: `synthetic` 698,896
- runtime_est_src: `derived` 698,896

## helios_uranus

- jobs: **329,063**
- runtime_est_s quality: MdAPE **94.7%**, Spearman **0.306**, within factor-2 of actual **28.0%** (overestimate >2x: 38.9%, underestimate <0.5x: 33.1%)
- job_type: `training` 100.0% (329,063)
- vram_gb_src: `synthetic` 329,063
- job_type_src: `derived` 329,063
- priority_src: `synthetic` 329,063
- deadline_src: `synthetic` 329,063
- runtime_est_src: `derived` 329,063

## helios_venus

- jobs: **125,303**
- runtime_est_s quality: MdAPE **99.5%**, Spearman **0.201**, within factor-2 of actual **23.7%** (overestimate >2x: 46.4%, underestimate <0.5x: 29.9%)
- job_type: `training` 100.0% (125,303)
- vram_gb_src: `synthetic` 125,303
- job_type_src: `derived` 125,303
- priority_src: `synthetic` 125,303
- deadline_src: `synthetic` 125,303
- runtime_est_src: `derived` 125,303

