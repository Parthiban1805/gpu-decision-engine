# P4 — heuristic baselines on the pre-registered regimes

Generated 2026-10-12T21:01:09 by `python tasks.py baselines`. Regimes: `configs/regimes.yaml` (FROZEN, pre-registered in `reports/sim/regime_prereg.md` before any P4 result). Schedulers: fifo, priority, sjf, edf, best_fit, easy_backfill. Paired seeds (Amendment 1): [31501, 31002, 31003, 31004, 31005, 31506, 31007, 31008, 31009, 31010]. Est mode: stored.

**Headline disclosures (binding):** on capped-1h, **64% of offered work is excluded by the 6 h duration cap** (measured, `reports/sim/p3b_regime.md`); **deadlines/SLOs/VRAM are synthetic** for all trace jobs; `batch_inference` is a **short-small proxy** (derived, ≤600 s & ≤0.5 GPU, not tuned); overlay jobs are fully synthetic. Every table below carries its regime tag.

## Main comparison (rho 0.8)

### regime: capped-1h — rho 0.8, overlay 0% (mean over paired seeds)

| scheduler | util(AW) | rho hat tr | wait p50 h | wait p95 h | DL miss | SLO miss(ov) | makespan d | frag |
|---|---|---|---|---|---|---|---|---|
| edf | 0.722 | 0.722 | 0.08 | 12.10 | 0.299 | - | 3.54 | 0.047 |
| sjf | 0.716 | 0.716 | 0.15 | 11.91 | 0.307 | - | 3.54 | 0.053 |
| priority | 0.724 | 0.724 | 0.46 | 12.95 | 0.398 | - | 3.52 | 0.044 |
| best_fit | 0.725 | 0.725 | 1.75 | 16.53 | 0.351 | - | 3.54 | 0.040 |
| easy_backfill | 0.707 | 0.707 | 4.84 | 16.69 | 0.665 | - | 3.61 | 0.013 |
| fifo | 0.662 | 0.662 | 12.15 | 21.32 | 0.765 | - | 3.77 | 0.007 |

### regime: capped-1h — rho 0.8, overlay 25% (mean over paired seeds)

| scheduler | util(AW) | rho hat tr | wait p50 h | wait p95 h | DL miss | SLO miss(ov) | makespan d | frag |
|---|---|---|---|---|---|---|---|---|
| edf | 0.722 | 0.722 | 0.10 | 12.09 | 0.312 | 0.274 | 3.54 | 0.044 |
| sjf | 0.716 | 0.716 | 0.14 | 11.94 | 0.310 | 0.279 | 3.54 | 0.050 |
| priority | 0.720 | 0.719 | 0.25 | 13.90 | 0.353 | 0.289 | 3.54 | 0.042 |
| best_fit | 0.726 | 0.725 | 1.43 | 16.03 | 0.350 | 0.339 | 3.54 | 0.039 |
| easy_backfill | 0.712 | 0.712 | 4.82 | 16.58 | 0.650 | 0.658 | 3.60 | 0.011 |
| fifo | 0.663 | 0.663 | 12.05 | 21.32 | 0.765 | 0.831 | 3.77 | 0.005 |

### regime: stress-24h — rho 0.8, overlay 0% (mean over paired seeds)

| scheduler | util(AW) | rho hat tr | wait p50 h | wait p95 h | DL miss | SLO miss(ov) | makespan d | frag |
|---|---|---|---|---|---|---|---|---|
| edf | 0.766 | 0.766 | 9.08 | 201.75 | 0.558 | - | 169.89 | 0.022 |
| best_fit | 0.759 | 0.759 | 14.32 | 215.26 | 0.586 | - | 170.95 | 0.012 |
| priority | 0.769 | 0.769 | 24.39 | 251.23 | 0.637 | - | 169.42 | 0.018 |
| sjf | 0.750 | 0.750 | 43.32 | 230.95 | 0.581 | - | 172.61 | 0.018 |
| easy_backfill | 0.696 | 0.696 | 372.89 | 762.18 | 0.910 | - | 187.43 | 0.004 |
| fifo | 0.636 | 0.636 | 707.54 | 1076.08 | 0.961 | - | 206.80 | 0.001 |

### regime: stress-24h — rho 0.8, overlay 25% (mean over paired seeds)

| scheduler | util(AW) | rho hat tr | wait p50 h | wait p95 h | DL miss | SLO miss(ov) | makespan d | frag |
|---|---|---|---|---|---|---|---|---|
| edf | 0.766 | 0.766 | 9.52 | 188.75 | 0.557 | 0.532 | 169.99 | 0.020 |
| best_fit | 0.760 | 0.760 | 12.02 | 201.64 | 0.584 | 0.556 | 170.96 | 0.011 |
| priority | 0.764 | 0.764 | 15.76 | 273.17 | 0.620 | 0.586 | 170.12 | 0.016 |
| sjf | 0.742 | 0.742 | 54.08 | 255.88 | 0.608 | 0.596 | 174.33 | 0.016 |
| easy_backfill | 0.695 | 0.695 | 381.78 | 772.54 | 0.912 | 0.905 | 187.73 | 0.003 |
| fifo | 0.636 | 0.636 | 710.35 | 1079.25 | 0.961 | 0.962 | 207.02 | 0.001 |

### regime: uncapped-1h — rho 0.8, overlay 25% (optional cheap cell)

| scheduler | util(AW) | wait p50 h | wait p95 h | DL miss | SLO miss(ov) | makespan d |
|---|---|---|---|---|---|---|
| sjf | 0.630 | 0.05 | 16.98 | 0.239 | 0.198 | 7.79 |
| edf | 0.630 | 0.06 | 15.45 | 0.226 | 0.205 | 7.80 |
| best_fit | 0.634 | 0.11 | 18.57 | 0.288 | 0.246 | 7.79 |
| priority | 0.633 | 0.13 | 18.72 | 0.297 | 0.239 | 7.77 |
| easy_backfill | 0.598 | 9.04 | 43.59 | 0.618 | 0.615 | 8.56 |
| fifo | 0.571 | 20.24 | 58.19 | 0.744 | 0.765 | 8.92 |

## Clairvoyant ablation — regime: capped-1h (rho 0.8, overlay 25%)

*EASY and SJF re-run with ORACLE estimates (runtime_est = actual). Labeled **clairvoyant**: an ablation isolating estimate quality, NOT an online competitor — oracle estimates are not available at decision time. Question (Amendment 2): does EASY's gap to the reordering policies close when its reservations are exact?*

| scheduler | estimates | util(AW) | wait p50 h | wait p95 h | DL miss | SLO miss(ov) |
|---|---|---|---|---|---|---|
| easy_backfill | stored | 0.712 | 4.82 | 16.58 | 0.650 | 0.658 |
| easy_backfill | oracle (clairvoyant) | 0.733 | 0.87 | 12.31 | 0.439 | 0.362 |
| sjf | stored | 0.716 | 0.14 | 11.94 | 0.310 | 0.279 |
| sjf | oracle (clairvoyant) | 0.717 | 0.08 | 9.81 | 0.269 | 0.255 |
| edf (reference) | stored | 0.722 | 0.10 | 12.09 | 0.312 | 0.274 |

## rho sweep — regime: capped-1h (overlay 25%)

| scheduler | rho | util(AW) | wait p50 h | DL miss | SLO miss(ov) | makespan d |
|---|---|---|---|---|---|---|
| best_fit | 0.6 | 0.563 | 0.88 | 0.246 | 0.232 | 4.48 |
| easy_backfill | 0.6 | 0.561 | 2.37 | 0.440 | 0.469 | 4.50 |
| edf | 0.6 | 0.562 | 0.00 | 0.201 | 0.190 | 4.49 |
| fifo | 0.6 | 0.560 | 5.75 | 0.580 | 0.643 | 4.52 |
| priority | 0.6 | 0.563 | 0.42 | 0.265 | 0.216 | 4.47 |
| sjf | 0.6 | 0.563 | 0.01 | 0.229 | 0.208 | 4.47 |
| best_fit | 0.8 | 0.726 | 1.43 | 0.350 | 0.339 | 3.54 |
| easy_backfill | 0.8 | 0.712 | 4.82 | 0.650 | 0.658 | 3.60 |
| edf | 0.8 | 0.722 | 0.10 | 0.312 | 0.274 | 3.54 |
| fifo | 0.8 | 0.663 | 12.05 | 0.765 | 0.831 | 3.77 |
| priority | 0.8 | 0.720 | 0.25 | 0.353 | 0.289 | 3.54 |
| sjf | 0.8 | 0.716 | 0.14 | 0.310 | 0.279 | 3.54 |
| best_fit | 0.95 | 0.781 | 1.75 | 0.392 | 0.340 | 3.24 |
| easy_backfill | 0.95 | 0.738 | 6.95 | 0.752 | 0.746 | 3.42 |
| edf | 0.95 | 0.780 | 0.24 | 0.330 | 0.299 | 3.25 |
| fifo | 0.95 | 0.684 | 16.38 | 0.823 | 0.879 | 3.61 |
| priority | 0.95 | 0.797 | 0.40 | 0.406 | 0.312 | 3.22 |
| sjf | 0.95 | 0.777 | 0.18 | 0.321 | 0.274 | 3.27 |
| best_fit | 1.1 | 0.822 | 1.95 | 0.430 | 0.394 | 3.11 |
| easy_backfill | 1.1 | 0.756 | 9.13 | 0.788 | 0.795 | 3.34 |
| edf | 1.1 | 0.816 | 0.29 | 0.344 | 0.336 | 3.12 |
| fifo | 1.1 | 0.703 | 19.87 | 0.839 | 0.902 | 3.54 |
| priority | 1.1 | 0.815 | 0.64 | 0.429 | 0.351 | 3.10 |
| sjf | 1.1 | 0.801 | 0.15 | 0.327 | 0.302 | 3.16 |

## Ranking sanity (PLAN P4 gate)

- Deadline-heavy synthetic mix ({'training': 0.2, 'batch_inference': 0.8}, 800 jobs): EDF deadline miss **0.766** vs FIFO **0.970** -> EDF <= FIFO: **PASS**

## Cap sensitivity — regime family capped-1h (3 h / 6 h / 12 h)

*Pre-registered scope: report ONLY whether the scheduler ranking is stable. The 6 h cap is frozen and is NOT re-picked. Rollout/harness config is identical at every cap: All caps run the identical six-scheduler harness config of this file on the regimes.yaml benchmark seeds (Amendment 1 list), overlay 0.25, FIFO-compatible engine; no per-cap policy or horizon changes.*

| cap | ranking by trace wait p50 | ranking by DL miss | stable vs 6 h (both) |
|---|---|---|---|
| 3 h | sjf > edf > best_fit > priority > easy_backfill > fifo | sjf > edf > best_fit > priority > easy_backfill > fifo | NO |
| 6 h | edf > sjf > priority > best_fit > easy_backfill > fifo | sjf > edf > best_fit > priority > easy_backfill > fifo | YES |
| 12 h | sjf > edf > priority > best_fit > easy_backfill > fifo | edf > sjf > best_fit > priority > easy_backfill > fifo | NO |

## Warm-up exclusion check — regime: capped-1h (rho 0.8, overlay 25%)

Metrics exclude jobs submitted in the first 10% / 25% / 50% of the arrival window (initial backlog / warm-up).

| scheduler | wait p50 h (all) | p50 h (excl 10%) | p50 h (excl 25%) | p50 h (excl 50%) | DL miss (all) | DL (10%) | DL (25%) | DL (50%) |
|---|---|---|---|---|---|---|---|---|
| fifo | 12.05 | 12.91 | 14.63 | 13.55 | 0.765 | 0.814 | 0.833 | 0.811 |
| priority | 0.25 | 0.26 | 0.30 | 0.39 | 0.353 | 0.361 | 0.340 | 0.281 |
| sjf | 0.14 | 0.17 | 0.15 | 0.20 | 0.310 | 0.318 | 0.289 | 0.271 |
| edf | 0.10 | 0.11 | 0.10 | 0.21 | 0.312 | 0.324 | 0.310 | 0.300 |
| best_fit | 1.43 | 1.45 | 1.46 | 1.44 | 0.350 | 0.354 | 0.335 | 0.294 |
| easy_backfill | 4.82 | 5.72 | 6.70 | 5.17 | 0.650 | 0.689 | 0.695 | 0.627 |

- Ranking by wait p50 at 10% exclusion unchanged vs no exclusion: **YES** (edf, sjf, priority, best_fit, easy_backfill, fifo)
- Ranking by wait p50 at 25% exclusion unchanged vs no exclusion: **YES** (edf, sjf, priority, best_fit, easy_backfill, fifo)
- Ranking by wait p50 at 50% exclusion unchanged vs no exclusion: **NO** (sjf, edf, priority, best_fit, easy_backfill, fifo)

## Horizon clarification (P3b claim correction)

The P3b sentence "no multi-day job straddles the 3600 s horizon" meant **multi-day RUNNING jobs** (excluded by the 6 h cap) — not that rollouts end with an idle cluster. Measured below: fraction of mid-run rollouts (capped-1h, FIFO, seed 31501, sampled every 5th decision point) that still have running jobs when the horizon binds:

| horizon | binds by | rollouts sampled | frac ending with running jobs |
|---|---|---|---|
| primary_prereg (n_arr=20, max_s=86400) | arrivals | 241 | 0.871 |
| seconds_3600 (n_arr=1000000, max_s=3600) | seconds | 241 | 0.983 |

So 6 h-capped jobs routinely straddle horizon end; what the cap removes is multi-day jobs. Labeling (P6) uses the primary horizon (n_arrivals=20 binds) with terminal penalties, per `configs/regimes.yaml`.

## Regime provenance and job-type mix

- capped-1h example cell (seed 31501): window_start 5872914, n_in_window 523, cap drops 12, infeasible drops 38, n_jobs 485, W 1.80M GPU-s, T_sim 4.33 d, scale 0.0096.
- Job-type mix per regime over the benchmark seeds (batch_inference = **short-small proxy**, derived; online_inference = synthetic overlay):

| regime | trace jobs | training | batch_inference (proxy) | overlay @25% |
|---|---|---|---|---|
| capped-1h | 4852 | 3888 (80.1%) | 964 (19.9%) | ~1212 |
| stress-24h | 113041 | 98566 (87.2%) | 14475 (12.8%) | ~28259 |

The 6 h cap shifts the mix toward short jobs (proxy share rises from 12.8% uncapped to 19.9% capped) — a disclosed consequence of the pre-registered filter, not a re-pick.

- All trace-derived fields carry P2 provenance (VRAM derived/synthetic; job_type derived; priority/deadline synthetic; runtime_est derived, MdAPE 45.3%). Overlay fields fully synthetic.
- Any 'scheduler X wins' statement below must hold on stress-24h or is scoped to capped-1h. Stress track is held out from P6 labeling.

## Split and window eligibility (Amendment 2E)

- Arrival-free trace prefix: **5 days** (7.3% of the trace span). Any 1 h window sampled there is degenerate; the prefix lies entirely inside the TRAIN band [0, 0.6), so val/test bands are unaffected.
- Split-seed capped-1h window eligibility (Amendment-1 rule applies at P6 sampling, replacements logged):

| split | seeds | degenerate (0 feasible) | min feasible jobs |
|---|---|---|---|
| train | 30 | 4 | 0 |
| val | 10 | 1 | 0 |
| test | 10 | 1 | 0 |

- Replacement seeds vs their originals (window band = position of window start as a fraction of trace span):

| replacement | original band | replacement band | feasible jobs |
|---|---|---|---|
| 31001->31501 | train | test | 485 |
| 31006->31506 | train | test | 361 |

Benchmark seeds are NOT split-assigned (splits use their own seed lists), so the band move train→test of the two replacements does not contaminate any split; it is disclosed for completeness. (benchmark seeds are not split-assigned; splits use their own seed lists)

## Cost

- 500 cells (480 main-matrix + 20 clairvoyant) + 180 sensitivity cells; 8,754,190 events total; total wall 5.1 min (workers=7, CPU-only).
