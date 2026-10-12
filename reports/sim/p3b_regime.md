# P3b regime diagnostic (alibaba2020, report-only)

## 1. Heavy-tail profile of feasible jobs (8x24 GB, 24 h windows)

| window | seed | n feasible | dropped | p50 s | p90 s | p99 s | max s | W top 0.1% | W top 1% | W top 10% | W total GPU-s | T_sim d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke (30001) | 30001 | 7630 | 636 | 714 | 13145 | 60508 | 604950 | 0.193 | 0.463 | 0.856 | 72.5M | 131.0 |
| alt (30002) | 30002 | 7550 | 625 | 716 | 18450 | 62229 | 604950 | 0.174 | 0.388 | 0.828 | 83.1M | 150.2 |

## 2. Window length x cluster (rho target 0.8, FIFO, no overlay)

| cluster | window | n feasible | W GPU-s | T_sim (d) | realized rho | util (AW) | makespan (d) | run wall s |
|---|---|---|---|---|---|---|---|---|
| small 8x24 | 1 h | 308 | 2.72M | 4.92 | 0.687 | 0.687 | 5.9 | 0 |
| small 8x24 | 6 h | 2075 | 21.67M | 39.19 | 0.617 | 0.617 | 51.3 | 2 |
| small 8x24 | 24 h | 7630 | 72.46M | 131.05 | 0.613 | 0.613 | 165.4 | 8 |
| medium 64 hetero | 1 h | 388 | 20.29M | 4.59 | 0.560 | 0.560 | 14.5 | 0 |
| medium 64 hetero | 6 h | 2297 | 66.51M | 15.03 | 0.367 | 0.367 | 38.6 | 7 |
| medium 64 hetero | 24 h | 8252 | 204.22M | 46.16 | 0.138 | 0.138 | 11.0 | 14 |

## 3. Candidate benchmark regimes (measured, FIFO, rho target 0.8)

| regime | n | cap drops | T_sim (d) | realized rho | makespan (d) | wait p50 | wait p95 | inter-arrival |
|---|---|---|---|---|---|---|---|---|
| small, 1 h + 6 h cap | 284 | 27 | 3.05 | 0.653 | 3.68 | 963 min | 26.3 h | 928 s |
| small, 1 h, no cap | 308 | 0 | 4.92 | 0.687 | 5.88 | 1111 min | 28.4 h | 1382 s |
| small, 24 h + 6 h cap | 7181 | 514 | 46.92 | 0.704 | 53.36 | 10485 min | 314.2 h | 565 s |
| small, 24 h, no cap (P3 smoke) | 7630 | 0 | 131.05 | 0.613 | 165.43 | 30608 min | 999.2 h | 1484 s |

## 4. Duration-cap sensitivity (cluster_small, 24 h window, seed 30001)

A cap is a FILTER (tagged, drops logged), not a modification: kept jobs are byte-identical. Overlay column = load share a 25% overlay (0.3 share x ~20 s = 6 GPU-s/job) would carry on the capped regime.

| cap | n kept | dropped | W GPU-s | T_sim (d) | mean GPU-s/job | overlay25 load share |
|---|---|---|---|---|---|---|
| 6 h | 7181 | 514 | 25.9M | 46.92 | 3613 | 0.00033 |
| 24 h | 7590 | 54 | 48.7M | 88.11 | 6419 | 0.00019 |
| 277778 h | 7630 | 0 | 72.5M | 131.05 | 9497 | 0.00013 |

## 5. Options (trade-offs)

**A. Duration-cap filter (6 h, tagged filter, drops logged, applied inside the fixed window before scaling).** Measured on the smoke window: drops 514/7695 in-window jobs (~p94 of durations) and 64% of W; T_sim 131 -> 47 d. Combined with 1 h windows: n=284, T_sim 3.05 d, realized rho 0.65, makespan 3.68 d (queue nearly drains inside the window), inter-arrival ~15.5 min. Strict FIFO still waits p50 16 h / p95 26 h there — that is the blocking floor P4 policies must beat, and at hour scale it is discriminative (uncapped: p50 21 d / p95 41 d). P6: the 3600 s rollout horizon sees ~4 arrivals and no multi-day job straddles the horizon. Overlay load share stays ~2e-4 in every regime (structural: mean trace job ~6k GPU-s vs ~6 GPU-s per overlay job) — the overlay's role is SLO/queue-position pressure, which priority-aware policies (P4) discriminate on. Cost: multi-day training jobs vanish from the primary regime, so it no longer represents the trace's GPU-hours; fragmentation stress from big jobs shrinks (1-6 h multi-GPU jobs remain); must be paired with an uncapped stress track so headline claims are also tested where whales dominate.

**B. Short windows (1 h), no cap (existing knobs only).** n=308, T_sim 4.92 d, realized rho 0.69, waits p50 18.5 h / p95 28 h. No new semantics. Cost: whales still stretch individual windows erratically (the medium cluster flipped realized rho 0.56 -> 0.14 between window lengths because the last feasible arrival can sit hours before the window end, which the stretch maps to sim-days of idle); per-seed job counts are small (need >= 10 seeds; runs are 0-8 s so that is cheap); P6 horizon is myopic in seconds terms (~2 arrivals per 3600 s) unless the n_arrivals=20 bound is what binds (config knob exists).

**C. cluster_medium (64 heterogeneous GPUs).** 8x capacity absorbs the tail (24 h window: T_sim 46 d vs 131 d; 2.8x more work feasible incl. up-to-64-GPU jobs). Keeps realism and exercises best-fit across 3 GPU types. Cost: does not fix the tail (top 1% of jobs still carry ~40% of W); realized rho drifts far and erratically from target (0.14-0.56: min-compute-factor placement plus arrival patchiness); P6 states are 8x bigger (fork still ms-scale, measured). Better as the P8 scale/heterogeneity axis than as the labeling regime.

## 6. Recommendation

Two-track benchmark, both reported from P4 onward:

1. **Primary (labeling + model comparison): cluster_small, 1 h windows, 6 h duration cap (tagged filter, in-window drops logged), rho target 0.8, >= 10 seeds.** Measured: 284 jobs/seed, T_sim 3.05 d, realized rho 0.65, makespan ~= window, run < 1 s, P6 horizons see several arrivals with bounded terminal penalties, and metric scales (hours) where scheduler differences are visible.
2. **Stress track (honesty): the P3 smoke regime (24 h window, uncapped) kept in every benchmark phase**, plus cluster_medium at P8 for scale/heterogeneity with its realized-rho drift measured and disclosed.

Rationale: the cap is a filter (kept jobs byte-identical, drops logged; 6 h chosen at ~p94 of durations for scale, not searched for performance); the uncapped track preserves the heavy-tail blocking that EASY backfill must beat; overlay load share is ~1e-4 structurally in all regimes, so overlay effects are judged on SLO rates, reported separately from trace load as required.

