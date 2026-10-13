# P6 dataset card — labeled decision states, PRIMARY (capped-1h) only

Generated 2026-10-13T10:42:28 by `python tasks.py label`. Regimes/config: `configs/regimes.yaml` (Amendments 1-4), `configs/experiment_p6.yaml`. Rollout policy: edf (frozen). Horizon: n_arrivals=20, max_seconds=86400 (n_arrivals binds). Terminal penalty: true remaining work (Amendment 2) PLUS projected lateness of deferred/running jobs (Amendment 4, estimates only). Deadlines/SLOs/VRAM synthetic; 64% of offered work excluded by the 6 h cap; batch_inference = short-small proxy. Stress track held out. Sampling: multi-candidate states only (n_feasible >= 2, Amendment 4).

- States: **2503** (train 1507, val 517, test 479); candidates/state mean **10.9** (min 1, max 20); tau = **0.0641** (0.5 x median positive TRAIN regret, Amendment 3).
- Throughput (real mix, before budget exercised): **433.2 rollouts/s** (2.3 ms/rollout, mean 11 rollouts/state => 216.58 states/s single-core). Budget (9/window/scheduler, cap 2700) chosen from this measurement, before labels were produced.
- Total labeling wall: 11.2 min (workers=7).

## 1. Hard-label composition (argmin == policy pick)

| argmin == | share |
|---|---|
| EDF pick | 29.5% |
| SJF pick | 12.3% |
| FIFO pick | 1.2% |
| WAIT pick | 0.6% |
| other candidate (not EDF/SJF/FIFO/WAIT) | 56.3% |

## 3. Multi-candidate-only sampling (Amendment 4)

- Single-feasible decision points (n_feasible <= 1) are NOT labeled (excluded from training loss and offline metric): train 63.4%, val 64.7%, test 63.9% of decision points with a non-empty queue were skipped.
- Labeled states are all n_feasible >= 2; budget 9/window/scheduler, cap 2700 (Amendment 4).

## 4. Soft-target entropy and J normalization

- soft-target entropy: mean 1.501, p50 1.461, max 2.995 (max possible ln(n_candidates) = 2.996).
- tau rule (Amendment 3, fixed pre-model): 0.5 x median positive TRAIN regret = 0.0641.
- J normalization (objective.py docstring, Amendment 3): mean |C|=0.106, |V|=17.521, |T|=18.015, |F|=0.028. V and T dominate J under the balanced preset in this regime (deadlines are synthetic and tight at rho 0.8); disclosed — the preset weights are unchanged (frozen).

## 5. Terminal penalty share and estimate-TP sensitivity

- mean TP (priority-hours) by candidate: WAIT 119.90, RUN 147.61.
- estimate-based TP (pre-registered 10% subsample, n=150): hard-label agreement **96.7%**, mean J rank correlation 0.989. WAIT-specific: true-TP argmin == WAIT in 2 states, est-TP argmin == WAIT in 2, both 2 (P(est WAIT | true WAIT) = 1.0, P(true WAIT | est WAIT) = 1.0). Agreement >= 80%: single label set carried into P7.


## 6. Candidate-count distribution

- mean 10.9, p50 10, p95 19, max 20; states exceeding K=16+1 (baseline picks beyond top-K): 19.6%.

## 7. Sampling stratification and splits

| split | best_fit | easy_backfill | edf | fifo | priority | sjf | total |
|---|---|---|---|---|---|---|---|
| test | 858 | 1126 | 800 | 1265 | 751 | 827 | 5627 |
| train | 2437 | 3188 | 2081 | 3750 | 2329 | 2199 | 15984 |
| val | 849 | 1016 | 817 | 1352 | 843 | 767 | 5644 |

Strata: time-in-window tercile x queue-length tercile per (window, scheduler); all states from one window stay in that window's split (splits are seed-assigned in regimes.yaml).

## 8. Degenerate split-seed replacements (Amendment-1 rule)

| split | original | replacement |
|---|---|---|
| train | 31103 | 31603 |
| train | 31105 | 31605 |
| train | 31111 | 31611 |
| train | 31127 | 31627 |
| val | 31207 | 31707 |
| test | 31304 | 31804 |

## 2. Amendment 4 precondition gate (REVISED labels, validation windows)

| seed | J EDF | J argmin | diff (EDF - argmin) |
|---|---|---|---|
| 31201 | 32.187 | 24.174 | +8.014 |
| 31202 | 15.957 | 17.246 | -1.289 |
| 31203 | 16.773 | 15.780 | +0.993 |
| 31204 | 36.059 | 23.957 | +12.103 |
| 31205 | 16.248 | 9.249 | +6.999 |
| 31206 | 9.527 | 8.809 | +0.718 |
| 31707 | 39.876 | 46.824 | -6.948 |
| 31208 | 17.455 | 15.124 | +2.332 |
| 31209 | 17.240 | 10.558 | +6.683 |
| 31210 | 11.999 | 11.265 | +0.734 |

- paired mean (J_EDF - J_argmin) **+3.0337**, bootstrap 95% CI [-0.0272, +6.3050]. Pass rule: CI excludes 0 on the favorable side OR mean >= 0.
- **GATE VERDICT: PASS — training allowed**
- Note: mean positive but CI includes 0 — passed on the mean rule; reported CI stands.

### Ablations (report only, 5 validation seeds)

| ablation | J EDF | J argmin | util EDF | util argmin |
|---|---|---|---|---|
| horizon n_arrivals=60 | 23.445 | 17.180 | 0.730 | 0.712 |
| work-conserving argmin | 23.445 | 20.237 | 0.730 | 0.718 |

- horizon-60 label composition on 30 kept val states: WAIT share 0.0% (n=20) -> 0.0% (n=60); mean candidates 10.6 -> 10.6.

## OR-Tools CP-SAT window reference (CLAIRVOYANT)

- 12 windows x 10 jobs on 4 GPUs (cumulative relaxation over GPU-share and VRAM; fractional shares exact, per-GPU packing relaxed — disclosed). Proven optimal: **12/12** within 10 s.

| window | status | proven | CP obj | EDF obj (same relaxed model) |
|---|---|---|---|---|
| 0 | OPTIMAL | True | 86440.0 | 110424 |
| 1 | OPTIMAL | True | 79650.0 | 101700 |
| 2 | OPTIMAL | True | 11585.0 | 13442 |
| 3 | OPTIMAL | True | 81881.0 | 83247 |
| 4 | OPTIMAL | True | 224985.0 | 243453 |
| 5 | OPTIMAL | True | 165775.0 | 1124663 |
| 6 | OPTIMAL | True | 34772.0 | 38411 |
| 7 | OPTIMAL | True | 92550.0 | 243070 |
| 8 | OPTIMAL | True | 100665.0 | 258490 |
| 9 | OPTIMAL | True | 93740.0 | 95212 |
| 10 | OPTIMAL | True | 63010.0 | 115081 |
| 11 | OPTIMAL | True | 71560.0 | 89124 |