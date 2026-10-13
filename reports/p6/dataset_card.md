# P6 dataset card — labeled decision states, PRIMARY (capped-1h) only

Generated 2026-10-13T07:20:21 by `python tasks.py label`. Regimes/config: `configs/regimes.yaml` (Amendments 1-3), `configs/experiment_p6.yaml`. Rollout policy: edf (frozen). Horizon: n_arrivals=20, max_seconds=86400 (n_arrivals binds). Terminal penalty: true remaining (pre-registered). Deadlines/SLOs/VRAM synthetic; 64% of offered work excluded by the 6 h cap; batch_inference = short-small proxy. Stress track held out.

- States: **1200** (train 720, val 240, test 240); candidates/state mean **4.8** (min 1, max 19); tau = **0.1611** (0.5 x median positive TRAIN regret, Amendment 3).
- Throughput (real mix, before budget exercised): **461.4 rollouts/s** (2.2 ms/rollout, mean 5 rollouts/state => 230.70 states/s single-core). Budget (4/window/scheduler, cap 1440) chosen from this measurement, before labels were produced.
- Total labeling wall: 13.4 min (workers=7).

## 1. Hard-label composition (argmin == policy pick)

| argmin == | share |
|---|---|
| EDF pick | 30.5% |
| SJF pick | 3.3% |
| FIFO pick | 1.0% |
| WAIT pick | 46.9% |
| other candidate (not EDF/SJF/FIFO/WAIT) | 18.2% |

## 3. Single-feasible-candidate states

- n_feasible <= 1: **834/1200 (69.5%)** — flagged in the Parquet (column n_feasible); kept in the dataset, excluded from nothing by default (P7 reports with/without as ablation).

## 4. Soft-target entropy and J normalization

- soft-target entropy: mean 0.666, p50 0.033, max 2.944 (max possible ln(n_candidates) = 2.944).
- tau rule (Amendment 3, fixed pre-model): 0.5 x median positive TRAIN regret = 0.1611.
- J normalization (objective.py docstring, Amendment 3): mean |C|=0.117, |V|=8.242, |T|=10.937, |F|=0.031. V and T dominate J under the balanced preset in this regime (deadlines are synthetic and tight at rho 0.8); disclosed — the preset weights are unchanged (frozen).

## 5. Terminal penalty share and estimate-TP sensitivity

- mean TP (priority-hours) by candidate: WAIT 145.48, RUN 205.67.
- estimate-based TP (pre-registered 10% subsample, n=72): hard-label agreement **98.6%**, mean J rank correlation 0.978. WAIT-specific: true-TP argmin == WAIT in 34 states, est-TP argmin == WAIT in 34, both 34 (P(est WAIT | true WAIT) = 1.0, P(true WAIT | est WAIT) = 1.0). Agreement >= 80%: single label set carried into P7.


## 6. Candidate-count distribution

- mean 4.8, p50 2, p95 18, max 19; states exceeding K=16+1 (baseline picks beyond top-K): 8.9%.

## 7. Sampling stratification and splits

| split | best_fit | easy_backfill | edf | fifo | priority | sjf | total |
|---|---|---|---|---|---|---|---|
| test | 140 | 300 | 100 | 502 | 89 | 113 | 1244 |
| train | 371 | 761 | 299 | 1304 | 359 | 304 | 3398 |
| val | 65 | 290 | 92 | 514 | 72 | 62 | 1095 |

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

## 2. Oracle headroom (argmin-at-every-state vs EDF, closed loop)

| seed | J EDF | J argmin | diff |
|---|---|---|---|
| 31501 | 24.771 | 33.259 | -8.487 |
| 31002 | 14.388 | 31.164 | -16.777 |
| 31003 | 14.306 | 13.778 | +0.528 |
| 31004 | 61.226 | 89.168 | -27.942 |
| 31005 | 15.046 | 13.346 | +1.700 |
| 31506 | 25.847 | 20.781 | +5.066 |
| 31007 | 13.863 | 12.109 | +1.754 |
| 31008 | 12.642 | 11.936 | +0.707 |
| 31009 | 7.653 | 7.894 | -0.241 |
| 31010 | 14.087 | 11.957 | +2.130 |

- paired mean improvement (J_EDF - J_argmin) **-4.1562**, bootstrap 95% CI [-10.8356, +1.3173] over 10 seeds.
- **FLAGGED BEFORE P7 (per pre-registration): the greedy horizon-argmin oracle does NOT beat EDF on trajectory J** (mean favors EDF; CI spans zero). Interpretation: optimizing the 20-arrival horizon J greedily at every state is myopic w.r.t. trajectory J, and 69.5% of states have <=1 feasible candidate, so most states carry no decision freedom. The learnable signal is per-state rollout regret; expectations that V1 beats EDF's closed-loop trajectory J should be tempered accordingly — this is exactly what the P7 gate will test.
- argmin policy wall: 77 s/seed (labels computed live at every decision point).

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