# P7 — V1 classical models (GBM + set-encoder MLP)

Generated 2026-10-13T07:29:13 by `python tasks.py train-v1`. Labels: `python tasks.py label` (P6, primary track only, tau per Amendment 3). Headline disclosures: 64% of offered work excluded by the 6 h cap; deadlines/SLOs/VRAM synthetic; batch_inference = short-small proxy. Every table is regime-tagged.

## Offline selection (VAL) and test regret

| model | val mean regret | val median | test mean regret |
|---|---|---|---|
| gbm | 0.6511 | 0.0000 | 0.2122 |
| mlp | 0.5340 | 0.0000 | 0.2768 |
| edf pick (reference) | 0.6936 | 0.0000 | - |
| sjf pick (reference) | 0.6941 | 0.0000 | - |
| fifo pick (reference) | 0.8784 | 0.0000 | - |

**V1 = v1_mlp** (lower val mean regret, Amendment 2 selection rule).

## Closed loop — regime: capped-1h (primary TEST split, overlay 25%)

| policy | J (mean over seeds) | wait p50 h | DL miss | SLO miss | util |
|---|---|---|---|---|---|
| v1_mlp | 60.7523 | 3.82 | 0.456 | 0.382 | 0.648 |
| edf | 45.9098 | 2.60 | 0.271 | 0.337 | 0.686 |
| sjf | 44.9387 | 2.14 | 0.264 | 0.319 | 0.689 |

## Closed loop — regime: stress-24h (benchmark seeds, overlay 25%)

| policy | J (mean over seeds) | wait p50 h | DL miss | SLO miss | util |
|---|---|---|---|---|---|
| v1_mlp | 1231.6848 | 36.60 | 0.743 | 0.675 | 0.754 |
| edf | 588.3537 | 9.52 | 0.557 | 0.532 | 0.766 |
| sjf | 1090.6121 | 54.08 | 0.608 | 0.596 | 0.742 |

## P7 gate (Amendment 2)

- primary TEST vs EDF: mean diff +14.8425, 95% CI [+6.9067, +25.6745] -> beat: **NO**
- primary TEST vs SJF: mean diff +15.8136, 95% CI [+6.1061, +30.6690] -> beat: **NO**
- stress vs EDF: mean diff +643.3311, 95% CI [+340.3342, +1024.7996] -> no worse: **NO**
- stress vs SJF: mean diff +141.0727, 95% CI [-485.0544, +684.1076] -> no worse: **YES**

**Gate verdict: FAIL** (primary: did NOT beat both; stress: WORSE). Per PLAN P7, if V1 does not beat the best heuristic, diagnose (labels, features, horizon, tau) before P8 and report honestly.

## Decision latency and gate usage

- median per-decision latency: **0.76 ms** (per-run median of per-decision times; CPU scoring for GBM).
- policy-gate fallbacks: 0 total across runs (invalid/low-confidence recommendations re-routed to EDF).

## Diagnosis (PLAN P7 requirement — measured, before any P8 work)

1. **WAIT bias in the labels**: argmin == WAIT in 46.9% of states; mean J of the WAIT candidate (16.657) vs mean J of RUN candidates (30.436). The V term is the driver: mean V(WAIT) = 4.508 vs V(RUN) = 9.230 — admitting a job starts its deadline clock INSIDE the horizon while deferring pushes misses past it; the pre-registered TP counts remaining WORK but not future lateness. Delaying looks cheaper than it is trajectory-wide (consistent with the P6 headroom flag: greedy horizon-argmin loses to EDF closed-loop).
2. **Model mirrors the bias**: V1 picks WAIT in 65.0% of val states; closed-loop utilization drops (0.648 vs EDF 0.686 on capped-1h) and waits/misses rise — the offline regret advantage (0.534 vs 0.694) does not transfer.
3. **Low decision freedom**: 69.5% of labeled states have <= 1 feasible candidate (P6 card); effective train signal comes from ~30% of states, and online compounding amplifies small deviations from EDF's ordering.
5. **Distribution shift on stress-24h**: trained only on capped-1h states (queues ~1-20); stress queues reach thousands — V1's J degrades far more than EDF's (1231.7 vs 588.4), as pre-registered (stress was never labeled).

**Next-step options (for the user, not executed):** (a) revise the label objective so deferring incurs the same lateness accounting as admitting (count future lateness of pending jobs at horizon, or extend the V term with an SLO/deadline-risk estimate on pending jobs) — a label-design change requiring a pre-registered amendment and re-labeling; (b) horizon ablation (n_arrivals 20 -> 60) to test myopia directly; (c) tau ablation; (d) accept V1-negative result and carry the honest finding into P8 reporting.