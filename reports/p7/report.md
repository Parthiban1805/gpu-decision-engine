# P7 — V1 classical models (GBM + set-encoder MLP)

Generated 2026-10-13T10:51:07 by `python tasks.py train-v1`. Labels: `python tasks.py label` (P6, primary track only, tau per Amendment 3). Headline disclosures: 64% of offered work excluded by the 6 h cap; deadlines/SLOs/VRAM synthetic; batch_inference = short-small proxy. Every table is regime-tagged.

## Offline selection (VAL) and test regret

| model | val mean regret | val median | test mean regret |
|---|---|---|---|
| gbm | 0.9578 | 0.0187 | 0.6431 |
| mlp | 1.1261 | 0.0474 | 0.7219 |
| edf pick (reference) | 1.1895 | 0.0246 | - |
| sjf pick (reference) | 0.9775 | 0.0157 | - |
| fifo pick (reference) | 2.0050 | 0.3060 | - |

**V1 = v1_gbm** (lower val mean regret, Amendment 2 selection rule).

## Closed loop — regime: capped-1h (primary TEST split, overlay 25%)

| policy | J (mean over seeds) | wait p50 h | DL miss | SLO miss | util |
|---|---|---|---|---|---|
| v1_gbm | 69.6003 | 24.98 | 0.333 | 0.331 | 0.392 |
| edf | 45.9098 | 2.60 | 0.271 | 0.337 | 0.686 |
| sjf | 44.9387 | 2.14 | 0.264 | 0.319 | 0.689 |

## Closed loop — regime: stress-24h (benchmark seeds, overlay 25%)

| policy | J (mean over seeds) | wait p50 h | DL miss | SLO miss | util |
|---|---|---|---|---|---|
| v1_gbm | 19245.4932 | 59.65 | 0.048 | 0.038 | 0.052 |
| edf | 588.3537 | 9.52 | 0.557 | 0.532 | 0.766 |
| sjf | 1090.6121 | 54.08 | 0.608 | 0.596 | 0.742 |

## Closed loop — regime: capped-1h, VALIDATION windows (Amendment 4)

| policy | J (mean) | wait p50 h | DL miss | SLO miss | util |
|---|---|---|---|---|---|
| v1_gbm | 48.4436 | 0.13 | 0.177 | 0.127 | 0.385 |
| edf | 21.3323 | 0.15 | 0.307 | 0.305 | 0.708 |
| sjf | 18.7178 | 0.08 | 0.263 | 0.306 | 0.703 |

- val vs EDF: mean diff +27.1114, 95% CI [-0.8532, +78.5952]
- val vs SJF: mean diff +29.7258, 95% CI [-0.5862, +84.2726]

## Closed loop — regime: capped-1h, FRESH HEADLINE windows (Amendment 4 seeds, never used anywhere before)

| policy | J (mean) | wait p50 h | DL miss | SLO miss | util |
|---|---|---|---|---|---|
| v1_gbm | 22.5766 | 0.20 | 0.245 | 0.181 | 0.378 |
| edf | 20.2302 | 0.07 | 0.206 | 0.323 | 0.689 |
| sjf | 15.5852 | 0.04 | 0.208 | 0.278 | 0.682 |

- fresh vs EDF: mean diff +2.3464, 95% CI [-0.7811, +5.9004]
- fresh vs SJF: mean diff +6.9914, 95% CI [+1.2206, +13.0385]

## P7 gate (Amendment 2)

- primary TEST vs EDF: mean diff +23.6905, 95% CI [+0.2419, +47.6810] -> beat: **NO**
- primary TEST vs SJF: mean diff +24.6616, 95% CI [-0.1887, +51.9691] -> beat: **NO**
- stress vs EDF: mean diff +18657.1395, 95% CI [+14923.2443, +22244.4094] -> no worse: **NO**
- stress vs SJF: mean diff +18154.8811, 95% CI [+14576.5050, +21726.5549] -> no worse: **NO**

**Gate verdict: FAIL** (primary: did NOT beat both; stress: WORSE). Per PLAN P7, if V1 does not beat the best heuristic, diagnose (labels, features, horizon, tau) before P8 and report honestly.

## Decision latency and gate usage

- median per-decision latency: **0.93 ms** (per-run median of per-decision times; CPU scoring for GBM).
- policy-gate fallbacks: 0 total across runs (invalid/low-confidence recommendations re-routed to EDF).

## Diagnosis (PLAN P7 requirement — measured; post-Amendment-4)

1. **The label WAIT-bias is fixed; the policy WAIT-bias is not.** argmin == WAIT in only 0.6% of states now (was 46.9% pre-revision; mean J WAIT 46.319 vs RUN 53.870). Yet V1 picks WAIT in 39.7% of VAL states: per-state, WAIT is CHEAP in the counterfactual because the EDF rollout re-admits the job at the very next decision point, so its measured regret stays small. A policy that generalizes to 'WAIT often' never re-admits — each WAIT leaves the cluster idle until the next external event.
2. **Closed-loop compounding:** V1 utilization 0.385 vs EDF 0.694 on capped-1h; wait p50 explodes. The offline regret advantage (0.958 vs EDF pick 1.189) does not transfer: the forced-action-then-EDF counterfactual systematically understates the cost of a WAIT-ing policy — the covariate-shift problem PLAN §5.2 assigns to DAgger (P11).
3. **The signal is real but needs the work-conserving constraint** (pre-registered ablation, P6 card): the work-conserving argmin beats EDF closed-loop (J 20.24 vs 23.45 on 5 val seeds; util 0.718 vs 0.730). Removing WAIT from the model's action space at the POLICY layer (labels/objective untouched) is the obvious next experiment — reported as an option, not executed (would need its own pre-registered gate run).
4. **Stress-24h collapse (J 19245, util 0.052):** trained only on capped-1h multi-candidate states; under stress the model WAITs at scale and the cluster idles — distribution shift, as pre-registered (stress never labeled).

**Options for the user (not executed):** (a) work-conserving ModelPolicy (policy layer only; pre-registered gate run); (b) DAgger (P11) relabeling on the model's own state distribution; (c) accept the V1 negative result — per-state counterfactual labels with an EDF rollout undervalue admission for closed-loop control in this regime.