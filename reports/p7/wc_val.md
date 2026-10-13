# Amendment 5 val: Stage 1 — model + work-conserving rule on VALIDATION windows (Amendment 5 pass rule)

Generated 2026-10-13T12:11:24 by `python tasks.py train-v1 --stage val`. Policy: **model + work-conserving rule** (WAIT masked out whenever any candidate fits; policy layer only, labels/objective/checkpoints unchanged per Amendment 5). Disclosures: 64% of offered work excluded by the 6 h cap; deadlines/SLOs/VRAM synthetic; batch_inference = short-small proxy. Regime tag: capped-1h.

| policy | J (mean) | wait p50 h | DL miss | SLO miss | util |
|---|---|---|---|---|---|
| v1_wc | 22.8855 | 0.16 | 0.300 | 0.308 | 0.712 |
| edf | 21.3323 | 0.15 | 0.307 | 0.305 | 0.708 |
| sjf | 18.7178 | 0.08 | 0.263 | 0.306 | 0.703 |

## Pass rule (Amendment 5): beat EDF AND SJF, paired 95% CI

- vs EDF: mean diff +1.5532, 95% CI [-1.0996, +4.6673] -> **NOT beaten**
- vs SJF: mean diff +4.1676, 95% CI [-0.1367, +8.8916] -> **NOT beaten**
- **STAGE VERDICT: FAIL**

## Amendment 5 reporting items

- model WAIT rate BEFORE masking: mean 75.2% of decision points (raw argmax over the unmasked candidate set)
- utilization: V1-WC 0.712 vs EDF 0.708 vs SJF 0.703
- overlay SLO miss: V1-WC 0.308 vs EDF 0.305 vs SJF 0.306