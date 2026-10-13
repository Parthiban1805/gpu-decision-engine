# Pre-registered benchmark regimes (P3b decision) — FROZEN

Written 2026-10-12, **before any P4 scheduler result exists**, per the P3b
decision. Machine-readable definition: `configs/regimes.yaml` (this file and
that config are committed together; the commit is the freeze point). Basis:
`reports/sim/p3b_regime.md` (regime diagnostic, measured) and the P3 smoke
report `reports/sim/p3_smoke.md`.

## What is frozen

- PRIMARY **capped-1h**: alibaba2020, `cluster_small.yaml` (8×24 GB), **1 h
  trace windows**, **6 h duration cap** applied as a *filter inside the window
  before time-scaling* (drops logged, kept jobs byte-identical), ρ target 0.8.
- STRESS **stress-24h**: the P3 smoke regime — same dataset/cluster, **24 h
  window, uncapped** (whales retained), ρ 0.8, run in **every** phase.
- The cap value, the window lengths and stress ρ must not change after any
  scheduler result is seen. Adding schedulers/metrics/analyses is allowed;
  re-picking these constants post hoc is not.

## Horizons (PLAN.md Appendix A change)

The PLAN Appendix A base horizon `{n_arrivals: 20, max_seconds: 3600}` becomes
**per-regime** (recorded in `configs/regimes.yaml`):

- primary capped-1h: `{n_arrivals: 20, max_seconds: 86400}` — the 20-arrival
  bound binds; max_seconds is raised so it cannot bind first (capped-1h
  inter-arrival ≈ 928 s ⇒ 20 arrivals ≈ 5.2 h < 24 h).
- stress-24h: `{n_arrivals: 20, max_seconds: 3600}` — the PLAN base value is
  kept.

## Load sweep

ρ ∈ {0.6, 0.8, 0.95, 1.1} runs on the **primary regime only**; the stress
track stays at ρ = 0.8.

## Splits (defined now, used from P6/P7)

Chronological split at the **window level**: a window belongs to train/val/test
by its start position as a fraction of the trace span — train [0.0, 0.6),
val [0.6, 0.8), test [0.8, 1.0) — with no window overlap across splits.
Split seed lists are disjoint from each other and from the benchmark seeds.
Cross-trace test on Helios: apply the same 6 h cap and report it; also report
uncapped.

## Seeds

Benchmark (paired across schedulers, ≥10): 31001–31010 — disjoint from the P3b
diagnostic seeds 30001/30002. Splits: train 31101–31130, val 31201–31210,
test 31301–31310. Full lists in `configs/regimes.yaml`.

**Amendment 1 (2026-10-12, still before any scheduler result):** seeds 31001
and 31006 drew 1 h windows in the trace's first ~5 days, where arrivals are
~0/h (0 capped-feasible jobs — degenerate cells, measured before any
scheduler was run). Rule: a benchmark seed whose capped-1h window yields 0
feasible jobs is replaced by seed+500, repeatedly, until non-empty.
31001→31501 (485 capped-feasible jobs), 31006→31506 (361). Effective benchmark
seeds: 31501, 31002–31005, 31506, 31007–31010. Cap, window and ρ unchanged.

## P7 gate (defined now)

V1 must **(a) beat the best heuristic on the primary TEST split** — mean
objective J (balanced preset, PLAN §5.1; lower is better) with the paired
bootstrap 95% CI of (J_V1 − J_best_heuristic) excluding zero on the favorable
side — **and (b) be no worse than the best heuristic on the stress track**
(CI includes zero or favors V1). Both numbers reported.

## Reporting rules (binding from P4 onward)

1. Every table and plot carries a regime tag (capped-1h / stress-24h /
   uncapped-1h).
2. Headlines must state: **64% of offered work is excluded by the 6 h cap**
   (measured on the smoke window) and **deadlines/SLOs/VRAM are synthetic**.
3. Report the job-type mix per regime, with `batch_inference` labeled as a
   **short-small proxy** (derived, ≤600 s & ≤0.5 GPU heuristic, not tuned).
4. Any "scheduler X wins" claim must hold on stress-24h or be explicitly
   scoped to capped-1h.
5. stress-24h is **held out for labeling/DAgger from P6 onward** (P6 samples
   states from the primary track only).

## Scope of P4 analyses under this pre-registration

- Cap sensitivity at 3 h / 12 h (alongside the frozen 6 h): report ONLY
  whether the scheduler ranking is stable; the cap is not re-picked.
- Warm-up check on capped-1h: metrics with vs without an initial warm-up
  exclusion; report whether rankings change.
- Optional cheap cell: uncapped-1h (P3b option B), reported under its own tag.

## Amendment 2 (2026-10-12, after P4 acceptance, BEFORE any P5/P6 work)

- **Best heuristic for the P7 gate:** EDF and SJF tied in P4. The primary-side
  gate is **beat BOTH** (each pairwise, paired bootstrap 95% CI on mean J,
  balanced preset); the stress-side gate is no worse than min-J(EDF, SJF),
  with the EDF/SJF tie broken by balanced-preset J on the VAL split once P6
  computes J. All pairwise comparisons reported.
- **P6 rollout policy: `edf`**, fixed in `configs/regimes.yaml`, identical
  across all caps and regimes; never re-picked per cap.
- **P6 terminal penalty, form frozen:** TP = Σ_pending priority × (load +
  duration)/cf (full remaining work, never started) + Σ_running priority ×
  max(0, (load + duration)/cf − (H_end − start)). It **uses true remaining
  time** — labels are outcomes computed inside the rollout; features never see
  durations. Rationale: estimate-based penalties would bias labels toward the
  estimate distribution (MdAPE 45%). **Label sensitivity check planned:** in
  P6, recompute labels on a 10% subsample with the estimate-based penalty and
  report hard-label agreement + regret rank correlation; if agreement < 80%,
  both label sets are carried through P7.
- **Split seeds:** the Amendment-1 replacement rule (seed+500 until the
  capped-1h window is non-empty) applies to split seeds when first sampled in
  P6, with replacements logged.
- **Clairvoyant ablation (added analysis, not a competitor):** EASY and SJF
  re-run with oracle estimates and labeled clairvoyant; reported against the
  stored-estimate results to isolate the estimate-quality effect on EASY's
  reservations.

## Amendment 4 (2026-10-13, after the P7 gate failure and accepted diagnosis, BEFORE relabeling)

This is the **one and only allowed label revision**:

1. **Terminal penalty revision.** For jobs pending or running at horizon end,
   add **projected lateness = max(0, horizon_end + remaining_est −
   deadline_ts)** — priority-weighted, summed, normalized like V (÷ n_active),
   added to the V term. `remaining_est` uses **estimates only** (pending:
   load + runtime_est; running: max(0, that − elapsed)). The work-based TP
   (Amendment 2, true remaining) is unchanged. Rationale: the P7 diagnosis
   showed V(WAIT) 4.5 ≪ V(RUN) 9.2 — admitting starts the deadline clock
   inside the horizon while deferring pushes misses past it; deferral must be
   charged its projected lateness.
2. **Precondition gate.** The greedy-argmin policy from the revised labels
   must beat EDF on closed-loop trajectory J over the **validation windows**
   (paired seeds): pass = paired bootstrap 95% CI of (J_EDF − J_argmin)
   excludes 0 on the favorable side, **or** mean ≥ 0 (CI reported either
   way). **If it fails: stop and report; no model is trained.**
3. **Ablations, report only:** (a) horizon n_arrivals=60 (label composition +
   argmin-vs-EDF closed loop on 5 val seeds); (b) work-conserving mode
   (argmin over RUN candidates only, never WAIT while something fits; closed
   loop vs EDF on 5 val seeds).
4. **One-revision rule.** No further objective changes after seeing revised
   results. **Final headline numbers use fresh windows** — seeds 34001–34004,
   34505, 34006, 34507, 34008–34010 (34005/34007 were degenerate and took the
   Amendment-1 +500 replacement, verified before this commit; never used
   anywhere before).
5. **Sampling.** Multi-candidate states only (n_feasible ≥ 2): single-
   feasible states are excluded from the training loss and the offline metric
   (not labeled; share reported). Budget raised to ~**2,700** states (~1,620
   train / ~540 val / ~540 test) — enabled by the measured 453.8 rollouts/s.
   τ recomputed per the Amendment-3 rule on the revised train labels.

Outcome of this revision (relabel → precondition gate → retrain only on
pass) is reported with all numbers, including negatives.
