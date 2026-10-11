# GPU Decision Engine (GDE) — Implementation Plan for Claude Code

A learned, cost-aware scheduling decision model for GPU workloads (training + inference), CLEF-style:
a model that **scores a dynamically built set of candidate actions in one forward pass** (no text generation),
returns calibrated probabilities, and sits **above** a constraint/policy layer.
Everything is validated in simulation on **public GPU-cluster traces** before any real-cluster work.

> Read `AGENTS.md` first (operating rules). This file is the full spec. Work phase by phase.
> Never claim a result you have not measured. Every number reported must come from a command that can be re-run.

---

## 0. Ground rules for the build

1. **Phase gates.** Each phase ends with acceptance checks. Do not start the next phase until they pass. If a gate fails, fix or report; do not skip.
2. **Determinism.** Every simulator run takes a `seed`. Same seed + same trace + same policy ⇒ byte-identical event log (test this).
3. **No leakage.** A policy only ever sees information available at decision time. Actual runtimes are *labels/outcomes*, never features. Use `runtime_est_s` (a prediction) as the feature.
4. **Provenance.** Every field of the unified job schema is tagged `measured`, `derived`, or `synthetic`. Reports must state which fields were synthetic (see §2.3 — public traces lack VRAM, deadlines and SLOs).
5. **Baselines first.** The classical baseline (V1) and strong heuristics must exist and be benchmarked before any LLM backbone work. If an MLP/GBM matches the Qwen model, report that honestly.
6. **Safety.** The model never executes anything. A deterministic policy layer validates every recommendation; the API returns *recommended* and *final validated* action separately.
7. **Small commits, tests with every module, pinned dependencies, config-driven (YAML), no hard-coded paths.**
8. **Hardware check first.** Run `python tasks.py hwcheck` (runs `nvidia-smi` and the torch CUDA probe, writes `docs/env.md`). **Compute split:** the simulator, baselines and labeling (P3–P6) stay CPU-only and parallelize across cores (`multiprocessing`); MLP training (P7) and the Qwen scorer (P9+) use the GPU. All training/inference code picks its device via `gde.runtime.resolve_device()` from `configs/runtime.yaml` (`auto | cuda | cpu`) with a CPU fallback. Phase 9+ (Qwen) needs a GPU; if none, stop at Phase 8 and tell the user (Colab/cloud GPU is the fallback). Measured fit of the Qwen backbone on this machine's GPU: `python tasks.py vram-probe` → `docs/env.md`.

---

## 1. Scope

**In scope (v1):** single-cluster simulator; decision = *which pending job to admit next, or WAIT*; best-fit GPU placement by deterministic rule; later extension to joint (job, GPU) choice; classical baselines; OR-Tools reference; GBM/MLP model; Qwen-backbone candidate-scoring model; joint multi-head model; benchmark harness; FastAPI service; minimal dashboard.

**Out of scope until the user approves:** Kubernetes/Kueue integration, live migration, killing/preempting running jobs, tenant-quota overrides, reinforcement learning from scratch, production telemetry (DCGM). Document these as future work only.

**Success criteria (to be *measured*, not assumed):** lower cost per completed job and idle-GPU time; fewer deadline/SLO violations; less fragmentation; decision latency far below LLM-reasoning latency; generalization to unseen workload mixes and *unseen clusters* without per-job-label retraining.

---

## 2. Data: public traces

### 2.1 Datasets to use

| Priority | Dataset | What it is | Use in GDE | Access |
|---|---|---|---|---|
| **Primary** | **Alibaba `cluster-trace-gpu-v2020`** (PAI MLaaS) | Training + inference jobs, 6,500+ GPUs on ~1,800 machines, July–Aug 2020; GPU requested as a percentage (`plan_gpu`, 50.0 = half a GPU); `gpu_type` per task | Main training/validation source. Closest public match to a mixed train/inference workload | `github.com/alibaba/clusterdata` → subdirectory for the 2020 GPU trace (README lists download links; an alternative data repo on GitHub is mentioned). Read its README + LICENSE |
| Secondary | **Alibaba `cluster-trace-gpu-v2023`** (`fgd_gpu_trace_2023`) | ~6,200 GPUs on ~1,200 machines, heterogeneous resource specs, built for fragmentation studies | Fragmentation metric validation; heterogeneous GPU capacities | same repo, `fgd_gpu_trace_2023` folder |
| Secondary | **Helios (SenseTime)** — `S-Lab-System-Group/HeliosData` | 4 clusters (Earth, Saturn, Uranus, Venus), ~3.36M jobs (~1.58M GPU jobs), 6 months, ~343 MB | **Cross-cluster generalization test set** (train on Alibaba, test on Helios) | `git clone` of HeliosData; each cluster has `cluster_log.csv` and `cluster_gpu_number.csv` |
| Secondary | **Microsoft Philly traces** — `msr-fiddle/philly-traces` | 117,325 DNN-training jobs, Aug–Dec 2017; ~0.98 GB compressed (~6.6 GB unpacked); machine list gives GPUs and **per-GPU memory** per server | Training-only workload; only source here with real per-GPU memory in the machine list | `trace-data.tar.gz` in the repo (Git LFS); parsing notebook included |
| Optional | **AcmeTrace** — `internlm/AcmeTrace` | LLM-development jobs, Mar–Aug 2023, ~880k jobs, job traces ~109 MB on GitHub (utilization files ~80 GB on HuggingFace — **do not download**) | LLM-era job-mix shift test | GitHub job traces only |
| Stretch | Alibaba `cluster-trace-gpu-v2026` | Anonymized six-month trace, up to 155,410 GPUs / 37,707 servers | Only after everything else works; use one small time slice | check size before downloading |
| Unverified | ATLAS (from the original plan) | learning-augmented scheduling benchmark | Verify it exists and is downloadable before relying on it; otherwise drop | — |

Notes:
- Older Alibaba traces (2017/2018) require a short survey for the link; the GPU traces' README states its own access path. **Read each README at download time and follow it. Do not hard-code or guess URLs.** If a source is unreachable, log it, continue with the others, and fall back to the synthetic generator (§2.4) — never block the project on a download.
- Check each dataset's `LICENSE`/terms and write them to `data/raw/<name>/SOURCES.md` with URL, retrieval date, file sizes and SHA-256 checksums. Cite the associated papers in the README (Alibaba: "MLaaS in the Wild", NSDI'22; Helios: SC'21; Philly: ATC'19).
- Schemas below are *what we need to find*, not verified column lists. **Phase 1 must read each dataset's schema doc and record the true column names in `docs/data_dictionary.md`.**

### 2.2 What public traces do NOT contain (be explicit about it)

| Needed by GDE | In traces? | Handling |
|---|---|---|
| Submit time, requested GPUs/CPU/memory, run duration, final status | Yes (names vary) | `measured` |
| GPU type / machine specs | Mostly | `measured` |
| Per-job **VRAM requirement** | Mostly no (Alibaba gives GPU %, Philly gives per-GPU memory of machines) | `derived`: `vram_gb = gpu_fraction × gpu_mem_gb(gpu_type)` for fractional jobs; for whole-GPU jobs sample a fraction from a seeded Beta distribution (config) → tag `synthetic` |
| **Deadlines / SLOs** | No | `synthetic`: seeded slack distributions per job type (§2.3) |
| Online-inference vs training vs batch-inference label | Partly (Alibaba has inference/training workloads; others don't) | use trace label when present (`measured`); otherwise heuristic on duration + GPU fraction + dataset (`derived`), config-driven |
| Priority | Partly (queue/VC/user group) | map to 1–10 (`derived`), configurable |
| Runtime **estimate** (what a scheduler would know at submit) | No | `derived`: train a small runtime predictor on *past* jobs only (per user/group/type features) or use noisy actual (config noise level) for ablation |
| Counterfactual outcomes ("what if we had chosen B") | **Never** | generated by simulator rollouts (§5) |
| Energy | No | modeled: `P = P_idle + (P_peak − P_idle) × util` per GPU type, config constants, tagged as model assumptions |

### 2.3 Synthetic deadline/SLO rules (config-driven, seeded)
- `training`: `deadline = submit + slack × duration`, `slack ~ LogUniform(2, 10)`.
- `batch_inference`: `slack ~ LogUniform(1.5, 4)`.
- `online_inference`: SLO is a **max queue wait** `slo_s ~ {1, 2, 5, 10}` seconds (plus model-load time `load_s` configurable); violating it counts as an SLO miss. Online inference is modeled as short jobs with high priority.
- All parameters in `configs/workload_synthesis.yaml`; every report prints them.

### 2.4 Synthetic workload generator (needed anyway)
`src/ingestion/synthetic.py`: Poisson/bursty arrivals (MMPP), job-type mix, lognormal durations, known deadlines. Used for (a) simulator correctness tests with analytically known answers, (b) distribution-shift tests (burst, mix change), (c) fallback if downloads fail.

### 2.5 Unified job schema (Parquet)
```python
# src/ingestion/schema.py  (pydantic or pyarrow schema)
job_id: str                 # unique within dataset, stable
dataset: str                # alibaba2020 | alibaba2023 | helios_earth | philly | synthetic ...
submit_ts: float            # seconds from trace start
duration_s: float           # ACTUAL runtime — outcome only, never a model input
runtime_est_s: float        # prediction available at submit — model input
num_gpus: float             # may be fractional (GPU share); gpu_milli/1000 style
gpu_type_req: str | None    # None = any
vram_gb: float
cpu_cores: float
mem_gb: float
job_type: str               # training | online_inference | batch_inference
priority: int               # 1..10
deadline_ts: float          # absolute; for online_inference = submit_ts + slo_s
user_group: str | None
final_status: str           # pass | failed | killed  (outcome only)
# provenance flags, one per derived field:
vram_gb_src: str; job_type_src: str; priority_src: str; deadline_src: str; runtime_est_src: str
```

---

## 3. Repository layout

```
gpu-decision-engine/
├── AGENTS.md
├── PLAN.md
├── README.md
├── pyproject.toml / requirements.txt   # pinned versions (requirements.in = top-level, requirements.txt = lock)
├── tasks.py                            # python tasks.py setup | data | sim-smoke | baselines | label | train-v1 | bench | api | test
├── configs/
│   ├── cluster_small.yaml              # 8 GPUs
│   ├── cluster_medium.yaml             # 64 GPUs, heterogeneous
│   ├── objectives.yaml                 # weights w1..w4, price/power constants
│   ├── workload_synthesis.yaml         # §2.3
│   ├── runtime.yaml                    # device: auto | cuda | cpu (training/inference only)
│   ├── backbone.yaml                   # Qwen model id + pinned revision
│   └── experiment_*.yaml
├── data/{raw,interim,processed,splits,labels}/    # gitignored; keep SOURCES.md + checksums
├── docs/data_dictionary.md             # true column names per dataset (Phase 1 output)
├── docs/env.md                         # hardware check + measured Qwen VRAM (generated)
├── scripts/
│   ├── hw_check.py                     # writes docs/env.md
│   ├── vram_probe.py                   # Qwen frozen-forward / LoRA-step peak VRAM
│   ├── download_data.py                # reads each README, downloads, checksums
│   ├── make_splits.py
│   └── make_report.py
├── src/gde/
│   ├── ingestion/{load_alibaba2020,load_alibaba2023,load_helios,load_philly,synthetic,normalize,schema,runtime_predictor}.py
│   ├── simulator/{events,cluster,workload,engine,metrics,snapshot}.py
│   ├── decisions/{schema,candidates,constraints,policy_gate}.py
│   ├── features/state_builder.py
│   ├── baselines/{fifo,priority,sjf,edf,best_fit,easy_backfill,ortools_ref}.py
│   ├── labeling/{rollout,oracle_labels,dataset_builder}.py
│   ├── models/{gbm,mlp_setencoder,qwen_scorer,joint_heads,calibration,train}.py
│   ├── evaluation/{benchmark,stats,calibration_eval,latency}.py
│   └── api/{main,schemas}.py
├── dashboard/                          # React, last
├── tests/
├── checkpoints/ experiments/ reports/
```

---

## 4. Core contracts

### 4.1 Cluster (`configs/cluster_*.yaml`)
Per GPU: `gpu_id, type, vram_gb, compute_factor (relative speed), p_idle_w, p_peak_w, price_per_hour (config placeholder, NOT real market prices)`. Cluster: `power_budget_kw`, optional pools. Start: 8 identical GPUs; then 64 with 2–3 types.

### 4.2 Simulator engine (custom discrete-event loop, **not SimPy**)
Reason: label generation needs thousands of cheap *state snapshots + forked rollouts*; a plain `heapq` event loop with an immutable-ish state object is easy to deep-copy and fast. SimPy generators are hard to fork. Keep the SimPy option out unless there's a strong reason.

Events: `JOB_ARRIVE`, `JOB_START`, `JOB_FINISH`, `DECISION_POINT`. A **decision point** occurs after every arrival and every finish: build candidates, ask the policy, validate, apply, repeat until the policy returns `WAIT` or nothing is feasible.

Runtime model: job runtime on GPU of type g = `duration_s / compute_factor(g)` (documented assumption); fractional-GPU jobs share a GPU if VRAM and share fit; no interference model in v1 (add optional slowdown factor later, off by default).

Metrics recorded per job: `wait_s, start_ts, finish_ts, deadline_met, slo_met, gpu_hours, energy_kwh`; per cluster over time: utilization, VRAM used, **fragmentation** (§4.4), idle GPU-hours, queue length.

### 4.3 Candidates & constraints
- Candidates at a decision point: top-`K` (default 16) pending jobs after a cheap pre-rank (e.g., by priority then deadline slack) **that fit** (VRAM, GPU share, power headroom, allowed GPU type) + `WAIT`. Pending jobs beyond `K` are not candidates this round (record `n_pending`, `n_truncated` as state features).
- `constraints.py` hard-rejects: VRAM overcommit, no free share, power budget exceeded, job type not allowed on GPU, operator limits from config. Pure functions, 100% unit-tested.
- Placement for a chosen job: deterministic best-fit (least leftover VRAM that fits). v2 of the action space: candidates become (job, gpu_id) pairs.

### 4.4 Fragmentation metric (define once, document)
`F = Σ over GPUs of free_vram_gb on GPUs whose free VRAM < min_vram_of_any_pending_job` divided by total VRAM, sampled at decision points while the queue is non-empty (stranded capacity). Test on hand-built cases.

### 4.5 Decision record (what training data and the API both use)
```json
{
  "state": {"cluster": {...}, "gpus": [...], "running": [...], "pending_summary": {...}, "time_features": {...}},
  "candidates": [{"action_id": "RUN:<job_id>", "job": {...features...}, "fits": true}, {"action_id": "WAIT"}],
  "allowed_actions": ["RUN:job_a","RUN:job_b","WAIT"]
}
```
Response:
```json
{
  "model_version": "...",
  "scores": {"RUN:job_a": 0.76, "RUN:job_b": 0.19, "WAIT": 0.05},
  "recommended_action": "RUN:job_a",
  "confidence": 0.76,
  "expected": {"cost": ..., "deadline_risk": ...},        // only after V3
  "policy_gate": {"valid": true, "final_action": "RUN:job_a", "reasons": []},
  "decision_latency_ms": ...
}
```
**Architecture rule:** actions are scored *per state*, never as a fixed class list of job IDs. Output is a masked softmax over the candidates present in that state.

---

## 5. Objective, labels and counterfactuals

### 5.1 Objective (configurable, `configs/objectives.yaml`)
`J = w1·C_compute + w2·V_deadline + w3·T_completion + w4·F_frag`, each term normalized (e.g., per job or by horizon) so weights are comparable.
- `C_compute`: energy_kWh × energy_price + idle_GPU_hours × idle_penalty (cluster is fixed-size, so "cost" must capture waste and energy, not just rented time). All constants are config assumptions.
- `V_deadline`: Σ priority-weighted lateness (training/batch) + large fixed penalty per online-inference SLO miss.
- `T_completion`: priority-weighted wait + run time.
- `F_frag`: §4.4 averaged over the horizon.
Ship **three presets**: `latency_critical` (SLO penalty dominant), `cost_saver`, `balanced`.

### 5.2 Counterfactual label generation (`labeling/`)
For each sampled decision state `s`:
1. Snapshot the simulator.
2. For each candidate action `a`: fork, apply `a`, then continue with a fixed **rollout policy** (default: best heuristic from baselines, config) for horizon `H` (next `N` arrivals or `T` seconds). Add a **terminal penalty** for jobs still pending/running at horizon end (prevents "delaying looks free" artifacts).
3. Compute `J(a)`. Store: `J` for all candidates, `regret(a) = J(a) − min J`, hard label `argmin`, soft target `softmax(−J/τ)`.
4. Run states in parallel (`multiprocessing`); cache to Parquet in `data/labels/`.
Sample states from the *baseline* schedulers' trajectories first; Phase 11 adds DAgger (states visited by the learned policy, relabeled).

**OR-Tools reference (`baselines/ortools_ref.py`)**: CP-SAT over small windows (≤ ~12 jobs, few GPUs) for offline reference. This is **clairvoyant** (knows true durations), so it is an *upper-bound reference*, not an online competitor. Report whether CP-SAT proved optimality or stopped on a time limit; if not proven, call it a heuristic reference.

---

## 6. Models

### V1 — classical (must exist before anything else)
- **Features per (state, candidate):** job (type one-hot, priority, vram, num_gpus, runtime_est, deadline slack, wait so far), cluster (free GPUs, free VRAM, largest free block, utilization, power headroom, queue length/stats, running-job remaining-estimate stats), interaction (vram/free_vram_largest, slack/runtime_est, fits-on-which-types).
- **Heads:** (a) LightGBM LambdaRank or per-candidate regret regression → softmax over candidates; (b) small set-encoder MLP (DeepSets or 1–2 layer attention over candidates + context vector) trained with soft cross-entropy.
- Output: masked softmax over candidates.

### V2 — CLEF-like backbone scorer
- Backbone: a small Qwen model (e.g., Qwen2.5-0.5B or the current small Qwen3 variant — **check Hugging Face for what exists and what fits VRAM at run time; record the exact model id/revision in the config**). Current pick: `configs/backbone.yaml` (`Qwen/Qwen3-0.6B-Base`, pinned revision); measured VRAM in `docs/env.md`.
- **One forward pass per state.** Two input encodings to ablate: (i) *text-serialized* compact state with a marker token per candidate (`<cand_i>`); (ii) *numeric-embedding* variant: project features to the hidden size and feed as input embeddings (usually better, test it).
- Read hidden states at candidate-marker positions → MLP head → logit per candidate → masked softmax. No autoregressive generation.
- Training stages: (1) frozen backbone + trainable head; (2) LoRA via PEFT + head. Keep LoRA only if held-out decision quality improves enough to justify added memory/latency.
- Loss: `L = CE_soft(label-smoothed) + λ·Brier`; choose λ on validation. Then **temperature scaling** on validation for calibration.
- Report decision latency (p50/p95, batch=1) on CPU and GPU, and VRAM footprint.

### V3 — joint decision model
Shared encoder with heads per candidate: action logit, predicted `J` (regression), deadline-violation risk (BCE). Add cross-field interaction (e.g., risk head conditioned on action head). Compare against V2 with *identical* data. Joint is kept only if it beats independent scoring on regret or calibration.

### Calibration honesty
The softmax probability measures *agreement with the oracle/rollout label*, not the chance that an outcome is good. Evaluate with ECE, reliability diagram, and Brier on held-out states; say this in the report.

---

## 7. Evaluation protocol

**Schedulers compared (identical traces + seeds):** FIFO, Priority, SJF (by estimate), EDF, Best-fit, **EASY backfilling**, OR-Tools windowed reference (clairvoyant, labeled as such), V1, V2, V3. Beat strong heuristics, not just FIFO.

**Splits (no leakage):**
1. Chronological split inside Alibaba2020 (train → val → test).
2. **Cross-trace test:** train on Alibaba2020, test on Helios and/or Philly (different clusters, GPU counts, mixes).
3. **Stress tests:** load factor ρ ∈ {0.6, 0.8, 0.95, 1.1}; burst arrivals; mix shifts (more online inference); heterogeneous GPU capacities; noisy runtime estimates (σ ∈ {0, 0.25, 0.5}).
Traces come from clusters of thousands of GPUs; for simulation, **sample time windows and scale arrival rate / job sizes to a target load on the 8- and 64-GPU simulated clusters**, and record the scaling in the experiment config.

**Metrics:** GPU utilization; throughput; p50/p95 wait (and inference queue latency); deadline/SLO violation rate; cost per completed job; fragmentation; idle GPU-hours; decision latency; **regret vs rollout oracle**; Brier/ECE.
**Statistics:** ≥ 5 seeds per setting; paired comparison vs best heuristic; bootstrap 95% CIs; report wins/ties/losses, not just means. Auto-generate `reports/<exp>/report.md` with tables and plots, including a "synthetic fields used" section.

---

## 8. Policy/safety layer and API

- `policy_gate.py`: re-validates the model's top action with `constraints.py`; if invalid, or confidence below threshold, or model error/timeouts → fallback to the best heuristic (config) and record `reasons`. Never allow overcommit; never preempt/kill in v1.
- FastAPI endpoints: `POST /decision`, `POST /simulate`, `GET /metrics`, `GET /health`. Pydantic request/response models from §4.5. Response always includes both `recommended_action` and `policy_gate.final_action`. Add request-size limits and input validation tests.
- Dashboard (React, minimal, last): cluster utilization, pending/running jobs, recommendation + alternative scores, cost/deadline impact, comparison vs FIFO/priority. Read-only.

---

## 9. Phase plan with gates

| # | Phase | Deliverables | Gate (must pass) |
|---|---|---|---|
| **P0** | Scaffold | repo, pinned env (Python 3.11/3.12 venv), `tasks.py` task runner, CI-style `python tasks.py test`, hardware check printed to `docs/env.md` | `python tasks.py test` runs; env reproducible |
| **P1** | Data acquisition | `scripts/download_data.py`, `SOURCES.md` w/ checksums + licenses, `docs/data_dictionary.md` with real column names, EDA notebook/script with per-dataset summary stats (job counts, duration/GPU distributions, type shares) | ≥ 1 real dataset (Alibaba2020 preferred) loaded; unavailable ones logged, not faked |
| **P2** | Normalization | per-dataset loaders → unified Parquet (§2.5) with provenance flags; cleaning report (dropped rows + reasons); impossible-request detection; runtime predictor trained on past jobs only | schema validation passes; no negative durations; provenance flags populated; leakage test: `runtime_est_s` never computed from future jobs |
| **P3** | Simulator + FIFO | engine, cluster, metrics, snapshot/fork, synthetic generator, FIFO | **tests:** conservation (GPU-hours used ≤ capacity × time), no VRAM overcommit ever, determinism hash equal across 2 runs, hand-computed 3-job scenario matches exactly, snapshot→fork reproduces original trajectory |
| **P4** | Baselines | Priority, SJF, EDF, Best-fit, EASY backfill; benchmark harness v0 | all run end-to-end on 3 seeds; ranking sanity (e.g., EDF ≤ FIFO on deadline-miss in a deadline-heavy synthetic mix); results table generated |
| **P5** | Decision layer | candidates, constraints, policy_gate, state_builder | property-based tests (hypothesis): a gated action never violates constraints for random states; feature builder has no access to outcome fields (static check/test) |
| **P6** | Labeling | rollout engine, objective presets, parallel dataset builder, OR-Tools window reference | label determinism; soft targets sum to 1 over candidates; rollout throughput measured and logged; dataset card (n states, candidates per state, class balance, regret distribution) written |
| **P7** | V1 models | GBM + set-encoder MLP, training scripts, calibration | **Decision gate:** report regret + J vs best heuristic on val. If V1 does not beat the best heuristic, diagnose (labels, features, horizon, τ) *before* P8 and report to the user |
| **P8** | Full benchmark of V1 | closed-loop runs in simulator with learned policy; cross-trace + stress tests; report | report generated incl. CIs and synthetic-field disclosure. **Stop here and summarize if no GPU available** |
| **P9** | V2 Qwen scorer | frozen-backbone run, numeric-vs-text ablation, LoRA run, calibration, latency/VRAM table | closed-loop benchmark vs V1 + heuristics; keep LoRA only if justified |
| **P10** | V3 joint heads | multi-head model, ablations vs V2 | same data/splits as V2; report win/loss honestly |
| **P11** | DAgger loop | iterate: run learned policy → collect visited states → relabel → retrain (2–3 rounds) | regret on learned-policy state distribution decreases or report no gain |
| **P12** | API + gate | FastAPI, policy gate integration, tests, load test for latency | endpoints tested; response always carries recommended + final action; fallback path tested by injecting a bad model output |
| **P13** | Dashboard | minimal React read-only UI | renders from `/simulate` + `/decision`; visual smoke test |
| **P14** | Final report + README | reproduction commands, results, limitations, future-work (shadow mode → human-approved → limited test cluster; Kueue) | a new clone can reproduce the headline table with documented commands |

Suggested order of effort: P0–P4 first (foundation), P5–P8 (first real result), then P9–P11 (the CLEF-like contribution), P12–P14 (productization).

---

## 10. Testing strategy
- Unit: constraints, fragmentation, metrics, cost/energy math, schema validation.
- Property-based (`hypothesis`): no overcommit; gated actions always feasible; candidate masks never include infeasible actions.
- Simulator invariants: conservation, determinism, fork equivalence, event ordering.
- Golden tests: tiny hand-solved scenarios for each baseline (expected schedule written by hand).
- Leakage tests: feature columns ∩ outcome columns = ∅; time-ordered split has no overlap.
- Model tests: masked softmax sums to 1; shape tests; overfit-a-tiny-batch test; checkpoint reload gives identical outputs.
- API tests: schema validation, fallback behavior, latency smoke.

---

## 11. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Public traces lack VRAM/deadlines/SLOs | Explicit synthetic rules (§2.3), provenance flags, sensitivity runs over slack/VRAM parameters, disclose in every report |
| Labels reflect rollout-policy bias | Use several rollout policies; compare labels; DAgger; evaluate closed-loop, not just offline accuracy |
| Rollouts too slow | Short horizon, parallelism, state subsampling, cap K, cache; measure and tune in P6 |
| Learned model overfits Alibaba | Cross-trace test (Helios/Philly), stress tests, noisy-estimate ablation |
| LLM backbone adds cost without gain | V1 gate, frozen-vs-LoRA ablation, latency/VRAM table; report negative results |
| Calibration misread as outcome probability | Document meaning (agreement with oracle label); ECE/Brier on held-out states |
| Download blocked / dataset moved | Log, continue with remaining datasets + synthetic generator |
| Scope creep into K8s | Out of scope until user approves |

---

## 12. Definition of done
- [ ] `python tasks.py setup`, `python tasks.py data` and `python tasks.py test` work on a clean clone; data provenance + licenses recorded
- [ ] Simulator passes determinism/conservation/fork tests
- [ ] ≥ 6 heuristic baselines + OR-Tools reference benchmarked, with CIs
- [ ] V1, V2, V3 trained on identical splits; closed-loop results on in-distribution, cross-trace, and stress settings
- [ ] Calibration (ECE/Brier) and decision latency reported
- [ ] API returns recommended and validated final action; fallback tested
- [ ] Final report states clearly which results used synthetic fields and which claims are *not* supported
- [ ] Future-work section: shadow mode → human-approved → limited reversible changes in a test cluster (Kueue-based), no preemption/quota override by default

---

## Appendix A — starter configs

`configs/objectives.yaml`
```yaml
presets:
  balanced:         {w_cost: 1.0, w_deadline: 2.0, w_completion: 1.0, w_frag: 0.5}
  latency_critical: {w_cost: 0.3, w_deadline: 5.0, w_completion: 1.0, w_frag: 0.2}
  cost_saver:       {w_cost: 2.0, w_deadline: 1.0, w_completion: 0.5, w_frag: 1.0}
constants:                       # ASSUMPTIONS, not measured prices
  energy_price_per_kwh: 0.10
  idle_gpu_penalty_per_hour: 0.5
  slo_miss_penalty: 50.0
horizon: {n_arrivals: 20, max_seconds: 3600, terminal_penalty: true}
labeling: {tau: 1.0, top_k_candidates: 16, rollout_policy: best_heuristic}
```

`configs/cluster_small.yaml`
```yaml
gpus:
  - {count: 8, type: generic_24gb, vram_gb: 24, compute_factor: 1.0, p_idle_w: 60, p_peak_w: 300, price_per_hour: 1.0}
power_budget_kw: 3.0
```

## Appendix B — Task runner targets
Cross-platform (no GNU make): `python tasks.py <target>`, stdlib-only, runs everything inside `.venv`.
`python tasks.py setup` · `python tasks.py data` · `python tasks.py sim-smoke` · `python tasks.py baselines` · `python tasks.py label` · `python tasks.py train-v1` · `python tasks.py bench` · `python tasks.py train-v2` · `python tasks.py api` · `python tasks.py test` · `python tasks.py hwcheck` · `python tasks.py vram-probe` · `python tasks.py lock`

## Appendix C — First prompt to give Claude Code
> Read AGENTS.md and PLAN.md. Start with Phase P0, then P1. After each phase, run its gate checks, commit, and give me a short status (what passed, what numbers, what's next). Stop and ask me if a dataset download needs a manual step (survey/login) or if any gate fails. Do not proceed past P8 without confirming a GPU is available.