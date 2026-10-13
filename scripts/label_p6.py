"""P6: build the labeled decision dataset on the PRIMARY track (capped-1h),
write the dataset card (all pre-registered items), measure rollout throughput,
oracle headroom (argmin-vs-EDF, paired bootstrap CI), the estimate-TP label
sensitivity check, and the OR-Tools clairvoyant window reference.

Re-run: python tasks.py label
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from gde.baselines import ALL_POLICIES
from gde.baselines.ortools_ref import edf_window_objective, solve_window
from gde.evaluation.benchmark import load_regimes, make_overlay_cfg, regime_spec
from gde.labeling.dataset_builder import (
    apply_tau,
    label_one,
    replace_degenerate_seeds,
    sample_states,
    tau_from_labels,
    to_parquet,
)
from gde.labeling.objective import load_preset, trajectory_objective
from gde.labeling.rollout import label_state
from gde.paths import CONFIGS, DATA, REPORTS
from gde.simulator.cluster import ClusterSpec
from gde.simulator.engine import Engine, EventKind
from gde.simulator.snapshot import snapshot
from gde.simulator.workload import TraceConfig, generate_overlay, load_trace_window


def _jobs_for_seed(regimes: dict, overlay_raw: dict, seed: int, rho: float = 0.8):
    spec = regime_spec(regimes, "capped-1h")
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    jobs, tman = load_trace_window(
        DATA / "processed" / f"{spec['dataset']}.parquet", cluster,
        TraceConfig(window_s=3600.0, rho=rho, max_jobs=10**9,
                    duration_cap_s=float(spec["duration_cap_s"])), seed=seed)
    overlay, _ = generate_overlay(
        len(jobs), tman["t_sim_s"], make_overlay_cfg(overlay_raw, 0.25), seed=seed)
    return jobs + overlay, cluster, tman["t_sim_s"]


# ---------------------------------------------------------------------------
# throughput measurement (before fixing the budget)
# ---------------------------------------------------------------------------

def measure_throughput(regimes: dict, exp: dict, overlay_raw: dict) -> dict:
    const = load_preset("balanced")
    horizon = regimes["primary"]["horizon"]
    seed = regimes["splits"]["split_seeds"]["train"][0]
    jobs, cluster, t_sim = _jobs_for_seed(regimes, overlay_raw, seed)
    engine = Engine(jobs, cluster, ALL_POLICIES["edf"](), arrival_horizon=t_sim)
    states = []
    while len(engine.state.queue) and len(states) < 10:
        ev = engine.state.queue.peek()
        if ev.kind == EventKind.DECISION_POINT and engine.state.pending:
            states.append(snapshot(engine.state))
        engine.step()
    t0 = time.perf_counter()
    n_rollouts = 0
    for s in states:
        labels = label_state(s, const, tau=1.0,
                             n_arrivals=horizon["n_arrivals"],
                             max_seconds=horizon["max_seconds"],
                             k=exp["k_candidates"])
        n_rollouts += len(labels.candidates)
    wall = time.perf_counter() - t0
    return {"n_states": len(states), "n_rollouts": n_rollouts, "wall_s": wall,
            "states_per_s": len(states) / wall, "rollouts_per_s": n_rollouts / wall,
            "mean_rollout_ms": 1000 * wall / max(n_rollouts, 1)}


# ---------------------------------------------------------------------------
# oracle headroom (item 2): argmin-at-every-state vs EDF, closed loop
# ---------------------------------------------------------------------------

class _ArgminOracle:
    """Policy that labels the live state at every DP and takes the argmin.
    Clairvoyant-ish (uses true durations INSIDE rollouts) — reference only."""

    name = "argmin_oracle"

    def __init__(self, const, horizon, k):
        self.const, self.horizon, self.k = const, horizon, k
        self.engine = None

    def bind(self, engine: Engine) -> None:
        self.engine = engine

    def decide(self, view):
        labels = label_state(self.engine.state, self.const, tau=1.0,
                             n_arrivals=self.horizon["n_arrivals"],
                             max_seconds=self.horizon["max_seconds"], k=self.k)
        job = None
        for action_id, jid in zip(labels.candidates, labels.job_ids):
            if action_id == labels.hard_label:
                job = jid
                break
        return job


class _ArgminWorkConserving:
    """Amendment 4 ablation: argmin over RUN candidates only — never WAIT
    while any fitting candidate exists."""

    name = "argmin_work_conserving"

    def __init__(self, const, horizon, k):
        self.const, self.horizon, self.k = const, horizon, k
        self.engine = None

    def bind(self, engine: Engine) -> None:
        self.engine = engine

    def decide(self, view):
        labels = label_state(self.engine.state, self.const, tau=1.0,
                             n_arrivals=self.horizon["n_arrivals"],
                             max_seconds=self.horizon["max_seconds"], k=self.k)
        best_job, best_j = None, float("inf")
        for action_id, jid, j in zip(labels.candidates, labels.job_ids,
                                     labels.j_values):
            if action_id != "WAIT" and j < best_j:
                best_job, best_j = jid, j
        return best_job  # WAIT only when nothing fits


def _argmin_seed(args: tuple) -> dict:
    """Closed-loop J for EDF vs a live-argmin policy on one seed."""
    seed, regimes, overlay_raw, horizon, k, mode = args
    const = load_preset("balanced")
    jobs, cluster, t_sim = _jobs_for_seed(regimes, overlay_raw, seed)
    edf = Engine(jobs, cluster, ALL_POLICIES["edf"](), arrival_horizon=t_sim).run()
    policy_cls = _ArgminOracle if mode == "argmin" else _ArgminWorkConserving
    oracle = policy_cls(const, horizon, k)
    eng = Engine(jobs, cluster, oracle, arrival_horizon=t_sim)
    oracle.bind(eng)
    argmin = eng.run()
    out = {
        "seed": seed, "mode": mode,
        "J_edf": trajectory_objective(edf.state, const, t_sim)["J"],
        "J_argmin": trajectory_objective(argmin.state, const, t_sim)["J"],
        "wait_p50_edf": edf.summary["groups"]["trace"]["wait_s_p50"],
        "wait_p50_argmin": argmin.summary["groups"]["trace"]["wait_s_p50"],
        "util_edf": edf.summary["cluster"]["utilization_arrival_window"],
        "util_argmin": argmin.summary["cluster"]["utilization_arrival_window"],
        "wall_argmin_s": argmin.wall_s,
    }
    if mode == "argmin":
        # label composition along the argmin trajectory is not directly
        # observable; the horizon-60 composition ablation covers it instead
        pass
    return out


def paired_bootstrap_ci(diffs: np.ndarray, n: int = 2000, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(0)
    boots = [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
             for _ in range(n)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(np.mean(diffs)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(diffs))}


def _fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return "-"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


# ---------------------------------------------------------------------------
# OR-Tools windows
# ---------------------------------------------------------------------------

def ortools_reference(regimes: dict, exp: dict, overlay_raw: dict) -> list[dict]:
    spec = regime_spec(regimes, "capped-1h")
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    cfg = exp["ortools"]
    rng = np.random.default_rng(cfg["seed"])
    out = []
    for w in range(cfg["n_windows"]):
        seed = regimes["benchmark_seeds"][w % len(regimes["benchmark_seeds"])]
        jobs, _, _ = _jobs_for_seed(regimes, overlay_raw, seed)
        idx = rng.choice(len(jobs), size=cfg["window_jobs"], replace=False)
        win = [{"job_id": f"j{i}", "arrive": float(rng.uniform(0, 600)),
                "dur": float(jobs[i].duration_s), "share": jobs[i].share_per_gpu,
                "vram": jobs[i].per_gpu_vram, "priority": float(jobs[i].priority)}
               for i in idx]
        cp_obj, proven, status = solve_window(
            win, cfg["window_gpus"], cfg["window_gpus"] * 24.0,
            time_limit_s=cfg["time_limit_s"])
        edf_obj = edf_window_objective(win, cfg["window_gpus"],
                                       cfg["window_gpus"] * 24.0)
        out.append({"window": w, "status": status, "proven_optimal": proven,
                    "cp_obj": cp_obj, "edf_relaxed_obj": edf_obj})
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIGS / "experiment_p6.yaml"))
    parser.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 1))
    parser.add_argument("--skip-headroom", action="store_true")
    args = parser.parse_args()
    exp = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    regimes = load_regimes()
    overlay_raw = yaml.safe_load((CONFIGS / "sim_p3.yaml").read_text(encoding="utf-8"))["overlay"]
    horizon = regimes["primary"]["horizon"]
    const = load_preset("balanced")

    t_start = time.perf_counter()

    # 0) throughput BEFORE budget is exercised (budget pre-fixed in config;
    #    measurement validates it — see card)
    thr = measure_throughput(regimes, exp, overlay_raw)
    print(f"throughput: {thr['states_per_s']:.2f} states/s, "
          f"{thr['rollouts_per_s']:.1f} rollouts/s, "
          f"mean rollout {thr['mean_rollout_ms']:.1f} ms", flush=True)

    # 1) sample + label states per split
    records: dict[str, list[dict]] = {}
    replacement_log: list[dict] = []
    val_keep_states = []
    spec = regime_spec(regimes, "capped-1h")
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    parquet_path = DATA / "processed" / f"{spec['dataset']}.parquet"
    for split in ("train", "val", "test"):
        _, log = replace_degenerate_seeds(
            regimes["splits"]["split_seeds"][split], cluster, 3600.0,
            float(spec["duration_cap_s"]), parquet_path)
        replacement_log += [{**r, "split": split} for r in log]

    all_meta: dict[str, dict] = {}
    mc_stats: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in ("train", "val", "test"):
            sampled, stats = sample_states(regimes, exp, split)
            mc_stats[split] = stats
            if split == "train":
                # pre-registered sensitivity subsample (deterministic draw)
                rng_sub = np.random.default_rng(4242)
                sub_ids = set(rng_sub.choice(
                    len(sampled),
                    size=max(1, int(exp["sensitivity"]["subsample_frac"] * len(sampled))),
                    replace=False))
            else:
                sub_ids = set()
            payloads = [({"state_id": s.state_id, "split": s.split, "seed": s.seed,
                          "scheduler": s.scheduler, "t_frac": s.t_frac, "qlen": s.qlen,
                          "edf_pick": s.edf_pick, "sjf_pick": s.sjf_pick,
                          "fifo_pick": s.fifo_pick, "state": s.state},
                         horizon, const.__dict__, exp["k_candidates"], i in sub_ids)
                        for i, s in enumerate(sampled)]
            print(f"labeling {split}: {len(payloads)} states "
                  f"({len(sub_ids)} with est-TP sensitivity)", flush=True)
            records[split] = list(pool.map(label_one, payloads))
            if split == "val":
                val_keep_states = [s.state for s in sampled[:30]]
            for s in sampled:
                all_meta[s.state_id] = vars(s) | {"state": None}

    # 2) tau from TRAIN only (Amendment 3), then soft targets everywhere
    tau = tau_from_labels(records["train"])
    all_records = records["train"] + records["val"] + records["test"]
    for r in all_records:
        apply_tau([r], tau)  # in-place per record
    df = to_parquet(all_records, tau, Path(exp["report"]["parquet"]))

    # 3) sensitivity subsample: estimate-based TP agreement (pre-registered)
    sub = [r for r in records["train"] if "j_est_values" in r]
    agree = sum(1 for r in sub if r["hard_label"] == r["hard_est_label"])
    wait_true = [r for r in sub if r["hard_label"] == "WAIT"]
    wait_est = [r for r in sub if r["hard_est_label"] == "WAIT"]
    both_wait = [r for r in sub
                 if r["hard_label"] == "WAIT" and r["hard_est_label"] == "WAIT"]

    def _spearman(a: list[float], b: list[float]) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    rank_corrs = [c for c in (_spearman(r["j_values"], r["j_est_values"])
                              for r in sub) if np.isfinite(c)]
    sensitivity = {
        "n_sub": len(sub),
        "hard_agreement": agree / len(sub) if sub else None,
        "wait_true_n": len(wait_true), "wait_est_n": len(wait_est),
        "both_wait_n": len(both_wait),
        "p_est_wait_given_true_wait": (len(both_wait) / len(wait_true)) if wait_true else None,
        "p_true_wait_given_est_wait": (len(both_wait) / len(wait_est)) if wait_est else None,
        "mean_j_rank_correlation": float(np.mean(rank_corrs)) if rank_corrs else None,
    }

    # 4) OR-Tools windows
    ort = ortools_reference(regimes, exp, overlay_raw)

    # 5) Amendment 4 precondition gate: argmin (REVISED labels) vs EDF on the
    #    VALIDATION windows, closed loop, paired
    spec = regime_spec(regimes, "capped-1h")
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    val_seeds, val_repl = replace_degenerate_seeds(
        regimes["splits"]["split_seeds"]["val"], cluster, 3600.0,
        float(spec["duration_cap_s"]),
        DATA / "processed" / f"{spec['dataset']}.parquet")
    gate_payloads = [(seed, regimes, overlay_raw, horizon, exp["k_candidates"], "argmin")
                     for seed in val_seeds]
    print(f"precondition gate: {len(gate_payloads)} closed-loop pairs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gate_rows = list(pool.map(_argmin_seed, gate_payloads))
    gate_diffs = np.array([g["J_edf"] - g["J_argmin"] for g in gate_rows])
    gate_ci = paired_bootstrap_ci(
        gate_diffs, n=exp["headroom"]["bootstrap"]["n"],
        alpha=exp["headroom"]["bootstrap"]["alpha"])
    gate_pass = bool(gate_ci["ci_lo"] > 0 or gate_ci["mean"] >= 0)

    # 6) ablations (report only, 5 val seeds)
    abl_seeds = val_seeds[: exp["precondition"]["ablation_seeds"]]
    h60 = dict(horizon)
    h60["n_arrivals"] = 60
    abl_payloads = ([(s, regimes, overlay_raw, h60, exp["k_candidates"], "argmin")
                     for s in abl_seeds]
                    + [(s, regimes, overlay_raw, horizon, exp["k_candidates"], "wc")
                       for s in abl_seeds])
    print(f"ablations: {len(abl_payloads)} closed-loop runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        abl_rows = list(pool.map(_argmin_seed, abl_payloads))
    h60_rows = [r for r in abl_rows if r["mode"] == "argmin"]
    wc_rows = [r for r in abl_rows if r["mode"] == "wc"]

    # horizon-60 label composition on kept val states (report only)
    comp60 = {"n_states": 0, "wait_share_20": None, "wait_share_60": None,
              "mean_cands_20": None, "mean_cands_60": None}
    if val_keep_states:
        w20 = w60 = 0
        c20 = c60 = 0
        for st in val_keep_states:
            l20 = label_state(st, const, tau=1.0, n_arrivals=horizon["n_arrivals"],
                              max_seconds=horizon["max_seconds"], k=exp["k_candidates"])
            l60 = label_state(st, const, tau=1.0, n_arrivals=h60["n_arrivals"],
                              max_seconds=h60["max_seconds"], k=exp["k_candidates"])
            w20 += l20.hard_label == "WAIT"
            w60 += l60.hard_label == "WAIT"
            c20 += len(l20.candidates)
            c60 += len(l60.candidates)
        n = len(val_keep_states)
        comp60 = {"n_states": n, "wait_share_20": w20 / n, "wait_share_60": w60 / n,
                  "mean_cands_20": c20 / n, "mean_cands_60": c60 / n}

    wall_total = time.perf_counter() - t_start

    # ---- dataset card -------------------------------------------------------
    lines: list[str] = []
    lines.append("# P6 dataset card — labeled decision states, PRIMARY (capped-1h) only")
    lines.append("")
    lines.append(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%S')} by `python tasks.py label`. "
                 "Regimes/config: `configs/regimes.yaml` (Amendments 1-4), "
                 "`configs/experiment_p6.yaml`. Rollout policy: edf (frozen). "
                 f"Horizon: n_arrivals={horizon['n_arrivals']}, "
                 f"max_seconds={horizon['max_seconds']} (n_arrivals binds). "
                 "Terminal penalty: true remaining work (Amendment 2) PLUS projected "
                 "lateness of deferred/running jobs (Amendment 4, estimates only). "
                 "Deadlines/SLOs/VRAM synthetic; 64% of offered work excluded by "
                 "the 6 h cap; batch_inference = short-small proxy. "
                 "Stress track held out. Sampling: multi-candidate states only "
                 "(n_feasible >= 2, Amendment 4).")
    n_states = len(all_records)
    train_df = df[df["split"] == "train"]
    lines.append("")
    lines.append(f"- States: **{n_states}** (train {len(records['train'])}, "
                 f"val {len(records['val'])}, test {len(records['test'])}); "
                 f"candidates/state mean **{df.groupby('state_id').size().mean():.1f}** "
                 f"(min {df.groupby('state_id').size().min()}, max "
                 f"{df.groupby('state_id').size().max()}); tau = **{tau:.4f}** "
                 "(0.5 x median positive TRAIN regret, Amendment 3).")
    lines.append(f"- Throughput (real mix, before budget exercised): "
                 f"**{thr['rollouts_per_s']:.1f} rollouts/s** "
                 f"({thr['mean_rollout_ms']:.1f} ms/rollout, mean "
                 f"{df.groupby('state_id').size().mean():.0f} rollouts/state "
                 f"=> {thr['states_per_s']:.2f} states/s single-core). Budget "
                 f"({exp['state_budget']['states_per_window_scheduler']}/window/scheduler, "
                 f"cap {exp['state_budget']['max_states_total']}) chosen from this "
                 "measurement, before labels were produced.")
    lines.append(f"- Total labeling wall: {wall_total / 60:.1f} min "
                 f"(workers={args.workers}).")

    # item 1: argmin shares
    hard_vs = {"edf": 0, "sjf": 0, "fifo": 0, "wait": 0}
    for r in all_records:
        if r["hard_label"] == "WAIT":
            hard_vs["wait"] += 1
        elif r["hard_label"] == f"RUN:{r['edf_pick']}":
            hard_vs["edf"] += 1
        elif r["hard_label"] == f"RUN:{r['sjf_pick']}":
            hard_vs["sjf"] += 1
        elif r["hard_label"] == f"RUN:{r['fifo_pick']}":
            hard_vs["fifo"] += 1
    lines.append("")
    lines.append("## 1. Hard-label composition (argmin == policy pick)")
    lines.append("")
    lines.append("| argmin == | share |")
    lines.append("|---|---|")
    for k, v in hard_vs.items():
        lines.append(f"| {k.upper()} pick | {100 * v / n_states:.1f}% |")
    other = n_states - sum(hard_vs.values())
    lines.append(f"| other candidate (not EDF/SJF/FIFO/WAIT) | {100 * other / n_states:.1f}% |")

    # item 3: multi-candidate-only sampling (Amendment 4)
    single_share = {
        split: (1.0 - s["multi_candidate"] / s["dp_with_pending"])
        if s["dp_with_pending"] else None
        for split, s in mc_stats.items()}
    lines.append("")
    lines.append("## 3. Multi-candidate-only sampling (Amendment 4)")
    lines.append("")
    lines.append(f"- Single-feasible decision points (n_feasible <= 1) are NOT labeled "
                 f"(excluded from training loss and offline metric): "
                 + ", ".join(f"{split} {100 * v:.1f}%" if v is not None else f"{split} n/a"
                             for split, v in single_share.items())
                 + " of decision points with a non-empty queue were skipped.")
    lines.append(f"- Labeled states are all n_feasible >= 2; budget "
                 f"{exp['state_budget']['states_per_window_scheduler']}/window/scheduler, "
                 f"cap {exp['state_budget']['max_states_total']} (Amendment 4).")

    # item 4: entropy + normalization
    ents = [r["entropy"] for r in all_records]
    lines.append("")
    lines.append("## 4. Soft-target entropy and J normalization")
    lines.append("")
    lines.append(f"- soft-target entropy: mean {np.mean(ents):.3f}, p50 "
                 f"{np.percentile(ents, 50):.3f}, max {np.max(ents):.3f} "
                 f"(max possible ln(n_candidates) = {np.log(df.groupby('state_id').size().max()):.3f}).")
    lines.append(f"- tau rule (Amendment 3, fixed pre-model): 0.5 x median positive "
                 f"TRAIN regret = {tau:.4f}.")
    lines.append("- J normalization (objective.py docstring, Amendment 3): mean "
                 f"|C|={np.abs(df['C']).mean():.3f}, |V|={df['V'].mean():.3f}, "
                 f"|T|={df['T'].mean():.3f}, |F|={df['F'].mean():.3f}. V and T "
                 "dominate J under the balanced preset in this regime (deadlines are "
                 "synthetic and tight at rho 0.8); disclosed — the preset weights are "
                 "unchanged (frozen).")

    # item 5: TP share by candidate type + WAIT agreement
    tp_share_wait = df[df["is_wait"] == 1]["tp_priority_hours"].mean() if (df["is_wait"] == 1).any() else 0
    lines.append("")
    lines.append("## 5. Terminal penalty share and estimate-TP sensitivity")
    lines.append("")
    lines.append(f"- mean TP (priority-hours) by candidate: WAIT {tp_share_wait:.2f}, "
                 f"RUN {df[df['is_wait'] == 0]['tp_priority_hours'].mean():.2f}.")
    if sensitivity["n_sub"]:
        lines.append(f"- estimate-based TP (pre-registered 10% subsample, n="
                     f"{sensitivity['n_sub']}): hard-label agreement "
                     f"**{100 * sensitivity['hard_agreement']:.1f}%**, mean J rank "
                     f"correlation {sensitivity['mean_j_rank_correlation']:.3f}. "
                     f"WAIT-specific: true-TP argmin == WAIT in "
                     f"{sensitivity['wait_true_n']} states, est-TP argmin == WAIT in "
                     f"{sensitivity['wait_est_n']}, both {sensitivity['both_wait_n']} "
                     f"(P(est WAIT | true WAIT) = "
                     f"{sensitivity['p_est_wait_given_true_wait']}, P(true WAIT | est WAIT) = "
                     f"{sensitivity['p_true_wait_given_est_wait']}). "
                     + ("Agreement >= 80%: single label set carried into P7."
                        if (sensitivity["hard_agreement"] or 0) >= 0.8
                        else "**Agreement < 80%: both label sets carried into P7.**"))
    lines.append("")

    # item 6: candidate-count distribution
    counts = df.groupby("state_id").size()
    lines.append("")
    lines.append("## 6. Candidate-count distribution")
    lines.append("")
    lines.append(f"- mean {counts.mean():.1f}, p50 {counts.median():.0f}, p95 "
                 f"{counts.quantile(0.95):.0f}, max {counts.max()}; "
                 f"states exceeding K=16+1 (baseline picks beyond top-K): "
                 f"{100 * (counts > 17).mean():.1f}%.")

    # item 7: stratification + splits
    lines.append("")
    lines.append("## 7. Sampling stratification and splits")
    lines.append("")
    strat = df.groupby(["split", "scheduler"]).size().unstack(fill_value=0)
    lines.append("| split | " + " | ".join(str(c) for c in strat.columns) + " | total |")
    lines.append("|---|" + "---|" * (len(strat.columns) + 1))
    for split, row in strat.iterrows():
        lines.append(f"| {split} | " + " | ".join(str(int(v)) for v in row)
                     + f" | {int(row.sum())} |")
    lines.append("")
    lines.append("Strata: time-in-window tercile x queue-length tercile per "
                 "(window, scheduler); all states from one window stay in that "
                 "window's split (splits are seed-assigned in regimes.yaml).")

    # item 8: replacements
    lines.append("")
    lines.append("## 8. Degenerate split-seed replacements (Amendment-1 rule)")
    lines.append("")
    if replacement_log:
        lines.append("| split | original | replacement |")
        lines.append("|---|---|---|")
        for r in replacement_log:
            lines.append(f"| {r['split']} | {r['original']} | {r['replacement']} |")
    else:
        lines.append("none this run (rule: seed+500 until non-empty; earlier "
                     "measurement found train 4/30, val 1/10, test 1/10 degenerate)")

    # Amendment 4 precondition gate + ablations (replaces the P6 headroom item)
    lines.append("")
    lines.append("## 2. Amendment 4 precondition gate (REVISED labels, validation windows)")
    lines.append("")
    lines.append("| seed | J EDF | J argmin | diff (EDF - argmin) |")
    lines.append("|---|---|---|---|")
    for g in gate_rows:
        lines.append(f"| {g['seed']} | {g['J_edf']:.3f} | {g['J_argmin']:.3f} "
                     f"| {g['J_edf'] - g['J_argmin']:+.3f} |")
    lines.append("")
    lines.append(f"- paired mean (J_EDF - J_argmin) **{gate_ci['mean']:+.4f}**, "
                 f"bootstrap 95% CI [{gate_ci['ci_lo']:+.4f}, {gate_ci['ci_hi']:+.4f}]. "
                 f"Pass rule: CI excludes 0 on the favorable side OR mean >= 0.")
    lines.append(f"- **GATE VERDICT: {'PASS — training allowed' if gate_pass else 'FAIL — stop, do not train'}**")
    if gate_pass and gate_ci["mean"] > 0 and gate_ci["ci_lo"] <= 0:
        lines.append("- Note: mean positive but CI includes 0 — passed on the mean "
                     "rule; reported CI stands.")
    lines.append("")
    lines.append("### Ablations (report only, 5 validation seeds)")
    lines.append("")
    lines.append("| ablation | J EDF | J argmin | util EDF | util argmin |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| horizon n_arrivals=60 | {_fmt(np.mean([r['J_edf'] for r in h60_rows]), 3)} "
                 f"| {_fmt(np.mean([r['J_argmin'] for r in h60_rows]), 3)} "
                 f"| {_fmt(np.mean([r['util_edf'] for r in h60_rows]), 3)} "
                 f"| {_fmt(np.mean([r['util_argmin'] for r in h60_rows]), 3)} |")
    lines.append(f"| work-conserving argmin | {_fmt(np.mean([r['J_edf'] for r in wc_rows]), 3)} "
                 f"| {_fmt(np.mean([r['J_argmin'] for r in wc_rows]), 3)} "
                 f"| {_fmt(np.mean([r['util_edf'] for r in wc_rows]), 3)} "
                 f"| {_fmt(np.mean([r['util_argmin'] for r in wc_rows]), 3)} |")
    lines.append("")
    lines.append(f"- horizon-60 label composition on {comp60['n_states']} kept val "
                 f"states: WAIT share {100 * comp60['wait_share_20']:.1f}% (n=20) -> "
                 f"{100 * comp60['wait_share_60']:.1f}% (n=60); mean candidates "
                 f"{comp60['mean_cands_20']:.1f} -> {comp60['mean_cands_60']:.1f}.")

    # OR-Tools
    proven = sum(1 for o in ort if o["proven_optimal"])
    lines.append("")
    lines.append("## OR-Tools CP-SAT window reference (CLAIRVOYANT)")
    lines.append("")
    lines.append(f"- {len(ort)} windows x {exp['ortools']['window_jobs']} jobs on "
                 f"{exp['ortools']['window_gpus']} GPUs (cumulative relaxation over "
                 "GPU-share and VRAM; fractional shares exact, per-GPU packing "
                 "relaxed — disclosed). Proven optimal: "
                 f"**{proven}/{len(ort)}** within {exp['ortools']['time_limit_s']} s.")
    lines.append("")
    lines.append("| window | status | proven | CP obj | EDF obj (same relaxed model) |")
    lines.append("|---|---|---|---|---|")
    for o in ort:
        lines.append(f"| {o['window']} | {o['status']} | {o['proven_optimal']} "
                     f"| {o['cp_obj']} | {o['edf_relaxed_obj']:.0f} |")

    out_md = REPORTS / "p6" / "dataset_card.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    meta = {"tau": tau, "n_states": n_states, "throughput": thr,
            "replacement_log": replacement_log,
            "gate": {"rows": gate_rows, "ci": gate_ci, "pass": gate_pass},
            "ablations": {"horizon60_rows": h60_rows, "wc_rows": wc_rows,
                          "composition": comp60},
            "multi_candidate_stats": mc_stats,
            "ortools": ort, "sensitivity": sensitivity,
            "wall_min": wall_total / 60}
    (REPORTS / "p6").mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "p6" / "labels_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"card: {out_md}")
    print(f"precondition gate: {'PASS' if gate_pass else 'FAIL'} "
          f"(mean {gate_ci['mean']:+.4f}, CI [{gate_ci['ci_lo']:+.4f}, "
          f"{gate_ci['ci_hi']:+.4f}])")
    if not gate_pass:
        raise SystemExit("Amendment 4 precondition gate FAILED — stop; do not train.")


if __name__ == "__main__":
    main()
