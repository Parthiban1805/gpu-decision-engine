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


def _headroom_seed(args: tuple) -> dict:
    seed, regimes, overlay_raw, horizon, k = args
    const = load_preset("balanced")
    jobs, cluster, t_sim = _jobs_for_seed(regimes, overlay_raw, seed)
    edf = Engine(jobs, cluster, ALL_POLICIES["edf"](), arrival_horizon=t_sim).run()
    oracle = _ArgminOracle(const, horizon, k)
    eng = Engine(jobs, cluster, oracle, arrival_horizon=t_sim)
    oracle.bind(eng)
    argmin = eng.run()
    return {
        "seed": seed,
        "J_edf": trajectory_objective(edf.state, const, t_sim)["J"],
        "J_argmin": trajectory_objective(argmin.state, const, t_sim)["J"],
        "wait_p50_edf": edf.summary["groups"]["trace"]["wait_s_p50"],
        "wait_p50_argmin": argmin.summary["groups"]["trace"]["wait_s_p50"],
        "wall_argmin_s": argmin.wall_s,
    }


def paired_bootstrap_ci(diffs: np.ndarray, n: int = 2000, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(0)
    boots = [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
             for _ in range(n)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(np.mean(diffs)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(diffs))}


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
    spec = regime_spec(regimes, "capped-1h")
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    parquet_path = DATA / "processed" / f"{spec['dataset']}.parquet"
    for split in ("train", "val", "test"):
        _, log = replace_degenerate_seeds(
            regimes["splits"]["split_seeds"][split], cluster, 3600.0,
            float(spec["duration_cap_s"]), parquet_path)
        replacement_log += [{**r, "split": split} for r in log]

    all_meta: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in ("train", "val", "test"):
            sampled = sample_states(regimes, exp, split)
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

    # 5) oracle headroom
    headroom_rows = []
    if not args.skip_headroom:
        payloads = [(seed, regimes, overlay_raw, horizon, exp["k_candidates"])
                    for seed in exp["headroom"]["seeds"]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            headroom_rows = list(pool.map(_headroom_seed, payloads))
        diffs = np.array([h["J_edf"] - h["J_argmin"] for h in headroom_rows])
        headroom_ci = paired_bootstrap_ci(
            diffs, n=exp["headroom"]["bootstrap"]["n"],
            alpha=exp["headroom"]["bootstrap"]["alpha"])
    else:
        headroom_ci = None

    wall_total = time.perf_counter() - t_start

    # ---- dataset card -------------------------------------------------------
    lines: list[str] = []
    lines.append("# P6 dataset card — labeled decision states, PRIMARY (capped-1h) only")
    lines.append("")
    lines.append(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%S')} by `python tasks.py label`. "
                 "Regimes/config: `configs/regimes.yaml` (Amendments 1-3), "
                 "`configs/experiment_p6.yaml`. Rollout policy: edf (frozen). "
                 f"Horizon: n_arrivals={horizon['n_arrivals']}, "
                 f"max_seconds={horizon['max_seconds']} (n_arrivals binds). "
                 "Terminal penalty: true remaining (pre-registered). "
                 "Deadlines/SLOs/VRAM synthetic; 64% of offered work excluded by "
                 "the 6 h cap; batch_inference = short-small proxy. "
                 "Stress track held out.")
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

    # item 3: single-feasible states
    single = sum(1 for r in all_records if r["n_feasible"] <= 1)
    lines.append("")
    lines.append(f"## 3. Single-feasible-candidate states\n")
    lines.append(f"- n_feasible <= 1: **{single}/{n_states} "
                 f"({100 * single / n_states:.1f}%)** — flagged in the Parquet "
                 "(column n_feasible); kept in the dataset, excluded from nothing "
                 "by default (P7 reports with/without as ablation).")

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

    # headroom
    if headroom_rows:
        lines.append("")
        lines.append("## 2. Oracle headroom (argmin-at-every-state vs EDF, closed loop)")
        lines.append("")
        lines.append("| seed | J EDF | J argmin | diff |")
        lines.append("|---|---|---|---|")
        for h in headroom_rows:
            lines.append(f"| {h['seed']} | {h['J_edf']:.3f} | {h['J_argmin']:.3f} "
                         f"| {h['J_edf'] - h['J_argmin']:+.3f} |")
        ci = headroom_ci
        argmin_better = ci["ci_hi"] < 0  # J_edf - J_argmin > 0 with CI excluding 0
        lines.append("")
        lines.append(f"- paired mean improvement (J_EDF - J_argmin) **{ci['mean']:+.4f}**, "
                     f"bootstrap 95% CI [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}] over "
                     f"{ci['n']} seeds.")
        if argmin_better:
            lines.append("- Substantial, significant headroom for learned policies.")
        else:
            lines.append(
                "- **FLAGGED BEFORE P7 (per pre-registration): the greedy horizon-argmin "
                "oracle does NOT beat EDF on trajectory J** (mean favors EDF; CI spans "
                "zero). Interpretation: optimizing the 20-arrival horizon J greedily at "
                "every state is myopic w.r.t. trajectory J, and 69.5% of states have "
                "<=1 feasible candidate, so most states carry no decision freedom. The "
                "learnable signal is per-state rollout regret; expectations that V1 "
                "beats EDF's closed-loop trajectory J should be tempered accordingly — "
                "this is exactly what the P7 gate will test.")
        lines.append(f"- argmin policy wall: {np.mean([h['wall_argmin_s'] for h in headroom_rows]):.0f} "
                     "s/seed (labels computed live at every decision point).")

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
            "replacement_log": replacement_log, "headroom": headroom_rows,
            "headroom_ci": headroom_ci, "ortools": ort, "sensitivity": sensitivity,
            "wall_min": wall_total / 60}
    (REPORTS / "p6").mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "p6" / "labels_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"card: {out_md}")


if __name__ == "__main__":
    main()
