"""P6 dataset builder: stratified state sampling on the primary (capped-1h)
track only, parallel rollout labeling, tau from train regrets (Amendment 3),
split-seed replacement (Amendment 1 rule), Parquet output.

States are sampled from every scheduler's trajectories, stratified by
time-in-window tercile x queue-length tercile within each (window, scheduler);
all states from one window stay in that window's split (windows are
seed-assigned to splits in configs/regimes.yaml).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gde.baselines import ALL_POLICIES
from gde.evaluation.benchmark import load_regimes, make_overlay_cfg, regime_spec
from gde.labeling.objective import ObjectiveConstants, load_preset
from gde.labeling.rollout import _softmax_neg_j, label_state
from gde.paths import CONFIGS, DATA
from gde.simulator.cluster import ClusterSpec
from gde.simulator.engine import Engine, EngineConfig, EventKind
from gde.simulator.snapshot import snapshot
from gde.simulator.workload import TraceConfig, generate_overlay, load_trace_window


@dataclass
class SampledState:
    state: object                    # SimState snapshot
    split: str
    seed: int
    scheduler: str
    t_frac: float
    qlen: int
    edf_pick: str | None
    sjf_pick: str | None
    fifo_pick: str | None
    state_id: str = ""


def replace_degenerate_seeds(seeds: list[int], cluster: ClusterSpec,
                             window_s: float, cap_s: float | None,
                             parquet: Path) -> tuple[list[int], list[dict]]:
    """Amendment-1 rule: seed+500 until the capped window is non-empty."""
    replaced: list[int] = []
    out: list[int] = []
    for seed in seeds:
        s = seed
        for _ in range(50):
            try:
                jobs, _ = load_trace_window(
                    parquet, cluster,
                    TraceConfig(window_s=window_s, rho=0.8, max_jobs=10**9,
                                duration_cap_s=cap_s), seed=s)
                if jobs:
                    break
            except ValueError:
                pass
            replaced.append(s)
            s += 500
        out.append(s)
    log = [{"original": o, "replacement": r} for o, r in zip(seeds, out) if o != r]
    return out, log


def _n_feasible_fast(state, cap_at: int = 2) -> int:
    """Count feasible pending jobs, early-exiting at cap_at (Amendment 4
    filter only needs to know whether n_feasible >= 2)."""
    cluster = state.cluster
    n = 0
    for job_id in state.pending:
        job = state.jobs[job_id]
        if cluster.any_fit(job.n_occupied, job.share_per_gpu,
                           job.per_gpu_vram, job.gpu_type_req):
            n += 1
            if n >= cap_at:
                return n
    return n


def sample_states(regimes: dict, exp: dict, split: str) -> tuple[list[SampledState], dict]:
    """Sample MULTI-CANDIDATE states (n_feasible >= 2, Amendment 4) from every
    scheduler's trajectories, stratified by time-in-window tercile x
    queue-length tercile. Returns (states, stats) where stats reports the
    single-feasible share that was skipped."""
    spec = regime_spec(regimes, "capped-1h")
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    parquet = DATA / "processed" / f"{spec['dataset']}.parquet"
    cap = float(spec["duration_cap_s"])
    seeds, _ = replace_degenerate_seeds(
        regimes["splits"]["split_seeds"][split], cluster, 3600.0, cap, parquet)
    per_ws = exp["state_budget"]["states_per_window_scheduler"]

    import yaml
    overlay_raw = yaml.safe_load((CONFIGS / "sim_p3.yaml").read_text(encoding="utf-8"))["overlay"]
    overlay_cfg = make_overlay_cfg(overlay_raw, exp["overlay_share"])

    stats = {"dp_with_pending": 0, "multi_candidate": 0}
    sampled: list[SampledState] = []
    for seed in seeds:
        jobs, tman = load_trace_window(
            parquet, cluster,
            TraceConfig(window_s=3600.0, rho=0.8, max_jobs=10**9, duration_cap_s=cap),
            seed=seed)
        overlay, _ = generate_overlay(len(jobs), tman["t_sim_s"], overlay_cfg, seed=seed)
        t_sim = tman["t_sim_s"]
        for sched_i, sched in enumerate(
                regimes["p6_labeling"]["state_sampling"]["schedulers"]):
            engine = Engine(jobs + overlay, cluster, ALL_POLICIES[sched](),
                            arrival_horizon=t_sim)
            points: list[tuple[float, int, object]] = []
            while len(engine.state.queue):
                ev = engine.state.queue.peek()
                if ev.kind == EventKind.DECISION_POINT and engine.state.pending:
                    stats["dp_with_pending"] += 1
                    if _n_feasible_fast(engine.state) >= 2:
                        stats["multi_candidate"] += 1
                        points.append((engine.state.now / t_sim,
                                       len(engine.state.pending),
                                       snapshot(engine.state)))
                engine.step()
            if not points:
                continue
            view = None
            # terciles by time and queue length
            ts = sorted(p[0] for p in points)
            qs = sorted(p[1] for p in points)
            t_cut = [ts[len(ts) // 3], ts[2 * len(ts) // 3]]
            q_cut = [qs[len(qs) // 3], qs[2 * len(qs) // 3]]
            cells: dict[tuple[int, int], list] = {}
            for p in points:
                ti = 0 if p[0] <= t_cut[0] else (1 if p[0] <= t_cut[1] else 2)
                qi = 0 if p[1] <= q_cut[0] else (1 if p[1] <= q_cut[1] else 2)
                cells.setdefault((ti, qi), []).append(p)
            rng = np.random.default_rng(seed * 100 + sched_i)
            n_taken = 0
            for cell_key in sorted(cells):
                cell = cells[cell_key]
                take = min(max(per_ws // 9, 1), len(cell))
                for p in rng.choice(len(cell), size=take, replace=False):
                    t_frac, qlen, snap = cell[int(p)]
                    vh = _ViewHolder(snap)
                    sampled.append(SampledState(
                        state=snap, split=split, seed=seed, scheduler=sched,
                        t_frac=t_frac, qlen=qlen,
                        edf_pick=ALL_POLICIES["edf"]().decide(vh.view),
                        sjf_pick=ALL_POLICIES["sjf"]().decide(vh.view),
                        fifo_pick=ALL_POLICIES["fifo"]().decide(vh.view)))
                    n_taken += 1
                    if n_taken >= per_ws:
                        break
                if n_taken >= per_ws:
                    break
    for i, s in enumerate(sampled):
        s.state_id = f"{s.split}:{s.seed}:{s.scheduler}:{i}"
    return sampled, stats


class _ViewHolder:
    def __init__(self, state) -> None:
        engine = Engine.__new__(Engine)
        engine.state = state
        engine.config = EngineConfig()
        self.view = engine.build_view()


def label_one(args: tuple) -> dict:
    """Worker: full label (J vector + feature rows) for one sampled state;
    optionally also the estimate-based-TP variant (sensitivity subsample)."""
    ss_meta, horizon, const_dict, k, est_tp = args
    const = ObjectiveConstants(**const_dict)
    labels = label_state(ss_meta["state"], const, tau=1.0,  # tau applied later
                         n_arrivals=horizon["n_arrivals"],
                         max_seconds=horizon["max_seconds"], k=k)
    from gde.decisions.candidates import build_candidates
    from gde.features.state_builder import build_feature_matrix
    vh = _ViewHolder(ss_meta["state"])
    cand_set = build_candidates(vh.view, k=k)
    feature_rows = build_feature_matrix(vh.view, cand_set)
    out = {
        "state_id": ss_meta["state_id"], "split": ss_meta["split"],
        "seed": ss_meta["seed"], "scheduler": ss_meta["scheduler"],
        "t_frac": ss_meta["t_frac"], "qlen": ss_meta["qlen"],
        "edf_pick": ss_meta["edf_pick"], "sjf_pick": ss_meta["sjf_pick"],
        "fifo_pick": ss_meta["fifo_pick"],
        "candidates": list(labels.candidates), "job_ids": list(labels.job_ids),
        "j_values": list(labels.j_values), "n_feasible": labels.n_feasible,
        "n_pending": labels.n_pending,
        "components": [dict(o.components) for o in labels.outcomes],
        "rollout_runtime_s": [o.rollout_runtime_s for o in labels.outcomes],
        "feature_rows": feature_rows,
    }
    if est_tp:
        est = label_state(ss_meta["state"], const, tau=1.0,
                          n_arrivals=horizon["n_arrivals"],
                          max_seconds=horizon["max_seconds"], k=k,
                          est_based_tp=True)
        out["j_est_values"] = list(est.j_values)
        out["hard_est_label"] = est.candidates[int(np.argmin(est.j_values))]
    return out


def tau_from_labels(records: list[dict]) -> float:
    """Amendment 3: tau = 0.5 x median positive regret over TRAIN states."""
    positive = []
    for r in records:
        j_min = min(r["j_values"])
        positive.extend(j - j_min for j in r["j_values"] if j - j_min > 1e-12)
    if not positive:
        return 1.0
    return 0.5 * float(np.median(positive))


def apply_tau(records: list[dict], tau: float) -> None:
    for r in records:
        j = r["j_values"]
        j_min = min(j)
        r["regret"] = [v - j_min for v in j]
        r["hard_label"] = r["candidates"][int(np.argmin(j))]
        r["soft_target"] = _softmax_neg_j(j, tau)
        r["entropy"] = float(-sum(p * math.log(max(p, 1e-12)) for p in r["soft_target"]))


def to_parquet(records: list[dict], tau: float, path: Path) -> pd.DataFrame:
    from gde.features.state_builder import FEATURE_COLUMNS
    rows = []
    for r in records:
        for i, action_id in enumerate(r["candidates"]):
            feats = r.get("feature_rows", [{} for _ in r["candidates"]])[i]
            row = {
                "state_id": r["state_id"], "split": r["split"], "seed": r["seed"],
                "scheduler": r["scheduler"], "t_frac": r["t_frac"], "qlen": r["qlen"],
                "n_feasible": r["n_feasible"], "n_pending": r["n_pending"],
                "action_id": action_id, "job_id": r["job_ids"][i],
                "candidate_rank": i, "is_wait": int(action_id == "WAIT"),
                "J": r["j_values"][i], "regret": r["regret"][i],
                "soft_target": r["soft_target"][i],
                "hard_label": r["hard_label"], "tau": tau,
                "n_candidates": len(r["candidates"]),
                "edf_pick": r["edf_pick"], "sjf_pick": r["sjf_pick"],
                "fifo_pick": r["fifo_pick"],
                "tp_priority_hours": r["components"][i]["tp_priority_hours"],
                "C": r["components"][i]["C"], "V": r["components"][i]["V"],
                "T": r["components"][i]["T"], "F": r["components"][i]["F"],
            }
            row.update({c: float(feats.get(c, 0.0)) for c in FEATURE_COLUMNS})
            rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
