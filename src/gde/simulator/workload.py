"""SimJob spec, trace replayer (window + load scaling) and synthetic overlay.

Replay recipe (PLAN.md §7): sample a time window from a processed Parquet file,
filter to jobs feasible on the simulated cluster (BEFORE scaling, else the
1000-GPU trace tails dominate offered work), then compress the window so the
offered load hits target rho on the simulated cluster. Durations are never
scaled — only the arrival rate is. Deadline slack is preserved in SECONDS
(deadline_sim = submit_sim + (deadline_ts - submit_ts)); deadlines are
slack-relative so this keeps them meaningfully attached to unscaled durations.

Offered work W counts the GPU-seconds a job will actually occupy:
share_per_gpu * n_occupied * duration_s — so for fractional >1 jobs (e.g. 1.5
GPUs ceiled to 2) W counts 2 GPUs, and realized rho matches the target.
compute_factor is taken as 1.0 here; heterogeneous clusters shift realized rho
by their mean factor, which the run report measures and prints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gde.simulator.cluster import EPS, ClusterSpec

TRACE_GROUP = "trace"
OVERLAY_GROUP = "overlay"


@dataclass(frozen=True)
class SimJob:
    job_id: str
    group: str                      # trace | overlay
    dataset: str
    submit_ts: float
    duration_s: float               # outcome only, never shown to a policy
    runtime_est_s: float            # decision-time feature
    num_gpus: float
    n_occupied: int                 # GPUs actually occupied (1 for share jobs)
    share_per_gpu: float            # share taken on each occupied GPU
    per_gpu_vram: float
    gpu_type_req: str | None
    job_type: str
    priority: int
    deadline_ts: float
    load_s: float                   # model load time (overlay online inference)
    slo_s: float | None             # max queue wait SLO (online inference)
    # provenance tags (measured / derived / synthetic) for derived fields
    vram_src: str = "measured"
    job_type_src: str = "measured"
    priority_src: str = "measured"
    deadline_src: str = "measured"
    runtime_est_src: str = "measured"

    @property
    def expected_gpu_seconds(self) -> float:
        """Offered work as it will actually be consumed on the cluster."""
        return self.n_occupied * self.share_per_gpu * self.duration_s

    def synthetic_all(self) -> bool:
        """True when every derived field is synthetic (overlay jobs)."""
        return all(tag == "synthetic" for tag in (
            self.vram_src, self.job_type_src, self.priority_src,
            self.deadline_src, self.runtime_est_src))


def placement_shape(num_gpus: float, vram_gb: float) -> tuple[int, float, float]:
    """Map (num_gpus, total job VRAM) -> (n_occupied, share_per_gpu, per_gpu_vram).

    num_gpus <= 1: one GPU, fractional share. Integer num_gpus > 1: that many
    whole GPUs. Fractional num_gpus > 1: ceiled up to whole GPUs (count these —
    they consume more than they ask for).
    """
    if num_gpus <= 1.0 + EPS:
        return 1, float(num_gpus), float(vram_gb)
    if abs(num_gpus - round(num_gpus)) < EPS:
        k = int(round(num_gpus))
    else:
        k = int(math.ceil(num_gpus - EPS))
    return k, 1.0, float(vram_gb) / k


@dataclass(frozen=True)
class TraceConfig:
    window_s: float = 86400.0
    rho: float = 0.8
    est_mode: str = "stored"        # stored | oracle | noisy
    noise_sigma: float = 0.25
    max_jobs: int = 5000
    gpu_type: str = "ignore"        # ignore | strict (sim types are abstract)


@dataclass(frozen=True)
class OverlayConfig:
    share: float = 0.0              # extra arrivals as a fraction of trace rate
    duration_mu: float = 2.505      # lognormal params -> ~5..30 s after clipping
    duration_sigma: float = 0.436
    duration_min_s: float = 5.0
    duration_max_s: float = 30.0
    num_gpus_min: float = 0.1
    num_gpus_max: float = 0.5
    priorities: tuple[int, ...] = (9, 10)
    slo_choices: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    model_load_s: float = 0.0
    vram_min_gb: float = 2.0
    vram_max_gb: float = 10.0

    def is_active(self) -> bool:
        return self.share > 0.0


def load_trace_window(parquet_path: str | Path, cluster: ClusterSpec, cfg: TraceConfig,
                      seed: int) -> tuple[list[SimJob], dict]:
    """Sample a window, scale arrivals to target rho, return jobs + manifest."""
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(parquet_path)
    dataset = str(df["dataset"].iloc[0]) if len(df) else "unknown"
    manifest: dict = {
        "dataset": dataset,
        "parquet": str(parquet_path),
        "seed": seed,
        "window_s": cfg.window_s,
        "rho_target": cfg.rho,
        "est_mode": cfg.est_mode,
        "noise_sigma": cfg.noise_sigma if cfg.est_mode == "noisy" else None,
        "gpu_type": cfg.gpu_type,
        "max_jobs": cfg.max_jobs,
    }

    t_min = float(df["submit_ts"].min())
    t_max = float(df["submit_ts"].max())
    span = t_max - t_min
    window_s = min(cfg.window_s, span)
    manifest["window_s"] = window_s
    window_start = t_min + float(rng.uniform(0.0, max(span - window_s, 0.0)))
    # closed interval so a job landing exactly on the window end is kept
    win = df[(df["submit_ts"] >= window_start) & (df["submit_ts"] <= window_start + window_s)]
    manifest["window_start"] = window_start
    manifest["n_jobs_in_window"] = int(len(win))

    dropped = {"too_many_gpus": 0, "vram_too_big": 0, "nonpositive_duration": 0}
    feasible_indices: list[int] = []
    n_fractional_ceiled = 0
    per_type: dict[str, float] = {}    # per-type max per-GPU VRAM capacity
    per_type_count: dict[str, int] = {}
    for g in cluster.gpus:  # ClusterSpec.gpus is a tuple of GpuSpec
        per_type[g.type] = max(per_type.get(g.type, 0.0), g.vram_gb)
        per_type_count[g.type] = per_type_count.get(g.type, 0) + 1

    shapes: dict[int, tuple[int, float, float]] = {}
    for idx, row in win.iterrows():
        num_gpus = float(row["num_gpus"])
        vram = float(row["vram_gb"])
        dur = float(row["duration_s"])
        if dur <= 0:
            dropped["nonpositive_duration"] += 1
            continue
        n_occ, share, pg_vram = placement_shape(num_gpus, vram)
        type_req = row["gpu_type_req"]
        type_req = None if (pd.isna(type_req) or cfg.gpu_type == "ignore") else str(type_req)
        types = [t for t in per_type if type_req is None or t == type_req]
        if not types or max((per_type[t] for t in types), default=0.0) < pg_vram - EPS:
            dropped["vram_too_big"] += 1
            continue
        if sum(per_type_count[t] for t in types) < n_occ:
            dropped["too_many_gpus"] += 1
            continue
        if num_gpus > 1.0 + EPS and abs(num_gpus - round(num_gpus)) >= EPS:
            n_fractional_ceiled += 1
        feasible_indices.append(idx)
        shapes[idx] = (n_occ, share, pg_vram)
    manifest["dropped"] = dropped
    manifest["dropped_total"] = int(len(win) - len(feasible_indices))
    manifest["n_fractional_ceiled"] = n_fractional_ceiled

    if not feasible_indices:
        raise ValueError(f"no feasible jobs in window {window_start}+{window_s}s of {parquet_path}")

    if len(feasible_indices) > cfg.max_jobs:
        keep = rng.choice(feasible_indices, size=cfg.max_jobs, replace=False)
        keep = sorted(int(i) for i in keep)
        manifest["max_jobs_applied"] = True
    else:
        keep = sorted(feasible_indices)
        manifest["max_jobs_applied"] = False
    sub = win.loc[keep]

    # offered work from GPUs actually occupied (amendment A)
    work_gpu_s = 0.0
    for idx, row in sub.iterrows():
        n_occ, share, _ = shapes[idx]
        work_gpu_s += n_occ * share * float(row["duration_s"])
    t_sim = work_gpu_s / (cfg.rho * cluster.n_gpus)
    scale = t_sim / window_s
    manifest["work_gpu_s"] = work_gpu_s
    manifest["t_sim_s"] = t_sim
    manifest["arrival_scale_factor"] = window_s / t_sim
    manifest["n_jobs"] = int(len(sub))

    est_rng = np.random.default_rng(seed + 10_000)
    jobs: list[SimJob] = []
    for idx, row in sub.iterrows():
        n_occ, share, pg_vram = shapes[idx]
        submit_sim = (float(row["submit_ts"]) - window_start) * scale
        duration = float(row["duration_s"])
        deadline_sim = submit_sim + (float(row["deadline_ts"]) - float(row["submit_ts"]))
        est, est_src = _estimate(duration, cfg, row, est_rng)
        jobs.append(SimJob(
            job_id=f"{dataset}::{row['job_id']}",
            group=TRACE_GROUP,
            dataset=dataset,
            submit_ts=submit_sim,
            duration_s=duration,
            runtime_est_s=est,
            num_gpus=float(row["num_gpus"]),
            n_occupied=n_occ,
            share_per_gpu=share,
            per_gpu_vram=pg_vram,
            gpu_type_req=None if cfg.gpu_type == "ignore" else (
                None if pd.isna(row["gpu_type_req"]) else str(row["gpu_type_req"])),
            job_type=str(row["job_type"]),
            priority=int(row["priority"]),
            deadline_ts=deadline_sim,
            load_s=0.0,
            slo_s=None,
            vram_src=str(row["vram_gb_src"]),
            job_type_src=str(row["job_type_src"]),
            priority_src=str(row["priority_src"]),
            deadline_src=str(row["deadline_src"]),
            runtime_est_src=est_src,
        ))
    jobs.sort(key=lambda j: (j.submit_ts, j.job_id))
    return jobs, manifest


def sim_jobs_from_unified_df(df: pd.DataFrame, cluster: ClusterSpec | None = None,
                             group: str = TRACE_GROUP, est_mode: str = "stored",
                             sigma: float = 0.0, seed: int = 0,
                             gpu_type: str = "ignore") -> tuple[list[SimJob], dict]:
    """Convert a unified-schema frame (PLAN §2.5) to SimJobs.

    When `cluster` is given, jobs that cannot fit it are dropped (counted in
    the returned manifest) — same rule the trace replayer applies.
    """
    est_rng = np.random.default_rng(seed + 10_000)
    per_type: dict[str, float] = {}
    per_type_count: dict[str, int] = {}
    if cluster is not None:
        for g in cluster.gpus:  # ClusterSpec.gpus is a tuple of GpuSpec
            per_type[g.type] = max(per_type.get(g.type, 0.0), g.vram_gb)
            per_type_count[g.type] = per_type_count.get(g.type, 0) + 1
    dropped = 0
    jobs: list[SimJob] = []
    for _, row in df.iterrows():
        n_occ, share, pg_vram = placement_shape(float(row["num_gpus"]), float(row["vram_gb"]))
        if cluster is not None:
            type_req = row["gpu_type_req"]
            type_req = None if (pd.isna(type_req) or gpu_type == "ignore") else str(type_req)
            types = [t for t in per_type if type_req is None or t == type_req]
            if (not types or max((per_type[t] for t in types), default=0.0) < pg_vram - EPS
                    or sum(per_type_count[t] for t in types) < n_occ):
                dropped += 1
                continue
        duration = float(row["duration_s"])
        if est_mode == "oracle":
            est, est_src = duration, "synthetic"
        elif est_mode == "noisy":
            est, est_src = max(duration * float(np.exp(est_rng.normal(0.0, sigma))), 1e-6), "synthetic"
        else:
            est, est_src = float(row["runtime_est_s"]), str(row["runtime_est_src"])
        jt = str(row["job_type"])
        jobs.append(SimJob(
            job_id=f"{row['dataset']}::{row['job_id']}",
            group=group, dataset=str(row["dataset"]),
            submit_ts=float(row["submit_ts"]), duration_s=duration, runtime_est_s=est,
            num_gpus=float(row["num_gpus"]), n_occupied=n_occ, share_per_gpu=share,
            per_gpu_vram=pg_vram,
            gpu_type_req=None if gpu_type == "ignore" else (
                None if pd.isna(row["gpu_type_req"]) else str(row["gpu_type_req"])),
            job_type=jt, priority=int(row["priority"]),
            deadline_ts=float(row["deadline_ts"]),
            load_s=0.0, slo_s=None,
            vram_src=str(row["vram_gb_src"]), job_type_src=str(row["job_type_src"]),
            priority_src=str(row["priority_src"]), deadline_src=str(row["deadline_src"]),
            runtime_est_src=est_src,
        ))
    jobs.sort(key=lambda j: (j.submit_ts, j.job_id))
    return jobs, {"n_input": int(len(df)), "n_dropped_infeasible": dropped}


def _estimate(duration: float, cfg: TraceConfig, row: pd.Series,
              rng: np.random.Generator) -> tuple[float, str]:
    if cfg.est_mode == "oracle":
        return duration, "synthetic"
    if cfg.est_mode == "noisy":
        est = duration * float(np.exp(rng.normal(0.0, cfg.noise_sigma)))
        return max(est, 1e-6), "synthetic"
    return float(row["runtime_est_s"]), str(row["runtime_est_src"])


def generate_overlay(n_trace_jobs: int, t_sim_s: float, cfg: OverlayConfig,
                     seed: int) -> tuple[list[SimJob], dict]:
    """Extra online-inference arrivals on top of the trace (all fields synthetic)."""
    if not cfg.is_active():
        return [], {"share": 0.0, "n_jobs": 0, "model_load_s": cfg.model_load_s, "seed": None}
    rng = np.random.default_rng(seed + 20_000)
    n = int(round(cfg.share * n_trace_jobs))
    # Poisson process conditioned on n arrivals in [0, t_sim]: sorted uniforms.
    arrivals = np.sort(rng.uniform(0.0, t_sim_s, size=n))
    durations = np.clip(rng.lognormal(cfg.duration_mu, cfg.duration_sigma, size=n),
                        cfg.duration_min_s, cfg.duration_max_s)
    shares = rng.uniform(cfg.num_gpus_min, cfg.num_gpus_max, size=n)
    vrams = rng.uniform(cfg.vram_min_gb, cfg.vram_max_gb, size=n)
    priorities = rng.integers(0, len(cfg.priorities), size=n)
    slos = rng.integers(0, len(cfg.slo_choices), size=n)
    jobs = [
        SimJob(
            job_id=f"overlay::{i}",
            group=OVERLAY_GROUP,
            dataset="overlay",
            submit_ts=float(arrivals[i]),
            duration_s=float(durations[i]),
            runtime_est_s=float(cfg.model_load_s + durations[i]),
            num_gpus=float(shares[i]),
            n_occupied=1,
            share_per_gpu=float(shares[i]),
            per_gpu_vram=float(vrams[i]),
            gpu_type_req=None,
            job_type="online_inference",
            priority=int(cfg.priorities[int(priorities[i])]),
            deadline_ts=float(arrivals[i]) + float(cfg.slo_choices[int(slos[i])]),
            load_s=float(cfg.model_load_s),
            slo_s=float(cfg.slo_choices[int(slos[i])]),
            vram_src="synthetic", job_type_src="synthetic", priority_src="synthetic",
            deadline_src="synthetic", runtime_est_src="synthetic",
        )
        for i in range(n)
    ]
    jobs.sort(key=lambda j: (j.submit_ts, j.job_id))
    manifest = {
        "share": cfg.share,
        "n_jobs": n,
        "model_load_s": cfg.model_load_s,
        "seed": seed,
        "duration": {"mu": cfg.duration_mu, "sigma": cfg.duration_sigma,
                     "min_s": cfg.duration_min_s, "max_s": cfg.duration_max_s},
        "num_gpus": {"min": cfg.num_gpus_min, "max": cfg.num_gpus_max},
        "slo_s": list(cfg.slo_choices),
    }
    return jobs, manifest
