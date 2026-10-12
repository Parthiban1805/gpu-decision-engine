"""Per-job records, cluster samples and run summaries (PLAN.md §4.2, §4.4).

Split by group (trace | overlay): overlay load is reported separately from
trace load as required. Utilization / VRAM / energy are integrated over the
arrival window (amendment C); makespan covers the full run including the
queue drain after the last arrival.

Fragmentation (PLAN §4.4): F = (free VRAM on GPUs whose free VRAM is below the
smallest per-GPU VRAM requirement among pending jobs) / total VRAM, sampled at
decision points with a non-empty queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from gde.simulator.cluster import ClusterState

if TYPE_CHECKING:  # runtime-free type hint only; engine imports this module
    from gde.simulator.engine import SimState


@dataclass
class JobRecord:
    job_id: str
    group: str
    dataset: str
    job_type: str
    priority: int
    submit_ts: float
    deadline_ts: float
    slo_s: float | None
    load_s: float
    runtime_est_s: float
    duration_s: float           # outcome, kept for analysis only
    start_ts: float | None = None
    finish_ts: float | None = None
    wait_s: float | None = None
    run_s: float | None = None
    gpu_ids: tuple[int, ...] = ()
    gpu_hours: float | None = None
    energy_kwh: float | None = None      # incremental attribution (see engine)
    deadline_met: bool | None = None
    slo_met: bool | None = None
    final: bool = False

    @classmethod
    def from_job(cls, job) -> "JobRecord":
        return cls(
            job_id=job.job_id, group=job.group, dataset=job.dataset,
            job_type=job.job_type, priority=job.priority, submit_ts=job.submit_ts,
            deadline_ts=job.deadline_ts, slo_s=job.slo_s, load_s=job.load_s,
            runtime_est_s=job.runtime_est_s, duration_s=job.duration_s,
        )

    def clone(self) -> "JobRecord":
        # Snapshots share finalized records but clone running ones: both the
        # original and the fork mutate them independently afterwards.
        return JobRecord(**self.__dict__)


@dataclass(frozen=True)
class ClusterSample:
    t: float
    queue_len: int
    util: float
    vram_used: float
    frag: float | None     # sampled only while the queue is non-empty


def fragmentation(free_vrams: Iterable[float], min_pending_vram: float,
                  total_vram: float) -> float:
    """Stranded free VRAM / total VRAM (PLAN.md §4.4)."""
    free = list(free_vrams)
    if total_vram <= 0:
        return 0.0
    stranded = sum(v for v in free if v < min_pending_vram - 1e-9)
    return stranded / total_vram


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(int(q * (len(s) - 1)), len(s) - 1)
    return s[idx]


def _group_summary(records: list[JobRecord], n_gpus: float, horizon_s: float | None) -> dict:
    waits = [r.wait_s for r in records if r.wait_s is not None]
    in_window_gpu_hours = 0.0
    for r in records:
        if r.start_ts is None:
            continue
        if horizon_s is None:
            in_window_gpu_hours += r.gpu_hours or 0.0
        elif r.finish_ts is not None and r.finish_ts > r.start_ts:
            lo = min(r.start_ts, horizon_s)
            hi = min(r.finish_ts, horizon_s)
            frac = max(0.0, (hi - lo) / (r.finish_ts - r.start_ts))
            in_window_gpu_hours += (r.gpu_hours or 0.0) * frac
    deadline_jobs = [r for r in records if r.job_type != "online_inference"]
    slo_jobs = [r for r in records if r.job_type == "online_inference"]
    out = {
        "n_jobs": len(records),
        "wait_s_mean": sum(waits) / len(waits) if waits else 0.0,
        "wait_s_p50": _pct(waits, 0.50),
        "wait_s_p95": _pct(waits, 0.95),
        "gpu_hours": sum(r.gpu_hours or 0.0 for r in records),
        "gpu_hours_in_window": in_window_gpu_hours,
        "energy_kwh_incremental": sum(r.energy_kwh or 0.0 for r in records),
        "deadline_miss_rate": (
            sum(1 for r in deadline_jobs if r.deadline_met is False) / len(deadline_jobs)
            if deadline_jobs else None),
        "slo_miss_rate": (
            sum(1 for r in slo_jobs if r.slo_met is False) / len(slo_jobs)
            if slo_jobs else None),
    }
    if horizon_s and horizon_s > 0:
        out["realized_load"] = in_window_gpu_hours / (n_gpus * horizon_s / 3600.0)
    else:
        out["realized_load"] = None
    return out


def summarize(state: SimState, arrival_horizon: float | None) -> dict:
    records = list(state.records.values())
    cluster = state.cluster
    n = cluster.spec.n_gpus
    horizon = arrival_horizon

    aw_share_time = sum(g.share_time_aw_s for g in cluster.gpus.values())
    aw_energy = sum(g.energy_aw_j for g in cluster.gpus.values())
    total_share_time = sum(g.share_time_s for g in cluster.gpus.values())
    total_energy = sum(g.energy_j for g in cluster.gpus.values())
    total_vram_time = sum(g.vram_time_gb_s for g in cluster.gpus.values())
    finishes = [r.finish_ts for r in records if r.finish_ts is not None]
    makespan = max(finishes) if finishes else 0.0
    frags = [s.frag for s in state.samples if s.frag is not None]
    queue_lens = [s.queue_len for s in state.samples]

    if horizon and horizon > 0:
        utilization_aw = aw_share_time / (n * horizon)
        idle_gpu_hours_aw = (n * horizon - aw_share_time) / 3600.0
    else:
        utilization_aw = None
        idle_gpu_hours_aw = None

    return {
        "cluster": {
            "n_gpus": n,
            "arrival_window_s": horizon,
            "makespan_s": makespan,
            "utilization_arrival_window": utilization_aw,
            "idle_gpu_hours_arrival_window": idle_gpu_hours_aw,
            "gpu_hours_total": total_share_time / 3600.0,
            "energy_kwh_arrival_window": aw_energy / 3.6e6,
            "energy_kwh_total": total_energy / 3.6e6,
            "avg_vram_used_gb_arrival_window": (
                sum(g.vram_time_aw_gb_s for g in cluster.gpus.values()) / horizon
                if horizon and horizon > 0 else None),
            "avg_vram_used_gb_total": (total_vram_time / makespan) if makespan > 0 else None,
            "fragmentation_mean": (sum(frags) / len(frags)) if frags else None,
            "fragmentation_max": max(frags) if frags else None,
            "queue_len_p50": _pct([float(q) for q in queue_lens], 0.50),
            "queue_len_p95": _pct([float(q) for q in queue_lens], 0.95),
            "events_processed": state.events_processed,
            "decision_points": len(state.samples),
        },
        "groups": {
            group: _group_summary([r for r in records if r.group == group], n, horizon)
            for group in sorted({r.group for r in records})
        },
    }
