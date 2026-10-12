"""Objective J (PLAN.md §5.1) — exact formulas, pre-registered (Amendment 3).

J = w_cost·C + w_deadline·V + w_completion·T + w_frag·F over an evaluation
window, with every term normalized so the preset weights are comparable:

- C = (energy_kwh × energy_price + idle_gpu_hours × idle_penalty) / (N_gpus × H_hours)
      — dollars per GPU-hour of the window.
- V = [ Σ_{finished training/batch in window} priority × max(0, finish − deadline)_h
       + slo_miss_penalty × (# finished online jobs that missed SLO) ] / n_active
- T = [ Σ_{finished in window} priority × (wait + run)_h + TP ] / n_active
      (hours per active job; TP is the pre-registered terminal penalty in the
      same priority·hour units)
- F = mean fragmentation sampled at decision points in the window (0..1).
- n_active = jobs finished in window + pending at end + running at end.
- TP (true-remaining, pre-registered Amendment 2):
      Σ_{pending at end} priority × (load + duration)/cf          [never started]
    + Σ_{running at end} priority × max(0, (load + duration)/cf − elapsed)
  Estimate variant (label sensitivity check): runtime_est in place of duration.

The same component math without horizon windowing (whole trajectory, TP = 0)
gives trajectory J for closed-loop policy comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from gde.paths import CONFIGS
from gde.simulator.cluster import ClusterState
from gde.simulator.engine import SimState
from gde.simulator.metrics import JobRecord


@dataclass(frozen=True)
class ObjectiveConstants:
    w_cost: float
    w_deadline: float
    w_completion: float
    w_frag: float
    energy_price_per_kwh: float
    idle_gpu_penalty_per_hour: float
    slo_miss_penalty: float


def load_preset(name: str = "balanced", path: Path | None = None) -> ObjectiveConstants:
    raw = yaml.safe_load((path or CONFIGS / "objectives.yaml").read_text(encoding="utf-8"))
    p = raw["presets"][name]
    c = raw["constants"]
    return ObjectiveConstants(
        w_cost=p["w_cost"], w_deadline=p["w_deadline"],
        w_completion=p["w_completion"], w_frag=p["w_frag"],
        energy_price_per_kwh=c["energy_price_per_kwh"],
        idle_gpu_penalty_per_hour=c["idle_gpu_penalty_per_hour"],
        slo_miss_penalty=c["slo_miss_penalty"],
    )


def _true_remaining_hours_from_job(job, est_based: bool) -> float:
    est_or_dur = job.runtime_est_s if est_based else job.duration_s
    return (job.load_s + est_or_dur) / 3600.0


def terminal_penalty(state: SimState, t_end: float, est_based: bool = False) -> float:
    """Pre-registered TP in priority·hours (Amendment 2). compute_factor is
    taken as 1.0 (cluster_small; documented simplification)."""
    jobs = state.jobs
    total = 0.0
    for job_id in state.pending:
        job = jobs[job_id]
        total += job.priority * _true_remaining_hours_from_job(job, est_based)
    for job_id, placement in state.cluster.placements.items():
        job = jobs[job_id]
        occ = _true_remaining_hours_from_job(job, est_based)
        elapsed = max(0.0, (t_end - placement.start_ts) / 3600.0)
        total += job.priority * max(0.0, occ - elapsed)
    return total


def integral_base(state: SimState) -> tuple[float, float]:
    """Per-gpu integral sums at snapshot time; window value = final - base."""
    return (sum(g.energy_j for g in state.cluster.gpus.values()),
            sum(g.share_time_s for g in state.cluster.gpus.values()))


def window_objective(state: SimState, t0: float, t1: float,
                     const: ObjectiveConstants, est_based_tp: bool = False,
                     base: tuple[float, float] | None = None) -> dict:
    """J components over [t0, t1]; `state` is the FINAL state of the fork run.
    Integrals accumulate from run start, so the window is the delta against
    `base` captured at fork time (energy_j, share_time_s). Default base (0,0)
    is correct for windows starting at the run's t=0 (e.g. whole trajectories)."""
    h_hours = max((t1 - t0) / 3600.0, 1e-9)
    n_gpus = state.cluster.spec.n_gpus
    base_energy_j, base_share_time_s = base or (0.0, 0.0)

    energy_kwh = (sum(g.energy_j for g in state.cluster.gpus.values())
                  - base_energy_j) / 3.6e6
    share_hours = (sum(g.share_time_s for g in state.cluster.gpus.values())
                   - base_share_time_s) / 3600.0
    idle_hours = n_gpus * h_hours - share_hours
    cost_raw = (energy_kwh * const.energy_price_per_kwh
                + idle_hours * const.idle_gpu_penalty_per_hour)

    finished = [r for r in state.records.values()
                if r.final and r.finish_ts is not None and t0 < r.finish_ts <= t1]
    lateness = 0.0
    slo_misses = 0
    completion = 0.0
    for r in finished:
        if r.job_type == "online_inference":
            if r.slo_met is False:
                slo_misses += 1
        else:
            lateness += r.priority * max(0.0, (r.finish_ts - r.deadline_ts) / 3600.0)
        completion += r.priority * ((r.wait_s or 0.0) + (r.run_s or 0.0)) / 3600.0

    tp = terminal_penalty(state, t1, est_based=est_based_tp)
    n_active = len(finished) + len(state.pending) + len(state.cluster.placements)

    frags = [s.frag for s in state.samples
             if s.frag is not None and t0 <= s.t <= t1]

    c = cost_raw / (n_gpus * h_hours)
    v = (lateness + const.slo_miss_penalty * slo_misses) / max(n_active, 1)
    t = (completion + tp) / max(n_active, 1)
    f = (sum(frags) / len(frags)) if frags else 0.0
    j = const.w_cost * c + const.w_deadline * v + const.w_completion * t + const.w_frag * f
    return {"J": j, "C": c, "V": v, "T": t, "F": f,
            "tp_priority_hours": tp, "slo_misses": slo_misses,
            "lateness_priority_hours": lateness, "n_active": n_active}


def trajectory_objective(result_state: SimState, const: ObjectiveConstants,
                         horizon_s: float) -> dict:
    """Whole-trajectory J for closed-loop policy comparison (TP = 0; the run
    is complete so nothing is pending/running at the end)."""
    return window_objective(result_state, 0.0, horizon_s, const)
