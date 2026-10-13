"""OR-Tools CP-SAT windowed reference (PLAN.md §5.2) — CLAIRVOYANT.

Knows true durations (outcome information), so it is an offline upper-bound
reference, never an online competitor. Model: non-preemptive jobs on two
cumulative resources — GPU-share capacity N and VRAM capacity sum(VRAM) —
which RELAXES per-GPU bin-packing (disclosed); fractional shares are exact
under this relaxation. Objective: minimize sum of priority-weighted completion
times. Reports CP-SAT status: proven optimal vs time-limited.
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class WindowResult:
    status: str                 # OPTIMAL | FEASIBLE | INFEASIBLE
    proven_optimal: bool
    cp_obj: float | None
    edf_obj: float | None


def solve_window(jobs: list[dict], n_gpus: int, total_vram: float,
                 time_limit_s: float = 10.0) -> tuple[float | None, bool, str]:
    """jobs: {job_id, arrive, dur, share (GPUs), vram, priority}. Returns
    (objective, proven_optimal, status)."""
    model = cp_model.CpModel()
    horizon = int(sum(j["dur"] for j in jobs) + max(j["arrive"] for j in jobs)) + 1
    starts, ends = [], []
    for j in jobs:
        s = model.NewIntVar(int(j["arrive"]), horizon, f"s_{j['job_id']}")
        e = model.NewIntVar(int(j["arrive"]), horizon, f"e_{j['job_id']}")
        model.Add(e == s + int(j["dur"]))
        starts.append(s)
        ends.append(e)
    # cumulative: GPU-share (x1000) and VRAM (x100) over interval variables
    share_demands = [int(round(j["share"] * 1000)) for j in jobs]
    vram_demands = [int(round(j["vram"] * 100)) for j in jobs]
    intervals = [model.NewIntervalVar(s, int(j["dur"]), e, f"iv_{j['job_id']}")
                 for j, s, e in zip(jobs, starts, ends)]
    model.AddCumulative(intervals, share_demands, int(n_gpus * 1000))
    model.AddCumulative(intervals, vram_demands, int(total_vram * 100))
    weighted = []
    for j, e in zip(jobs, ends):
        w = int(j["priority"])
        weighted.append((e, w))
    obj = model.NewIntVar(0, int(horizon * sum(j["priority"] for j in jobs)) + 1, "obj")
    model.Add(obj == sum(w * e for e, w in weighted))
    model.Minimize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return float(solver.ObjectiveValue()), status == cp_model.OPTIMAL, solver.StatusName(status)
    return None, False, solver.StatusName(status)


def edf_window_objective(jobs: list[dict], n_gpus: int, total_vram: float) -> float | None:
    """Greedy EDF (earliest deadline among fitting jobs, deadlines approximated
    by arrival order here — same jobs, same relaxed capacity check) for a
    like-for-like objective comparison inside the relaxed model."""
    import heapq

    remaining = sorted(jobs, key=lambda j: (j["arrive"], j["job_id"]))
    t = 0.0
    used_share = 0.0
    used_vram = 0.0
    running: list[tuple[float, float, float]] = []  # (end, share, vram)
    obj = 0.0
    while remaining or running:
        while running and running[0][0] <= t:
            _, s, v = heapq.heappop(running)
            used_share -= s
            used_vram -= v
        progressed = False
        for j in list(remaining):
            if j["arrive"] <= t and used_share + j["share"] <= n_gpus + 1e-9 \
                    and used_vram + j["vram"] <= total_vram + 1e-9:
                end = t + j["dur"]
                heapq.heappush(running, (end, j["share"], j["vram"]))
                used_share += j["share"]
                used_vram += j["vram"]
                remaining.remove(j)
                obj += j["priority"] * end
                progressed = True
        if remaining and not progressed:
            # advance to the next arrival or the next completion, whichever
            # comes first (completions free capacity for blocked arrivals)
            future = [j["arrive"] for j in remaining if j["arrive"] > t]
            times = ([min(future)] if future else []) + ([running[0][0]] if running else [])
            if not times:  # nothing pending can ever run — cannot happen with
                break      # positive capacities, but guard anyway
            t = max(t, min(times))
            continue
        if running:
            t = running[0][0]
        elif remaining:
            t = max(t, min(j["arrive"] for j in remaining))
    return obj
