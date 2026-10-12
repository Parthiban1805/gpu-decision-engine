"""EASY backfill (Lifka'95 style, first-fit):

- The FIFO head keeps priority: if it fits now, run it.
- Otherwise compute a RESERVATION for the head — the earliest time it fits
  assuming running jobs finish at their ESTIMATED finish times (runtime_est_s
  feature, never actuals; future arrivals ignored).
- A later pending job may backfill now iff it fits now AND its estimated
  finish is <= the reservation (so the head is never delayed). Backfill scan
  is FIFO order over the first `view_pending_cap` pending jobs.

Reservation math is exact on the decision view (per-GPU share/VRAM replay of
estimated releases). On heterogeneous clusters a backfill candidate's
estimated run uses the slowest eligible compute_factor (conservative).
"""

from __future__ import annotations

from gde.simulator.engine import DecisionView, GpuView, PendingView, RunningView
from gde.simulator.workload import placement_shape

EPS = 1e-9


def _shape(pv: PendingView) -> tuple[int, float, float]:
    n_occ, share, _ = placement_shape(pv.num_gpus, 1.0)
    return n_occ, share, pv.per_gpu_vram


def _fits(free: dict[int, tuple[float, float]], gpus: tuple[GpuView, ...],
          n_occ: int, share: float, pg_vram: float, gpu_type_req: str | None) -> bool:
    eligible = [
        g for g in gpus
        if (gpu_type_req is None or g.type == gpu_type_req)
        and free[g.gpu_id][0] >= share - EPS and free[g.gpu_id][1] >= pg_vram - EPS
    ]
    return len(eligible) >= n_occ


def _reservation_time(view: DecisionView, head: PendingView) -> float | None:
    """Earliest estimated time the head fits; None if never (by estimates)."""
    n_occ, share, pg_vram = _shape(head)
    free = {g.gpu_id: (g.free_share, g.free_vram) for g in view.gpus}
    if _fits(free, view.gpus, n_occ, share, pg_vram, head.gpu_type_req):
        return view.now
    releases = sorted(view.running, key=lambda r: (r.est_finish_ts, r.job_id))
    i = 0
    while i < len(releases):
        t = releases[i].est_finish_ts
        while i < len(releases) and releases[i].est_finish_ts <= t + EPS:
            r = releases[i]
            for gpu_id in r.gpu_ids:
                free[gpu_id] = (free[gpu_id][0] + r.share_per_gpu,
                                free[gpu_id][1] + r.per_gpu_vram)
            i += 1
        if _fits(free, view.gpus, n_occ, share, pg_vram, head.gpu_type_req):
            return t
    return None


def _est_run_s(pv: PendingView, view: DecisionView) -> float:
    n_occ, share, _ = _shape(pv)
    cfs = [
        g.compute_factor for g in view.gpus
        if (pv.gpu_type_req is None or g.type == pv.gpu_type_req)
        and g.free_share >= share - EPS and g.free_vram >= pv.per_gpu_vram - EPS
    ]
    slowest = min(cfs) if cfs else 1.0
    return (pv.load_s + pv.runtime_est_s) / slowest


class EasyBackfillPolicy:
    name = "easy_backfill"

    def decide(self, view: DecisionView) -> str | None:
        if not view.pending:
            return None
        head = view.pending[0]
        if head.fits:
            return head.job_id
        t_reserve = _reservation_time(view, head)
        if t_reserve is None:
            # estimates say the head never fits (stale/underestimates): fall
            # back to conservative first-fit that fits now, FIFO order
            for pv in view.pending:
                if pv.fits:
                    return pv.job_id
            return None
        for pv in view.pending[1:]:
            if not pv.fits:
                continue
            if view.now + _est_run_s(pv, view) <= t_reserve + EPS:
                return pv.job_id
        return None
