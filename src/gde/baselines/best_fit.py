"""Best-fit baseline (job selection): among pending jobs that fit, admit the
one whose placement is SNUGGEST — the smallest leftover VRAM on its best GPU
— to fill fragmentation rather than open new gaps. Complements the engine's
deterministic best-fit placement. Ties by arrival order."""

from __future__ import annotations

from gde.simulator.engine import DecisionView, PendingView
from gde.simulator.workload import placement_shape


def snugness(pv: PendingView, view: DecisionView) -> float:
    """Best (smallest) leftover VRAM this job can achieve right now; inf if it
    cannot fit (should not happen for jobs flagged fits=True)."""
    n_occ, share, _ = placement_shape(pv.num_gpus, 1.0)
    pg_vram = pv.per_gpu_vram
    eligible = [
        g for g in view.gpus
        if (pv.gpu_type_req is None or g.type == pv.gpu_type_req)
        and g.free_share >= share - 1e-9 and g.free_vram >= pg_vram - 1e-9
    ]
    if len(eligible) < n_occ:
        return float("inf")
    return min(g.free_vram - pg_vram for g in eligible)


class BestFitPolicy:
    name = "best_fit"

    def decide(self, view: DecisionView) -> str | None:
        feasible = [pv for pv in view.pending if pv.fits]
        if not feasible:
            return None
        scored = [(snugness(pv, view), pv.arrival_index, pv) for pv in feasible]
        scored = [s for s in scored if s[0] != float("inf")]
        if not scored:
            return None
        return min(scored, key=lambda s: (s[0], s[1]))[2].job_id
