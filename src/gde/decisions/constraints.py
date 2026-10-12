"""Hard constraint checks (PLAN.md §4.3). Pure functions over the live
cluster state; 100% unit-tested. The engine's admission path
(``ClusterState.any_fit``/``best_fit``) implements the same math — tests assert
consistency so the gate can never reject what the engine would accept or vice
versa. Operator limits come from config (none defined in v1 beyond the power
budget and per-job GPU cap)."""

from __future__ import annotations

from dataclasses import dataclass

from gde.simulator.cluster import EPS, ClusterState


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reasons: tuple[str, ...]


def check_placement(cluster: ClusterState, n_occupied: int, share_per_gpu: float,
                    per_gpu_vram: float, gpu_type_req: str | None,
                    operator_max_gpus_per_job: int | None = None) -> CheckResult:
    """Validate admitting one job right now. Empty reasons == admissible."""
    reasons: list[str] = []

    by_type = [g for g in cluster.gpus.values()
               if gpu_type_req is None or g.spec.type == gpu_type_req]
    if not by_type:
        reasons.append("type_not_available")
        return CheckResult(False, tuple(reasons))

    if operator_max_gpus_per_job is not None and n_occupied > operator_max_gpus_per_job:
        reasons.append("operator_max_gpus_per_job")

    share_ok = [g for g in by_type if g.free_share >= share_per_gpu - EPS]
    vram_ok = [g for g in by_type if g.free_vram >= per_gpu_vram - EPS]
    both_ok = [g for g in share_ok if g in vram_ok]
    if len(both_ok) < n_occupied:
        if len(share_ok) < n_occupied and len(vram_ok) >= n_occupied:
            reasons.append("no_free_share")
        elif len(vram_ok) < n_occupied and len(share_ok) >= n_occupied:
            reasons.append("no_free_vram")
        else:
            reasons.append("insufficient_free_gpus")

    if not cluster._power_ok(tuple(g.gpu_id for g in both_ok[:n_occupied]), share_per_gpu) \
            and not reasons:
        reasons.append("power_budget")
    return CheckResult(not reasons, tuple(reasons))
