"""Policy/safety gate (PLAN.md §8, delivered in P5): the model never executes
anything. Every recommendation is re-validated with constraints.py; invalid,
unknown, expired or low-confidence recommendations fall back to a deterministic
heuristic (pre-registered: EDF) which is itself validated. If even the
fallback is inadmissible the gate returns WAIT. Overcommit is impossible by
construction — the gate only ever returns WAIT or a job that passes the same
checks the engine applies at admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from gde.baselines.edf import EdfPolicy
from gde.decisions.constraints import check_placement
from gde.simulator.cluster import ClusterState
from gde.simulator.engine import DecisionView, Policy
from gde.simulator.workload import SimJob


@dataclass(frozen=True)
class GateResult:
    recommended_action: str | None
    final_action: str | None      # job_id or None (= WAIT)
    valid: bool
    fallback_used: bool
    reasons: tuple[str, ...]


class PolicyGate:
    def __init__(self, fallback: Policy | None = None,
                 min_confidence: float | None = None) -> None:
        self.fallback = fallback or EdfPolicy()   # pre-registered rollout/best heuristic
        self.min_confidence = min_confidence

    def gate(self, action: str | None, view: DecisionView, cluster: ClusterState,
             jobs: dict[str, SimJob], confidence: float | None = None) -> GateResult:
        reasons: list[str] = []
        if action is None or action == "WAIT":
            return GateResult(action, None, True, False, ())

        if confidence is not None and self.min_confidence is not None \
                and confidence < self.min_confidence:
            reasons.append("confidence_below_threshold")

        pending_ids = {pv.job_id for pv in view.pending}
        job = jobs.get(action)
        if job is None or action not in pending_ids:
            reasons.append("unknown_or_not_pending")
            return self._fallback(view, cluster, jobs, tuple(reasons))

        if not self._admissible(job, cluster):
            reasons.append("constraint_violation")
            return self._fallback(view, cluster, jobs, tuple(reasons))

        if reasons:  # confidence below threshold but otherwise valid
            return self._fallback(view, cluster, jobs, tuple(reasons))
        return GateResult(action, action, True, False, ())

    def _fallback(self, view: DecisionView, cluster: ClusterState,
                  jobs: dict[str, SimJob], reasons: tuple[str, ...]) -> GateResult:
        choice = self.fallback.decide(view)
        if choice is not None and choice in {pv.job_id for pv in view.pending} \
                and self._admissible(jobs[choice], cluster):
            return GateResult(None, choice, False, True, reasons)
        return GateResult(None, None, False, True, reasons + ("fallback_inadmissible",))

    @staticmethod
    def _admissible(job: SimJob, cluster: ClusterState) -> bool:
        return check_placement(
            cluster, job.n_occupied, job.share_per_gpu, job.per_gpu_vram,
            job.gpu_type_req,
        ).ok
