"""Decision record / response schemas (PLAN.md §4.5).

The record is what training data (P6) and the API (P12) both use: a state
summary, the candidate list and the allowed actions — decision-time fields
only. The response adds model scores, the policy-gate outcome and latency;
`expected` stays empty until V3 per PLAN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gde.decisions.candidates import CandidateSet
from gde.decisions.policy_gate import GateResult
from gde.simulator.engine import DecisionView


@dataclass(frozen=True)
class DecisionRecord:
    now: float
    state_summary: dict
    candidates: tuple[dict, ...]
    allowed_actions: tuple[str, ...]
    n_pending: int
    n_truncated: int


def build_decision_record(view: DecisionView, candidates: CandidateSet) -> DecisionRecord:
    return DecisionRecord(
        now=view.now,
        state_summary={
            "n_pending": view.n_pending,
            "n_running": view.n_running,
            "utilization": (sum(g.used_share for g in view.gpus) / len(view.gpus)
                            if view.gpus else 0.0),
            "free_vram_total": sum(g.free_vram for g in view.gpus),
            "queue_len": view.n_pending,
        },
        candidates=tuple(
            {"action_id": c.action_id, "job_id": c.job_id, "source": c.source,
             "rank": c.rank}
            for c in candidates.candidates
        ),
        allowed_actions=candidates.allowed_actions,
        n_pending=candidates.n_pending,
        n_truncated=candidates.n_truncated,
    )


@dataclass
class DecisionResponse:
    model_version: str
    scores: dict[str, float]
    recommended_action: str
    confidence: float | None
    policy_gate: GateResult
    decision_latency_ms: float
    expected: dict = field(default_factory=dict)   # only after V3 (PLAN §4.5)
