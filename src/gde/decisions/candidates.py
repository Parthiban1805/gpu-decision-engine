"""Candidate construction (PLAN.md §4.3 + P5 conditions).

Candidates = top-K feasible pending jobs by a deterministic pre-rank
(priority desc, then deadline slack asc, then arrival order) UNION the current
pick of every baseline policy (so each baseline's choice is always a
candidate), plus WAIT always last. Order is fully deterministic and depends
only on decision-time fields — runtime estimates or actual durations never
affect the pre-rank, and no outcome field is visible here.
"""

from __future__ import annotations

from dataclasses import dataclass

from gde.baselines import ALL_POLICIES
from gde.simulator.engine import DecisionView, PendingView

DEFAULT_BASELINES: tuple[str, ...] = (
    "fifo", "priority", "sjf", "edf", "best_fit", "easy_backfill",
)


def pre_rank_key(pv: PendingView) -> tuple[int, float, int]:
    """Priority first, then deadline slack, then arrival order (all
    decision-time). Lower is better after negating priority."""
    return (-pv.priority, pv.deadline_ts, pv.arrival_index)


@dataclass(frozen=True)
class Candidate:
    action_id: str            # "RUN:<job_id>" | "WAIT"
    job_id: str | None
    source: str               # "top-k" | "baseline:<name>" | "wait"
    rank: int                 # position in the candidate list


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[Candidate, ...]
    n_pending: int
    n_truncated: int          # feasible jobs beyond top-K (pre-rank cut)
    allowed_actions: tuple[str, ...]

    def job_ids(self) -> tuple[str, ...]:
        return tuple(c.job_id for c in self.candidates if c.job_id is not None)


def build_candidates(view: DecisionView, k: int = 16,
                     baselines: tuple[str, ...] = DEFAULT_BASELINES) -> CandidateSet:
    feasible = [pv for pv in view.pending if pv.fits]
    ranked = sorted(feasible, key=pre_rank_key)
    top = ranked[: k]
    top_ids = {pv.job_id for pv in top}

    picked: list[tuple[str, str]] = []  # (job_id, source) in baseline order
    for name in baselines:
        choice = ALL_POLICIES[name]().decide(view)
        if choice is not None and choice not in top_ids \
                and choice in {pv.job_id for pv in feasible} \
                and all(choice != j for j, _ in picked):
            picked.append((choice, f"baseline:{name}"))

    candidates: list[Candidate] = [
        Candidate(f"RUN:{pv.job_id}", pv.job_id, "top-k", i)
        for i, pv in enumerate(top)
    ]
    base_offset = len(candidates)
    candidates += [
        Candidate(f"RUN:{job_id}", job_id, src, base_offset + i)
        for i, (job_id, src) in enumerate(picked)
    ]
    candidates.append(Candidate("WAIT", None, "wait", len(candidates)))
    return CandidateSet(
        candidates=tuple(candidates),
        n_pending=view.n_pending,
        n_truncated=max(0, len(feasible) - len(top)) + view.n_truncated,
        allowed_actions=tuple(c.action_id for c in candidates),
    )
