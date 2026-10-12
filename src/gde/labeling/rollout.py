"""Counterfactual rollout labeling (PLAN.md §5.2, primary track only).

For a sampled decision state s and each candidate action a (from
decisions.candidates): fork, apply a, continue with the pre-registered
rollout policy (edf) until the pre-registered horizon binds (n_arrivals=20
on capped-1h; max_seconds=86400 cannot bind first), then compute J(a)
(objective.py, true-remaining terminal penalty). regret(a) = J(a) - min J;
hard label = argmin; soft target = softmax(-J / tau).

Deterministic by construction: forks of a deterministic engine, no RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gde.baselines.edf import EdfPolicy
from gde.decisions.candidates import CandidateSet, build_candidates
from gde.labeling.objective import (
    ObjectiveConstants,
    integral_base,
    window_objective,
)
from gde.simulator.engine import Engine, EngineConfig, EventKind, Policy, SimState
from gde.simulator.snapshot import fork


class ForcedFirstPolicy:
    """Applies the candidate action once, then hands over to the rollout
    policy for the rest of the rollout."""

    def __init__(self, forced_job_id: str | None, rollout: Policy) -> None:
        self.name = f"forced({forced_job_id or 'WAIT'})+{getattr(rollout, 'name', 'rollout')}"
        self._forced = forced_job_id
        self._rollout = rollout
        self._done = False

    def decide(self, view):  # DecisionView
        if not self._done:
            self._done = True
            if self._forced is not None:
                return self._forced
        return self._rollout.decide(view)


def run_until_horizon(engine: Engine, t0: float, n_arrivals: int,
                      max_seconds: float) -> int:
    """Step the forked engine until the horizon binds (pre-registered
    semantics: n_arrivals NEW arrivals, or max_seconds elapsed). Advances the
    clock/integrals exactly to the horizon end so J windows are consistent.
    Returns arrivals processed."""
    arrivals = 0
    t_end = t0 + max_seconds
    while len(engine.state.queue):
        ev = engine.state.queue.peek()
        if ev.time > t_end:
            break
        ev = engine.step()
        if ev.kind == EventKind.JOB_ARRIVE:
            arrivals += 1
            if arrivals >= n_arrivals:
                break
    # advance integrals exactly to min(horizon end, last event time) — after
    # the loop the clock sits at the last processed event
    stop = min(t_end, engine.state.now if engine.state.queue else engine.state.now)
    if engine.state.now < stop:
        engine._advance_clock(stop)
    return arrivals


@dataclass(frozen=True)
class CandidateOutcome:
    action_id: str
    job_id: str | None
    j: float
    components: dict
    rollout_runtime_s: float


@dataclass(frozen=True)
class StateLabels:
    now: float
    candidates: tuple[str, ...]
    job_ids: tuple[str | None, ...]
    j_values: tuple[float, ...]
    regret: tuple[float, ...]
    hard_label: str
    soft_target: tuple[float, ...]
    outcomes: tuple[CandidateOutcome, ...]
    n_feasible: int
    n_pending: int


def evaluate_candidate(state: SimState, candidate_job_id: str | None,
                       const: ObjectiveConstants, n_arrivals: int,
                       max_seconds: float, rollout: Policy | None = None,
                       est_based_tp: bool = False) -> CandidateOutcome:
    import time as _t

    rollout = rollout or EdfPolicy()
    t0 = state.now
    base = integral_base(state)
    engine = fork(state, ForcedFirstPolicy(candidate_job_id, rollout))
    t_start = _t.perf_counter()
    run_until_horizon(engine, t0, n_arrivals, max_seconds)
    wall = _t.perf_counter() - t_start
    t_end = min(t0 + max_seconds, max(engine.state.now, t0))
    comp = window_objective(engine.state, t0, t_end, const,
                            est_based_tp=est_based_tp, base=base)
    return CandidateOutcome(
        action_id="WAIT" if candidate_job_id is None else f"RUN:{candidate_job_id}",
        job_id=candidate_job_id, j=comp["J"], components=comp, rollout_runtime_s=wall,
    )


def label_state(state: SimState, const: ObjectiveConstants, tau: float,
                n_arrivals: int = 20, max_seconds: float = 86400.0,
                k: int = 16, est_based_tp: bool = False) -> StateLabels:
    view_holder = _ViewHolder(state)
    candidates: CandidateSet = build_candidates(view_holder.view, k=k)
    outcomes = tuple(
        evaluate_candidate(state, c.job_id, const, n_arrivals, max_seconds,
                           est_based_tp=est_based_tp)
        for c in candidates.candidates
    )
    j_values = [o.j for o in outcomes]
    j_min = min(j_values)
    regrets = [j - j_min for j in j_values]
    hard_idx = j_values.index(j_min)
    # tie-break the hard label by candidate order (deterministic)
    soft = _softmax_neg_j(j_values, tau)
    n_feasible = sum(1 for pv in view_holder.view.pending if pv.fits)
    return StateLabels(
        now=state.now,
        candidates=tuple(c.action_id for c in candidates.candidates),
        job_ids=tuple(c.job_id for c in candidates.candidates),
        j_values=tuple(j_values), regret=tuple(regrets),
        hard_label=outcomes[hard_idx].action_id,
        soft_target=tuple(soft), outcomes=outcomes,
        n_feasible=n_feasible, n_pending=view_holder.view.n_pending,
    )


def _softmax_neg_j(j_values: list[float], tau: float) -> list[float]:
    tau = max(tau, 1e-9)
    z = [-j / tau for j in j_values]
    zmax = max(z)
    exp = [math.exp(v - zmax) for v in z]
    s = sum(exp)
    return [v / s for v in exp]


class _ViewHolder:
    """build_view() lives on Engine; a throwaway view of the state suffices."""

    def __init__(self, state: SimState) -> None:
        engine = Engine.__new__(Engine)
        engine.state = state
        engine.config = EngineConfig()
        self.view = engine.build_view()
