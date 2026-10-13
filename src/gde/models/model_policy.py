"""ModelPolicy: wraps a trained V1 scorer + the P5 policy gate for closed-loop
simulation (and later the API). The model NEVER executes anything: its
recommendation is re-validated by the gate (fallback: pre-registered EDF)."""

from __future__ import annotations

import time

from gde.decisions.candidates import build_candidates
from gde.decisions.policy_gate import PolicyGate
from gde.features.state_builder import build_feature_matrix
from gde.simulator.engine import Engine, DecisionView, Policy


class ModelPolicy(Policy):
    def __init__(self, scorer, gate: PolicyGate | None = None, k: int = 16,
                 min_confidence: float | None = None) -> None:
        """scorer: object with .scores(rows: list[dict]) -> list[float]."""
        self.scorer = scorer
        self.gate = gate or PolicyGate(min_confidence=min_confidence)
        self.k = k
        self.engine: Engine | None = None
        self.latencies_ms: list[float] = []
        self.fallbacks = 0
        self.decisions = 0
        self.name = f"model:{type(scorer).__name__}"

    def bind(self, engine: Engine) -> None:
        self.engine = engine

    def decide(self, view: DecisionView):
        assert self.engine is not None, "bind(engine) required before run()"
        t0 = time.perf_counter()
        candidates = build_candidates(view, k=self.k)
        rows = build_feature_matrix(view, candidates)
        scores = self.scorer.scores(rows)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        recommended = candidates.candidates[best_idx]
        job_id = recommended.job_id  # None for WAIT
        self.decisions += 1
        result = self.gate.gate(job_id, view, self.engine.state.cluster,
                                self.engine.state.jobs,
                                confidence=scores[best_idx])
        if result.fallback_used:
            self.fallbacks += 1
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return result.final_action
