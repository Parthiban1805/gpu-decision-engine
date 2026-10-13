"""ModelPolicy: wraps a trained V1 scorer + the P5 policy gate for closed-loop
simulation (and later the API). The model NEVER executes anything: its
recommendation is re-validated by the gate (fallback: pre-registered EDF)."""

from __future__ import annotations

import time

from gde.decisions.candidates import build_candidates
from gde.decisions.policy_gate import PolicyGate
from gde.features.state_builder import FEATURE_COLUMNS, build_feature_matrix
from gde.simulator.engine import Engine, DecisionView, Policy


class MlpScorerAdapter:
    """Adapt SetEncoderScorer to the scores(rows) interface ModelPolicy uses."""

    def __init__(self, model) -> None:
        self.model = model
        self.name = "mlp"

    def scores(self, rows: list[dict]) -> list[float]:
        import torch
        x = torch.tensor([[float(r.get(c, 0.0)) for c in FEATURE_COLUMNS]
                          for r in rows], dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(1, x.size(1), dtype=torch.bool)
        return self.model.scores(x, mask)[0].tolist()


class ModelPolicy(Policy):
    def __init__(self, scorer, gate: PolicyGate | None = None, k: int = 16,
                 min_confidence: float | None = None,
                 work_conserving: bool = False) -> None:
        """scorer: object with .scores(rows: list[dict]) -> list[float].

        work_conserving=True implements the pre-registered Amendment 5 rule:
        when any candidate job fits, WAIT is masked out of the candidate set
        given to the scorer (the model only ranks fitting jobs); WAIT remains
        the only option when nothing fits."""
        self.scorer = scorer
        self.gate = gate or PolicyGate(min_confidence=min_confidence)
        self.k = k
        self.work_conserving = work_conserving
        self.engine: Engine | None = None
        self.latencies_ms: list[float] = []
        self.fallbacks = 0
        self.decisions = 0
        self.pre_mask_wait_picks = 0   # Amendment 5 reporting: raw argmax == WAIT
        self.wait_returns = 0          # WAIT actually returned (nothing fit)
        self.name = (f"model:{type(scorer).__name__}"
                     + ("+wc" if work_conserving else ""))

    def bind(self, engine: Engine) -> None:
        self.engine = engine

    def decide(self, view: DecisionView):
        assert self.engine is not None, "bind(engine) required before run()"
        t0 = time.perf_counter()
        candidates = build_candidates(view, k=self.k)
        rows = build_feature_matrix(view, candidates)
        scores = self.scorer.scores(rows)
        self.decisions += 1
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if candidates.candidates[best_idx].action_id == "WAIT":
            self.pre_mask_wait_picks += 1

        if self.work_conserving:
            run_idx = [i for i, c in enumerate(candidates.candidates)
                       if c.job_id is not None]
            if run_idx:  # something fits -> WAIT masked out (Amendment 5)
                best_idx = max(run_idx, key=lambda i: scores[i])
            else:
                self.wait_returns += 1
                self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                return None
        recommended = candidates.candidates[best_idx]
        job_id = recommended.job_id  # None for WAIT
        result = self.gate.gate(job_id, view, self.engine.state.cluster,
                                self.engine.state.jobs,
                                confidence=scores[best_idx])
        if result.fallback_used:
            self.fallbacks += 1
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return result.final_action
