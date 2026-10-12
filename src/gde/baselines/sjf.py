"""SJF baseline: shortest job first by the submit-time runtime ESTIMATE
(decision-time feature, never the actual duration). Ties by arrival order."""

from __future__ import annotations

from gde.simulator.engine import DecisionView


class SjfPolicy:
    name = "sjf"

    def decide(self, view: DecisionView) -> str | None:
        feasible = [pv for pv in view.pending if pv.fits]
        if not feasible:
            return None
        best = min(feasible, key=lambda pv: (pv.runtime_est_s, pv.arrival_index))
        return best.job_id
