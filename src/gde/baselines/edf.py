"""EDF baseline: earliest deadline first among pending jobs that fit.
deadline_ts is decision-time info (synthetic for trace jobs — disclosed);
online-inference deadlines are submit + SLO. Ties by arrival order."""

from __future__ import annotations

from gde.simulator.engine import DecisionView


class EdfPolicy:
    name = "edf"

    def decide(self, view: DecisionView) -> str | None:
        feasible = [pv for pv in view.pending if pv.fits]
        if not feasible:
            return None
        best = min(feasible, key=lambda pv: (pv.deadline_ts, pv.arrival_index))
        return best.job_id
