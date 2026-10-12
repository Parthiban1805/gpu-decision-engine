"""Priority baseline: among pending jobs that FIT, admit the highest priority;
ties broken by arrival order (oldest first). Skips blocked jobs (unlike FIFO)."""

from __future__ import annotations

from gde.simulator.engine import DecisionView


class PriorityPolicy:
    name = "priority"

    def decide(self, view: DecisionView) -> str | None:
        feasible = [pv for pv in view.pending if pv.fits]
        if not feasible:
            return None
        best = max(feasible, key=lambda pv: (pv.priority, -pv.arrival_index))
        return best.job_id
