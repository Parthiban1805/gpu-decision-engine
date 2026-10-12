"""FIFO baseline: strict head-of-line, no skipping.

Admits the oldest pending job when it fits; if the head does not fit the policy
returns WAIT even if a later job would fit. Strict head-of-line blocking is the
point — it is what EASY backfilling (P4) exists to relieve.
"""

from __future__ import annotations

from gde.simulator.engine import DecisionView


class FifoPolicy:
    name = "fifo"

    def decide(self, view: DecisionView) -> str | None:
        if view.head_job_id is not None and view.head_fits:
            return view.head_job_id
        return None
