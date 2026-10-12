"""Heuristic scheduler baselines (PLAN.md P4). All policies are decision-time
only: they read the DecisionView (runtime estimates, never actual durations)."""

from gde.baselines.best_fit import BestFitPolicy
from gde.baselines.edf import EdfPolicy
from gde.baselines.easy_backfill import EasyBackfillPolicy
from gde.baselines.fifo import FifoPolicy
from gde.baselines.priority import PriorityPolicy
from gde.baselines.sjf import SjfPolicy

ALL_POLICIES = {
    "fifo": FifoPolicy,
    "priority": PriorityPolicy,
    "sjf": SjfPolicy,
    "edf": EdfPolicy,
    "best_fit": BestFitPolicy,
    "easy_backfill": EasyBackfillPolicy,
}

__all__ = [
    "FifoPolicy", "PriorityPolicy", "SjfPolicy", "EdfPolicy",
    "BestFitPolicy", "EasyBackfillPolicy", "ALL_POLICIES",
]
