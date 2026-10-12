"""Decision layer (PLAN.md P5): candidates, constraints, policy gate, schemas."""

from gde.decisions.candidates import Candidate, CandidateSet, build_candidates
from gde.decisions.constraints import CheckResult, check_placement
from gde.decisions.policy_gate import GateResult, PolicyGate
from gde.decisions.schema import DecisionRecord, DecisionResponse, build_decision_record

__all__ = [
    "Candidate", "CandidateSet", "build_candidates",
    "CheckResult", "check_placement",
    "GateResult", "PolicyGate",
    "DecisionRecord", "DecisionResponse", "build_decision_record",
]
