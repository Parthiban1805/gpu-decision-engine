"""Feature builder for the V1 models (PLAN.md §6) — decision-time only.

Every feature derives from the DecisionView (pending/running views, GPU views)
and the candidate: job features, cluster features, interactions. Outcome
fields (duration_s, final_status, start/finish) are structurally out of reach:
the builder consumes views, and the no-leakage test (P5 gate) runs it under
oracle-estimate mode to show that features track the ESTIMATE, not the actual
duration.
"""

from __future__ import annotations

import math

from gde.decisions.candidates import Candidate, CandidateSet
from gde.simulator.engine import DecisionView, PendingView

JOB_COLUMNS = [
    "job_is_training", "job_is_batch_inference", "job_is_online_inference",
    "job_priority", "job_num_gpus", "job_per_gpu_vram",
    "job_log_runtime_est_s", "job_deadline_slack_s", "job_wait_s",
    "job_is_overlay",
]
STATE_COLUMNS = [
    "state_n_pending", "state_n_truncated", "state_n_running",
    "state_utilization", "state_free_vram_total", "state_free_vram_largest",
    "state_free_share_total", "state_power_headroom_w",
    "state_queue_est_mean_s", "state_queue_est_p90_s",
    "state_running_est_remaining_mean_s", "state_running_est_remaining_max_s",
]
INTERACTION_COLUMNS = [
    "int_vram_over_free_largest", "int_slack_over_est", "int_fits",
    "int_priority_minus_queue_max",
]
FEATURE_COLUMNS = JOB_COLUMNS + STATE_COLUMNS + INTERACTION_COLUMNS


def state_features(view: DecisionView) -> dict[str, float]:
    gpus = view.gpus
    free_vrams = sorted((g.free_vram for g in gpus), reverse=True)
    ests = sorted(pv.runtime_est_s for pv in view.pending)
    remaining = sorted(max(0.0, r.est_finish_ts - view.now) for r in view.running)
    power_now = sum(
        g.p_idle_w + (g.p_peak_w - g.p_idle_w) * g.used_share for g in gpus)
    return {
        "state_n_pending": float(view.n_pending),
        "state_n_truncated": float(view.n_truncated),
        "state_n_running": float(view.n_running),
        "state_utilization": sum(g.used_share for g in gpus) / len(gpus) if gpus else 0.0,
        "state_free_vram_total": sum(g.free_vram for g in gpus),
        "state_free_vram_largest": free_vrams[0] if free_vrams else 0.0,
        "state_free_share_total": sum(g.free_share for g in gpus),
        "state_power_headroom_w": max(0.0, view.power_budget_w - power_now)
        if view.power_budget_w else 0.0,
        "state_queue_est_mean_s": (sum(ests) / len(ests)) if ests else 0.0,
        "state_queue_est_p90_s": ests[min(int(0.9 * (len(ests) - 1)), len(ests) - 1)]
        if ests else 0.0,
        "state_running_est_remaining_mean_s": (sum(remaining) / len(remaining))
        if remaining else 0.0,
        "state_running_est_remaining_max_s": remaining[-1] if remaining else 0.0,
    }


def candidate_features(view: DecisionView, pv: PendingView) -> dict[str, float]:
    return {
        "job_is_training": 1.0 if pv.job_type == "training" else 0.0,
        "job_is_batch_inference": 1.0 if pv.job_type == "batch_inference" else 0.0,
        "job_is_online_inference": 1.0 if pv.job_type == "online_inference" else 0.0,
        "job_priority": float(pv.priority),
        "job_num_gpus": pv.num_gpus,
        "job_per_gpu_vram": pv.per_gpu_vram,
        "job_log_runtime_est_s": math.log1p(max(pv.runtime_est_s, 0.0)),
        "job_deadline_slack_s": pv.deadline_ts - view.now,
        "job_wait_s": pv.wait_s,
        "job_is_overlay": 1.0 if pv.group == "overlay" else 0.0,
    }


def interaction_features(view: DecisionView, pv: PendingView,
                         state: dict[str, float]) -> dict[str, float]:
    largest = max(state["state_free_vram_largest"], 1e-9)
    est = max(pv.runtime_est_s, 1e-9)
    return {
        "int_vram_over_free_largest": pv.per_gpu_vram / largest,
        "int_slack_over_est": max(pv.deadline_ts - view.now, 0.0) / est,
        "int_fits": 1.0 if pv.fits else 0.0,
        "int_priority_minus_queue_max": float(
            pv.priority - max((p.priority for p in view.pending), default=pv.priority)),
    }


def build_feature_row(view: DecisionView, candidate: Candidate,
                      state: dict[str, float] | None = None) -> dict[str, float]:
    state = state or state_features(view)
    row: dict[str, float] = dict(state)
    if candidate.job_id is None:  # WAIT: neutral job/interaction block
        row.update({c: 0.0 for c in JOB_COLUMNS + INTERACTION_COLUMNS})
        return row
    pv = next(p for p in view.pending if p.job_id == candidate.job_id)
    row.update(candidate_features(view, pv))
    row.update(interaction_features(view, pv, state))
    return row


def build_feature_matrix(view: DecisionView,
                         candidates: CandidateSet) -> list[dict[str, float]]:
    state = state_features(view)
    return [build_feature_row(view, c, state) for c in candidates.candidates]
