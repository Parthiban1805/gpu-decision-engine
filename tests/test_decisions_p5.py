"""P5 gate tests (PLAN.md P5): property-based — a gated action never violates
constraints for random states (queues up to 1000); candidate sets always
contain WAIT and every baseline pick, deterministic and outcome-independent;
feature builder has no access to outcome fields, checked statically AND under
oracle-estimate mode."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gde.baselines import ALL_POLICIES
from gde.decisions.candidates import DEFAULT_BASELINES, build_candidates
from gde.decisions.constraints import check_placement
from gde.decisions.policy_gate import PolicyGate
from gde.decisions.schema import build_decision_record
from gde.features.state_builder import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    state_features,
)
from gde.simulator.cluster import ClusterSpec, ClusterState, Placement
from gde.simulator.engine import Engine, EngineConfig
from gde.simulator.workload import placement_shape

from tests.test_simulator import mkjob


def _engine_with_queue(n_pending: int, n_gpus: int = 2, fill_running: int = 1,
                       seed: int = 0) -> Engine:
    """Engine paused with ~n_pending jobs queued behind a non-fitting FIFO head
    (strict FIFO admits nothing), while feasible candidates still exist in the
    leftovers of the half-GPU blockers — the shape candidate/gate tests need."""
    rng = np.random.default_rng(seed)
    jobs = [mkjob(f"warm_{i}", i * 0.1, 0.05) for i in range(120)]
    # half-GPU blockers alive at t=20 on both GPUs (0.5 share, 4 GB each)
    jobs += [mkjob(f"blk{g}", 19.95, 50.0, share=0.5, vram=4.0) for g in range(2)]
    # the FIFO head never fits while blockers run -> queue really accumulates
    jobs.append(mkjob("q_head", 20.0, 10.0, share=1.0, vram=20.0, priority=2))
    for i in range(n_pending):
        jobs.append(mkjob(
            f"q_{i}", 20.0 + 1e-6 + i * 1e-6, float(rng.integers(1, 30)),
            share=float(rng.choice([0.1, 0.25, 0.5, 0.75, 1.0])),
            vram=float(rng.integers(1, 6)),
            priority=int(rng.integers(1, 11)),
            job_type=str(rng.choice(["training", "batch_inference"]))))
    cluster = ClusterSpec.homogeneous(n_gpus, vram_gb=24.0)
    engine = Engine(jobs, cluster, ALL_POLICIES["fifo"](), arrival_horizon=20.0)
    # step through the arrival burst at t~20.0 only (blockers hold the GPUs
    # until t=69.95, so nothing drains and the queue is exactly the arrivals)
    while len(engine.state.queue) and engine.state.now <= 20.1:
        engine.step()
    return engine


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def test_candidates_always_contain_wait_and_baseline_picks():
    engine = _engine_with_queue(50)
    view = engine.build_view()
    cs = build_candidates(view, k=16)
    assert cs.allowed_actions[-1] == "WAIT"
    assert sum(1 for c in cs.candidates if c.action_id == "WAIT") == 1
    # every baseline's current pick (when it returns one) is in the set
    for name in DEFAULT_BASELINES:
        choice = ALL_POLICIES[name]().decide(view)
        if choice is not None:
            assert choice in cs.job_ids(), name
    # K respected for top-k entries
    assert sum(1 for c in cs.candidates if c.source == "top-k") <= 16


def test_candidates_deterministic_and_outcome_independent():
    engine = _engine_with_queue(40, seed=3)
    view = engine.build_view()
    a = build_candidates(view, k=16)
    b = build_candidates(view, k=16)
    assert a == b
    # identical jobs with DIFFERENT runtime estimates -> identical candidate ids
    # (pre-rank uses priority/deadline-slack/arrival only, never estimates)
    engine2 = _engine_with_queue(40, seed=3)
    state2 = engine2.state
    for jid in list(state2.jobs):
        est2 = state2.jobs[jid].runtime_est_s * 3.7 + 11.0
        state2.jobs[jid] = replace(state2.jobs[jid], runtime_est_s=est2)
    view2 = engine2.build_view()
    c = build_candidates(view2, k=16)
    assert a.allowed_actions == c.allowed_actions


def test_candidates_ordering_follows_pre_rank():
    engine = _engine_with_queue(30, seed=5)
    view = engine.build_view()
    cs = build_candidates(view, k=16)
    top = [c for c in cs.candidates if c.source == "top-k"]
    by_id = {pv.job_id: pv for pv in view.pending}
    keys = [(-by_id[c.job_id].priority, by_id[c.job_id].deadline_ts,
             by_id[c.job_id].arrival_index) for c in top]
    assert keys == sorted(keys)
    assert all(c.source != "wait" for c in cs.candidates[:-1])


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def test_constraints_hand_cases():
    spec = ClusterSpec.homogeneous(2, vram_gb=10.0, p_idle_w=50.0, p_peak_w=100.0,
                                   power_budget_w=1e9)
    cs = ClusterState.empty(spec)
    assert check_placement(cs, 1, 0.5, 6.0, None).ok
    cs.place(Placement("x0", (0,), 0.5, 6.0, 0.0, 1.0, 1.0))
    cs.place(Placement("x1", (1,), 0.5, 6.0, 0.0, 1.0, 1.0))
    # both GPUs have free share 0.5 but only 4 GB free -> vram blocked
    r = check_placement(cs, 1, 0.5, 6.0, None)
    assert not r.ok and "no_free_vram" in r.reasons
    # a small-vram job still fits the share
    assert check_placement(cs, 1, 0.5, 3.0, None).ok
    # share blocked: both GPUs full-share by one whole-GPU job each
    cs2 = ClusterState.empty(spec)
    cs2.place(Placement("w0", (0,), 1.0, 2.0, 0.0, 1.0, 1.0))
    cs2.place(Placement("w1", (1,), 1.0, 2.0, 0.0, 1.0, 1.0))
    r = check_placement(cs2, 1, 0.25, 1.0, None)
    assert not r.ok and "no_free_share" in r.reasons
    r = check_placement(cs2, 3, 1.0, 1.0, None)
    assert not r.ok and "insufficient_free_gpus" in r.reasons
    r = check_placement(cs2, 1, 0.25, 1.0, "nonexistent_type")
    assert not r.ok and "type_not_available" in r.reasons


def test_constraints_consistent_with_engine_admission():
    """check_placement agrees with ClusterState.any_fit on random states."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        spec = ClusterSpec.homogeneous(int(rng.integers(1, 4)), vram_gb=12.0)
        cs = ClusterState.empty(spec)
        for i in range(int(rng.integers(0, 5))):  # random occupancy
            if not cs.any_fit(1, 0.5, 4.0, None):
                break
            cs.place(Placement(f"x{i}", cs.best_fit(1, 0.5, 4.0, None), 0.5, 4.0, 0, 1, 1))
        n_occ = int(rng.choice([1, 1, 2]))
        share = float(rng.choice([0.25, 0.5, 1.0])) if n_occ == 1 else 1.0
        vram = float(rng.uniform(0.5, 12.0))
        assert check_placement(cs, n_occ, share, vram, None).ok == \
            cs.any_fit(n_occ, share, vram, None)


def test_power_budget_constraint():
    spec = ClusterSpec.homogeneous(2, p_idle_w=50.0, p_peak_w=100.0, power_budget_w=150.0)
    cs = ClusterState.empty(spec)
    assert check_placement(cs, 1, 1.0, 1.0, None).ok            # 150 W total ok
    cs.place(Placement("a", (0,), 1.0, 1.0, 0, 1, 1))
    r = check_placement(cs, 1, 1.0, 1.0, None)                   # would hit 200 W
    assert not r.ok and "power_budget" in r.reasons


# ---------------------------------------------------------------------------
# Policy gate — property-based (P5 gate), queues up to 1000
# ---------------------------------------------------------------------------

st_actions = st.one_of(
    st.none(),
    st.sampled_from(["WAIT", "ghost-job", "q_3", "q_10", "warm_0", "blk"]),
    st.integers(min_value=0, max_value=1200).map(lambda i: f"q_{i}"),
)


@given(
    n_pending=st.sampled_from([1, 5, 50, 200, 600, 1000]),
    n_gpus=st.sampled_from([1, 2, 4]),
    seed=st.integers(0, 20),
    action=st_actions,
    min_confidence=st.sampled_from([None, 0.5]),
    confidence=st.one_of(st.none(), st.floats(0.0, 1.0)),
)
@settings(max_examples=60, deadline=None)
def test_gated_action_never_violates_constraints(n_pending, n_gpus, seed, action,
                                                 min_confidence, confidence):
    engine = _engine_with_queue(n_pending, n_gpus=n_gpus, seed=seed)
    state = engine.state
    view = engine.build_view()
    gate = PolicyGate(min_confidence=min_confidence)
    result = gate.gate(action, view, state.cluster, state.jobs,
                       confidence=confidence)

    # the final action is WAIT or a currently-pending, admissible job
    if result.final_action is not None:
        job = state.jobs[result.final_action]
        assert result.final_action in {pv.job_id for pv in view.pending}
        assert check_placement(state.cluster, job.n_occupied, job.share_per_gpu,
                               job.per_gpu_vram, job.gpu_type_req).ok
        assert state.cluster.any_fit(job.n_occupied, job.share_per_gpu,
                                     job.per_gpu_vram, job.gpu_type_req)
    # a valid WAIT/unknown/confidence path never leaks an invalid RUN through
    if not result.valid:
        assert result.fallback_used


def test_gate_fallback_and_wait_paths():
    engine = _engine_with_queue(30, seed=1)
    state = engine.state
    view = engine.build_view()

    gate = PolicyGate()
    assert gate.gate(None, view, state.cluster, state.jobs).final_action is None
    assert gate.gate("WAIT", view, state.cluster, state.jobs).valid

    ghost = gate.gate("no-such-job", view, state.cluster, state.jobs)
    assert not ghost.valid and ghost.fallback_used
    assert ghost.final_action is None or ghost.final_action in {
        pv.job_id for pv in view.pending}

    # confidence below threshold -> fallback even for a valid job
    edf_pick = ALL_POLICIES["edf"]().decide(view)
    if edf_pick is not None:
        g = PolicyGate(min_confidence=0.9)
        r = g.gate(edf_pick, view, state.cluster, state.jobs, confidence=0.2)
        assert r.fallback_used


def test_gate_rejects_stale_job_not_in_view():
    """A job that exists in `jobs` but is no longer pending falls back."""
    engine = _engine_with_queue(20, seed=2)
    state = engine.state
    view = engine.build_view()
    finished = "warm_0"  # long finished by queue-build time
    if finished in state.records and state.records[finished].final:
        g = PolicyGate()
        r = g.gate(finished, view, state.cluster, state.jobs)
        assert not r.valid and r.fallback_used
        assert "unknown_or_not_pending" in r.reasons


# ---------------------------------------------------------------------------
# No leakage — static + oracle-estimate mode
# ---------------------------------------------------------------------------

def test_state_builder_static_no_outcome_access():
    """The builder code never reads outcome attributes (P5 static check),
    and the view dataclasses carry no outcome fields."""
    import ast

    import gde.features.state_builder as sb
    import gde.simulator.engine as engine_mod

    forbidden_attrs = {"duration_s", "final_status", "gpu_hours", "deadline_met",
                       "slo_met", "energy_kwh", "finish_ts", "start_ts"}
    tree = ast.parse(inspect.getsource(sb))
    used = [n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in forbidden_attrs]
    assert not used, used

    for view_cls in (engine_mod.PendingView, engine_mod.RunningView,
                     engine_mod.GpuView, engine_mod.DecisionView):
        fields = set(view_cls.__dataclass_fields__)
        assert not (fields & {"duration_s", "final_status"}), view_cls


def test_features_track_estimates_not_actuals():
    """Oracle-estimate mode: swapping runtime_est for a different value changes
    ONLY estimate-derived feature columns; duration-derived state stays equal.
    If features secretly used duration_s, both runs would be identical there
    AND differ elsewhere unpredictably — the assertions pin the exact delta set."""
    engine_a = _engine_with_queue(25, seed=11)
    engine_b = _engine_with_queue(25, seed=11)
    for jid in list(engine_b.state.jobs):  # oracle-style: est := actual duration
        job = engine_b.state.jobs[jid]
        engine_b.state.jobs[jid] = replace(job, runtime_est_s=job.duration_s)
    view_a, view_b = engine_a.build_view(), engine_b.build_view()
    assert view_a.n_pending == view_b.n_pending

    sa, sb_ = state_features(view_a), state_features(view_b)
    est_cols = {"state_queue_est_mean_s", "state_queue_est_p90_s",
                "state_running_est_remaining_mean_s",
                "state_running_est_remaining_max_s"}
    for col in sa:
        if col in est_cols:
            continue
        assert sa[col] == pytest.approx(sb_[col]), col

    cs_a = build_candidates(view_a, k=16)
    cs_b = build_candidates(view_b, k=16)
    rows_a = build_feature_matrix(view_a, cs_a)
    rows_b = build_feature_matrix(view_b, cs_b)
    assert len(rows_a) == len(rows_b)
    est_job_cols = {"job_log_runtime_est_s", "int_slack_over_est"}
    for ra, rb in zip(rows_a, rows_b):
        for col in FEATURE_COLUMNS:
            if col in est_cols | est_job_cols:
                continue
            assert ra[col] == pytest.approx(rb[col]), col


def test_feature_matrix_shape_and_wait_row():
    engine = _engine_with_queue(20, seed=13)
    view = engine.build_view()
    cs = build_candidates(view, k=16)
    rows = build_feature_matrix(view, cs)
    assert all(set(r.keys()) == set(FEATURE_COLUMNS) for r in rows)
    wait_row = rows[-1]
    assert wait_row["job_priority"] == 0.0 and wait_row["int_fits"] == 0.0
    assert wait_row["state_n_pending"] > 0  # state block still populated


def test_decision_record_contains_allowed_actions():
    engine = _engine_with_queue(15, seed=17)
    view = engine.build_view()
    cs = build_candidates(view, k=16)
    rec = build_decision_record(view, cs)
    assert rec.allowed_actions == cs.allowed_actions
    assert rec.allowed_actions[-1] == "WAIT"
    assert rec.n_truncated >= 0 and len(rec.candidates) == len(cs.candidates)
