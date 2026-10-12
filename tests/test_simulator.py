"""P3 gate tests (PLAN.md phase table) + simulator units.

Gates:
1. conservation: GPU-hours used <= capacity x time
2. no VRAM overcommit, ever (log replay)
3. determinism: identical event-log hash across two same-seed runs
4. hand-computed 3-job scenario matches exactly
5. snapshot -> fork reproduces the original trajectory
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gde.baselines.fifo import FifoPolicy
from gde.ingestion.schema import COLUMNS, validate_jobs
from gde.ingestion.synthetic import SyntheticConfig, generate_workload
from gde.paths import CONFIGS, DATA
from gde.simulator.cluster import ClusterSpec, ClusterState, Placement
from gde.simulator.engine import Engine, EngineConfig, PendingView
from gde.simulator.events import EventKind
from gde.simulator.metrics import fragmentation
from gde.simulator.snapshot import fork, snapshot
from gde.simulator.workload import (
    OverlayConfig,
    SimJob,
    TraceConfig,
    generate_overlay,
    load_trace_window,
    placement_shape,
    sim_jobs_from_unified_df,
)


def mkjob(job_id: str, submit: float, duration: float, share: float = 1.0,
          vram: float = 4.0, job_type: str = "training", priority: int = 5,
          group: str = "trace", slo_s: float | None = None, load_s: float = 0.0,
          deadline_slack: float = 10.0) -> SimJob:
    n_occ, share_per_gpu, pg_vram = placement_shape(share, vram)
    return SimJob(
        job_id=job_id, group=group, dataset=group, submit_ts=submit, duration_s=duration,
        runtime_est_s=duration, num_gpus=share, n_occupied=n_occ, share_per_gpu=share_per_gpu,
        per_gpu_vram=pg_vram, gpu_type_req=None, job_type=job_type, priority=priority,
        deadline_ts=submit + deadline_slack * max(duration, 1.0), load_s=load_s, slo_s=slo_s,
    )


def run_sim(jobs, cluster=None, policy=None, horizon=None, config=None):
    cluster = cluster or ClusterSpec.homogeneous(1, vram_gb=24.0)
    engine = Engine(jobs, cluster, policy or FifoPolicy(), config=config,
                    arrival_horizon=horizon if horizon is not None else _horizon(jobs))
    return engine.run()


def _horizon(jobs) -> float:
    return max((j.submit_ts for j in jobs), default=0.0)


def parse_log(log: list[str]):
    out = []
    for line in log:
        t, seq, kind, job_id = line.split("\t")
        out.append((float(t), int(seq), EventKind[kind], None if job_id == "-" else job_id))
    return out


# --------------------------------------------------------------------------
# Gate 4: hand-computed 3-job scenario
# --------------------------------------------------------------------------

def test_hand_computed_three_job_scenario():
    """1 GPU, 24 GB. A(0.5 share, 8 GB, 100 s), B(0.6 share, 8 GB, 50 s) both at
    t=0; C(0.4 share, 16 GB, 40 s) at t=10. FIFO strict:
    - t=0: A starts; B blocked (0.5+0.6 > 1.0).
    - t=10: C arrives, B still head -> WAIT.
    - t=100: A finishes; B starts (0.6, 8 GB); C starts (0.6+0.4=1.0, 8+16=24 GB fits).
    - t=140: C finishes; t=150: B finishes.
    """
    jobs = [
        mkjob("A", 0.0, 100.0, share=0.5, vram=8.0),
        mkjob("B", 0.0, 50.0, share=0.6, vram=8.0),
        mkjob("C", 10.0, 40.0, share=0.4, vram=16.0),
    ]
    result = run_sim(jobs)

    kinds_times = [(t, k.name, j) for t, _, k, j in parse_log(result.log)]
    expected = [
        (0.0, "JOB_ARRIVE", "A"), (0.0, "JOB_ARRIVE", "B"),
        (0.0, "DECISION_POINT", None), (0.0, "JOB_START", "A"),
        (0.0, "DECISION_POINT", None),
        (10.0, "JOB_ARRIVE", "C"), (10.0, "DECISION_POINT", None),
        (100.0, "JOB_FINISH", "A"), (100.0, "DECISION_POINT", None),
        (100.0, "JOB_START", "B"), (100.0, "JOB_START", "C"),
        (140.0, "JOB_FINISH", "C"), (140.0, "DECISION_POINT", None),
        (150.0, "JOB_FINISH", "B"), (150.0, "DECISION_POINT", None),
    ]
    assert kinds_times == expected

    r = result.records
    assert (r["A"].start_ts, r["A"].finish_ts, r["A"].wait_s) == (0.0, 100.0, 0.0)
    assert (r["B"].start_ts, r["B"].finish_ts, r["B"].wait_s) == (100.0, 150.0, 100.0)
    assert (r["C"].start_ts, r["C"].finish_ts, r["C"].wait_s) == (100.0, 140.0, 90.0)
    assert r["A"].gpu_hours == pytest.approx(0.5 * 100 / 3600)
    assert r["B"].gpu_hours == pytest.approx(0.6 * 50 / 3600)
    assert r["C"].gpu_hours == pytest.approx(0.4 * 40 / 3600)
    assert result.summary["cluster"]["makespan_s"] == 150.0


def test_hand_computed_share_conflict_blocks_head():
    """Same-timestamp jobs whose shares exceed 1.0: second waits for the first."""
    jobs = [mkjob("A", 0.0, 30.0, share=0.6, vram=4.0), mkjob("B", 0.0, 20.0, share=0.6, vram=4.0)]
    result = run_sim(jobs)
    assert result.records["B"].start_ts == 30.0


def test_hand_computed_vram_conflict_blocks_head():
    """Shares fit (0.5+0.5) but VRAM does not (6+6 > 10): FIFO strict waits."""
    cluster = ClusterSpec.homogeneous(1, vram_gb=10.0)
    jobs = [mkjob("A", 0.0, 30.0, share=0.5, vram=6.0), mkjob("B", 0.0, 20.0, share=0.5, vram=6.0)]
    result = run_sim(jobs, cluster=cluster)
    assert result.records["A"].start_ts == 0.0
    assert result.records["B"].start_ts == 30.0


# --------------------------------------------------------------------------
# Gates 1+2: conservation and no-overcommit (property-based)
# --------------------------------------------------------------------------

@st.composite
def workloads(draw, max_jobs=30, max_gpus=3):
    n_gpus = draw(st.integers(1, max_gpus))
    n_jobs = draw(st.integers(1, max_jobs))
    shares = [round(x, 2) for x in np.arange(0.1, 1.01, 0.1)]
    shares += [g for g in (2.0, 3.0) if g <= n_gpus]  # keep multi-GPU jobs feasible
    jobs = []
    for i in range(n_jobs):
        submit = float(draw(st.integers(0, 50)))
        duration = float(draw(st.integers(1, 60)))
        share = float(draw(st.sampled_from(shares)))
        vram = float(draw(st.integers(1, 8)))
        jobs.append(mkjob(f"j{i}", submit, duration, share=share, vram=vram))
    cluster = ClusterSpec.homogeneous(n_gpus, vram_gb=12.0, p_idle_w=10.0, p_peak_w=100.0)
    return jobs, cluster


def _invariants(jobs, cluster):
    result = run_sim(jobs, cluster=cluster)
    assert not result.warnings, result.warnings
    # every job finishes
    assert all(r.finish_ts is not None for r in result.records.values())
    # Gate 1: conservation — GPU-hours used never exceed capacity x time
    makespan = result.summary["cluster"]["makespan_s"]
    total_gpu_hours = sum(r.gpu_hours or 0.0 for r in result.records.values())
    capacity_gpu_hours = cluster.n_gpus * makespan / 3600.0
    assert total_gpu_hours <= capacity_gpu_hours + 1e-9
    # Gate 2: no VRAM/share overcommit at any event (log replay)
    spec_by_id = {g.gpu_id: g for g in cluster.gpus}
    used_vram = {i: 0.0 for i in spec_by_id}
    used_share = {i: 0.0 for i in spec_by_id}
    power = {i: spec_by_id[i].p_idle_w for i in spec_by_id}
    for t, _seq, kind, job_id in parse_log(result.log):
        if kind not in (EventKind.JOB_START, EventKind.JOB_FINISH):
            continue
        job = result.state.jobs[job_id]
        sign = 1.0 if kind == EventKind.JOB_START else -1.0
        for gpu_id in result.records[job_id].gpu_ids:
            used_vram[gpu_id] += sign * job.per_gpu_vram
            used_share[gpu_id] += sign * job.share_per_gpu
            power[gpu_id] += sign * (spec_by_id[gpu_id].p_peak_w
                                     - spec_by_id[gpu_id].p_idle_w) * job.share_per_gpu
            assert used_vram[gpu_id] <= spec_by_id[gpu_id].vram_gb + 1e-9, (t, job_id)
            assert used_share[gpu_id] <= 1.0 + 1e-9, (t, job_id)
        assert sum(power.values()) <= cluster.power_budget_w + 1e-6
    return result


@settings(max_examples=75, deadline=None)
@given(workloads())
def test_property_conservation_and_no_overcommit(case):
    jobs, cluster = case
    _invariants(jobs, cluster)


def test_power_budget_blocks_admission():
    """2 GPUs, budget 150 W: idle 2x50, peak 100 each. One full-share job ->
    100+50=150 fits; the second would reach 200 -> must wait."""
    cluster = ClusterSpec.homogeneous(2, p_idle_w=50.0, p_peak_w=100.0, power_budget_w=150.0)
    jobs = [mkjob("A", 0.0, 100.0), mkjob("B", 0.0, 100.0)]
    result = run_sim(jobs, cluster=cluster)
    assert result.records["A"].start_ts == 0.0
    assert result.records["B"].start_ts == 100.0  # blocked by power, not by capacity


# --------------------------------------------------------------------------
# Gate 3: determinism
# --------------------------------------------------------------------------

def _synthetic_jobs(seed: int, n=120) -> list[SimJob]:
    cfg = SyntheticConfig(n_jobs=n, rate_per_s=0.2)
    df = generate_workload(cfg, seed)
    jobs, _ = sim_jobs_from_unified_df(df, cluster=ClusterSpec.homogeneous(4, vram_gb=24.0))
    return jobs


def test_gate_determinism_same_seed_same_hash():
    jobs = _synthetic_jobs(7)
    a = run_sim(jobs, cluster=ClusterSpec.homogeneous(4, vram_gb=24.0))
    b = run_sim(jobs, cluster=ClusterSpec.homogeneous(4, vram_gb=24.0))
    assert a.digest == b.digest
    assert a.log == b.log


def test_different_seed_changes_log():
    a = run_sim(_synthetic_jobs(7), cluster=ClusterSpec.homogeneous(4, vram_gb=24.0))
    b = run_sim(_synthetic_jobs(8), cluster=ClusterSpec.homogeneous(4, vram_gb=24.0))
    assert a.digest != b.digest


# --------------------------------------------------------------------------
# Gate 5: fork equivalence
# --------------------------------------------------------------------------

def test_gate_fork_reproduces_trajectory():
    jobs = _synthetic_jobs(21, n=200)
    cluster = ClusterSpec.homogeneous(4, vram_gb=24.0)
    horizon = _horizon(jobs)
    reference = Engine(jobs, cluster, FifoPolicy(), arrival_horizon=horizon).run()

    for snap_at in (1, 37, 200, 700):
        engine = Engine(jobs, cluster, FifoPolicy(), arrival_horizon=horizon)
        for _ in range(snap_at):
            if not len(engine.state.queue):
                break
            engine.step()
        state = snapshot(engine.state)
        original = fork(state, FifoPolicy(), arrival_horizon=horizon).run()
        forked = fork(engine.state, FifoPolicy(), arrival_horizon=horizon).run()
        assert original.digest == forked.digest == reference.digest
        assert {k: r.__dict__ for k, r in original.records.items()} == \
               {k: r.__dict__ for k, r in forked.records.items()}
        # the snapshot itself is isolated: continuing it twice agrees exactly


def test_fork_does_not_mutate_parent_state():
    jobs = _synthetic_jobs(33, n=100)
    cluster = ClusterSpec.homogeneous(2, vram_gb=24.0)
    engine = Engine(jobs, cluster, FifoPolicy(), arrival_horizon=_horizon(jobs))
    for _ in range(50):
        engine.step()
    before = snapshot(engine.state)
    fork(engine.state, FifoPolicy()).run()
    after = snapshot(engine.state)
    assert before.log == after.log
    assert {k: r.__dict__ for k, r in before.records.items()} == \
           {k: r.__dict__ for k, r in after.records.items()}


# --------------------------------------------------------------------------
# Placement rules
# --------------------------------------------------------------------------

def test_best_fit_prefers_least_leftover_and_low_id():
    spec = ClusterSpec.homogeneous(3, vram_gb=24.0)
    cs = ClusterState.empty(spec)
    # pre-load gpu 0 with 10 GB, gpu 1 with 4 GB, gpu 2 free
    cs.place(Placement(job_id="x0", gpu_ids=(0,), share_per_gpu=0.1, per_gpu_vram=10.0,
                       start_ts=0, finish_ts=1, run_s=1))
    cs.place(Placement(job_id="x1", gpu_ids=(1,), share_per_gpu=0.1, per_gpu_vram=4.0,
                       start_ts=0, finish_ts=1, run_s=1))
    # job needing 5 GB: leftovers 24-10-5=9, 24-4-5=15, 24-5=19 -> gpu 0
    assert cs.best_fit(1, 0.2, 5.0, None) == (0,)
    # job needing 21 GB: gpu0/gpu1 have 14/20 GB free -> only gpu 2 qualifies
    assert cs.best_fit(1, 0.2, 21.0, None) == (2,)
    # tie between identical leftovers -> lowest gpu_id
    empty = ClusterState.empty(ClusterSpec.homogeneous(2, vram_gb=24.0))
    assert empty.best_fit(1, 0.2, 5.0, None) == (0,)


def test_multi_gpu_placement_uses_whole_gpus():
    jobs = [mkjob("big", 0.0, 10.0, share=2.0, vram=8.0),   # 2 whole GPUs, 4 GB each
            mkjob("small", 0.0, 5.0, share=0.5, vram=4.0)]
    cluster = ClusterSpec.homogeneous(2, vram_gb=24.0)
    result = run_sim(jobs, cluster=cluster)
    assert sorted(result.records["big"].gpu_ids) == [0, 1]
    # 'small' cannot share a whole-GPU placement (share sum > 1) -> waits
    assert result.records["small"].start_ts == 10.0
    assert result.records["big"].gpu_hours == pytest.approx(2 * 10 / 3600)


def test_fractional_above_one_ceils_to_whole_gpus():
    n_occ, share, pg_vram = placement_shape(1.5, 20.0)
    assert (n_occ, share, pg_vram) == (2, 1.0, 10.0)
    assert placement_shape(2.0, 8.0) == (2, 1.0, 4.0)
    assert placement_shape(0.4, 6.0) == (1, 0.4, 6.0)


def test_compute_factor_scales_runtime():
    spec = ClusterSpec.homogeneous(1, compute_factor=2.0)
    engine = Engine([mkjob("A", 0.0, 100.0)], spec, FifoPolicy(), arrival_horizon=0.0)
    result = engine.run()
    assert result.records["A"].finish_ts == pytest.approx(50.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_fragmentation_hand_case():
    # free vrams [4, 24], smallest pending need 8 GB: only gpu0's 4 GB stranded
    assert fragmentation([4.0, 24.0], 8.0, 48.0) == pytest.approx(4.0 / 48.0)
    assert fragmentation([24.0, 24.0], 8.0, 48.0) == pytest.approx(0.0)
    assert fragmentation([0.0, 0.0], 8.0, 48.0) == pytest.approx(0.0)


def test_fragmentation_sampled_in_engine():
    jobs = [mkjob("A", 0.0, 100.0, share=0.5, vram=20.0),
            mkjob("B", 0.0, 10.0, share=0.5, vram=8.0)]
    result = run_sim(jobs)  # 1 GPU 24 GB; A runs, B pending needs 8 > free 4
    frags = [s.frag for s in result.state.samples if s.frag is not None]
    # sampled at every DP with a non-empty queue: [empty cluster -> 0,
    # A running while B waits -> 4/24 stranded, A finished -> 0]
    assert len(frags) == 3
    assert frags[1] == pytest.approx(4.0 / 24.0)
    assert result.summary["cluster"]["fragmentation_mean"] == pytest.approx((4.0 / 24.0) / 3)


def test_utilization_over_arrival_window_only():
    """Long job finishing past the horizon: utilization counted in-window only,
    makespan still covers the drain (amendment C)."""
    jobs = [mkjob("A", 0.0, 100.0, share=1.0)]
    result = run_sim(jobs, horizon=10.0)
    assert result.summary["cluster"]["utilization_arrival_window"] == pytest.approx(1.0)
    assert result.summary["cluster"]["makespan_s"] == 100.0
    assert result.summary["cluster"]["gpu_hours_total"] == pytest.approx(100.0 / 3600.0)


def test_group_split_metrics():
    trace = [mkjob("t0", 0.0, 10.0, group="trace")]
    overlay = [mkjob("o0", 1.0, 5.0, group="overlay", job_type="online_inference", slo_s=2.0)]
    result = run_sim(trace + overlay)
    groups = result.summary["groups"]
    assert set(groups) == {"trace", "overlay"}
    assert groups["trace"]["n_jobs"] == 1 and groups["overlay"]["n_jobs"] == 1


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------

def test_overlay_generation_properties():
    cfg = OverlayConfig(share=0.25)
    jobs, manifest = generate_overlay(n_trace_jobs=400, t_sim_s=1000.0, cfg=cfg, seed=5)
    assert len(jobs) == 100
    for j in jobs:
        assert j.group == "overlay" and j.job_type == "online_inference"
        assert j.synthetic_all()
        assert 5.0 <= j.duration_s <= 30.0
        assert 0.1 <= j.num_gpus <= 0.5
        assert j.priority in (9, 10)
        assert j.slo_s in (1.0, 2.0, 5.0, 10.0)
        assert 0.0 <= j.submit_ts <= 1000.0
        assert j.deadline_ts == j.submit_ts + j.slo_s
    assert manifest["n_jobs"] == 100 and manifest["share"] == 0.25


def test_overlay_slo_clock_includes_load():
    """SLO met iff start + load_s - submit <= slo_s (decision B)."""
    blocker = mkjob("blk", 0.0, 4.0, share=1.0, vram=4.0)
    late = mkjob("late", 0.0, 1.0, share=0.5, vram=1.0, job_type="online_inference",
                 slo_s=5.0, load_s=2.0, group="overlay")
    early = mkjob("early", 0.0, 1.0, share=0.5, vram=1.0, job_type="online_inference",
                  slo_s=5.0, load_s=0.0, group="overlay")
    cluster = ClusterSpec.homogeneous(1, vram_gb=8.0)
    # FIFO strict: blk head -> ov, ov2 wait until 4.0
    result = run_sim([blocker, late, early], cluster=cluster)
    assert result.records["late"].start_ts == 4.0
    assert result.records["late"].slo_met is False     # 4 + 2 - 0 = 6 > 5
    assert result.records["early"].start_ts == 4.0
    assert result.records["early"].slo_met is True      # 4 + 0 - 0 = 4 <= 5


def test_overlay_load_reported_separately():
    trace = [mkjob("t0", 0.0, 100.0)]
    overlay, _ = generate_overlay(10, 200.0, OverlayConfig(share=1.0), seed=3)
    result = run_sim(trace + overlay, horizon=200.0)
    groups = result.summary["groups"]
    assert groups["trace"]["realized_load"] is not None
    assert groups["overlay"]["realized_load"] is not None
    total = groups["trace"]["gpu_hours_in_window"] + groups["overlay"]["gpu_hours_in_window"]
    assert total <= 1 * 200.0 / 3600.0 + 1e-9  # 1 GPU


# --------------------------------------------------------------------------
# Replayer
# --------------------------------------------------------------------------

def _tiny_unified_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "dataset": "unittest", "gpu_type_req": None, "cpu_cores": 0.0, "mem_gb": 0.0,
        "user_group": None, "final_status": "pass",
        "vram_gb_src": "derived", "job_type_src": "derived", "priority_src": "synthetic",
        "deadline_src": "synthetic", "runtime_est_src": "derived",
    }
    defaults = {"duration_s": 100.0, "num_gpus": 1.0, "vram_gb": 4.0, "priority": 5,
                "job_type": "training"}
    full = []
    for r in rows:
        row = {**defaults, **base, **r}
        if "deadline_ts" not in row:
            row["deadline_ts"] = row["submit_ts"] + 3.0 * row["duration_s"]
        if "runtime_est_s" not in row:
            row["runtime_est_s"] = 2.0 * row["duration_s"]
        full.append(row)
    return pd.DataFrame(full, columns=COLUMNS)


@pytest.fixture()
def tiny_parquet(tmp_path):
    df = _tiny_unified_df([
        {"job_id": str(i), "submit_ts": float(100 * i), "duration_s": 100.0}
        for i in range(4)
    ] + [  # infeasible on a 1-GPU cluster
        {"job_id": "big", "submit_ts": 50.0, "duration_s": 100.0, "num_gpus": 5.0, "vram_gb": 20.0},
        {"job_id": "fat", "submit_ts": 60.0, "duration_s": 100.0, "vram_gb": 99.0},
    ])
    path = tmp_path / "tiny.parquet"
    df.to_parquet(path)
    return path


def test_replayer_window_scale_and_drops(tiny_parquet):
    """4 feasible jobs of 100 s each on 1 GPU at rho=0.8: W=400 gpu-s,
    T_sim = 400/0.8 = 500 s; the window clamps to the 300 s trace span, so
    submits 0/100/200/300 -> 0/166.67/333.33/500 (scale 500/300)."""
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    jobs, manifest = load_trace_window(
        tiny_parquet, cluster, TraceConfig(window_s=400.0, rho=0.8, est_mode="stored"), seed=1)
    assert manifest["dropped"] == {"too_many_gpus": 1, "vram_too_big": 1, "nonpositive_duration": 0}
    assert manifest["n_jobs"] == 4
    assert manifest["work_gpu_s"] == pytest.approx(400.0)
    assert manifest["t_sim_s"] == pytest.approx(500.0)
    assert manifest["arrival_scale_factor"] == pytest.approx(300.0 / 500.0)
    assert [j.submit_ts for j in jobs] == pytest.approx([0.0, 500 / 3, 1000 / 3, 500.0])
    assert jobs[0].job_id == "unittest::0"


def test_replayer_preserves_deadline_slack_in_seconds(tiny_parquet):
    """Amendment B: deadline_sim - submit_sim == deadline_ts - submit_ts exactly,
    while submit times are rescaled."""
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    raw = pd.read_parquet(tiny_parquet)
    jobs, _ = load_trace_window(
        tiny_parquet, cluster, TraceConfig(window_s=400.0, rho=0.8, est_mode="stored"), seed=1)
    kept = {j.job_id.split("::")[1]: j for j in jobs}
    for _, row in raw.iterrows():
        if row["job_id"] in kept:
            j = kept[row["job_id"]]
            assert j.deadline_ts - j.submit_ts == pytest.approx(
                row["deadline_ts"] - row["submit_ts"], abs=1e-9)
            assert j.submit_ts == pytest.approx(row["submit_ts"] * 500.0 / 300.0)


def test_replayer_offered_work_counts_occupied_gpus(tmp_path):
    """1.5-GPU job (ceiled to 2) and a 0.5-share job on 2 GPUs: W = 2*dur + 0.5*dur."""
    df = _tiny_unified_df([
        {"job_id": "frac", "submit_ts": 0.0, "duration_s": 100.0, "num_gpus": 1.5, "vram_gb": 20.0},
        {"job_id": "share", "submit_ts": 1.0, "duration_s": 100.0, "num_gpus": 0.5, "vram_gb": 4.0},
    ])
    path = tmp_path / "frac.parquet"
    df.to_parquet(path)
    cluster = ClusterSpec.homogeneous(2, vram_gb=24.0)
    jobs, manifest = load_trace_window(
        path, cluster, TraceConfig(window_s=10.0, rho=0.5, est_mode="stored"), seed=1)
    assert manifest["n_fractional_ceiled"] == 1
    assert manifest["work_gpu_s"] == pytest.approx(2.0 * 100.0 + 0.5 * 100.0)
    assert jobs[0].n_occupied == 2 and jobs[0].share_per_gpu == 1.0


def test_replayer_estimate_modes(tiny_parquet):
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    base = TraceConfig(window_s=400.0, rho=0.8)
    stored, m1 = load_trace_window(tiny_parquet, cluster, TraceConfig(**{**base.__dict__, "est_mode": "stored"}), seed=1)
    oracle, _ = load_trace_window(tiny_parquet, cluster, TraceConfig(**{**base.__dict__, "est_mode": "oracle"}), seed=1)
    noisy, _ = load_trace_window(tiny_parquet, cluster, TraceConfig(**{**base.__dict__, "est_mode": "noisy", "noise_sigma": 0.25}), seed=1)
    noisy2, _ = load_trace_window(tiny_parquet, cluster, TraceConfig(**{**base.__dict__, "est_mode": "noisy", "noise_sigma": 0.25}), seed=1)
    assert all(j.runtime_est_s == 200.0 and j.runtime_est_src == "derived" for j in stored)
    assert all(j.runtime_est_s == j.duration_s and j.runtime_est_src == "synthetic" for j in oracle)
    assert all(j.runtime_est_s > 0 and j.runtime_est_src == "synthetic" for j in noisy)
    assert [j.runtime_est_s for j in noisy] == [j.runtime_est_s for j in noisy2]  # seeded


def test_replayer_realized_load_hits_target(tiny_parquet):
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    jobs, manifest = load_trace_window(
        tiny_parquet, cluster, TraceConfig(window_s=400.0, rho=0.8), seed=1)
    result = run_sim(jobs, cluster=cluster, horizon=manifest["t_sim_s"])
    realized = result.summary["groups"]["trace"]["realized_load"]
    # 4 jobs x 100 s on 1 GPU, no waiting; the last job (submit 500 == horizon)
    # runs entirely past the arrival window, so in-window work is 300/500 = 0.6
    # (amendment C: load is measured over the arrival window only).
    assert realized == pytest.approx(0.6, abs=0.01)


def test_replayer_deterministic(tiny_parquet):
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    cfg = TraceConfig(window_s=400.0, rho=0.8)
    a, ma = load_trace_window(tiny_parquet, cluster, cfg, seed=9)
    b, mb = load_trace_window(tiny_parquet, cluster, cfg, seed=9)
    assert a == b and ma == mb


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------

def test_max_decisions_guard():
    cfg = EngineConfig(max_decisions_per_timestep=2)
    jobs = [mkjob(f"j{i}", 0.0, 10.0) for i in range(5)]
    cluster = ClusterSpec.homogeneous(8, vram_gb=24.0)
    result = run_sim(jobs, cluster=cluster, config=cfg)
    assert any("max-decisions guard" in w for w in result.warnings)
    # run still completes: everything drains via later decision points
    assert all(r.finish_ts is not None for r in result.records.values())


def test_policy_must_return_feasible_pending_job():
    class BadPolicy:
        name = "bad"
        def decide(self, view):
            return "nonexistent-job"

    engine = Engine([mkjob("A", 0.0, 5.0)], ClusterSpec.homogeneous(1), BadPolicy())
    with pytest.raises(ValueError, match="unknown/non-pending"):
        engine.run()


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------

def test_synthetic_generator_valid_schema_and_seeded():
    cfg = SyntheticConfig(n_jobs=300, rate_per_s=0.5)
    df1 = generate_workload(cfg, seed=42)
    df2 = generate_workload(cfg, seed=42)
    assert df1.equals(df2)
    assert validate_jobs(df1) == []
    assert set(df1["job_type"]) <= {"training", "batch_inference", "online_inference"}


def test_synthetic_poisson_rate():
    cfg = SyntheticConfig(n_jobs=4000, mode="poisson", rate_per_s=1.0,
                          job_mix={"training": 1.0})
    df = generate_workload(cfg, seed=3)
    span = df["submit_ts"].max()
    assert 3000 < span < 6000  # ~n/rate within sampling noise


def test_synthetic_bursty_clusters_arrivals():
    """MMPP: per-bucket count CV^2 well above the Poisson ~1."""
    import numpy as np
    cfg = SyntheticConfig(n_jobs=4000, mode="bursty", rate_per_s=0.2,
                          bursty_rate_high=4.0, bursty_mean_stay_s=50.0,
                          job_mix={"training": 1.0})
    df = generate_workload(cfg, seed=3)
    counts, _ = np.histogram(df["submit_ts"].to_numpy(), bins=20)
    cv2_bursty = counts.var() / max(counts.mean(), 1e-9)
    poisson = generate_workload(
        SyntheticConfig(n_jobs=4000, mode="poisson", rate_per_s=df["submit_ts"].max() / 4000,
                        job_mix={"training": 1.0}), seed=3)
    pcounts, _ = np.histogram(poisson["submit_ts"].to_numpy(), bins=20)
    cv2_poisson = pcounts.var() / max(pcounts.mean(), 1e-9)
    assert cv2_bursty > 2.0 * max(cv2_poisson, 0.5)


# --------------------------------------------------------------------------
# Views / leakage structure
# --------------------------------------------------------------------------

def test_decision_view_has_no_outcome_fields():
    fields = PendingView.__dataclass_fields__.keys()
    forbidden = {"duration_s", "final_status"}
    assert not (fields & forbidden)
    view_fields_ok = {"runtime_est_s"} <= fields  # estimate yes, actual no
    assert view_fields_ok


def test_view_pending_cap_and_truncation():
    class NeverAdmit:  # freezes the queue so the view is deterministic
        name = "never"
        def decide(self, view):
            return None

    jobs = [mkjob(f"j{i}", 0.0, 10.0) for i in range(100)]
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    engine = Engine(jobs, cluster, NeverAdmit(),
                    config=EngineConfig(view_pending_cap=10), arrival_horizon=0.0)
    while len(engine.state.queue):
        engine.step()
    view = engine.build_view()
    assert view.n_pending == 100
    assert len(view.pending) == 10
    assert view.n_truncated == 90
    assert view.pending[0].job_id == "j0" and view.pending[0].fits is True


def test_engine_smoke_on_real_trace_window():
    """One small real-data cell end-to-end (skipped if the parquet is absent)."""
    path = DATA / "processed" / "alibaba2020.parquet"
    if not path.exists():
        pytest.skip("alibaba2020.parquet not downloaded")
    cluster = ClusterSpec.from_yaml(CONFIGS / "cluster_small.yaml")
    jobs, manifest = load_trace_window(
        path, cluster, TraceConfig(window_s=3600.0, rho=0.5, max_jobs=300), seed=123)
    result = run_sim(jobs, cluster=cluster, horizon=manifest["t_sim_s"])
    assert not result.warnings
    assert all(r.finish_ts is not None for r in result.records.values())
    total_gpu_hours = sum(r.gpu_hours or 0.0 for r in result.records.values())
    cap = cluster.n_gpus * result.summary["cluster"]["makespan_s"] / 3600.0
    assert total_gpu_hours <= cap + 1e-9
