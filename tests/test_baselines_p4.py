"""P4 policy tests: hand-solved golden cases (PLAN.md §10), the frozen cap
filter, EASY reservation semantics, determinism and the harness plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gde.baselines import ALL_POLICIES
from gde.baselines.easy_backfill import EasyBackfillPolicy
from gde.simulator.cluster import ClusterSpec
from gde.simulator.engine import Engine
from gde.simulator.events import EventKind
from gde.simulator.workload import TraceConfig, load_trace_window, placement_shape

from tests.test_simulator import _tiny_unified_df, mkjob, run_sim  # reuse helpers


def run_policy(jobs, policy, cluster=None, horizon=None):
    return run_sim(jobs, cluster=cluster, policy=policy, horizon=horizon)


def test_priority_picks_highest_priority_that_fits():
    """FIFO head is low priority and fits; a later high-priority job fits too —
    priority skips the head and runs the high-priority job first."""
    jobs = [
        mkjob("low", 0.0, 50.0, share=0.5, vram=4.0, priority=2),
        mkjob("high", 1.0, 10.0, share=0.5, vram=4.0, priority=9),
    ]
    r = run_policy(jobs, ALL_POLICIES["priority"]())
    assert r.records["high"].start_ts == 1.0
    assert r.records["low"].start_ts == 0.0  # low started before high arrived
    # after both running, nothing pending -> completion order high (11) < low (50)


def test_priority_respects_fitting():
    """1 GPU. At t=0 all three fit: priority runs big (prio 9) first; the
    others then no longer fit (share 0.9 used). At t=10: small (prio 3) is
    picked over blocker (prio 1), and blocker (0.25+0.75=1.0, 2+20=22 GB)
    still fits alongside it."""
    jobs = [
        mkjob("small", 0.0, 10.0, share=0.25, vram=2.0, priority=3),
        mkjob("big", 0.0, 10.0, share=0.9, vram=20.0, priority=9),
        mkjob("blocker", 0.0, 40.0, share=0.75, vram=20.0, priority=1),
    ]
    r = run_policy(jobs, ALL_POLICIES["priority"]())
    assert r.records["big"].start_ts == 0.0
    assert r.records["small"].start_ts == 10.0
    assert r.records["blocker"].start_ts == 10.0


def test_sjf_uses_estimate_not_actual():
    """SJF must order by runtime_est_s; job B (longer actual, shorter EST)
    runs before job A."""
    a = mkjob("A", 0.0, 10.0, share=1.0)
    b = mkjob("B", 0.0, 60.0, share=1.0)
    # give B the smaller estimate (swap the defaults)
    from dataclasses import replace
    b = replace(b, runtime_est_s=5.0)
    a = replace(a, runtime_est_s=50.0)
    r = run_policy([a, b], ALL_POLICIES["sjf"]())
    assert r.records["B"].start_ts == 0.0
    assert r.records["A"].start_ts == 60.0  # actual durations drive finish times


def test_edf_meets_earliest_deadline_first():
    j1 = mkjob("late", 0.0, 10.0, share=0.5, deadline_slack=8.0)
    j2 = mkjob("early", 0.0, 10.0, share=0.5, deadline_slack=2.0)
    r = run_policy([j1, j2], ALL_POLICIES["edf"]())
    assert r.records["early"].start_ts == 0.0
    assert r.records["late"].start_ts == 0.0  # both fit simultaneously
    # force serialization: full-share jobs -> EDF order visible in start times
    j1 = mkjob("late", 0.0, 10.0, share=1.0, deadline_slack=8.0)
    j2 = mkjob("early", 0.0, 10.0, share=1.0, deadline_slack=2.0)
    r = run_policy([j1, j2], ALL_POLICIES["edf"]())
    assert r.records["early"].start_ts == 0.0
    assert r.records["late"].start_ts == 10.0


def test_best_fit_prefers_snug_job():
    """2 GPUs. At t=0 both occupiers fit: best_fit admits occupier2 (22 GB,
    snugness 2) first -> placed gpu0 (leftover 2); occupier (14 GB) -> gpu1
    (leftover 10). At t=1 the snug 2-GB job (leftover 0 on gpu0) is preferred
    over the looser 4-GB job; both are admitted."""
    jobs = [
        mkjob("occupier", 0.0, 100.0, share=0.1, vram=14.0),
        mkjob("occupier2", 0.0, 100.0, share=0.1, vram=22.0),
        mkjob("loose", 1.0, 10.0, share=0.1, vram=4.0, priority=1),
        mkjob("snug", 1.0, 10.0, share=0.1, vram=2.0, priority=9),
    ]
    cluster = ClusterSpec.homogeneous(2, vram_gb=24.0)
    r = run_policy(jobs, ALL_POLICIES["best_fit"](), cluster=cluster)
    assert r.records["snug"].start_ts == 1.0
    assert r.records["loose"].start_ts == 1.0
    assert r.records["snug"].gpu_ids == (0,)   # filled the 2 GB gap on gpu0
    assert r.records["loose"].gpu_ids == (1,)


def test_easy_backfills_when_head_blocked():
    """Head (whole GPU, long) is blocked by a running job; a short job may
    backfill now because its estimated finish precedes the head reservation."""
    jobs = [
        mkjob("running", 0.0, 100.0, share=1.0, vram=4.0),     # runs 0..100
        mkjob("head", 1.0, 50.0, share=1.0, vram=4.0),         # FIFO head, waits
        mkjob("small", 2.0, 10.0, share=0.25, vram=1.0),
    ]
    r = run_policy(jobs, ALL_POLICIES["easy_backfill"]())
    # 'small' cannot backfill onto the GPU share-wise (running holds share 1.0).
    # At t=100 'running' frees the GPU but the FIFO head takes it first; small
    # waits behind head exactly like FIFO -> 150.
    assert r.records["small"].start_ts == 150.0
    assert r.records["head"].start_ts == 100.0

    # now make the running job a fractional one so backfill share exists
    jobs = [
        mkjob("running", 0.0, 100.0, share=0.5, vram=4.0),
        mkjob("head", 1.0, 50.0, share=0.9, vram=4.0),         # blocked (0.5+0.9>1)
        mkjob("small", 2.0, 10.0, share=0.25, vram=1.0),       # fits 0.5+0.25<=1
    ]
    r = run_policy(jobs, ALL_POLICIES["easy_backfill"]())
    # head reservation est: running frees at 100 -> head at 100; small est finish
    # 2+10=12 <= 100 -> backfill at t=2
    assert r.records["small"].start_ts == 2.0
    assert r.records["head"].start_ts == 100.0


def test_easy_never_delays_head_reservation():
    """A backfill candidate whose ESTIMATED finish exceeds the reservation is
    rejected (head not delayed), even though it fits now."""
    jobs = [
        mkjob("running", 0.0, 20.0, share=0.5, vram=4.0),      # frees at 20
        mkjob("head", 1.0, 5.0, share=0.9, vram=4.0),          # reserved at 20
        mkjob("longbf", 2.0, 50.0, share=0.25, vram=1.0),      # est finish 52 > 20
    ]
    r = run_policy(jobs, ALL_POLICIES["easy_backfill"]())
    # longbf was NOT allowed to backfill before the reservation; at t=20 the
    # head starts (runs to 25), longbf follows on the freed GPU.
    assert r.records["head"].start_ts == 20.0
    assert r.records["longbf"].start_ts == 25.0


def test_easy_reservation_uses_estimates_not_actuals():
    from dataclasses import replace
    running = replace(mkjob("running", 0.0, 30.0, share=0.5, vram=4.0),
                      runtime_est_s=10.0)                       # est finish 10, actual 30
    head = mkjob("head", 1.0, 5.0, share=0.9, vram=4.0)
    small = mkjob("small", 2.0, 4.0, share=0.25, vram=1.0)      # est finish 6
    jobs = [running, head, small]
    r = run_policy(jobs, ALL_POLICIES["easy_backfill"]())
    # reservation believes GPUs free at t=10; small (est finish 6 <= 10) backfills at 2
    assert r.records["small"].start_ts == 2.0
    # head still starts when running ACTUALLY finishes: 30
    assert r.records["head"].start_ts == 30.0


def test_all_policies_deterministic_and_drain(tmp_path):
    """Every policy: two runs -> identical digest; all jobs finish; invariants
    hold on a mixed random-ish workload."""
    rng = np.random.default_rng(99)
    jobs = []
    for i in range(60):
        jobs.append(mkjob(
            f"j{i}", float(rng.integers(0, 200)),
            float(rng.integers(1, 120)),
            share=float(rng.choice([0.1, 0.25, 0.5, 1.0, 2.0])),
            vram=float(rng.integers(1, 8)),
            priority=int(rng.integers(1, 11))))
    cluster = ClusterSpec.homogeneous(2, vram_gb=12.0)
    for name, policy_cls in ALL_POLICIES.items():
        a = run_sim(jobs, cluster=cluster, policy=policy_cls(), horizon=200.0)
        b = run_sim(jobs, cluster=cluster, policy=policy_cls(), horizon=200.0)
        assert a.digest == b.digest, name
        assert not a.warnings, (name, a.warnings)
        assert all(r.finish_ts is not None for r in a.records.values()), name
        total = sum(r.gpu_hours or 0.0 for r in a.records.values())
        cap = cluster.n_gpus * a.summary["cluster"]["makespan_s"] / 3600.0
        assert total <= cap + 1e-9, name


def test_cap_filter_drops_before_scaling(tmp_path):
    """The frozen 6 h cap: drops counted, applied inside the window BEFORE
    W/time-scaling (T_sim computed from capped work)."""
    rows = [{"job_id": f"j{i}", "submit_ts": float(i * 10), "duration_s": d}
            for i, d in enumerate([100.0, 200.0, 300.0, 25000.0, 40000.0])]
    path = tmp_path / "cap.parquet"
    _tiny_unified_df(rows).to_parquet(path)
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    cfg_cap = TraceConfig(window_s=400.0, rho=0.8, duration_cap_s=21600.0)
    cfg_none = TraceConfig(window_s=400.0, rho=0.8)
    jobs, m = load_trace_window(path, cluster, cfg_cap, seed=1)
    _, m0 = load_trace_window(path, cluster, cfg_none, seed=1)
    assert m["n_cap_dropped"] == 2
    assert m["n_jobs"] == 3
    assert m["work_gpu_s"] == pytest.approx(600.0)
    assert m["t_sim_s"] == pytest.approx(600.0 / 0.8)
    assert all(j.duration_s <= 21600.0 for j in jobs)
    assert m0["n_cap_dropped"] == 0 and m0["n_jobs"] == 5


def test_policies_never_see_actual_durations():
    """Decision views expose estimates; the leakage split (P5 hardens this)."""
    from gde.simulator.engine import PendingView, RunningView
    pending_fields = set(PendingView.__dataclass_fields__)
    running_fields = set(RunningView.__dataclass_fields__)
    assert "duration_s" not in pending_fields
    assert "duration_s" not in running_fields
    assert "runtime_est_s" in pending_fields and "runtime_est_s" in running_fields


def test_engine_runs_all_policies_on_capped_regime_smoke():
    """One tiny capped-1h-like real cell per policy (skips without data).
    Uses a benchmark seed known to have a non-empty window (Amendment 1)."""
    from gde.paths import DATA
    path = DATA / "processed" / "alibaba2020.parquet"
    if not path.exists():
        pytest.skip("alibaba2020.parquet not downloaded")
    from gde.evaluation.benchmark import load_regimes, regime_spec, make_overlay_cfg
    import yaml
    from gde.paths import CONFIGS
    from gde.simulator.workload import generate_overlay
    seed = 31002  # non-empty window per Amendment 1 measurement
    regimes = load_regimes()
    spec = regime_spec(regimes, "capped-1h")
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    jobs, tman = load_trace_window(
        path, cluster,
        TraceConfig(window_s=3600.0, rho=0.6, max_jobs=150,
                    duration_cap_s=float(spec["duration_cap_s"])), seed=seed)
    overlay_raw = yaml.safe_load((CONFIGS / "sim_p3.yaml").read_text(encoding="utf-8"))["overlay"]
    overlay, _ = generate_overlay(len(jobs), tman["t_sim_s"],
                                  make_overlay_cfg(overlay_raw, 0.25), seed=seed)
    for name, policy_cls in ALL_POLICIES.items():
        r = run_sim(jobs + overlay, cluster=cluster, policy=policy_cls(),
                    horizon=tman["t_sim_s"])
        assert not r.warnings, name
        assert all(rec.finish_ts is not None for rec in r.records.values()), name
