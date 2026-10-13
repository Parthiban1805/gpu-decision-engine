"""P6 gate tests: label determinism, soft targets sum to 1, TP true-vs-est,
J window math, dataset schema."""

from __future__ import annotations

import math

import pytest

from gde.baselines import ALL_POLICIES
from gde.labeling.objective import load_preset, terminal_penalty, window_objective
from gde.labeling.rollout import label_state
from gde.simulator.cluster import ClusterSpec
from gde.simulator.engine import Engine, EventKind
from gde.simulator.snapshot import snapshot

from tests.test_simulator import mkjob


def _state_with_queue(seed: int = 0):
    import numpy as np
    rng = np.random.default_rng(seed)
    jobs = [mkjob(f"warm_{i}", i * 0.1, 0.05) for i in range(60)]
    jobs += [mkjob("blk", 19.95, 40.0, share=0.5, vram=4.0) for _ in range(2)]
    jobs.append(mkjob("head", 20.0, 8.0, share=1.0, vram=20.0, priority=2))
    for i in range(12):
        jobs.append(mkjob(
            f"q_{i}", 20.0 + 1e-6 + i * 1e-6, float(rng.integers(1, 20)),
            share=float(rng.choice([0.1, 0.25, 0.5])),
            vram=float(rng.integers(1, 5)),
            priority=int(rng.integers(1, 11))))
    cluster = ClusterSpec.homogeneous(2, vram_gb=24.0)
    engine = Engine(jobs, cluster, ALL_POLICIES["fifo"](), arrival_horizon=20.0)
    while len(engine.state.queue) and engine.state.now <= 20.1:
        engine.step()
    return snapshot(engine.state)


CONST = load_preset("balanced")


def test_labels_deterministic():
    state = _state_with_queue()
    a = label_state(state, CONST, tau=0.5, n_arrivals=5, max_seconds=500.0, k=8)
    b = label_state(state, CONST, tau=0.5, n_arrivals=5, max_seconds=500.0, k=8)

    def content(labels):
        return (labels.candidates, labels.job_ids, labels.j_values, labels.regret,
                labels.hard_label, labels.soft_target,
                tuple((o.action_id, o.j, tuple(sorted(o.components.items())))
                      for o in labels.outcomes))

    assert content(a) == content(b)  # timing fields excluded (instrumentation)


def test_soft_targets_sum_to_one_and_hard_is_argmin():
    labels = label_state(_state_with_queue(), CONST, tau=0.7,
                         n_arrivals=5, max_seconds=500.0, k=8)
    assert len(labels.soft_target) == len(labels.candidates)
    assert sum(labels.soft_target) == pytest.approx(1.0, abs=1e-12)
    assert all(p >= 0 for p in labels.soft_target)
    j_min = min(labels.j_values)
    assert labels.j_values[list(labels.candidates).index(labels.hard_label)] == j_min
    # argmin gets the max soft mass
    assert labels.soft_target[list(labels.candidates).index(labels.hard_label)] == \
        max(labels.soft_target)


def test_run_beats_wait_when_action_matters():
    """1 GPU, two fitting jobs at t=0: a (long, loose deadline), b (short,
    tight deadline). WAIT lets the EDF rollout run b first (earlier deadline);
    RUN:a forces a first, delaying b past its deadline -> higher J.
    RUN:b equals WAIT (same trajectory as the rollout's own choice)."""
    jobs = [mkjob("a", 0.0, 100.0, share=1.0, vram=4.0, priority=5, deadline_slack=20.0),
            mkjob("b", 0.0, 10.0, share=1.0, vram=4.0, priority=9, deadline_slack=2.0)]
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    engine = Engine(jobs, cluster, ALL_POLICIES["fifo"](), arrival_horizon=0.0)
    engine.step()  # ARRIVE a
    state = snapshot(engine.state)
    labels = label_state(state, CONST, tau=0.5, n_arrivals=2, max_seconds=400.0, k=8)
    j = {c: v for c, v in zip(labels.candidates, labels.j_values)}
    assert "RUN:a" in j and "WAIT" in j
    assert j["RUN:a"] > j["WAIT"] + 1e-9      # forcing a hurts (b misses deadline)
    if "RUN:b" in j:
        assert j["RUN:b"] == pytest.approx(j["WAIT"])  # rollout would pick b anyway


def test_terminal_penalty_true_vs_estimate():
    """Estimate-based TP differs from true TP exactly when est != duration."""
    from dataclasses import replace
    state = _state_with_queue()
    tp_true = terminal_penalty(state, state.now, est_based=False)
    # scale every estimate by 2x -> est TP roughly doubles for pending jobs
    mutated = snapshot(state)
    for jid in list(mutated.jobs):
        job = mutated.jobs[jid]
        mutated.jobs[jid] = replace(job, runtime_est_s=job.duration_s * 2.0)
    tp_est2 = terminal_penalty(mutated, mutated.now, est_based=True)
    assert tp_true > 0
    assert tp_est2 == pytest.approx(2.0 * tp_true, rel=0.35)  # pending-dominated


def test_projected_lateness_hand_case():
    """Amendment 4: pending job with deadline 100 s, est remaining 50 s,
    horizon end t=80 -> lateness 30 s; est-only (actual duration irrelevant);
    running-job variant subtracts elapsed; no lateness when deadline is far."""
    from dataclasses import replace

    from gde.labeling.objective import projected_lateness_penalty
    pending = mkjob("p", 0.0, 999.0, share=0.5, deadline_slack=100.0 / 999.0)
    pending = replace(pending, runtime_est_s=50.0, deadline_ts=100.0)
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0)
    engine = Engine([pending], cluster, ALL_POLICIES["fifo"](), arrival_horizon=0.0)
    engine.step()  # ARRIVE p
    state = snapshot(engine.state)
    # pending: H_end 80 + est 50 - deadline 100 = 30 s of lateness
    assert projected_lateness_penalty(state, 80.0) == pytest.approx(
        pending.priority * 30.0 / 3600.0)
    assert projected_lateness_penalty(state, 40.0) == pytest.approx(0.0)  # 40+50<=100
    # actual duration must not matter (estimates only)
    mutated = replace(state.jobs["p"], duration_s=12345.0)
    import dataclasses
    state2 = snapshot(state)
    state2.jobs["p"] = mutated
    assert projected_lateness_penalty(state2, 80.0) == \
        pytest.approx(projected_lateness_penalty(state, 80.0))


def test_window_objective_math_hand_case():
    """Single 0.5-share, 1 h job finishing inside the window: C/V/T/F sanity."""
    jobs = [mkjob("a", 0.0, 3600.0, share=0.5, vram=4.0)]
    cluster = ClusterSpec.homogeneous(1, vram_gb=24.0, p_idle_w=0.0, p_peak_w=1000.0)
    result = Engine(jobs, cluster, ALL_POLICIES["fifo"](), arrival_horizon=0.0).run()
    comp = window_objective(result.state, 0.0, 7200.0, CONST)
    # energy: 0.5 share -> 500 W for 3600 s = 0.5 kWh (p_idle = 0); the GPU is
    # half idle during the job and fully idle for the second hour -> 1.5
    # idle-hours; C = (0.5*price + 1.5*idle_penalty) / (1*2h)
    expected_c = (0.5 * CONST.energy_price_per_kwh
                  + 1.5 * CONST.idle_gpu_penalty_per_hour) / 2.0
    assert comp["C"] == pytest.approx(expected_c, rel=0.02)
    assert comp["n_active"] == 1
    assert comp["T"] == pytest.approx(5.0 * 1.0 / 1.0, rel=0.01)  # prio 5 x 1 h / 1 job


def test_parquet_schema_roundtrip(tmp_path):
    from gde.labeling.dataset_builder import apply_tau, tau_from_labels, to_parquet
    state = _state_with_queue()
    rec = {
        "state_id": "s1", "split": "train", "seed": 1, "scheduler": "fifo",
        "t_frac": 0.5, "qlen": 3, "edf_pick": None, "sjf_pick": None,
        "fifo_pick": None, "candidates": ["RUN:a", "WAIT"],
        "job_ids": ["a", None], "j_values": [1.0, 2.0], "n_feasible": 1,
        "n_pending": 3,
        "components": [{"tp_priority_hours": 0.1, "C": 0.1, "V": 0.0, "T": 0.3, "F": 0.0}] * 2,
    }
    tau = tau_from_labels([rec])
    apply_tau([rec], tau)
    df = to_parquet([rec], tau, tmp_path / "x.parquet")
    assert len(df) == 2 and df["soft_target"].sum() == pytest.approx(1.0)
    assert df["hard_label"].nunique() == 1 and df["hard_label"].iloc[0] == "RUN:a"
    import pandas as pd
    assert isinstance(pd.read_parquet(tmp_path / "x.parquet"), pd.DataFrame)
