"""Discrete-event engine: custom heapq loop (PLAN.md §4.2), no SimPy.

The engine has NO internal randomness — every stochastic input (trace window,
estimate noise, overlay arrivals) is baked into the job list by seeded
generators before construction. Same jobs + same cluster + same policy =>
byte-identical event log, and snapshots fork by copying containers only.

Decision-time information only: the policy receives a DecisionView whose job
fields exclude outcomes (duration_s, final_status). Actual durations live on
SimJob but never cross into the view; the leakage split is enforced by test
(P5 hardens it).

Event flow per instant: state-changing events (ARRIVE/START/FINISH) settle
first, then DECISION_POINT handlers run; each decision point admits jobs until
the policy returns WAIT, nothing is feasible, or the max-decisions guard trips.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Protocol

from gde.simulator.cluster import EPS, ClusterSpec, ClusterState, Placement
from gde.simulator.events import Event, EventKind, EventQueue, event_log_digest
from gde.simulator.metrics import ClusterSample, JobRecord, summarize
from gde.simulator.workload import SimJob


class Policy(Protocol):
    name: str

    def decide(self, view: "DecisionView") -> str | None:
        """Return the job_id of a pending job to admit now, or None for WAIT."""
        ...


@dataclass(frozen=True)
class EngineConfig:
    max_decisions_per_timestep: int = 1000
    view_pending_cap: int = 64    # fits flags computed for at most this many pending jobs


@dataclass(frozen=True)
class PendingView:
    """Decision-time view of one pending job (no outcome fields)."""
    job_id: str
    group: str
    job_type: str
    priority: int
    submit_ts: float
    wait_s: float
    runtime_est_s: float
    num_gpus: float
    per_gpu_vram: float
    deadline_ts: float
    slo_s: float | None
    load_s: float
    gpu_type_req: str | None
    fits: bool
    arrival_index: int


@dataclass(frozen=True)
class GpuView:
    gpu_id: int
    type: str
    vram_gb: float
    compute_factor: float
    p_idle_w: float
    p_peak_w: float
    used_share: float
    used_vram: float
    free_share: float
    free_vram: float


@dataclass(frozen=True)
class RunningView:
    """Decision-time view of a running job. est_finish_ts uses runtime_est_s
    (the feature), never the actual duration — no leakage into policies."""
    job_id: str
    gpu_ids: tuple[int, ...]
    start_ts: float
    est_finish_ts: float
    share_per_gpu: float
    per_gpu_vram: float
    priority: int
    job_type: str
    runtime_est_s: float


@dataclass(frozen=True)
class DecisionView:
    now: float
    n_pending: int
    n_truncated: int
    pending: tuple[PendingView, ...]
    head_job_id: str | None
    head_fits: bool
    gpus: tuple[GpuView, ...]
    running: tuple[RunningView, ...]
    n_running: int
    power_budget_w: float = 0.0   # 0 = unconstrained


@dataclass
class SimState:
    """Everything that evolves; snapshot() = copy of these containers."""
    cluster: ClusterState
    jobs: dict[str, SimJob]
    pending: dict[str, None]                      # FIFO order (insertion-ordered)
    records: dict[str, JobRecord]
    queue: EventQueue
    log: list[str]
    samples: list[ClusterSample]
    warnings: list[str]
    now: float = 0.0
    seq: int = 0
    events_processed: int = 0

    def copy(self) -> "SimState":
        return SimState(
            cluster=self.cluster.copy(),
            jobs=dict(self.jobs),                 # SimJobs are frozen -> shared
            pending=dict(self.pending),
            records={k: (r if r.final else r.clone()) for k, r in self.records.items()},
            queue=self.queue.copy(),
            log=list(self.log),
            samples=list(self.samples),
            warnings=list(self.warnings),
            now=self.now, seq=self.seq, events_processed=self.events_processed,
        )

    def digest(self) -> str:
        return event_log_digest(self.log)


@dataclass
class RunResult:
    manifest: dict
    log: list[str]
    records: dict[str, JobRecord]
    state: SimState
    summary: dict
    digest: str
    wall_s: float
    warnings: list[str]

    @property
    def events_per_s(self) -> float:
        return self.state.events_processed / self.wall_s if self.wall_s > 0 else float("nan")


class Engine:
    def __init__(self, jobs: list[SimJob], cluster_spec: ClusterSpec, policy: Policy,
                 config: EngineConfig | None = None, arrival_horizon: float | None = None,
                 manifest: dict | None = None) -> None:
        self.config = config or EngineConfig()
        self.policy = policy
        self.arrival_horizon = arrival_horizon
        self.manifest = dict(manifest or {})
        self.manifest["policy"] = getattr(policy, "name", type(policy).__name__)
        self.manifest["engine_config"] = {
            "max_decisions_per_timestep": self.config.max_decisions_per_timestep,
            "view_pending_cap": self.config.view_pending_cap,
        }
        ordered = sorted(jobs, key=lambda j: (j.submit_ts, j.job_id))
        state = SimState(
            cluster=ClusterState.empty(cluster_spec),
            jobs={j.job_id: j for j in ordered},
            pending={}, records={j.job_id: JobRecord.from_job(j) for j in ordered},
            queue=EventQueue(), log=[], samples=[], warnings=[],
        )
        self.state = state
        for job in ordered:  # seq assignment in arrival order => deterministic
            self._push(EventKind.JOB_ARRIVE, job.submit_ts, job.job_id)

    @classmethod
    def from_state(cls, state: SimState, policy: Policy, config: EngineConfig | None = None,
                   arrival_horizon: float | None = None, manifest: dict | None = None) -> "Engine":
        """Continue a run from a (copied) state; used by fork()."""
        engine = cls.__new__(cls)
        engine.config = config or EngineConfig()
        engine.policy = policy
        engine.arrival_horizon = arrival_horizon
        engine.manifest = dict(manifest or {})
        engine.manifest["policy"] = getattr(policy, "name", type(policy).__name__)
        engine.manifest["forked"] = True
        engine.state = state
        return engine

    # -- event plumbing ---------------------------------------------------

    def _push(self, kind: EventKind, at: float, job_id: str | None = None) -> Event:
        event = Event(time=at, seq=self.state.seq, kind=kind, job_id=job_id)
        self.state.seq += 1
        self.state.queue.push(event)
        return event

    def _advance_clock(self, to: float) -> None:
        horizon = self.arrival_horizon if self.arrival_horizon is not None else float("inf")
        self.state.cluster.advance(self.state.now, to, horizon)
        self.state.now = to

    # -- main loop --------------------------------------------------------

    def run(self) -> RunResult:
        start = _time.perf_counter()
        while len(self.state.queue):
            self.step()
        wall = _time.perf_counter() - start
        if self.state.pending:
            self.state.warnings.append(
                f"{len(self.state.pending)} jobs still pending at end of run (deadlock?)")
        self.manifest["events_processed"] = self.state.events_processed
        summary = summarize(self.state, self.arrival_horizon)
        self.manifest["makespan_s"] = summary["cluster"]["makespan_s"]
        return RunResult(
            manifest=self.manifest, log=self.state.log, records=self.state.records,
            state=self.state, summary=summary, digest=event_log_digest(self.state.log),
            wall_s=wall, warnings=list(self.state.warnings),
        )

    def step(self) -> Event:
        """Process exactly one event; public so tests/benchmarks can drive runs."""
        event = self.state.queue.pop()
        self._advance_clock(event.time)
        self.state.log.append(event.log_line())
        self.state.events_processed += 1
        if event.kind == EventKind.JOB_ARRIVE:
            self._on_arrive(event)
        elif event.kind == EventKind.JOB_START:
            self._on_start(event)
        elif event.kind == EventKind.JOB_FINISH:
            self._on_finish(event)
        elif event.kind == EventKind.DECISION_POINT:
            self._on_decision_point()
        return event

    # -- handlers ----------------------------------------------------------

    def _on_arrive(self, event: Event) -> None:
        self.state.pending[event.job_id] = None
        self._push(EventKind.DECISION_POINT, event.time)

    def _on_start(self, event: Event) -> None:
        record = self.state.records[event.job_id]
        record.start_ts = event.time
        record.wait_s = event.time - record.submit_ts

    def _on_finish(self, event: Event) -> None:
        placement = self.state.cluster.release(event.job_id)
        record = self.state.records[event.job_id]
        record.finish_ts = event.time
        record.gpu_ids = placement.gpu_ids
        record.run_s = placement.run_s
        record.gpu_hours = placement.run_s * placement.share_per_gpu * len(placement.gpu_ids) / 3600.0
        record.energy_kwh = _incremental_energy_kwh(self.state.cluster, placement)
        job = self.state.jobs[event.job_id]
        record.deadline_met = (event.time <= job.deadline_ts + EPS) if job.job_type != "online_inference" else None
        if job.job_type == "online_inference":
            # SLO clock includes model load (decision): start + load_s - submit <= slo_s
            record.slo_met = (record.start_ts + job.load_s - job.submit_ts) <= (job.slo_s or 0.0) + EPS
        record.final = True
        self._push(EventKind.DECISION_POINT, event.time)

    def _on_decision_point(self) -> None:
        self._sample()
        decisions = 0
        while True:
            if not self.state.pending:
                return
            view = self.build_view()
            choice = self.policy.decide(view)
            if choice is None:
                return
            if decisions >= self.config.max_decisions_per_timestep:
                self.state.warnings.append(
                    f"max-decisions guard ({self.config.max_decisions_per_timestep}) tripped at t={self.state.now}")
                return
            job = self.state.jobs.get(choice)
            if job is None or choice not in self.state.pending:
                raise ValueError(f"policy {self.policy.name!r} returned unknown/non-pending job {choice!r}")
            gpu_ids = self.state.cluster.best_fit(
                job.n_occupied, job.share_per_gpu, job.per_gpu_vram, job.gpu_type_req)
            if gpu_ids is None:
                raise ValueError(
                    f"policy {self.policy.name!r} chose infeasible job {choice!r} at t={self.state.now}")
            self._apply(job, gpu_ids)
            decisions += 1

    def _apply(self, job: SimJob, gpu_ids: tuple[int, ...]) -> None:
        slowest_cf = min(self.state.cluster.gpus[i].spec.compute_factor for i in gpu_ids)
        run_s = (job.load_s + job.duration_s) / slowest_cf
        self.state.pending.pop(job.job_id)
        self.state.cluster.place(Placement(
            job_id=job.job_id, gpu_ids=gpu_ids, share_per_gpu=job.share_per_gpu,
            per_gpu_vram=job.per_gpu_vram, start_ts=self.state.now,
            finish_ts=self.state.now + run_s, run_s=run_s,
        ))
        self._push(EventKind.JOB_START, self.state.now, job.job_id)
        self._push(EventKind.JOB_FINISH, self.state.now + run_s, job.job_id)

    # -- decision-time view --------------------------------------------------

    def build_view(self) -> DecisionView:
        state = self.state
        cluster = state.cluster
        gpus = tuple(
            GpuView(g.gpu_id, g.spec.type, g.spec.vram_gb, g.spec.compute_factor,
                    g.spec.p_idle_w, g.spec.p_peak_w,
                    g.used_share, g.used_vram, g.free_share, g.free_vram)
            for g in cluster.gpus.values()
        )
        pending_ids = list(state.pending.keys())
        head_job_id = pending_ids[0] if pending_ids else None
        head_fits = False
        if head_job_id is not None:
            head = state.jobs[head_job_id]
            head_fits = cluster.best_fit(
                head.n_occupied, head.share_per_gpu, head.per_gpu_vram, head.gpu_type_req) is not None
        views: list[PendingView] = []
        for arrival_index, job_id in enumerate(pending_ids[: self.config.view_pending_cap]):
            job = state.jobs[job_id]
            if arrival_index == 0:
                fits = head_fits
            else:
                fits = cluster.any_fit(
                    job.n_occupied, job.share_per_gpu, job.per_gpu_vram, job.gpu_type_req)
            views.append(PendingView(
                job_id=job.job_id, group=job.group, job_type=job.job_type,
                priority=job.priority, submit_ts=job.submit_ts,
                wait_s=state.now - job.submit_ts, runtime_est_s=job.runtime_est_s,
                num_gpus=job.num_gpus, per_gpu_vram=job.per_gpu_vram,
                deadline_ts=job.deadline_ts, slo_s=job.slo_s, load_s=job.load_s,
                gpu_type_req=job.gpu_type_req, fits=fits, arrival_index=arrival_index,
            ))
        running_views = []
        for job_id, placement in cluster.placements.items():
            job = state.jobs[job_id]
            slowest_cf = min(cluster.gpus[i].spec.compute_factor for i in placement.gpu_ids)
            est_run = (job.load_s + job.runtime_est_s) / slowest_cf
            running_views.append(RunningView(
                job_id=job_id, gpu_ids=placement.gpu_ids,
                start_ts=placement.start_ts,
                est_finish_ts=placement.start_ts + est_run,
                share_per_gpu=placement.share_per_gpu,
                per_gpu_vram=placement.per_gpu_vram,
                priority=job.priority, job_type=job.job_type,
                runtime_est_s=job.runtime_est_s,
            ))
        return DecisionView(
            now=state.now, n_pending=len(pending_ids),
            n_truncated=max(0, len(pending_ids) - self.config.view_pending_cap),
            pending=tuple(views), head_job_id=head_job_id, head_fits=head_fits,
            gpus=gpus, running=tuple(running_views), n_running=len(running_views),
            power_budget_w=cluster.spec.power_budget_w,
        )

    def _sample(self) -> None:
        state = self.state
        n_pending = len(state.pending)
        frag = None
        if n_pending > 0:
            min_vram = min(state.jobs[j].per_gpu_vram for j in state.pending)
            free = [g.free_vram for g in state.cluster.gpus.values()]
            stranded = sum(v for v in free if v < min_vram - EPS)
            frag = stranded / state.cluster.spec.total_vram_gb
        state.samples.append(ClusterSample(
            t=state.now, queue_len=n_pending,
            util=state.cluster.used_share_total() / state.cluster.spec.n_gpus,
            vram_used=state.cluster.used_vram_total(),
            frag=frag,
        ))


def _incremental_energy_kwh(cluster: ClusterState, placement: Placement) -> float:
    """Per-job energy attribution: incremental power only (idle power stays cluster-level)."""
    joules = sum(
        (cluster.gpus[i].spec.p_peak_w - cluster.gpus[i].spec.p_idle_w) * placement.share_per_gpu
        for i in placement.gpu_ids
    ) * placement.run_s
    return joules / 3.6e6
