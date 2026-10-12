"""Cluster spec (configs/cluster_*.yaml) and mutable per-GPU state (PLAN.md §4.1).

Fractional-GPU sharing: one GPU hosts several jobs iff the SUM of their GPU
shares fits in 1.0 AND the SUM of their per-GPU VRAM fits in the GPU's VRAM.
Whole/multi-GPU jobs occupy ``n_occupied`` GPUs exclusively (share 1.0 each);
fractional-share jobs never co-locate with them because the share sum would
exceed 1. Runtime model: a job's occupancy lasts ``load_s + duration_s`` divided
by ``compute_factor`` of the slowest GPU in its placement (v1 has no
interference term; a shared GPU runs each tenant at full speed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

EPS = 1e-9


@dataclass(frozen=True)
class GpuSpec:
    gpu_id: int
    type: str
    vram_gb: float
    compute_factor: float
    p_idle_w: float
    p_peak_w: float
    price_per_hour: float


@dataclass(frozen=True)
class ClusterSpec:
    gpus: tuple[GpuSpec, ...]
    power_budget_w: float

    @property
    def n_gpus(self) -> int:
        return len(self.gpus)

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClusterSpec":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        gpus: list[GpuSpec] = []
        for entry in raw["gpus"]:
            for _ in range(int(entry["count"])):
                gpus.append(GpuSpec(
                    gpu_id=len(gpus),
                    type=str(entry["type"]),
                    vram_gb=float(entry["vram_gb"]),
                    compute_factor=float(entry["compute_factor"]),
                    p_idle_w=float(entry["p_idle_w"]),
                    p_peak_w=float(entry["p_peak_w"]),
                    price_per_hour=float(entry["price_per_hour"]),
                ))
        return cls(gpus=tuple(gpus), power_budget_w=float(raw.get("power_budget_kw", 0.0)) * 1000.0)

    @classmethod
    def homogeneous(cls, n: int, vram_gb: float = 24.0, compute_factor: float = 1.0,
                    p_idle_w: float = 60.0, p_peak_w: float = 300.0,
                    power_budget_w: float = 1e9, gpu_type: str = "generic") -> "ClusterSpec":
        return cls(
            gpus=tuple(GpuSpec(i, gpu_type, vram_gb, compute_factor, p_idle_w, p_peak_w, 1.0)
                       for i in range(n)),
            power_budget_w=power_budget_w,
        )


@dataclass(frozen=True)
class Placement:
    """Where a running job sits; frozen so snapshots can share it."""
    job_id: str
    gpu_ids: tuple[int, ...]
    share_per_gpu: float
    per_gpu_vram: float
    start_ts: float
    finish_ts: float
    run_s: float


@dataclass
class GpuState:
    spec: GpuSpec
    used_share: float = 0.0
    used_vram: float = 0.0
    # Integral accumulators, advanced by ClusterState.advance().
    share_time_s: float = 0.0        # int(used_share dt) over the whole run
    vram_time_gb_s: float = 0.0      # int(used_vram dt)
    energy_j: float = 0.0            # int(P dt), P = p_idle + (p_peak - p_idle) * used_share
    share_time_aw_s: float = 0.0     # same, clamped to the arrival window
    vram_time_aw_gb_s: float = 0.0
    energy_aw_j: float = 0.0

    @property
    def gpu_id(self) -> int:
        return self.spec.gpu_id

    @property
    def free_share(self) -> float:
        return 1.0 - self.used_share

    @property
    def free_vram(self) -> float:
        return self.spec.vram_gb - self.used_vram

    @property
    def power_w(self) -> float:
        return self.spec.p_idle_w + (self.spec.p_peak_w - self.spec.p_idle_w) * self.used_share

    def copy(self) -> "GpuState":
        return GpuState(
            spec=self.spec, used_share=self.used_share, used_vram=self.used_vram,
            share_time_s=self.share_time_s, vram_time_gb_s=self.vram_time_gb_s,
            energy_j=self.energy_j,
            share_time_aw_s=self.share_time_aw_s, vram_time_aw_gb_s=self.vram_time_aw_gb_s,
            energy_aw_j=self.energy_aw_j,
        )


@dataclass
class ClusterState:
    spec: ClusterSpec
    gpus: dict[int, GpuState]
    placements: dict[str, Placement] = field(default_factory=dict)  # running jobs

    @classmethod
    def empty(cls, spec: ClusterSpec) -> "ClusterState":
        return cls(spec=spec, gpus={g.gpu_id: GpuState(spec=g) for g in spec.gpus})

    def advance(self, from_t: float, to_t: float, arrival_horizon: float) -> None:
        """Accumulate per-GPU integrals over (from_t, to_t], split at the horizon."""
        dt = to_t - from_t
        if dt <= 0:
            return
        dt_aw = min(to_t, arrival_horizon) - from_t
        dt_aw = max(0.0, min(dt_aw, dt))
        for g in self.gpus.values():
            p = g.power_w
            g.share_time_s += g.used_share * dt
            g.vram_time_gb_s += g.used_vram * dt
            g.energy_j += p * dt
            if dt_aw > 0:
                g.share_time_aw_s += g.used_share * dt_aw
                g.vram_time_aw_gb_s += g.used_vram * dt_aw
                g.energy_aw_j += p * dt_aw

    def any_fit(self, n_occupied: int, share_per_gpu: float, per_gpu_vram: float,
                gpu_type_req: str | None) -> bool:
        """Feasibility-only check (no placement sort); used for view `fits` flags."""
        eligible = [
            g for g in self.gpus.values()
            if (gpu_type_req is None or g.spec.type == gpu_type_req)
            and g.free_share >= share_per_gpu - EPS
            and g.free_vram >= per_gpu_vram - EPS
        ]
        if len(eligible) < n_occupied or not self._power_ok(
                tuple(g.gpu_id for g in eligible[:n_occupied]), share_per_gpu):
            return False
        return True

    def best_fit(self, n_occupied: int, share_per_gpu: float, per_gpu_vram: float,
                 gpu_type_req: str | None) -> tuple[int, ...] | None:
        """Deterministic best-fit: least leftover VRAM that fits; tie-break gpu_id."""
        eligible = [
            g for g in self.gpus.values()
            if (gpu_type_req is None or g.spec.type == gpu_type_req)
            and g.free_share >= share_per_gpu - EPS
            and g.free_vram >= per_gpu_vram - EPS
        ]
        if len(eligible) < n_occupied:
            return None
        eligible.sort(key=lambda g: (g.spec.vram_gb - g.used_vram - per_gpu_vram, g.gpu_id))
        chosen = tuple(g.gpu_id for g in eligible[:n_occupied])
        if not self._power_ok(chosen, share_per_gpu):
            return None
        return chosen

    def _power_ok(self, gpu_ids: tuple[int, ...], share_per_gpu: float) -> bool:
        if self.spec.power_budget_w <= 0:
            return True
        total = sum(g.power_w for g in self.gpus.values())
        delta = sum((self.gpus[i].spec.p_peak_w - self.gpus[i].spec.p_idle_w) * share_per_gpu
                    for i in gpu_ids)
        return total + delta <= self.spec.power_budget_w + EPS

    def instant_power_w(self) -> float:
        return sum(g.power_w for g in self.gpus.values())

    def place(self, placement: Placement) -> None:
        for gpu_id in placement.gpu_ids:
            g = self.gpus[gpu_id]
            g.used_share += placement.share_per_gpu
            g.used_vram += placement.per_gpu_vram
        self.placements[placement.job_id] = placement

    def release(self, job_id: str) -> Placement:
        p = self.placements.pop(job_id)
        for gpu_id in p.gpu_ids:
            g = self.gpus[gpu_id]
            g.used_share = max(0.0, g.used_share - p.share_per_gpu)
            g.used_vram = max(0.0, g.used_vram - p.per_gpu_vram)
        return p

    def copy(self) -> "ClusterState":
        return ClusterState(
            spec=self.spec,
            gpus={i: g.copy() for i, g in self.gpus.items()},
            placements=dict(self.placements),  # Placements are frozen -> shared
        )

    def used_share_total(self) -> float:
        return sum(g.used_share for g in self.gpus.values())

    def used_vram_total(self) -> float:
        return sum(g.used_vram for g in self.gpus.values())
