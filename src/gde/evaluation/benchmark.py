"""Benchmark harness v0 (PLAN.md P4): run scheduler cells under the
pre-registered two-track regimes (configs/regimes.yaml, FROZEN).

A cell = (scheduler, regime, rho, overlay_share, seed). Seeds are PAIRED:
every scheduler sees the same seeded workload instances per (regime, rho,
overlay, seed). Parallelized with ProcessPoolExecutor (CPU-only); results are
independent of execution order (engine determinism by seed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from gde.baselines import ALL_POLICIES
from gde.paths import CONFIGS, DATA
from gde.simulator.cluster import ClusterSpec
from gde.simulator.engine import Engine, EngineConfig, EventKind
from gde.simulator.snapshot import fork, snapshot
from gde.simulator.workload import (
    OverlayConfig,
    TraceConfig,
    generate_overlay,
    load_trace_window,
)


@dataclass(frozen=True)
class Cell:
    scheduler: str
    regime: str          # capped-1h | stress-24h | uncapped-1h
    rho: float
    overlay_share: float
    seed: int
    est_mode: str | None = None  # None -> experiment default; 'oracle' = clairvoyant ablation
    tag: str = ""        # e.g. 'clairvoyant'


def load_regimes(path: Path | None = None) -> dict:
    path = path or (CONFIGS / "regimes.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def regime_spec(regimes: dict, name: str) -> dict:
    """Flatten a regime entry (primary/stress/optional_cell) into run params."""
    if name == "capped-1h":
        spec = regimes["primary"]
    elif name == "stress-24h":
        spec = regimes["stress"]
    elif name == "uncapped-1h":
        base = dict(regimes["primary"])
        opt = regimes.get("optional_cell", {})
        base.update(opt)
        base["rho_sweep"] = opt.get("rho", [0.8])
        base["duration_cap_s"] = None
        spec = base
    else:
        raise ValueError(f"unknown regime {name!r}")
    return spec


def make_overlay_cfg(raw_overlay: dict, share: float) -> OverlayConfig:
    return OverlayConfig(
        share=share,
        duration_mu=raw_overlay["duration"]["mu"],
        duration_sigma=raw_overlay["duration"]["sigma"],
        duration_min_s=raw_overlay["duration"]["min_s"],
        duration_max_s=raw_overlay["duration"]["max_s"],
        num_gpus_min=raw_overlay["num_gpus"]["min"],
        num_gpus_max=raw_overlay["num_gpus"]["max"],
        priorities=tuple(raw_overlay["priority"]),
        slo_choices=tuple(float(s) for s in raw_overlay["slo_s"]),
        model_load_s=raw_overlay["model_load_s"],
        vram_min_gb=raw_overlay["vram_gb"]["min"],
        vram_max_gb=raw_overlay["vram_gb"]["max"],
    )


def run_cell(cell: Cell, regimes: dict, overlay_raw: dict,
             est_mode: str = "stored", warmup_fracs: tuple[float, ...] = (0.1, 0.25, 0.5)) -> dict:
    """One (scheduler, regime, rho, overlay, seed) FIFO-compatible run.

    Returns the summary plus `warmup` blocks recomputing trace-group
    wait/deadline metrics excluding jobs submitted before frac * T_sim for
    each requested warm-up fraction."""
    spec = regime_spec(regimes, cell.regime)
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    cap = spec.get("duration_cap_s")
    est = cell.est_mode or est_mode
    trace_cfg = TraceConfig(
        window_s=spec["window_s"], rho=cell.rho, est_mode=est,
        max_jobs=10**9, gpu_type=spec.get("gpu_type", "ignore"),
        duration_cap_s=float(cap) if cap else None,
    )
    jobs, tman = load_trace_window(DATA / "processed" / f"{spec['dataset']}.parquet",
                                   cluster, trace_cfg, seed=cell.seed)
    overlay_cfg = make_overlay_cfg(overlay_raw, cell.overlay_share)
    overlay, oman = generate_overlay(len(jobs), tman["t_sim_s"], overlay_cfg, seed=cell.seed)
    engine = Engine(
        jobs + overlay, cluster, ALL_POLICIES[cell.scheduler](),
        config=EngineConfig(), arrival_horizon=tman["t_sim_s"],
        manifest={"cell": cell.__dict__, "trace": tman, "overlay": oman},
    )
    t0 = time.perf_counter()
    result = engine.run()
    wall = time.perf_counter() - t0

    warmup: dict[str, dict] = {}
    for frac in warmup_fracs:
        cutoff = frac * tman["t_sim_s"]
        rows = [r for r in result.records.values()
                if r.group == "trace" and r.submit_ts >= cutoff]
        waits = sorted(r.wait_s for r in rows if r.wait_s is not None)
        dl = [r for r in rows if r.job_type != "online_inference"]
        warmup[str(frac)] = {
            "n_jobs": len(rows),
            "wait_s_p50": waits[len(waits) // 2] if waits else None,
            "wait_s_p95": waits[min(int(0.95 * (len(waits) - 1)), len(waits) - 1)] if waits else None,
            "deadline_miss_rate": (sum(1 for r in dl if r.deadline_met is False) / len(dl)) if dl else None,
        }
    return {
        "cell": cell.__dict__, "trace_manifest": tman, "overlay_manifest": oman,
        "summary": result.summary, "digest": result.digest,
        "warnings": result.warnings, "wall_s": wall,
        "events": result.state.events_processed,
        "warmup": warmup,
    }


def run_cell_parallel(args: tuple) -> dict:
    """Top-level worker for ProcessPoolExecutor (Windows spawn-safe)."""
    cell, regimes, overlay_raw, est_mode, warmup_fracs = args
    return run_cell(cell, regimes, overlay_raw, est_mode, warmup_fracs)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else None


def key_of(result: dict) -> tuple:
    c = result["cell"]
    return (c["regime"], c["rho"], c["overlay_share"], c["scheduler"], c["seed"])


def group_key(result: dict) -> tuple:
    """Aggregation key: clairvoyant/ablation cells never merge with stored runs."""
    c = result["cell"]
    return (c["regime"], c["rho"], c["overlay_share"], c["scheduler"],
            c.get("est_mode") or "stored", c.get("tag") or "")


def aggregate(results: list[dict], warmup_frac: float | None = None) -> dict:
    """Group by (regime, rho, overlay, scheduler, est, tag); mean over paired
    seeds. warmup_frac not None swaps trace wait/deadline metrics for that
    warm-up exclusion variant (P4 pre-registered check: do rankings change?)."""
    grouped: dict[tuple, list[dict]] = {}
    for r in results:
        grouped.setdefault(group_key(r), []).append(r)
    out: dict[tuple, dict] = {}
    for k, runs in grouped.items():
        g = [r["summary"]["groups"] for r in runs]
        cl = [r["summary"]["cluster"] for r in runs]
        trace = [x.get("trace", {}) for x in g]
        if warmup_frac is not None:
            trace = [dict(t, **r["warmup"][str(warmup_frac)])
                     for t, r in zip(trace, runs)]
        overlay = [x.get("overlay", {}) for x in g]
        out[k] = {
            "n_seeds": len(runs),
            "util_aw": _mean([c["utilization_arrival_window"] for c in cl]),
            "makespan_s": _mean([c["makespan_s"] for c in cl]),
            "frag_mean": _mean([c["fragmentation_mean"] for c in cl]),
            "energy_kwh_aw": _mean([c["energy_kwh_arrival_window"] for c in cl]),
            "realized_rho_trace": _mean([t.get("realized_load") for t in trace]),
            "realized_rho_overlay": _mean([t.get("realized_load") for t in overlay]),
            "trace_wait_p50_s": _mean([t.get("wait_s_p50") for t in trace]),
            "trace_wait_p95_s": _mean([t.get("wait_s_p95") for t in trace]),
            "trace_dl_miss": _mean([t.get("deadline_miss_rate") for t in trace]),
            "ov_wait_p50_s": _mean([t.get("wait_s_p50") for t in overlay]),
            "ov_slo_miss": _mean([t.get("slo_miss_rate") for t in overlay]),
            "events": _mean([float(r["events"]) for r in runs]),
            "wall_s": _mean([r["wall_s"] for r in runs]),
        }
    return out


def rank_by(agg: dict, regime: str, rho: float, overlay: float,
            metric: str, est: str = "stored", tag: str = "",
            lower_is_better: bool = True) -> list[tuple[str, float]]:
    rows = []
    for key, m in agg.items():
        (reg, r, ov, sched, k_est, k_tag) = key
        if (reg, r, ov, k_est, k_tag) == (regime, rho, overlay, est, tag) \
                and m.get(metric) is not None:
            rows.append((sched, m[metric]))
    rows.sort(key=lambda x: x[1], reverse=not lower_is_better)
    return rows


# ---------------------------------------------------------------------------
# Amendment 2 (E): split/eligibility report for the seed lists
# ---------------------------------------------------------------------------

def window_eligibility(regimes: dict) -> dict:
    """Per split-seed capped-1h window eligibility (Amendment-1 rule applies
    at P6 sampling), the benchmark replacement seeds' split bands, and the
    arrival-free trace prefix that makes early windows degenerate."""
    import collections

    cluster = ClusterSpec.from_yaml(CONFIGS / regimes["primary"]["cluster"])
    path = DATA / "processed" / "alibaba2020.parquet"
    cap = float(regimes["primary"]["duration_cap_s"])
    cfg = TraceConfig(window_s=3600.0, rho=0.8, max_jobs=10**9, duration_cap_s=cap)

    import pandas as pd
    df = pd.read_parquet(path, columns=["submit_ts"])
    t_min = float(df["submit_ts"].min())
    t_max = float(df["submit_ts"].max())
    span = t_max - t_min

    def window_start_fraction(seed: int) -> float:
        return float(np.random.default_rng(seed).uniform(0.0, span - 3600.0)) / span

    def n_feasible(seed: int) -> int:
        try:
            jobs, _ = load_trace_window(path, cluster, cfg, seed=seed)
            return len(jobs)
        except ValueError:
            return 0

    splits = {}
    for split, seeds in regimes["splits"]["split_seeds"].items():
        counts = [n_feasible(s) for s in seeds]
        splits[split] = {
            "n_seeds": len(seeds),
            "n_degenerate": sum(1 for c in counts if c == 0),
            "min_feasible": min(counts),
        }

    bands = regimes["splits"]["window_bands"]

    def band_of(seed: int) -> str:
        f = window_start_fraction(seed)
        for name, (lo, hi) in bands.items():
            if lo <= f < hi:
                return name
        return "test" if f >= bands["test"][0] else "train"

    replacements = {}
    for orig, repl in ((31001, 31501), (31006, 31506)):
        replacements[f"{orig}->{repl}"] = {
            "orig_band": band_of(orig), "repl_band": band_of(repl),
            "repl_feasible_jobs": n_feasible(repl),
        }

    # arrival-free prefix: first trailing day from t_min whose hourly count
    # stays under 1 job/h in every hour of that day
    submits = df["submit_ts"].to_numpy()
    day = 86400.0
    prefix_days = 0
    while prefix_days * day < span:
        lo = t_min + prefix_days * day
        hi = lo + day
        n = int(((submits >= lo) & (submits < hi)).sum())
        if n >= 24:  # at least ~1 job/hour somewhere -> no longer arrival-free
            break
        prefix_days += 1
    return {
        "splits": splits,
        "replacement_seeds": replacements,
        "arrival_free_prefix_days": prefix_days,
        "arrival_free_prefix_span_fraction": prefix_days * day / span,
        "note": "benchmark seeds are not split-assigned; splits use their own seed lists",
    }


# ---------------------------------------------------------------------------
# Step 6: rollout straddle fraction (horizon claim correction)
# ---------------------------------------------------------------------------

def rollout_straddle_fraction(scheduler: str, regime: str, seed: int, regimes: dict,
                              overlay_raw: dict, n_arrivals: int, max_seconds: float,
                              sample_every: int = 5) -> dict:
    """Fraction of mid-run rollouts that still have running jobs at horizon end.

    Horizon end = when n_arrivals NEW arrivals have been processed OR
    max_seconds elapsed, whichever binds first (regimes.yaml semantics).
    """
    spec = regime_spec(regimes, regime)
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    cap = spec.get("duration_cap_s")
    trace_cfg = TraceConfig(
        window_s=spec["window_s"], rho=0.8, est_mode="stored",
        max_jobs=10**9, gpu_type="ignore",
        duration_cap_s=float(cap) if cap else None,
    )
    jobs, tman = load_trace_window(DATA / "processed" / f"{spec['dataset']}.parquet",
                                   cluster, trace_cfg, seed=seed)
    overlay_cfg = make_overlay_cfg(overlay_raw, 0.25)
    overlay, _ = generate_overlay(len(jobs), tman["t_sim_s"], overlay_cfg, seed=seed)
    engine = Engine(jobs + overlay, cluster, ALL_POLICIES[scheduler](),
                    arrival_horizon=tman["t_sim_s"])
    n_dp = 0
    n_rollouts = 0
    n_with_running = 0
    n_bound_by_arrivals = 0
    while len(engine.state.queue):
        event = engine.state.queue.peek()
        if event.kind == EventKind.DECISION_POINT:
            n_dp += 1
            if n_dp % sample_every == 0 and engine.state.pending:
                state = snapshot(engine.state)
                forked = fork(state, ALL_POLICIES[scheduler](),
                              arrival_horizon=tman["t_sim_s"])
                t0 = state.now
                arrivals = 0
                while len(forked.state.queue):
                    ev = forked.state.queue.peek()
                    if ev.time > t0 + max_seconds:
                        break
                    ev = forked.step()
                    if ev.kind == EventKind.JOB_ARRIVE:
                        arrivals += 1
                        if arrivals >= n_arrivals:
                            break
                n_rollouts += 1
                if arrivals >= n_arrivals:
                    n_bound_by_arrivals += 1
                if len(forked.state.cluster.placements) > 0:
                    n_with_running += 1
        engine.step()
    return {
        "scheduler": scheduler, "regime": regime, "seed": seed,
        "n_arrivals": n_arrivals, "max_seconds": max_seconds,
        "n_rollouts": n_rollouts,
        "frac_with_running_jobs": n_with_running / n_rollouts if n_rollouts else None,
        "frac_bound_by_arrivals": n_bound_by_arrivals / n_rollouts if n_rollouts else None,
    }
