"""Synthetic workload generator (PLAN.md §2.4): Poisson and bursty (MMPP).

Produces a DataFrame in the unified job schema (§2.5) — it validates against
`schema.validate_jobs` — with every derived field tagged `synthetic`. Used for
(a) simulator correctness tests with hand-checkable structure, (b) distribution
shift tests, (c) fallback when trace downloads fail.

Bursty mode is a 2-state Markov-modulated Poisson process: the arrival rate
switches between rate_low and rate_high with exponential sojourns, giving
clustered arrivals (CV of per-bucket counts well above the Poisson 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gde.ingestion.schema import COLUMNS

JOB_TYPE_PARAMS_DEFAULT: dict[str, dict] = {
    "training": {
        "duration": {"mu": 8.0, "sigma": 1.2},          # lognormal seconds
        "num_gpus": {"kind": "int", "choices": [1.0, 2.0, 4.0, 8.0]},
        "vram_per_gpu": {"min": 8.0, "max": 20.0},
        "priority": {"min": 1, "max": 6},
        "slack": {"dist": "loguniform", "min": 2.0, "max": 10.0},
    },
    "batch_inference": {
        "duration": {"mu": 5.0, "sigma": 0.8},
        "num_gpus": {"kind": "share", "min": 0.1, "max": 0.5},
        "vram_per_gpu": {"min": 2.0, "max": 8.0},
        "priority": {"min": 4, "max": 8},
        "slack": {"dist": "loguniform", "min": 1.5, "max": 4.0},
    },
    "online_inference": {
        "duration": {"mu": 2.5, "sigma": 0.45},          # ~5-30 s
        "num_gpus": {"kind": "share", "min": 0.1, "max": 0.5},
        "vram_per_gpu": {"min": 2.0, "max": 8.0},
        "priority": {"min": 9, "max": 10},
        "slo_s": [1.0, 2.0, 5.0, 10.0],
    },
}


@dataclass
class SyntheticConfig:
    n_jobs: int = 1000
    mode: str = "poisson"            # poisson | bursty
    rate_per_s: float = 0.05         # mean arrival rate (poisson; low state for bursty)
    bursty_rate_high: float = 0.5    # high-state rate (bursty)
    bursty_mean_stay_s: float = 60.0  # mean sojourn per state (bursty)
    job_mix: dict[str, float] = field(default_factory=lambda: {
        "training": 0.7, "batch_inference": 0.2, "online_inference": 0.1})
    model_load_s: float = 0.0
    type_params: dict[str, dict] = field(
        default_factory=lambda: {k: dict(v) for k, v in JOB_TYPE_PARAMS_DEFAULT.items()})


def _arrival_times(rng: np.random.Generator, cfg: SyntheticConfig) -> np.ndarray:
    """Sorted arrival timestamps over the naturally-terminated horizon."""
    times: list[float] = []
    if cfg.mode == "poisson":
        t, scale = 0.0, 1.0 / cfg.rate_per_s
        for _ in range(cfg.n_jobs):
            t += rng.exponential(scale)
            times.append(t)
    elif cfg.mode == "bursty":
        t, high = 0.0, False
        while len(times) < cfg.n_jobs:
            rate = cfg.bursty_rate_high if high else cfg.rate_per_s
            stay = rng.exponential(cfg.bursty_mean_stay_s)
            end = t + stay
            while t < end and len(times) < cfg.n_jobs:
                t += rng.exponential(1.0 / rate)
                if t < end:
                    times.append(t)
            t, high = end, not high
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")
    return np.array(times)


def generate_workload(cfg: SyntheticConfig, seed: int) -> pd.DataFrame:
    """Unified-schema frame; every derived field provenance-tagged synthetic."""
    rng = np.random.default_rng(seed)
    arrivals = _arrival_times(rng, cfg)
    types = list(cfg.job_mix)
    probs = np.array([cfg.job_mix[t] for t in types], dtype=float)
    probs /= probs.sum()
    job_types = rng.choice(types, size=cfg.n_jobs, p=probs)

    rows: list[dict] = []
    for i in range(cfg.n_jobs):
        jt = str(job_types[i])
        p = cfg.type_params[jt]
        dur = float(rng.lognormal(p["duration"]["mu"], p["duration"]["sigma"]))
        ng = p["num_gpus"]
        num_gpus = (float(ng["choices"][rng.integers(0, len(ng["choices"]))])
                    if ng["kind"] == "int"
                    else float(rng.uniform(ng["min"], ng["max"])))
        n_occ = 1 if num_gpus <= 1 else int(round(num_gpus))
        vram = float(rng.uniform(p["vram_per_gpu"]["min"], p["vram_per_gpu"]["max"])) * n_occ
        priority = int(rng.integers(p["priority"]["min"], p["priority"]["max"] + 1))
        submit = float(arrivals[i])
        if jt == "online_inference":
            slo = float(p["slo_s"][rng.integers(0, len(p["slo_s"]))])
            deadline = submit + slo
        else:
            slack = float(rng.uniform(np.log(p["slack"]["min"]), np.log(p["slack"]["max"])))
            deadline = submit + np.exp(slack) * dur
        rows.append({
            "job_id": f"synthetic_{seed}_{i}",
            "dataset": "synthetic",
            "submit_ts": submit,
            "duration_s": dur,
            "runtime_est_s": dur * float(np.exp(rng.normal(0.0, 0.25))),  # noisy estimate
            "num_gpus": num_gpus,
            "gpu_type_req": None,
            "vram_gb": vram,
            "cpu_cores": 0.0,
            "mem_gb": 0.0,
            "job_type": jt,
            "priority": priority,
            "deadline_ts": deadline,
            "user_group": None,
            "final_status": "pass",
            "vram_gb_src": "synthetic",
            "job_type_src": "synthetic",
            "priority_src": "synthetic",
            "deadline_src": "synthetic",
            "runtime_est_src": "synthetic",
        })
    return pd.DataFrame(rows, columns=COLUMNS)
