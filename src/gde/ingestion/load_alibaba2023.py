"""Loader: Alibaba cluster-trace-gpu-v2023 (openb, fgd_gpu_trace_2023) -> unified frame.

Pod list `openb_pod_list_gpuspec33.csv` = the default list augmented with `gpu_spec`
(GPU-type constraints; docs/data_dictionary.md). The trace is a snapshot: ~64% of pods
are still Running/Pending (deletion_time is the trace end, not a real end) — those are
censored and dropped. `gpu_milli` is the GPU share in milli-GPU for sharing pods
(num_gpu == 1); `qos` is the only class field -> priority mapping (config).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET = "alibaba2023"
_KEEP_PHASE = {"Succeeded": "pass", "Failed": "failed"}


def load(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Return (core frame, drop counts by reason, notes). Core frame is pre-synthesis."""
    pods = pd.read_csv(Path(raw_dir) / "csv" / "openb_pod_list_gpuspec33.csv")
    drops: dict[str, int] = {}
    notes = [f"{DATASET}: pod rows read {len(pods)} (gpuspec33 list with GPU-type spec)"]

    phase_map = pods["pod_phase"].map(_KEEP_PHASE)
    drops["phase_not_finished (Running/Pending, censored snapshot)"] = int(phase_map.isna().sum())
    pods = pods[phase_map.notna()].copy()
    pods["final_status"] = phase_map[phase_map.notna()]

    pods["submit_ts"] = pd.to_numeric(pods["creation_time"], errors="coerce")
    sched = pd.to_numeric(pods["scheduled_time"], errors="coerce")
    dele = pd.to_numeric(pods["deletion_time"], errors="coerce")
    pods["duration_s"] = dele - sched
    pods["num_gpus"] = pods["num_gpu"] * pods["gpu_milli"] / 1000.0
    pods["gpu_fractional"] = (pods["gpu_milli"] < 1000) & (pods["num_gpu"] > 0)
    pods["cpu_cores"] = pods["cpu_milli"] / 1000.0
    pods["mem_gb"] = pods["memory_mib"] / 1024.0

    checks = [
        ("no_submit_time", lambda d: d["submit_ts"].isna()),
        ("no_scheduled_time", lambda d: pd.to_numeric(d["scheduled_time"], errors="coerce").isna()),
        ("no_gpu_request", lambda d: d["num_gpus"].fillna(0) <= 0),
        ("nonpositive_or_missing_duration", lambda d: ~(d["duration_s"] > 0)),
    ]
    for reason, failing in checks:
        mask = failing(pods)
        drops[reason] = int(mask.sum())
        pods = pods[~mask]

    # gpu_spec lists allowed types ("V100M16|V100M32"); keep the first listed as the
    # canonical request (deterministic; missing = unconstrained -> None).
    spec = pods["gpu_spec"].fillna("").astype(str).str.split("|", regex=False).str[0]
    pods["gpu_type_req"] = spec.where(spec != "", None)

    pods["user_group"] = None
    pods["priority"] = pods["qos"].map({"LS": 8, "Burstable": 5, "Guaranteed": 9, "BE": 2})
    drops["unknown_qos"] = int(pods["priority"].isna().sum())
    pods = pods[pods["priority"].notna()]
    pods["priority"] = pods["priority"].astype("int64")
    pods["priority_src"] = "derived"

    pods["job_id"] = pods["name"].astype(str)
    pods["dataset"] = DATASET
    pods["submit_ts"] = pods["submit_ts"] - pods["submit_ts"].min()
    cols = ["job_id", "dataset", "submit_ts", "duration_s", "num_gpus", "gpu_fractional",
            "gpu_type_req", "cpu_cores", "mem_gb", "user_group", "final_status",
            "priority", "priority_src"]
    return pods[cols].reset_index(drop=True), drops, notes
