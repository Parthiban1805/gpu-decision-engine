"""Loader: SenseTime Helios cluster logs -> unified job frames (one per cluster).

Per docs/data_dictionary.md: `duration` = end - start (actual runtime) and `queue` =
start - submit are given in seconds; states COMPLETED/CANCELLED/FAILED/TIMEOUT/NODE_FAIL
map to pass/killed/failed (TIMEOUT and NODE_FAIL count as failed per the README);
SUSPENDED/RUNNING are unfinished. gpu_num == 0 are CPU jobs (out of scope). No memory,
GPU-type or job-type fields exist (vram -> synthetic Beta, PLAN.md §2.2).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASETS = {"earth": "helios_earth", "saturn": "helios_saturn",
            "uranus": "helios_uranus", "venus": "helios_venus"}
_STATE_MAP = {"COMPLETED": "pass", "CANCELLED": "killed",
              "FAILED": "failed", "TIMEOUT": "failed", "NODE_FAIL": "failed"}
_DT = "%Y-%m-%d %H:%M:%S"


def load(cluster: str, raw_root: Path) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Return (core frame, drop counts by reason, notes) for one Helios cluster."""
    dataset = DATASETS[cluster.lower()]
    path = Path(raw_root) / "data" / cluster.capitalize() / "cluster_log.csv"
    df = pd.read_csv(path)
    drops: dict[str, int] = {}
    notes = [f"{dataset}: rows read {len(df)} from {path.name}"]

    state_map = df["state"].map(_STATE_MAP)
    drops["state_not_finished (SUSPENDED/RUNNING)"] = int(state_map.isna().sum())
    df = df[state_map.notna()].copy()
    df["final_status"] = state_map[state_map.notna()]

    submit = pd.to_datetime(df["submit_time"], format=_DT)
    df["submit_ts"] = (submit - submit.min()).dt.total_seconds()
    df["duration_s"] = pd.to_numeric(df["duration"], errors="coerce").astype(float)
    df["num_gpus"] = pd.to_numeric(df["gpu_num"], errors="coerce").astype(float)
    df["gpu_fractional"] = False                      # whole-GPU shares only in this trace
    df["gpu_type_req"] = None
    df["cpu_cores"] = pd.to_numeric(df["cpu_num"], errors="coerce").astype(float)
    df["mem_gb"] = float("nan")
    df["user_group"] = df["vc"].astype(str)

    checks = [
        ("cpu_job (gpu_num == 0)", lambda d: d["num_gpus"].fillna(0) <= 0),
        ("no_submit_time", lambda d: d["submit_ts"].isna()),
        ("nonpositive_or_missing_duration", lambda d: ~(d["duration_s"] > 0)),
    ]
    for reason, failing in checks:
        mask = failing(df)
        drops[reason] = int(mask.sum())
        df = df[~mask]

    df["priority"] = 5                                # no class field in this trace
    df["priority_src"] = "synthetic"
    df["job_id"] = df["job_id"].astype(str)
    df["dataset"] = dataset
    df["submit_ts"] = df["submit_ts"] - df["submit_ts"].min()
    cols = ["job_id", "dataset", "submit_ts", "duration_s", "num_gpus", "gpu_fractional",
            "gpu_type_req", "cpu_cores", "mem_gb", "user_group", "final_status",
            "priority", "priority_src"]
    return df[cols].reset_index(drop=True), drops, notes
