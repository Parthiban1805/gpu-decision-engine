"""Loader: Alibaba cluster-trace-gpu-v2020 (PAI) -> unified job frame (PLAN.md §2.5).

Column semantics from the trace README (docs/data_dictionary.md, verified in P1):
- job table `inst_id` is the job id; `start_time` is SUBMISSION (the job/task gap is 0 s
  for all Terminated jobs, so the trace carries no queueing delay — waits come from the
  simulator).
- GPU request per job = sum(plan_gpu * inst_num) / 100 over its tasks (50.0 = half a GPU).
- `gpu_type_spec` (group-tag table) is the type REQUESTED (empty = any); the task table's
  `gpu_type` is the type assigned — an outcome, never used here.
- Status: only Terminated is successful. Running/Waiting jobs are censored (no end time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET = "alibaba2020"
_KEEP_STATUS = {"Terminated": "pass", "Failed": "failed"}
_TASK_AGG: dict[str, Any] = {
    "gpu_pct": ("plan_gpu", "sum"),       # filled with 0 beforehand; multiplied by inst_num
    "cpu_pct": ("plan_cpu", "sum"),
    "mem_gb": ("plan_mem", "sum"),
    "t_start": ("start_time", "min"),
    "t_end": ("end_time", "max"),
    "shared_instance": ("shared_instance", "max"),
}


def load(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Return (core frame, drop counts by reason, notes). Core frame is pre-synthesis."""
    raw_dir = Path(raw_dir)

    def read(table: str) -> pd.DataFrame:
        names = (raw_dir / f"{table}.header").read_text(encoding="utf-8").strip().split(",")
        return pd.read_csv(raw_dir / f"{table}.csv", names=names, header=None,
                           dtype=str, keep_default_na=False, na_values=[""])

    job = read("pai_job_table")
    task = read("pai_task_table")
    tag = read("pai_group_tag_table")
    drops: dict[str, int] = {}
    notes = [f"{DATASET}: job rows read {len(job)}, task rows {len(task)}, tag rows {len(tag)}"]

    status_map = job["status"].map(_KEEP_STATUS)
    drops["status_not_finished (Running/Waiting, censored)"] = int(status_map.isna().sum())
    job = job[status_map.notna()].copy()
    job["final_status"] = status_map[status_map.notna()]

    dup = int(job["inst_id"].duplicated().sum())
    if dup:
        drops["duplicate_job_id"] = dup
        job = job.drop_duplicates("inst_id", keep="first")

    for col in ("plan_gpu", "plan_cpu", "plan_mem", "inst_num", "start_time", "end_time"):
        task[col] = pd.to_numeric(task[col], errors="coerce")
    task["shared_instance"] = ((task["plan_gpu"] > 0) & (task["plan_gpu"] < 100)).astype(float)
    for col in ("plan_gpu", "plan_cpu", "plan_mem"):
        task[col] = task[col] * task["inst_num"]          # per-instance request * instances

    tg = task.groupby("job_name", sort=False).agg(**_TASK_AGG)
    notes.append(f"{DATASET}: jobs without any task row: "
                 f"{int((~job['job_name'].isin(tg.index)).sum())} (duration from job table)")

    df = job.merge(tg, how="left", on="job_name")
    df["duration_s"] = df["t_end"] - df["t_start"]                       # task span
    job_span = pd.to_numeric(df["end_time"], errors="coerce") - pd.to_numeric(df["start_time"], errors="coerce")
    df["duration_s"] = df["duration_s"].fillna(job_span)                 # fallback: job-table span
    df["submit_ts"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["num_gpus"] = df["gpu_pct"] / 100.0
    df["cpu_cores"] = df["cpu_pct"] / 100.0                              # 600 = 6 cores
    df["mem_gb"] = df["mem_gb"]
    df["gpu_fractional"] = ((df["num_gpus"] % 1).abs() > 1e-9) | (df["shared_instance"] == 1.0)

    checks = [
        ("no_submit_time", lambda d: d["submit_ts"].isna()),
        ("no_gpu_request", lambda d: d["num_gpus"].fillna(0) <= 0),
        ("nonpositive_or_missing_duration", lambda d: ~(d["duration_s"] > 0)),
    ]
    for reason, failing in checks:
        mask = failing(df)
        drops[reason] = int(mask.sum())
        df = df[~mask]

    tag_first = tag.sort_values("inst_id").drop_duplicates("inst_id", keep="first")
    df = df.merge(tag_first[["inst_id", "gpu_type_spec", "group", "user"]], on="inst_id", how="left")
    df["gpu_type_req"] = df["gpu_type_spec"].fillna("")
    df["gpu_type_req"] = df["gpu_type_req"].where(df["gpu_type_req"] != "", None)
    df["user_group"] = df["group"].fillna(df["user_y"]).astype(object).where(
        df["group"].notna() | df["user_y"].notna(), None)
    df["priority"] = 5                    # trace has no priority/class field
    df["priority_src"] = "synthetic"

    df["job_id"] = df["inst_id"].astype(str)
    df["dataset"] = DATASET
    df["submit_ts"] = df["submit_ts"] - df["submit_ts"].min()
    cols = ["job_id", "dataset", "submit_ts", "duration_s", "num_gpus", "gpu_fractional",
            "gpu_type_req", "cpu_cores", "mem_gb", "user_group", "final_status",
            "priority", "priority_src"]
    return df[cols].reset_index(drop=True), drops, notes
