"""Unified job schema (PLAN.md §2.5) and its validator.

Every normalized dataset is written as Parquet with exactly these columns. The five
`*_src` columns tag how each derived field was produced (measured / derived / synthetic);
reports must disclose the synthetic ones.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

JOB_TYPES = ("training", "batch_inference", "online_inference")
FINAL_STATUSES = ("pass", "failed", "killed")
PROVENANCE_TAGS = ("measured", "derived", "synthetic")

JOB_SCHEMA = pa.schema([
    ("job_id", pa.string()),
    ("dataset", pa.string()),
    ("submit_ts", pa.float64()),        # seconds from the dataset's first submission
    ("duration_s", pa.float64()),       # ACTUAL runtime — outcome, never a model input
    ("runtime_est_s", pa.float64()),    # prediction known at submit — model input
    ("num_gpus", pa.float64()),         # may be fractional (GPU share)
    ("gpu_type_req", pa.string()),      # None = any type
    ("vram_gb", pa.float64()),
    ("cpu_cores", pa.float64()),
    ("mem_gb", pa.float64()),
    ("job_type", pa.string()),
    ("priority", pa.int64()),           # 1..10
    ("deadline_ts", pa.float64()),      # absolute; online_inference = submit + slo_s
    ("user_group", pa.string()),
    ("final_status", pa.string()),      # pass | failed | killed (outcome only)
    ("vram_gb_src", pa.string()),
    ("job_type_src", pa.string()),
    ("priority_src", pa.string()),
    ("deadline_src", pa.string()),
    ("runtime_est_src", pa.string()),
])

COLUMNS = [field.name for field in JOB_SCHEMA]
FLAG_COLUMNS = ["vram_gb_src", "job_type_src", "priority_src", "deadline_src", "runtime_est_src"]

# Columns that must be non-null in every row (gpu_type_req / mem_gb / user_group may be).
REQUIRED = [c for c in COLUMNS if c not in ("gpu_type_req", "mem_gb", "user_group")]


def validate_jobs(df: pd.DataFrame) -> list[str]:
    """Return a list of violation messages; empty list means the frame is valid."""
    violations: list[str] = []
    missing = set(COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(COLUMNS)
    if missing:
        violations.append(f"missing columns: {sorted(missing)}")
    if extra:
        violations.append(f"unexpected columns: {sorted(extra)}")
    if missing:
        return violations

    for col in REQUIRED:
        n = int(df[col].isna().sum())
        if n:
            violations.append(f"{col}: {n} null values")

    checks: list[tuple[str, pd.Series]] = [
        ("duration_s must be >= 0", df["duration_s"] < 0),
        ("submit_ts must be >= 0", df["submit_ts"] < 0),
        ("num_gpus must be > 0", df["num_gpus"] <= 0),
        ("vram_gb must be > 0", df["vram_gb"] <= 0),
        ("cpu_cores must be >= 0", df["cpu_cores"] < 0),
        ("mem_gb must be >= 0 where present", df["mem_gb"] < 0),
        ("priority must be in 1..10", ~df["priority"].between(1, 10)),
        ("deadline_ts must be >= submit_ts", df["deadline_ts"] < df["submit_ts"]),
        ("runtime_est_s must be > 0", df["runtime_est_s"] <= 0),
        (f"job_type must be one of {JOB_TYPES}", ~df["job_type"].isin(JOB_TYPES)),
        (f"final_status must be one of {FINAL_STATUSES}", ~df["final_status"].isin(FINAL_STATUSES)),
    ]
    for name, bad in checks:
        n = int(bad.fillna(False).sum())
        if n:
            violations.append(f"{name}: {n} rows")
    for col in FLAG_COLUMNS:
        bad = ~df[col].isin(PROVENANCE_TAGS)
        if int(bad.sum()):
            violations.append(f"{col} must be one of {PROVENANCE_TAGS}: {int(bad.sum())} rows")
    dup = int(df.duplicated(subset=["dataset", "job_id"]).sum())
    if dup:
        violations.append(f"duplicate (dataset, job_id): {dup} rows")
    return violations
