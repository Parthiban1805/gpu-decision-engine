"""P1 exploratory summary of the downloaded raw traces. Prints and writes reports/eda/p1_summary.md.

Per dataset: job counts, duration and GPU-request distributions, job-type shares (or the closest
proxy the trace offers; stated explicitly when a trace has no job-type field). Raw data only; no
normalization (that is P2).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from gde import paths

RAW = paths.DATA / "raw"
QS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


def qtable(s: pd.Series, unit: str, label: str) -> list[str]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return [f"{label}: no data", ""]
    q = s.quantile(QS)
    head = "| n | mean | " + " | ".join(f"p{int(x * 100)}" for x in QS) + " | max |"
    row = f"| {len(s):,} | {s.mean():,.1f} | " + " | ".join(f"{v:,.1f}" for v in q) + f" | {s.max():,.1f} |"
    return [f"{label} ({unit}):", "", head, "|" + "---|" * (len(QS) + 3), row, ""]


def shares(s: pd.Series, label: str, top: int = 12) -> list[str]:
    s = s.fillna("<missing>").astype(str).replace("", "<empty>")
    vc = s.value_counts()
    total = int(vc.sum())
    rows = [f"| {k} | {v:,} | {100 * v / total:.1f}% |" for k, v in vc.head(top).items()]
    if len(vc) > top:
        rest = int(vc.iloc[top:].sum())
        rows.append(f"| ({len(vc) - top} others) | {rest:,} | {100 * rest / total:.1f}% |")
    return [f"{label} (n={total:,}):", "", "| value | count | share |", "|---|---|---|", *rows, ""]


def bins(s: pd.Series, edges: list[float], labels: list[str], label: str) -> list[str]:
    cut = pd.cut(pd.to_numeric(s, errors="coerce"), bins=edges, labels=labels, right=True, include_lowest=True)
    return shares(cut.astype(str), label, top=len(labels) + 1)


def read_headerless(dir_: Path, table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    cols = (dir_ / f"{table}.header").read_text(encoding="utf-8").strip().split(",")
    return pd.read_csv(dir_ / f"{table}.csv", header=None, names=cols, usecols=usecols, low_memory=False)


# ---------------------------------------------------------------- per dataset

def eda_alibaba2020(d: Path) -> list[str]:
    out = ["## alibaba2020 (cluster-trace-gpu-v2020, PAI)", ""]
    jobs = read_headerless(d, "pai_job_table")
    tasks = read_headerless(d, "pai_task_table")
    span = (jobs["end_time"].max() - jobs["start_time"].min()) / 86400
    out += [f"- Jobs: **{len(jobs):,}** (rows in pai_job_table; job_name unique: {jobs['job_name'].is_unique}); "
            f"tasks: {len(tasks):,}; users: {jobs['user'].nunique():,}; span ≈ {span:.1f} days.", ""]
    out += shares(jobs["status"], "Job status (README: only 'Terminated' is successful)")

    t = tasks.copy()
    t["gpus"] = t["plan_gpu"].fillna(0) * t["inst_num"].fillna(0) / 100.0
    agg = t.groupby("job_name").agg(task_start=("start_time", "min"), task_end=("end_time", "max"),
                                    gpus=("gpus", "sum"), n_tasks=("task_name", "size"),
                                    max_inst_plan_gpu=("plan_gpu", "max"))
    j = jobs.set_index("job_name").join(agg, how="left")
    j["runtime_s"] = j["task_end"] - j["task_start"]
    j["wait_s"] = j["task_start"] - j["start_time"]
    term = j[j["status"] == "Terminated"]
    out += [f"Jobs with no task rows: {j['task_start'].isna().sum():,}.", ""]
    out += qtable(term["runtime_s"], "s", "Runtime of Terminated jobs = max(task.end_time) - min(task.start_time)")
    out += qtable(term["wait_s"], "s", "Wait of Terminated jobs = min(task.start_time) - job.start_time (submit)")
    out += qtable(j["gpus"], "GPUs", "GPU request per job = Σ plan_gpu × inst_num / 100 over tasks")
    out += bins(j["gpus"], [-0.001, 0, 0.999, 1, 2, 8, np.inf], ["0", "(0,1) fractional", "1", "(1,2]", "(2,8]", ">8"],
                "GPU request per job, binned")
    gpu_tasks = tasks[tasks["plan_gpu"].fillna(0) > 0]
    out += bins(gpu_tasks["plan_gpu"], [0, 25, 50, 99.999, 100, np.inf], ["(0,25]", "(25,50]", "(50,100)", "100", ">100"],
                "Per-instance plan_gpu (%) of GPU tasks")
    out += shares(tasks["gpu_type"], "Task gpu_type (assigned GPU type)")
    out += shares(tasks["task_name"], "Task role / task_name (job-type proxy; no explicit train/inference field)")
    gt = d / "pai_group_tag_table.csv"
    if gt.exists():
        tags = read_headerless(d, "pai_group_tag_table", usecols=["inst_id", "gpu_type_spec", "workload"])
        out += shares(tags["workload"], "group_tag.workload (README: ~9% of instances tagged)")
        out += shares(tags["gpu_type_spec"], "group_tag.gpu_type_spec (requested GPU type; empty = any)")
    ms = d / "pai_machine_spec.csv"
    if ms.exists():
        m = read_headerless(d, "pai_machine_spec")
        g = m.groupby("gpu_type").agg(machines=("machine", "size"), gpus=("cap_gpu", "sum"))
        out += ["Machines by gpu_type (pai_machine_spec):", "", "| gpu_type | machines | GPUs |", "|---|---|---|",
                *[f"| {k} | {r.machines:,} | {int(r.gpus):,} |" for k, r in g.iterrows()],
                f"| **total** | {len(m):,} | {int(m['cap_gpu'].sum()):,} |", ""]
    return out


def eda_helios(d: Path) -> list[str]:
    out = ["## helios (SenseTime HeliosData)", "",
           "Job type: **no job-type field** in `cluster_log.csv` (README schema); reported as N/A.", ""]
    clusters = sorted(p.name for p in (d / "data").iterdir() if (p / "cluster_log.csv").exists())
    summary = ["| cluster | jobs | GPU jobs | GPU-job share | median GPU-job duration (s) | GPUs (median daily total) | span (days) |",
               "|---|---|---|---|---|---|---|"]
    detail: list[str] = []
    for c in clusters:
        log = pd.read_csv(d / "data" / c / "cluster_log.csv", low_memory=False)
        gpu = log[log["gpu_num"] > 0]
        cap = pd.read_csv(d / "data" / c / "cluster_gpu_number.csv")
        submit = pd.to_datetime(log["submit_time"], errors="coerce")
        span = (submit.max() - submit.min()).total_seconds() / 86400
        summary.append(f"| {c} | {len(log):,} | {len(gpu):,} | {100 * len(gpu) / len(log):.1f}% | "
                       f"{gpu['duration'].median():,.0f} | {cap['total'].median():,.0f} | {span:.0f} |")
        detail += [f"### {c}", ""]
        detail += shares(log["state"], "Job state (all jobs)")
        detail += qtable(gpu["duration"], "s", "Duration of GPU jobs (README: end_time - start_time)")
        detail += qtable(gpu["queue"], "s", "Queue time of GPU jobs (README: start_time - submit_time)")
        detail += bins(log["gpu_num"], [-0.5, 0.5, 1, 4, 8, 16, np.inf], ["0 (CPU job)", "1", "2-4", "5-8", "9-16", ">16"],
                       "gpu_num per job (all jobs)")
    return out + summary + [""] + detail


def eda_alibaba2023(d: Path) -> list[str]:
    out = ["## alibaba2023 (cluster-trace-gpu-v2023, openb)", ""]
    pods = pd.read_csv(d / "csv" / "openb_pod_list_default.csv")
    nodes = pd.read_csv(d / "csv" / "openb_node_list_all_node.csv")
    out += [f"- `openb_pod_list_default.csv`: **{len(pods):,}** pods (tasks); `openb_node_list_all_node.csv`: "
            f"{len(nodes):,} nodes, {int(nodes['gpu'].sum()):,} GPUs.",
            "- Job type: no train/inference field; `qos` is the closest proxy (README: Burstable, BE, LS).", ""]
    out += shares(pods["qos"], "qos")
    out += shares(pods["pod_phase"], "pod_phase")
    dur = pods["deletion_time"] - pods["scheduled_time"]
    out += qtable(dur[pods["pod_phase"] == "Succeeded"], "s", "deletion_time - scheduled_time of Succeeded pods")
    out += shares(pods["num_gpu"], "num_gpu per pod")
    share = pods[(pods["num_gpu"] == 1) & (pods["gpu_milli"] < 1000)]
    out += [f"GPU-sharing pods (num_gpu==1, gpu_milli<1000): {len(share):,} "
            f"({100 * len(share) / max(1, (pods['num_gpu'] > 0).sum()):.1f}% of GPU pods).", ""]
    out += qtable(share["gpu_milli"], "milli-GPU", "gpu_milli of GPU-sharing pods")
    out += shares(pods["gpu_spec"], "gpu_spec (required GPU types; missing = unconstrained)")
    out += shares(nodes["model"], "Node GPU model (all nodes)")
    return out


EDA: dict[str, tuple[str, Callable[[Path], list[str]]]] = {
    "alibaba2020": ("pai_job_table.csv", eda_alibaba2020),
    "helios": ("data", eda_helios),
    "alibaba2023": ("csv/openb_pod_list_default.csv", eda_alibaba2023),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", choices=sorted(EDA), default=list(EDA))
    ap.add_argument("--out", type=Path, default=paths.REPORTS / "eda" / "p1_summary.md")
    args = ap.parse_args()

    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    doc = ["# P1 EDA summary (raw traces)", "",
           f"Generated {now} by `python tasks.py eda` (`scripts/eda.py`) from files under `data/raw/`. "
           "All values are computed from measured trace fields; nothing synthetic is used here.", ""]
    for name in args.datasets:
        marker, fn = EDA[name]
        d = RAW / name
        if not (d / marker).exists():
            doc += [f"## {name}", "", f"Not available locally (`{marker}` missing); see `data/raw/DOWNLOADS.md`.", ""]
            continue
        doc += fn(d)
    text = "\n".join(doc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    print(f"\nwrote {args.out.relative_to(paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
