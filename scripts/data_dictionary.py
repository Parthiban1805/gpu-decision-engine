"""Generate docs/data_dictionary.md from the downloaded files (PLAN.md P1).

Column names come from the files themselves (or the trace's own .header files for headerless CSVs)
and are cross-checked against the columns documented in each dataset's README
(data/raw/<name>/README.upstream.md, saved at download time). Mismatches are reported, not hidden.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

from gde import paths

RAW = paths.DATA / "raw"
SAMPLE_ROWS = 100_000


def count_rows(path: Path, has_header: bool) -> int:
    with path.open("rb") as fh:
        n = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 24), b""))
    return n - (1 if has_header else 0)


# ------------------------------------------------------------ README column extraction

def readme_cols_alibaba2020(readme: str, table: str) -> list[str]:
    section = re.split(r"\n## ", readme)
    body = next((s for s in section if s.startswith(table)), "")
    return re.findall(r"^\|\s*([A-Za-z_][\w]*)\s*\|", body.split("**Note", 1)[0], re.MULTILINE)[1:]


def readme_cols_alibaba2023(readme: str, file: str) -> list[str]:
    anchor = {"openb_node_list_gpu_node.csv": "openb_node_list_all_node.csv",
              "openb_pod_list_gpuspec33.csv": "openb_pod_list_default.csv"}.get(file, file)
    body = readme.split(f"### [{anchor}]", 1)[-1]
    header = next(ln for ln in body.splitlines() if ln.startswith("|"))
    return [c.strip() for c in header.strip("|").split("|") if c.strip()]


def readme_cols_helios(readme: str, file: str) -> list[str]:
    heading = re.search(rf"^### `+{re.escape(file)}`+\s*$", readme, re.MULTILINE)
    if heading is None:
        return []
    body = readme[heading.end():].split("#### Schema", 1)[-1]
    body = re.split(r"\n### ", body, maxsplit=1)[0]
    return re.findall(r"^\|\s*`([^`]+)`\s*\|", body, re.MULTILINE)


# ------------------------------------------------------------ table rendering

def column_table(df: pd.DataFrame, sample_note: str) -> list[str]:
    lines = [f"| Column | dtype ({sample_note}) | non-null % | example |", "|---|---|---|---|"]
    for col in df.columns:
        s = df[col]
        ex = s.dropna()
        example = str(ex.iloc[0]) if not ex.empty else ""
        example = example if len(example) <= 40 else example[:37] + "..."
        lines.append(f"| `{col}` | {s.dtype} | {100 * s.notna().mean():.1f} | `{example}` |")
    return lines


def compare(actual: list[str], documented: list[str], pattern_cols: dict[str, str] | None = None) -> str:
    pattern_cols = pattern_cols or {}
    doc_plain = [d for d in documented if d not in pattern_cols]
    missing = [d for d in doc_plain if d not in actual]
    extra = [a for a in actual if a not in doc_plain
             and not any(re.fullmatch(p, a) for d, p in pattern_cols.items() if d in documented)]
    if not documented:
        return "README documents no column list for this file."
    if not missing and not extra:
        return "Columns match the README."
    return (f"**Mismatch vs README** — documented but absent: {missing or 'none'}; "
            f"present but undocumented: {extra or 'none'}.")


def file_section(title: str, path: Path, df: pd.DataFrame, rows: int, check: str, header_src: str) -> list[str]:
    return [f"### `{title}`", "",
            f"- Path: `{path.relative_to(paths.REPO_ROOT).as_posix()}` — {rows:,} rows; column names from {header_src}.",
            f"- README check: {check}", "",
            *column_table(df, f"first {min(rows, SAMPLE_ROWS):,} rows"), ""]


# ------------------------------------------------------------ datasets

def alibaba2020(d: Path) -> list[str]:
    readme = (d / "README.upstream.md").read_text(encoding="utf-8")
    out: list[str] = []
    for hdr in sorted(d.glob("*.header")):
        table, csv = hdr.stem, d / f"{hdr.stem}.csv"
        if not csv.exists():
            continue
        cols = hdr.read_text(encoding="utf-8").strip().split(",")
        df = pd.read_csv(csv, header=None, names=cols, nrows=SAMPLE_ROWS, low_memory=False)
        out += file_section(f"{table}.csv", csv, df, count_rows(csv, False),
                            compare(cols, readme_cols_alibaba2020(readme, table)),
                            f"`{hdr.name}` (CSV has no header row)")
    return out


def alibaba2023(d: Path) -> list[str]:
    readme = (d / "README.upstream.md").read_text(encoding="utf-8")
    out: list[str] = []
    for csv in sorted((d / "csv").glob("*.csv")):
        df = pd.read_csv(csv, nrows=SAMPLE_ROWS)
        out += file_section(csv.name, csv, df, count_rows(csv, True),
                            compare(list(df.columns), readme_cols_alibaba2023(readme, csv.name)), "the CSV header row")
    return out


def helios(d: Path) -> list[str]:
    readme = (d / "README.upstream.md").read_text(encoding="utf-8")
    out: list[str] = []
    for cluster in sorted(p for p in (d / "data").iterdir() if p.is_dir()):
        for name in ("cluster_log.csv", "cluster_gpu_number.csv"):
            csv = cluster / name
            df = pd.read_csv(csv, nrows=SAMPLE_ROWS, low_memory=False)
            documented = readme_cols_helios(readme, name)
            check = compare(list(df.columns), documented, {"vc***": r"vc\w+"})
            if name == "cluster_gpu_number.csv":
                df = df[["date", *[c for c in df.columns if c != "date"][:3], "total"]]
                check += f" ({len(pd.read_csv(csv, nrows=1).columns) - 2} VC columns; first 3 shown.)"
            out += file_section(f"{cluster.name}/{name}", csv, df, count_rows(csv, True), check, "the CSV header row")
    return out


DATASETS = {"alibaba2020": alibaba2020, "alibaba2023": alibaba2023, "helios": helios}

NOTES: dict[str, list[str]] = {
    "alibaba2020": [
        "Time: `start_time`/`end_time` are seconds with a constant offset removed (README); in UTC+8 they keep "
        "the original time of day and weekday but fake dates.",
        "`pai_job_table.start_time` is job **submission**; `pai_task_table.start_time` is task launch. "
        "README: the gap between job.start_time and the earliest task.start_time is the wait (scheduling latency). "
        "**Measured in P1 EDA: that gap is 0 s for all 732,355 Terminated jobs** (p99 and max both 0), so the trace "
        "carries no usable queueing delay; it gives arrivals, requests and runtimes, and waits must come from the "
        "simulator.",
        "`inst_id` should be treated as the job id (README); joins to `pai_group_tag_table.inst_id`.",
        "`plan_gpu` is a percentage per instance (50.0 = half a GPU); job GPU request = Σ plan_gpu × inst_num / 100. "
        "`plan_cpu` is percent of a core (600 = 6 cores); `plan_mem` is GB.",
        "`gpu_type` in the task table is the type **assigned** (outcome-ish); `gpu_type_spec` in the group-tag table "
        "is the type **requested** (empty = any) — use the latter as a decision-time feature.",
        "`status` values per README: Running, Terminated, Failed, Waiting; only Terminated is successful.",
        "No explicit training/inference column (PLAN.md §2.2 assumed Alibaba has one; it does not in these tables). "
        "Proxies: `task_name` (tensorflow 49%, worker 22%, ps 15%, PyTorchWorker 9%, evaluator 1.4%, ...) and "
        "`group_tag.workload` (tagged for ~10% of jobs: bert, ctr, nmt, ...). Any job_type mapping is `derived`.",
        "No VRAM requirement column in the downloaded tables. `pai_sensor_table` (not downloaded by default, 388 MB) has "
        "measured `max_gpu_wrk_mem` per instance — usage, an outcome, not a request.",
    ],
    "alibaba2023": [
        "Times (`creation_time`, `scheduled_time`, `deletion_time`) are seconds (README).",
        "`gpu_milli` is the GPU share for GPU-sharing pods (num_gpu == 1), in milli-GPU (README).",
        "`gpu_spec` lists allowed GPU types separated by `|`; missing = unconstrained. `openb_pod_list_gpuspec33.csv` is "
        "the default list augmented with GPU-type constraints (README); `openb_node_list_gpu_node.csv` is the GPU-node "
        "subset of `openb_node_list_all_node.csv`.",
        "No train/inference field; `qos` (LS / Burstable / BE) is the closest proxy. No license file upstream.",
    ],
    "helios": [
        "One `cluster_log.csv` + `cluster_gpu_number.csv` per cluster (Earth, Saturn, Uranus, Venus).",
        "`submit_time`/`start_time`/`end_time` are datetime strings; `duration` = end − start and `queue` = start − submit, "
        "both seconds (README notes 3 and 4).",
        "`state`: COMPLETED, CANCELLED, FAILED, TIMEOUT, NODE_FAIL (README treats the last two as failed).",
        "No job-type, GPU-type, memory or VRAM fields. `gpu_num` = 0 means a CPU job.",
        "`cluster_gpu_number.csv`: daily GPU count per virtual cluster (`vc*` columns) plus `total`.",
    ],
}


def main() -> None:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    doc = ["# Data dictionary (raw traces)", "",
           f"Generated {now} by `scripts/data_dictionary.py` (run by `python tasks.py data`) from the files "
           "under `data/raw/`. Column names are read from the files (or the trace's `.header` files) and checked "
           "against each dataset's README. Acquisition status: `data/raw/DOWNLOADS.md`.", ""]
    for name, fn in DATASETS.items():
        d = RAW / name
        doc += [f"## {name}", ""]
        if not (d / "README.upstream.md").exists():
            doc += ["Not downloaded; see `data/raw/DOWNLOADS.md`.", ""]
            continue
        doc += ["Notes from the README (relevant to normalization in P2):", "", *[f"- {n}" for n in NOTES[name]], ""]
        doc += fn(d)
    doc += ["## philly", "", "Not downloaded (awaiting user approval; ~0.98 GB compressed). Schema per its README: "
            "`cluster_job_log` (JSON: status, vc, jobid, attempts[start_time, end_time, detail[ip, gpus]], "
            "submitted_time, user), `cluster_machine_list` (machineId, number of GPUs, single GPU mem), plus per-minute "
            "GPU/CPU/memory utilization CSVs. Column names will be verified against the files when downloaded.", ""]
    out = paths.DOCS / "data_dictionary.md"
    out.write_text("\n".join(doc), encoding="utf-8")
    print(f"wrote {out.relative_to(paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
