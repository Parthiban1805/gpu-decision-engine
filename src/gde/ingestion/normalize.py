"""Normalize raw traces into unified job Parquet (PLAN.md §2.5) + cleaning report.

Pipeline per dataset: loader (measured fields, drop log) -> runtime estimates from past
jobs only -> job-type heuristic -> VRAM synthesis (derived for fractional shares, seeded
Beta for whole GPUs) -> synthetic deadlines (§2.3) -> schema validation -> Parquet.
Everything seeded/derived is echoed in the report (reports/normalize/cleaning_report.md).

Leakage notes (tested in tests/test_normalize.py):
- runtime_est_s uses only PRIOR jobs (grouped by user_group+job_type, then job_type).
- job_type is derived from runtime_est_s and num_gpus — both decision-time quantities —
  not from actual duration (PLAN §2.2 allows duration; we prefer the estimate).
- deadline_ts is synthetic and DOES use actual duration by design (a label-side field,
  disclosed); policies see the deadline but never duration_s.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from gde import paths
from gde.ingestion import load_alibaba2020, load_alibaba2023, load_helios
from gde.ingestion.runtime_predictor import add_runtime_estimates
from gde.ingestion.schema import validate_jobs


@dataclass
class DatasetResult:
    dataset: str
    rows_out: int = 0
    drops: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    stats: list[str] = field(default_factory=list)
    parquet: str = ""
    skipped: bool = False


def load_config() -> dict[str, Any]:
    return yaml.safe_load((paths.CONFIGS / "workload_synthesis.yaml").read_text(encoding="utf-8"))


def _builders(raw_root: Path) -> dict[str, Callable[[], tuple[pd.DataFrame, dict[str, int], list[str]]]]:
    out: dict[str, Callable[[], tuple[pd.DataFrame, dict[str, int], list[str]]]] = {
        "alibaba2020": lambda: load_alibaba2020.load(raw_root / "alibaba2020"),
        "alibaba2023": lambda: load_alibaba2023.load(raw_root / "alibaba2023"),
    }
    out.update({name: (lambda c=cluster: load_helios.load(c, raw_root / "helios"))
                for cluster, name in load_helios.DATASETS.items()})
    return out


def synthesize(df: pd.DataFrame, cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Add runtime_est_s, job_type, vram_gb, deadline_ts (+ provenance flags). Deterministic."""
    rng = np.random.default_rng(seed)
    df = df.sort_values(["submit_ts", "job_id"], kind="mergesort").reset_index(drop=True)

    # Two prior-only passes: job_type is classified from an estimate (decision-time), and
    # the final estimate then conditions on that job_type. Neither pass sees own/future
    # durations, so no circularity in the leakage sense.
    df = add_runtime_estimates(df, fallback_levels=(("user_group",), ()))
    jh = cfg["job_type_heuristic"]
    df["job_type"] = np.where(
        (df["runtime_est_s"] <= jh["short_job_s"]) & (df["num_gpus"] <= jh["small_gpu"]),
        "batch_inference", jh["default"])
    df["job_type_src"] = "derived"
    df = add_runtime_estimates(df)
    df["runtime_est_src"] = "derived"

    vc = cfg["vram"]
    mem = df["gpu_type_req"].map(vc["gpu_mem_gb"]).fillna(vc["default_mem_gb"]).astype(float)
    beta = vc["beta"]
    frac = np.clip(rng.beta(beta["alpha"], beta["beta"], size=len(df)), beta["min_frac"], 1.0)
    df["vram_gb"] = np.where(df["gpu_fractional"], df["num_gpus"] * mem,
                             df["num_gpus"] * frac * mem).astype(float)
    df["vram_gb_src"] = np.where(df["gpu_fractional"], "derived", "synthetic")
    df = df.drop(columns=["gpu_fractional"])          # loader helper, not part of the schema

    slack_cfg = cfg["deadline_slack"]

    def loguniform(spec: dict[str, float]) -> np.ndarray:
        return np.exp(rng.uniform(np.log(spec["min"]), np.log(spec["max"]), size=len(df)))

    oi = cfg["online_inference"]
    slo = np.asarray(rng.choice(oi["slo_s_choices"], size=len(df)), dtype=float)
    deadline = np.select(
        [df["job_type"] == "training", df["job_type"] == "batch_inference"],
        [df["submit_ts"] + loguniform(slack_cfg["training"]) * df["duration_s"],
         df["submit_ts"] + loguniform(slack_cfg["batch_inference"]) * df["duration_s"]],
        default=df["submit_ts"] + slo + oi["load_s"])
    df["deadline_ts"] = deadline.astype(float)
    df["deadline_src"] = "synthetic"
    return df


def _stats_block(df: pd.DataFrame) -> list[str]:
    q = df["duration_s"].quantile([0.5, 0.95]).round(1).tolist()
    gpu = df["num_gpus"].quantile([0.5, 0.95]).round(2).tolist()
    share = (df["job_type"].value_counts(normalize=True) * 100).round(1)
    status = (df["final_status"].value_counts(normalize=True) * 100).round(1)
    return [
        f"- rows out: **{len(df)}**; submit span: {df['submit_ts'].max() / 86400:.1f} days",
        f"- duration_s p50/p95: {q[0]} / {q[1]} s; num_gpus p50/p95: {gpu[0]} / {gpu[1]}",
        "- job_type: " + ", ".join(f"{k} {v}%" for k, v in share.items())
        + " (derived; online_inference never comes from these traces)",
        "- final_status: " + ", ".join(f"{k} {v}%" for k, v in status.items()),
    ]


def _report(results: list[DatasetResult], cfg: dict[str, Any]) -> str:
    lines = [
        "# P2 normalization: cleaning report", "",
        f"Generated {dt.datetime.now().astimezone().isoformat(timespec='seconds')} by "
        "`python tasks.py normalize` (`src/gde/ingestion/normalize.py`).", "",
        "## Synthetic / derived fields used (PLAN.md §2.2 disclosure)", "",
        "- `deadline_ts`: **synthetic** for every dataset (no trace has deadlines/SLOs). "
        "Training slack ~ LogUniform(2, 10), batch_inference ~ LogUniform(1.5, 4) around "
        "the ACTUAL duration (label-side field by design).",
        "- `vram_gb`: **derived** for fractional-GPU jobs (share x per-type VRAM constant), "
        "**synthetic** (seeded Beta) for whole-GPU jobs.",
        "- `priority`: **synthetic** default 5 for alibaba2020/helios (no class field); "
        "**derived** from `qos` for alibaba2023.",
        "- `job_type`: **derived** heuristic on runtime_est_s and num_gpus (no trace label).",
        "- `runtime_est_s`: **derived** expanding log-mean of PRIOR jobs only "
        "(user_group+job_type -> job_type -> global; first job gets a 3600 s constant).",
        "- GPU VRAM capacity constants are ASSUMPTIONS: "
        + ", ".join(f"{k}={v}GB" for k, v in cfg["vram"]["gpu_mem_gb"].items())
        + f", default {cfg['vram']['default_mem_gb']}GB.",
        "- Seeds: " + ", ".join(f"{k}={v}" for k, v in cfg["seeds"].items()) + ".", "",
    ]
    for r in results:
        lines += [f"## {r.dataset}", ""]
        if r.skipped:
            lines += [f"- SKIPPED: {r.notes[-1] if r.notes else 'raw data missing'}", ""]
            continue
        lines += r.notes + ["", "Drops (pipeline order):", "",
                            "| Reason | Rows |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in r.drops.items()] or ["| (none) | 0 |"]
        lines += [""] + r.stats + [""]
        if r.violations:
            lines += ["**SCHEMA VIOLATIONS:**"] + [f"- {v}" for v in r.violations]
        else:
            lines += [f"- schema validation: **PASS** -> `{r.parquet}`"]
        lines += [""]
    return "\n".join(lines) + "\n"


def run(datasets: list[str] | None = None, raw_root: Path | None = None,
        out_dir: Path | None = None, report_path: Path | None = None) -> list[DatasetResult]:
    cfg = load_config()
    raw_root = Path(raw_root) if raw_root else paths.DATA / "raw"
    out_dir = Path(out_dir) if out_dir else paths.DATA / "processed"
    report_path = report_path or paths.REPORTS / "normalize" / "cleaning_report.md"
    builders = _builders(raw_root)
    if datasets is None:
        datasets = list(builders)

    results: list[DatasetResult] = []
    for name in datasets:
        if name not in builders:
            raise ValueError(f"unknown dataset {name!r}; known: {sorted(builders)}")
        print(f"== {name}", flush=True)
        try:
            df, drops, notes = builders[name]()
        except FileNotFoundError as exc:
            print(f"   skipped (raw data missing): {exc}")
            results.append(DatasetResult(name, skipped=True, notes=[f"raw data missing: {exc}"]))
            continue
        df = synthesize(df, cfg, int(cfg["seeds"][name]))
        violations = validate_jobs(df)
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet = out_dir / f"{name}.parquet"
        df.to_parquet(parquet, engine="pyarrow", index=False)
        print(f"   {len(df)} rows -> {parquet.name} "
              f"({'VALID' if not violations else 'INVALID: ' + '; '.join(violations)})")
        results.append(DatasetResult(name, rows_out=len(df), drops=drops, notes=notes,
                                     violations=violations, stats=_stats_block(df),
                                     parquet=str(parquet.relative_to(paths.REPO_ROOT))))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(results, cfg), encoding="utf-8")
    print(f"report -> {report_path}")
    return results
