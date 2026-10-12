"""P2b data-quality follow-up on the unified job Parquet (run after `python tasks.py normalize`).

Reports, per dataset:
- runtime_est_s quality vs duration_s: median absolute percentage error (MdAPE), Spearman
  rank correlation, and the share of jobs whose estimate is within a factor of 2 of the
  actual duration (0.5 <= est/actual <= 2).
- job_type shares and provenance (`*_src`) counts.

Reads only data/processed/*.parquet; writes reports/normalize/p2b_quality_report.md.
Spearman is computed as Pearson on average ranks (exact; avoids the scipy dependency).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gde import paths

PROV_COLS = ["vram_gb_src", "job_type_src", "priority_src", "deadline_src", "runtime_est_src"]


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation of average ranks == Spearman rho (ties get averaged ranks)."""
    ra, rb = a.rank(method="average"), b.rank(method="average")
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def quality_block(df: pd.DataFrame) -> dict[str, float]:
    est = df["runtime_est_s"].astype(float)
    act = df["duration_s"].astype(float)
    ok = act > 0
    est, act = est[ok], act[ok]
    ratio = est / act
    return {
        "n": float(ok.sum()),
        "mdape_pct": float((100.0 * (est - act).abs() / act).median()),
        "spearman": _spearman(est, act),
        "within_2x_pct": float(100.0 * ((ratio >= 0.5) & (ratio <= 2.0)).mean()),
        "over_est_pct": float(100.0 * (ratio > 2.0).mean()),
        "under_est_pct": float(100.0 * (ratio < 0.5).mean()),
    }


def run(datasets: list[str] | None = None, out: Path | None = None) -> str:
    proc = paths.DATA / "processed"
    files = sorted(proc.glob("*.parquet")) if datasets is None else [proc / f"{d}.parquet" for d in datasets]
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit(f"no parquet files under {proc}; run `python tasks.py normalize` first")

    lines = [
        "# P2b data-quality follow-up (runtime estimates, job_type, provenance)", "",
        f"Generated {dt.datetime.now().astimezone().isoformat(timespec='seconds')} by "
        "`python scripts/p2b_quality_report.py` from `data/processed/*.parquet`.", "",
        "`runtime_est_s` is the stored (second-pass) estimate: expanding geometric mean of "
        "PRIOR jobs' durations, grouped by (user_group, job_type) -> (job_type) -> global, "
        "cold-start constant 3600 s. Actual `duration_s` is an outcome, never an input.", "",
    ]
    print(lines[2])
    for f in files:
        df = pd.read_parquet(f)
        name = f.stem
        q = quality_block(df)
        print(f"\n== {name}: {int(q['n']):,} jobs")
        print(f"   MdAPE {q['mdape_pct']:.1f}% | Spearman {q['spearman']:.3f} | "
              f"within factor-2 {q['within_2x_pct']:.1f}% "
              f"(over>2x {q['over_est_pct']:.1f}%, under<0.5x {q['under_est_pct']:.1f}%)")
        jt = df["job_type"].value_counts()
        print("   job_type: " + ", ".join(f"{k} {v / len(df) * 100:.1f}% ({v:,})" for k, v in jt.items()))

        lines += [f"## {name}", "", f"- jobs: **{len(df):,}**",
                  f"- runtime_est_s quality: MdAPE **{q['mdape_pct']:.1f}%**, Spearman "
                  f"**{q['spearman']:.3f}**, within factor-2 of actual **{q['within_2x_pct']:.1f}%** "
                  f"(overestimate >2x: {q['over_est_pct']:.1f}%, underestimate <0.5x: {q['under_est_pct']:.1f}%)",
                  "- job_type: " + ", ".join(f"`{k}` {v / len(df) * 100:.1f}% ({v:,})" for k, v in jt.items())]
        for col in PROV_COLS:
            if col not in df.columns:
                continue
            vc = df[col].value_counts()
            lines.append(f"- {col}: " + ", ".join(f"`{k}` {v:,}" for k, v in vc.items()))
        lines.append("")
    text = "\n".join(lines) + "\n"
    out = out or paths.REPORTS / "normalize" / "p2b_quality_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nreport -> {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of parquet stems (default: all)")
    args = ap.parse_args()
    run(args.datasets)
    sys.exit(0)
