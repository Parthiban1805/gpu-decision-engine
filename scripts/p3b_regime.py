"""P3b regime diagnostic (report only — no simulator or data changes).

Questions (user, 2026-10-12):
1. alibaba2020 smoke window + one other seeded window: duration percentiles of
   the feasible jobs and the share of offered work W from the top 0.1/1/10% of
   jobs (ranked by W contribution).
2. job count, T_sim, realized rho for 1 h / 6 h / 24 h windows on cluster_small
   (8 GPUs) and cluster_medium (64 GPUs), rho target 0.8, FIFO, no overlay.
3-4. options + recommendation live in the printed/report narrative.

Re-run: python tasks.py p3b  (or .venv python scripts/p3b_regime.py)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gde.baselines.fifo import FifoPolicy
from gde.paths import CONFIGS, DATA, REPORTS
from gde.simulator.cluster import ClusterSpec
from gde.simulator.engine import Engine
from gde.simulator.workload import TraceConfig, load_trace_window

PARQUET = DATA / "processed" / "alibaba2020.parquet"
SEED_A = 30001  # the smoke window seed
SEED_B = 30002


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q * 100))


def window_stats(cluster: ClusterSpec, window_s: float, seed: int) -> dict:
    cfg = TraceConfig(window_s=window_s, rho=0.8, est_mode="stored",
                      max_jobs=10**9, gpu_type="ignore")
    jobs, manifest = load_trace_window(PARQUET, cluster, cfg, seed)
    durs = [j.duration_s for j in jobs]
    w = sorted((j.expected_gpu_seconds for j in jobs), reverse=True)
    total = sum(w)
    n = len(w)

    def top_share(frac: float) -> float:
        k = max(1, int(np.ceil(frac * n)))
        return sum(w[:k]) / total

    manifest.update({
        "n_jobs": n,
        "dur_p50_s": _pct(durs, 0.50), "dur_p90_s": _pct(durs, 0.90),
        "dur_p99_s": _pct(durs, 0.99), "dur_max_s": max(durs),
        "w_top0.1": top_share(0.001), "w_top1": top_share(0.01),
        "w_top10": top_share(0.10),
        "t_sim_days": manifest["t_sim_s"] / 86400.0,
    })
    return manifest


def regime_run(cluster: ClusterSpec, window_s: float, seed: int,
               duration_cap_s: float | None = None) -> dict:
    """FIFO run on ONE fixed window (same for all cells of a comparison);
    optional duration cap applied inside that window before replay."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path, _, n_drops = _window_parquet(window_s, seed, duration_cap_s, Path(td))
        cfg = TraceConfig(window_s=window_s, rho=0.8, est_mode="stored",
                          max_jobs=10**9, gpu_type="ignore")
        jobs, manifest = load_trace_window(path, cluster, cfg, seed)
    if duration_cap_s is not None:
        manifest["cap"] = {"cap_s": duration_cap_s, "n_dropped": n_drops}
    manifest["t_sim_days"] = manifest["t_sim_s"] / 86400.0
    t0 = time.perf_counter()
    engine = Engine(jobs, cluster, FifoPolicy(), arrival_horizon=manifest["t_sim_s"])
    result = engine.run()
    manifest["run_s"] = time.perf_counter() - t0
    manifest["realized_rho"] = result.summary["groups"]["trace"]["realized_load"]
    manifest["util_aw"] = result.summary["cluster"]["utilization_arrival_window"]
    manifest["makespan_days"] = result.summary["cluster"]["makespan_s"] / 86400.0
    manifest["wait_p50_min"] = result.summary["groups"]["trace"]["wait_s_p50"] / 60.0
    manifest["wait_p95_h"] = result.summary["groups"]["trace"]["wait_s_p95"] / 3600.0
    manifest["interarrival_s"] = manifest["t_sim_s"] / max(manifest["n_jobs"], 1)
    return manifest


def _window_parquet(window_s: float, seed: int, cap_s: float | None,
                    tmpdir: Path) -> tuple[Path, int, int]:
    """Slice ONE fixed window (same formula/seed as the replayer's first rng
    draw, so seed 30001 reproduces the smoke window), optionally apply the
    duration-cap filter inside it, and write a temp Parquet. load_trace_window
    on that slice then clamps to the slice span -> the same window content.
    Returns (path, n_in_window_after_cap, n_cap_drops_in_window)."""
    import pandas as pd

    df = pd.read_parquet(PARQUET)
    t_min = float(df["submit_ts"].min())
    t_max = float(df["submit_ts"].max())
    span = t_max - t_min
    ws = min(window_s, span)
    start = t_min + float(np.random.default_rng(seed).uniform(0.0, max(span - ws, 0.0)))
    win = df[(df["submit_ts"] >= start) & (df["submit_ts"] <= start + ws)]
    n_window = int(len(win))
    if cap_s is not None:
        win = win[win["duration_s"] <= cap_s]
    out = tmpdir / f"win_{int(window_s)}_{seed}_{int(cap_s or 0)}.parquet"
    win.to_parquet(out)
    return out, int(len(win)), n_window - int(len(win))


def cap_sensitivity(cluster: ClusterSpec, window_s: float, seed: int,
                    caps: tuple[float, ...]) -> list[dict]:
    """Duration-cap filter sensitivity inside ONE fixed window: effect on
    n/W/T_sim and on the load share a 25% overlay would carry (estimate,
    6 GPU-s per overlay job)."""
    import tempfile

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for cap in caps:
            path, _, n_drops = _window_parquet(
                window_s, seed, None if cap > 10**8 else cap, tmpdir)
            cfg = TraceConfig(window_s=window_s, rho=0.8, est_mode="stored",
                              max_jobs=10**9, gpu_type="ignore")
            jobs, _ = load_trace_window(path, cluster, cfg, seed)
            w = sum(j.expected_gpu_seconds for j in jobs)
            t_sim = w / (0.8 * cluster.n_gpus)
            overlay_load = (0.25 * len(jobs) * 6.0) / (cluster.n_gpus * t_sim) if jobs else 0.0
            rows.append({
                "cap_h": cap / 3600, "n_kept": len(jobs), "dropped": n_drops,
                "w_mgpu_s": w / 1e6, "t_sim_days": t_sim / 86400,
                "mean_gpu_s_per_job": w / len(jobs) if jobs else 0.0,
                "overlay25_load_share": overlay_load,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-runs", action="store_true",
                        help="skip the FIFO runs (tables 1 only)")
    args = parser.parse_args()

    small = ClusterSpec.from_yaml(CONFIGS / "cluster_small.yaml")
    medium = ClusterSpec.from_yaml(CONFIGS / "cluster_medium.yaml")

    lines: list[str] = ["# P3b regime diagnostic (alibaba2020, report-only)", ""]

    # ---- table 1: window heavy-tail profile (cluster_small feasibility) ----
    lines.append("## 1. Heavy-tail profile of feasible jobs (8x24 GB, 24 h windows)")
    lines.append("")
    lines.append("| window | seed | n feasible | dropped | p50 s | p90 s | p99 s | max s | "
                 "W top 0.1% | W top 1% | W top 10% | W total GPU-s | T_sim d |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for seed, label in ((SEED_A, "smoke (30001)"), (SEED_B, "alt (30002)")):
        m = window_stats(small, 86400.0, seed)
        lines.append(
            f"| {label} | {seed} | {m['n_jobs']} | {m['dropped_total']} "
            f"| {m['dur_p50_s']:.0f} | {m['dur_p90_s']:.0f} | {m['dur_p99_s']:.0f} "
            f"| {m['dur_max_s']:.0f} | {m['w_top0.1']:.3f} | {m['w_top1']:.3f} "
            f"| {m['w_top10']:.3f} | {m['work_gpu_s'] / 1e6:.1f}M | {m['t_sim_days']:.1f} |")
    lines.append("")

    if not args.skip_runs:
        lines.append("## 2. Window length x cluster (rho target 0.8, FIFO, no overlay)")
        lines.append("")
        lines.append("| cluster | window | n feasible | W GPU-s | T_sim (d) | realized rho | "
                     "util (AW) | makespan (d) | run wall s |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cluster, cname in ((small, "small 8x24"), (medium, "medium 64 hetero")):
            for window_s in (3600.0, 21600.0, 86400.0):
                m = regime_run(cluster, window_s, SEED_A)
                lines.append(
                    f"| {cname} | {window_s / 3600:.0f} h | {m['n_jobs']} "
                    f"| {m['work_gpu_s'] / 1e6:.2f}M | {m['t_sim_days']:.2f} "
                    f"| {m['realized_rho']:.3f} | {m['util_aw']:.3f} "
                    f"| {m['makespan_days']:.1f} | {m['run_s']:.0f} |")
        lines.append("")

        lines.append("## 3. Candidate benchmark regimes (measured, FIFO, rho target 0.8)")
        lines.append("")
        lines.append("| regime | n | cap drops | T_sim (d) | realized rho | makespan (d) | "
                     "wait p50 | wait p95 | inter-arrival |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cluster, cname, window_s, cap in (
            (small, "small, 1 h + 6 h cap", 3600.0, 3600.0 * 6),
            (small, "small, 1 h, no cap", 3600.0, None),
            (small, "small, 24 h + 6 h cap", 86400.0, 3600.0 * 6),
            (small, "small, 24 h, no cap (P3 smoke)", 86400.0, None),
        ):
            m = regime_run(cluster, window_s, SEED_A, duration_cap_s=cap)
            drops = m.get("cap", {}).get("n_dropped", 0)
            lines.append(
                f"| {cname} | {m['n_jobs']} | {drops} | {m['t_sim_days']:.2f} "
                f"| {m['realized_rho']:.3f} | {m['makespan_days']:.2f} "
                f"| {m['wait_p50_min']:.0f} min | {m['wait_p95_h']:.1f} h "
                f"| {m['interarrival_s']:.0f} s |")
        lines.append("")

    lines.append("## 4. Duration-cap sensitivity (cluster_small, 24 h window, seed 30001)")
    lines.append("")
    lines.append("A cap is a FILTER (tagged, drops logged), not a modification: kept jobs are "
                 "byte-identical. Overlay column = load share a 25% overlay (0.3 share x ~20 s "
                 "= 6 GPU-s/job) would carry on the capped regime.")
    lines.append("")
    lines.append("| cap | n kept | dropped | W GPU-s | T_sim (d) | mean GPU-s/job | overlay25 load share |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in cap_sensitivity(small, 86400.0, SEED_A, (3600.0 * 6, 86400.0, 10**9)):
        cap_label = "none" if row["cap_h"] > 1e8 else f"{row['cap_h']:.0f} h"
        lines.append(
            f"| {cap_label} | {row['n_kept']} | {row['dropped']} | {row['w_mgpu_s']:.1f}M "
            f"| {row['t_sim_days']:.2f} | {row['mean_gpu_s_per_job']:.0f} "
            f"| {row['overlay25_load_share']:.5f} |")
    lines.append("")

    lines.extend([
        "## 5. Options (trade-offs)",
        "",
        "**A. Duration-cap filter (6 h, tagged filter, drops logged, applied inside the "
        "fixed window before scaling).** Measured on the smoke window: drops 514/7695 "
        "in-window jobs (~p94 of durations) and 64% of W; T_sim 131 -> 47 d. Combined "
        "with 1 h windows: n=284, T_sim 3.05 d, realized rho 0.65, makespan 3.68 d "
        "(queue nearly drains inside the window), inter-arrival ~15.5 min. Strict FIFO "
        "still waits p50 16 h / p95 26 h there — that is the blocking floor P4 policies "
        "must beat, and at hour scale it is discriminative (uncapped: p50 21 d / p95 "
        "41 d). P6: the 3600 s rollout horizon sees ~4 arrivals and no multi-day job "
        "straddles the horizon. Overlay load share stays ~2e-4 in every regime "
        "(structural: mean trace job ~6k GPU-s vs ~6 GPU-s per overlay job) — the "
        "overlay's role is SLO/queue-position pressure, which priority-aware policies "
        "(P4) discriminate on. Cost: multi-day training jobs vanish from the primary "
        "regime, so it no longer represents the trace's GPU-hours; fragmentation "
        "stress from big jobs shrinks (1-6 h multi-GPU jobs remain); must be paired "
        "with an uncapped stress track so headline claims are also tested where "
        "whales dominate.",
        "",
        "**B. Short windows (1 h), no cap (existing knobs only).** n=308, T_sim 4.92 d, "
        "realized rho 0.69, waits p50 18.5 h / p95 28 h. No new semantics. Cost: whales "
        "still stretch individual windows erratically (the medium cluster flipped "
        "realized rho 0.56 -> 0.14 between window lengths because the last feasible "
        "arrival can sit hours before the window end, which the stretch maps to "
        "sim-days of idle); per-seed job counts are small (need >= 10 seeds; runs are "
        "0-8 s so that is cheap); P6 horizon is myopic in seconds terms (~2 arrivals "
        "per 3600 s) unless the n_arrivals=20 bound is what binds (config knob "
        "exists).",
        "",
        "**C. cluster_medium (64 heterogeneous GPUs).** 8x capacity absorbs the tail "
        "(24 h window: T_sim 46 d vs 131 d; 2.8x more work feasible incl. "
        "up-to-64-GPU jobs). Keeps realism and exercises best-fit across 3 GPU types. "
        "Cost: does not fix the tail (top 1% of jobs still carry ~40% of W); realized "
        "rho drifts far and erratically from target (0.14-0.56: min-compute-factor "
        "placement plus arrival patchiness); P6 states are 8x bigger (fork still "
        "ms-scale, measured). Better as the P8 scale/heterogeneity axis than as the "
        "labeling regime.",
        "",
        "## 6. Recommendation",
        "",
        "Two-track benchmark, both reported from P4 onward:",
        "",
        "1. **Primary (labeling + model comparison): cluster_small, 1 h windows, 6 h "
        "duration cap (tagged filter, in-window drops logged), rho target 0.8, >= 10 "
        "seeds.** Measured: 284 jobs/seed, T_sim 3.05 d, realized rho 0.65, makespan "
        "~= window, run < 1 s, P6 horizons see several arrivals with bounded terminal "
        "penalties, and metric scales (hours) where scheduler differences are visible.",
        "2. **Stress track (honesty): the P3 smoke regime (24 h window, uncapped) kept "
        "in every benchmark phase**, plus cluster_medium at P8 for scale/heterogeneity "
        "with its realized-rho drift measured and disclosed.",
        "",
        "Rationale: the cap is a filter (kept jobs byte-identical, drops logged; 6 h "
        "chosen at ~p94 of durations for scale, not searched for performance); the "
        "uncapped track preserves the heavy-tail blocking that EASY backfill must "
        "beat; overlay load share is ~1e-4 structurally in all regimes, so overlay "
        "effects are judged on SLO rates, reported separately from trace load as "
        "required.",
        "",
    ])

    text = "\n".join(lines)
    print(text)
    out = REPORTS / "sim" / "p3b_regime.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
