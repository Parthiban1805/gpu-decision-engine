"""P4 benchmark: six heuristic schedulers on the pre-registered two-track
regimes (capped-1h + stress-24h), paired seeds, plus the pre-registered
analyses: rho sweep (primary), cap sensitivity (ranking stability only),
warm-up exclusion check, optional uncapped-1h cell, sanity check and the
horizon/straddle correction.

Re-run: python tasks.py baselines  (or .venv python scripts/bench_p4.py)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from gde.baselines import ALL_POLICIES
from gde.evaluation.benchmark import (
    Cell,
    aggregate,
    load_regimes,
    rank_by,
    rollout_straddle_fraction,
    run_cell_parallel,
)
from gde.ingestion.synthetic import SyntheticConfig, generate_workload
from gde.paths import CONFIGS, REPORTS
from gde.simulator.cluster import ClusterSpec
from gde.simulator.engine import Engine
from gde.simulator.workload import sim_jobs_from_unified_df


def _fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return "-"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def _run_cells(cells, regimes, overlay_raw, cfg, workers):
    args = [(c, regimes, overlay_raw, cfg["est_mode"], cfg["warmup_frac"]) for c in cells]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(run_cell_parallel, args))
    return [run_cell_parallel(a) for a in args]


def _main_table(agg, regime, rho, overlay, lines):
    lines.append(f"### regime: {regime} — rho {rho}, overlay {int(overlay * 100)}% "
                 f"(mean over paired seeds)")
    lines.append("")
    lines.append("| scheduler | util(AW) | rho hat tr | wait p50 h | wait p95 h | "
                 "DL miss | SLO miss(ov) | makespan d | frag |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    rows = {k: v for k, v in agg.items()
            if k[0] == regime and k[1] == rho and k[2] == overlay}
    for (reg, r, ov, sched), m in sorted(rows.items(), key=lambda kv: kv[1]["trace_wait_p50_s"]):
        lines.append(
            f"| {sched} | {_fmt(m['util_aw'])} | {_fmt(m['realized_rho_trace'])} "
            f"| {_fmt((m['trace_wait_p50_s'] or 0) / 3600, 2)} "
            f"| {_fmt((m['trace_wait_p95_s'] or 0) / 3600, 2)} "
            f"| {_fmt(m['trace_dl_miss'])} | {_fmt(m['ov_slo_miss'])} "
            f"| {_fmt((m['makespan_s'] or 0) / 86400, 2)} | {_fmt(m['frag_mean'])} |")
    lines.append("")


def sanity_check(cfg) -> dict:
    """PLAN P4 gate: EDF <= FIFO on deadline misses in a deadline-heavy mix."""
    sc = SyntheticConfig(
        n_jobs=cfg["sanity_check"]["n_jobs"],
        rate_per_s=cfg["sanity_check"]["rate_per_s"],
        job_mix=cfg["sanity_check"]["mix"], mode="poisson")
    df = generate_workload(sc, seed=cfg["sanity_check"]["seed"])
    cluster = ClusterSpec.homogeneous(cfg["sanity_check"]["cluster_gpus"], vram_gb=24.0)
    jobs, _ = sim_jobs_from_unified_df(df, cluster=cluster)
    out = {}
    for name in ("fifo", "edf"):
        horizon = max(j.submit_ts for j in jobs)
        res = Engine(jobs, cluster, ALL_POLICIES[name](), arrival_horizon=horizon).run()
        out[name] = res.summary["groups"]["trace"]["deadline_miss_rate"]
    out["pass"] = out["edf"] <= out["fifo"]
    return out


def ranking_of(agg, regime, rho, overlay, metric) -> list[str]:
    return [s for s, _ in rank_by(agg, regime, rho, overlay, metric)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIGS / "experiment_p4.yaml"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--serial", action="store_true", help="no multiprocessing")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    regimes = load_regimes()
    overlay_raw = yaml.safe_load((CONFIGS / "sim_p3.yaml").read_text(encoding="utf-8"))["overlay"]
    seeds = regimes["benchmark_seeds"]
    schedulers = cfg["schedulers"]
    workers = 1 if args.serial else args.workers

    t_start = time.perf_counter()
    lines: list[str] = []
    lines.append("# P4 — heuristic baselines on the pre-registered regimes")
    lines.append("")
    lines.append(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%S')} by `python tasks.py baselines`. "
                 "Regimes: `configs/regimes.yaml` (FROZEN, pre-registered in "
                 "`reports/sim/regime_prereg.md` before any P4 result). "
                 f"Schedulers: {', '.join(schedulers)}. "
                 f"Paired seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)}), disjoint from the "
                 "P3b diagnostic seeds. Est mode: stored.")
    lines.append("")
    lines.append("**Headline disclosures (binding):** on capped-1h, **64% of offered work is "
                 "excluded by the 6 h duration cap** (measured, `reports/sim/p3b_regime.md`); "
                 "**deadlines/SLOs/VRAM are synthetic** for all trace jobs; `batch_inference` "
                 "is a **short-small proxy** (derived, ≤600 s & ≤0.5 GPU, not tuned); overlay "
                 "jobs are fully synthetic. Every table below carries its regime tag.")
    lines.append("")

    # ---- main matrix -----------------------------------------------------
    cells = []
    primary_rhos = regimes["primary"]["rho_sweep"]
    for rho in primary_rhos:
        for ov in (cfg["overlay_shares"] if rho == regimes["primary"]["rho_default"] else [0.25]):
            for s in schedulers:
                for seed in seeds:
                    cells.append(Cell(s, "capped-1h", rho, ov, seed))
    for ov in cfg["overlay_shares"]:
        for s in schedulers:
            for seed in seeds:
                cells.append(Cell(s, "stress-24h", 0.8, ov, seed))
    for s in schedulers:  # optional uncapped-1h cell
        for seed in seeds:
            cells.append(Cell(s, "uncapped-1h", 0.8, 0.25, seed))

    print(f"main matrix: {len(cells)} cells, workers={workers}", flush=True)
    results = _run_cells(cells, regimes, overlay_raw, cfg, workers)
    bad = [r for r in results if r["warnings"]]
    if bad:
        raise SystemExit(f"cells with warnings (deadlock/guard): {[r['cell'] for r in bad]}")

    agg = aggregate(results)
    agg_warm = aggregate(results, use_warmup=True)

    lines.append("## Main comparison (rho 0.8)")
    lines.append("")
    _main_table(agg, "capped-1h", 0.8, 0.0, lines)
    _main_table(agg, "capped-1h", 0.8, 0.25, lines)
    _main_table(agg, "stress-24h", 0.8, 0.0, lines)
    _main_table(agg, "stress-24h", 0.8, 0.25, lines)
    lines.append("### regime: uncapped-1h — rho 0.8, overlay 25% (optional cheap cell)")
    lines.append("")
    lines.append("| scheduler | util(AW) | wait p50 h | wait p95 h | DL miss | SLO miss(ov) | makespan d |")
    lines.append("|---|---|---|---|---|---|---|")
    for (reg, r, ov, sched), m in sorted(
            ((k, v) for k, v in agg.items() if k[0] == "uncapped-1h"),
            key=lambda kv: kv[1]["trace_wait_p50_s"]):
        lines.append(
            f"| {sched} | {_fmt(m['util_aw'])} "
            f"| {_fmt((m['trace_wait_p50_s'] or 0) / 3600, 2)} "
            f"| {_fmt((m['trace_wait_p95_s'] or 0) / 3600, 2)} "
            f"| {_fmt(m['trace_dl_miss'])} | {_fmt(m['ov_slo_miss'])} "
            f"| {_fmt((m['makespan_s'] or 0) / 86400, 2)} |")
    lines.append("")

    # ---- rho sweep (primary only) -----------------------------------------
    lines.append("## rho sweep — regime: capped-1h (overlay 25%)")
    lines.append("")
    lines.append("| scheduler | rho | util(AW) | wait p50 h | DL miss | SLO miss(ov) | makespan d |")
    lines.append("|---|---|---|---|---|---|---|")
    for rho in primary_rhos:
        for (reg, r, ov, sched), m in sorted(
                ((k, v) for k, v in agg.items()
                 if k[0] == "capped-1h" and k[1] == rho and k[2] == 0.25),
                key=lambda kv: (kv[0][3])):
            lines.append(
                f"| {sched} | {rho} | {_fmt(m['util_aw'])} "
                f"| {_fmt((m['trace_wait_p50_s'] or 0) / 3600, 2)} "
                f"| {_fmt(m['trace_dl_miss'])} | {_fmt(m['ov_slo_miss'])} "
                f"| {_fmt((m['makespan_s'] or 0) / 86400, 2)} |")
    lines.append("")

    # ---- sanity check ------------------------------------------------------
    sc = sanity_check(cfg)
    lines.append("## Ranking sanity (PLAN P4 gate)")
    lines.append("")
    lines.append(f"- Deadline-heavy synthetic mix "
                 f"({cfg['sanity_check']['mix']}, {cfg['sanity_check']['n_jobs']} jobs): "
                 f"EDF deadline miss **{_fmt(sc['edf'])}** vs FIFO **{_fmt(sc['fifo'])}** "
                 f"-> EDF <= FIFO: **{'PASS' if sc['pass'] else 'FAIL'}**")
    lines.append("")

    # ---- cap sensitivity (ranking stability only) ---------------------------
    lines.append("## Cap sensitivity — regime family capped-1h (3 h / 6 h / 12 h)")
    lines.append("")
    lines.append("*Pre-registered scope: report ONLY whether the scheduler ranking is "
                 "stable. The 6 h cap is frozen and is NOT re-picked. Rollout/harness "
                 "config is identical at every cap: " +
                 cfg["cap_sensitivity"]["rollout_policy_note"].replace("\n", " ").strip() + "*")
    lines.append("")
    cap_results = {}
    sensitivity_cells_run = 0
    for cap in cfg["cap_sensitivity"]["caps_s"]:
        reg_copy = json.loads(json.dumps(regimes))  # deep copy for the override
        reg_copy["primary"]["duration_cap_s"] = cap
        cells_cap = [Cell(s, "capped-1h", cfg["cap_sensitivity"]["rho"],
                          cfg["cap_sensitivity"]["overlay_share"], seed)
                     for s in schedulers for seed in seeds]
        print(f"cap {cap}s: {len(cells_cap)} cells", flush=True)
        results_cap = _run_cells(cells_cap, reg_copy, overlay_raw, cfg, workers)
        sensitivity_cells_run += len(cells_cap)
        cap_results[cap] = aggregate(results_cap)
    lines.append("| cap | ranking by trace wait p50 | ranking by DL miss | stable vs 6 h (both) |")
    lines.append("|---|---|---|---|")
    base_wait = ranking_of(cap_results[21600.0], "capped-1h", 0.8, 0.25, "trace_wait_p50_s")
    base_dl = ranking_of(cap_results[21600.0], "capped-1h", 0.8, 0.25, "trace_dl_miss")
    stability = {}
    for cap, agg_cap in cap_results.items():
        rw = ranking_of(agg_cap, "capped-1h", 0.8, 0.25, "trace_wait_p50_s")
        rd = ranking_of(agg_cap, "capped-1h", 0.8, 0.25, "trace_dl_miss")
        same = (rw == base_wait and rd == base_dl)
        stability[f"{cap}s"] = same
        lines.append(f"| {cap / 3600:.0f} h | {' > '.join(rw)} | {' > '.join(rd)} | "
                     f"{'YES' if same else 'NO'} |")
    lines.append("")

    # ---- warm-up check ------------------------------------------------------
    lines.append("## Warm-up exclusion check — regime: capped-1h (rho 0.8, overlay 25%)")
    lines.append("")
    lines.append(f"Metrics exclude jobs submitted in the first {int(cfg['warmup_frac'] * 100)}% "
                 "of the arrival window (initial backlog/warm-up).")
    lines.append("")
    lines.append("| scheduler | wait p50 h (all) | wait p50 h (warm-excl) | DL miss (all) | DL miss (warm-excl) |")
    lines.append("|---|---|---|---|---|")
    rank_all = ranking_of(agg, "capped-1h", 0.8, 0.25, "trace_wait_p50_s")
    rank_warm = ranking_of(agg_warm, "capped-1h", 0.8, 0.25, "trace_wait_p50_s")
    for sched in schedulers:
        m = agg[("capped-1h", 0.8, 0.25, sched)]
        mw = agg_warm[("capped-1h", 0.8, 0.25, sched)]
        lines.append(
            f"| {sched} | {_fmt((m['trace_wait_p50_s'] or 0) / 3600, 2)} "
            f"| {_fmt((mw['trace_wait_p50_s'] or 0) / 3600, 2)} "
            f"| {_fmt(m['trace_dl_miss'])} | {_fmt(mw['trace_dl_miss'])} |")
    lines.append("")
    lines.append(f"Ranking by wait p50 unchanged after warm-up exclusion: "
                 f"**{'YES' if rank_all == rank_warm else 'NO'}** "
                 f"(all: {' > '.join(rank_all)}; warm-excl: {' > '.join(rank_warm)}).")
    lines.append("")

    # ---- straddle / horizon claim correction (step 6) ------------------------
    lines.append("## Horizon clarification (P3b claim correction)")
    lines.append("")
    lines.append("The P3b sentence \"no multi-day job straddles the 3600 s horizon\" meant "
                 "**multi-day RUNNING jobs** (excluded by the 6 h cap) — not that rollouts "
                 "end with an idle cluster. Measured below: fraction of mid-run rollouts "
                 "(capped-1h, FIFO, seed 31001, sampled every "
                 f"{cfg['straddle']['sample_every']}th decision point) that still have "
                 "running jobs when the horizon binds:")
    lines.append("")
    lines.append("| horizon | binds by | rollouts sampled | frac ending with running jobs |")
    lines.append("|---|---|---|---|")
    straddle_raw = {}
    for h in cfg["straddle"]["horizons"]:
        out = rollout_straddle_fraction(
            cfg["straddle"]["scheduler"], "capped-1h", seeds[0], regimes, overlay_raw,
            n_arrivals=h["n_arrivals"], max_seconds=h["max_seconds"],
            sample_every=cfg["straddle"]["sample_every"])
        straddle_raw[h["name"]] = out
        lines.append(
            f"| {h['name']} (n_arr={h['n_arrivals']}, max_s={h['max_seconds']}) "
            f"| {'arrivals' if (out['frac_bound_by_arrivals'] or 0) > 0.5 else 'seconds'} "
            f"| {out['n_rollouts']} | {_fmt(out['frac_with_running_jobs'])} |")
    lines.append("")
    lines.append("So 6 h-capped jobs routinely straddle horizon end; what the cap removes is "
                 "multi-day jobs. Labeling (P6) uses the primary horizon "
                 "(n_arrivals=20 binds) with terminal penalties, per `configs/regimes.yaml`.")
    lines.append("")

    # ---- provenance / mix ----------------------------------------------------
    r0 = next(r for r in results if r["cell"]["regime"] == "capped-1h")
    tman = r0["trace_manifest"]
    lines.append("## Regime provenance and job-type mix")
    lines.append("")
    lines.append(f"- capped-1h example cell (seed {seeds[0]}): window_start "
                 f"{tman['window_start']:.0f}, n_in_window {tman['n_jobs_in_window']}, "
                 f"cap drops {tman['n_cap_dropped']}, infeasible drops {tman['dropped_total']}, "
                 f"n_jobs {tman['n_jobs']}, W {tman['work_gpu_s'] / 1e6:.2f}M GPU-s, "
                 f"T_sim {tman['t_sim_s'] / 86400:.2f} d, scale "
                 f"{tman['arrival_scale_factor']:.4f}.")
    lines.append("- All trace-derived fields carry P2 provenance (VRAM derived/synthetic; "
                 "job_type derived; priority/deadline synthetic; runtime_est derived, "
                 "MdAPE 45.3%). Overlay fields fully synthetic.")
    lines.append("- Any 'scheduler X wins' statement below must hold on stress-24h or is "
                 "scoped to capped-1h. Stress track is held out from P6 labeling.")
    lines.append("")

    wall_total = time.perf_counter() - t_start
    total_events = sum(r["events"] for r in results)
    lines.append("## Cost")
    lines.append("")
    lines.append(f"- {len(results)} main-matrix cells + {len(sensitivity_cells_run)} "
                 f"sensitivity cells; {total_events:,} events total; total wall "
                 f"{wall_total / 60:.1f} min (workers={workers}, CPU-only).")
    lines.append("")

    out_md = REPORTS / "p4" / "report.md"
    out_json = REPORTS / "p4" / "results.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    json.dump({
        "aggregate": {"/".join(map(str, k)): v for k, v in agg.items()},
        "aggregate_warmup": {"/".join(map(str, k)): v for k, v in agg_warm.items()},
        "cap_sensitivity": {str(c): {"/".join(map(str, k)): v for k, v in a.items()}
                            for c, a in cap_results.items()},
        "sanity": sc, "straddle": straddle_raw, "stability": stability,
        "cells": len(results), "wall_min": wall_total / 60,
    }, open(out_json, "w", encoding="utf-8"), indent=2, default=str)
    print(f"report: {out_md}")
    if not sc["pass"]:
        raise SystemExit("sanity check FAILED (EDF > FIFO on deadline misses)")


if __name__ == "__main__":
    main()
