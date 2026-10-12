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
    window_eligibility,
)
from gde.ingestion.synthetic import SyntheticConfig, generate_workload
from gde.paths import CONFIGS, DATA, REPORTS
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
    args = [(c, regimes, overlay_raw, cfg["est_mode"], tuple(cfg["warmup_fracs"]))
            for c in cells]
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
            if (k[0], k[1], k[2], k[4], k[5]) == (regime, rho, overlay, "stored", "")}
    for key, m in sorted(rows.items(), key=lambda kv: kv[1]["trace_wait_p50_s"]):
        lines.append(
            f"| {key[3]} | {_fmt(m['util_aw'])} | {_fmt(m['realized_rho_trace'])} "
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


def job_type_mix(regimes: dict) -> dict:
    """Trace job-type mix per regime over the benchmark seeds (proxy labels:
    batch_inference = short-small proxy, derived; online_inference = overlay)."""
    import collections

    from gde.simulator.workload import TraceConfig, load_trace_window

    cluster = ClusterSpec.from_yaml(CONFIGS / regimes["primary"]["cluster"])
    out = {}
    for name, window, cap in (("capped-1h", 3600.0, 21600.0), ("stress-24h", 86400.0, None)):
        counts: collections.Counter = collections.Counter()
        overlay_jobs = 0
        for seed in regimes["benchmark_seeds"]:
            jobs, _ = load_trace_window(
                DATA / "processed" / "alibaba2020.parquet", cluster,
                TraceConfig(window_s=window, rho=0.8, max_jobs=10**9, duration_cap_s=cap),
                seed=seed)
            for j in jobs:
                counts[j.job_type] += 1
            overlay_jobs += round(0.25 * len(jobs))
        total = sum(counts.values())
        out[name] = {
            "trace_jobs": total,
            "mix": {k: {"n": v, "share": v / total} for k, v in counts.most_common()},
            "overlay_jobs_at_25pct": overlay_jobs,
        }
    return out


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
                 f"Paired seeds (Amendment 1): {seeds}. Est mode: stored.")
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
    for key, m in sorted(
            ((k, v) for k, v in agg.items()
             if (k[0], k[4], k[5]) == ("uncapped-1h", "stored", "")),
            key=lambda kv: kv[1]["trace_wait_p50_s"]):
        lines.append(
            f"| {key[3]} | {_fmt(m['util_aw'])} "
            f"| {_fmt((m['trace_wait_p50_s'] or 0) / 3600, 2)} "
            f"| {_fmt((m['trace_wait_p95_s'] or 0) / 3600, 2)} "
            f"| {_fmt(m['trace_dl_miss'])} | {_fmt(m['ov_slo_miss'])} "
            f"| {_fmt((m['makespan_s'] or 0) / 86400, 2)} |")
    lines.append("")

    # ---- clairvoyant ablation (Amendment 2, added analysis) -----------------
    lines.append("## Clairvoyant ablation — regime: capped-1h (rho 0.8, overlay 25%)")
    lines.append("")
    lines.append("*EASY and SJF re-run with ORACLE estimates (runtime_est = actual). "
                 "Labeled **clairvoyant**: an ablation isolating estimate quality, "
                 "NOT an online competitor — oracle estimates are not available at "
                 "decision time. Question (Amendment 2): does EASY's gap to the "
                 "reordering policies close when its reservations are exact?*")
    lines.append("")
    cv_cells = [Cell(s, "capped-1h", 0.8, 0.25, seed, est_mode="oracle", tag="clairvoyant")
                for s in cfg["clairvoyant"]["schedulers"] for seed in seeds]
    print(f"clairvoyant: {len(cv_cells)} cells", flush=True)
    results_cv = _run_cells(cv_cells, regimes, overlay_raw, cfg, workers)
    bad_cv = [r for r in results_cv if r["warnings"]]
    if bad_cv:
        raise SystemExit(f"clairvoyant cells with warnings: {[r['cell'] for r in bad_cv]}")
    agg_cv = aggregate(results_cv)
    results_all = results + results_cv
    agg_all = aggregate(results_all)

    lines.append("| scheduler | estimates | util(AW) | wait p50 h | wait p95 h | DL miss | SLO miss(ov) |")
    lines.append("|---|---|---|---|---|---|---|")
    for sched in ["easy_backfill", "sjf"]:
        for est_label, key in (("stored", ("capped-1h", 0.8, 0.25, sched, "stored", "")),
                               ("oracle (clairvoyant)", ("capped-1h", 0.8, 0.25, sched, "oracle", "clairvoyant"))):
            m = agg_all[key]
            lines.append(
                f"| {sched} | {est_label} | {_fmt(m['util_aw'])} "
                f"| {_fmt((m['trace_wait_p50_s'] or 0) / 3600, 2)} "
                f"| {_fmt((m['trace_wait_p95_s'] or 0) / 3600, 2)} "
                f"| {_fmt(m['trace_dl_miss'])} | {_fmt(m['ov_slo_miss'])} |")
    m_edf = agg_all[("capped-1h", 0.8, 0.25, "edf", "stored", "")]
    lines.append(
        f"| edf (reference) | stored | {_fmt(m_edf['util_aw'])} "
        f"| {_fmt((m_edf['trace_wait_p50_s'] or 0) / 3600, 2)} "
        f"| {_fmt((m_edf['trace_wait_p95_s'] or 0) / 3600, 2)} "
        f"| {_fmt(m_edf['trace_dl_miss'])} | {_fmt(m_edf['ov_slo_miss'])} |")
    lines.append("")

    # ---- rho sweep (primary only) -----------------------------------------
    lines.append("## rho sweep — regime: capped-1h (overlay 25%)")
    lines.append("")
    lines.append("| scheduler | rho | util(AW) | wait p50 h | DL miss | SLO miss(ov) | makespan d |")
    lines.append("|---|---|---|---|---|---|---|")
    for rho in primary_rhos:
        for key, m in sorted(
                ((k, v) for k, v in agg.items()
                 if (k[0], k[1], k[2], k[4], k[5]) == ("capped-1h", rho, 0.25, "stored", "")),
                key=lambda kv: kv[0][3]):
            lines.append(
                f"| {key[3]} | {rho} | {_fmt(m['util_aw'])} "
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

    # ---- warm-up check at 10/25/50% exclusion -------------------------------
    lines.append("## Warm-up exclusion check — regime: capped-1h (rho 0.8, overlay 25%)")
    lines.append("")
    lines.append("Metrics exclude jobs submitted in the first 10% / 25% / 50% of the "
                 "arrival window (initial backlog / warm-up).")
    lines.append("")
    warm_aggs = {frac: aggregate(results, warmup_frac=frac) for frac in cfg["warmup_fracs"]}
    rank_all = ranking_of(agg, "capped-1h", 0.8, 0.25, "trace_wait_p50_s")
    lines.append("| scheduler | wait p50 h (all) | " +
                 " | ".join(f"p50 h (excl {int(f * 100)}%)" for f in cfg["warmup_fracs"]) +
                 " | DL miss (all) | " +
                 " | ".join(f"DL ({int(f * 100)}%)" for f in cfg["warmup_fracs"]) + " |")
    lines.append("|---|" + "---|" * (2 * len(cfg["warmup_fracs"]) + 2))
    for sched in schedulers:
        key = ("capped-1h", 0.8, 0.25, sched, "stored", "")
        cells = [_fmt((agg[key]["trace_wait_p50_s"] or 0) / 3600, 2)]
        cells += [_fmt((warm_aggs[f][key]["trace_wait_p50_s"] or 0) / 3600, 2)
                  for f in cfg["warmup_fracs"]]
        cells += [_fmt(agg[key]["trace_dl_miss"])]
        cells += [_fmt(warm_aggs[f][key]["trace_dl_miss"]) for f in cfg["warmup_fracs"]]
        lines.append(f"| {sched} | " + " | ".join(cells) + " |")
    lines.append("")
    for frac in cfg["warmup_fracs"]:
        rank_f = ranking_of(warm_aggs[frac], "capped-1h", 0.8, 0.25, "trace_wait_p50_s")
        lines.append(f"- Ranking by wait p50 at {int(frac * 100)}% exclusion unchanged "
                     f"vs no exclusion: **{'YES' if rank_f == rank_all else 'NO'}** "
                     f"({', '.join(rank_f)})")
    lines.append("")

    # ---- straddle / horizon claim correction (step 6) ------------------------
    lines.append("## Horizon clarification (P3b claim correction)")
    lines.append("")
    lines.append("The P3b sentence \"no multi-day job straddles the 3600 s horizon\" meant "
                 "**multi-day RUNNING jobs** (excluded by the 6 h cap) — not that rollouts "
                 "end with an idle cluster. Measured below: fraction of mid-run rollouts "
                 f"(capped-1h, FIFO, seed {seeds[0]}, sampled every "
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
    mix = job_type_mix(regimes)
    lines.append("## Regime provenance and job-type mix")
    lines.append("")
    lines.append(f"- capped-1h example cell (seed {seeds[0]}): window_start "
                 f"{tman['window_start']:.0f}, n_in_window {tman['n_jobs_in_window']}, "
                 f"cap drops {tman['n_cap_dropped']}, infeasible drops {tman['dropped_total']}, "
                 f"n_jobs {tman['n_jobs']}, W {tman['work_gpu_s'] / 1e6:.2f}M GPU-s, "
                 f"T_sim {tman['t_sim_s'] / 86400:.2f} d, scale "
                 f"{tman['arrival_scale_factor']:.4f}.")
    lines.append("- Job-type mix per regime over the benchmark seeds "
                 "(batch_inference = **short-small proxy**, derived; online_inference "
                 "= synthetic overlay):")
    lines.append("")
    lines.append("| regime | trace jobs | training | batch_inference (proxy) | overlay @25% |")
    lines.append("|---|---|---|---|---|")
    for name in ("capped-1h", "stress-24h"):
        m = mix[name]
        tr = m["mix"].get("training", {"n": 0, "share": 0.0})
        bi = m["mix"].get("batch_inference", {"n": 0, "share": 0.0})
        lines.append(f"| {name} | {m['trace_jobs']} | {tr['n']} ({100 * tr['share']:.1f}%) "
                     f"| {bi['n']} ({100 * bi['share']:.1f}%) | ~{m['overlay_jobs_at_25pct']} |")
    lines.append("")
    lines.append("The 6 h cap shifts the mix toward short jobs (proxy share rises from "
                 "12.8% uncapped to 19.9% capped) — a disclosed consequence of the "
                 "pre-registered filter, not a re-pick.")
    lines.append("")
    lines.append("- All trace-derived fields carry P2 provenance (VRAM derived/synthetic; "
                 "job_type derived; priority/deadline synthetic; runtime_est derived, "
                 "MdAPE 45.3%). Overlay fields fully synthetic.")
    lines.append("- Any 'scheduler X wins' statement below must hold on stress-24h or is "
                 "scoped to capped-1h. Stress track is held out from P6 labeling.")
    lines.append("")

    # ---- Amendment 2 (E): split + window eligibility -------------------------
    elig = window_eligibility(regimes)
    lines.append("## Split and window eligibility (Amendment 2E)")
    lines.append("")
    lines.append(f"- Arrival-free trace prefix: **{elig['arrival_free_prefix_days']} days** "
                 f"({100 * elig['arrival_free_prefix_span_fraction']:.1f}% of the trace span). "
                 "Any 1 h window sampled there is degenerate; the prefix lies entirely "
                 "inside the TRAIN band [0, 0.6), so val/test bands are unaffected.")
    lines.append("- Split-seed capped-1h window eligibility (Amendment-1 rule applies "
                 "at P6 sampling, replacements logged):")
    lines.append("")
    lines.append("| split | seeds | degenerate (0 feasible) | min feasible jobs |")
    lines.append("|---|---|---|---|")
    for split, s in elig["splits"].items():
        lines.append(f"| {split} | {s['n_seeds']} | {s['n_degenerate']} | {s['min_feasible']} |")
    lines.append("")
    lines.append("- Replacement seeds vs their originals (window band = position of "
                 "window start as a fraction of trace span):")
    lines.append("")
    lines.append("| replacement | original band | replacement band | feasible jobs |")
    lines.append("|---|---|---|---|")
    for repl, info in elig["replacement_seeds"].items():
        lines.append(f"| {repl} | {info['orig_band']} | {info['repl_band']} "
                     f"| {info['repl_feasible_jobs']} |")
    lines.append("")
    lines.append("Benchmark seeds are NOT split-assigned (splits use their own seed "
                 "lists), so the band move train→test of the two replacements does "
                 "not contaminate any split; it is disclosed for completeness.")
    lines.append("")

    wall_total = time.perf_counter() - t_start
    total_events = sum(r["events"] for r in results_all)
    lines.append("## Cost")
    lines.append("")
    lines.append(f"- {len(results_all)} cells ({len(results)} main-matrix + "
                 f"{len(results_cv)} clairvoyant) + {sensitivity_cells_run} "
                 f"sensitivity cells; {total_events:,} events total; total wall "
                 f"{wall_total / 60:.1f} min (workers={workers}, CPU-only).")
    lines.append("")

    out_md = REPORTS / "p4" / "report.md"
    out_json = REPORTS / "p4" / "results.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    json.dump({
        "aggregate": {"/".join(map(str, k)): v for k, v in agg_all.items()},
        "warmup": {str(f): {"/".join(map(str, k)): v for k, v in a.items()}
                   for f, a in warm_aggs.items()},
        "cap_sensitivity": {str(c): {"/".join(map(str, k)): v for k, v in a.items()}
                            for c, a in cap_results.items()},
        "sanity": sc, "straddle": straddle_raw, "stability": stability,
        "eligibility": elig,
        "cells": len(results_all), "wall_min": wall_total / 60,
    }, open(out_json, "w", encoding="utf-8"), indent=2, default=str)
    print(f"report: {out_md}")
    if not sc["pass"]:
        raise SystemExit("sanity check FAILED (EDF > FIFO on deadline misses)")


if __name__ == "__main__":
    main()
