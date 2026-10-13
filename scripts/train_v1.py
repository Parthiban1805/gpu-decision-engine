"""P7: train V1 (GBM + set-encoder MLP) on the P6 labels, evaluate offline on
val/test, then CLOSED-LOOP on the primary TEST split and the stress track
against EDF and SJF (Amendment 2 gate: beat both on primary TEST; no worse
than min-J(EDF,SJF) on stress). Paired bootstrap CIs. Regime tags everywhere.

Re-run: python tasks.py train-v1
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from gde.baselines import ALL_POLICIES
from gde.evaluation.benchmark import load_regimes, make_overlay_cfg, regime_spec
from gde.labeling.dataset_builder import replace_degenerate_seeds
from gde.labeling.objective import load_preset, trajectory_objective
from gde.models.gbm import GbmScorer
from gde.models.model_policy import ModelPolicy
from gde.models.mlp_setencoder import load_mlp
from gde.models.train import train_v1
from gde.paths import CHECKPOINTS, CONFIGS, DATA, REPORTS
from gde.simulator.cluster import ClusterSpec
from gde.simulator.engine import Engine
from gde.simulator.workload import TraceConfig, generate_overlay, load_trace_window

PARQUET = DATA / "labels" / "p6_primary.parquet"


def _fmt(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def _run_policy(args: tuple) -> dict:
    """Closed-loop trajectory J for one (policy-kind, regime, seed)."""
    kind, regime_name, seed, model_paths = args
    spec = regime_spec(load_regimes(), regime_name)
    import yaml as _y
    overlay_raw = _y.safe_load((CONFIGS / "sim_p3.yaml").read_text(encoding="utf-8"))["overlay"]
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    cap = spec.get("duration_cap_s")
    jobs, tman = load_trace_window(
        DATA / "processed" / f"{spec['dataset']}.parquet", cluster,
        TraceConfig(window_s=spec["window_s"], rho=0.8, max_jobs=10**9,
                    duration_cap_s=float(cap) if cap else None), seed=seed)
    overlay, _ = generate_overlay(len(jobs), tman["t_sim_s"],
                                  make_overlay_cfg(overlay_raw, 0.25), seed=seed)
    const = load_preset("balanced")
    t_sim = tman["t_sim_s"]
    if kind in ("edf", "sjf", "fifo"):
        policy = ALL_POLICIES[kind]()
        result = Engine(jobs + overlay, cluster, policy, arrival_horizon=t_sim).run()
        j = trajectory_objective(result.state, const, t_sim)["J"]
        lat = None
        fallbacks = 0
    else:
        scorer = (GbmScorer.load(Path(model_paths["gbm"]))
                  if kind == "v1_gbm" else load_mlp(Path(model_paths["mlp"])))
        policy = ModelPolicy(scorer)
        engine = Engine(jobs + overlay, cluster, policy, arrival_horizon=t_sim)
        policy.bind(engine)
        result = engine.run()
        j = trajectory_objective(result.state, const, t_sim)["J"]
        lat = float(np.median(policy.latencies_ms)) if policy.latencies_ms else None
        fallbacks = policy.fallbacks
    return {"kind": kind, "regime": regime_name, "seed": seed, "J": j,
            "wait_p50_s": result.summary["groups"]["trace"]["wait_s_p50"],
            "dl_miss": result.summary["groups"]["trace"]["deadline_miss_rate"],
            "slo_miss": result.summary["groups"]["overlay"].get("slo_miss_rate"),
            "util": result.summary["cluster"]["utilization_arrival_window"],
            "decisions": len(policy.latencies_ms) if kind.startswith("v1") else None,
            "latency_ms": lat, "fallbacks": fallbacks}


def paired_ci(diffs, n=2000, alpha=0.05):
    rng = np.random.default_rng(0)
    diffs = np.asarray(diffs, dtype=float)
    boots = [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
             for _ in range(n)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(np.mean(diffs)), "ci_lo": float(lo), "ci_hi": float(hi)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 1))
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse checkpoints/v1 and only re-evaluate")
    args = parser.parse_args()
    regimes = load_regimes()
    exp = yaml.safe_load((CONFIGS / "experiment_p6.yaml").read_text(encoding="utf-8"))

    # ---- train (or load) ----------------------------------------------------
    if not args.skip_train:
        print("training V1 (GBM + MLP)...", flush=True)
        train_v1(PARQUET)
    metrics = json.loads((CHECKPOINTS / "v1" / "metrics.json").read_text(encoding="utf-8"))
    model_paths = {"gbm": str(CHECKPOINTS / "v1" / "gbm.txt"),
                   "mlp": str(CHECKPOINTS / "v1" / "mlp.pt")}
    v1_kind = ("v1_gbm" if metrics["gbm"]["val"]["mean_regret"]
               <= metrics["mlp"]["val"]["mean_regret"] else "v1_mlp")
    print(f"V1 selected by val regret: {v1_kind} "
          f"(gbm {metrics['gbm']['val']['mean_regret']:.4f} vs "
          f"mlp {metrics['mlp']['val']['mean_regret']:.4f})", flush=True)

    # ---- closed-loop: primary TEST split seeds -------------------------------
    spec = regime_spec(regimes, "capped-1h")
    cluster = ClusterSpec.from_yaml(CONFIGS / spec["cluster"])
    test_seeds, test_repl = replace_degenerate_seeds(
        regimes["splits"]["split_seeds"]["test"], cluster, 3600.0,
        float(spec["duration_cap_s"]),
        DATA / "processed" / f"{spec['dataset']}.parquet")
    bench_seeds = regimes["benchmark_seeds"]
    kinds = [v1_kind, "edf", "sjf"]
    cells = [(k, "capped-1h", s, model_paths) for k in kinds for s in test_seeds]
    cells += [(k, "stress-24h", s, model_paths) for k in kinds for s in bench_seeds]
    print(f"closed-loop: {len(cells)} runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run_policy, cells))

    def J_of(kind, regime):
        return {r["seed"]: r["J"] for r in rows if r["kind"] == kind and r["regime"] == regime}

    def paired(kind_a, kind_b, regime):
        a, b = J_of(kind_a, regime), J_of(kind_b, regime)
        seeds = sorted(set(a) & set(b))
        return paired_ci([a[s] - b[s] for s in seeds])

    primary_edf = paired(v1_kind, "edf", "capped-1h")
    primary_sjf = paired(v1_kind, "sjf", "capped-1h")
    stress_edf = paired(v1_kind, "edf", "stress-24h")
    stress_sjf = paired(v1_kind, "sjf", "stress-24h")
    beat = lambda ci: ci["ci_hi"] < 0
    no_worse = lambda ci: ci["ci_lo"] <= 0
    gate_primary = beat(primary_edf) and beat(primary_sjf)
    gate_stress = no_worse(stress_edf) and no_worse(stress_sjf)

    # ---- report ---------------------------------------------------------------
    lines: list[str] = []
    lines.append("# P7 — V1 classical models (GBM + set-encoder MLP)")
    lines.append("")
    lines.append(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%S')} by `python tasks.py "
                 "train-v1`. Labels: `python tasks.py label` (P6, primary track only, "
                 "tau per Amendment 3). Headline disclosures: 64% of offered work "
                 "excluded by the 6 h cap; deadlines/SLOs/VRAM synthetic; "
                 "batch_inference = short-small proxy. Every table is regime-tagged.")
    lines.append("")
    lines.append("## Offline selection (VAL) and test regret")
    lines.append("")
    lines.append("| model | val mean regret | val median | test mean regret |")
    lines.append("|---|---|---|---|")
    for name in ("gbm", "mlp"):
        m = metrics[name]
        lines.append(f"| {name} | {_fmt(m['val']['mean_regret'], 4)} "
                     f"| {_fmt(m['val']['median_regret'], 4)} "
                     f"| {_fmt(m['test']['mean_regret'], 4)} |")
    for b, mm in metrics["baselines"].items():
        lines.append(f"| {b} pick (reference) | {_fmt(mm['mean_regret'], 4)} "
                     f"| {_fmt(mm['median_regret'], 4)} | - |")
    lines.append("")
    lines.append(f"**V1 = {v1_kind}** (lower val mean regret, Amendment 2 selection rule).")

    lines.append("")
    lines.append("## Closed loop — regime: capped-1h (primary TEST split, overlay 25%)")
    lines.append("")
    lines.append("| policy | J (mean over seeds) | wait p50 h | DL miss | SLO miss | util |")
    lines.append("|---|---|---|---|---|---|")
    for kind in kinds:
        rs = [r for r in rows if r["kind"] == kind and r["regime"] == "capped-1h"]
        lines.append(f"| {kind} | {_fmt(np.mean([r['J'] for r in rs]), 4)} "
                     f"| {_fmt(np.mean([r['wait_p50_s'] for r in rs]) / 3600, 2)} "
                     f"| {_fmt(np.mean([r['dl_miss'] for r in rs]))} "
                     f"| {_fmt(np.mean([r['slo_miss'] for r in rs if r['slo_miss'] is not None]))} "
                     f"| {_fmt(np.mean([r['util'] for r in rs]))} |")
    lines.append("")
    lines.append("## Closed loop — regime: stress-24h (benchmark seeds, overlay 25%)")
    lines.append("")
    lines.append("| policy | J (mean over seeds) | wait p50 h | DL miss | SLO miss | util |")
    lines.append("|---|---|---|---|---|---|")
    for kind in kinds:
        rs = [r for r in rows if r["kind"] == kind and r["regime"] == "stress-24h"]
        lines.append(f"| {kind} | {_fmt(np.mean([r['J'] for r in rs]), 4)} "
                     f"| {_fmt(np.mean([r['wait_p50_s'] for r in rs]) / 3600, 2)} "
                     f"| {_fmt(np.mean([r['dl_miss'] for r in rs]))} "
                     f"| {_fmt(np.mean([r['slo_miss'] for r in rs if r['slo_miss'] is not None]))} "
                     f"| {_fmt(np.mean([r['util'] for r in rs]))} |")
    lines.append("")
    lines.append("## P7 gate (Amendment 2)")
    lines.append("")
    lines.append(f"- primary TEST vs EDF: mean diff {primary_edf['mean']:+.4f}, "
                 f"95% CI [{primary_edf['ci_lo']:+.4f}, {primary_edf['ci_hi']:+.4f}] "
                 f"-> beat: **{'YES' if beat(primary_edf) else 'NO'}**")
    lines.append(f"- primary TEST vs SJF: mean diff {primary_sjf['mean']:+.4f}, "
                 f"95% CI [{primary_sjf['ci_lo']:+.4f}, {primary_sjf['ci_hi']:+.4f}] "
                 f"-> beat: **{'YES' if beat(primary_sjf) else 'NO'}**")
    lines.append(f"- stress vs EDF: mean diff {stress_edf['mean']:+.4f}, "
                 f"95% CI [{stress_edf['ci_lo']:+.4f}, {stress_edf['ci_hi']:+.4f}] "
                 f"-> no worse: **{'YES' if no_worse(stress_edf) else 'NO'}**")
    lines.append(f"- stress vs SJF: mean diff {stress_sjf['mean']:+.4f}, "
                 f"95% CI [{stress_sjf['ci_lo']:+.4f}, {stress_sjf['ci_hi']:+.4f}] "
                 f"-> no worse: **{'YES' if no_worse(stress_sjf) else 'NO'}**")
    lines.append("")
    verdict = "PASS" if (gate_primary and gate_stress) else "FAIL"
    lines.append(f"**Gate verdict: {verdict}** "
                 f"(primary: {'beat both' if gate_primary else 'did NOT beat both'}; "
                 f"stress: {'no worse' if gate_stress else 'WORSE'}). "
                 "Per PLAN P7, if V1 does not beat the best heuristic, diagnose "
                 "(labels, features, horizon, tau) before P8 and report honestly.")

    v1_rows = [r for r in rows if r["kind"] == v1_kind]
    lats = [r["latency_ms"] for r in v1_rows if r["latency_ms"] is not None]
    lines.append("")
    lines.append("## Decision latency and gate usage")
    lines.append("")
    if lats:
        lines.append(f"- median per-decision latency: **{np.mean(lats):.2f} ms** "
                     f"(per-run median of per-decision times; CPU scoring for GBM).")
    fb = [r["fallbacks"] for r in v1_rows if r["fallbacks"] is not None]
    if fb:
        lines.append(f"- policy-gate fallbacks: {sum(fb)} total across runs "
                     f"(invalid/low-confidence recommendations re-routed to EDF).")

    out = REPORTS / "p7" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    json.dump({"metrics": metrics, "v1_kind": v1_kind,
               "rows": rows, "gate": {"primary": gate_primary, "stress": gate_stress},
               "ci": {"primary_edf": primary_edf, "primary_sjf": primary_sjf,
                      "stress_edf": stress_edf, "stress_sjf": stress_sjf}},
              open(REPORTS / "p7" / "results.json", "w", encoding="utf-8"),
              indent=2, default=str)
    print(f"report: {out}")
    print(f"gate: primary={'PASS' if gate_primary else 'FAIL'} "
          f"stress={'PASS' if gate_stress else 'FAIL'}")


if __name__ == "__main__":
    main()
