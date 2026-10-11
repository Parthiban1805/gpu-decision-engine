# CLAUDE.md — GPU Decision Engine (GDE)

Project: learned, cost-aware GPU workload scheduling decision model (CLEF-style candidate scoring), validated in simulation on public GPU-cluster traces. Full spec: `PLAN.md`. Read it before doing anything.

## How to work
- Follow `PLAN.md` phases P0 → P14 in order. Run each phase's gate checks before moving on. Commit after each phase.
- Give a short status after every phase: what passed, real measured numbers, what's next.
- Ask the user before: manual dataset steps (surveys/logins), anything touching a real cluster, anything beyond the PLAN scope, or continuing past P8 without a confirmed GPU.

## Hard rules
- Simulator is deterministic by seed; same inputs ⇒ identical event log. Test it.
- Policies see only decision-time information. Actual durations/outcomes are labels, never features.
- Tag every derived or synthetic job field with provenance (`measured` / `derived` / `synthetic`) and disclose synthetic fields in every report.
- The model never executes actions. Every recommendation passes through the policy gate; API returns both the recommended and the validated final action.
- Baselines (FIFO, priority, SJF, EDF, best-fit, EASY backfill) and the V1 classical model come before any LLM backbone. Report negative results honestly.
- Never state a result that wasn't produced by a re-runnable command. OR-Tools results are an optimum only if solved to proven optimality; they are clairvoyant references, not online competitors.
- Never guess dataset URLs or column names: read each dataset's README/schema and record findings in `docs/data_dictionary.md`. If a download fails, log it and continue.

## Conventions
- Python 3.11/3.12 in a venv (`.venv`, created by `python tasks.py setup`), pinned dependencies (`requirements.in` → lock in `requirements.txt` via `python tasks.py lock`), YAML configs, no hard-coded paths, type hints, pytest (+hypothesis for property tests).
- Custom `heapq` discrete-event simulator (cheap snapshot/fork for rollouts), not SimPy.
- Data lives under `data/` (gitignored); keep `SOURCES.md` + checksums per dataset.
- Windows host: no GNU make. Commands: `python tasks.py test`, `python tasks.py data`, `python tasks.py baselines`, `python tasks.py label`, `python tasks.py bench`.