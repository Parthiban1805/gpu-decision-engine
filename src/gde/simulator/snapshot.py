"""Cheap snapshot/fork (PLAN.md §4.2).

snapshot() copies containers only: the event-heap list (a plain list copy
preserves the heap invariant), pending/record dicts, per-GPU accumulators and
the log prefix. Immutable SimJob and finalized Placement/JobRecord objects are
shared, not deep-copied; unfinalized JobRecords are cloned because both the
original and the fork mutate them. There is no RNG state inside the engine, so
nothing else needs saving.
"""

from __future__ import annotations

from gde.simulator.engine import Engine, EngineConfig, Policy, SimState


def snapshot(state: SimState) -> SimState:
    return state.copy()


def fork(state: SimState, policy: Policy, config: EngineConfig | None = None,
         arrival_horizon: float | None = None, manifest: dict | None = None) -> Engine:
    """A new engine that continues from a copy of `state`."""
    return Engine.from_state(
        snapshot(state), policy, config=config,
        arrival_horizon=arrival_horizon, manifest=manifest,
    )
