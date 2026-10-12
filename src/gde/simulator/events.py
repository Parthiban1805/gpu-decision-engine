"""Event types and the heapq event queue (PLAN.md §4.2).

Ordering: heap key is ``(time, rank, seq)``. ``rank`` puts every state-changing
event (ARRIVE/START/FINISH) before any DECISION_POINT at the same timestamp, so
a decision point always observes the fully settled state of its instant. ``seq``
is a global monotonic push counter that makes the order total and the event log
byte-identical across identical runs.
"""

from __future__ import annotations

import hashlib
import heapq
from enum import IntEnum
from typing import Iterator, NamedTuple


class EventKind(IntEnum):
    JOB_ARRIVE = 0
    JOB_START = 1
    JOB_FINISH = 2
    DECISION_POINT = 3


# State-changing events sort before decision points at equal time.
KIND_RANK: dict[EventKind, int] = {
    EventKind.JOB_ARRIVE: 0,
    EventKind.JOB_START: 0,
    EventKind.JOB_FINISH: 0,
    EventKind.DECISION_POINT: 1,
}


class Event(NamedTuple):
    time: float
    seq: int
    kind: EventKind
    job_id: str | None = None

    @property
    def key(self) -> tuple[float, int, int]:
        return (self.time, KIND_RANK[self.kind], self.seq)

    def log_line(self) -> str:
        """Canonical serialization: 9-decimal fixed-point time keeps it stable."""
        return f"{self.time:.9f}\t{self.seq}\t{self.kind.name}\t{self.job_id or '-'}"


def event_log_digest(log: list[str]) -> str:
    """SHA-256 over the canonical log; equal across same-seed runs (gate test)."""
    return hashlib.sha256("\n".join(log).encode("utf-8")).hexdigest()


class EventQueue:
    """Thin deterministic wrapper around heapq; copying it is a cheap fork."""

    def __init__(self) -> None:
        self._heap: list[tuple[tuple[float, int, int], Event]] = []

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, (event.key, event))

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[1]

    def peek(self) -> Event:
        return self._heap[0][1]

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Event]:
        return iter(e for _, e in self._heap)

    def copy(self) -> "EventQueue":
        clone = EventQueue()
        clone._heap = self._heap.copy()  # list copy preserves the heap invariant
        return clone
