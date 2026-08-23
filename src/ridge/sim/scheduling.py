"""Monotonic scheduling primitives for Stage-1 telemetry collection."""

from __future__ import annotations

from dataclasses import dataclass


def telemetry_should_yield_to_control(
    *,
    now_mono: float,
    next_control_deadline_mono: float | None,
    estimated_collection_sec: float,
    maximum_control_lag_sec: float = 0.5,
) -> bool:
    """Return whether a due telemetry cycle must wait for an imminent control action."""
    if next_control_deadline_mono is None or estimated_collection_sec <= 0.0:
        return False
    now = float(now_mono)
    deadline = float(next_control_deadline_mono)
    # A collection may safely finish after the transition deadline while the predicted delay remains within the certification limit. 
    # Waiting is only necessary for the part of the blocking window that would exceed it.
    guard_sec = max(0.0, float(estimated_collection_sec) - float(maximum_control_lag_sec))
    return now < deadline <= now + guard_sec


@dataclass(frozen=True)
class SnapshotSlot:
    """One fixed-rate telemetry opportunity."""

    snapshot_id: int
    scheduled_mono: float


class FixedRateTelemetryScheduler:
    """Schedule non-overlapping telemetry cycles without catch-up bursts."""

    def __init__(self, interval_sec: float, start_mono: float):
        if interval_sec <= 0:
            raise ValueError("interval_sec must be positive")
        self.interval_sec = float(interval_sec)
        self.start_mono = float(start_mono)
        self._next_snapshot_id = 0
        self._next_deadline = self.start_mono
        self._active_slot: SnapshotSlot | None = None

    @property
    def next_deadline(self) -> float:
        """Return the monotonic deadline of the next telemetry slot."""
        return self._next_deadline

    @property
    def active(self) -> bool:
        """Return whether a telemetry cycle is in progress."""
        return self._active_slot is not None

    def ready(self, now_mono: float) -> bool:
        """Return whether no cycle is active and the next deadline has passed."""
        return not self.active and float(now_mono) >= self._next_deadline

    def begin(self, now_mono: float) -> SnapshotSlot:
        """Claim the due slot as the active telemetry cycle and return it."""
        if self.active:
            raise RuntimeError("A telemetry cycle is already active")
        if not self.ready(now_mono):
            raise RuntimeError("The next telemetry cycle is not due")
        slot = SnapshotSlot(self._next_snapshot_id, self._next_deadline)
        self._active_slot = slot
        return slot

    def skip_expired(self, now_mono: float) -> list[SnapshotSlot]:
        """Drop old idle-time deadlines while retaining the latest due slot."""
        if self.active:
            raise RuntimeError("Cannot skip deadlines during an active telemetry cycle")
        now = float(now_mono)
        skipped: list[SnapshotSlot] = []
        while self._next_deadline + self.interval_sec <= now:
            skipped.append(SnapshotSlot(self._next_snapshot_id, self._next_deadline))
            self._advance()
        return skipped

    def finish(self, completed_mono: float) -> list[SnapshotSlot]:
        """Complete the active cycle and return deadlines missed by it."""
        if self._active_slot is None:
            raise RuntimeError("No telemetry cycle is active")
        completed = float(completed_mono)
        self._active_slot = None
        self._advance()

        skipped: list[SnapshotSlot] = []
        while self._next_deadline <= completed:
            skipped.append(SnapshotSlot(self._next_snapshot_id, self._next_deadline))
            self._advance()
        return skipped

    def seconds_until_due(self, now_mono: float) -> float:
        """Return the non-negative wait until the next deadline."""
        return max(0.0, self._next_deadline - float(now_mono))

    def _advance(self) -> None:
        """Move to the next snapshot identifier and derive its deadline from the origin."""
        self._next_snapshot_id += 1
        self._next_deadline = self.start_mono + self._next_snapshot_id * self.interval_sec
