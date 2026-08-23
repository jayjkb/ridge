"""Guards shared by the Stage-1 generator and the per-episode generator."""

from __future__ import annotations


def require_positive(name: str, value: int | float) -> None:
    """Raise ValueError unless the named argument is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def require_non_negative(name: str, value: int | float) -> None:
    """Raise ValueError unless the named argument is zero or positive."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def require_range(name: str, value: float, lower: float, upper: float) -> None:
    """Raise ValueError unless the named argument lies within the inclusive bounds."""
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")


def require_duration_after_window(duration: int, fault_duration_sec: int) -> None:
    """Raise ValueError unless the measured period is longer than the fault duration."""
    # An episode as long as its fault would contain no healthy snapshots.
    if duration <= fault_duration_sec:
        raise ValueError("--duration must be longer than fault duration")


def require_fault_window_fits(duration: int, start_offset: int, fault_duration: int) -> None:
    """Raise ValueError unless the fault window ends within the measured period."""
    if start_offset + fault_duration > duration:
        raise ValueError(
            "Fault window does not fit: fault start offset + fault duration "
            "must be <= telemetry collection duration"
        )
