"""Argument resolution and range checks shared by the Stage-1 generators."""

from __future__ import annotations

from ridge.common.validation import require_non_negative, require_positive


def resolved_burst_mean_gap_sec(interval_sec: int, configured: float | None) -> float:
    """Default the burst cadence to three collection intervals, floored at 1s."""
    return float(configured) if configured is not None else max(float(interval_sec) * 3.0, 1.0)


def resolve_bounded_range(
    base_name: str,
    min_value: int,
    max_value: int,
    *,
    positive: bool,
) -> tuple[int, int]:
    """Validate a min and max pair of second-valued arguments and return them unchanged."""
    if positive:
        require_positive(f"--{base_name}-min-sec", min_value)
        require_positive(f"--{base_name}-max-sec", max_value)
    else:
        require_non_negative(f"--{base_name}-min-sec", min_value)
        require_non_negative(f"--{base_name}-max-sec", max_value)
    if min_value > max_value:
        raise ValueError(f"--{base_name}-min-sec must be <= --{base_name}-max-sec")
    return min_value, max_value


def require_min_max_bounds(min_name: str, min_value: int, max_name: str, max_value: int) -> None:
    """Raise ValueError unless both values are positive and the minimum does not exceed the maximum."""
    require_positive(min_name, min_value)
    require_positive(max_name, max_value)
    if min_value > max_value:
        raise ValueError(f"{min_name} must be <= {max_name}")


def require_offset_before_duration(offset_value: int, duration_value: int) -> None:
    """Raise ValueError unless the fault onset offset lies within the measured period."""
    if offset_value >= duration_value:
        raise ValueError("Fault start offset must be < --duration")


def require_drain_phase_ratios(ratios: tuple[float, float, float, float]) -> None:
    """Raise ValueError unless every drain phase fraction is positive and the four sum to one."""
    for value in ratios:
        if value <= 0:
            raise ValueError("Drain phase ratios must be > 0")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Drain phase ratios must sum to 1.0")
