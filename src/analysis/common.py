"""Helpers shared by the Stage 1 analysis scripts."""

from __future__ import annotations

import logging
import math


def configure_logging(level_name: str) -> None:
    """Configure root logging at the named level with a timestamped format."""
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def progress_interval(total: int) -> int:
    """Return the episode count between progress messages, about twenty per analysis."""
    if total <= 20:
        return 1
    return max(10, total // 20)


def parse_optional_bool(value: object) -> bool | None:
    """Parse a serialized boolean, returning None for missing, NaN, or unrecognized values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Divide two optional numbers, treating zero over zero as one and other zero denominators as None."""
    if numerator is None or denominator is None or denominator < 0:
        return None
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    return float(numerator / denominator)


def print_section(title: str, lines: list[str]) -> None:
    """Print a title followed by its lines as an indented bullet list."""
    print(title)
    for line in lines:
        print(f"  - {line}")


def format_float(value: float | None, digits: int = 3) -> str:
    """Format a float with fixed decimals, or n/a when it is None."""
    return "n/a" if value is None else f"{value:.{digits}f}"
