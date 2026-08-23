"""Filesystem and numeric helpers shared across the pipeline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


def read_csv_rows(path: Path, *, required: bool = False) -> list[dict[str, str]]:
    """Read a CSV file into string-valued row dictionaries, empty when an optinal file is absent."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write row dictionaries to a CSV file with the keys of the first row as header."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    """Return the parsed content of a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_json_object(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    """Load a JSON object. With ``missing_ok`` an absent file reads as ``{}``."""
    if missing_ok and not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    """Write a payload as indented JSON with sorted keys and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def as_float(value: object) -> float:
    """Coerce to float, returning NaN when the value is missing or malformed."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite_float(value: object) -> float | None:
    """Coerce to float, returning None for missing, malformed or non-finite values."""
    parsed = as_float(value)
    return parsed if math.isfinite(parsed) else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linearly interpolated percentile, matching the Stage-1 acceptance gate."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
