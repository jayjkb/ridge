"""Aggregate Stage-6 test metrics across seeds into mean and standard deviation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> int:
    """Report the mean, sample standard deviation, and count of every scalar test metric across the summaries as JSON."""
    parser = argparse.ArgumentParser(description="Aggregate test metrics across seeds")
    parser.add_argument(
        "summaries", nargs="+", type=Path, help="Stage-6 evaluation summary files, one per seed"
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    values: dict[str, list[float]] = defaultdict(list)
    for path in args.summaries:
        payload = json.loads(path.read_text())
        if "test_metrics" not in payload:
            raise SystemExit(
                f"{path} has no 'test_metrics' block. Pass Stage-6 evaluation summaries, "
                "not Stage-5 training metrics."
            )
        metrics = payload["test_metrics"]
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key].append(float(value))
    report = {}
    for key, series in sorted(values.items()):
        mean = sum(series) / len(series)
        variance = (
            sum((v - mean) ** 2 for v in series) / (len(series) - 1) if len(series) > 1 else 0.0
        )
        report[key] = {"mean": mean, "std": math.sqrt(variance), "n": len(series)}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
