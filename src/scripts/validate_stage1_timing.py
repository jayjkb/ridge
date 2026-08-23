"""Validate Stage-1 telemetry cadence before accepting a pilot or dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ridge.io.stage1_timing import TimingThresholds, evaluate_dataset_timing


def build_parser() -> argparse.ArgumentParser:
    """Build the parser whose threshold defaults come from TimingThresholds."""
    defaults = TimingThresholds()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--minimum-completeness", type=float, default=defaults.minimum_completeness)
    parser.add_argument("--start-lag-p95-sec", type=float, default=defaults.start_lag_p95_sec)
    parser.add_argument("--start-lag-p99-sec", type=float, default=defaults.start_lag_p99_sec)
    parser.add_argument("--probe-age-p95-sec", type=float, default=defaults.probe_age_p95_sec)
    parser.add_argument("--probe-max-age-sec", type=float, default=defaults.probe_max_age_sec)
    parser.add_argument("--fault-lag-p95-sec", type=float, default=defaults.fault_lag_p95_sec)
    parser.add_argument(
        "--telemetry-blocking-budget-sec",
        type=float,
        default=defaults.telemetry_blocking_budget_sec,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate the dataset timing checks, print the JSON report, and return zero only when every check passed."""
    args = build_parser().parse_args(argv)
    thresholds = TimingThresholds(
        minimum_completeness=args.minimum_completeness,
        start_lag_p95_sec=args.start_lag_p95_sec,
        start_lag_p99_sec=args.start_lag_p99_sec,
        probe_age_p95_sec=args.probe_age_p95_sec,
        probe_max_age_sec=args.probe_max_age_sec,
        fault_lag_p95_sec=args.fault_lag_p95_sec,
        telemetry_blocking_budget_sec=args.telemetry_blocking_budget_sec,
    )
    report = evaluate_dataset_timing(args.dataset_root.resolve(), thresholds=thresholds)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
