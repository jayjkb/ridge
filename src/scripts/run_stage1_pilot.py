"""Run the gated Stage-1 correctness and concurrency pilot on the target machine."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ridge.common.contracts import FAULT_CATEGORIES
from ridge.io.stage1_timing import evaluate_dataset_timing
from scripts.audit_stage1_features import audit_dataset

CONCURRENCY_TIERS = (4, 8, 12)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the output root, interpreter, dry run, and reserved CPU count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter containing the target Mininet environment.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
    )
    parser.add_argument(
        "--reserve-cpus",
        type=int,
        default=2,
        help="Leave this many CPUs outside the pilot process affinity (default: 2).",
    )
    return parser


def _configure_cpu_affinity(reserve_cpus: int, *, apply: bool) -> dict[str, list[int]]:
    """Pin the pilot to all but the last reserved CPUs and return the active and reserved sets."""
    if reserve_cpus < 0:
        raise ValueError("--reserve-cpus must be non-negative")
    available = sorted(os.sched_getaffinity(0))
    if reserve_cpus >= len(available):
        raise ValueError(
            f"--reserve-cpus={reserve_cpus} leaves no CPUs for the pilot "
            f"(available={len(available)})"
        )
    active = available[:-reserve_cpus] if reserve_cpus else available
    reserved = available[-reserve_cpus:] if reserve_cpus else []
    if apply:
        os.sched_setaffinity(0, active)
    return {"pilot_cpus": active, "reserved_cpus": reserved}


def _generation_command(
    python: Path,
    output: Path,
    *,
    runs: int,
    workers: int,
    seed: int,
    fault_fraction: float,
    fault_category: str,
) -> list[str]:
    """Build the command line that runs the generator for one pilot job."""
    return [
        str(python),
        "-m",
        "ridge",
        "generate",
        "--runs",
        str(runs),
        "--workers",
        str(workers),
        "--output",
        str(output),
        "--duration",
        "180",
        "--interval",
        "2",
        "--burst-mean-gap-sec",
        "2.5",
        "--traffic-flow-min",
        "20",
        "--traffic-flow-max",
        "24",
        "--ping-pair-min",
        "4",
        "--ping-pair-max",
        "6",
        "--probe-packets",
        "1",
        "--probe-timeout-sec",
        "1.0",
        "--probe-cadence-sec",
        "1.0",
        "--warmup-sec",
        "60",
        "--fault-start-offset-min-sec",
        "35",
        "--fault-start-offset-max-sec",
        "70",
        "--fault-duration-min-sec",
        "25",
        "--fault-duration-max-sec",
        "45",
        "--fault-fraction",
        str(fault_fraction),
        "--fault-category",
        fault_category,
        "--seed",
        str(seed),
        "--min-training-runs",
        "0",
    ]


def _run_job(command: list[str], output: Path, *, dry_run: bool) -> dict[str, Any]:
    """Run one generation job, or print it on a dry run, and return its timing, gate, and certification results."""
    if dry_run:
        print(" ".join(command))
        return {"output": str(output), "dry_run": True}
    completed = subprocess.run(command, check=False)
    try:
        report = evaluate_dataset_timing(output)
    except Exception as exc:
        report = {
            "passed": False,
            "validation_error": f"{type(exc).__name__}: {exc}",
        }
    gate_path = output / "generation_gate.json"
    generation_gate = (
        json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    )
    realism_path = output / "realism_checks.json"
    realism_checks = (
        json.loads(realism_path.read_text(encoding="utf-8")) if realism_path.exists() else {}
    )
    report["generator_returncode"] = completed.returncode
    report["generation_gate"] = generation_gate
    report["realism_checks"] = realism_checks
    report["passed"] = bool(
        completed.returncode == 0
        and report.get("passed")
        and generation_gate.get("passed", False)
        and realism_checks.get("certification_passed", False)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the correctness and concurrency pilot jobs, audit the features, repeat the best tier, and write the pilot report."""
    args = build_parser().parse_args(argv)
    affinity = _configure_cpu_affinity(args.reserve_cpus, apply=not args.dry_run)
    print(
        f"pilot CPU affinity: active={affinity['pilot_cpus']} reserved={affinity['reserved_cpus']}",
        flush=True,
    )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"Pilot output root must not exist: {output_root}")
    if not args.dry_run:
        output_root.mkdir(parents=True)

    jobs: list[tuple[str, list[str], Path]] = []
    healthy_output = output_root / "correctness_healthy"
    jobs.append(
        (
            "correctness_healthy",
            _generation_command(
                args.python,
                healthy_output,
                runs=2,
                workers=1,
                seed=42,
                fault_fraction=0.0,
                fault_category="mixed",
            ),
            healthy_output,
        )
    )
    for category_index, category in enumerate(FAULT_CATEGORIES):
        output = output_root / f"correctness_{category}"
        jobs.append(
            (
                f"correctness_{category}",
                _generation_command(
                    args.python,
                    output,
                    runs=2,
                    workers=1,
                    seed=100 + category_index * 10,
                    fault_fraction=1.0,
                    fault_category=category,
                ),
                output,
            )
        )
    for workers in CONCURRENCY_TIERS:
        output = output_root / f"concurrency_{workers:02d}"
        jobs.append(
            (
                f"concurrency_{workers:02d}",
                _generation_command(
                    args.python,
                    output,
                    runs=16,
                    workers=workers,
                    seed=4242,
                    fault_fraction=0.5,
                    fault_category="mixed",
                ),
                output,
            )
        )

    reports: dict[str, Any] = {}
    for name, command, output in jobs:
        reports[name] = _run_job(command, output, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    feature_audits: dict[str, Any] = {}
    for name, _command, output in jobs:
        if not name.startswith("correctness_"):
            continue
        try:
            feature_audits[name] = audit_dataset(output)
        except Exception as exc:
            feature_audits[name] = {"audit_error": f"{type(exc).__name__}: {exc}"}

    passing_tiers = [
        workers
        for workers in CONCURRENCY_TIERS
        if reports[f"concurrency_{workers:02d}"].get("passed")
    ]
    highest = max(passing_tiers, default=None)
    if highest is not None:
        repeat_output = output_root / f"concurrency_{highest:02d}_repeat"
        repeat_command = _generation_command(
            args.python,
            repeat_output,
            runs=16,
            workers=highest,
            seed=4242,
            fault_fraction=0.5,
            fault_category="mixed",
        )
        reports[f"concurrency_{highest:02d}_repeat"] = _run_job(
            repeat_command, repeat_output, dry_run=False
        )

    feature_audits_passed = bool(feature_audits) and not any(
        "audit_error" in report for report in feature_audits.values()
    )
    correctness_passed = all(
        reports[name].get("passed", False)
        for name, _command, _output in jobs
        if name.startswith("correctness_")
    )
    repeat_passed = bool(
        highest is not None
        and reports.get(f"concurrency_{highest:02d}_repeat", {}).get("passed", False)
    )
    qualified_names = (
        {f"concurrency_{highest:02d}", f"concurrency_{highest:02d}_repeat"}
        if highest is not None
        else set()
    )
    required_resource_reports = [
        report
        for name, report in reports.items()
        if name.startswith("correctness_") or name in qualified_names
    ]
    resource_headroom_verified = bool(required_resource_reports) and all(
        report.get("generation_gate", {}).get("resource_summary", {}).get("headroom_passed", False)
        for report in required_resource_reports
    )
    summary = {
        "passed": correctness_passed
        and repeat_passed
        and resource_headroom_verified
        and feature_audits_passed,
        "timing_qualified_worker_count": highest,
        "resource_headroom_verified": resource_headroom_verified,
        "paired_distribution_reviewed": False,
        "full_generation_authorized": False,
        "feature_audits_passed": feature_audits_passed,
        "cpu_affinity": affinity,
        "note": (
            "Timing qualification alone does not authorize full generation. "
            "Review paired telemetry distributions and fault observability first. "
            "Failed higher tiers are retained as expected calibration evidence."
        ),
        "reports": reports,
        "feature_audits": feature_audits,
    }
    (output_root / "pilot_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
