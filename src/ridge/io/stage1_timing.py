"""Cadence validation for Stage-1 telemetry artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ridge.common.contracts import (
    FAULT_CATEGORIES,
    STAGE1_ARTIFACT_TYPE,
    TELEMETRY_FAMILIES,
    parse_bool,
    require_artifact_contract,
)
from ridge.common.io import (
    finite_float,
    percentile,
    read_csv_rows,
    read_json_object,
)

_SUCCESS_STATUSES = frozenset({"ok", "complete", "completed", "fresh"})

_CATCHUP_EARLY_FIRE_TOLERANCE_SEC = 0.05


@dataclass(frozen=True)
class TimingThresholds:
    """Acceptance thresholds for cadence, completeness, probe freshness, and fault lag."""

    minimum_completeness: float = 0.99
    start_lag_p95_sec: float = 0.5
    start_lag_p99_sec: float = 1.0
    probe_age_p95_sec: float = 2.0
    probe_max_age_sec: float = 4.0
    minimum_probe_launch_coverage: float = 0.99
    fault_lag_p95_sec: float = 0.5
    telemetry_blocking_budget_sec: float = 1.0


def _row_float(row: dict[str, str], key: str) -> float | None:
    """Return a row field as a finite float, or None."""
    return finite_float(row.get(key, ""))


def _status_ok(value: object) -> bool:
    """Return whether a status value is one of the success spellings."""
    return str(value).strip().lower() in _SUCCESS_STATUSES


def _rows_by_snapshot(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    """Group rows by integer SnapshotId, skipping rows without one."""
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            snapshot_id = int(row.get("SnapshotId", ""))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(snapshot_id, []).append(row)
    return grouped


def _probe_snapshot_coverage(
    run_dir: Path,
    completed_rows: list[dict[str, str]],
    probe_timing_rows: list[dict[str, str]],
    *,
    maximum_age_sec: float,
) -> tuple[list[str], list[float]]:
    """Require one fresh, age-bounded probe result per pair and complete snapshot."""
    ping_rows = read_csv_rows(run_dir / "ping_stats.csv")
    expected_pairs = {
        (str(row.get("Source", "")).strip(), str(row.get("Destination", "")).strip())
        for row in ping_rows
    }
    expected_pairs.discard(("", ""))
    errors: list[str] = []
    accepted_ages: list[float] = []
    if not expected_pairs:
        return ["probe timing: expected pair catalog is empty"], accepted_ages

    by_snapshot: dict[int, list[dict[str, str]]] = {}
    for row in probe_timing_rows:
        raw_snapshot_id = str(row.get("SnapshotId", "")).strip()
        if not raw_snapshot_id:
            # Launch/completion lifecycle events intentionally have no SnapshotId.
            continue
        try:
            snapshot_id = int(raw_snapshot_id)
        except ValueError:
            errors.append(f"probe timing: invalid SnapshotId {raw_snapshot_id!r}")
            continue
        by_snapshot.setdefault(snapshot_id, []).append(row)

    for timing_row in completed_rows:
        snapshot_id = int(timing_row["SnapshotId"])
        rows = by_snapshot.get(snapshot_id, [])
        observed_pairs = [
            (
                str(row.get("Source", "")).strip(),
                str(row.get("Destination", "")).strip(),
            )
            for row in rows
        ]
        if set(observed_pairs) != expected_pairs or len(observed_pairs) != len(expected_pairs):
            errors.append(
                f"snapshot {snapshot_id}: probe timing pair set is incomplete or duplicated"
            )
            continue
        for row, pair in zip(rows, observed_pairs):
            status = str(row.get("Status", "")).strip().lower()
            if status != "fresh":
                errors.append(
                    f"snapshot {snapshot_id}: probe {pair[0]}->{pair[1]} status={status or 'missing'}"
                )
                continue
            age = _row_float(row, "ResultAgeSec")
            if age is None:
                errors.append(
                    f"snapshot {snapshot_id}: probe {pair[0]}->{pair[1]} has no finite result age"
                )
                continue
            accepted_ages.append(age)
            if age > maximum_age_sec:
                errors.append(
                    f"snapshot {snapshot_id}: probe {pair[0]}->{pair[1]} age "
                    f"{age:.6f}s exceeds {maximum_age_sec:.6f}s"
                )
    return errors, accepted_ages


def _probe_lifecycle_coverage(
    run_dir: Path,
    probe_timing_rows: list[dict[str, str]],
    *,
    duration_sec: float,
    cadence_sec: float,
    minimum_coverage: float,
) -> tuple[list[str], float, int]:
    """Validate that cached freshness is backed by the configured launch cadence."""
    expected_pairs = {
        (str(row.get("Source", "")).strip(), str(row.get("Destination", "")).strip())
        for row in read_csv_rows(run_dir / "ping_stats.csv")
        if str(row.get("Source", "")).strip() and str(row.get("Destination", "")).strip()
    }
    if not expected_pairs or cadence_sec <= 0.0:
        return ["probe lifecycle: pair catalog or cadence is invalid"], 0.0, 0

    expected_opportunities = math.ceil(duration_sec / cadence_sec)
    errors: list[str] = []
    pair_coverages: list[float] = []
    skipped_count = 0
    for pair in sorted(expected_pairs):
        lifecycle_rows = []
        for row in probe_timing_rows:
            if str(row.get("SnapshotId", "")).strip():
                continue
            if (
                str(row.get("Source", "")).strip(),
                str(row.get("Destination", "")).strip(),
            ) != pair:
                continue
            status = str(row.get("Status", "")).strip().lower()
            if status not in {"launched", "skipped_deadline"}:
                continue
            scheduled_offset = _row_float(row, "ScheduledOffsetSec")
            if scheduled_offset is None or not (0.0 <= scheduled_offset < duration_sec):
                continue
            lifecycle_rows.append(row)

        launch_offsets = [
            round(float(row["ScheduledOffsetSec"]), 6)
            for row in lifecycle_rows
            if str(row.get("Status", "")).strip().lower() == "launched"
        ]
        skipped_offsets = [
            round(float(row["ScheduledOffsetSec"]), 6)
            for row in lifecycle_rows
            if str(row.get("Status", "")).strip().lower() == "skipped_deadline"
        ]
        skipped_count += len(skipped_offsets)
        if len(set(launch_offsets)) != len(launch_offsets):
            errors.append(f"probe lifecycle: duplicate launch opportunity for {pair[0]}->{pair[1]}")
        opportunity_count = len(set([*launch_offsets, *skipped_offsets]))
        if opportunity_count < expected_opportunities:
            errors.append(
                f"probe lifecycle: {pair[0]}->{pair[1]} recorded {opportunity_count}/"
                f"{expected_opportunities} scheduled opportunities"
            )
        coverage = min(1.0, len(set(launch_offsets)) / max(1, expected_opportunities))
        pair_coverages.append(coverage)
        if coverage < minimum_coverage:
            errors.append(
                f"probe lifecycle: {pair[0]}->{pair[1]} launch coverage "
                f"{coverage:.6f} is below {minimum_coverage:.6f}"
            )
        recorded_opportunities = len(launch_offsets) + len(skipped_offsets)
        skipped_fraction = len(skipped_offsets) / max(1, recorded_opportunities)
        if skipped_fraction > 1.0 - minimum_coverage + 1e-12:
            errors.append(
                f"probe lifecycle: {pair[0]}->{pair[1]} skipped fraction "
                f"{skipped_fraction:.6f} exceeds {1.0 - minimum_coverage:.6f}"
            )
    return errors, min(pair_coverages, default=0.0), skipped_count


def _entity_completeness_errors(
    run_dir: Path,
    completed_rows: list[dict[str, str]],
) -> list[str]:
    """Validate exact entities and timestamp alignment for complete cycles."""
    topology_nodes = read_csv_rows(run_dir / "topology_nodes.csv")
    topology_links = read_csv_rows(run_dir / "topology_links.csv")
    expected_nodes = {str(row.get("Node", "")) for row in topology_nodes}
    expected_interfaces = {
        (str(row.get("Source", "")), str(row.get("SourceInterface", ""))) for row in topology_links
    } | {
        (str(row.get("Destination", "")), str(row.get("DestinationInterface", "")))
        for row in topology_links
    }
    family_rows = {
        "node": read_csv_rows(run_dir / "node_stats.csv"),
        "interface": read_csv_rows(run_dir / "interface_stats.csv"),
        "queue": read_csv_rows(run_dir / "queue_stats.csv"),
        "route": read_csv_rows(run_dir / "route_stats.csv"),
        "neighbor": read_csv_rows(run_dir / "neighbor_stats.csv"),
        "probe": read_csv_rows(run_dir / "ping_stats.csv"),
    }
    grouped = {name: _rows_by_snapshot(rows) for name, rows in family_rows.items()}
    expected_probe_ids = {
        f"{row.get('Source', '')}->{row.get('Destination', '')}" for row in family_rows["probe"]
    }
    errors: list[str] = []
    if not expected_nodes or not expected_interfaces or not expected_probe_ids:
        errors.append("topology or probe catalog is empty")
        return errors

    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = read_json_object(metadata_path)
        try:
            expected_probe_count = int(metadata.get("ping_pair_count", -1))
        except (TypeError, ValueError):
            expected_probe_count = -1
        if expected_probe_count != len(expected_probe_ids):
            errors.append(
                "probe catalog size does not match run_metadata.json: "
                f"{len(expected_probe_ids)} != {expected_probe_count}"
            )

    for timing_row in completed_rows:
        snapshot_id = int(timing_row["SnapshotId"])
        snapshot_rows = {name: rows.get(snapshot_id, []) for name, rows in grouped.items()}
        for family, rows in snapshot_rows.items():
            prefix = "Ping" if family == "probe" else family.title()
            try:
                recorded_count = int(timing_row.get(f"{prefix}RowCount", ""))
            except (TypeError, ValueError):
                recorded_count = -1
            if recorded_count != len(rows):
                errors.append(
                    f"snapshot {snapshot_id}: {family} timing row count "
                    f"{recorded_count} != artifact row count {len(rows)}"
                )

        timestamps = {
            str(row.get("Timestamp", "")) for rows in snapshot_rows.values() for row in rows
        }
        timing_timestamp = str(timing_row.get("ActualStartTimestamp", "")).strip()
        if timestamps != {timing_timestamp} or not timing_timestamp:
            errors.append(f"snapshot {snapshot_id}: telemetry timestamps are not aligned")

        for family in ("node", "route", "neighbor"):
            actual_nodes = {str(row.get("Node", "")) for row in snapshot_rows[family]}
            if actual_nodes != expected_nodes:
                errors.append(f"snapshot {snapshot_id}: {family} entity set mismatch")
        actual_interfaces = {
            (str(row.get("Node", "")), str(row.get("Interface", "")))
            for row in snapshot_rows["interface"]
        }
        if actual_interfaces != expected_interfaces:
            errors.append(f"snapshot {snapshot_id}: interface entity set mismatch")
        actual_queue_interfaces = {
            (str(row.get("Node", "")), str(row.get("Interface", "")))
            for row in snapshot_rows["queue"]
        }
        if not expected_interfaces.issubset(actual_queue_interfaces):
            errors.append(f"snapshot {snapshot_id}: queue entity set mismatch")
        actual_probe_ids = {
            f"{row.get('Source', '')}->{row.get('Destination', '')}"
            for row in snapshot_rows["probe"]
        }
        if actual_probe_ids != expected_probe_ids:
            errors.append(f"snapshot {snapshot_id}: probe entity set mismatch")
    return errors


def evaluate_run_timing(
    run_dir: Path,
    *,
    duration_sec: float,
    interval_sec: float,
    thresholds: TimingThresholds = TimingThresholds(),
) -> dict[str, Any]:
    """Evaluate cadence, completeness, probe freshness, and fault timing for one episode."""
    timing_rows = read_csv_rows(run_dir / "telemetry_timing.csv")
    probe_rows = read_csv_rows(run_dir / "probe_timing.csv")
    fault_rows = read_csv_rows(run_dir / "fault_timing.csv")
    metadata = read_json_object(run_dir / "run_metadata.json")
    expected_count = math.ceil(duration_sec / interval_sec)

    snapshot_ids: list[int] = []
    completed_rows: list[dict[str, str]] = []
    family_errors: list[str] = []
    for row in timing_rows:
        try:
            snapshot_id = int(row.get("SnapshotId", ""))
        except (TypeError, ValueError):
            continue
        snapshot_ids.append(snapshot_id)
        if not _status_ok(row.get("Status")):
            continue
        completed_rows.append(row)
        for family in TELEMETRY_FAMILIES:
            prefix = "Ping" if family == "probe" else family.title()
            if not _status_ok(row.get(f"{prefix}Status")):
                family_errors.append(f"snapshot {snapshot_id}: {family} status")
                continue
            try:
                row_count = int(row.get(f"{prefix}RowCount", "0") or 0)
            except ValueError:
                row_count = 0
            if row_count <= 0:
                family_errors.append(f"snapshot {snapshot_id}: {family} has no rows")

    expected_ids = set(range(expected_count))
    observed_ids = set(snapshot_ids)
    unexplained_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    duplicate_ids = sorted(
        snapshot_id for snapshot_id in observed_ids if snapshot_ids.count(snapshot_id) > 1
    )
    completeness = len(completed_rows) / expected_count if expected_count else 0.0
    start_lags = [
        value for row in completed_rows if (value := _row_float(row, "StartLagSec")) is not None
    ]
    cycle_durations = [
        value for row in completed_rows if (value := _row_float(row, "DurationSec")) is not None
    ]
    control_blocking_durations = []
    for row in completed_rows:
        control_duration = _row_float(row, "ControlBlockingDurationSec")
        if control_duration is None:
            control_duration = _row_float(row, "DurationSec")
        if control_duration is not None:
            control_blocking_durations.append(control_duration)

    ordered_completed = sorted(completed_rows, key=lambda row: int(row["SnapshotId"]))
    overlap_count = 0
    catchup_burst_count = 0
    previous_end: float | None = None
    previous_start: float | None = None
    for row in ordered_completed:
        start = _row_float(row, "ActualStartOffsetSec")
        end = _row_float(row, "ActualEndOffsetSec")
        if start is None or end is None:
            continue
        if previous_end is not None and start < previous_end - 1e-9:
            overlap_count += 1
        if previous_start is not None and start - previous_start < interval_sec * 0.5:
            scheduled = _row_float(row, "ScheduledOffsetSec")
            fired_ahead_of_schedule = (
                scheduled is None or start < scheduled - _CATCHUP_EARLY_FIRE_TOLERANCE_SEC
            )
            if fired_ahead_of_schedule:
                catchup_burst_count += 1
        previous_start = start
        previous_end = end

    probe_coverage_errors, accepted_probe_ages = _probe_snapshot_coverage(
        run_dir,
        completed_rows,
        probe_rows,
        maximum_age_sec=thresholds.probe_max_age_sec,
    )
    try:
        probe_cadence_sec = float(metadata.get("probe_cadence_sec", ""))
    except (TypeError, ValueError):
        probe_cadence_sec = math.nan
    probe_lifecycle_errors, probe_launch_coverage, probe_skipped_deadline_count = (
        _probe_lifecycle_coverage(
            run_dir,
            probe_rows,
            duration_sec=duration_sec,
            cadence_sec=probe_cadence_sec,
            minimum_coverage=thresholds.minimum_probe_launch_coverage,
        )
    )
    fault_lags = [value for row in fault_rows if (value := _row_float(row, "LagSec")) is not None]
    fault_offsets = [
        value for row in fault_rows if (value := _row_float(row, "ActualOffsetSec")) is not None
    ]
    fault_events_ordered = all(
        later > earlier for earlier, later in zip(fault_offsets, fault_offsets[1:])
    )
    fault_category = str(metadata.get("fault_category", "none") or "none")
    fault_expected = fault_category in FAULT_CATEGORIES
    fault_event_names = [str(row.get("Event", "")).strip() for row in fault_rows]
    terminal_event_present = any(
        event == "recovery" or event.endswith(":complete") for event in fault_event_names
    )
    fault_timing_evidence = (
        bool(fault_rows)
        and fault_event_names.count("fault_injected") == 1
        and terminal_event_present
        and bool(str(metadata.get("fault_start_ts", "")).strip())
        and bool(str(metadata.get("fault_end_ts", "")).strip())
        and all(
            str(row.get("FaultCategory", fault_category)) == fault_category for row in fault_rows
        )
    )
    if not fault_expected:
        fault_timing_evidence = not fault_rows

    # OSPF is the only routing mode, so acceptance is exactly "every router started and converged". 
    # An episode whose FRR failed never reaches here.
    try:
        router_count = int(metadata.get("frr_router_count", -1))
        started_count = int(metadata.get("frr_start_success_count", -1))
        converged_count = int(metadata.get("frr_converged_router_count", -1))
    except (TypeError, ValueError):
        router_count = started_count = converged_count = -1
    frr_fully_converged = bool(
        router_count > 0 and started_count == router_count and converged_count == router_count
    )

    cleanup_healthy = not metadata.get("cleanup_errors") and all(
        int(metadata.get(key, 0) or 0) == 0
        for key in ("lingering_ditg", "lingering_receivers", "lingering_active_faults")
    )
    baseline_traffic_healthy = bool(
        parse_bool(
            metadata.get("baseline_traffic_health_pass", False),
            field="baseline_traffic_health_pass",
        )
        and int(metadata.get("baseline_traffic_attempted_count", 0) or 0) > 0
        and int(metadata.get("baseline_traffic_launched_count", 0) or 0)
        == int(metadata.get("baseline_traffic_attempted_count", 0) or 0)
        and int(metadata.get("baseline_traffic_failed_count", 0) or 0) == 0
        and int(metadata.get("traffic_runtime_check_count", 0) or 0) > 0
        and int(metadata.get("traffic_runtime_failure_count", 0) or 0) == 0
    )
    burst_traffic_healthy = parse_bool(
        metadata.get("traffic_burst_health_pass", False),
        field="traffic_burst_health_pass",
    )
    baseline_probe_healthy = bool(
        parse_bool(metadata.get("baseline_health_pass", False), field="baseline_health_pass")
        and int(metadata.get("baseline_missing_probe_sample_count", 0) or 0) == 0
        and int(metadata.get("baseline_loss_sample_count", 0) or 0) > 0
        and int(metadata.get("baseline_rtt_sample_count", 0) or 0) > 0
        and int(metadata.get("baseline_timeout_sample_count", 0) or 0) > 0
    )

    lag_p95 = percentile(start_lags, 0.95)
    lag_p99 = percentile(start_lags, 0.99)
    probe_age_p95 = percentile(accepted_probe_ages, 0.95)
    fault_lag_p95 = percentile([abs(value) for value in fault_lags], 0.95)
    family_errors.extend(_entity_completeness_errors(run_dir, completed_rows))
    checks = {
        "timing_artifact_present": bool(timing_rows),
        "snapshot_ids_explicit": not unexplained_ids and not unexpected_ids and not duplicate_ids,
        "completeness": completeness >= thresholds.minimum_completeness,
        "family_completeness": not family_errors,
        "start_lag_p95": lag_p95 is not None and lag_p95 <= thresholds.start_lag_p95_sec,
        "start_lag_p99": lag_p99 is not None and lag_p99 <= thresholds.start_lag_p99_sec,
        "telemetry_blocking_budget": bool(control_blocking_durations)
        and max(control_blocking_durations) <= thresholds.telemetry_blocking_budget_sec,
        "no_overlap": overlap_count == 0,
        "no_catchup_bursts": catchup_burst_count == 0,
        "probe_snapshot_coverage": not probe_coverage_errors,
        "probe_launch_cadence": not probe_lifecycle_errors,
        "probe_freshness_evidence": bool(accepted_probe_ages),
        "probe_age_p95": probe_age_p95 is not None
        and probe_age_p95 <= thresholds.probe_age_p95_sec,
        "probe_max_age": bool(accepted_probe_ages)
        and max(accepted_probe_ages) <= thresholds.probe_max_age_sec,
        "fault_timing_evidence": fault_timing_evidence,
        "fault_events_ordered": fault_events_ordered,
        "fault_lag_p95": (
            fault_lag_p95 is not None and fault_lag_p95 <= thresholds.fault_lag_p95_sec
        )
        if fault_expected
        else fault_lag_p95 is None,
        "frr_fully_converged": frr_fully_converged,
        "cleanup_healthy": cleanup_healthy,
        "baseline_traffic_healthy": baseline_traffic_healthy,
        "burst_traffic_healthy": burst_traffic_healthy,
        "baseline_probe_healthy": baseline_probe_healthy,
    }
    return {
        "run_dir": str(run_dir),
        "passed": all(checks.values()),
        "checks": checks,
        "expected_snapshot_count": expected_count,
        "timing_row_count": len(timing_rows),
        "completed_snapshot_count": len(completed_rows),
        "completeness": completeness,
        "unexplained_snapshot_ids": unexplained_ids,
        "unexpected_snapshot_ids": unexpected_ids,
        "duplicate_snapshot_ids": duplicate_ids,
        "family_errors": family_errors,
        "probe_coverage_errors": probe_coverage_errors,
        "probe_lifecycle_errors": probe_lifecycle_errors,
        "probe_launch_coverage_min": probe_launch_coverage,
        "probe_skipped_deadline_count": probe_skipped_deadline_count,
        "start_lag_p95_sec": lag_p95,
        "start_lag_p99_sec": lag_p99,
        "max_cycle_duration_sec": max(cycle_durations, default=None),
        "max_control_blocking_duration_sec": max(control_blocking_durations, default=None),
        "overlap_count": overlap_count,
        "catchup_burst_count": catchup_burst_count,
        "probe_age_p95_sec": probe_age_p95,
        "probe_max_age_sec": max(accepted_probe_ages) if accepted_probe_ages else None,
        "fault_lag_p95_sec": fault_lag_p95,
    }


def evaluate_dataset_timing(
    dataset_root: Path,
    *,
    thresholds: TimingThresholds = TimingThresholds(),
) -> dict[str, Any]:
    """Evaluate all successful episodes in a generated Stage-1 dataset."""
    provenance = read_json_object(dataset_root / "generation_provenance.json")
    require_artifact_contract(provenance, artifact_type=STAGE1_ARTIFACT_TYPE)
    config = provenance.get("resolved_config", provenance)
    duration_sec = float(config["duration_sec"])
    interval_sec = float(config["interval_sec"])
    manifest_rows = read_csv_rows(dataset_root / "manifest.csv")
    reports: list[dict[str, Any]] = []
    for row in manifest_rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        run_id = int(row["run_id"])
        log_dir = str(row.get("log_dir", "")).strip()
        candidate = Path(log_dir) if log_dir else Path(f"run_{run_id:06d}")
        run_dir = candidate if candidate.is_absolute() else dataset_root / candidate
        reports.append(
            evaluate_run_timing(
                run_dir,
                duration_sec=duration_sec,
                interval_sec=interval_sec,
                thresholds=thresholds,
            )
        )

    total_expected = sum(report["expected_snapshot_count"] for report in reports)
    total_complete = sum(report["completed_snapshot_count"] for report in reports)
    return {
        "dataset_root": str(dataset_root),
        "passed": bool(reports) and all(report["passed"] for report in reports),
        "thresholds": asdict(thresholds),
        "run_count": len(reports),
        "passed_run_count": sum(1 for report in reports if report["passed"]),
        "aggregate_completeness": total_complete / total_expected if total_expected else 0.0,
        "runs": reports,
    }
