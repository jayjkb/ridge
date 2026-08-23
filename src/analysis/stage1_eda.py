"""Validate a Stage 1 dataset for downstream temporal graph learning.

Example:
    python src/analysis/stage1_eda.py \
      --dataset-root /data/stage1-dataset \
      --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.common import (
    configure_logging,
    format_float,
    parse_optional_bool,
    print_section,
    progress_interval,
    safe_ratio,
)
from ridge.common.io import write_json
from ridge.io.stage1_dataset import (
    load_manifest,
    load_run_artifacts,
    resolve_run_dir,
    validate_dataset_root,
)

DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_LOG_LEVEL = "INFO"
BASELINE_THRESHOLDS = {
    "baseline_mean_packet_loss_pct_max": 5.0,
    "baseline_timeout_ratio_max": 0.1,
    "baseline_p95_rtt_ms_max": 250.0,
}
REQUIRED_ROOT_FILES = ("manifest.csv", "generation_provenance.json")
REQUIRED_RUN_FILES = (
    "run_metadata.json",
    "topology_nodes.csv",
    "topology_links.csv",
    "node_stats.csv",
    "interface_stats.csv",
    "ping_stats.csv",
)
SPARSE_TARGET_THRESHOLD = 10
LOGGER = logging.getLogger("stage1_eda")


@dataclass(frozen=True)
class RunAnalysis:
    """Per-episode integrity, timing, baseline, and observability findings of the readiness analysis."""

    run_id: int
    status: str
    category: str
    root_cause_kind: str
    fault_target: str
    run_dir_resolved: bool
    missing_required_files: list[str]
    has_fault_log_rows: bool
    cadence_median_sec: float | None
    interface_monotonic: bool | None
    ping_monotonic: bool | None
    queue_monotonic: bool | None
    route_monotonic: bool | None
    interface_duplicates: int
    ping_duplicates: int
    queue_duplicates: int
    route_duplicates: int
    baseline_health_pass: bool | None
    baseline_loss_exceeds: bool | None
    baseline_timeout_exceeds: bool | None
    baseline_rtt_exceeds: bool | None
    observable_probe: bool | None
    observable_queue: bool | None
    observable_interface: bool | None
    observable_any: bool | None
    rtt_uplift_ratio: float | None
    loss_uplift_pct: float | None
    timeout_fault_ratio: float | None
    queue_backlog_uplift_ratio: float | None
    interface_drop_uplift_ratio: float | None
    fault_window_valid: bool
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse the dataset root, worker count, output path, episode cap, and log level."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="Path to the Stage 1 dataset root."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Worker threads for per-run analysis (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <dataset-root>/analysis_stage1_eda.json.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for smoke tests and quick validation passes.",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help=f"Progress logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def _float_from_value(value: object) -> float | None:
    """Convert a value to float, returning None when it is missing, malformed, or NaN."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return float(parsed)


def _timestamp_stats(
    frame: pd.DataFrame,
    entity_columns: tuple[str, ...],
    column: str = "Timestamp",
) -> tuple[pd.Series, bool | None, int]:
    """Return the distinct snapshot timestamps, whether they increase monotonically, and the duplicate entity row count."""
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype="datetime64[ns, UTC]"), None, 0
    working = frame.copy()
    working["_timestamp"] = pd.to_datetime(working[column], errors="coerce", utc=True)
    working = working.dropna(subset=["_timestamp"])
    if working.empty:
        return pd.Series(dtype="datetime64[ns, UTC]"), None, 0
    snapshot_key = "SnapshotId" if "SnapshotId" in working.columns else column
    keys = [snapshot_key, *[name for name in entity_columns if name in working.columns]]
    duplicate_count = int(working.duplicated(subset=keys, keep="first").sum())
    timestamps = working.drop_duplicates(subset=[snapshot_key])["_timestamp"].reset_index(drop=True)
    if timestamps.empty:
        return timestamps, None, 0
    monotonic = bool(timestamps.is_monotonic_increasing)
    return timestamps, monotonic, duplicate_count


def _median_cadence_seconds(timestamps: pd.Series) -> float | None:
    """Return the median gap in seconds between consecutive timestamps, or None below two."""
    if len(timestamps) < 2:
        return None
    deltas = timestamps.diff().dt.total_seconds().dropna()
    if deltas.empty:
        return None
    return float(deltas.median())


def _numeric_mean(frame: pd.DataFrame, column: str) -> float | None:
    """Return the mean of a numeric column, or None when the column is absent or empty."""
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _queue_window_max(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, column: str
) -> float | None:
    """Return the maximum of a queue column within a half-open time window, or None."""
    if frame.empty or "Timestamp" not in frame.columns or column not in frame.columns:
        return None
    ts = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
    values = pd.to_numeric(frame[column], errors="coerce")
    subset = values[(ts >= start) & (ts < end)].dropna()
    if subset.empty:
        return None
    return float(subset.max())


def _interface_drop_window_mean(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> float | None:
    """Return the mean of the larger per-interface drop rate within a half-open time window, or None."""
    if frame.empty or "Timestamp" not in frame.columns:
        return None
    if "TX_DropsPerSec" not in frame.columns and "RX_DropsPerSec" not in frame.columns:
        return None
    ts = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
    tx = (
        pd.to_numeric(frame.get("TX_DropsPerSec"), errors="coerce")
        if "TX_DropsPerSec" in frame.columns
        else pd.Series(dtype=float)
    )
    rx = (
        pd.to_numeric(frame.get("RX_DropsPerSec"), errors="coerce")
        if "RX_DropsPerSec" in frame.columns
        else pd.Series(dtype=float)
    )
    if len(tx) == 0:
        tx = pd.Series(0.0, index=frame.index)
    if len(rx) == 0:
        rx = pd.Series(0.0, index=frame.index)
    values = pd.concat([tx, rx], axis=1).max(axis=1)
    subset = values[(ts >= start) & (ts < end)].dropna()
    if subset.empty:
        return None
    return float(subset.mean())


def _probe_observability(
    ping_stats: pd.DataFrame, fault_start: pd.Timestamp, fault_end: pd.Timestamp
) -> dict[str, Any]:
    """Compare probe round-trip time, loss, and timeouts before and during the fault window."""
    if ping_stats.empty or "Timestamp" not in ping_stats.columns:
        return {
            "observable_probe": None,
            "rtt_uplift_ratio": None,
            "loss_uplift_pct": None,
            "timeout_fault_ratio": None,
        }
    frame = ping_stats.copy()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
    baseline = frame[frame["Timestamp"] < fault_start].copy()
    fault = frame[(frame["Timestamp"] >= fault_start) & (frame["Timestamp"] <= fault_end)].copy()
    if baseline.empty or fault.empty:
        return {
            "observable_probe": None,
            "rtt_uplift_ratio": None,
            "loss_uplift_pct": None,
            "timeout_fault_ratio": None,
        }

    baseline_rtt = _numeric_mean(baseline, "AvgRTT")
    fault_rtt = _numeric_mean(fault, "AvgRTT")
    baseline_loss = _numeric_mean(baseline, "PacketLoss")
    fault_loss = _numeric_mean(fault, "PacketLoss")
    fault_timeout = _numeric_mean(fault, "TimeoutFlag")

    rtt_uplift_ratio = safe_ratio(fault_rtt, baseline_rtt)
    loss_uplift_pct = (
        None if fault_loss is None or baseline_loss is None else float(fault_loss - baseline_loss)
    )
    timeout_fault_ratio = fault_timeout
    observable = bool(
        (rtt_uplift_ratio is not None and rtt_uplift_ratio >= 2.0)
        or (loss_uplift_pct is not None and loss_uplift_pct >= 5.0)
        or (timeout_fault_ratio is not None and timeout_fault_ratio > 0.0)
    )
    return {
        "observable_probe": observable,
        "rtt_uplift_ratio": rtt_uplift_ratio,
        "loss_uplift_pct": loss_uplift_pct,
        "timeout_fault_ratio": timeout_fault_ratio,
    }


def _queue_observability(
    queue_stats: pd.DataFrame, fault_start: pd.Timestamp, fault_end: pd.Timestamp
) -> dict[str, Any]:
    """Compare the maximum queue backlog before and during the fault window."""
    baseline_max = _queue_window_max(
        queue_stats, pd.Timestamp.min.tz_localize("UTC"), fault_start, "Backlog_Bytes"
    )
    fault_max = _queue_window_max(queue_stats, fault_start, fault_end, "Backlog_Bytes")
    uplift_ratio = safe_ratio(fault_max, baseline_max)
    observable = None
    if baseline_max is not None and fault_max is not None:
        observable = bool(
            (baseline_max == 0 and fault_max > 0)
            or (uplift_ratio is not None and uplift_ratio >= 1.5)
        )
    return {
        "observable_queue": observable,
        "queue_backlog_uplift_ratio": uplift_ratio,
    }


def _interface_observability(
    interface_stats: pd.DataFrame, fault_start: pd.Timestamp, fault_end: pd.Timestamp
) -> dict[str, Any]:
    """Compare the mean interface drop rate before and during the fault window."""
    baseline_drop = _interface_drop_window_mean(
        interface_stats, pd.Timestamp.min.tz_localize("UTC"), fault_start
    )
    fault_drop = _interface_drop_window_mean(interface_stats, fault_start, fault_end)
    uplift_ratio = safe_ratio(fault_drop, baseline_drop)
    observable = None
    if baseline_drop is not None and fault_drop is not None:
        observable = bool(
            (baseline_drop == 0 and fault_drop > 0)
            or (uplift_ratio is not None and uplift_ratio >= 1.5)
        )
    return {
        "observable_interface": observable,
        "interface_drop_uplift_ratio": uplift_ratio,
    }


def _analyze_run(run_row: pd.Series, dataset_root: Path) -> RunAnalysis:
    """Load one episode's artifacts and compute its integrity, timing, baseline, and observability findings."""
    run_id = int(run_row["run_id"])
    status = str(run_row.get("status", "") or "")
    category = str(run_row.get("fault_category", "") or "")
    root_cause_kind = str(run_row.get("root_cause_kind", "") or "")
    fault_target = str(run_row.get("fault_target", "") or "")
    baseline_health_pass = parse_optional_bool(run_row.get("baseline_health_pass"))
    baseline_loss = _float_from_value(run_row.get("baseline_mean_packet_loss_pct"))
    baseline_timeout = _float_from_value(run_row.get("baseline_timeout_ratio"))
    baseline_rtt = _float_from_value(run_row.get("baseline_p95_rtt_ms"))

    try:
        run_dir = resolve_run_dir(dataset_root, run_row)
        missing_required = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
        artifacts = load_run_artifacts(run_dir)
    except Exception as exc:
        return RunAnalysis(
            run_id=run_id,
            status=status,
            category=category,
            root_cause_kind=root_cause_kind,
            fault_target=fault_target,
            run_dir_resolved=False,
            missing_required_files=list(REQUIRED_RUN_FILES),
            has_fault_log_rows=False,
            cadence_median_sec=None,
            interface_monotonic=None,
            ping_monotonic=None,
            queue_monotonic=None,
            route_monotonic=None,
            interface_duplicates=0,
            ping_duplicates=0,
            queue_duplicates=0,
            route_duplicates=0,
            baseline_health_pass=baseline_health_pass,
            baseline_loss_exceeds=None,
            baseline_timeout_exceeds=None,
            baseline_rtt_exceeds=None,
            observable_probe=None,
            observable_queue=None,
            observable_interface=None,
            observable_any=None,
            rtt_uplift_ratio=None,
            loss_uplift_pct=None,
            timeout_fault_ratio=None,
            queue_backlog_uplift_ratio=None,
            interface_drop_uplift_ratio=None,
            fault_window_valid=False,
            error=str(exc),
        )

    interface_ts, interface_monotonic, interface_dupes = _timestamp_stats(
        artifacts.interface_stats, ("Node", "Interface")
    )
    ping_ts, ping_monotonic, ping_dupes = _timestamp_stats(
        artifacts.ping_stats, ("Source", "Destination")
    )
    queue_ts, queue_monotonic, queue_dupes = _timestamp_stats(
        artifacts.queue_stats, ("Node", "Interface", "Qdisc")
    )
    route_ts, route_monotonic, route_dupes = _timestamp_stats(artifacts.route_stats, ("Node",))
    cadence_median = _median_cadence_seconds(interface_ts)

    fault_start = pd.to_datetime(run_row.get("fault_start_ts"), errors="coerce", utc=True)
    fault_end = pd.to_datetime(run_row.get("fault_end_ts"), errors="coerce", utc=True)
    fault_window_valid = bool(
        pd.notna(fault_start) and pd.notna(fault_end) and fault_end >= fault_start
    )

    probe_obs = {
        "observable_probe": None,
        "rtt_uplift_ratio": None,
        "loss_uplift_pct": None,
        "timeout_fault_ratio": None,
    }
    queue_obs = {"observable_queue": None, "queue_backlog_uplift_ratio": None}
    interface_obs = {"observable_interface": None, "interface_drop_uplift_ratio": None}
    observable_any = None

    if category not in {"", "none"} and fault_window_valid:
        probe_obs = _probe_observability(artifacts.ping_stats, fault_start, fault_end)
        queue_obs = _queue_observability(artifacts.queue_stats, fault_start, fault_end)
        interface_obs = _interface_observability(artifacts.interface_stats, fault_start, fault_end)
        evidence = [
            probe_obs["observable_probe"],
            queue_obs["observable_queue"],
            interface_obs["observable_interface"],
        ]
        if any(value is not None for value in evidence):
            observable_any = bool(any(value is True for value in evidence))

    return RunAnalysis(
        run_id=run_id,
        status=status,
        category=category,
        root_cause_kind=root_cause_kind,
        fault_target=fault_target,
        run_dir_resolved=True,
        missing_required_files=missing_required,
        has_fault_log_rows=not artifacts.fault_log.empty,
        cadence_median_sec=cadence_median,
        interface_monotonic=interface_monotonic,
        ping_monotonic=ping_monotonic,
        queue_monotonic=queue_monotonic,
        route_monotonic=route_monotonic,
        interface_duplicates=interface_dupes,
        ping_duplicates=ping_dupes,
        queue_duplicates=queue_dupes,
        route_duplicates=route_dupes,
        baseline_health_pass=baseline_health_pass,
        baseline_loss_exceeds=None
        if baseline_loss is None
        else bool(baseline_loss > BASELINE_THRESHOLDS["baseline_mean_packet_loss_pct_max"]),
        baseline_timeout_exceeds=None
        if baseline_timeout is None
        else bool(baseline_timeout > BASELINE_THRESHOLDS["baseline_timeout_ratio_max"]),
        baseline_rtt_exceeds=None
        if baseline_rtt is None
        else bool(baseline_rtt > BASELINE_THRESHOLDS["baseline_p95_rtt_ms_max"]),
        observable_probe=probe_obs["observable_probe"],
        observable_queue=queue_obs["observable_queue"],
        observable_interface=interface_obs["observable_interface"],
        observable_any=observable_any,
        rtt_uplift_ratio=probe_obs["rtt_uplift_ratio"],
        loss_uplift_pct=probe_obs["loss_uplift_pct"],
        timeout_fault_ratio=probe_obs["timeout_fault_ratio"],
        queue_backlog_uplift_ratio=queue_obs["queue_backlog_uplift_ratio"],
        interface_drop_uplift_ratio=interface_obs["interface_drop_uplift_ratio"],
        fault_window_valid=fault_window_valid,
        error=None,
    )


def _count_bool(records: list[RunAnalysis], field: str, expected: bool = True) -> int:
    """Count the records whose field equals the expected boolean."""
    return sum(1 for record in records if getattr(record, field) is expected)


def _median_from_records(records: list[RunAnalysis], field: str) -> float | None:
    """Return the median of a record field over the records where it is set, or None."""
    values = [getattr(record, field) for record in records if getattr(record, field) is not None]
    if not values:
        return None
    return float(pd.Series(values, dtype=float).median())


def _distribution_from_series(series: pd.Series, keep_empty: bool = False) -> dict[str, int]:
    """Count the values of a series as text, dropping blanks unless asked to keep them."""
    values = series.fillna("").astype(str)
    if not keep_empty:
        values = values[values != ""]
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _summarize_integrity(
    manifest: pd.DataFrame, records: list[RunAnalysis], dataset_root: Path
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Summarize file presence, episode resolution, and analysis errors with their checks."""
    root_files = {name: (dataset_root / name).exists() for name in REQUIRED_ROOT_FILES}
    ok_runs = manifest.loc[manifest["status"].astype(str) == "ok"]
    faulty_ok = ok_runs.loc[
        ok_runs["fault_category"].astype(str).ne("none")
        & ok_runs["fault_category"].astype(str).ne("")
    ]
    # Fault labels should remain auditable from per-episode artifacts, not only from manifest metadata.
    fault_log_missing = sum(
        1
        for record in records
        if record.category not in {"", "none"}
        and record.run_dir_resolved
        and not record.has_fault_log_rows
    )
    summary = {
        "required_root_files": root_files,
        "total_runs": int(len(manifest)),
        "ok_runs": int(len(ok_runs)),
        "resolved_run_dirs": sum(1 for record in records if record.run_dir_resolved),
        "runs_with_missing_required_files": sum(
            1 for record in records if record.missing_required_files
        ),
        "faulty_ok_runs": int(len(faulty_ok)),
        "faulty_runs_missing_fault_log_rows": int(fault_log_missing),
        "run_errors": sum(1 for record in records if record.error is not None),
    }
    checks = {
        "root_files_present": all(root_files.values()),
        "run_resolution_ok": summary["resolved_run_dirs"] == len(records),
        "required_run_files_ok": summary["runs_with_missing_required_files"] == 0,
        "fault_logs_present_for_faults": summary["faulty_runs_missing_fault_log_rows"] == 0,
        "run_errors_ok": summary["run_errors"] == 0,
    }
    return summary, checks


def _summarize_labels(manifest: pd.DataFrame) -> tuple[dict[str, Any], dict[str, bool]]:
    """Summarize healthy and faulty episode counts, category and target distributions, and sparse targets with their checks."""
    ok = manifest.loc[manifest["status"].astype(str) == "ok"].copy()
    ok["fault_category_clean"] = ok["fault_category"].fillna("none").astype(str).replace("", "none")
    ok["is_faulty"] = ok["fault_category_clean"] != "none"
    healthy_runs = int((~ok["is_faulty"]).sum())
    faulty_runs = int(ok["is_faulty"].sum())
    target_counts = _distribution_from_series(ok.loc[ok["is_faulty"], "fault_target"])
    # Sparse targets make supervised RCA unstable and weaken per-target evaluation claims.
    sparse_targets = {
        target: count for target, count in target_counts.items() if count < SPARSE_TARGET_THRESHOLD
    }
    summary = {
        "healthy_runs": healthy_runs,
        "faulty_runs": faulty_runs,
        "fault_category_counts": _distribution_from_series(
            ok["fault_category_clean"], keep_empty=True
        ),
        "root_cause_kind_counts": _distribution_from_series(ok["root_cause_kind"]),
        "fault_target_counts": target_counts,
        "sparse_targets": sparse_targets,
    }
    checks = {
        "has_healthy_runs": healthy_runs > 0,
        "has_faulty_runs": faulty_runs > 0,
        "no_sparse_targets": len(sparse_targets) == 0,
    }
    return summary, checks


def _load_expected_interval(dataset_root: Path) -> float | None:
    """Return the configured telemetry cadence from Stage 1 provenance."""
    path = dataset_root / "generation_provenance.json"
    try:
        with path.open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        resolved_config = provenance.get("resolved_config", {})
        raw_interval = provenance.get("interval_sec")
        if raw_interval is None and isinstance(resolved_config, dict):
            raw_interval = resolved_config.get("interval_sec")
        interval = float(raw_interval)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return interval if math.isfinite(interval) and interval > 0 else None


def _summarize_temporal(
    manifest: pd.DataFrame,
    records: list[RunAnalysis],
    expected_interval_sec: float | None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Summarize snapshot cadence, duration, and timestamp integrity against the expected collection interval with their checks."""
    cadence_values = [
        record.cadence_median_sec for record in records if record.cadence_median_sec is not None
    ]
    duration = (
        pd.to_numeric(manifest.get("fault_duration_sec"), errors="coerce")
        if "fault_duration_sec" in manifest.columns
        else pd.Series(dtype=float)
    )
    if duration.empty and {"fault_start_ts", "fault_end_ts"}.issubset(manifest.columns):
        duration = (
            pd.to_datetime(manifest["fault_end_ts"], errors="coerce", utc=True)
            - pd.to_datetime(manifest["fault_start_ts"], errors="coerce", utc=True)
        ).dt.total_seconds()
    offsets = (
        pd.to_numeric(manifest.get("fault_start_offset_sec"), errors="coerce")
        if "fault_start_offset_sec" in manifest.columns
        else pd.Series(dtype=float)
    )

    summary = {
        "expected_interval_sec": expected_interval_sec,
        "median_cadence_sec": None
        if not cadence_values
        else float(pd.Series(cadence_values).median()),
        "p95_cadence_sec": None
        if not cadence_values
        else float(pd.Series(cadence_values).quantile(0.95)),
        "runs_with_non_monotonic_interface": _count_bool(records, "interface_monotonic", False),
        "runs_with_non_monotonic_ping": _count_bool(records, "ping_monotonic", False),
        "runs_with_non_monotonic_queue": _count_bool(records, "queue_monotonic", False),
        "runs_with_non_monotonic_route": _count_bool(records, "route_monotonic", False),
        "total_interface_duplicates": sum(record.interface_duplicates for record in records),
        "total_ping_duplicates": sum(record.ping_duplicates for record in records),
        "total_queue_duplicates": sum(record.queue_duplicates for record in records),
        "total_route_duplicates": sum(record.route_duplicates for record in records),
        "fault_start_offset_summary_sec": _series_summary(offsets),
        "fault_duration_summary_sec": _series_summary(duration),
    }
    cadence_median = summary["median_cadence_sec"]
    checks = {
        "cadence_target_available": expected_interval_sec is not None,
        "cadence_close_to_target": bool(
            cadence_median is not None
            and expected_interval_sec is not None
            and cadence_median <= expected_interval_sec * 1.2
        ),
        "interface_timestamps_monotonic": summary["runs_with_non_monotonic_interface"] == 0,
    }
    return summary, checks


def _series_summary(series: pd.Series) -> dict[str, float] | None:
    """Return the minimum, median, and maximum of a numeric series, or None when empty."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def _summarize_baseline(
    manifest: pd.DataFrame, records: list[RunAnalysis]
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Summarize the validity check pass rate and baseline probe statistics with their checks."""
    pass_values = [
        parse_optional_bool(value)
        for value in manifest.get("baseline_health_pass", pd.Series(dtype=object))
    ]
    known_pass_values = [value for value in pass_values if value is not None]
    summary = {
        "thresholds": dict(BASELINE_THRESHOLDS),
        "baseline_health_pass_rate": None
        if not known_pass_values
        else float(sum(1 for value in known_pass_values if value) / len(known_pass_values)),
        "baseline_loss_summary": _series_summary(
            manifest.get("baseline_mean_packet_loss_pct", pd.Series(dtype=float))
        ),
        "baseline_timeout_summary": _series_summary(
            manifest.get("baseline_timeout_ratio", pd.Series(dtype=float))
        ),
        "baseline_rtt_summary": _series_summary(
            manifest.get("baseline_p95_rtt_ms", pd.Series(dtype=float))
        ),
        "loss_threshold_violations": _count_bool(records, "baseline_loss_exceeds", True),
        "timeout_threshold_violations": _count_bool(records, "baseline_timeout_exceeds", True),
        "rtt_threshold_violations": _count_bool(records, "baseline_rtt_exceeds", True),
    }
    checks = {
        # Thresholds come directly from the documented Stage 1 baseline-health gates.
        "baseline_pass_rate_nonzero": bool(
            summary["baseline_health_pass_rate"] is None
            or summary["baseline_health_pass_rate"] > 0.0
        ),
        "loss_threshold_ok": summary["loss_threshold_violations"] == 0,
        "timeout_threshold_ok": summary["timeout_threshold_violations"] == 0,
        "rtt_threshold_ok": summary["rtt_threshold_violations"] == 0,
    }
    return summary, checks


def _summarize_observability(records: list[RunAnalysis]) -> tuple[dict[str, Any], dict[str, bool]]:
    """Summarize how many faulty episodes show a visible probe, queue, or interface change, with their checks."""
    faulty = [record for record in records if record.category not in {"", "none"}]
    valid_faulty = [record for record in faulty if record.fault_window_valid]
    observable_known = [
        record.observable_any for record in valid_faulty if record.observable_any is not None
    ]
    summary = {
        "faulty_runs_analyzed": len(valid_faulty),
        "observable_run_rate": None
        if not observable_known
        else float(sum(1 for value in observable_known if value) / len(observable_known)),
        "probe_observable_runs": _count_bool(valid_faulty, "observable_probe", True),
        "queue_observable_runs": _count_bool(valid_faulty, "observable_queue", True),
        "interface_observable_runs": _count_bool(valid_faulty, "observable_interface", True),
        "median_rtt_uplift_ratio": _median_from_records(valid_faulty, "rtt_uplift_ratio"),
        "median_loss_uplift_pct": _median_from_records(valid_faulty, "loss_uplift_pct"),
        "median_timeout_fault_ratio": _median_from_records(valid_faulty, "timeout_fault_ratio"),
    }
    checks = {
        "fault_windows_present": len(valid_faulty) > 0,
        # Goal is not perfect separability, only evidence that a substantial share of injected faults perturb observable signals.
        "observability_nontrivial": bool(
            summary["observable_run_rate"] is None or summary["observable_run_rate"] >= 0.5
        ),
    }
    return summary, checks


def _summarize_leakage_risk(manifest: pd.DataFrame) -> tuple[dict[str, Any], dict[str, bool]]:
    """Summarize routing mode, traffic, probe, and fault timing metadata for shortcut-risk discussion, without checks."""
    ok = manifest.loc[manifest["status"].astype(str) == "ok"].copy()
    ok["fault_category_clean"] = ok["fault_category"].fillna("none").astype(str).replace("", "none")
    summary = {
        "routing_mode_effective_counts": _distribution_from_series(
            ok.get("routing_mode_effective", pd.Series(dtype=object))
        ),
        "traffic_flow_count_summary": _series_summary(
            ok.get("traffic_flow_count", pd.Series(dtype=float))
        ),
        "ping_pair_count_summary": _series_summary(
            ok.get("ping_pair_count", pd.Series(dtype=float))
        ),
        "fault_start_offset_summary_sec": _series_summary(
            ok.get("fault_start_offset_sec", pd.Series(dtype=float))
        ),
        "fault_duration_summary_sec": _series_summary(
            ok.get("fault_duration_sec", pd.Series(dtype=float))
        ),
    }
    checks: dict[str, bool] = {}
    return summary, checks


def _build_verdict(checks: dict[str, bool]) -> dict[str, Any]:
    """Derive the pass, usable-with-caveats, or fail verdict and its caveats from the failed checks."""
    failed = sorted(name for name, value in checks.items() if value is False)
    caveats: list[str] = []
    status = "pass"
    if failed:
        if any(
            name in failed
            for name in (
                "root_files_present",
                "run_resolution_ok",
                "required_run_files_ok",
                "run_errors_ok",
            )
        ):
            status = "fail"
        else:
            status = "usable_with_caveats"
        caveats = [_caveat_for_check(name) for name in failed]
    return {
        "status": status,
        "failed_checks": failed,
        "caveats": caveats,
    }


def _caveat_for_check(name: str) -> str:
    """Return the plain-language caveat for a failed check name."""
    messages = {
        "root_files_present": "Required dataset-level files are missing.",
        "run_resolution_ok": "Some run directories could not be resolved.",
        "required_run_files_ok": "Some runs are missing required telemetry artifacts.",
        "fault_logs_present_for_faults": "Some faulty runs lack fault-log evidence.",
        "run_errors_ok": "One or more runs failed during readiness analysis.",
        "has_healthy_runs": "No healthy runs were found in the successful manifest subset.",
        "has_faulty_runs": "No faulty runs were found in the successful manifest subset.",
        "no_sparse_targets": "Some fault targets are sparsely represented for supervised learning.",
        "cadence_target_available": "The configured telemetry interval is missing from generation provenance.",
        "cadence_close_to_target": "Observed telemetry cadence drifts beyond the configured profile.",
        "interface_timestamps_monotonic": "Some interface telemetry timestamps are not monotonic.",
        "baseline_pass_rate_nonzero": "Baseline health checks do not pass on any known runs.",
        "loss_threshold_ok": "Some runs exceed the documented baseline packet-loss threshold.",
        "timeout_threshold_ok": "Some runs exceed the documented baseline timeout threshold.",
        "rtt_threshold_ok": "Some runs exceed the documented baseline RTT threshold.",
        "fault_windows_present": "Fault windows are unavailable for faulty-run observability analysis.",
        "observability_nontrivial": "Fault observability is weak under the current heuristic checks.",
    }
    return messages.get(name, name.replace("_", " "))


def _json_ready(records: list[RunAnalysis]) -> list[dict[str, Any]]:
    """Convert the per-episode analysis records to plain dictionaries for JSON."""
    return [asdict(record) for record in records]


def main() -> None:
    """Analyze every manifest episode in a thread pool, summarize the checks, and write the JSON report and verdict."""
    args = parse_args()
    configure_logging(args.log_level)
    LOGGER.info("Starting Stage 1 TGNN readiness analysis")
    LOGGER.info("Dataset root: %s", args.dataset_root)
    validate_dataset_root(args.dataset_root)
    output_json = args.output_json or (args.dataset_root / "analysis_stage1_eda.json")

    LOGGER.info("Loading manifest")
    manifest = load_manifest(args.dataset_root)
    if args.max_runs is not None:
        manifest = manifest.head(args.max_runs).copy()
        LOGGER.info("Applying max-runs cap: %d", args.max_runs)
    LOGGER.info("Loaded %d manifest rows for analysis", len(manifest))

    rows = [row for _, row in manifest.iterrows()]
    records: list[RunAnalysis] = []
    worker_count = max(1, int(args.workers))
    LOGGER.info("Analyzing %d runs with %d worker(s)", len(rows), worker_count)
    progress_every = progress_interval(len(rows))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_analyze_run, row, args.dataset_root): int(row["run_id"])
            for row in rows
        }
        completed = 0
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                LOGGER.exception("Unexpected failure while analyzing run_id=%s: %s", run_id, exc)
                raise
            records.append(record)
            completed += 1
            if record.error is not None:
                LOGGER.warning("Run %06d analyzed with error: %s", record.run_id, record.error)
            elif LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "Run %06d complete: category=%s cadence=%s observable=%s",
                    record.run_id,
                    record.category or "none",
                    format_float(record.cadence_median_sec),
                    record.observable_any,
                )
            if completed % progress_every == 0 or completed == len(rows):
                LOGGER.info("Completed %d/%d runs", completed, len(rows))
    records.sort(key=lambda record: record.run_id)

    LOGGER.info("Summarizing integrity and provenance")
    dataset_summary, dataset_checks = _summarize_integrity(manifest, records, args.dataset_root)
    LOGGER.info("Summarizing label balance")
    label_summary, label_checks = _summarize_labels(manifest)
    LOGGER.info("Summarizing temporal quality")
    temporal_summary, temporal_checks = _summarize_temporal(
        manifest,
        records,
        _load_expected_interval(args.dataset_root),
    )
    LOGGER.info("Summarizing baseline-health validity")
    baseline_summary, baseline_checks = _summarize_baseline(manifest, records)
    LOGGER.info("Summarizing fault observability")
    observability_summary, observability_checks = _summarize_observability(records)
    LOGGER.info("Summarizing leakage-risk metadata")
    leakage_risk_summary, leakage_checks = _summarize_leakage_risk(manifest)

    checks = {
        **dataset_checks,
        **label_checks,
        **temporal_checks,
        **baseline_checks,
        **observability_checks,
        **leakage_checks,
    }
    verdict = _build_verdict(checks)

    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "workers": int(args.workers),
            "max_runs": args.max_runs,
            "output_json": str(output_json),
        },
        "dataset_summary": dataset_summary,
        "label_summary": label_summary,
        "temporal_summary": temporal_summary,
        "baseline_summary": baseline_summary,
        "observability_summary": observability_summary,
        "leakage_risk_summary": leakage_risk_summary,
        "checks": checks,
        "verdict": verdict,
        "per_run": _json_ready(records),
    }
    LOGGER.info("Writing JSON report to %s", output_json)
    write_json(output_json, report)
    LOGGER.info("Rendering concise CLI summary")

    print_section(
        "Dataset",
        [
            f"runs={dataset_summary['total_runs']}, ok={dataset_summary['ok_runs']}, resolved={dataset_summary['resolved_run_dirs']}",
            f"missing required run files={dataset_summary['runs_with_missing_required_files']}, run errors={dataset_summary['run_errors']}",
            f"faulty ok runs={dataset_summary['faulty_ok_runs']}, faulty runs missing fault logs={dataset_summary['faulty_runs_missing_fault_log_rows']}",
        ],
    )
    print_section(
        "Labels",
        [
            f"healthy={label_summary['healthy_runs']}, faulty={label_summary['faulty_runs']}",
            f"fault categories={label_summary['fault_category_counts']}",
            f"sparse targets={len(label_summary['sparse_targets'])}",
        ],
    )
    print_section(
        "Temporal",
        [
            f"median cadence sec={format_float(temporal_summary['median_cadence_sec'])}, p95 cadence sec={format_float(temporal_summary['p95_cadence_sec'])}",
            f"non-monotonic interface runs={temporal_summary['runs_with_non_monotonic_interface']}, total interface duplicates={temporal_summary['total_interface_duplicates']}",
            f"fault start offsets={temporal_summary['fault_start_offset_summary_sec']}",
        ],
    )
    print_section(
        "Baseline",
        [
            f"baseline pass rate={format_float(baseline_summary['baseline_health_pass_rate'])}",
            f"threshold violations: loss={baseline_summary['loss_threshold_violations']}, timeout={baseline_summary['timeout_threshold_violations']}, rtt={baseline_summary['rtt_threshold_violations']}",
            f"thresholds={baseline_summary['thresholds']}",
        ],
    )
    print_section(
        "Observability",
        [
            f"faulty runs analyzed={observability_summary['faulty_runs_analyzed']}, observable rate={format_float(observability_summary['observable_run_rate'])}",
            f"probe-observable={observability_summary['probe_observable_runs']}, queue-observable={observability_summary['queue_observable_runs']}, interface-observable={observability_summary['interface_observable_runs']}",
            f"median rtt uplift={format_float(observability_summary['median_rtt_uplift_ratio'])}, median loss uplift pct={format_float(observability_summary['median_loss_uplift_pct'])}",
        ],
    )
    print_section(
        "Leakage Risk",
        [
            f"effective routing modes={leakage_risk_summary['routing_mode_effective_counts']}",
            f"ping pair summary={leakage_risk_summary['ping_pair_count_summary']}",
            f"traffic flow summary={leakage_risk_summary['traffic_flow_count_summary']}",
        ],
    )
    print_section(
        "Verdict",
        [
            f"status={verdict['status']}",
            f"failed checks={verdict['failed_checks'] if verdict['failed_checks'] else 'none'}",
            f"json report={output_json}",
        ],
    )
    LOGGER.info("Stage 1 TGNN readiness analysis finished with status=%s", verdict["status"])


if __name__ == "__main__":
    main()
