"""Mutable state of one Stage-1 episode, from artifact paths and writers to metadata and counters."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ridge.common.io import percentile

from .common import (
    DITG_DATA_PORT_MIN,
    FAULT_TIMING_HEADERS,
    HOST_HEADERS,
    INTERFACE_HEADERS,
    NEIGHBOR_HEADERS,
    NODE_HEADERS,
    PING_HEADERS,
    PROBE_TIMING_HEADERS,
    QUEUE_HEADERS,
    ROUTE_HEADERS,
    TELEMETRY_TIMING_HEADERS,
    BufferedCsvWriter,
    FaultLogger,
    ensure_csv,
    utc_timestamp,
)


def _rounded_percentile(values: list[float], fraction: float) -> float | None:
    """Return the interpolated percentile rounded to six decimals, or None for no values."""
    value = percentile(values, fraction)
    return None if value is None else round(value, 6)


class SimulatorState:
    """Per-episode container for artifact paths, buffered CSV writers, metadata, active faults, and the timing origin."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.node_csv = log_dir / "node_stats.csv"
        self.host_csv = log_dir / "host_stats.csv"
        self.interface_csv = log_dir / "interface_stats.csv"
        self.queue_csv = log_dir / "queue_stats.csv"
        self.ping_csv = log_dir / "ping_stats.csv"
        self.route_csv = log_dir / "route_stats.csv"
        self.neighbor_csv = log_dir / "neighbor_stats.csv"
        self.fault_csv = log_dir / "fault_log.csv"
        self.telemetry_timing_csv = log_dir / "telemetry_timing.csv"
        self.probe_timing_csv = log_dir / "probe_timing.csv"
        self.fault_timing_csv = log_dir / "fault_timing.csv"
        self.metadata_json = log_dir / "run_metadata.json"
        self.topology_nodes_csv = log_dir / "topology_nodes.csv"
        self.topology_links_csv = log_dir / "topology_links.csv"
        self.previous_interface_counters: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.previous_host_cpu_counters: Dict[str, float] = {}
        self.ditg_processes: List[Dict[str, str]] = []
        self.ditg_receivers: Dict[str, str] = {}
        self.ditg_next_data_port = DITG_DATA_PORT_MIN
        self.ditg_next_flow_id = 1
        self.active_faults: List[Dict[str, object]] = []
        self.timing_origin_mono: float | None = None
        self._writers: Dict[Path, BufferedCsvWriter] = {}
        self._telemetry_timing_stats: List[Dict[str, object]] = []
        self.metadata: Dict[str, object] = {
            "root_cause_kind": "none",
            "root_cause_id": "",
            "fault_target": "",
            "fault_category": "none",
            "fault_start_ts": "",
            "fault_end_ts": "",
            "phase_timestamps": [],
            "phase_count": 0,
            "target_link_role": "",
            "fault_schedule_mode": "single_fault",
            "traffic_profile_id": "",
            "traffic_flow_count": 0,
            "ping_pair_count": 0,
            "traffic_base_rate_pps": 0,
            "traffic_phase_rates_pps": [],
            "traffic_burst_count": 0,
            "traffic_burst_scheduled_count": 0,
            "traffic_burst_successful_count": 0,
            "traffic_burst_failed_count": 0,
            "traffic_burst_receiver_restart_count": 0,
            "traffic_burst_first_failure_reason": "",
            "traffic_burst_health_pass": True,
            "baseline_traffic_attempted_count": 0,
            "baseline_traffic_launched_count": 0,
            "baseline_traffic_failed_count": 0,
            "baseline_traffic_first_failure_reason": "",
            "baseline_traffic_health_pass": False,
            "traffic_runtime_check_count": 0,
            "traffic_runtime_failure_count": 0,
            "traffic_runtime_first_failure": "",
            "traffic_protocol_mix": {},
        }

        ensure_csv(self.node_csv, NODE_HEADERS)
        ensure_csv(self.host_csv, HOST_HEADERS)
        ensure_csv(self.interface_csv, INTERFACE_HEADERS)
        ensure_csv(self.queue_csv, QUEUE_HEADERS)
        ensure_csv(self.ping_csv, PING_HEADERS)
        ensure_csv(self.route_csv, ROUTE_HEADERS)
        ensure_csv(self.neighbor_csv, NEIGHBOR_HEADERS)
        ensure_csv(self.telemetry_timing_csv, TELEMETRY_TIMING_HEADERS)
        ensure_csv(self.probe_timing_csv, PROBE_TIMING_HEADERS)
        ensure_csv(self.fault_timing_csv, FAULT_TIMING_HEADERS)
        self.fault_logger = FaultLogger(self.fault_csv)
        self.write_metadata()

    def update_metadata(self, **fields: object) -> None:
        """Merge fields into the episode metadata and rewrite the metadata file."""
        self.metadata.update(fields)
        self.write_metadata()

    def append_phase_timestamp(
        self,
        phase_name: str,
        phase_state: str,
        timestamp: str,
        fault_category: str,
        *,
        flush: bool = True,
    ) -> None:
        """Append one fault phase transition to the metadata, rewriting the file unless flush is off."""
        phase_entries = list(self.metadata.get("phase_timestamps", []))
        phase_entries.append(
            {
                "phase_name": phase_name,
                "phase_state": phase_state,
                "timestamp": timestamp,
                "fault_category": fault_category,
            }
        )
        self.metadata.update(phase_timestamps=phase_entries)
        if flush:
            self.write_metadata()

    def set_timing_origin(self, origin_mono: float) -> None:
        """Fix the monotonic instant from which every offset column is measured."""
        self.timing_origin_mono = float(origin_mono)

    def timing_offset(self, mono: float | None) -> float | str:
        """Return the offset from the timing origin in seconds, or an empty string when unset."""
        if mono is None or self.timing_origin_mono is None:
            return ""
        return round(float(mono) - self.timing_origin_mono, 6)

    def append_rows(self, path: Path, headers: List[str], rows: Iterable[Dict[str, object]]) -> int:
        """Append rows through the buffered writer for the path, creating the writer on first use."""
        writer = self._writers.get(path)
        if writer is None:
            writer = BufferedCsvWriter(path, headers)
            self._writers[path] = writer
        return writer.write_rows(rows)

    def record_telemetry_timing(self, row: Dict[str, object]) -> None:
        """Write one telemetry timing row and keep it for the episode summary."""
        normalized = {header: row.get(header, "") for header in TELEMETRY_TIMING_HEADERS}
        self.append_rows(self.telemetry_timing_csv, TELEMETRY_TIMING_HEADERS, [normalized])
        self._telemetry_timing_stats.append(normalized)

    def record_probe_timing(self, row: Dict[str, object]) -> None:
        """Write one probe timing row in the column order of the header."""
        normalized = {header: row.get(header, "") for header in PROBE_TIMING_HEADERS}
        self.append_rows(self.probe_timing_csv, PROBE_TIMING_HEADERS, [normalized])

    def record_fault_timing(
        self,
        *,
        event: str,
        fault_category: str,
        target: str,
        scheduled_mono: float,
        actual_mono: float,
    ) -> None:
        """Write one fault event row with its scheduled and actual offsets and lag."""
        self.append_rows(
            self.fault_timing_csv,
            FAULT_TIMING_HEADERS,
            [
                {
                    "Event": event,
                    "FaultCategory": fault_category,
                    "Target": target,
                    "ScheduledOffsetSec": self.timing_offset(scheduled_mono),
                    "ActualOffsetSec": self.timing_offset(actual_mono),
                    "LagSec": round(max(0.0, actual_mono - scheduled_mono), 6),
                    "Timestamp": utc_timestamp(),
                }
            ],
        )

    def telemetry_timing_summary(self) -> Dict[str, object]:
        """Summarize snapshot counts, completeness ratio, and timing percentiles over the recorded cycles."""
        rows = self._telemetry_timing_stats
        complete = [row for row in rows if row.get("Status") == "complete"]
        collected = [row for row in rows if row.get("Status") in {"complete", "partial", "error"}]
        skipped = [row for row in rows if row.get("Status") == "skipped"]

        def finite_values(field: str) -> List[float]:
            """Return the sorted finite values of a column over the collected rows."""
            values: List[float] = []
            for row in collected:
                try:
                    value = float(row.get(field, ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            return sorted(values)

        lags = finite_values("StartLagSec")
        durations = finite_values("DurationSec")
        control_blocking_durations = finite_values("ControlBlockingDurationSec")
        scheduled_count = len(rows)
        return {
            "scheduled_snapshot_count": scheduled_count,
            "complete_snapshot_count": len(complete),
            "partial_snapshot_count": sum(row.get("Status") == "partial" for row in rows),
            "error_snapshot_count": sum(row.get("Status") == "error" for row in rows),
            "skipped_snapshot_count": len(skipped),
            "snapshot_completeness_ratio": round(len(complete) / scheduled_count, 6)
            if scheduled_count
            else 0.0,
            "telemetry_start_lag_p95_sec": _rounded_percentile(lags, 0.95),
            "telemetry_start_lag_p99_sec": _rounded_percentile(lags, 0.99),
            "telemetry_duration_p95_sec": _rounded_percentile(durations, 0.95),
            "telemetry_control_blocking_p95_sec": _rounded_percentile(
                control_blocking_durations, 0.95
            ),
            "telemetry_overrun_count": sum(
                str(row.get("Overrun", "")).lower() == "true" for row in collected
            ),
        }

    def flush(self) -> None:
        """Flush every buffered CSV writer of the episode."""
        for writer in self._writers.values():
            writer.flush()

    def close(self) -> None:
        """Close every buffered CSV writer and forget it."""
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def __del__(self) -> None:
        """Close the writers on garbage collection, ignoring any error."""
        try:
            self.close()
        except Exception:
            pass

    def write_metadata(self) -> None:
        """Write the metadata dictionary as JSON with sorted keys."""
        with self.metadata_json.open("w") as handle:
            json.dump(self.metadata, handle, sort_keys=True)
            handle.write("\n")
