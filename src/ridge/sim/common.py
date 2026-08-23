"""Column orders and append helpers for the CSV artifacts of one Stage-1 episode."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, TextIO

from mininet.log import info

# D-ITG reserves 9003-10003 for dynamic ITGLog channels.
DITG_DATA_PORT_MIN = 11000
INTERFACE_HEADERS = [
    "SnapshotId",
    "Timestamp",
    "Node",
    "Interface",
    "TX_Bytes",
    "RX_Bytes",
    "TX_Packets",
    "RX_Packets",
    "TX_Errors",
    "RX_Errors",
    "TX_Drops",
    "RX_Drops",
    "TX_KBPS",
    "RX_KBPS",
    "TX_PacketsPerSec",
    "RX_PacketsPerSec",
    "TX_DropsPerSec",
    "RX_DropsPerSec",
    "TX_ErrorsPerSec",
    "RX_ErrorsPerSec",
]
QUEUE_HEADERS = [
    "SnapshotId",
    "Timestamp",
    "Node",
    "Interface",
    "Qdisc",
    "Bytes",
    "Packets",
    "Drops",
    "Overlimits",
    "Backlog_Bytes",
    "Backlog_Packets",
    "Requeues",
]
ROUTE_HEADERS = [
    "SnapshotId",
    "Timestamp",
    "Node",
    "RouteCount",
    "DefaultRouteCount",
    "HostPrefixRouteCount",
    "KernelRouteCount",
    "StaticRouteCount",
    "OspfRouteCount",
]
NEIGHBOR_HEADERS = [
    "SnapshotId",
    "Timestamp",
    "Node",
    "NeighborCount",
    "ReachableCount",
    "StaleCount",
    "FailedCount",
]
NODE_HEADERS = [
    "SnapshotId",
    "Timestamp",
    "Node",
]
HOST_HEADERS = [
    "SnapshotId",
    "Timestamp",
    "CPUPercent",
    "LoadAvg1",
    "LoadAvg5",
    "MemoryPercent",
    "ProcCount",
]
PING_HEADERS = [
    "SnapshotId",
    "Timestamp",
    "Source",
    "Destination",
    "MinRTT",
    "AvgRTT",
    "MaxRTT",
    "MdevRTT",
    "PacketLoss",
    "TimeoutFlag",
]
FAULT_HEADERS = [
    "Timestamp",
    "Action",
    "Target",
    "Parameters",
    "FaultType",
    "FaultCategory",
]
TELEMETRY_TIMING_HEADERS = [
    "SnapshotId",
    "Status",
    "ScheduledOffsetSec",
    "ActualStartOffsetSec",
    "ActualEndOffsetSec",
    "ActualStartTimestamp",
    "ActualEndTimestamp",
    "StartLagSec",
    "DurationSec",
    "ControlBlockingDurationSec",
    "ControlServiceDurationSec",
    "PersistenceDurationSec",
    "Overrun",
    "SkippedReason",
    "NodeDurationSec",
    "NodeStatus",
    "NodeRowCount",
    "HostDurationSec",
    "HostStatus",
    "HostRowCount",
    "InterfaceDurationSec",
    "InterfaceStatus",
    "InterfaceRowCount",
    "QueueDurationSec",
    "QueueStatus",
    "QueueRowCount",
    "RouteDurationSec",
    "RouteStatus",
    "RouteRowCount",
    "NeighborDurationSec",
    "NeighborStatus",
    "NeighborRowCount",
    "PingDurationSec",
    "PingStatus",
    "PingRowCount",
    "Error",
]
PROBE_TIMING_HEADERS = [
    "SnapshotId",
    "Source",
    "Destination",
    "Status",
    "ScheduledOffsetSec",
    "LaunchOffsetSec",
    "CompletedOffsetSec",
    "LaunchLagSec",
    "DurationSec",
    "ResultAgeSec",
    "TimedOut",
    "Timestamp",
]
FAULT_TIMING_HEADERS = [
    "Event",
    "FaultCategory",
    "Target",
    "ScheduledOffsetSec",
    "ActualOffsetSec",
    "LagSec",
    "Timestamp",
]


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def ensure_csv(path: Path, headers: List[str]) -> None:
    """Create the CSV file with its header row unless it already holds data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()


def append_rows(path: Path, headers: List[str], rows: Iterable[Dict[str, object]]) -> None:
    """Append rows to a CSV file, creating it with its header when absent."""
    rows = list(rows)
    if not rows:
        return
    ensure_csv(path, headers)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writerows(rows)


class BufferedCsvWriter:
    """Keep one append handle open and periodically flush buffered CSV rows."""

    def __init__(self, path: Path, headers: List[str], flush_every_rows: int = 512):
        ensure_csv(path, headers)
        self.path = path
        self.headers = headers
        self.flush_every_rows = max(1, int(flush_every_rows))
        self._handle: TextIO = path.open("a", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=headers)
        self._rows_since_flush = 0
        self._closed = False

    def write_rows(self, rows: Iterable[Dict[str, object]]) -> int:
        """Write rows through the open handle and flush once the row threshold is reached."""
        if self._closed:
            raise RuntimeError(f"CSV writer for {self.path} is closed")
        written = 0
        for row in rows:
            self._writer.writerow(row)
            written += 1
        self._rows_since_flush += written
        if self._rows_since_flush >= self.flush_every_rows:
            self.flush()
        return written

    def flush(self) -> None:
        """Flush the open handle and reset the row counter."""
        if self._closed:
            return
        self._handle.flush()
        self._rows_since_flush = 0

    def close(self) -> None:
        """Flush and close the handle so later writes raise."""
        if self._closed:
            return
        self.flush()
        self._handle.close()
        self._closed = True


class FaultLogger:
    """Logger that appends one row per fault action to the fault CSV and the Mininet log."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        ensure_csv(self.csv_path, FAULT_HEADERS)

    def log(
        self,
        action: str,
        target: str,
        parameters: str,
        fault_type: str,
        fault_category: str = "",
    ) -> None:
        """Record one fault action with the current UTC timestamp in the CSV and the Mininet log."""
        append_rows(
            self.csv_path,
            FAULT_HEADERS,
            [
                {
                    "Timestamp": utc_timestamp(),
                    "Action": action,
                    "Target": target,
                    "Parameters": parameters,
                    "FaultType": fault_type,
                    "FaultCategory": fault_category,
                }
            ],
        )
        info(
            f"*** FaultLog action={action} target={target} "
            f"type={fault_type} category={fault_category or '-'} "
            f"params={parameters}\n"
        )
