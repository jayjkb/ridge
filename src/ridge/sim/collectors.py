"""Collectors that turn node commands, emulation-host statistics, and asynchronous probes into the rows of one snapshot."""

from __future__ import annotations

import dataclasses
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from mininet.net import Mininet

from .common import utc_timestamp
from .state import SimulatorState
from .topology import logical_interface_name, logical_name, node_primary_ip

PING_COUNT = 1
PING_TIMEOUT_SEC = 1.0
SECTION_MARKER_PREFIX = "__RIDGE_SECTION_"
NODE_TELEMETRY_SECTIONS = (
    ("NET_DEV", "cat /proc/net/dev"),
    ("QDISC", "tc -s qdisc show"),
    ("ROUTES", "ip -4 route show"),
    ("NEIGHBORS", "ip -4 neigh show"),
)
NODE_TELEMETRY_COMMAND = " ; ".join(
    f"printf '{SECTION_MARKER_PREFIX}{name}__\\n' ; {command}"
    for name, command in NODE_TELEMETRY_SECTIONS
)


@dataclasses.dataclass
class NodeTelemetrySnapshot:
    """Raw command output and parsed interface counters collected from one node at one instant."""

    timestamp: str
    now: float
    node_name: str
    runtime_interfaces: List[str]
    logical_interfaces: Dict[str, str]
    interface_stats: Dict[str, Dict[str, int]]
    qdisc_raw: str
    routes_raw: str
    neighbors_raw: str


@dataclasses.dataclass
class TelemetryBatch:
    """Rows and per-family collection diagnostics for one snapshot slot."""

    node_rows: List[Dict[str, object]]
    interface_rows: List[Dict[str, object]]
    queue_rows: List[Dict[str, object]]
    route_rows: List[Dict[str, object]]
    neighbor_rows: List[Dict[str, object]]
    durations_sec: Dict[str, float]
    statuses: Dict[str, str]
    errors: List[str]


def parse_proc_net_dev(output: str) -> Dict[str, Dict[str, int]]:
    """Parse /proc/net/dev into byte, packet, error, and drop counters keyed by interface."""
    stats = {}
    for line in output.strip().splitlines()[2:]:
        if ":" not in line:
            continue
        iface, data = line.split(":", 1)
        fields = data.split()
        if len(fields) < 16:
            continue
        stats[iface.strip()] = {
            "rx_bytes": int(fields[0]),
            "rx_packets": int(fields[1]),
            "rx_errs": int(fields[2]),
            "rx_drops": int(fields[3]),
            "tx_bytes": int(fields[8]),
            "tx_packets": int(fields[9]),
            "tx_errs": int(fields[10]),
            "tx_drops": int(fields[11]),
        }
    return stats


def parse_cpu_line(output: str) -> Dict[str, int]:
    """Parse the aggregate cpu line of /proc/stat into its eight jiffy counters."""
    for line in output.strip().splitlines():
        if line.startswith("cpu "):
            fields = line.split()[1:]
            padded = fields + ["0"] * max(0, 8 - len(fields))
            return {
                "user": int(padded[0]),
                "nice": int(padded[1]),
                "system": int(padded[2]),
                "idle": int(padded[3]),
                "iowait": int(padded[4]),
                "irq": int(padded[5]),
                "softirq": int(padded[6]),
                "steal": int(padded[7]),
            }
    return {
        "user": 0,
        "nice": 0,
        "system": 0,
        "idle": 0,
        "iowait": 0,
        "irq": 0,
        "softirq": 0,
        "steal": 0,
    }


def parse_meminfo(output: str) -> tuple[float, float]:
    """Return total memory in kilobytes and the used percentage from /proc/meminfo."""
    total_kb = 0.0
    available_kb = math.nan
    for line in output.strip().splitlines():
        if line.startswith("MemTotal:"):
            total_kb = float(line.split()[1])
        elif line.startswith("MemAvailable:"):
            available_kb = float(line.split()[1])
    if not total_kb or not math.isfinite(available_kb):
        return 0.0, 0.0
    used_percent = max(0.0, min(100.0, (1.0 - available_kb / total_kb) * 100.0))
    return total_kb, used_percent


def parse_loadavg(output: str) -> tuple[float, float]:
    """Return the one and five minute load averages from /proc/loadavg."""
    fields = output.strip().split()
    if len(fields) < 2:
        return 0.0, 0.0
    return _safe_float(fields[0]), _safe_float(fields[1])


def rate(
    current_values: Dict[str, float],
    previous_values: Optional[Dict[str, float]],
    elapsed: float,
    field: str,
) -> float:
    """Return the non-negative per-second change of a counter since the previous values."""
    if not previous_values or elapsed <= 0:
        return 0.0
    return max(0.0, (current_values[field] - previous_values[field]) / elapsed)


def node_data_interfaces(node: object) -> List[str]:
    """Return the runtime interface names of a node except loopback."""
    return [intf.name for intf in node.intfList() if intf.name != "lo"]


def parse_sectioned_output(output: str) -> Dict[str, str]:
    """Split combined command output into sections by their printed markers."""
    pattern = re.compile(rf"^{re.escape(SECTION_MARKER_PREFIX)}([A-Z_]+)__$")
    sections: Dict[str, List[str]] = {}
    current: str | None = None
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def parse_queue_rows(node: object, raw: str, timestamp: str) -> List[Dict[str, object]]:
    """Parse tc qdisc statistics into one queue row per qdisc with logical interface names."""
    rows: List[Dict[str, object]] = []
    sent_pattern = re.compile(
        r"Sent\s+(\d+)\s+bytes\s+(\d+)\s+pkt\s+\(dropped\s+(\d+),\s+overlimits\s+(\d+)\s+requeues\s+(\d+)\)"
    )
    backlog_pattern = re.compile(r"backlog\s+(\d+)b\s+(\d+)p")
    qdisc_pattern = re.compile(r"qdisc\s+(\S+)\s+\d+:\s+dev\s+(\S+)")
    node_name = logical_name(node)
    blocks = [block.strip() for block in raw.split("qdisc ") if block.strip()]
    for block in blocks:
        header = f"qdisc {block.splitlines()[0]}"
        qdisc_match = qdisc_pattern.search(header)
        sent_match = sent_pattern.search(block)
        backlog_match = backlog_pattern.search(block)
        if qdisc_match is None or sent_match is None:
            continue
        qdisc_name = qdisc_match.group(1)
        runtime_interface = qdisc_match.group(2)
        interface_name = logical_interface_name(node, runtime_interface)
        backlog_bytes = int(backlog_match.group(1)) if backlog_match else 0
        backlog_packets = int(backlog_match.group(2)) if backlog_match else 0
        rows.append(
            {
                "Timestamp": timestamp,
                "Node": node_name,
                "Interface": interface_name,
                "Qdisc": qdisc_name,
                "Bytes": int(sent_match.group(1)),
                "Packets": int(sent_match.group(2)),
                "Drops": int(sent_match.group(3)),
                "Overlimits": int(sent_match.group(4)),
                "Backlog_Bytes": backlog_bytes,
                "Backlog_Packets": backlog_packets,
                "Requeues": int(sent_match.group(5)),
            }
        )
    return rows


def parse_route_row(node_name: str, raw: str, timestamp: str) -> Dict[str, object]:
    """Count the routes of a node by default, host prefix, and protocol origin."""
    routes = [line.strip() for line in raw.splitlines() if line.strip()]
    default_count = sum(1 for line in routes if line.startswith("default "))
    host_prefix_count = sum(1 for line in routes if line.startswith("10.10."))
    kernel_count = sum(1 for line in routes if " proto kernel " in f" {line} ")
    static_count = sum(1 for line in routes if " proto static " in f" {line} ")
    ospf_count = sum(
        1 for line in routes if " proto ospf " in f" {line} " or " proto zebra " in f" {line} "
    )
    return {
        "Timestamp": timestamp,
        "Node": node_name,
        "RouteCount": len(routes),
        "DefaultRouteCount": default_count,
        "HostPrefixRouteCount": host_prefix_count,
        "KernelRouteCount": kernel_count,
        "StaticRouteCount": static_count,
        "OspfRouteCount": ospf_count,
    }


def parse_neighbor_row(node_name: str, raw: str, timestamp: str) -> Dict[str, object]:
    """Count the neighbor table entries of a node by reachability state."""
    neighbors = [line.strip() for line in raw.splitlines() if line.strip()]
    return {
        "Timestamp": timestamp,
        "Node": node_name,
        "NeighborCount": len(neighbors),
        "ReachableCount": sum(1 for line in neighbors if " REACHABLE" in line),
        "StaleCount": sum(1 for line in neighbors if " STALE" in line),
        "FailedCount": sum(1 for line in neighbors if " FAILED" in line),
    }


def collect_host_stats(
    state: SimulatorState,
    *,
    snapshot_id: int,
    timestamp: str,
    now_mono: float | None = None,
) -> Dict[str, object]:
    """Collect one emulation-host diagnostic row (never a model input)."""
    now = time.monotonic() if now_mono is None else float(now_mono)
    cpu = parse_cpu_line(Path("/proc/stat").read_text(encoding="utf-8"))
    load1, load5 = parse_loadavg(Path("/proc/loadavg").read_text(encoding="utf-8"))
    _total_memory_kb, memory_percent = parse_meminfo(
        Path("/proc/meminfo").read_text(encoding="utf-8")
    )
    proc_count = sum(entry.name.isdigit() for entry in Path("/proc").iterdir())

    total = float(sum(cpu.values()))
    idle = float(cpu["idle"] + cpu["iowait"])
    previous = state.previous_host_cpu_counters
    cpu_percent = 0.0
    if previous:
        total_delta = max(0.0, total - float(previous["total"]))
        idle_delta = max(0.0, idle - float(previous["idle"]))
        if total_delta > 0:
            cpu_percent = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
    state.previous_host_cpu_counters = {"total": total, "idle": idle, "timestamp": now}
    return {
        "SnapshotId": snapshot_id,
        "Timestamp": timestamp,
        "CPUPercent": round(cpu_percent, 4),
        "LoadAvg1": round(load1, 4),
        "LoadAvg5": round(load5, 4),
        "MemoryPercent": round(memory_percent, 4),
        "ProcCount": proc_count,
    }


def collect_node_snapshot(node: object, timestamp: str, now: float) -> NodeTelemetrySnapshot:
    """Run the combined telemetry command in one node and parse its sections."""
    runtime_interfaces = node_data_interfaces(node)
    sections = parse_sectioned_output(node.cmd(NODE_TELEMETRY_COMMAND))
    return NodeTelemetrySnapshot(
        timestamp=timestamp,
        now=now,
        node_name=logical_name(node),
        runtime_interfaces=runtime_interfaces,
        logical_interfaces={
            iface: logical_interface_name(node, iface) for iface in runtime_interfaces
        },
        interface_stats=parse_proc_net_dev(sections.get("NET_DEV", "")),
        qdisc_raw=sections.get("QDISC", ""),
        routes_raw=sections.get("ROUTES", ""),
        neighbors_raw=sections.get("NEIGHBORS", ""),
    )


def collect_node_snapshots(
    net: Mininet,
    timestamp: str | None = None,
    now: float | None = None,
    *,
    errors: List[str] | None = None,
) -> List[NodeTelemetrySnapshot]:
    """Collect one logically atomic node batch with a shared timestamp."""
    collected_at = utc_timestamp() if timestamp is None else timestamp
    collected_now = time.monotonic() if now is None else float(now)
    snapshots: List[NodeTelemetrySnapshot] = []
    for node in list(net.hosts) + list(net.switches):
        try:
            snapshots.append(collect_node_snapshot(node, timestamp=collected_at, now=collected_now))
        except Exception as exc:
            if errors is None:
                raise
            errors.append(f"node={logical_name(node)}:{type(exc).__name__}:{exc}")
    return snapshots


def snapshot_to_node_row(snapshot: NodeTelemetrySnapshot) -> Dict[str, object]:
    """Return node identity for per-snapshot entity completeness.

    Mininet provides network namespaces, but its nodes share the emulation host's PID and mount namespaces.
    CPU, load, memory, and process-count values therefore describe the emulation host and are not node-local and are collected once in
    ``host_stats.csv`` instead of being duplicated for every node.
    """
    return {
        "Timestamp": snapshot.timestamp,
        "Node": snapshot.node_name,
    }


def snapshot_to_interface_rows(
    snapshot: NodeTelemetrySnapshot, state: SimulatorState
) -> List[Dict[str, object]]:
    """Build interface rows with per-second rates against the previous snapshot and store the new counters."""
    rows: List[Dict[str, object]] = []
    for iface in snapshot.runtime_interfaces:
        values = snapshot.interface_stats.get(iface)
        if values is None:
            continue
        interface_name = snapshot.logical_interfaces.get(iface, iface)
        key = (snapshot.node_name, interface_name)
        previous = state.previous_interface_counters.get(key)
        elapsed = snapshot.now - previous["timestamp"] if previous else 0.0
        rows.append(
            {
                "Timestamp": snapshot.timestamp,
                "Node": snapshot.node_name,
                "Interface": interface_name,
                "TX_Bytes": int(values["tx_bytes"]),
                "RX_Bytes": int(values["rx_bytes"]),
                "TX_Packets": int(values["tx_packets"]),
                "RX_Packets": int(values["rx_packets"]),
                "TX_Errors": int(values["tx_errs"]),
                "RX_Errors": int(values["rx_errs"]),
                "TX_Drops": int(values["tx_drops"]),
                "RX_Drops": int(values["rx_drops"]),
                "TX_KBPS": round(rate(values, previous, elapsed, "tx_bytes") * 8 / 1000, 4),
                "RX_KBPS": round(rate(values, previous, elapsed, "rx_bytes") * 8 / 1000, 4),
                "TX_PacketsPerSec": round(rate(values, previous, elapsed, "tx_packets"), 4),
                "RX_PacketsPerSec": round(rate(values, previous, elapsed, "rx_packets"), 4),
                "TX_DropsPerSec": round(rate(values, previous, elapsed, "tx_drops"), 4),
                "RX_DropsPerSec": round(rate(values, previous, elapsed, "rx_drops"), 4),
                "TX_ErrorsPerSec": round(rate(values, previous, elapsed, "tx_errs"), 4),
                "RX_ErrorsPerSec": round(rate(values, previous, elapsed, "rx_errs"), 4),
            }
        )
        state.previous_interface_counters[key] = {
            "timestamp": snapshot.now,
            "tx_bytes": float(values["tx_bytes"]),
            "rx_bytes": float(values["rx_bytes"]),
            "tx_packets": float(values["tx_packets"]),
            "rx_packets": float(values["rx_packets"]),
            "tx_drops": float(values["tx_drops"]),
            "rx_drops": float(values["rx_drops"]),
            "tx_errs": float(values["tx_errs"]),
            "rx_errs": float(values["rx_errs"]),
        }
    return rows


def snapshot_to_queue_rows(
    snapshot: NodeTelemetrySnapshot, node: object
) -> List[Dict[str, object]]:
    """Build queue rows from the qdisc output of a node snapshot."""
    return parse_queue_rows(node, snapshot.qdisc_raw, snapshot.timestamp)


def snapshot_to_route_row(snapshot: NodeTelemetrySnapshot) -> Dict[str, object]:
    """Build the route row from the routing output of a node snapshot."""
    return parse_route_row(snapshot.node_name, snapshot.routes_raw, snapshot.timestamp)


def snapshot_to_neighbor_row(snapshot: NodeTelemetrySnapshot) -> Dict[str, object]:
    """Build the neighbor row from the neighbor output of a node snapshot."""
    return parse_neighbor_row(snapshot.node_name, snapshot.neighbors_raw, snapshot.timestamp)


def collect_telemetry_batch(
    net: Mininet,
    state: SimulatorState,
    *,
    snapshot_id: int | None = None,
    timestamp: str | None = None,
) -> TelemetryBatch:
    """Collect one atomic snapshot while preserving per-family failures."""
    collection_errors: List[str] = []
    raw_started = time.monotonic()
    snapshots = collect_node_snapshots(
        net,
        timestamp=timestamp,
        now=raw_started,
        errors=collection_errors,
    )
    raw_duration = time.monotonic() - raw_started
    node_by_name = {logical_name(node): node for node in list(net.hosts) + list(net.switches)}
    rows_by_family: Dict[str, List[Dict[str, object]]] = {
        "Node": [],
        "Interface": [],
        "Queue": [],
        "Route": [],
        "Neighbor": [],
    }
    family_errors: Dict[str, List[str]] = {name: [] for name in rows_by_family}
    durations: Dict[str, float] = {}

    converters: Dict[str, Callable[[NodeTelemetrySnapshot], object]] = {
        "Node": snapshot_to_node_row,
        "Interface": lambda snapshot: snapshot_to_interface_rows(snapshot, state),
        "Queue": lambda snapshot: snapshot_to_queue_rows(
            snapshot, node_by_name[snapshot.node_name]
        ),
        "Route": snapshot_to_route_row,
        "Neighbor": snapshot_to_neighbor_row,
    }
    for family, converter in converters.items():
        started = time.monotonic()
        for snapshot in snapshots:
            try:
                converted = converter(snapshot)
                if isinstance(converted, list):
                    rows_by_family[family].extend(converted)
                else:
                    rows_by_family[family].append(converted)
            except Exception as exc:
                family_errors[family].append(
                    f"family={family} node={snapshot.node_name}:{type(exc).__name__}:{exc}"
                )
        durations[family] = time.monotonic() - started

    if snapshot_id is not None:
        for rows in rows_by_family.values():
            for row in rows:
                row["SnapshotId"] = snapshot_id

    expected_nodes = len(list(net.hosts) + list(net.switches))
    statuses: Dict[str, str] = {}
    for family in rows_by_family:
        errors = [*collection_errors, *family_errors[family]]
        if not snapshots or not rows_by_family[family]:
            statuses[family] = "error" if errors else "empty"
        elif errors or len(snapshots) != expected_nodes:
            statuses[family] = "partial"
        else:
            statuses[family] = "complete"

    durations["Node"] += raw_duration
    errors = [*collection_errors]
    for entries in family_errors.values():
        errors.extend(entries)
    return TelemetryBatch(
        node_rows=rows_by_family["Node"],
        interface_rows=rows_by_family["Interface"],
        queue_rows=rows_by_family["Queue"],
        route_rows=rows_by_family["Route"],
        neighbor_rows=rows_by_family["Neighbor"],
        durations_sec=durations,
        statuses=statuses,
        errors=errors,
    )


def parse_ping_output(output: str) -> Dict[str, Optional[float]]:
    """Parse packet loss and round-trip time statistics from ping output, NaN when absent."""
    loss_match = re.search(r"(\d+(?:\.\d+)?)%\s+packet loss", output)
    rtt_match = re.search(
        r"rtt min/avg/max/mdev = "
        r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?) ms",
        output,
    )
    return {
        "packet_loss": float(loss_match.group(1)) if loss_match else math.nan,
        "min_rtt": float(rtt_match.group(1)) if rtt_match else math.nan,
        "avg_rtt": float(rtt_match.group(2)) if rtt_match else math.nan,
        "max_rtt": float(rtt_match.group(3)) if rtt_match else math.nan,
        "mdev_rtt": float(rtt_match.group(4)) if rtt_match else math.nan,
    }


def _ping_missing_row(timestamp: str, source_name: str, destination_name: str) -> Dict[str, object]:
    """Return a probe row with every measurement set to NaN."""
    return {
        "Timestamp": timestamp,
        "Source": source_name,
        "Destination": destination_name,
        "MinRTT": math.nan,
        "AvgRTT": math.nan,
        "MaxRTT": math.nan,
        "MdevRTT": math.nan,
        "PacketLoss": math.nan,
        # Missing/stale results are not successful probes.
        "TimeoutFlag": math.nan,
    }


def _ping_timeout_row(timestamp: str, source_name: str, destination_name: str) -> Dict[str, object]:
    """Return a finite timeout outcome when a probe exceeds its hard deadline."""
    row = _ping_missing_row(timestamp, source_name, destination_name)
    row.update(PacketLoss=100.0, TimeoutFlag=1.0)
    return row


def _ping_row(
    timestamp: str, source_name: str, destination_name: str, output: str
) -> Dict[str, object]:
    """Build a probe row from ping output with the timeout flag set on total loss."""
    metrics = parse_ping_output(output)
    timeout_flag = (
        1.0 if "100% packet loss" in output or "Destination Host Unreachable" in output else 0.0
    )
    return {
        "Timestamp": timestamp,
        "Source": source_name,
        "Destination": destination_name,
        "MinRTT": metrics["min_rtt"],
        "AvgRTT": metrics["avg_rtt"],
        "MaxRTT": metrics["max_rtt"],
        "MdevRTT": metrics["mdev_rtt"],
        "PacketLoss": metrics["packet_loss"],
        "TimeoutFlag": timeout_flag,
    }


def _ping_command(destination_ip: str, packets: int, timeout_sec: float) -> list[str]:
    """Return the ping command line with its packet count and a whole-second deadline."""
    deadline_sec = max(1, int(math.ceil(timeout_sec)))
    return [
        "ping",
        "-c",
        str(max(1, packets)),
        "-W",
        str(deadline_sec),
        "-w",
        str(deadline_sec),
        destination_ip,
    ]


@dataclasses.dataclass
class ProbeSample:
    """Latest result of one probe pair with its completion instant."""

    completed_at_mono: float
    row: Dict[str, object]
    timed_out: bool = False


@dataclasses.dataclass
class ProbeProcess:
    """One running ping process with its pair and scheduling instants."""

    source_name: str
    destination_name: str
    pair_key: tuple[str, str]
    process: subprocess.Popen[bytes]
    scheduled_at_mono: float
    launched_at_mono: float


class AsyncPingProbeScheduler:
    """Scheduler that runs one probe per source at a time on a fixed cadence and keeps the freshest result per pair."""
    
    def __init__(
        self,
        host_pairs: List[Tuple[object, object]],
        packets: int = PING_COUNT,
        timeout_sec: float = PING_TIMEOUT_SEC,
        cadence_sec: float = 1.0,
        freshness_sec: float = 10.0,
        timing_sink: Callable[[Dict[str, object]], None] | None = None,
        timing_origin_mono: float | None = None,
    ):
        self.host_pairs = list(host_pairs)
        self.packets = max(1, int(packets))
        self.timeout_sec = max(0.001, float(timeout_sec))
        self.cadence_sec = max(0.001, float(cadence_sec))
        self.freshness_sec = max(0.001, float(freshness_sec))
        self.timing_sink = timing_sink
        self.timing_origin_mono = timing_origin_mono
        self._pairs_by_source: Dict[str, List[Tuple[object, object]]] = {}
        self._source_cursor: Dict[str, int] = {}
        self._source_active: Dict[str, ProbeProcess] = {}
        self._next_due_mono: Dict[tuple[str, str], float] = {}
        self._latest_samples: Dict[tuple[str, str], ProbeSample] = {}
        for source, destination in self.host_pairs:
            self._pairs_by_source.setdefault(logical_name(source), []).append((source, destination))
        for source_name in self._pairs_by_source:
            self._source_cursor[source_name] = 0

    def tick(self, now: float | None = None) -> None:
        """Collect finished probes and launch the next due pair of every idle source."""
        current = time.monotonic() if now is None else float(now)
        if self.timing_origin_mono is None:
            self.timing_origin_mono = current
        self._collect_finished(current)
        for source_name, pairs in self._pairs_by_source.items():
            if source_name in self._source_active or not pairs:
                continue
            pair = self._next_eligible_pair(source_name, pairs, current)
            if pair is None:
                continue
            source, destination = pair
            self._launch_probe(source, destination, current)

    def latest_rows(
        self,
        timestamp: str | None = None,
        now: float | None = None,
        snapshot_id: int | None = None,
    ) -> List[Dict[str, object]]:
        """Return one row per configured pair, fresh results or NaN rows for missing and stale ones."""
        current = time.monotonic() if now is None else float(now)
        row_timestamp = utc_timestamp() if timestamp is None else timestamp
        rows: List[Dict[str, object]] = []
        for source, destination in self.host_pairs:
            key = (logical_name(source), logical_name(destination))
            sample = self._latest_samples.get(key)
            result_age = None if sample is None else max(0.0, current - sample.completed_at_mono)
            if sample is None or result_age is None or result_age > self.freshness_sec:
                rows.append(_ping_missing_row(row_timestamp, key[0], key[1]))
                self._emit_timing(
                    source=key[0],
                    destination=key[1],
                    status="missing" if sample is None else "stale",
                    snapshot_id=snapshot_id,
                    result_age_sec=result_age,
                )
                continue
            row = dict(sample.row)
            row["Timestamp"] = row_timestamp
            rows.append(row)
            self._emit_timing(
                source=key[0],
                destination=key[1],
                status="fresh",
                snapshot_id=snapshot_id,
                completed_at_mono=sample.completed_at_mono,
                result_age_sec=result_age,
                timed_out=sample.timed_out,
            )
        return rows

    def missing_sample_pairs(self) -> list[tuple[str, str]]:
        """Return configured pairs that have not produced any result yet."""
        return [
            (logical_name(source), logical_name(destination))
            for source, destination in self.host_pairs
            if (logical_name(source), logical_name(destination)) not in self._latest_samples
        ]

    def close(self) -> None:
        """Terminate every running probe and record it as cancelled."""
        for probe in list(self._source_active.values()):
            self._emit_timing(
                source=probe.source_name,
                destination=probe.destination_name,
                status="cancelled",
                scheduled_at_mono=probe.scheduled_at_mono,
                launched_at_mono=probe.launched_at_mono,
                completed_at_mono=time.monotonic(),
            )
            try:
                probe.process.terminate()
            except Exception:
                continue
        for probe in list(self._source_active.values()):
            try:
                probe.process.communicate(timeout=0.2)
            except Exception:
                try:
                    probe.process.kill()
                    probe.process.communicate()
                except Exception:
                    pass
        self._source_active.clear()

    def _collect_finished(self, now: float) -> None:
        """Turn completed or overdue probe processes into samples and timing rows."""
        finished_sources: List[str] = []
        for source_name, probe in self._source_active.items():
            forced_timeout = False
            if probe.process.poll() is None:
                hard_timeout = max(1.0, self.timeout_sec) + 0.5
                if now - probe.launched_at_mono <= hard_timeout:
                    continue
                forced_timeout = True
                try:
                    probe.process.terminate()
                    stdout, stderr = probe.process.communicate(timeout=0.1)
                except Exception:
                    probe.process.kill()
                    stdout, stderr = probe.process.communicate()
            else:
                stdout, stderr = probe.process.communicate()
            output = stdout.decode("utf-8", errors="replace")
            err_output = stderr.decode("utf-8", errors="replace")
            if err_output:
                output = f"{output}\n{err_output}".strip()
            timestamp = utc_timestamp()
            status = "scheduler_timeout" if forced_timeout else "error"
            if forced_timeout:
                # The process exceeded the scheduler's bounded hard deadline.
                row = _ping_timeout_row(timestamp, probe.source_name, probe.destination_name)
            else:
                row = _ping_row(timestamp, probe.source_name, probe.destination_name, output)
                if row["TimeoutFlag"] == 1.0:
                    status = "timeout"
                elif math.isfinite(float(row["PacketLoss"])):
                    status = "complete"
            self._latest_samples[probe.pair_key] = ProbeSample(
                completed_at_mono=now,
                row=row,
                timed_out=status in {"timeout", "scheduler_timeout"},
            )
            self._emit_timing(
                source=probe.source_name,
                destination=probe.destination_name,
                status=status,
                scheduled_at_mono=probe.scheduled_at_mono,
                launched_at_mono=probe.launched_at_mono,
                completed_at_mono=now,
                timed_out=status in {"timeout", "scheduler_timeout"},
            )
            finished_sources.append(source_name)
        for source_name in finished_sources:
            self._source_active.pop(source_name, None)

    def _next_eligible_pair(
        self,
        source_name: str,
        pairs: List[Tuple[object, object]],
        now: float,
    ) -> Tuple[object, object] | None:
        """Return the next due pair of a source in round-robin order, dropping missed deadlines."""
        start = self._source_cursor.get(source_name, 0)
        for offset in range(len(pairs)):
            index = (start + offset) % len(pairs)
            source, destination = pairs[index]
            key = (logical_name(source), logical_name(destination))
            due = self._next_due_mono.setdefault(key, now)
            if now >= due:
                # Drop older fixed-rate opportunities instead of accumulating a queue 
                # that would later create a probe burst.
                while due + self.cadence_sec <= now:
                    self._emit_timing(
                        source=key[0],
                        destination=key[1],
                        status="skipped_deadline",
                        scheduled_at_mono=due,
                    )
                    due += self.cadence_sec
                self._next_due_mono[key] = due
                self._source_cursor[source_name] = (index + 1) % len(pairs)
                return source, destination
        return None

    def _launch_probe(self, source: object, destination: object, now: float) -> None:
        """Start a ping from source to destination and record the launch."""
        source_name = logical_name(source)
        destination_name = logical_name(destination)
        destination_ip = node_primary_ip(destination) or destination.IP()
        process = source.popen(
            _ping_command(destination_ip, packets=self.packets, timeout_sec=self.timeout_sec)
        )
        pair_key = (source_name, destination_name)
        scheduled_at = self._next_due_mono.get(pair_key, now)
        self._next_due_mono[pair_key] = scheduled_at + self.cadence_sec
        self._source_active[source_name] = ProbeProcess(
            source_name=source_name,
            destination_name=destination_name,
            pair_key=pair_key,
            process=process,
            scheduled_at_mono=scheduled_at,
            launched_at_mono=now,
        )
        self._emit_timing(
            source=source_name,
            destination=destination_name,
            status="launched",
            scheduled_at_mono=scheduled_at,
            launched_at_mono=now,
        )

    def _emit_timing(
        self,
        *,
        source: str,
        destination: str,
        status: str,
        snapshot_id: int | None = None,
        scheduled_at_mono: float | None = None,
        launched_at_mono: float | None = None,
        completed_at_mono: float | None = None,
        result_age_sec: float | None = None,
        timed_out: bool = False,
    ) -> None:
        """Send one probe timing row with offsets from the timing origin to the sink."""
        if self.timing_sink is None:
            return
        origin = self.timing_origin_mono

        def offset(value: float | None) -> float | str:
            """Return the offset from the timing origin, or an empty string when unset."""
            if value is None or origin is None:
                return ""
            return round(value - origin, 6)

        launch_lag = ""
        if scheduled_at_mono is not None and launched_at_mono is not None:
            launch_lag = round(max(0.0, launched_at_mono - scheduled_at_mono), 6)
        duration = ""
        if launched_at_mono is not None and completed_at_mono is not None:
            duration = round(max(0.0, completed_at_mono - launched_at_mono), 6)
        self.timing_sink(
            {
                "SnapshotId": "" if snapshot_id is None else snapshot_id,
                "Source": source,
                "Destination": destination,
                "Status": status,
                "ScheduledOffsetSec": offset(scheduled_at_mono),
                "LaunchOffsetSec": offset(launched_at_mono),
                "CompletedOffsetSec": offset(completed_at_mono),
                "LaunchLagSec": launch_lag,
                "DurationSec": duration,
                "ResultAgeSec": ""
                if result_age_sec is None
                else round(max(0.0, result_age_sec), 6),
                "TimedOut": bool(timed_out),
                "Timestamp": utc_timestamp(),
            }
        )


def _safe_float(value: object) -> float:
    """Convert to float, returning zero when the value is malformed."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
