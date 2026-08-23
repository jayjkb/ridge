"""Planning, launch, and validation of the D-ITG background traffic and congestion bursts of an episode."""

from __future__ import annotations

import re
import shlex
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mininet.log import info
from mininet.net import Mininet

from .common import DITG_DATA_PORT_MIN
from .state import SimulatorState
from .topology import logical_name, logical_node, node_primary_ip

DITG_DATA_PORT_MAX = 60000

_STARTED_FLOW_PATTERN = re.compile(r"Started sending packets of flow ID:\s*\d+")
_FATAL_PATTERNS = (
    re.compile(r"\*\*\s*ERROR\s*\*\*", re.IGNORECASE),
    re.compile(r"\bERROR_TERMINATE\b", re.IGNORECASE),
    re.compile(r"\bFlow\s+\d+\s+aborted\b", re.IGNORECASE),
    re.compile(r"\bReceiver is down\b", re.IGNORECASE),
    re.compile(r"\bFinish requested caused by errors\b", re.IGNORECASE),
    re.compile(r"\bError - FlowSender interrupted by an error\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class DitgFlow:
    """One pre-planned D-ITG flow, timed relative to a shared activation."""

    flow_id: int
    kind: str
    burst_id: int | None
    src: str
    dst: str
    start_offset_ms: int
    duration_ms: int
    rate_pps: int
    packet_size: int
    protocol: str
    data_port: int
    on_duration_ms: int | None = None
    off_duration_ms: int | None = None

    def __post_init__(self) -> None:
        """Reject a flow whose identifiers, endpoints, timing, protocol, port, or burst settings are invalid."""
        if self.flow_id < 1:
            raise ValueError("D-ITG flow_id must be positive")
        if self.kind not in {"baseline", "burst"}:
            raise ValueError("D-ITG flow kind must be 'baseline' or 'burst'")
        if self.kind == "baseline" and self.burst_id is not None:
            raise ValueError("Baseline D-ITG flows cannot have a burst_id")
        if self.kind == "burst" and (self.burst_id is None or self.burst_id < 0):
            raise ValueError("Burst D-ITG flows require a non-negative burst_id")
        if not self.src or not self.dst or self.src == self.dst:
            raise ValueError("D-ITG flows require distinct source and destination nodes")
        if self.start_offset_ms < 0:
            raise ValueError("D-ITG start_offset_ms cannot be negative")
        if min(self.duration_ms, self.rate_pps, self.packet_size) <= 0:
            raise ValueError("D-ITG duration, rate, and packet size must be positive")
        if self.protocol.upper() not in {"UDP", "TCP"}:
            raise ValueError("D-ITG protocol must be UDP or TCP")
        if not DITG_DATA_PORT_MIN <= self.data_port <= DITG_DATA_PORT_MAX:
            raise ValueError(
                f"D-ITG data_port must be in [{DITG_DATA_PORT_MIN}, {DITG_DATA_PORT_MAX}]"
            )
        if (self.on_duration_ms is None) != (self.off_duration_ms is None):
            raise ValueError("D-ITG ON/OFF durations must be provided together")
        if (
            self.on_duration_ms is not None
            and min(self.on_duration_ms, self.off_duration_ms or 0) <= 0
        ):
            raise ValueError("D-ITG ON/OFF durations must be positive")


def allocate_ditg_data_port(state: SimulatorState) -> int:
    """Allocate an episode-local D-ITG data port outside its logging range."""
    port = max(DITG_DATA_PORT_MIN, int(getattr(state, "ditg_next_data_port", 0) or 0))
    if port > DITG_DATA_PORT_MAX:
        raise RuntimeError("D-ITG episode data-port range exhausted")
    state.ditg_next_data_port = port + 1
    return port


def build_ditg_flows(
    state: SimulatorState,
    pairs: Sequence[tuple[object, object]],
    *,
    kind: str,
    burst_id: int | None,
    start_offset_ms: int,
    duration_ms: int,
    rate_pps: int,
    packet_size: int,
    protocol: str,
    on_duration_ms: int | None = None,
    off_duration_ms: int | None = None,
) -> list[DitgFlow]:
    """Build deterministic flow records and allocate their IDs and data ports."""
    next_flow_id = max(1, int(getattr(state, "ditg_next_flow_id", 1) or 1))
    flows: list[DitgFlow] = []
    for index, (src, dst) in enumerate(pairs):
        flows.append(
            DitgFlow(
                flow_id=next_flow_id + index,
                kind=kind,
                burst_id=burst_id,
                src=src if isinstance(src, str) else logical_name(src),
                dst=dst if isinstance(dst, str) else logical_name(dst),
                start_offset_ms=start_offset_ms,
                duration_ms=duration_ms,
                rate_pps=rate_pps,
                packet_size=packet_size,
                protocol=protocol.upper(),
                data_port=allocate_ditg_data_port(state),
                on_duration_ms=on_duration_ms,
                off_duration_ms=off_duration_ms,
            )
        )
    state.ditg_next_flow_id = next_flow_id + len(flows)
    return flows


def _safe_name(value: str) -> str:
    """Replace characters unsafe in file names with underscores."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _background_pid(node: object, command: str) -> str:
    """Run a backgrounding command in a node and return the process identifier it echoes."""
    output = node.cmd(f"sh -c {shlex.quote(command)}").strip().splitlines()
    return output[-1].strip() if output else ""


def _pid_is_alive(node: object, pid: str) -> bool:
    """Return whether a process with the given identifier still exists in the node."""
    if not pid or not str(pid).strip().isdigit():
        return False
    output = node.cmd(f"kill -0 {pid} >/dev/null 2>&1; echo $?").strip().splitlines()
    return bool(output) and output[-1] == "0"


def _receiver_command(stderr_log: Path) -> str:
    """Start a receiver with compact text diagnostics only."""
    return f"ITGRecv </dev/null >>{shlex.quote(str(stderr_log))} 2>&1 & echo $!"


def _script_line(flow: DitgFlow, destination_ip: str) -> str:
    """Return the ITGSend script line for one flow, with the burst-mode option last."""
    options = [
        "-a",
        shlex.quote(destination_ip),
        "-rp",
        str(flow.data_port),
        "-T",
        flow.protocol.upper(),
        "-C",
        str(flow.rate_pps),
        "-c",
        str(flow.packet_size),
        "-t",
        str(flow.duration_ms),
        "-d",
        str(flow.start_offset_ms),
    ]
    if flow.on_duration_ms is not None:
        # D-ITG requires the native burst-mode option to be last.
        options.extend(
            (
                "-B",
                "C",
                str(flow.on_duration_ms),
                "C",
                str(flow.off_duration_ms),
            )
        )
    return " ".join(options)


def _source_master_command(
    script: Path,
    stderr_log: Path,
    *,
    activation_mono: float,
) -> str:
    """Return a command that sleeps until the shared activation instant and then runs ITGSend."""
    delay_sec = max(0.0, float(activation_mono) - time.monotonic())
    foreground = (
        f"sleep {delay_sec:.6f}; exec ITGSend {shlex.quote(str(script))} "
        f"</dev/null >>{shlex.quote(str(stderr_log))} 2>&1"
    )
    return f"({foreground}) & echo $!"


def _record_process(
    state: SimulatorState,
    *,
    role: str,
    node: str,
    pid: str,
    stderr_log: Path,
    expected_flow_count: int,
    script: Path | None = None,
) -> None:
    """Append one launched D-ITG process with its role, log path, and expected flow count to the episode state."""
    state.ditg_processes.append(
        {
            "role": role,
            "node": node,
            "pid": pid,
            "stderr_log": str(stderr_log),
            "script": str(script or ""),
            "expected_flow_count": str(expected_flow_count),
        }
    )


def _validated_plan(flows: Sequence[DitgFlow]) -> list[DitgFlow]:
    """Return the flows sorted by identifier after checking that identifiers and data ports are unique."""
    ordered = sorted(flows, key=lambda flow: flow.flow_id)
    if len({flow.flow_id for flow in ordered}) != len(ordered):
        raise ValueError("D-ITG flow IDs must be unique within an episode")
    if len({flow.data_port for flow in ordered}) != len(ordered):
        raise ValueError("D-ITG data ports must be unique within an episode")
    return ordered


def launch_ditg_plan(
    net: Mininet,
    state: SimulatorState,
    flows: Sequence[DitgFlow],
    *,
    activation_mono: float,
) -> dict[str, object]:
    """Launch one receiver per destination and one multi-flow master per source."""
    ordered = _validated_plan(flows)
    by_source: dict[str, list[DitgFlow]] = defaultdict(list)
    by_destination: dict[str, list[DitgFlow]] = defaultdict(list)
    for flow in ordered:
        by_source[flow.src].append(flow)
        by_destination[flow.dst].append(flow)

    result: dict[str, object] = {
        "planned_flow_count": len(ordered),
        "baseline_flow_count": sum(flow.kind == "baseline" for flow in ordered),
        "burst_flow_count": sum(flow.kind == "burst" for flow in ordered),
        "source_process_attempted_count": len(by_source),
        "source_process_launched_count": 0,
        "source_process_failed_count": 0,
        "receiver_process_attempted_count": len(by_destination),
        "receiver_process_launched_count": 0,
        "receiver_process_failed_count": 0,
        "first_failure_reason": "",
    }
    if not ordered:
        return result

    info(
        f"*** Starting D-ITG plan flows={len(ordered)} "
        f"sources={len(by_source)} destinations={len(by_destination)}\n"
    )
    log_dir = state.log_dir / "ditg"
    log_dir.mkdir(parents=True, exist_ok=True)

    destination_ips: dict[str, str] = {}
    for destination in sorted(by_destination):
        node = logical_node(net, destination)
        destination_ips[destination] = node_primary_ip(node) or node.IP()
        safe_destination = _safe_name(destination)
        stderr_log = log_dir / f"receiver_{safe_destination}.stderr.log"
        pid = _background_pid(node, _receiver_command(stderr_log))
        _record_process(
            state,
            role="receiver",
            node=destination,
            pid=pid,
            stderr_log=stderr_log,
            expected_flow_count=len(by_destination[destination]),
        )
        state.ditg_receivers[destination] = pid
        if _pid_is_alive(node, pid):
            result["receiver_process_launched_count"] = (
                int(result["receiver_process_launched_count"]) + 1
            )
        else:
            result["receiver_process_failed_count"] = (
                int(result["receiver_process_failed_count"]) + 1
            )
            if not result["first_failure_reason"]:
                result["first_failure_reason"] = f"receiver_launch_failed:{destination}"

    # Every script is materialized before any source master can activate.
    source_files: dict[str, tuple[Path, Path]] = {}
    for source in sorted(by_source):
        safe_source = _safe_name(source)
        script = log_dir / f"source_{safe_source}.ditg.script"
        stderr_log = log_dir / f"source_{safe_source}.stderr.log"
        lines = [
            _script_line(flow, destination_ips[flow.dst])
            for flow in sorted(by_source[source], key=lambda flow: flow.flow_id)
        ]
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        source_files[source] = (script, stderr_log)

    for source in sorted(by_source):
        node = logical_node(net, source)
        script, stderr_log = source_files[source]
        pid = _background_pid(
            node,
            _source_master_command(
                script,
                stderr_log,
                activation_mono=activation_mono,
            ),
        )
        _record_process(
            state,
            role="source_master",
            node=source,
            pid=pid,
            stderr_log=stderr_log,
            script=script,
            expected_flow_count=len(by_source[source]),
        )
        if _pid_is_alive(node, pid):
            result["source_process_launched_count"] = (
                int(result["source_process_launched_count"]) + 1
            )
        else:
            result["source_process_failed_count"] = int(result["source_process_failed_count"]) + 1
            if not result["first_failure_reason"]:
                result["first_failure_reason"] = f"source_master_launch_failed:{source}"

    return result


def ditg_runtime_failures(state: SimulatorState, net: Mininet) -> list[dict[str, str]]:
    """Report any planned receiver or source master that is no longer alive."""
    failures: list[dict[str, str]] = []
    for process in state.ditg_processes:
        node = logical_node(net, str(process.get("node", "")))
        if not _pid_is_alive(node, str(process.get("pid", ""))):
            failure = dict(process)
            failure["reason"] = "process_not_alive"
            failures.append(failure)
    return failures


def validate_ditg_diagnostics(state: SimulatorState) -> dict[str, object]:
    """Validate fatal output and Started-flow counts after planned traffic ends."""
    failures: list[str] = []
    fatal_marker_count = 0
    expected_started = 0
    observed_started = 0
    for process in state.ditg_processes:
        role = str(process.get("role", ""))
        stderr_path = Path(str(process.get("stderr_log", "")))
        if not stderr_path.is_file():
            failures.append(f"missing_diagnostic_log:{role}:{process.get('node', '')}")
            continue
        contents = stderr_path.read_text(encoding="utf-8", errors="replace")
        process_fatal_count = sum(len(pattern.findall(contents)) for pattern in _FATAL_PATTERNS)
        fatal_marker_count += process_fatal_count
        if process_fatal_count:
            failures.append(
                f"fatal_diagnostic_marker:{role}:{process.get('node', '')}:"
                f"count={process_fatal_count}"
            )
        if role != "source_master":
            continue
        expected = int(process.get("expected_flow_count", "0") or 0)
        observed = len(_STARTED_FLOW_PATTERN.findall(contents))
        expected_started += expected
        observed_started += observed
        if observed != expected:
            failures.append(
                f"started_flow_count_mismatch:{process.get('node', '')}:"
                f"expected={expected}:observed={observed}"
            )
    return {
        "passed": not failures,
        "expected_started_flow_count": expected_started,
        "observed_started_flow_count": observed_started,
        "fatal_marker_count": fatal_marker_count,
        "failures": failures,
    }


def stop_ditg_traffic(state: SimulatorState, net: Mininet) -> None:
    """Stop only D-ITG PIDs recorded by this episode."""
    info("*** Stopping D-ITG traffic\n")
    for process in reversed(state.ditg_processes):
        pid = str(process.get("pid", ""))
        node_name = str(process.get("node", ""))
        if not pid.isdigit() or not node_name:
            continue
        try:
            node = logical_node(net, node_name)
            node.cmd(f"kill {pid} >/dev/null 2>&1 || true")
            if _pid_is_alive(node, pid):
                node.cmd(f"kill -9 {pid} >/dev/null 2>&1 || true")
        except Exception as exc:
            info(f"*** failed to stop D-ITG pid={pid} node={node_name}: {exc}\n")
    state.ditg_processes.clear()
    state.ditg_receivers.clear()
