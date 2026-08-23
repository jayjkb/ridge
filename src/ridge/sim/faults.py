"""Admissible fault targets and the injection, maintenance, and recovery of the four Stage-1 fault categories."""

from __future__ import annotations

import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from mininet.net import Mininet

from ridge.common.contracts import (
    FAULT_CATEGORY_DRAIN,
    FAULT_CATEGORY_FIBER_CUT,
    FAULT_CATEGORY_LINK_DEGRADATION,
    FAULT_CATEGORY_LINK_FLAP,
)
from ridge.common.contracts import (
    FAULT_CATEGORY_NONE as FAULT_CATEGORY_NONE,
)

from .common import utc_timestamp
from .state import SimulatorState
from .topology import (
    EDGE_ROLE_FABRIC,
    EDGE_ROLE_WAN,
    candidate_edge_ids_by_roles,
    canonical_edge_id,
    edge_role,
    logical_interface_name,
    logical_name,
    logical_node,
    topology_spec,
)

FAULT_SCHEDULE_NONE = "none"
FAULT_SCHEDULE_STAGED_DRAIN = "staged_drain"
FAULT_SCHEDULE_INSTANT_LINK_DOWN = "instant_link_down"
FAULT_SCHEDULE_NETEM_LINK_DEGRADATION = "netem_link_degradation"
FAULT_SCHEDULE_BURSTY_LINK_FLAP = "bursty_link_flap"
OBSERVED_NODE_FAULT_TARGETS = tuple(sorted(str(name) for name in topology_spec()["switches"]))

OBSERVED_EDGE_FAULT_TARGETS = tuple(
    sorted(
        canonical_edge_id(str(link["src"]), str(link["dst"])) for link in topology_spec()["links"]
    )
)

# Routers whose full drain still leaves every host pair connected by an alternative path.
_DRAIN_NODE_FAULT_TARGETS_RAW = (
    "r19",
    "r21",
    "r25",
)

_FIBER_CUT_EDGE_FAULT_TARGETS_RAW = (
    "r18<->r4",
    "r28<->r4",
    "r21<->r4",
    "r25<->r6",
    "r15<->r8",
    "r25<->r8",
    "r15<->r21",
    "r18<->r28",
    "r21<->r25",
    "r19<->r28",
    "r19<->r6",
)

RAMP_MIN_RATE_MBPS = 0.5
RAMP_BURST = "32kbit"
RAMP_LATENCY = "400ms"
LINK_FLAP_COUNT_RANGE = (2, 5)
LINK_FLAP_DOWN_DURATION_RANGE_SEC = (1.5, 5.0)
LINK_FLAP_UP_GAP_RANGE_SEC = (3.0, 10.0)
TC_TIME_ABS_TOLERANCE_MS = 0.1


@dataclass(frozen=True)
class LinkDegradationSeverityProfile:
    """One link degradation severity level with its delay, jitter, and loss sampling ranges."""
    name: str
    delay_ms: tuple[float, float]
    jitter_ms: tuple[float, float]
    loss_pct: tuple[float, float]


LINK_DEGRADATION_SEVERITY_PROFILES: tuple[LinkDegradationSeverityProfile, ...] = (
    LinkDegradationSeverityProfile("mild", (5.0, 10.0), (1.0, 3.0), (0.2, 0.8)),
    LinkDegradationSeverityProfile("medium", (15.0, 30.0), (3.0, 8.0), (1.0, 3.0)),
    LinkDegradationSeverityProfile("severe", (40.0, 80.0), (8.0, 20.0), (3.0, 8.0)),
)


@dataclass(frozen=True)
class DrainPhaseRatios:
    """Fractions of the fault window given to the four drain phases."""
    ramp_down: float
    link_down: float
    hold_down: float
    ramp_up: float


@dataclass(frozen=True)
class DrainPhaseDurations:
    """Durations in seconds of the four drain phases."""
    ramp_down: float
    link_down: float
    hold_down: float
    ramp_up: float


@dataclass(frozen=True)
class DrainConfig:
    """Fault window length, number of rate steps, and phase fractions of one drain."""
    duration_sec: int
    ramp_steps: int
    phase_ratios: DrainPhaseRatios


def _host_connectivity_preserved(
    link_excluded: Callable[[dict[str, Any], str, str], bool],
) -> bool:
    """BFS check that every host stays reachable once excluded links are removed."""
    spec = topology_spec()
    hosts = [str(name) for name in spec["hosts"]]
    graph: dict[str, set[str]] = {str(name): set() for name in [*spec["switches"], *spec["hosts"]]}
    for link in spec["links"]:
        src = str(link["src"])
        dst = str(link["dst"])
        if link_excluded(link, src, dst):
            continue
        graph[src].add(dst)
        graph[dst].add(src)

    start = hosts[0]
    seen = {start}
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if nbr in seen:
                continue
            seen.add(nbr)
            queue.append(nbr)
    return all(host in seen for host in hosts)


def _host_connectivity_preserved_when_drained(fault_target: str) -> bool:
    """Return whether every host stays reachable with the router's infrastructure links removed."""
    def _excluded(link: dict[str, Any], src: str, dst: str) -> bool:
        """Exclude non-access links that touch the drained router."""
        profile = str(link["profile"])
        return profile != "host" and (src == fault_target or dst == fault_target)

    return _host_connectivity_preserved(_excluded)


def _validated_drain_node_fault_targets() -> tuple[str, ...]:
    """Return the drain targets after checking each exists and is not a single point of failure."""
    validated: list[str] = []
    for node in _DRAIN_NODE_FAULT_TARGETS_RAW:
        if node not in OBSERVED_NODE_FAULT_TARGETS:
            raise ValueError(f"Drain target {node} is not present in the active topology")
        if not _host_connectivity_preserved_when_drained(node):
            raise ValueError(f"Drain target {node} is a host-connectivity SPOF under full drain")
        validated.append(node)
    return tuple(validated)


DRAIN_NODE_FAULT_TARGETS = _validated_drain_node_fault_targets()


def _host_connectivity_preserved_when_edge_removed(edge_id: str) -> bool:
    """Return whether every host stays reachable with one link removed."""
    try:
        src, dst = edge_id.split("<->", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid edge fault target={edge_id}") from exc
    removed_edge = canonical_edge_id(src, dst)

    def _excluded(_link: dict[str, Any], src: str, dst: str) -> bool:
        """Exclude the link whose canonical identifier matches the removed one."""
        return canonical_edge_id(src, dst) == removed_edge

    return _host_connectivity_preserved(_excluded)


def _validated_edge_fault_targets() -> tuple[str, ...]:
    """Return the link fault targets after checking each exists, is infrastructure, and is not a single point of failure."""
    validated: list[str] = []
    infrastructure_edges = set(candidate_edge_ids_by_roles({EDGE_ROLE_FABRIC, EDGE_ROLE_WAN}))
    for edge_id in _FIBER_CUT_EDGE_FAULT_TARGETS_RAW:
        try:
            src, dst = edge_id.split("<->", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid fiber cut target={edge_id}") from exc
        normalized = canonical_edge_id(src, dst)
        if normalized not in OBSERVED_EDGE_FAULT_TARGETS:
            raise ValueError(f"Fiber cut target {edge_id} is not present in the active topology")
        if normalized not in infrastructure_edges:
            raise ValueError(f"Fiber cut target {edge_id} is not an infrastructure edge")
        if not _host_connectivity_preserved_when_edge_removed(normalized):
            raise ValueError(
                f"Fiber cut target {edge_id} is a host-connectivity SPOF under edge removal"
            )
        validated.append(normalized)
    return tuple(sorted(validated))


EDGE_FAULT_TARGETS = _validated_edge_fault_targets()


def _log_fault_action(
    state: SimulatorState,
    action: str,
    target: str,
    parameters: str,
    fault_type: str,
    category: str,
) -> None:
    """Write one fault action row through the episode's fault logger."""
    state.fault_logger.log(
        action=action,
        target=target,
        parameters=parameters,
        fault_type=fault_type,
        fault_category=category,
    )


def _link_nodes(edge_id: str) -> tuple[str, str]:
    """Split a link identifier into its two endpoint names."""
    try:
        src, dst = edge_id.split("<->", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid edge fault target={edge_id}") from exc
    if not src or not dst:
        raise ValueError(f"Invalid edge fault target={edge_id}")
    return src, dst


def _edge_bandwidth_mbps(edge_id: str) -> float:
    """Return the configured bandwidth of a link, min is the minimum drain rate."""
    return max(float(_edge_attrs(edge_id)["bw"]), RAMP_MIN_RATE_MBPS)


def _edge_attrs(edge_id: str) -> dict[str, Any]:
    """Return a copy of the configured attributes of a link in the topology."""
    src, dst = _link_nodes(edge_id)
    for link in topology_spec()["links"]:
        lsrc = str(link["src"])
        ldst = str(link["dst"])
        if canonical_edge_id(lsrc, ldst) == canonical_edge_id(src, dst):
            return dict(link["attrs"])
    raise ValueError(f"Unknown edge fault target={edge_id}")


def _delay_ms(delay: object) -> float:
    """Convert a delay given with a us, ms, or s suffix to milliseconds."""
    text = str(delay).strip().lower()
    if text.endswith("us"):
        return float(text[:-2]) / 1000.0
    if text.endswith("ms"):
        return float(text[:-2])
    if text.endswith("s"):
        return float(text[:-1]) * 1000.0
    return float(text)


def _tc_time_ms(value: str, unit: str) -> float:
    """Convert a tc time value and unit to milliseconds."""
    amount = float(value)
    if unit == "us":
        return amount / 1000.0
    if unit == "ms":
        return amount
    if unit == "s":
        return amount * 1000.0
    raise ValueError(f"Unsupported tc time unit={unit}")


def _netem_delay_jitter_ms(qdisc: str) -> tuple[float, float | None]:
    """Parse the delay and optional jitter in milliseconds from netem qdisc output."""
    match = re.search(
        r"\bdelay\s+([0-9.]+)(us|ms|s)(?:\s+([0-9.]+)(us|ms|s))?\b",
        qdisc,
    )
    if not match:
        return math.nan, None
    delay_ms = _tc_time_ms(match.group(1), match.group(2))
    jitter_ms = _tc_time_ms(match.group(3), match.group(4)) if match.group(3) is not None else None
    return delay_ms, jitter_ms


def _resolve_edge_targets_for_node(fault_target: str) -> list[str]:
    """Return the infrastructure links attached to a drain target router."""
    if fault_target not in DRAIN_NODE_FAULT_TARGETS:
        raise ValueError(f"Unsupported observed node fault target={fault_target}")
    edge_ids: list[str] = []
    for link in topology_spec()["links"]:
        src = str(link["src"])
        dst = str(link["dst"])
        profile = str(link["profile"])
        if profile == "host":
            continue
        if src == fault_target or dst == fault_target:
            edge_ids.append(canonical_edge_id(src, dst))
    unique = sorted(set(edge_ids))
    if not unique:
        raise ValueError(f"No infrastructure links for fault target={fault_target}")
    return unique


def _get_link_interfaces(node: object, peer: object) -> tuple[str, str]:
    """Return the interface names at both ends of the link between two nodes."""
    connections = node.connectionsTo(peer)
    if not connections:
        raise ValueError(f"No interface found between {node.name} and {peer.name}")
    intf, peer_intf = connections[0]
    return intf.name, peer_intf.name


_COMMAND_STATUS_MARKER = "__RIDGE_COMMAND_STATUS__"


def _checked_node_command(node: object, command: str) -> str:
    """Run a shell command in a node and raise RuntimeError on a non-zero exit status."""
    wrapped = (
        f"{command}; __ridge_status=$?; printf '\n{_COMMAND_STATUS_MARKER}%s\n' \"$__ridge_status\""
    )
    output = str(node.cmd(wrapped))
    marker_index = output.rfind(_COMMAND_STATUS_MARKER)
    if marker_index < 0:
        raise RuntimeError(f"Network command did not report status: {command}")
    try:
        status_lines = output[marker_index + len(_COMMAND_STATUS_MARKER) :].strip().splitlines()
        status = int(status_lines[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Network command returned invalid status: {command}") from exc
    if status != 0:
        detail = output[:marker_index].strip()
        raise RuntimeError(f"Network command failed ({status}): {command}: {detail}")
    return output[:marker_index].strip()


def _run_tbf(node: object, interface: str, rate_mbps: float) -> None:
    """Replace the root qdisc of an interface with a token bucket at the given rate and verify it."""
    _checked_node_command(
        node,
        f"tc qdisc replace dev {interface} root "
        f"tbf rate {rate_mbps:.3f}mbit burst {RAMP_BURST} latency {RAMP_LATENCY}",
    )
    qdisc = _checked_node_command(node, f"tc qdisc show dev {interface}")
    if not re.search(r"\bqdisc\s+tbf\b", qdisc):
        raise RuntimeError(f"TBF postcondition failed for {logical_name(node)}:{interface}")


def _set_link_state(node: object, interface: str, up: bool) -> None:
    """Set an interface up or down and verify the resulting flag."""
    state = "up" if up else "down"
    _checked_node_command(node, f"ip link set dev {interface} {state}")
    link = _checked_node_command(node, f"ip -o link show dev {interface}")
    flags_match = re.search(r"<([^>]*)>", link)
    flags = set(flags_match.group(1).split(",")) if flags_match else set()
    if ("UP" in flags) != up:
        raise RuntimeError(
            f"Link-state postcondition failed for {logical_name(node)}:{interface}; "
            f"expected admin {'UP' if up else 'DOWN'}"
        )


def _configure_tc_interface(node: object, interface: str, attrs: dict[str, Any]) -> None:
    """Apply delay, jitter, loss, and shaping attributes to an interface and verify each one."""
    intf = node.intf(interface) if hasattr(node, "intf") else None
    if intf is not None and hasattr(intf, "config"):
        intf.config(**attrs)
    else:
        delay = attrs.get("delay")
        jitter = attrs.get("jitter")
        loss = attrs.get("loss")
        netem_parts = ["netem"]
        if delay is not None:
            netem_parts.extend(["delay", str(delay)])
            if jitter is not None:
                netem_parts.append(str(jitter))
        if loss is not None:
            netem_parts.extend(["loss", f"{float(loss):.4f}%"])
        _checked_node_command(
            node, f"tc qdisc replace dev {interface} root {' '.join(netem_parts)}"
        )
    qdisc = _checked_node_command(node, f"tc qdisc show dev {interface}")
    if not qdisc or "noqueue" in qdisc:
        raise RuntimeError(f"Qdisc postcondition failed for {logical_name(node)}:{interface}")
    if attrs.get("use_tbf") and not re.search(r"\bqdisc\s+tbf\b", qdisc):
        raise RuntimeError(
            f"TBF restoration postcondition failed for {logical_name(node)}:{interface}"
        )
    delay = attrs.get("delay")
    if delay is not None:
        observed_delay_ms, observed_jitter_ms = _netem_delay_jitter_ms(qdisc)
        expected_delay_ms = _delay_ms(delay)
        if not math.isclose(
            observed_delay_ms,
            expected_delay_ms,
            rel_tol=0.0,
            abs_tol=TC_TIME_ABS_TOLERANCE_MS,
        ):
            raise RuntimeError(
                f"Delay postcondition failed for {logical_name(node)}:{interface}; "
                f"expected_ms={expected_delay_ms:.6f} observed_ms={observed_delay_ms:.6f}"
            )
        jitter = attrs.get("jitter")
        if jitter is not None:
            expected_jitter_ms = _delay_ms(jitter)
            if observed_jitter_ms is None or not math.isclose(
                observed_jitter_ms,
                expected_jitter_ms,
                rel_tol=0.0,
                abs_tol=TC_TIME_ABS_TOLERANCE_MS,
            ):
                raise RuntimeError(
                    f"Jitter postcondition failed for {logical_name(node)}:{interface}; "
                    f"expected_ms={expected_jitter_ms:.6f} observed_ms={observed_jitter_ms}"
                )
    loss = attrs.get("loss")
    if loss is not None:
        loss_match = re.search(r"\bloss\s+([0-9.]+)%", qdisc)
        observed_loss = float(loss_match.group(1)) if loss_match else math.nan
        if not math.isclose(observed_loss, float(loss), abs_tol=0.01):
            raise RuntimeError(f"Loss postcondition failed for {logical_name(node)}:{interface}")


def _restore_tc_interface(node: object, interface: str, edge_id: str) -> None:
    """Reapply the configured topology attributes of a link to one of its interfaces."""
    _configure_tc_interface(node, interface, _edge_attrs(edge_id))


def _sample_link_degradation(rng: random.Random) -> dict[str, float | str]:
    """Sample a severity level and its delay, jitter, and loss values."""
    profile = rng.choice(LINK_DEGRADATION_SEVERITY_PROFILES)
    return {
        "severity": profile.name,
        "delay_ms": rng.uniform(*profile.delay_ms),
        "jitter_ms": rng.uniform(*profile.jitter_ms),
        "loss_pct": rng.uniform(*profile.loss_pct),
    }


def _sample_link_flap_events(
    rng: random.Random, duration_sec: int
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    """Sample the flap count and down and up durations, shrinking the schedule to fit the fault window."""
    event_count = rng.randint(*LINK_FLAP_COUNT_RANGE)
    down_durations = [rng.uniform(*LINK_FLAP_DOWN_DURATION_RANGE_SEC) for _ in range(event_count)]
    up_gaps = [rng.uniform(*LINK_FLAP_UP_GAP_RANGE_SEC) for _ in range(max(0, event_count - 1))]
    budget = max(0.0, float(duration_sec))

    while event_count > 1 and sum(down_durations) + sum(up_gaps) > budget:
        event_count -= 1
        down_durations = down_durations[:event_count]
        up_gaps = up_gaps[: max(0, event_count - 1)]

    if sum(down_durations) + sum(up_gaps) > budget:
        down_durations = [min(down_durations[0], budget)]
        up_gaps = []

    events: list[dict[str, Any]] = []
    offset = 0.0
    for index, down_duration in enumerate(down_durations):
        flap_index = index + 1
        events.append(
            {"at_offset": round(offset, 4), "action": "link_down", "flap_index": flap_index}
        )
        offset += float(down_duration)
        events.append(
            {"at_offset": round(offset, 4), "action": "link_up", "flap_index": flap_index}
        )
        if index < len(up_gaps):
            offset += float(up_gaps[index])
    events.append(
        {
            "at_offset": round(max(offset, budget), 4),
            "action": "complete",
            "flap_index": len(down_durations),
        }
    )
    return (
        events,
        [round(value, 4) for value in down_durations],
        [round(value, 4) for value in up_gaps],
    )


def _phase_durations(duration_sec: int, ratios: DrainPhaseRatios) -> DrainPhaseDurations:
    """Convert drain phase fractions into seconds of the fault window."""
    return DrainPhaseDurations(
        ramp_down=float(duration_sec) * ratios.ramp_down,
        link_down=float(duration_sec) * ratios.link_down,
        hold_down=float(duration_sec) * ratios.hold_down,
        ramp_up=float(duration_sec) * ratios.ramp_up,
    )


def _append_phase(state: SimulatorState, phase_name: str, phase_state: str) -> None:
    """Record a drain phase boundary in the metadata without rewriting the file."""
    state.append_phase_timestamp(
        phase_name=phase_name,
        phase_state=phase_state,
        timestamp=utc_timestamp(),
        fault_category=FAULT_CATEGORY_DRAIN,
        flush=False,
    )


def _record_phase_count(state: SimulatorState) -> None:
    """Store the number of recorded phase boundaries in the metadata."""
    phase_entries = list(state.metadata.get("phase_timestamps", []))
    state.update_metadata(phase_count=len(phase_entries))


def _schedule_event(
    fault: dict[str, Any], at_offset: float, phase: str, action: str, **extra: Any
) -> None:
    """Append a timed action to the fault's event list."""
    events = fault.setdefault("events", [])
    payload: dict[str, Any] = {"at_offset": at_offset, "phase": phase, "action": action}
    payload.update(extra)
    events.append(payload)


def _build_events(fault: dict[str, Any], cfg: DrainConfig, baseline_rate_mbps: float) -> None:
    """Schedule the rate steps down, interfaces down, hold, interfaces up, and rate steps up of a drain."""
    durations = _phase_durations(cfg.duration_sec, cfg.phase_ratios)
    step_count = max(1, int(cfg.ramp_steps))

    ramp_down_start = 0.0
    link_down_start = ramp_down_start + durations.ramp_down
    hold_down_start = link_down_start + durations.link_down
    link_up_start = hold_down_start + durations.hold_down
    ramp_up_start = link_up_start
    clear_start = ramp_up_start + durations.ramp_up

    for idx in range(1, step_count + 1):
        frac = idx / step_count
        rate = baseline_rate_mbps - (baseline_rate_mbps - RAMP_MIN_RATE_MBPS) * frac
        at = ramp_down_start + durations.ramp_down * frac
        _schedule_event(
            fault, at, "ramp_down", "rate", rate_mbps=max(RAMP_MIN_RATE_MBPS, rate), step=idx
        )

    _schedule_event(fault, link_down_start, "link_down", "link_down")
    _schedule_event(fault, hold_down_start, "hold_down", "hold_down")
    _schedule_event(fault, link_up_start, "link_up", "link_up")

    for idx in range(1, step_count + 1):
        frac = idx / step_count
        rate = RAMP_MIN_RATE_MBPS + (baseline_rate_mbps - RAMP_MIN_RATE_MBPS) * frac
        at = ramp_up_start + durations.ramp_up * frac
        _schedule_event(
            fault, at, "ramp_up", "rate", rate_mbps=max(RAMP_MIN_RATE_MBPS, rate), step=idx
        )

    _schedule_event(fault, clear_start, "ramp_up", "clear_qdisc")
    _schedule_event(fault, clear_start, "done", "complete")


def _apply_rate_step(
    state: SimulatorState, fault: dict[str, Any], rate_mbps: float, phase: str, step: int
) -> None:
    """Shape every endpoint of the drained router to one rate and log each change."""
    for endpoint in list(fault.get("endpoints", [])):
        node = endpoint["node"]
        intf = endpoint["interface"]
        _run_tbf(node, intf, rate_mbps)
        _log_fault_action(
            state,
            action="drain_rate_step",
            target=f"{logical_name(node)}:{logical_interface_name(node, intf)}",
            parameters=f"phase={phase} step={step} rate_mbps={rate_mbps:.3f}",
            fault_type=FAULT_CATEGORY_DRAIN,
            category=FAULT_CATEGORY_DRAIN,
        )


def _apply_link_transition(
    state: SimulatorState, fault: dict[str, Any], up: bool, phase: str
) -> None:
    """Bring every endpoint of the drained router down or up and log each change."""
    action = "set_link_up" if up else "set_link_down"
    for endpoint in list(fault.get("endpoints", [])):
        node = endpoint["node"]
        intf = endpoint["interface"]
        _set_link_state(node, intf, up=up)
        _log_fault_action(
            state,
            action=action,
            target=f"{logical_name(node)}:{logical_interface_name(node, intf)}",
            parameters=f"phase={phase}",
            fault_type=FAULT_CATEGORY_DRAIN if not up else "recovery",
            category=FAULT_CATEGORY_DRAIN,
        )


def _restore_drain_links(state: SimulatorState, fault: dict[str, Any]) -> None:
    """Restore the configured attributes on every drained endpoint once."""
    if bool(fault.get("tc_restored", False)):
        return
    for endpoint in list(fault.get("endpoints", [])):
        node = endpoint["node"]
        intf = endpoint["interface"]
        edge_id = str(endpoint.get("edge_id", ""))
        if not edge_id:
            raise ValueError(
                f"Missing original edge profile for drain endpoint={logical_name(node)}:{intf}"
            )
        _restore_tc_interface(node, intf, edge_id)
        _log_fault_action(
            state,
            action="restore_link_tc",
            target=f"{logical_name(node)}:{logical_interface_name(node, intf)}",
            parameters=f"restore_profile={edge_id}",
            fault_type="recovery",
            category=FAULT_CATEGORY_DRAIN,
        )
    fault["tc_restored"] = True


def _execute_event(
    state: SimulatorState,
    fault: dict[str, Any],
    event: dict[str, Any],
    *,
    scheduled_mono: float,
) -> None:
    """Apply one due drain event, then record its timing and phase boundaries."""
    phase = str(event["phase"])
    action = str(event["action"])
    _append_phase(state, phase_name=phase, phase_state="start")
    if action == "rate":
        _apply_rate_step(
            state, fault, float(event["rate_mbps"]), phase=phase, step=int(event.get("step", 0))
        )
    elif action == "link_down":
        _apply_link_transition(state, fault, up=False, phase=phase)
    elif action == "hold_down":
        _log_fault_action(
            state,
            action="hold_link_down",
            target=",".join(list(fault.get("edge_ids", []))),
            parameters=f"phase={phase}",
            fault_type=FAULT_CATEGORY_DRAIN,
            category=FAULT_CATEGORY_DRAIN,
        )
    elif action == "link_up":
        _apply_link_transition(state, fault, up=True, phase=phase)
    elif action == "clear_qdisc":
        _restore_drain_links(state, fault)
    elif action == "complete":
        fault["completed"] = True
    else:
        raise ValueError(f"Unsupported staged-drain action={action}")
    state.record_fault_timing(
        event=f"{phase}:{action}",
        fault_category=FAULT_CATEGORY_DRAIN,
        target=str(fault.get("root_cause_id", "")),
        scheduled_mono=scheduled_mono,
        actual_mono=time.monotonic(),
    )
    _append_phase(state, phase_name=phase, phase_state="end")
    _record_phase_count(state)


def _execute_link_flap_event(
    state: SimulatorState,
    fault: dict[str, Any],
    event: dict[str, Any],
    *,
    scheduled_mono: float,
) -> None:
    """Apply one due link flap transition or completion, then record its timing."""
    action = str(event["action"])
    flap_index = int(event.get("flap_index", 0) or 0)
    phase = f"flap_{flap_index}_{'down' if action == 'link_down' else 'up'}"
    if action == "complete":
        _link_flap_cleanup(state, fault)
        fault["completed"] = True
        state.record_fault_timing(
            event=f"{phase}:{action}",
            fault_category=FAULT_CATEGORY_LINK_FLAP,
            target=str(fault.get("root_cause_id", "")),
            scheduled_mono=scheduled_mono,
            actual_mono=time.monotonic(),
        )
        return
    if action not in {"link_down", "link_up"}:
        raise ValueError(f"Unsupported link-flap action={action}")

    up = action == "link_up"
    _append_phase_for_category(
        state, phase_name=phase, phase_state="start", fault_category=FAULT_CATEGORY_LINK_FLAP
    )
    for endpoint in list(fault.get("endpoints", [])):
        node = endpoint["node"]
        intf = endpoint["interface"]
        _set_link_state(node, intf, up=up)
        _log_fault_action(
            state,
            action="set_link_up" if up else "set_link_down",
            target=f"{logical_name(node)}:{logical_interface_name(node, intf)}",
            parameters=f"phase={phase} flap_index={flap_index}",
            fault_type="recovery" if up else FAULT_CATEGORY_LINK_FLAP,
            category=FAULT_CATEGORY_LINK_FLAP,
        )
    state.record_fault_timing(
        event=f"{phase}:{action}",
        fault_category=FAULT_CATEGORY_LINK_FLAP,
        target=str(fault.get("root_cause_id", "")),
        scheduled_mono=scheduled_mono,
        actual_mono=time.monotonic(),
    )
    _append_phase_for_category(
        state, phase_name=phase, phase_state="end", fault_category=FAULT_CATEGORY_LINK_FLAP
    )
    _record_phase_count(state)


def _restore_endpoint_links(fault: dict[str, Any]) -> None:
    """Set every endpoint interface of a fault up again."""
    for endpoint in list(fault.get("endpoints", [])):
        node = endpoint["node"]
        intf = endpoint["interface"]
        _set_link_state(node, intf, up=True)


def _drain_cleanup(state: SimulatorState, fault: dict[str, Any]) -> None:
    """Bring the drained endpoints up and restore their configured attributes."""
    _restore_endpoint_links(fault)
    _restore_drain_links(state, fault)


def _append_phase_for_category(
    state: SimulatorState, phase_name: str, phase_state: str, fault_category: str
) -> None:
    """Record a phase boundary of the given fault category in the metadata without rewriting the file."""
    state.append_phase_timestamp(
        phase_name=phase_name,
        phase_state=phase_state,
        timestamp=utc_timestamp(),
        fault_category=fault_category,
        flush=False,
    )


def _fiber_cut_cleanup(state: SimulatorState, fault: dict[str, Any]) -> None:
    """Bring both endpoints of the cut link up again."""
    _restore_endpoint_links(fault)


def _link_flap_cleanup(state: SimulatorState, fault: dict[str, Any]) -> None:
    """Bring both endpoints of the flapping link up again."""
    _restore_endpoint_links(fault)


def _link_degradation_cleanup(state: SimulatorState, fault: dict[str, Any]) -> None:
    """Restore the configured attributes on both endpoints of the degraded link."""
    edge_id = str(fault.get("root_cause_id", ""))
    for endpoint in list(fault.get("endpoints", [])):
        node = endpoint["node"]
        intf = endpoint["interface"]
        _restore_tc_interface(node, intf, edge_id)


def _inject_fiber_cut_fault(
    net: Mininet,
    state: SimulatorState,
    fault_target: str,
    duration_sec: int,
    schedule_origin_mono: float,
) -> None:
    """Bring both interfaces of the target link down and schedule their recovery at the end of the fault window."""
    normalized = canonical_edge_id(*_link_nodes(fault_target))
    if normalized not in EDGE_FAULT_TARGETS:
        raise ValueError(f"Unsupported fiber cut target={fault_target}")

    src, dst = _link_nodes(normalized)
    left = logical_node(net, src)
    right = logical_node(net, dst)
    left_intf, right_intf = _get_link_interfaces(left, right)
    endpoints = (
        {"node": left, "interface": left_intf},
        {"node": right, "interface": right_intf},
    )

    for endpoint in endpoints:
        _set_link_state(endpoint["node"], endpoint["interface"], up=False)
        _log_fault_action(
            state,
            action="set_link_down",
            target=f"{logical_name(endpoint['node'])}:{logical_interface_name(endpoint['node'], endpoint['interface'])}",
            parameters="phase=link_down",
            fault_type="fiber_cut",
            category=FAULT_CATEGORY_FIBER_CUT,
        )

    state.update_metadata(
        root_cause_kind="edge",
        root_cause_id=normalized,
        fault_target=normalized,
        fault_category=FAULT_CATEGORY_FIBER_CUT,
        fault_schedule_mode=FAULT_SCHEDULE_INSTANT_LINK_DOWN,
        fault_start_ts=utc_timestamp(),
        fault_end_ts="",
        phase_timestamps=[],
        phase_count=0,
        target_link_role=edge_role(normalized),
        fiber_cut_target_edge=normalized,
        fiber_cut_duration_sec=duration_sec,
    )
    _append_phase_for_category(
        state, phase_name="link_down", phase_state="start", fault_category=FAULT_CATEGORY_FIBER_CUT
    )
    _append_phase_for_category(
        state, phase_name="link_down", phase_state="end", fault_category=FAULT_CATEGORY_FIBER_CUT
    )
    _record_phase_count(state)
    state.active_faults.append(
        {
            "description": f"fiber_cut:{normalized}",
            "fault_category": FAULT_CATEGORY_FIBER_CUT,
            "root_cause_kind": "edge",
            "root_cause_id": normalized,
            "edge_ids": [normalized],
            "endpoints": list(endpoints),
            "started_at": schedule_origin_mono,
            "recover_at": schedule_origin_mono + max(0, int(duration_sec)),
        }
    )


def _inject_link_flap_fault(
    net: Mininet,
    state: SimulatorState,
    fault_target: str,
    duration_sec: int,
    rng: random.Random,
    schedule_origin_mono: float,
) -> None:
    """Sample a flap schedule for the target link and register it as the active fault."""
    normalized = canonical_edge_id(*_link_nodes(fault_target))
    if normalized not in EDGE_FAULT_TARGETS:
        raise ValueError(f"Unsupported link flap target={fault_target}")

    src, dst = _link_nodes(normalized)
    left = logical_node(net, src)
    right = logical_node(net, dst)
    left_intf, right_intf = _get_link_interfaces(left, right)
    endpoints = (
        {"node": left, "interface": left_intf},
        {"node": right, "interface": right_intf},
    )
    events, down_durations, up_gaps = _sample_link_flap_events(rng, duration_sec)
    state.update_metadata(
        root_cause_kind="edge",
        root_cause_id=normalized,
        fault_target=normalized,
        fault_category=FAULT_CATEGORY_LINK_FLAP,
        fault_schedule_mode=FAULT_SCHEDULE_BURSTY_LINK_FLAP,
        fault_start_ts=utc_timestamp(),
        fault_end_ts="",
        phase_timestamps=[],
        phase_count=0,
        target_link_role=edge_role(normalized),
        link_flap_target_edge=normalized,
        link_flap_event_count=len(down_durations),
        link_flap_down_durations_sec=down_durations,
        link_flap_up_gaps_sec=up_gaps,
    )
    state.active_faults.append(
        {
            "description": f"link_flap:{normalized}",
            "fault_category": FAULT_CATEGORY_LINK_FLAP,
            "root_cause_kind": "edge",
            "root_cause_id": normalized,
            "edge_ids": [normalized],
            "endpoints": list(endpoints),
            "started_at": schedule_origin_mono,
            "events": events,
            "completed": False,
        }
    )


def _inject_link_degradation_fault(
    net: Mininet,
    state: SimulatorState,
    fault_target: str,
    duration_sec: int,
    rng: random.Random,
    schedule_origin_mono: float,
) -> None:
    """Add sampled delay, jitter, and loss to both interfaces of the target link and schedule recovery."""
    normalized = canonical_edge_id(*_link_nodes(fault_target))
    if normalized not in EDGE_FAULT_TARGETS:
        raise ValueError(f"Unsupported link degradation target={fault_target}")

    src, dst = _link_nodes(normalized)
    left = logical_node(net, src)
    right = logical_node(net, dst)
    left_intf, right_intf = _get_link_interfaces(left, right)
    endpoints = (
        {"node": left, "interface": left_intf},
        {"node": right, "interface": right_intf},
    )
    sampled = _sample_link_degradation(rng)
    base_attrs = _edge_attrs(normalized)
    degraded_attrs = dict(base_attrs)
    degraded_attrs["delay"] = (
        f"{_delay_ms(base_attrs.get('delay', 0.0)) + float(sampled['delay_ms']):.3f}ms"
    )
    degraded_attrs["jitter"] = f"{float(sampled['jitter_ms']):.3f}ms"
    degraded_attrs["loss"] = float(base_attrs.get("loss", 0.0) or 0.0) + float(sampled["loss_pct"])

    params = (
        f"severity={sampled['severity']} "
        f"delay_ms={float(sampled['delay_ms']):.3f} "
        f"jitter_ms={float(sampled['jitter_ms']):.3f} "
        f"loss_pct={float(sampled['loss_pct']):.3f}"
    )
    for endpoint in endpoints:
        _configure_tc_interface(endpoint["node"], endpoint["interface"], degraded_attrs)
        _log_fault_action(
            state,
            action="apply_link_degradation",
            target=f"{logical_name(endpoint['node'])}:{logical_interface_name(endpoint['node'], endpoint['interface'])}",
            parameters=params,
            fault_type=FAULT_CATEGORY_LINK_DEGRADATION,
            category=FAULT_CATEGORY_LINK_DEGRADATION,
        )

    state.update_metadata(
        root_cause_kind="edge",
        root_cause_id=normalized,
        fault_target=normalized,
        fault_category=FAULT_CATEGORY_LINK_DEGRADATION,
        fault_schedule_mode=FAULT_SCHEDULE_NETEM_LINK_DEGRADATION,
        fault_start_ts=utc_timestamp(),
        fault_end_ts="",
        phase_timestamps=[],
        phase_count=0,
        target_link_role=edge_role(normalized),
        link_degradation_target_edge=normalized,
        link_degradation_severity=sampled["severity"],
        link_degradation_delay_ms=round(float(sampled["delay_ms"]), 4),
        link_degradation_jitter_ms=round(float(sampled["jitter_ms"]), 4),
        link_degradation_loss_pct=round(float(sampled["loss_pct"]), 4),
    )
    _append_phase_for_category(
        state,
        phase_name="degraded",
        phase_state="start",
        fault_category=FAULT_CATEGORY_LINK_DEGRADATION,
    )
    _append_phase_for_category(
        state,
        phase_name="degraded",
        phase_state="end",
        fault_category=FAULT_CATEGORY_LINK_DEGRADATION,
    )
    _record_phase_count(state)
    state.active_faults.append(
        {
            "description": f"link_degradation:{normalized}",
            "fault_category": FAULT_CATEGORY_LINK_DEGRADATION,
            "root_cause_kind": "edge",
            "root_cause_id": normalized,
            "edge_ids": [normalized],
            "endpoints": list(endpoints),
            "started_at": schedule_origin_mono,
            "recover_at": schedule_origin_mono + max(0, int(duration_sec)),
        }
    )


def inject_requested_fault(
    net: Mininet,
    state: SimulatorState,
    root_cause_kind: str,
    fault_target: str,
    duration_sec: int,
    ramp_steps: int,
    phase_ratios: DrainPhaseRatios,
    fault_category: str = "",
    rng: random.Random | None = None,
    scheduled_at_mono: float | None = None,
    now_mono: float | None = None,
) -> None:
    """Inject the requested fault category on its target unless a fault is already active."""
    if state.active_faults:
        return
    injected_at_mono = time.monotonic() if now_mono is None else float(now_mono)
    scheduled_mono = injected_at_mono if scheduled_at_mono is None else float(scheduled_at_mono)
    if root_cause_kind == "edge":
        if fault_category == FAULT_CATEGORY_LINK_FLAP:
            _inject_link_flap_fault(
                net,
                state,
                fault_target=fault_target,
                duration_sec=duration_sec,
                rng=rng or random.Random(),
                schedule_origin_mono=scheduled_mono,
            )
            state.record_fault_timing(
                event="fault_injected",
                fault_category=FAULT_CATEGORY_LINK_FLAP,
                target=fault_target,
                scheduled_mono=scheduled_mono,
                actual_mono=time.monotonic(),
            )
            return
        if fault_category == FAULT_CATEGORY_LINK_DEGRADATION:
            _inject_link_degradation_fault(
                net,
                state,
                fault_target=fault_target,
                duration_sec=duration_sec,
                rng=rng or random.Random(),
                schedule_origin_mono=scheduled_mono,
            )
            state.record_fault_timing(
                event="fault_injected",
                fault_category=FAULT_CATEGORY_LINK_DEGRADATION,
                target=fault_target,
                scheduled_mono=scheduled_mono,
                actual_mono=time.monotonic(),
            )
            return
        _inject_fiber_cut_fault(
            net,
            state,
            fault_target=fault_target,
            duration_sec=duration_sec,
            schedule_origin_mono=scheduled_mono,
        )
        state.record_fault_timing(
            event="fault_injected",
            fault_category=FAULT_CATEGORY_FIBER_CUT,
            target=fault_target,
            scheduled_mono=scheduled_mono,
            actual_mono=time.monotonic(),
        )
        return
    if root_cause_kind != "node":
        raise ValueError("Supported fault scenarios require root_cause_kind=node|edge")
    if fault_target not in OBSERVED_NODE_FAULT_TARGETS:
        raise ValueError(f"Unsupported observed node fault target={fault_target}")

    edge_ids = _resolve_edge_targets_for_node(fault_target)
    endpoints: list[dict[str, Any]] = []
    baseline_candidates: list[float] = []
    for edge_id in edge_ids:
        src, dst = _link_nodes(edge_id)
        left = logical_node(net, src)
        right = logical_node(net, dst)
        left_intf, right_intf = _get_link_interfaces(left, right)
        endpoints.append({"node": left, "interface": left_intf, "edge_id": edge_id})
        endpoints.append({"node": right, "interface": right_intf, "edge_id": edge_id})
        baseline_candidates.append(_edge_bandwidth_mbps(edge_id))
    baseline_rate_mbps = min(baseline_candidates) if baseline_candidates else RAMP_MIN_RATE_MBPS

    cfg = DrainConfig(duration_sec=duration_sec, ramp_steps=ramp_steps, phase_ratios=phase_ratios)
    fault: dict[str, Any] = {
        "description": f"drain_staged:{fault_target}",
        "fault_category": FAULT_CATEGORY_DRAIN,
        "root_cause_kind": "node",
        "root_cause_id": fault_target,
        "edge_ids": edge_ids,
        "endpoints": endpoints,
        "started_at": scheduled_mono,
        "injected_at": injected_at_mono,
        "completed": False,
        "events": [],
    }
    _build_events(fault, cfg, baseline_rate_mbps=baseline_rate_mbps)

    state.update_metadata(
        root_cause_kind="node",
        root_cause_id=fault_target,
        fault_target=fault_target,
        fault_category=FAULT_CATEGORY_DRAIN,
        fault_schedule_mode=FAULT_SCHEDULE_STAGED_DRAIN,
        fault_start_ts=utc_timestamp(),
        fault_end_ts="",
        phase_timestamps=[],
        phase_count=0,
        drain_target_edge=",".join(edge_ids),
        drain_ramp_steps=cfg.ramp_steps,
        drain_phase_ratios={
            "ramp_down": cfg.phase_ratios.ramp_down,
            "link_down": cfg.phase_ratios.link_down,
            "hold_down": cfg.phase_ratios.hold_down,
            "ramp_up": cfg.phase_ratios.ramp_up,
        },
        drain_phase_durations_sec={
            "ramp_down": round(_phase_durations(cfg.duration_sec, cfg.phase_ratios).ramp_down, 3),
            "link_down": round(_phase_durations(cfg.duration_sec, cfg.phase_ratios).link_down, 3),
            "hold_down": round(_phase_durations(cfg.duration_sec, cfg.phase_ratios).hold_down, 3),
            "ramp_up": round(_phase_durations(cfg.duration_sec, cfg.phase_ratios).ramp_up, 3),
        },
    )
    state.active_faults.append(fault)
    state.record_fault_timing(
        event="fault_injected",
        fault_category=FAULT_CATEGORY_DRAIN,
        target=fault_target,
        scheduled_mono=scheduled_mono,
        actual_mono=time.monotonic(),
    )


def next_fault_transition_mono(state: SimulatorState) -> float | None:
    """Return the earliest pending transition deadline across active faults."""
    deadlines: list[float] = []
    for fault in state.active_faults:
        category = str(fault.get("fault_category", ""))
        if category in {FAULT_CATEGORY_DRAIN, FAULT_CATEGORY_LINK_FLAP}:
            started_at = float(fault["started_at"])
            deadlines.extend(
                started_at + float(event["at_offset"]) for event in list(fault.get("events", []))
            )
            continue
        if category in {FAULT_CATEGORY_FIBER_CUT, FAULT_CATEGORY_LINK_DEGRADATION}:
            recover_at = fault.get("recover_at")
            if recover_at is not None:
                deadlines.append(float(recover_at))
    return min(deadlines) if deadlines else None


def _log_fault_cleared(state: SimulatorState, fault: dict[str, Any], fault_category: str) -> None:
    """Log automatic recovery of a fault and stamp the fault end time once."""
    state.fault_logger.log(
        action="fault_cleared",
        target=str(fault["description"]),
        parameters="automatic_recovery=true",
        fault_type="recovery",
        fault_category=fault_category,
    )
    if state.metadata.get("fault_start_ts") and not state.metadata.get("fault_end_ts"):
        state.update_metadata(fault_end_ts=utc_timestamp())


def _recover_edge_fault(
    state: SimulatorState,
    fault: dict[str, Any],
    *,
    fault_category: str,
    recover_at: float,
    restore_endpoint: Callable[[Any, str], None],
) -> None:
    """Shared timed-recovery scaffold for one-shot link faults."""
    _append_phase_for_category(
        state, phase_name="recovery", phase_state="start", fault_category=fault_category
    )
    for endpoint in list(fault.get("endpoints", [])):
        restore_endpoint(endpoint["node"], endpoint["interface"])
    state.record_fault_timing(
        event="recovery",
        fault_category=fault_category,
        target=str(fault.get("root_cause_id", "")),
        scheduled_mono=recover_at,
        actual_mono=time.monotonic(),
    )
    _append_phase_for_category(
        state, phase_name="recovery", phase_state="end", fault_category=fault_category
    )
    _record_phase_count(state)
    _log_fault_cleared(state, fault, fault_category)


def maintain_active_faults(state: SimulatorState, now_mono: float | None = None) -> None:
    """Apply every due fault event and recovery, dropping faults that have completed."""
    if not state.active_faults:
        return
    now = time.monotonic() if now_mono is None else float(now_mono)
    remaining: list[dict[str, Any]] = []
    for fault in state.active_faults:
        if fault.get("fault_category") == FAULT_CATEGORY_LINK_FLAP:
            started_at = float(fault["started_at"])
            events = sorted(
                list(fault.get("events", [])), key=lambda entry: float(entry["at_offset"])
            )
            next_events: list[dict[str, Any]] = []
            for event in events:
                due = started_at + float(event["at_offset"])
                if now >= due:
                    _execute_link_flap_event(
                        state,
                        fault,
                        event,
                        scheduled_mono=due,
                    )
                else:
                    next_events.append(event)
            fault["events"] = next_events

            if fault.get("completed") and not next_events:
                _link_flap_cleanup(state, fault)
                _log_fault_cleared(state, fault, FAULT_CATEGORY_LINK_FLAP)
                continue

            remaining.append(fault)
            continue
        if fault.get("fault_category") == FAULT_CATEGORY_LINK_DEGRADATION:
            recover_at = float(fault.get("recover_at", 0.0))
            if now >= recover_at:
                edge_id = str(fault.get("root_cause_id", ""))

                def _restore_degraded(node: Any, intf: str, edge_id: str = edge_id) -> None:
                    """Restore one degraded endpoint and log the recovery."""
                    _restore_tc_interface(node, intf, edge_id)
                    _log_fault_action(
                        state,
                        action="restore_link_tc",
                        target=f"{logical_name(node)}:{logical_interface_name(node, intf)}",
                        parameters="phase=recovery",
                        fault_type="recovery",
                        category=FAULT_CATEGORY_LINK_DEGRADATION,
                    )

                _recover_edge_fault(
                    state,
                    fault,
                    fault_category=FAULT_CATEGORY_LINK_DEGRADATION,
                    recover_at=recover_at,
                    restore_endpoint=_restore_degraded,
                )
                continue

            remaining.append(fault)
            continue
        if fault.get("fault_category") == FAULT_CATEGORY_FIBER_CUT:
            recover_at = float(fault.get("recover_at", 0.0))
            if now >= recover_at:

                def _restore_cut(node: Any, intf: str) -> None:
                    """Bring one cut endpoint up and log the recovery."""
                    _set_link_state(node, intf, up=True)
                    _log_fault_action(
                        state,
                        action="set_link_up",
                        target=f"{logical_name(node)}:{logical_interface_name(node, intf)}",
                        parameters="phase=recovery",
                        fault_type="recovery",
                        category=FAULT_CATEGORY_FIBER_CUT,
                    )

                _recover_edge_fault(
                    state,
                    fault,
                    fault_category=FAULT_CATEGORY_FIBER_CUT,
                    recover_at=recover_at,
                    restore_endpoint=_restore_cut,
                )
                continue

            remaining.append(fault)
            continue
        if fault.get("fault_category") != FAULT_CATEGORY_DRAIN:
            continue
        started_at = float(fault["started_at"])
        events = sorted(list(fault.get("events", [])), key=lambda entry: float(entry["at_offset"]))
        next_events: list[dict[str, Any]] = []
        for event in events:
            due = started_at + float(event["at_offset"])
            if now >= due:
                _execute_event(
                    state,
                    fault,
                    event,
                    scheduled_mono=due,
                )
            else:
                next_events.append(event)
        fault["events"] = next_events

        if fault.get("completed") and not next_events:
            _drain_cleanup(state, fault)
            _log_fault_cleared(state, fault, FAULT_CATEGORY_DRAIN)
            continue

        remaining.append(fault)
    state.active_faults = remaining


def cleanup_active_faults(state: SimulatorState) -> int:
    """Force recovery of every active fault, log each cleanup, and return the count."""
    cleaned = 0
    for fault in list(state.active_faults):
        if fault.get("fault_category") == FAULT_CATEGORY_DRAIN:
            _drain_cleanup(state, fault)
            state.fault_logger.log(
                action="fault_cleanup",
                target=str(fault.get("description", "")),
                parameters="forced_cleanup=true",
                fault_type="recovery",
                fault_category=FAULT_CATEGORY_DRAIN,
            )
            cleaned += 1
            continue
        if fault.get("fault_category") == FAULT_CATEGORY_FIBER_CUT:
            _fiber_cut_cleanup(state, fault)
            state.fault_logger.log(
                action="fault_cleanup",
                target=str(fault.get("description", "")),
                parameters="forced_cleanup=true",
                fault_type="recovery",
                fault_category=FAULT_CATEGORY_FIBER_CUT,
            )
            cleaned += 1
            continue
        if fault.get("fault_category") == FAULT_CATEGORY_LINK_FLAP:
            _link_flap_cleanup(state, fault)
            state.fault_logger.log(
                action="fault_cleanup",
                target=str(fault.get("description", "")),
                parameters="forced_cleanup=true",
                fault_type="recovery",
                fault_category=FAULT_CATEGORY_LINK_FLAP,
            )
            cleaned += 1
            continue
        if fault.get("fault_category") == FAULT_CATEGORY_LINK_DEGRADATION:
            _link_degradation_cleanup(state, fault)
            state.fault_logger.log(
                action="fault_cleanup",
                target=str(fault.get("description", "")),
                parameters="forced_cleanup=true",
                fault_type="recovery",
                fault_category=FAULT_CATEGORY_LINK_DEGRADATION,
            )
            cleaned += 1
    state.active_faults.clear()
    return cleaned
