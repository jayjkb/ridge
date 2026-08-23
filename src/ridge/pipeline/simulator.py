"""Mininet-based synthetic data generator for RIDGE RCA."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from mininet.log import info, setLogLevel

from ridge.common.contracts import (
    STAGE1_PROBE_SCHEDULING_POLICY,
    STAGE1_SCHEDULING_POLICY,
)
from ridge.common.validation import (
    require_duration_after_window,
    require_fault_window_fits,
    require_non_negative,
    require_positive,
)
from ridge.pipeline.arg_validation import (
    require_drain_phase_ratios,
    require_min_max_bounds,
    require_offset_before_duration,
    resolved_burst_mean_gap_sec,
)
from ridge.pipeline.mininet_runtime import (
    cleanup_instance_runtime,
    cleanup_mininet_state,
    frr_runtime_dir,
    prepare_empty_directory,
)
from ridge.sim.collectors import (
    AsyncPingProbeScheduler,
    collect_host_stats,
    collect_telemetry_batch,
)
from ridge.sim.common import (
    HOST_HEADERS,
    INTERFACE_HEADERS,
    NEIGHBOR_HEADERS,
    NODE_HEADERS,
    PING_HEADERS,
    QUEUE_HEADERS,
    ROUTE_HEADERS,
    utc_timestamp,
)
from ridge.sim.faults import (
    DRAIN_NODE_FAULT_TARGETS,
    EDGE_FAULT_TARGETS,
    FAULT_CATEGORY_DRAIN,
    FAULT_CATEGORY_FIBER_CUT,
    FAULT_CATEGORY_LINK_DEGRADATION,
    FAULT_CATEGORY_LINK_FLAP,
    FAULT_CATEGORY_NONE,
    FAULT_SCHEDULE_BURSTY_LINK_FLAP,
    FAULT_SCHEDULE_INSTANT_LINK_DOWN,
    FAULT_SCHEDULE_NETEM_LINK_DEGRADATION,
    FAULT_SCHEDULE_NONE,
    FAULT_SCHEDULE_STAGED_DRAIN,
    DrainPhaseRatios,
    cleanup_active_faults,
    inject_requested_fault,
    maintain_active_faults,
    next_fault_transition_mono,
)
from ridge.sim.scheduling import (
    FixedRateTelemetryScheduler,
    SnapshotSlot,
    telemetry_should_yield_to_control,
)
from ridge.sim.state import SimulatorState
from ridge.sim.topology import (
    build_network,
    configure_network,
    export_topology_csv,
    host_site,
    logical_name,
    logical_node,
    node_role,
    topology_spec,
)
from ridge.sim.traffic import (
    DITG_DATA_PORT_MIN,
    build_ditg_flows,
    ditg_runtime_failures,
    launch_ditg_plan,
    stop_ditg_traffic,
    validate_ditg_diagnostics,
)

# Packet-size mix uses canonical Ethernet benchmarking frame sizes from RFC 2544,
# weighted to mimic the common bimodal shape seen in passive traces: many small
# control/ACK packets plus near-MTU data packets, with medium sizes still present.
REALISTIC_PACKET_SIZE_WEIGHTS: tuple[tuple[int, float], ...] = (
    (64, 0.18),
    (128, 0.08),
    (256, 0.14),
    (512, 0.12),
    (1024, 0.10),
    (1280, 0.16),
    (1518, 0.22),
)

# Per-flow baseline throughput profile in Mbps for this topology.
REALISTIC_FLOW_RATE_Mbps_WEIGHTS: tuple[tuple[float, float], ...] = (
    (0.08, 0.18),
    (0.15, 0.26),
    (0.30, 0.24),
    (0.50, 0.20),
    (0.80, 0.12),
)

CONTROL_TICK_SEC = 0.05
# An atomic telemetry batch may block serialized Mininet commands for at most
# this certified budget. Reserving the final 0.5 s prevents such a batch from
# pushing an imminent fault transition beyond its lag limit.
TELEMETRY_BLOCKING_BUDGET_SEC = 1.0
FAULT_TRANSITION_LAG_BUDGET_SEC = 0.5
FAULT_TRANSITION_GUARD_SEC = TELEMETRY_BLOCKING_BUDGET_SEC - FAULT_TRANSITION_LAG_BUDGET_SEC
# Keep source masters alive through final collection and integrity checks.
BASELINE_TRAFFIC_MIN_TAIL_SEC = 10
DITG_ACTIVATION_GUARD_SEC = 5.0
TRAFFIC_SCHEDULE_POLICY = "precomputed_multiflow_native_on_off_v3"


def _baseline_traffic_duration_sec(
    warmup_sec: int, collection_duration_sec: int, telemetry_interval_sec: float
) -> int:
    """Return a D-ITG duration that covers warmup, collection, and shutdown tail."""
    tail_sec = max(BASELINE_TRAFFIC_MIN_TAIL_SEC, math.ceil(2 * telemetry_interval_sec))
    return max(1, int(warmup_sec) + int(collection_duration_sec) + tail_sec)


def _burst_overlay_schedule(
    *,
    collection_duration_sec: int,
    telemetry_interval_sec: float,
    mean_gap_sec: float,
) -> dict[str, int | float]:
    """Resolve a bounded native D-ITG ON/OFF overlay for the measured period."""
    interval_sec = float(telemetry_interval_sec)
    start_offset_sec = min(interval_sec, max(0.0, float(collection_duration_sec) - 1.0))
    on_duration_ms = max(1000, int(round(interval_sec * 1000.0)))
    off_duration_ms = max(
        100,
        int(round(max(0.1, float(mean_gap_sec) - interval_sec) * 1000.0)),
    )
    duration_ms = max(
        1000,
        int(round((float(collection_duration_sec) - start_offset_sec) * 1000.0)),
    )
    cycle_ms = on_duration_ms + off_duration_ms
    return {
        "collection_start_offset_sec": round(start_offset_sec, 6),
        "duration_ms": duration_ms,
        "on_duration_ms": on_duration_ms,
        "off_duration_ms": off_duration_ms,
        "cycle_sec": round(cycle_ms / 1000.0, 6),
        "planned_cycle_count": max(1, math.ceil(duration_ms / cycle_ms)),
    }


REPRESENTATIVE_CROSS_SITE_PAIRS = (
    ("h1_1", "h4_1"),
    ("h4_1", "h1_1"),
    ("h2_1", "h5_1"),
    ("h5_1", "h2_1"),
    ("h3_1", "h4_2"),
    ("h4_2", "h3_2"),
    ("h1_3", "h5_2"),
    ("h5_2", "h1_2"),
)


def _cross_site_directed_pairs(
    host_names: list[str], rng: random.Random | None = None
) -> list[tuple[str, str]]:
    """Return every directed host pair across the two sites, shuffled when a generator is given."""
    site_a = sorted([name for name in host_names if host_site(name) == "site_a"])
    site_b = sorted([name for name in host_names if host_site(name) == "site_b"])
    if not site_a or not site_b:
        return []
    pairs = [(src, dst) for src in site_a for dst in site_b] + [
        (src, dst) for src in site_b for dst in site_a
    ]
    if rng is not None:
        rng.shuffle(pairs)
    return pairs


def policy_pair_pool(host_names: list[str], rng: random.Random) -> list[tuple[str, str]]:
    """Return the four core cross-site probe pairs in interleaved order followed by the shuffled extra pairs."""
    available_hosts = set(host_names)
    core_pairs = [
        pair
        for pair in REPRESENTATIVE_CROSS_SITE_PAIRS[:4]
        if pair[0] in available_hosts and pair[1] in available_hosts
    ]
    extra_pairs = [
        pair
        for pair in REPRESENTATIVE_CROSS_SITE_PAIRS[4:]
        if pair[0] in available_hosts and pair[1] in available_hosts
    ]
    if not core_pairs and not extra_pairs:
        return []
    interleaved_core = core_pairs[:1] + core_pairs[2:3] + core_pairs[1:2] + core_pairs[3:4]
    if not extra_pairs:
        return interleaved_core
    shuffled_extras = list(extra_pairs)
    rng.shuffle(shuffled_extras)
    return interleaved_core + shuffled_extras


def traffic_flow_pair_pool(
    host_names: list[str], rng: random.Random | None = None
) -> list[tuple[str, str]]:
    """Return the directed cross-site host pairs available to background traffic flows."""
    return _cross_site_directed_pairs(host_names, rng)


def validate_requested_pair_budget(
    label: str, count_max: int, candidate_pairs: list[tuple[str, str]]
) -> None:
    """Raise ValueError when a requested maximum exceeds the available pair count."""
    if count_max > len(candidate_pairs):
        raise ValueError(
            f"{label} requested max={count_max} exceeds available candidate pair count={len(candidate_pairs)}"
        )


def topology_host_names() -> list[str]:
    """Return the sorted host names of the topology."""
    return sorted(str(name) for name in topology_spec()["hosts"] if str(name).startswith("h"))


def validate_traffic_and_probe_args(args: argparse.Namespace) -> None:
    """Validate the traffic/probe bounds and pair budgets shared with the generator."""
    require_min_max_bounds(
        "--traffic-flow-min", args.traffic_flow_min, "--traffic-flow-max", args.traffic_flow_max
    )
    require_min_max_bounds(
        "--ping-pair-min", args.ping_pair_min, "--ping-pair-max", args.ping_pair_max
    )
    require_positive("--probe-packets", args.probe_packets)
    require_positive("--probe-timeout-sec", args.probe_timeout_sec)
    require_positive("--probe-cadence-sec", args.probe_cadence_sec)
    host_names = topology_host_names()
    validate_requested_pair_budget(
        "--traffic-flow-max",
        args.traffic_flow_max,
        traffic_flow_pair_pool(host_names),
    )
    validate_requested_pair_budget(
        "--ping-pair-max",
        args.ping_pair_max,
        policy_pair_pool(host_names, random.Random(0)),
    )


def _select_ping_pairs(
    net: object,
    rng: random.Random,
    host_names: list[str],
    count_min: int,
    count_max: int,
) -> list[tuple[object, object]]:
    """Select the active probe pairs, taking the core pairs first for small counts and sampling otherwise."""
    candidate_pairs = policy_pair_pool(host_names, rng)
    if not candidate_pairs:
        return []
    selected_count = min(len(candidate_pairs), rng.randint(count_min, count_max))
    if selected_count >= len(candidate_pairs):
        selected_pairs = list(candidate_pairs)
    elif selected_count <= 4:
        selected_pairs = list(candidate_pairs[:selected_count])
    else:
        selected_pairs = rng.sample(list(candidate_pairs), selected_count)
    return [(logical_node(net, src), logical_node(net, dst)) for src, dst in selected_pairs]


def _select_host_pairs(
    net: object,
    rng: random.Random,
    pair_pool: list[tuple[str, str]],
    count_min: int,
    count_max: int,
) -> list[tuple[object, object]]:
    """Sample a random number of host pairs from the pool within the configured bounds."""
    selected_count = rng.randint(count_min, count_max)
    selected_pairs = rng.sample(list(pair_pool), selected_count)
    return [(logical_node(net, src), logical_node(net, dst)) for src, dst in selected_pairs]


def parse_args() -> argparse.Namespace:
    """Build the argument parser of the per-episode generator and parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Length of the measured period in seconds, after the warmup period.",
    )
    parser.add_argument("--interval", type=int, default=5, help="Collection interval in seconds.")
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where CSV telemetry and fault logs will be written.",
    )
    parser.add_argument(
        "--instance-id",
        default="",
        help="Short alphanumeric runtime prefix for concurrent Mininet instances.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for this episode."
    )
    parser.add_argument(
        "--fault-mode",
        choices=("scenario", "none"),
        default="scenario",
        help="Fault schedule for a single episode.",
    )
    parser.add_argument(
        "--root-cause-kind",
        choices=("node", "edge"),
        default="",
        help="Candidate type, node or edge, for a deterministic single-fault episode.",
    )
    parser.add_argument(
        "--fault-target",
        default="",
        help="Router name or canonical link identifier for a deterministic single-fault episode.",
    )
    parser.add_argument(
        "--fault-category",
        choices=(
            FAULT_CATEGORY_DRAIN,
            FAULT_CATEGORY_FIBER_CUT,
            FAULT_CATEGORY_LINK_DEGRADATION,
            FAULT_CATEGORY_LINK_FLAP,
        ),
        default="",
        help="Fault category for deterministic scenarios. Link scenarios default to fiber_cut when omitted.",
    )
    parser.add_argument(
        "--warmup-sec",
        type=int,
        default=15,
        help="Traffic warmup in seconds before telemetry collection starts.",
    )
    parser.add_argument(
        "--fault-duration-sec",
        type=int,
        default=60,
        help="Duration of a deterministic single fault.",
    )
    parser.add_argument(
        "--fault-start-offset-sec",
        type=int,
        default=0,
        help="Delay in seconds after telemetry collection starts before injecting deterministic faults.",
    )
    parser.add_argument(
        "--drain-ramp-steps",
        type=int,
        default=5,
        help="Number of steps for ramp-down and ramp-up stages.",
    )
    parser.add_argument(
        "--drain-phase-ratio-ramp-down",
        type=float,
        default=0.25,
        help="Relative duration ratio for ramp-down phase.",
    )
    parser.add_argument(
        "--drain-phase-ratio-link-down",
        type=float,
        default=0.15,
        help="Relative duration ratio for link-down transition phase.",
    )
    parser.add_argument(
        "--drain-phase-ratio-hold-down",
        type=float,
        default=0.20,
        help="Relative duration ratio for hold-down phase.",
    )
    parser.add_argument(
        "--drain-phase-ratio-ramp-up",
        type=float,
        default=0.40,
        help="Relative duration ratio for ramp-up phase.",
    )
    parser.add_argument(
        "--traffic-flow-min",
        type=int,
        default=4,
        help="Minimum baseline traffic flow count sampled from pool.",
    )
    parser.add_argument(
        "--traffic-flow-max",
        type=int,
        default=6,
        help="Maximum baseline traffic flow count sampled from pool.",
    )
    parser.add_argument(
        "--ping-pair-min",
        type=int,
        default=4,
        help="Minimum probe pair count sampled from the pool.",
    )
    parser.add_argument(
        "--ping-pair-max",
        type=int,
        default=6,
        help="Maximum probe pair count sampled from the pool.",
    )
    parser.add_argument(
        "--probe-packets",
        type=int,
        default=1,
        help="ICMP echo count per probe execution.",
    )
    parser.add_argument(
        "--probe-timeout-sec",
        type=float,
        default=1.0,
        help="Per-probe timeout/deadline in seconds.",
    )
    parser.add_argument(
        "--probe-cadence-sec",
        type=float,
        default=1.0,
        help="Target cadence in seconds between background probes for the same pair.",
    )
    parser.add_argument(
        "--skip-startup-cleanup",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--burst-mean-gap-sec",
        type=float,
        default=None,
        help="Mean gap in seconds between congestion bursts. Defaults to three collection intervals.",
    )
    return parser.parse_args()


def cleanup_mininet() -> None:
    """Remove stale FRRouting runtime and Mininet state, logging through Mininet."""
    cleanup_mininet_state(
        lambda message: info(f"*** {message}\n"),
        banner="Cleaning stale Mininet state with mn -c",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Check every episode argument, including the fault scenario, window, and drain phase settings."""
    require_positive("--duration", args.duration)
    require_positive("--interval", args.interval)
    if args.burst_mean_gap_sec is not None:
        require_positive("--burst-mean-gap-sec", args.burst_mean_gap_sec)
    require_non_negative("--warmup-sec", args.warmup_sec)
    require_non_negative("--fault-start-offset-sec", args.fault_start_offset_sec)
    require_positive("--fault-duration-sec", args.fault_duration_sec)
    require_positive("--drain-ramp-steps", args.drain_ramp_steps)
    validate_traffic_and_probe_args(args)

    explicit_fault = bool(args.root_cause_kind or args.fault_target)
    fault_category = str(getattr(args, "fault_category", "") or "")
    if args.fault_mode == "scenario" and not explicit_fault:
        raise ValueError("--fault-mode scenario requires --root-cause-kind and --fault-target")
    if bool(args.root_cause_kind) != bool(args.fault_target):
        raise ValueError("--root-cause-kind and --fault-target must be provided together")
    if fault_category and not explicit_fault:
        raise ValueError("--fault-category requires --root-cause-kind and --fault-target")
    if not explicit_fault:
        return
    if args.root_cause_kind == "node" and fault_category and fault_category != FAULT_CATEGORY_DRAIN:
        raise ValueError(
            "Node fault scenarios require --fault-category drain when a category is provided"
        )
    if args.root_cause_kind == "edge" and fault_category == FAULT_CATEGORY_DRAIN:
        raise ValueError("Edge fault scenarios require fiber_cut, link_degradation, or link_flap")
    require_duration_after_window(args.duration, args.fault_duration_sec)
    require_offset_before_duration(args.fault_start_offset_sec, args.duration)
    require_fault_window_fits(args.duration, args.fault_start_offset_sec, args.fault_duration_sec)
    if args.root_cause_kind == "node" and args.fault_target not in DRAIN_NODE_FAULT_TARGETS:
        raise ValueError(f"Unsupported observed node fault target={args.fault_target}")
    if args.root_cause_kind == "node":
        ratios = (
            float(args.drain_phase_ratio_ramp_down),
            float(args.drain_phase_ratio_link_down),
            float(args.drain_phase_ratio_hold_down),
            float(args.drain_phase_ratio_ramp_up),
        )
        require_drain_phase_ratios(ratios)
        for name, value in (
            ("ramp_down", args.drain_phase_ratio_ramp_down),
            ("link_down", args.drain_phase_ratio_link_down),
            ("hold_down", args.drain_phase_ratio_hold_down),
            ("ramp_up", args.drain_phase_ratio_ramp_up),
        ):
            if float(args.fault_duration_sec) * float(value) < 0.25:
                raise ValueError(
                    f"Drain phase {name} is too short; increase --fault-duration-sec or phase ratio"
                )
        if node_role(args.fault_target) != "spine":
            raise ValueError("Drain scenarios currently support spine switch targets only")
        return
    if args.root_cause_kind == "edge" and args.fault_target not in EDGE_FAULT_TARGETS:
        raise ValueError(f"Unsupported edge fault target={args.fault_target}")


def prepare_run_log_directory(path: Path) -> None:
    """Refuse to append an episode to an existing episode directory."""
    prepare_empty_directory(path, kind="Log")


def _weighted_choice(rng: random.Random, weighted_values: tuple[tuple[float, float], ...]) -> float:
    """Draw one value from a weighted tuple of value and weight pairs."""
    values = [item[0] for item in weighted_values]
    weights = [item[1] for item in weighted_values]
    return float(rng.choices(values, weights=weights, k=1)[0])


def _sample_packet_size_bytes(rng: random.Random) -> int:
    """Sample a packet size from the weighted Ethernet frame sizes."""
    return int(_weighted_choice(rng, REALISTIC_PACKET_SIZE_WEIGHTS))


def _sample_flow_rate_pps(rng: random.Random, packet_size_bytes: int) -> int:
    """Sample a base sending rate in Mbps and convert it to packets per second for the packet size."""
    mbps = _weighted_choice(rng, REALISTIC_FLOW_RATE_Mbps_WEIGHTS)
    bits_per_packet = max(1, packet_size_bytes * 8)
    pps = int(round((mbps * 1_000_000) / bits_per_packet))
    return max(10, pps)


def _scaled_overlay_rate_pps(
    base_rate_pps: int,
    base_packet_size: int,
    rate_multiplier: float,
    *,
    overlay_packet_size: int = 1024,
) -> int:
    """Scale an overlay by bitrate when its packet size differs from baseline."""
    if min(base_rate_pps, base_packet_size, overlay_packet_size) <= 0:
        raise ValueError("Traffic rates and packet sizes must be positive")
    if rate_multiplier <= 0:
        raise ValueError("Traffic rate multiplier must be positive")
    equivalent_pps = (float(base_rate_pps) * float(base_packet_size) * rate_multiplier) / float(
        overlay_packet_size
    )
    return max(10, int(round(equivalent_pps)))


def _finite_float(value: object) -> float | None:
    """Convert to float, returning None for missing, malformed, or non-finite values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _probe_rows_complete(rows: list[dict[str, object]], expected_count: int) -> bool:
    """Return whether every configured pair has an explicit probe result."""
    if len(rows) != expected_count:
        return False
    metric_fields = ("PacketLoss", "AvgRTT", "TimeoutFlag")
    return all(
        any(_finite_float(row.get(field)) is not None for field in metric_fields) for row in rows
    )


def _baseline_health_summary(
    loss_samples: list[float],
    rtt_samples: list[float],
    timeout_samples: list[float],
    missing_probe_samples: int,
) -> dict[str, object]:
    """Summarize only finite baseline probes and fail closed on no evidence."""
    losses = [value for value in loss_samples if math.isfinite(value)]
    rtts = sorted(value for value in rtt_samples if math.isfinite(value))
    timeouts = [value for value in timeout_samples if math.isfinite(value)]
    mean_loss = sum(losses) / len(losses) if losses else math.nan
    timeout_ratio = sum(timeouts) / len(timeouts) if timeouts else math.nan
    if rtts:
        index = max(0, min(len(rtts) - 1, math.ceil(0.95 * len(rtts)) - 1))
        p95_rtt = rtts[index]
    else:
        p95_rtt = math.nan
    enough_samples = bool(losses and rtts and timeouts) and missing_probe_samples == 0
    health_pass = bool(
        enough_samples and mean_loss <= 5.0 and timeout_ratio <= 0.1 and p95_rtt <= 250.0
    )
    return {
        "baseline_mean_packet_loss_pct": round(mean_loss, 4) if math.isfinite(mean_loss) else None,
        "baseline_timeout_ratio": round(timeout_ratio, 4) if math.isfinite(timeout_ratio) else None,
        "baseline_p95_rtt_ms": round(p95_rtt, 4) if math.isfinite(p95_rtt) else None,
        "baseline_loss_sample_count": len(losses),
        "baseline_rtt_sample_count": len(rtts),
        "baseline_timeout_sample_count": len(timeouts),
        "baseline_missing_probe_sample_count": int(missing_probe_samples),
        "baseline_health_pass": health_pass,
    }


def _traffic_launch_complete(result: dict[str, object]) -> bool:
    """Return whether every planned source and receiver process launched without failure."""
    planned_flows = int(result.get("planned_flow_count", 0) or 0)
    source_attempted = int(result.get("source_process_attempted_count", 0) or 0)
    source_launched = int(result.get("source_process_launched_count", 0) or 0)
    source_failed = int(result.get("source_process_failed_count", 0) or 0)
    receiver_attempted = int(result.get("receiver_process_attempted_count", 0) or 0)
    receiver_launched = int(result.get("receiver_process_launched_count", 0) or 0)
    receiver_failed = int(result.get("receiver_process_failed_count", 0) or 0)
    return bool(
        planned_flows > 0
        and source_attempted > 0
        and receiver_attempted > 0
        and source_launched == source_attempted
        and receiver_launched == receiver_attempted
        and source_failed == 0
        and receiver_failed == 0
    )


def _skipped_timing_row(
    state: SimulatorState, slot: SnapshotSlot, reason: str
) -> dict[str, object]:
    """Build a skipped telemetry timing row for a missed snapshot slot."""
    return {
        "SnapshotId": slot.snapshot_id,
        "Status": "skipped",
        "ScheduledOffsetSec": state.timing_offset(slot.scheduled_mono),
        "Overrun": True,
        "SkippedReason": reason,
    }


def _effective_fault_window(args: argparse.Namespace, fault_mode: str) -> tuple[int, int]:
    """Return the fault onset offset and fault duration, or zeros without a fault scenario."""
    if fault_mode != "scenario":
        return 0, 0
    return int(args.fault_start_offset_sec), int(args.fault_duration_sec)


def run_simulation(
    args: argparse.Namespace, run_log_dir: Path, fault_mode: str = "scenario"
) -> None:
    """Run one episode from network construction through traffic, probes, telemetry, and the fault to cleanup."""
    if fault_mode not in {"scenario", "none"}:
        raise ValueError(f"Unsupported fault_mode={fault_mode}")

    fault_category = FAULT_CATEGORY_NONE
    fault_schedule_mode = FAULT_SCHEDULE_NONE
    if fault_mode == "scenario":
        if args.root_cause_kind == "node":
            fault_category = FAULT_CATEGORY_DRAIN
            fault_schedule_mode = FAULT_SCHEDULE_STAGED_DRAIN
        elif args.root_cause_kind == "edge":
            fault_category = (
                str(getattr(args, "fault_category", "") or "") or FAULT_CATEGORY_FIBER_CUT
            )
            fault_schedule_mode = (
                FAULT_SCHEDULE_NETEM_LINK_DEGRADATION
                if fault_category == FAULT_CATEGORY_LINK_DEGRADATION
                else (
                    FAULT_SCHEDULE_BURSTY_LINK_FLAP
                    if fault_category == FAULT_CATEGORY_LINK_FLAP
                    else FAULT_SCHEDULE_INSTANT_LINK_DOWN
                )
            )

    fault_start_offset_sec, fault_duration_sec = _effective_fault_window(args, fault_mode)
    state = SimulatorState(run_log_dir)
    state.update_metadata(
        instance_id=args.instance_id,
        simulator_pid=os.getpid(),
        frr_runtime_dir=str(frr_runtime_dir(args.instance_id)),
        simulation_scope="l3_only",
        root_cause_kind=args.root_cause_kind or FAULT_CATEGORY_NONE,
        root_cause_id=args.fault_target,
        fault_target=args.fault_target,
        fault_category=fault_category,
        fault_schedule_mode=fault_schedule_mode,
        warmup_sec=args.warmup_sec,
        fault_duration_sec=fault_duration_sec,
        fault_start_offset_sec=fault_start_offset_sec,
        drain_ramp_steps=args.drain_ramp_steps,
        drain_phase_ratios={
            "ramp_down": args.drain_phase_ratio_ramp_down,
            "link_down": args.drain_phase_ratio_link_down,
            "hold_down": args.drain_phase_ratio_hold_down,
            "ramp_up": args.drain_phase_ratio_ramp_up,
        },
        probe_packets=args.probe_packets,
        probe_timeout_sec=args.probe_timeout_sec,
        probe_cadence_sec=args.probe_cadence_sec,
        probe_result_freshness_sec=round(
            max(args.probe_timeout_sec, float(args.interval) * 2.0), 4
        ),
    )
    net = None
    probe_scheduler: AsyncPingProbeScheduler | None = None
    try:
        net = build_network(args.instance_id)
        net.start()
        routing_details = configure_network(net)
        export_topology_csv(state.log_dir, net)

        rng = random.Random(args.seed)
        host_names = sorted(
            logical_name(node) for node in net.hosts if logical_name(node).startswith("h")
        )
        traffic_pairs = _select_host_pairs(
            net,
            rng,
            traffic_flow_pair_pool(host_names, rng),
            args.traffic_flow_min,
            args.traffic_flow_max,
        )
        traffic_intents = [
            {"src": logical_name(src), "dst": logical_name(dst)} for src, dst in traffic_pairs
        ]
        ping_pairs = _select_ping_pairs(
            net,
            rng,
            host_names,
            count_min=args.ping_pair_min,
            count_max=args.ping_pair_max,
        )
        packet_size = _sample_packet_size_bytes(rng)
        base_rate_pps = _sample_flow_rate_pps(rng, packet_size)
        protocol = str(rng.choice(("UDP", "UDP", "TCP")))
        for intent in traffic_intents:
            intent["transport"] = protocol
            intent["packet_size"] = packet_size

        burst_mean_gap_sec = resolved_burst_mean_gap_sec(args.interval, args.burst_mean_gap_sec)
        burst_overlay = _burst_overlay_schedule(
            collection_duration_sec=args.duration,
            telemetry_interval_sec=args.interval,
            mean_gap_sec=burst_mean_gap_sec,
        )
        baseline_duration_sec = _baseline_traffic_duration_sec(
            args.warmup_sec, args.duration, args.interval
        )
        baseline_flows = build_ditg_flows(
            state,
            traffic_pairs,
            kind="baseline",
            burst_id=None,
            start_offset_ms=0,
            duration_ms=baseline_duration_sec * 1000,
            rate_pps=base_rate_pps,
            packet_size=packet_size,
            protocol=protocol,
        )
        traffic_flows = list(baseline_flows)
        burst_pairs = traffic_pairs[: max(1, min(2, len(traffic_pairs)))]
        burst_rng = random.Random(f"{args.seed}:traffic-overlay-v2")
        burst_overlay_flows = []
        burst_overlay_rates_pps: list[int] = []
        for burst_id, pair in enumerate(burst_pairs):
            burst_rate_pps = _scaled_overlay_rate_pps(
                base_rate_pps,
                packet_size,
                burst_rng.uniform(1.8, 2.4),
            )
            burst_overlay_rates_pps.append(burst_rate_pps)
            burst_overlay_flows.extend(
                build_ditg_flows(
                    state,
                    [pair],
                    kind="burst",
                    burst_id=burst_id,
                    start_offset_ms=int(
                        round(
                            (args.warmup_sec + float(burst_overlay["collection_start_offset_sec"]))
                            * 1000.0
                        )
                    ),
                    duration_ms=int(burst_overlay["duration_ms"]),
                    rate_pps=burst_rate_pps,
                    packet_size=1024,
                    protocol="UDP",
                    on_duration_ms=int(burst_overlay["on_duration_ms"]),
                    off_duration_ms=int(burst_overlay["off_duration_ms"]),
                )
            )
        traffic_flows.extend(burst_overlay_flows)
        planned_burst_cycle_count = int(burst_overlay["planned_cycle_count"])

        traffic_plan_artifact = {
            "schema_version": "ridge-stage1-traffic-plan-v2",
            "schedule_policy": TRAFFIC_SCHEDULE_POLICY,
            "activation_guard_sec": DITG_ACTIVATION_GUARD_SEC,
            "binary_packet_logging": False,
            "baseline_duration_sec": baseline_duration_sec,
            "traffic_pair_count": len(traffic_pairs),
            "burst_mode": "ditg_native_constant_on_off_overlay",
            "burst_overlay": {
                **burst_overlay,
                "flow_count": len(burst_overlay_flows),
                "rates_pps": burst_overlay_rates_pps,
                "rate_policy": "baseline_bitrate_multiplier_uniform_1.8_2.4",
            },
            "flows": [asdict(flow) for flow in traffic_flows],
        }
        with (state.log_dir / "traffic_matrix.json").open("w", encoding="utf-8") as handle:
            json.dump(traffic_plan_artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")

        activation_mono = time.monotonic() + DITG_ACTIVATION_GUARD_SEC
        traffic_launch = launch_ditg_plan(
            net,
            state,
            traffic_flows,
            activation_mono=activation_mono,
        )
        traffic_launch_pass = _traffic_launch_complete(traffic_launch)
        baseline_attempted_count = len(baseline_flows)
        baseline_launched_count = baseline_attempted_count if traffic_launch_pass else 0
        baseline_failed_count = baseline_attempted_count - baseline_launched_count
        explicit_fault = bool(args.root_cause_kind and args.fault_target)
        scenario_fault_injected = False
        state.update_metadata(
            traffic_profile_id=f"realistic-v2-seed-{args.seed}",
            traffic_flow_count=len(traffic_pairs),
            ping_pair_count=len(ping_pairs),
            traffic_base_rate_pps=base_rate_pps,
            traffic_phase_rates_pps=[
                base_rate_pps,
                base_rate_pps,
                base_rate_pps,
            ],
            traffic_protocol_mix={protocol: len(traffic_pairs)},
            traffic_burst_mean_gap_sec=round(burst_mean_gap_sec, 3),
            traffic_burst_min_interarrival_sec=float(burst_overlay["cycle_sec"]),
            traffic_burst_mode="ditg_native_constant_on_off_overlay",
            traffic_burst_on_duration_ms=int(burst_overlay["on_duration_ms"]),
            traffic_burst_off_duration_ms=int(burst_overlay["off_duration_ms"]),
            traffic_burst_overlay_flow_count=len(burst_overlay_flows),
            traffic_burst_overlay_rates_pps=burst_overlay_rates_pps,
            traffic_burst_rate_policy="baseline_bitrate_multiplier_uniform_1.8_2.4",
            traffic_matrix_profile_id="seeded_pair_native_on_off_v2",
            traffic_schedule_policy=TRAFFIC_SCHEDULE_POLICY,
            ditg_binary_packet_logging=False,
            ditg_data_port_policy="sequential_unique_per_episode",
            ditg_data_port_start=DITG_DATA_PORT_MIN,
            traffic_matrix_entries=traffic_intents,
            traffic_planned_flow_count=len(traffic_flows),
            traffic_planned_source_process_count=int(
                traffic_launch.get("source_process_attempted_count", 0) or 0
            ),
            traffic_planned_receiver_process_count=int(
                traffic_launch.get("receiver_process_attempted_count", 0) or 0
            ),
            traffic_burst_scheduled_count=planned_burst_cycle_count,
            traffic_burst_successful_count=0,
            traffic_burst_failed_count=0,
            traffic_burst_receiver_restart_count=0,
            traffic_burst_first_failure_reason=str(
                traffic_launch.get("first_failure_reason", "") or ""
            ),
            traffic_runtime_check_count=0,
            traffic_runtime_failure_count=0,
            traffic_runtime_first_failure="",
            baseline_traffic_attempted_count=baseline_attempted_count,
            baseline_traffic_launched_count=baseline_launched_count,
            baseline_traffic_failed_count=baseline_failed_count,
            baseline_traffic_first_failure_reason=str(
                traffic_launch.get("first_failure_reason", "") or ""
            ),
            baseline_traffic_health_pass=traffic_launch_pass,
            telemetry_interval_sec=float(args.interval),
            telemetry_scheduler=STAGE1_SCHEDULING_POLICY,
            telemetry_control_tick_sec=CONTROL_TICK_SEC,
            telemetry_blocking_budget_sec=TELEMETRY_BLOCKING_BUDGET_SEC,
            telemetry_fault_transition_guard_sec=FAULT_TRANSITION_GUARD_SEC,
            probe_scheduler=STAGE1_PROBE_SCHEDULING_POLICY,
            probe_precollection_warmup_sec=float(args.warmup_sec),
            **routing_details,
        )
        if not traffic_launch_pass:
            raise RuntimeError(
                "D-ITG traffic-plan launch integrity check failed: "
                f"sources={traffic_launch.get('source_process_launched_count', 0)}/"
                f"{traffic_launch.get('source_process_attempted_count', 0)} "
                f"receivers={traffic_launch.get('receiver_process_launched_count', 0)}/"
                f"{traffic_launch.get('receiver_process_attempted_count', 0)}"
            )

        collection_started_mono = activation_mono + float(args.warmup_sec)
        probe_scheduler = AsyncPingProbeScheduler(
            ping_pairs,
            packets=args.probe_packets,
            timeout_sec=args.probe_timeout_sec,
            cadence_sec=args.probe_cadence_sec,
            freshness_sec=max(args.probe_timeout_sec, float(args.interval) * 2.0),
            timing_sink=state.record_probe_timing,
            timing_origin_mono=collection_started_mono,
        )
        while (warmup_remaining := collection_started_mono - time.monotonic()) > 0.0:
            probe_scheduler.tick()
            runtime_failures = ditg_runtime_failures(state, net)
            if runtime_failures:
                first = runtime_failures[0]
                raise RuntimeError(
                    "D-ITG process exited before collection: "
                    f"{first.get('role', '')}:{first.get('node', '')}:"
                    f"pid={first.get('pid', '')}"
                )
            time.sleep(min(CONTROL_TICK_SEC, warmup_remaining))
        probe_scheduler.tick()
        if missing_probe_pairs := probe_scheduler.missing_sample_pairs():
            formatted_pairs = ", ".join(
                f"{source}->{destination}" for source, destination in missing_probe_pairs
            )
            raise RuntimeError(
                "Async probe warmup completed without a result for every pair: "
                f"{formatted_pairs}. Increase warmup or inspect probe timing diagnostics."
            )
        collection_end_mono = collection_started_mono + float(args.duration)
        deterministic_fault_deadline = collection_started_mono + float(args.fault_start_offset_sec)
        state.set_timing_origin(collection_started_mono)
        telemetry_scheduler = FixedRateTelemetryScheduler(
            float(args.interval), collection_started_mono
        )

        baseline_loss_samples: list[float] = []
        baseline_rtt_samples: list[float] = []
        baseline_timeout_samples: list[float] = []
        baseline_missing_probe_samples = 0
        baseline_max_link_kbps = 0.0
        next_traffic_prune_mono = collection_started_mono
        recent_collection_durations_sec: list[float] = [TELEMETRY_BLOCKING_BUDGET_SEC]

        def service_control() -> None:
            """Advance the probes, inject and maintain the fault, and check that the traffic processes are alive."""
            nonlocal scenario_fault_injected, next_traffic_prune_mono
            now_mono = time.monotonic()
            if probe_scheduler is not None:
                probe_scheduler.tick(now=now_mono)

            if (
                fault_mode == "scenario"
                and not scenario_fault_injected
                and explicit_fault
                and now_mono >= deterministic_fault_deadline
            ):
                inject_requested_fault(
                    net,
                    state,
                    root_cause_kind=args.root_cause_kind,
                    fault_target=args.fault_target,
                    duration_sec=args.fault_duration_sec,
                    ramp_steps=args.drain_ramp_steps,
                    phase_ratios=DrainPhaseRatios(
                        ramp_down=float(args.drain_phase_ratio_ramp_down),
                        link_down=float(args.drain_phase_ratio_link_down),
                        hold_down=float(args.drain_phase_ratio_hold_down),
                        ramp_up=float(args.drain_phase_ratio_ramp_up),
                    ),
                    fault_category=str(getattr(args, "fault_category", "") or ""),
                    rng=rng,
                    scheduled_at_mono=deterministic_fault_deadline,
                    now_mono=now_mono,
                )
                scenario_fault_injected = True
            maintain_active_faults(state, now_mono=time.monotonic())

            if now_mono >= next_traffic_prune_mono:
                premature_exits = ditg_runtime_failures(state, net)
                state.metadata["traffic_runtime_check_count"] = (
                    int(state.metadata.get("traffic_runtime_check_count", 0) or 0) + 1
                )
                if premature_exits:
                    first = premature_exits[0]
                    state.update_metadata(
                        traffic_runtime_failure_count=int(
                            state.metadata.get("traffic_runtime_failure_count", 0) or 0
                        )
                        + len(premature_exits),
                        traffic_runtime_first_failure=(
                            f"{first.get('role', '')}:{first.get('node', '')}:"
                            f"pid={first.get('pid', '')}"
                        ),
                        baseline_traffic_health_pass=False,
                    )
                    raise RuntimeError(
                        f"{len(premature_exits)} D-ITG process(es) exited unexpectedly"
                    )
                next_traffic_prune_mono = now_mono + 0.5

        while telemetry_scheduler.next_deadline < collection_end_mono:
            service_control()
            now_mono = time.monotonic()
            for skipped_slot in telemetry_scheduler.skip_expired(now_mono):
                if skipped_slot.scheduled_mono < collection_end_mono:
                    state.record_telemetry_timing(
                        _skipped_timing_row(state, skipped_slot, "control_loop_deadline_missed")
                    )
            if telemetry_scheduler.next_deadline >= collection_end_mono:
                break
            if not telemetry_scheduler.ready(now_mono):
                sleep_sec = min(
                    CONTROL_TICK_SEC,
                    telemetry_scheduler.seconds_until_due(now_mono),
                    max(0.0, collection_end_mono - now_mono),
                )
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                continue

            next_control_deadline = (
                deterministic_fault_deadline
                if fault_mode == "scenario" and explicit_fault and not scenario_fault_injected
                else next_fault_transition_mono(state)
            )
            estimated_collection_sec = max(
                TELEMETRY_BLOCKING_BUDGET_SEC,
                max(recent_collection_durations_sec[-8:]) + 2.0 * CONTROL_TICK_SEC,
            )
            if telemetry_should_yield_to_control(
                now_mono=now_mono,
                next_control_deadline_mono=next_control_deadline,
                estimated_collection_sec=min(float(args.interval), estimated_collection_sec),
                maximum_control_lag_sec=FAULT_TRANSITION_LAG_BUDGET_SEC,
            ):
                sleep_sec = min(
                    CONTROL_TICK_SEC,
                    max(0.0, float(next_control_deadline) - now_mono),
                )
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
                continue

            slot = telemetry_scheduler.begin(now_mono)
            cycle_started_mono = time.monotonic()
            cycle_started_timestamp = utc_timestamp()
            snapshot_timestamp = cycle_started_timestamp
            batch = collect_telemetry_batch(
                net,
                state,
                snapshot_id=slot.snapshot_id,
                timestamp=snapshot_timestamp,
            )
            host_started_mono = time.monotonic()
            host_errors: list[str] = []
            try:
                host_rows = [
                    collect_host_stats(
                        state,
                        snapshot_id=slot.snapshot_id,
                        timestamp=snapshot_timestamp,
                    )
                ]
                host_status = "complete"
            except Exception as exc:
                host_rows = []
                host_status = "error"
                host_errors.append(f"host:{type(exc).__name__}:{exc}")
            host_duration = time.monotonic() - host_started_mono

            ping_started_mono = time.monotonic()
            ping_errors: list[str] = []
            try:
                probe_scheduler.tick(now=ping_started_mono)
                ping_rows = probe_scheduler.latest_rows(
                    timestamp=snapshot_timestamp,
                    now=ping_started_mono,
                    snapshot_id=slot.snapshot_id,
                )
                ping_status = (
                    "complete" if _probe_rows_complete(ping_rows, len(ping_pairs)) else "partial"
                )
            except Exception as exc:
                ping_rows = []
                ping_status = "error"
                ping_errors.append(f"ping:{type(exc).__name__}:{exc}")
            ping_duration = time.monotonic() - ping_started_mono
            for row in ping_rows:
                row["SnapshotId"] = slot.snapshot_id

            control_blocking_completed_mono = time.monotonic()
            control_error: Exception | None = None
            control_started_mono = time.monotonic()
            try:
                service_control()
            except Exception as exc:
                control_error = exc
            control_service_duration = time.monotonic() - control_started_mono

            persistence_started_mono = time.monotonic()
            state.append_rows(state.node_csv, NODE_HEADERS, batch.node_rows)
            state.append_rows(state.host_csv, HOST_HEADERS, host_rows)
            state.append_rows(state.interface_csv, INTERFACE_HEADERS, batch.interface_rows)
            state.append_rows(state.ping_csv, PING_HEADERS, ping_rows)
            state.append_rows(state.queue_csv, QUEUE_HEADERS, batch.queue_rows)
            state.append_rows(state.route_csv, ROUTE_HEADERS, batch.route_rows)
            state.append_rows(state.neighbor_csv, NEIGHBOR_HEADERS, batch.neighbor_rows)
            persistence_duration = time.monotonic() - persistence_started_mono

            active_fault_present = bool(state.active_faults)
            if not active_fault_present and (
                fault_mode != "scenario" or not scenario_fault_injected
            ):
                for row in ping_rows:
                    packet_loss = _finite_float(row.get("PacketLoss"))
                    avg_rtt = _finite_float(row.get("AvgRTT"))
                    timeout_flag = _finite_float(row.get("TimeoutFlag"))
                    if packet_loss is not None:
                        baseline_loss_samples.append(packet_loss)
                    if avg_rtt is not None:
                        baseline_rtt_samples.append(avg_rtt)
                    if timeout_flag is not None:
                        baseline_timeout_samples.append(timeout_flag)
                    if packet_loss is None and avg_rtt is None and timeout_flag is None:
                        baseline_missing_probe_samples += 1
                for row in batch.interface_rows:
                    for field in ("TX_KBPS", "RX_KBPS"):
                        value = _finite_float(row.get(field))
                        if value is not None:
                            baseline_max_link_kbps = max(baseline_max_link_kbps, value)

            cycle_completed_mono = time.monotonic()
            recent_collection_durations_sec.append(
                control_blocking_completed_mono - cycle_started_mono
            )
            cycle_completed_timestamp = utc_timestamp()
            control_errors = (
                [f"control:{type(control_error).__name__}:{control_error}"]
                if control_error is not None
                else []
            )
            all_errors = [*batch.errors, *host_errors, *ping_errors, *control_errors]
            statuses = [*batch.statuses.values(), host_status, ping_status]
            if control_error is not None:
                cycle_status = "error"
            elif all(status == "complete" for status in statuses):
                cycle_status = "complete"
            elif any(status == "error" for status in statuses):
                cycle_status = "error"
            else:
                cycle_status = "partial"
            state.record_telemetry_timing(
                {
                    "SnapshotId": slot.snapshot_id,
                    "Status": cycle_status,
                    "ScheduledOffsetSec": state.timing_offset(slot.scheduled_mono),
                    "ActualStartOffsetSec": state.timing_offset(cycle_started_mono),
                    "ActualEndOffsetSec": state.timing_offset(cycle_completed_mono),
                    "ActualStartTimestamp": cycle_started_timestamp,
                    "ActualEndTimestamp": cycle_completed_timestamp,
                    "StartLagSec": round(max(0.0, cycle_started_mono - slot.scheduled_mono), 6),
                    "DurationSec": round(max(0.0, cycle_completed_mono - cycle_started_mono), 6),
                    "ControlBlockingDurationSec": round(
                        max(0.0, control_blocking_completed_mono - cycle_started_mono), 6
                    ),
                    "ControlServiceDurationSec": round(control_service_duration, 6),
                    "PersistenceDurationSec": round(persistence_duration, 6),
                    "Overrun": cycle_completed_mono - cycle_started_mono >= float(args.interval),
                    "NodeDurationSec": round(batch.durations_sec["Node"], 6),
                    "NodeStatus": batch.statuses["Node"],
                    "NodeRowCount": len(batch.node_rows),
                    "HostDurationSec": round(host_duration, 6),
                    "HostStatus": host_status,
                    "HostRowCount": len(host_rows),
                    "InterfaceDurationSec": round(batch.durations_sec["Interface"], 6),
                    "InterfaceStatus": batch.statuses["Interface"],
                    "InterfaceRowCount": len(batch.interface_rows),
                    "QueueDurationSec": round(batch.durations_sec["Queue"], 6),
                    "QueueStatus": batch.statuses["Queue"],
                    "QueueRowCount": len(batch.queue_rows),
                    "RouteDurationSec": round(batch.durations_sec["Route"], 6),
                    "RouteStatus": batch.statuses["Route"],
                    "RouteRowCount": len(batch.route_rows),
                    "NeighborDurationSec": round(batch.durations_sec["Neighbor"], 6),
                    "NeighborStatus": batch.statuses["Neighbor"],
                    "NeighborRowCount": len(batch.neighbor_rows),
                    "PingDurationSec": round(ping_duration, 6),
                    "PingStatus": ping_status,
                    "PingRowCount": len(ping_rows),
                    "Error": " | ".join(all_errors)[:4000],
                }
            )
            for skipped_slot in telemetry_scheduler.finish(cycle_completed_mono):
                if skipped_slot.scheduled_mono < collection_end_mono:
                    state.record_telemetry_timing(
                        _skipped_timing_row(state, skipped_slot, "previous_cycle_overrun")
                    )
            if control_error is not None:
                raise control_error

        # Keep the configured collection window alive after the final telemetry slot so the 1-second probe cadence, 
        # fault recovery, and process liveness remain observable through the exact boundary.
        while (remaining_sec := collection_end_mono - time.monotonic()) > 0.0:
            service_control()
            time.sleep(min(CONTROL_TICK_SEC, remaining_sec))
        service_control()
        traffic_diagnostics = validate_ditg_diagnostics(state)
        traffic_diagnostics_pass = bool(traffic_diagnostics["passed"])
        burst_count = planned_burst_cycle_count
        traffic_burst_successful_count = burst_count if traffic_diagnostics_pass else 0
        traffic_burst_failed_count = burst_count - traffic_burst_successful_count
        diagnostic_failures = [str(item) for item in traffic_diagnostics["failures"]]
        traffic_burst_health_pass = traffic_diagnostics_pass
        state.update_metadata(
            traffic_burst_count=burst_count,
            traffic_burst_successful_count=traffic_burst_successful_count,
            traffic_burst_failed_count=traffic_burst_failed_count,
            traffic_burst_health_pass=traffic_burst_health_pass,
            traffic_burst_first_failure_reason=(
                diagnostic_failures[0] if diagnostic_failures else ""
            ),
            traffic_expected_started_flow_count=int(
                traffic_diagnostics["expected_started_flow_count"]
            ),
            traffic_observed_started_flow_count=int(
                traffic_diagnostics["observed_started_flow_count"]
            ),
            traffic_fatal_diagnostic_marker_count=int(traffic_diagnostics["fatal_marker_count"]),
            traffic_diagnostic_failures=diagnostic_failures,
            baseline_traffic_launched_count=(
                baseline_attempted_count if traffic_diagnostics_pass else 0
            ),
            baseline_traffic_failed_count=(
                0 if traffic_diagnostics_pass else baseline_attempted_count
            ),
            baseline_traffic_health_pass=traffic_diagnostics_pass,
            **state.telemetry_timing_summary(),
        )
        state.update_metadata(
            baseline_max_link_kbps=round(baseline_max_link_kbps, 4),
            **_baseline_health_summary(
                baseline_loss_samples,
                baseline_rtt_samples,
                baseline_timeout_samples,
                baseline_missing_probe_samples,
            ),
        )
        if not traffic_burst_health_pass:
            raise RuntimeError(
                "D-ITG traffic diagnostic check failed: "
                f"expected_started={traffic_diagnostics['expected_started_flow_count']} "
                f"observed_started={traffic_diagnostics['observed_started_flow_count']} "
                f"fatal_markers={traffic_diagnostics['fatal_marker_count']}"
            )
    finally:
        if probe_scheduler is not None:
            probe_scheduler.close()
        cleanup_errors: list[str] = []
        fault_cleanup_count = 0
        if net is not None:
            try:
                fault_cleanup_count = cleanup_active_faults(state)
            except Exception as exc:
                cleanup_errors.append(f"fault_cleanup:{exc}")
            try:
                stop_ditg_traffic(state, net)
            except Exception as exc:
                cleanup_errors.append(f"ditg_cleanup:{exc}")
            try:
                net.stop()
            except Exception as exc:
                cleanup_errors.append(f"net_stop:{exc}")
        instance_cleanup: dict[str, int] = {}
        if args.instance_id:
            try:
                instance_cleanup = cleanup_instance_runtime(
                    args.instance_id,
                    run_log_dir,
                    lambda message: info(f"*** {message}\n"),
                )
            except Exception as exc:
                cleanup_errors.append(f"instance_cleanup:{exc}")
        state.update_metadata(
            cleanup_faults=fault_cleanup_count,
            cleanup_instance=instance_cleanup,
            cleanup_errors=cleanup_errors,
            lingering_ditg=len(state.ditg_processes),
            lingering_receivers=len(state.ditg_receivers),
            lingering_active_faults=len(state.active_faults),
        )
        state.close()


def run_from_args(args: argparse.Namespace) -> int:
    """Validate the arguments, run the episode, and return the process exit code."""
    setLogLevel("info")
    random.seed(args.seed)

    def handle_signal(signum: int, _frame: object) -> None:
        """Raise KeyboardInterrupt so a termination signal unwinds through the cleanup path."""
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        validate_args(args)
        prepare_run_log_directory(Path(args.log_dir))
        if not args.skip_startup_cleanup:
            cleanup_mininet()
        run_simulation(args, Path(args.log_dir), fault_mode=args.fault_mode)
    except KeyboardInterrupt:
        info("*** Simulation interrupted\n")
        return 130
    except Exception as exc:
        info(f"*** Simulation failed: {exc}\n")
        traceback.print_exc()
        return 1
    return 0


def main() -> int:
    """Parse the command line and run the per-episode generator."""
    return run_from_args(parse_args())


if __name__ == "__main__":
    sys.exit(main())
