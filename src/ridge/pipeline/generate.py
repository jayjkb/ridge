"""Run many RIDGE episodes through a small parallel worker pool."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ridge.common.contracts import (
    MAX_GENERATION_WORKERS,
    RAW_TELEMETRY_UNITS,
    SCHEMA_VERSION,
    STAGE1_ARTIFACT_TYPE,
    STAGE1_PROBE_SCHEDULING_POLICY,
    STAGE1_SCHEDULING_POLICY,
    parse_bool,
)
from ridge.common.io import write_json
from ridge.common.validation import (
    require_duration_after_window,
    require_fault_window_fits,
    require_non_negative,
    require_positive,
    require_range,
)
from ridge.io.stage1_timing import evaluate_run_timing
from ridge.pipeline.arg_validation import (
    require_drain_phase_ratios,
    require_offset_before_duration,
    resolve_bounded_range,
    resolved_burst_mean_gap_sec,
)
from ridge.pipeline.generation_health import (
    EpisodeRetryBudget,
    GenerationGate,
    HostResourceMonitor,
)
from ridge.pipeline.mininet_runtime import (
    cleanup_instance_runtime,
    cleanup_mininet_state,
    prepare_empty_directory,
)
from ridge.pipeline.simulator import (
    FAULT_TRANSITION_GUARD_SEC,
    TELEMETRY_BLOCKING_BUDGET_SEC,
    validate_traffic_and_probe_args,
)
from ridge.sim.common import utc_timestamp
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
    FAULT_SCHEDULE_STAGED_DRAIN,
    LINK_DEGRADATION_SEVERITY_PROFILES,
    LINK_FLAP_COUNT_RANGE,
    LINK_FLAP_DOWN_DURATION_RANGE_SEC,
    LINK_FLAP_UP_GAP_RANGE_SEC,
)
from ridge.sim.topology import node_role, runtime_name

MAX_WORKERS = MAX_GENERATION_WORKERS
FAULT_CATEGORY_MIXED = "mixed"

# Declares what a complete Stage-1 run directory contains.
STAGE1_CORE_ARTIFACT_FILENAMES = (
    "topology_nodes.csv",
    "topology_links.csv",
    "node_stats.csv",
    "host_stats.csv",
    "interface_stats.csv",
    "queue_stats.csv",
    "route_stats.csv",
    "neighbor_stats.csv",
    "ping_stats.csv",
    "fault_log.csv",
    "telemetry_timing.csv",
    "probe_timing.csv",
    "fault_timing.csv",
    "run_metadata.json",
    "traffic_matrix.json",
)


@dataclass(frozen=True)
class FaultTimingRange:
    """Bounds in seconds for the sampled fault duration and fault onset offset."""

    duration_min_sec: int
    duration_max_sec: int
    start_offset_min_sec: int
    start_offset_max_sec: int


@dataclass(frozen=True)
class DrainProfile:
    """Number of rate steps and the four phase fractions of a drain."""

    ramp_steps: int
    ramp_down_ratio: float
    link_down_ratio: float
    hold_down_ratio: float
    ramp_up_ratio: float

    @property
    def ratios(self) -> tuple[float, float, float, float]:
        """Return the four phase fractions in schedule order."""
        return (
            self.ramp_down_ratio,
            self.link_down_ratio,
            self.hold_down_ratio,
            self.ramp_up_ratio,
        )


@dataclass(frozen=True)
class EpisodeTemplate:
    """Fault identity of one planned episode before timing and seed are assigned."""

    fault_mode: str
    root_cause_kind: str
    root_cause_id: str
    fault_target: str
    fault_category: str
    fault_schedule_mode: str
    target_link_role: str
    is_healthy_baseline: int
    is_fault_episode: int


@dataclass(frozen=True)
class EpisodeSpec:
    """Fully resolved plan of one episode, including its seed, warmup, and fault window."""
    
    run_id: int
    instance_id: str
    seed: int
    warmup_sec: int
    fault_duration_sec: int
    fault_start_offset_sec: int
    fault_mode: str
    root_cause_kind: str
    root_cause_id: str
    fault_target: str
    fault_category: str
    fault_schedule_mode: str
    target_link_role: str
    is_healthy_baseline: int
    is_fault_episode: int

    def to_row(self) -> dict[str, object]:
        """Return the specification as a plain dictionary for the manifest and the worker boundary."""
        return asdict(self)


def base36(value: int) -> str:
    """Encode a non-negative integer in base 36 using digits and lowercase letters."""
    if value < 0:
        raise ValueError("base36 value must be >= 0")
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    digits = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits))


def instance_id_for_run(run_id: int) -> str:
    """Derive the Mininet instance identifier of an episode and check it fits Linux interface names."""
    instance_id = f"i{base36(run_id)}"
    # Validate against the longest current logical names and Linux IFNAMSIZ.
    runtime_name("h1_1", instance_id)
    return instance_id


def cleanup_mininet() -> None:
    """Remove stale FRRouting runtime and Mininet state, logging to standard output."""
    cleanup_mininet_state(
        lambda message: print(message, flush=True),
        banner="cleaning stale Mininet state with mn -c",
    )


def check_args(args: argparse.Namespace) -> None:
    """Validate every generation argument, including the worker cap, fault ranges, and drain fractions."""
    require_positive("--runs", args.runs)
    require_positive("--workers", args.workers)
    if args.workers > MAX_WORKERS:
        raise ValueError(
            f"--workers must be <= {MAX_WORKERS}; higher concurrency is outside the certified range"
        )
    require_positive("--duration", args.duration)
    require_positive("--interval", args.interval)
    if args.burst_mean_gap_sec is not None:
        require_positive("--burst-mean-gap-sec", args.burst_mean_gap_sec)
    validate_traffic_and_probe_args(args)
    require_non_negative("--warmup-sec", args.warmup_sec)
    _offset_min, offset_max = resolve_bounded_range(
        "fault-start-offset",
        args.fault_start_offset_min_sec,
        args.fault_start_offset_max_sec,
        positive=False,
    )
    _duration_min, duration_max = resolve_bounded_range(
        "fault-duration",
        args.fault_duration_min_sec,
        args.fault_duration_max_sec,
        positive=True,
    )
    require_offset_before_duration(offset_max, args.duration)
    require_duration_after_window(args.duration, duration_max)
    require_fault_window_fits(args.duration, offset_max, duration_max)
    require_non_negative("--min-training-runs", args.min_training_runs)
    require_range("--fault-fraction", args.fault_fraction, 0.0, 1.0)
    require_positive("--drain-ramp-steps", args.drain_ramp_steps)
    require_drain_phase_ratios(drain_profile_from_args(args).ratios)
    if args.min_training_runs and args.runs < args.min_training_runs:
        print(
            f"warning: --runs={args.runs} is below the RIDGE training guidance of {args.min_training_runs}",
            flush=True,
        )


def drain_profile_from_args(args: argparse.Namespace) -> DrainProfile:
    """Build the drain profile from the command-line drain arguments."""
    return DrainProfile(
        ramp_steps=int(args.drain_ramp_steps),
        ramp_down_ratio=float(args.drain_phase_ratio_ramp_down),
        link_down_ratio=float(args.drain_phase_ratio_link_down),
        hold_down_ratio=float(args.drain_phase_ratio_hold_down),
        ramp_up_ratio=float(args.drain_phase_ratio_ramp_up),
    )


def _fault_timing_ranges(args: argparse.Namespace) -> FaultTimingRange:
    """Return the validated fault duration and fault onset offset ranges."""
    duration_min, duration_max = resolve_bounded_range(
        "fault-duration",
        args.fault_duration_min_sec,
        args.fault_duration_max_sec,
        positive=True,
    )
    offset_min, offset_max = resolve_bounded_range(
        "fault-start-offset",
        args.fault_start_offset_min_sec,
        args.fault_start_offset_max_sec,
        positive=False,
    )
    return FaultTimingRange(
        duration_min_sec=duration_min,
        duration_max_sec=duration_max,
        start_offset_min_sec=offset_min,
        start_offset_max_sec=offset_max,
    )


def _fault_episode_templates(args: argparse.Namespace) -> list[EpisodeTemplate]:
    """Build one template per faulty episode, cycling through the selected categories and their targets."""
    faulty_count = round(args.runs * args.fault_fraction)
    spine_targets = [node for node in DRAIN_NODE_FAULT_TARGETS if node_role(node) == "spine"]
    if args.fault_category in {FAULT_CATEGORY_MIXED, FAULT_CATEGORY_DRAIN} and not spine_targets:
        raise ValueError("No observed spine node targets available for drain scenarios")
    if (
        args.fault_category in {FAULT_CATEGORY_MIXED, FAULT_CATEGORY_FIBER_CUT}
        and not EDGE_FAULT_TARGETS
    ):
        raise ValueError(
            "No validated infrastructure edge targets available for fiber cut scenarios"
        )
    if (
        args.fault_category in {FAULT_CATEGORY_MIXED, FAULT_CATEGORY_LINK_DEGRADATION}
        and not EDGE_FAULT_TARGETS
    ):
        raise ValueError(
            "No validated infrastructure edge targets available for link degradation scenarios"
        )
    if (
        args.fault_category in {FAULT_CATEGORY_MIXED, FAULT_CATEGORY_LINK_FLAP}
        and not EDGE_FAULT_TARGETS
    ):
        raise ValueError(
            "No validated infrastructure edge targets available for link flap scenarios"
        )

    def edge_template(edge_id: str, category: str, schedule_mode: str) -> EpisodeTemplate:
        """Build the template of a link fault episode for one target link."""
        return EpisodeTemplate(
            fault_mode="scenario",
            root_cause_kind="edge",
            root_cause_id=edge_id,
            fault_target=edge_id,
            fault_category=category,
            fault_schedule_mode=schedule_mode,
            target_link_role="",
            is_healthy_baseline=0,
            is_fault_episode=1,
        )

    if args.fault_category == FAULT_CATEGORY_DRAIN:
        return [
            EpisodeTemplate(
                fault_mode="scenario",
                root_cause_kind="node",
                root_cause_id=spine_targets[index % len(spine_targets)],
                fault_target=spine_targets[index % len(spine_targets)],
                fault_category=FAULT_CATEGORY_DRAIN,
                fault_schedule_mode=FAULT_SCHEDULE_STAGED_DRAIN,
                target_link_role="",
                is_healthy_baseline=0,
                is_fault_episode=1,
            )
            for index in range(faulty_count)
        ]

    if args.fault_category == FAULT_CATEGORY_FIBER_CUT:
        return [
            edge_template(
                EDGE_FAULT_TARGETS[index % len(EDGE_FAULT_TARGETS)],
                FAULT_CATEGORY_FIBER_CUT,
                FAULT_SCHEDULE_INSTANT_LINK_DOWN,
            )
            for index in range(faulty_count)
        ]

    if args.fault_category == FAULT_CATEGORY_LINK_DEGRADATION:
        return [
            edge_template(
                EDGE_FAULT_TARGETS[index % len(EDGE_FAULT_TARGETS)],
                FAULT_CATEGORY_LINK_DEGRADATION,
                FAULT_SCHEDULE_NETEM_LINK_DEGRADATION,
            )
            for index in range(faulty_count)
        ]

    if args.fault_category == FAULT_CATEGORY_LINK_FLAP:
        return [
            edge_template(
                EDGE_FAULT_TARGETS[index % len(EDGE_FAULT_TARGETS)],
                FAULT_CATEGORY_LINK_FLAP,
                FAULT_SCHEDULE_BURSTY_LINK_FLAP,
            )
            for index in range(faulty_count)
        ]

    templates: list[EpisodeTemplate] = []
    for index in range(faulty_count):
        category_index = index % 4
        target_index = index // 4
        if category_index == 0:
            target = spine_targets[target_index % len(spine_targets)]
            templates.append(
                EpisodeTemplate(
                    fault_mode="scenario",
                    root_cause_kind="node",
                    root_cause_id=target,
                    fault_target=target,
                    fault_category=FAULT_CATEGORY_DRAIN,
                    fault_schedule_mode=FAULT_SCHEDULE_STAGED_DRAIN,
                    target_link_role="",
                    is_healthy_baseline=0,
                    is_fault_episode=1,
                )
            )
            continue

        if category_index == 1:
            edge_id = EDGE_FAULT_TARGETS[target_index % len(EDGE_FAULT_TARGETS)]
            templates.append(
                edge_template(edge_id, FAULT_CATEGORY_FIBER_CUT, FAULT_SCHEDULE_INSTANT_LINK_DOWN)
            )
            continue

        if category_index == 2:
            edge_id = EDGE_FAULT_TARGETS[target_index % len(EDGE_FAULT_TARGETS)]
            templates.append(
                edge_template(
                    edge_id, FAULT_CATEGORY_LINK_DEGRADATION, FAULT_SCHEDULE_NETEM_LINK_DEGRADATION
                )
            )
            continue

        edge_id = EDGE_FAULT_TARGETS[target_index % len(EDGE_FAULT_TARGETS)]
        templates.append(
            edge_template(edge_id, FAULT_CATEGORY_LINK_FLAP, FAULT_SCHEDULE_BURSTY_LINK_FLAP)
        )
    return templates


def _healthy_episode_templates(args: argparse.Namespace) -> list[EpisodeTemplate]:
    """Build one template without a fault for every healthy episode."""
    healthy_count = args.runs - round(args.runs * args.fault_fraction)
    return [
        EpisodeTemplate(
            fault_mode=FAULT_CATEGORY_NONE,
            root_cause_kind=FAULT_CATEGORY_NONE,
            root_cause_id="",
            fault_target="",
            fault_category=FAULT_CATEGORY_NONE,
            fault_schedule_mode=FAULT_CATEGORY_NONE,
            target_link_role="",
            is_healthy_baseline=1,
            is_fault_episode=0,
        )
        for _ in range(healthy_count)
    ]


def build_episode_specs(args: argparse.Namespace) -> list[EpisodeSpec]:
    """Shuffle the templates with the base seed and sample each episode's warmup, fault window, and seed."""
    rng = random.Random(args.seed)
    episodes = _fault_episode_templates(args) + _healthy_episode_templates(args)
    rng.shuffle(episodes)
    timing = _fault_timing_ranges(args)
    plan: list[EpisodeSpec] = []
    for run_id, episode in enumerate(episodes):
        if episode.fault_mode == "scenario":
            base_duration = int(rng.randint(timing.duration_min_sec, timing.duration_max_sec))
        else:
            base_duration = 0
        warmup_low = max(0, args.warmup_sec - 10)
        warmup_high = args.warmup_sec + 20
        warmup = rng.randint(warmup_low, max(warmup_low, warmup_high))
        episode_duration = (
            max(1, min(base_duration, args.duration - 1)) if episode.fault_mode == "scenario" else 0
        )
        episode_fault_start_offset = (
            int(rng.randint(timing.start_offset_min_sec, timing.start_offset_max_sec))
            if episode.fault_mode == "scenario"
            else 0
        )
        plan.append(
            EpisodeSpec(
                run_id=run_id,
                instance_id=instance_id_for_run(run_id),
                seed=args.seed + run_id,
                warmup_sec=warmup,
                fault_duration_sec=episode_duration,
                fault_start_offset_sec=episode_fault_start_offset,
                fault_mode=episode.fault_mode,
                root_cause_kind=episode.root_cause_kind,
                root_cause_id=episode.root_cause_id,
                fault_target=episode.fault_target,
                fault_category=episode.fault_category,
                fault_schedule_mode=episode.fault_schedule_mode,
                target_link_role=episode.target_link_role,
                is_healthy_baseline=episode.is_healthy_baseline,
                is_fault_episode=episode.is_fault_episode,
            )
        )
    return plan


def build_episode_plan(args: argparse.Namespace) -> list[dict[str, object]]:
    """Return the episode specifications of the plan as plain dictionaries."""
    return [episode.to_row() for episode in build_episode_specs(args)]


def episodes_for_worker(
    episodes: list[dict[str, object]],
    worker_index: int,
    workers: int,
) -> list[dict[str, object]]:
    """Return the episodes assigned to a worker by round-robin on episode identifier."""
    return [episode for episode in episodes if int(episode["run_id"]) % workers == worker_index]


def read_run_metadata(run_dir: Path) -> dict[str, object]:
    """Return the metadata written by the per-episode generator, or an empty dictionary when absent."""
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as handle:
        return json.load(handle)


def _needs_parent_cleanup(
    result: subprocess.CompletedProcess[str], metadata: dict[str, object]
) -> bool:
    """Retain parent cleanup for crashes without repeating successful cleanup."""
    if result.returncode != 0 or not metadata:
        return True
    if metadata.get("cleanup_errors"):
        return True
    return any(
        int(metadata.get(key, 0) or 0) > 0
        for key in ("lingering_ditg", "lingering_receivers", "lingering_active_faults")
    )


def build_sim_command(
    simulator_module: str,
    run_dir: Path,
    duration: int,
    interval: int,
    burst_mean_gap_sec: float | None,
    traffic_flow_min: int,
    traffic_flow_max: int,
    ping_pair_min: int,
    ping_pair_max: int,
    probe_packets: int,
    probe_timeout_sec: float,
    probe_cadence_sec: float,
    fault_start_offset_sec: int,
    drain_ramp_steps: int,
    drain_phase_ratio_ramp_down: float,
    drain_phase_ratio_link_down: float,
    drain_phase_ratio_hold_down: float,
    drain_phase_ratio_ramp_up: float,
    episode: dict[str, object],
) -> list[str]:
    """Build the command line that runs the per-episode generator for one planned episode."""
    instance_id = str(episode.get("instance_id", instance_id_for_run(int(episode["run_id"]))))
    command = [
        sys.executable,
        "-m",
        simulator_module,
        "--duration",
        str(duration),
        "--interval",
        str(interval),
        "--burst-mean-gap-sec",
        str(resolved_burst_mean_gap_sec(interval, burst_mean_gap_sec)),
        "--seed",
        str(episode["seed"]),
        "--traffic-flow-min",
        str(traffic_flow_min),
        "--traffic-flow-max",
        str(traffic_flow_max),
        "--ping-pair-min",
        str(ping_pair_min),
        "--ping-pair-max",
        str(ping_pair_max),
        "--probe-packets",
        str(probe_packets),
        "--probe-timeout-sec",
        str(probe_timeout_sec),
        "--probe-cadence-sec",
        str(probe_cadence_sec),
        "--instance-id",
        instance_id,
        "--fault-mode",
        str(episode["fault_mode"]),
        "--skip-startup-cleanup",
        "--warmup-sec",
        str(episode["warmup_sec"]),
        "--log-dir",
        str(run_dir),
    ]
    if episode["fault_mode"] == "scenario":
        command.extend(
            [
                "--root-cause-kind",
                str(episode["root_cause_kind"]),
                "--fault-target",
                str(episode["fault_target"]),
                "--fault-duration-sec",
                str(episode["fault_duration_sec"]),
                "--fault-start-offset-sec",
                str(fault_start_offset_sec),
                "--drain-ramp-steps",
                str(drain_ramp_steps),
                "--drain-phase-ratio-ramp-down",
                str(drain_phase_ratio_ramp_down),
                "--drain-phase-ratio-link-down",
                str(drain_phase_ratio_link_down),
                "--drain-phase-ratio-hold-down",
                str(drain_phase_ratio_hold_down),
                "--drain-phase-ratio-ramp-up",
                str(drain_phase_ratio_ramp_up),
            ]
        )
        if episode.get("fault_category"):
            command.extend(["--fault-category", str(episode["fault_category"])])
    return command


def execute_episode(command: list[str], run_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run the per-episode generator and move its log into the episode directory afterwards."""
    temporary_log = run_dir.parent / f".{run_dir.name}.simulator.log"
    try:
        with temporary_log.open("w") as log:
            return subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
    finally:
        run_dir.mkdir(parents=True, exist_ok=True)
        if temporary_log.exists():
            temporary_log.replace(run_dir / "simulator.log")


def build_manifest_row(
    run_id: int,
    worker_index: int,
    episode: dict[str, object],
    metadata: dict[str, object],
    result: subprocess.CompletedProcess[str],
    fault_start_offset_sec: int,
    started_at: str,
    elapsed_sec: float,
    run_dir: Path,
) -> dict[str, object]:
    """Combine the plan, the episode metadata, and the process result into one manifest row."""
    fault_category = str(episode.get("fault_category", "none"))
    is_healthy_baseline = int(
        episode.get("is_healthy_baseline", int(episode.get("fault_mode", "none") == "none"))
    )
    is_fault_episode = int(
        episode.get("is_fault_episode", int(episode.get("fault_mode", "none") != "none"))
    )
    return {
        "run_id": run_id,
        "worker": worker_index,
        "instance_id": metadata.get("instance_id", episode.get("instance_id", "")),
        "simulator_pid": metadata.get("simulator_pid", ""),
        "frr_runtime_dir": metadata.get("frr_runtime_dir", ""),
        "seed": episode["seed"],
        "root_cause_kind": episode["root_cause_kind"],
        "root_cause_id": episode["root_cause_id"],
        "fault_target": episode["fault_target"],
        "fault_category": metadata.get("fault_category", fault_category),
        "target_link_role": metadata.get("target_link_role", episode.get("target_link_role", "")),
        "fault_schedule_mode": metadata.get(
            "fault_schedule_mode", episode.get("fault_schedule_mode", "single_fault")
        ),
        "fault_start_ts": metadata.get("fault_start_ts", ""),
        "fault_end_ts": metadata.get("fault_end_ts", ""),
        "warmup_sec": episode["warmup_sec"],
        "fault_duration_sec": episode["fault_duration_sec"],
        "fault_start_offset_sec": fault_start_offset_sec,
        "is_healthy_baseline": is_healthy_baseline,
        "is_fault_episode": is_fault_episode,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "parent_cleanup_performed": metadata.get("parent_cleanup_performed", False),
        "parent_cleanup_error": metadata.get("parent_cleanup_error", ""),
        "started_at": started_at,
        "finished_at": utc_timestamp(),
        "elapsed_sec": round(elapsed_sec, 3),
        "log_dir": run_dir,
        "traffic_profile_id": metadata.get("traffic_profile_id", ""),
        "traffic_flow_count": metadata.get("traffic_flow_count", 0),
        "ping_pair_count": metadata.get("ping_pair_count", 0),
        "probe_packets": metadata.get("probe_packets", 0),
        "probe_timeout_sec": metadata.get("probe_timeout_sec", 0),
        "probe_cadence_sec": metadata.get("probe_cadence_sec", 0),
        "probe_result_freshness_sec": metadata.get("probe_result_freshness_sec", 0),
        "traffic_base_rate_pps": metadata.get("traffic_base_rate_pps", 0),
        "traffic_phase_rates_pps": json.dumps(metadata.get("traffic_phase_rates_pps", [])),
        "traffic_burst_count": metadata.get("traffic_burst_count", 0),
        "traffic_burst_scheduled_count": metadata.get("traffic_burst_scheduled_count", 0),
        "traffic_burst_successful_count": metadata.get("traffic_burst_successful_count", 0),
        "traffic_burst_failed_count": metadata.get("traffic_burst_failed_count", 0),
        "traffic_burst_receiver_restart_count": metadata.get(
            "traffic_burst_receiver_restart_count", 0
        ),
        "traffic_burst_first_failure_reason": metadata.get(
            "traffic_burst_first_failure_reason", ""
        ),
        "traffic_burst_health_pass": metadata.get("traffic_burst_health_pass", True),
        "baseline_traffic_attempted_count": metadata.get("baseline_traffic_attempted_count", 0),
        "baseline_traffic_launched_count": metadata.get("baseline_traffic_launched_count", 0),
        "baseline_traffic_failed_count": metadata.get("baseline_traffic_failed_count", 0),
        "baseline_traffic_first_failure_reason": metadata.get(
            "baseline_traffic_first_failure_reason", ""
        ),
        "baseline_traffic_health_pass": metadata.get("baseline_traffic_health_pass", False),
        "traffic_runtime_check_count": metadata.get("traffic_runtime_check_count", 0),
        "traffic_runtime_failure_count": metadata.get("traffic_runtime_failure_count", 0),
        "traffic_runtime_first_failure": metadata.get("traffic_runtime_first_failure", ""),
        "traffic_protocol_mix": json.dumps(
            metadata.get("traffic_protocol_mix", {}), sort_keys=True
        ),
        "simulation_scope": metadata.get("simulation_scope", "l3_only"),
        "traffic_matrix_profile_id": metadata.get("traffic_matrix_profile_id", ""),
        "baseline_mean_packet_loss_pct": metadata.get("baseline_mean_packet_loss_pct", ""),
        "baseline_timeout_ratio": metadata.get("baseline_timeout_ratio", ""),
        "baseline_p95_rtt_ms": metadata.get("baseline_p95_rtt_ms", ""),
        "baseline_max_link_kbps": metadata.get("baseline_max_link_kbps", 0),
        "baseline_health_pass": metadata.get("baseline_health_pass", False),
    }


def run_episode(
    simulator_module: str,
    output: Path,
    worker_index: int,
    duration: int,
    interval: int,
    burst_mean_gap_sec: float | None,
    traffic_flow_min: int,
    traffic_flow_max: int,
    ping_pair_min: int,
    ping_pair_max: int,
    probe_packets: int,
    probe_timeout_sec: float,
    probe_cadence_sec: float,
    fault_start_offset_sec: int,
    drain_ramp_steps: int,
    drain_phase_ratio_ramp_down: float,
    drain_phase_ratio_link_down: float,
    drain_phase_ratio_hold_down: float,
    drain_phase_ratio_ramp_up: float,
    episode: dict[str, object],
    resource_monitor: HostResourceMonitor | None = None,
) -> dict[str, object]:
    """Run one planned episode as a subprocess, clean up after a crash, and return its manifest row."""
    run_id = int(episode["run_id"])
    episode = {**episode, "instance_id": episode.get("instance_id", instance_id_for_run(run_id))}
    run_dir = output / f"run_{run_id:06d}"
    started_at = utc_timestamp()
    start = time.monotonic()
    command = build_sim_command(
        simulator_module=simulator_module,
        run_dir=run_dir,
        duration=duration,
        interval=interval,
        burst_mean_gap_sec=burst_mean_gap_sec,
        traffic_flow_min=traffic_flow_min,
        traffic_flow_max=traffic_flow_max,
        ping_pair_min=ping_pair_min,
        ping_pair_max=ping_pair_max,
        probe_packets=probe_packets,
        probe_timeout_sec=probe_timeout_sec,
        probe_cadence_sec=probe_cadence_sec,
        fault_start_offset_sec=fault_start_offset_sec,
        drain_ramp_steps=drain_ramp_steps,
        drain_phase_ratio_ramp_down=drain_phase_ratio_ramp_down,
        drain_phase_ratio_link_down=drain_phase_ratio_link_down,
        drain_phase_ratio_hold_down=drain_phase_ratio_hold_down,
        drain_phase_ratio_ramp_up=drain_phase_ratio_ramp_up,
        episode=episode,
    )
    if resource_monitor is not None:
        resource_monitor.simulator_started()
    try:
        result = execute_episode(command, run_dir)
    finally:
        if resource_monitor is not None:
            resource_monitor.simulator_finished()
    metadata = read_run_metadata(run_dir)
    parent_cleanup_performed = False
    parent_cleanup_error = ""
    if _needs_parent_cleanup(result, metadata):
        parent_cleanup_performed = True
        try:
            cleanup_instance_runtime(
                str(episode["instance_id"]),
                run_dir,
                lambda message: print(f"post-run cleanup run_{run_id:06d}: {message}", flush=True),
            )
        except Exception as exc:
            parent_cleanup_error = f"{type(exc).__name__}: {exc}"
    metadata = {
        **metadata,
        "parent_cleanup_performed": parent_cleanup_performed,
        "parent_cleanup_error": parent_cleanup_error,
    }
    return build_manifest_row(
        run_id=run_id,
        worker_index=worker_index,
        episode=episode,
        metadata=metadata,
        result=result,
        fault_start_offset_sec=fault_start_offset_sec,
        started_at=started_at,
        elapsed_sec=time.monotonic() - start,
        run_dir=run_dir,
    )


def episode_is_acceptable(
    row: dict[str, object], *, duration_sec: float, interval_sec: float
) -> bool:
    """An episode is acceptable only if it completed and passes its per-episode timing gate."""
    if row.get("status") != "ok":
        return False
    try:
        report = evaluate_run_timing(
            Path(str(row["log_dir"])),
            duration_sec=duration_sec,
            interval_sec=interval_sec,
        )
    except Exception:
        return False
    return bool(report.get("passed"))


def archive_failed_attempt(output: Path, run_id: int, attempt: int) -> None:
    """Move a failed episode directory aside so the identical specification can be rerun.

    Archived attempts live under ``_failed_attempts/`` so they are preserved for diagnosis.
    """
    run_dir = output / f"run_{run_id:06d}"
    if not run_dir.exists():
        return
    archive_root = output / "_failed_attempts"
    archive_root.mkdir(exist_ok=True)
    dest = archive_root / f"run_{run_id:06d}_attempt{attempt}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(run_dir), str(dest))


def worker_runs(
    simulator_module: str,
    args: argparse.Namespace,
    worker_index: int,
    episodes: list[dict[str, object]],
    gate: GenerationGate | None = None,
    resource_monitor: HostResourceMonitor | None = None,
    retry_budget: EpisodeRetryBudget | None = None,
) -> list[dict[str, object]]:
    """Run a worker's episodes in turn, retrying transient failures within budget and recording each with the gate."""
    rows: list[dict[str, object]] = []
    if resource_monitor is not None:
        resource_monitor.worker_started()
    try:
        for episode in episodes_for_worker(episodes, worker_index, args.workers):
            if gate is not None and gate.stop_event.is_set():
                break
            run_id = int(episode["run_id"])
            attempt = 0
            while True:
                row = run_episode(
                    simulator_module=simulator_module,
                    output=args.output,
                    worker_index=worker_index,
                    duration=args.duration,
                    interval=args.interval,
                    burst_mean_gap_sec=args.burst_mean_gap_sec,
                    traffic_flow_min=args.traffic_flow_min,
                    traffic_flow_max=args.traffic_flow_max,
                    ping_pair_min=args.ping_pair_min,
                    ping_pair_max=args.ping_pair_max,
                    probe_packets=args.probe_packets,
                    probe_timeout_sec=args.probe_timeout_sec,
                    probe_cadence_sec=args.probe_cadence_sec,
                    fault_start_offset_sec=int(episode["fault_start_offset_sec"]),
                    drain_ramp_steps=args.drain_ramp_steps,
                    drain_phase_ratio_ramp_down=args.drain_phase_ratio_ramp_down,
                    drain_phase_ratio_link_down=args.drain_phase_ratio_link_down,
                    drain_phase_ratio_hold_down=args.drain_phase_ratio_hold_down,
                    drain_phase_ratio_ramp_up=args.drain_phase_ratio_ramp_up,
                    episode=episode,
                    resource_monitor=resource_monitor,
                )
                if retry_budget is None or episode_is_acceptable(
                    row, duration_sec=args.duration, interval_sec=args.interval
                ):
                    break
                if not retry_budget.try_consume(run_id):
                    if gate is not None:
                        gate.stop_event.set()
                    break
                attempt += 1
                archive_failed_attempt(args.output, run_id, attempt)
                print(
                    f"retry run_{run_id:06d} attempt {attempt} "
                    "after transient timing/health failure",
                    flush=True,
                )
            rows.append(row)
            if gate is not None:
                gate.record_episode(row, resource_monitor=resource_monitor)
            print(f"completed run_{run_id:06d} status={row['status']}", flush=True)
    finally:
        if resource_monitor is not None:
            resource_monitor.worker_finished()
    return rows


def write_manifest(output: Path, rows: list[dict[str, object]]) -> None:
    """Write the manifest rows sorted by episode identifier to manifest.csv."""
    if not rows:
        return
    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["run_id"])))


def write_realism_reports(
    output: Path,
    rows: list[dict[str, object]],
    *,
    expected_run_count: int | None = None,
) -> dict[str, bool]:
    """Summarize the manifest, evaluate the realism and certification checks, and write both reports."""
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    flow_counts = [int(row.get("traffic_flow_count", 0) or 0) for row in ok_rows]
    ping_counts = [int(row.get("ping_pair_count", 0) or 0) for row in ok_rows]
    burst_counts = [int(row.get("traffic_burst_count", 0) or 0) for row in ok_rows]
    burst_scheduled_counts = [
        int(row.get("traffic_burst_scheduled_count", 0) or 0) for row in ok_rows
    ]
    burst_successful_counts = [
        int(row.get("traffic_burst_successful_count", 0) or 0) for row in ok_rows
    ]
    burst_failed_counts = [int(row.get("traffic_burst_failed_count", 0) or 0) for row in ok_rows]
    baseline_health_passes = [
        parse_bool(row.get("baseline_health_pass", False), field="baseline_health_pass")
        for row in ok_rows
    ]

    def finite_values(key: str) -> list[float]:
        """Return the finite values of one manifest column over the successful episodes."""
        values: list[float] = []
        for row in ok_rows:
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    baseline_losses = finite_values("baseline_mean_packet_loss_pct")
    baseline_timeouts = finite_values("baseline_timeout_ratio")
    baseline_rtts = finite_values("baseline_p95_rtt_ms")
    warmups = [
        int(row.get("warmup_sec", 0) or 0)
        for row in ok_rows
        if int(row.get("is_fault_episode", 0) or 0) == 1
    ]
    durations = [
        int(row.get("fault_duration_sec", 0) or 0)
        for row in ok_rows
        if int(row.get("is_fault_episode", 0) or 0) == 1
    ]
    categories: dict[str, int] = {}
    protocols: dict[str, int] = {}
    for row in ok_rows:
        categories[str(row.get("fault_category", "none"))] = (
            categories.get(str(row.get("fault_category", "none")), 0) + 1
        )
        try:
            mix = json.loads(str(row.get("traffic_protocol_mix", "{}")))
        except json.JSONDecodeError:
            mix = {}
        for key, value in mix.items():
            protocols[str(key)] = protocols.get(str(key), 0) + int(value)
    summary = {
        "run_count": len(rows),
        "ok_count": len(ok_rows),
        "flow_count": {
            "min": min(flow_counts or [0]),
            "max": max(flow_counts or [0]),
            "mean": round(sum(flow_counts or [0]) / max(1, len(flow_counts)), 3),
        },
        "ping_pair_count": {
            "min": min(ping_counts or [0]),
            "max": max(ping_counts or [0]),
            "mean": round(sum(ping_counts or [0]) / max(1, len(ping_counts)), 3),
        },
        "burst_count": {
            "min": min(burst_counts or [0]),
            "max": max(burst_counts or [0]),
            "mean": round(sum(burst_counts or [0]) / max(1, len(burst_counts)), 3),
        },
        "burst_scheduled_count": {
            "min": min(burst_scheduled_counts or [0]),
            "max": max(burst_scheduled_counts or [0]),
            "mean": round(
                sum(burst_scheduled_counts or [0]) / max(1, len(burst_scheduled_counts)), 3
            ),
        },
        "burst_successful_count": {
            "min": min(burst_successful_counts or [0]),
            "max": max(burst_successful_counts or [0]),
            "mean": round(
                sum(burst_successful_counts or [0]) / max(1, len(burst_successful_counts)), 3
            ),
        },
        "burst_failed_count": {
            "min": min(burst_failed_counts or [0]),
            "max": max(burst_failed_counts or [0]),
            "mean": round(sum(burst_failed_counts or [0]) / max(1, len(burst_failed_counts)), 3),
        },
        "fault_warmup_sec": {
            "min": min(warmups or [0]),
            "max": max(warmups or [0]),
            "mean": round(sum(warmups or [0]) / max(1, len(warmups)), 3),
        },
        "fault_duration_sec": {
            "min": min(durations or [0]),
            "max": max(durations or [0]),
            "mean": round(sum(durations or [0]) / max(1, len(durations)), 3),
        },
        "fault_category_distribution": categories,
        "protocol_mix": protocols,
        "baseline_health_pass_rate": round(
            sum(1 for flag in baseline_health_passes if flag) / max(1, len(baseline_health_passes)),
            4,
        ),
        "baseline_metric_sample_counts": {
            "packet_loss": len(baseline_losses),
            "timeout_ratio": len(baseline_timeouts),
            "p95_rtt": len(baseline_rtts),
        },
        "baseline_mean_packet_loss_pct": {
            "min": round(min(baseline_losses or [0.0]), 4),
            "max": round(max(baseline_losses or [0.0]), 4),
            "mean": round(sum(baseline_losses or [0.0]) / max(1, len(baseline_losses)), 4),
        },
        "baseline_timeout_ratio": {
            "min": round(min(baseline_timeouts or [0.0]), 4),
            "max": round(max(baseline_timeouts or [0.0]), 4),
            "mean": round(sum(baseline_timeouts or [0.0]) / max(1, len(baseline_timeouts)), 4),
        },
        "baseline_p95_rtt_ms": {
            "min": round(min(baseline_rtts or [0.0]), 4),
            "max": round(max(baseline_rtts or [0.0]), 4),
            "mean": round(sum(baseline_rtts or [0.0]) / max(1, len(baseline_rtts)), 4),
        },
    }
    checks = {
        "flow_variability_non_trivial": bool(
            flow_counts and (max(flow_counts) - min(flow_counts) >= 1)
        ),
        "probe_coverage_non_trivial": bool(ping_counts and min(ping_counts) >= 4),
        "single_fault_label_integrity": all(
            (
                row.get("fault_category") == FAULT_CATEGORY_NONE
                and int(row.get("is_fault_episode", 0) or 0) == 0
            )
            or (
                row.get("fault_category") != FAULT_CATEGORY_NONE
                and int(row.get("is_fault_episode", 0) or 0) == 1
            )
            for row in rows
        ),
        "burst_presence": bool(sum(burst_counts) > 0),
        "burst_launch_integrity": all(
            int(row.get("traffic_burst_failed_count", 0) or 0) == 0
            and int(row.get("traffic_burst_scheduled_count", 0) or 0)
            == int(row.get("traffic_burst_successful_count", 0) or 0)
            and int(row.get("traffic_burst_receiver_restart_count", 0) or 0) == 0
            and parse_bool(
                row.get("traffic_burst_health_pass", False),
                field="traffic_burst_health_pass",
            )
            for row in ok_rows
        ),
        "l3_scope_only": all(
            str(row.get("simulation_scope", "l3_only")) == "l3_only" for row in ok_rows
        ),
        "baseline_health_pass_rate_ge_0_80": bool(
            baseline_health_passes
            and (
                sum(1 for flag in baseline_health_passes if flag)
                / max(1, len(baseline_health_passes))
            )
            >= 0.8
        ),
        "baseline_metrics_finite_and_complete": bool(
            ok_rows
            and len(baseline_losses) == len(ok_rows)
            and len(baseline_timeouts) == len(ok_rows)
            and len(baseline_rtts) == len(ok_rows)
        ),
    }
    certification_checks = {
        "all_observed_runs_completed": bool(rows) and len(ok_rows) == len(rows),
        "requested_episode_count_complete": len(rows)
        == (len(rows) if expected_run_count is None else expected_run_count),
        "minimum_probe_coverage": bool(ping_counts and min(ping_counts) >= 4),
        "single_fault_label_integrity": checks["single_fault_label_integrity"],
        "burst_presence": checks["burst_presence"],
        "burst_launch_integrity": checks["burst_launch_integrity"],
        "l3_scope_only": checks["l3_scope_only"],
        "all_baseline_probes_healthy": bool(baseline_health_passes) and all(baseline_health_passes),
        "baseline_metrics_finite_and_complete": checks["baseline_metrics_finite_and_complete"],
        "baseline_traffic_launch_and_runtime_integrity": bool(ok_rows)
        and all(
            int(row.get("baseline_traffic_attempted_count", 0) or 0) > 0
            and int(row.get("baseline_traffic_launched_count", 0) or 0)
            == int(row.get("baseline_traffic_attempted_count", 0) or 0)
            and int(row.get("baseline_traffic_failed_count", 0) or 0) == 0
            and parse_bool(
                row.get("baseline_traffic_health_pass", False),
                field="baseline_traffic_health_pass",
            )
            and int(row.get("traffic_runtime_check_count", 0) or 0) > 0
            and int(row.get("traffic_runtime_failure_count", 0) or 0) == 0
            for row in ok_rows
        ),
    }
    checks["certification_passed"] = all(certification_checks.values())
    checks["certification_checks"] = certification_checks
    with (output / "realism_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output / "realism_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return certification_checks


def write_generation_provenance(
    args: argparse.Namespace,
    output: Path,
    rows: list[dict[str, object]],
    resolved_episodes: list[dict[str, object]] | None = None,
) -> None:
    """Write the resolved configuration, episode plan, schema, topology, and source revision of the generation."""
    selected_fault_categories = (
        [
            FAULT_CATEGORY_DRAIN,
            FAULT_CATEGORY_FIBER_CUT,
            FAULT_CATEGORY_LINK_DEGRADATION,
            FAULT_CATEGORY_LINK_FLAP,
        ]
        if args.fault_category == FAULT_CATEGORY_MIXED
        else [args.fault_category]
    )
    schedule_by_category = {
        FAULT_CATEGORY_DRAIN: FAULT_SCHEDULE_STAGED_DRAIN,
        FAULT_CATEGORY_FIBER_CUT: FAULT_SCHEDULE_INSTANT_LINK_DOWN,
        FAULT_CATEGORY_LINK_DEGRADATION: FAULT_SCHEDULE_NETEM_LINK_DEGRADATION,
        FAULT_CATEGORY_LINK_FLAP: FAULT_SCHEDULE_BURSTY_LINK_FLAP,
    }
    selected_fault_schedule_modes = [
        schedule_by_category[category] for category in selected_fault_categories
    ]
    candidate_catalog = {
        "node_ids": sorted(set(DRAIN_NODE_FAULT_TARGETS)),
        "edge_ids": sorted(
            set(EDGE_FAULT_TARGETS) | set(EDGE_FAULT_TARGETS) | set(EDGE_FAULT_TARGETS)
        ),
        "fault_categories": selected_fault_categories,
    }
    resolved_config = {
        "workers": args.workers,
        "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "host_online_cpu_count": os.cpu_count(),
        "duration_sec": args.duration,
        "interval_sec": args.interval,
        "burst_mean_gap_sec": resolved_burst_mean_gap_sec(args.interval, args.burst_mean_gap_sec),
        "warmup_sec_default": args.warmup_sec,
        "fault_start_offset_min_sec": args.fault_start_offset_min_sec,
        "fault_start_offset_max_sec": args.fault_start_offset_max_sec,
        "fault_duration_min_sec": args.fault_duration_min_sec,
        "fault_duration_max_sec": args.fault_duration_max_sec,
        "fault_fraction": args.fault_fraction,
        "fault_category_selection": args.fault_category,
        "drain_ramp_steps": args.drain_ramp_steps,
        "drain_phase_ratios": {
            "ramp_down": args.drain_phase_ratio_ramp_down,
            "link_down": args.drain_phase_ratio_link_down,
            "hold_down": args.drain_phase_ratio_hold_down,
            "ramp_up": args.drain_phase_ratio_ramp_up,
        },
        "traffic_flow_min": args.traffic_flow_min,
        "traffic_flow_max": args.traffic_flow_max,
        "ditg_binary_packet_logging": False,
        "ping_pair_min": args.ping_pair_min,
        "ping_pair_max": args.ping_pair_max,
        "probe_packets": args.probe_packets,
        "probe_timeout_sec": args.probe_timeout_sec,
        "probe_cadence_sec": args.probe_cadence_sec,
        "probe_result_freshness_sec": max(args.probe_timeout_sec, float(args.interval) * 2.0),
        "probe_precollection_warmup_policy": "entire_resolved_episode_warmup",
        "simulation_scope": "l3_only",
        "seed": args.seed,
        "scheduling_policy": STAGE1_SCHEDULING_POLICY,
        "telemetry_blocking_budget_sec": TELEMETRY_BLOCKING_BUDGET_SEC,
        "telemetry_fault_transition_guard_sec": FAULT_TRANSITION_GUARD_SEC,
        "probe_scheduling_policy": STAGE1_PROBE_SCHEDULING_POLICY,
    }
    episode_source = resolved_episodes if resolved_episodes is not None else rows
    resolved_episode_specs = [
        {field.name: row.get(field.name) for field in fields(EpisodeSpec)}
        for row in sorted(episode_source, key=lambda item: int(item["run_id"]))
    ]

    successful_row = next((row for row in rows if row.get("status") == "ok"), None)
    telemetry_schema: dict[str, list[dict[str, str]]] = {}
    topology_identity: dict[str, list[dict[str, str]]] = {}
    if successful_row is not None:
        run_path = Path(str(successful_row["log_dir"]))
        if not run_path.is_absolute() and not run_path.exists():
            run_path = output / run_path
        for family, filename in (
            ("node", "node_stats.csv"),
            ("host_reference", "host_stats.csv"),
            ("interface", "interface_stats.csv"),
            ("queue", "queue_stats.csv"),
            ("route", "route_stats.csv"),
            ("neighbor", "neighbor_stats.csv"),
            ("probe", "ping_stats.csv"),
        ):
            with (run_path / filename).open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            telemetry_schema[family] = [
                {"name": column, "unit": RAW_TELEMETRY_UNITS.get(column, "categorical")}
                for column in header
            ]
        topology_payload: dict[str, list[dict[str, str]]] = {}
        for filename in ("topology_nodes.csv", "topology_links.csv"):
            with (run_path / filename).open(newline="", encoding="utf-8") as handle:
                topology_payload[filename] = list(csv.DictReader(handle))
        topology_identity = topology_payload

    repository_root = Path(__file__).resolve().parents[3]
    source_code_commit = ""
    source_code_dirty = True
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if revision_result.returncode == 0:
            source_code_commit = revision_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        source_code_dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    except OSError:
        pass

    source_code_revision = source_code_commit
    if source_code_dirty:
        source_code_revision = f"{source_code_commit or 'unknown'}+dirty"

    gate_path = output / "generation_gate.json"
    generation_gate = {}
    if gate_path.exists():
        with gate_path.open(encoding="utf-8") as handle:
            generation_gate = json.load(handle)
    source_validation_passed = bool(
        generation_gate.get("passed") is True
        and generation_gate.get("generation_halted") is False
        and len(rows) == args.runs
        and all(row.get("status") == "ok" for row in rows)
    )

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": STAGE1_ARTIFACT_TYPE,
        "source_code_revision": source_code_revision,
        "source_code_commit": source_code_commit,
        "source_code_dirty": source_code_dirty,
        "resolved_config": resolved_config,
        "resolved_episode_specs": resolved_episode_specs,
        "timing_completeness_summary": generation_gate,
        "source_validation_passed": source_validation_passed,
        "telemetry_schema": telemetry_schema,
        "topology": topology_identity,
        "candidate_catalog": candidate_catalog,
        "generator": "ridge.pipeline.generate",
        "generated_at_utc": utc_timestamp(),
        "python_executable": sys.executable,
        "run_count": len(rows),
        "requested_run_count": args.runs,
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "workers": args.workers,
        "seed": args.seed,
        "duration_sec": args.duration,
        "interval_sec": args.interval,
        "burst_mean_gap_sec": resolved_burst_mean_gap_sec(args.interval, args.burst_mean_gap_sec),
        "warmup_sec_default": args.warmup_sec,
        "fault_start_offset_min_sec": args.fault_start_offset_min_sec,
        "fault_start_offset_max_sec": args.fault_start_offset_max_sec,
        "fault_duration_min_sec": args.fault_duration_min_sec,
        "fault_duration_max_sec": args.fault_duration_max_sec,
        "fault_fraction": args.fault_fraction,
        "fault_category_selection": args.fault_category,
        "fault_categories": selected_fault_categories,
        "fault_schedule_modes": selected_fault_schedule_modes,
        "drain_node_fault_targets": list(DRAIN_NODE_FAULT_TARGETS),
        "fiber_cut_edge_fault_targets": list(EDGE_FAULT_TARGETS),
        "link_degradation_edge_fault_targets": list(EDGE_FAULT_TARGETS),
        "link_flap_edge_fault_targets": list(EDGE_FAULT_TARGETS),
        "link_flap_timing_defaults": {
            "count_range": list(LINK_FLAP_COUNT_RANGE),
            "down_duration_range_sec": list(LINK_FLAP_DOWN_DURATION_RANGE_SEC),
            "up_gap_range_sec": list(LINK_FLAP_UP_GAP_RANGE_SEC),
        },
        "link_degradation_severity_profiles": [
            {
                "name": profile.name,
                "delay_ms": list(profile.delay_ms),
                "jitter_ms": list(profile.jitter_ms),
                "loss_pct": list(profile.loss_pct),
            }
            for profile in LINK_DEGRADATION_SEVERITY_PROFILES
        ],
        "drain_ramp_steps": args.drain_ramp_steps,
        "drain_phase_ratios": {
            "ramp_down": args.drain_phase_ratio_ramp_down,
            "link_down": args.drain_phase_ratio_link_down,
            "hold_down": args.drain_phase_ratio_hold_down,
            "ramp_up": args.drain_phase_ratio_ramp_up,
        },
        "traffic_flow_min": args.traffic_flow_min,
        "traffic_flow_max": args.traffic_flow_max,
        "ping_pair_min": args.ping_pair_min,
        "ping_pair_max": args.ping_pair_max,
        "probe_packets": args.probe_packets,
        "probe_timeout_sec": args.probe_timeout_sec,
        "probe_cadence_sec": args.probe_cadence_sec,
        "probe_result_freshness_sec": max(args.probe_timeout_sec, float(args.interval) * 2.0),
        "simulation_scope": "l3_only",
    }
    with (output / "generation_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")


def prepare_output_directory(output: Path) -> None:
    """Create a fresh generation root without ever appending to a previous generation."""
    prepare_empty_directory(output, kind="Output")


def run_from_args(args: argparse.Namespace) -> int:
    """Plan the episodes, run them across workers with gates and retries, write the reports, and return the exit code."""
    check_args(args)
    simulator_module = "ridge.pipeline.simulator"
    prepare_output_directory(args.output)
    cleanup_mininet()
    episodes = build_episode_plan(args)
    rows: list[dict[str, object]] = []
    gate = GenerationGate(
        args.output / "generation_gate.json",
        duration_sec=args.duration,
        interval_sec=args.interval,
    )
    resource_monitor = HostResourceMonitor(args.output / "generation_resources.jsonl")
    resource_monitor.start()
    resource_summary: dict[str, object] = {}
    if args.max_total_retries >= 0:
        max_total_retries = args.max_total_retries
    else:
        max_total_retries = max(10, math.ceil(0.02 * args.runs))
    retry_budget = EpisodeRetryBudget(
        max_per_episode=args.max_episode_retries,
        max_total=max_total_retries,
    )
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    worker_runs,
                    simulator_module,
                    args,
                    worker_index,
                    episodes,
                    gate,
                    resource_monitor,
                    retry_budget,
                )
                for worker_index in range(args.workers)
            ]
            for future in futures:
                rows.extend(future.result())
    finally:
        resource_summary = resource_monitor.stop()
        cleanup_mininet()
    retry_summary = retry_budget.summary()
    write_json(args.output / "generation_retries.json", retry_summary)
    write_manifest(args.output, rows)
    realism_certification = write_realism_reports(
        args.output,
        rows,
        expected_run_count=args.runs,
    )
    certification_checks = {
        **realism_certification,
        "requested_worker_concurrency_observed": int(
            resource_summary.get("max_active_simulators", 0) or 0
        )
        == min(args.workers, args.runs),
    }
    gate_summary = gate.finalize(
        resource_summary,
        certification_checks=certification_checks,
    )
    write_generation_provenance(args, args.output, rows, episodes)
    summary = {
        "output_dir": str(args.output),
        "run_count": len(rows),
        "ok_count": sum(row["status"] == "ok" for row in rows),
        "failed_count": sum(row["status"] != "ok" for row in rows),
        "requested_run_count": args.runs,
        "gate": gate_summary,
        "retries": retry_summary,
    }
    print(json.dumps(summary, indent=2))
    return 0 if gate_summary["passed"] and len(rows) == args.runs else 2
