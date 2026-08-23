"""Topology loading and per-episode graph tensor construction."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from ridge.common.contracts import canonical_edge_id
from ridge.common.io import as_float, read_csv_rows, read_json
from ridge.io.stage1_dataset import resolve_run_dir_values
from ridge.models.features import (
    EDGE_DYNAMIC_FEATURES,
    NODE_DYNAMIC_FEATURES,
    PROBE_CONTINUOUS_FEATURES,
    PROBE_FEATURES,
)


def _snapshot_id(row: dict[str, str]) -> str:
    """Return the mandatory stable join key written by Stage 1."""
    snapshot_id = str(row.get("SnapshotId", "")).strip()
    if not snapshot_id:
        raise ValueError("Stage-1 telemetry row is missing SnapshotId")
    if not str(row.get("Timestamp", "")).strip():
        raise ValueError("Stage-1 telemetry row is missing Timestamp")
    return snapshot_id


def _group_rows_by_snapshot(
    rows: Sequence[dict[str, str]],
) -> OrderedDict[str, list[dict[str, str]]]:
    """Group telemetry rows by snapshot identifier in first-seen order."""
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(_snapshot_id(row), []).append(row)
    return grouped


def _ordered_snapshot_keys(keys: Iterable[str]) -> list[str]:
    """Return the distinct snapshot keys, numeric ones first in numeric order."""
    def sort_key(key: str) -> tuple[int, int | str]:
        """Sort numeric keys ahead of the rest and by value."""
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    return sorted(dict.fromkeys(keys), key=sort_key)


def _static_node_features(node_type: str, role: str) -> list[float]:
    """Return the four binary static descriptors of a node from its type and role."""
    node_type = node_type.lower()
    role = role.lower()
    return [
        float(node_type == "host"),
        float(node_type in {"switch", "router"}),
        float(role == "host"),
        float(role == "spine"),
    ]


def _static_edge_features(profile: str, bandwidth_mbps: float) -> list[float]:
    """Return the bandwidth and three link tier indicators that form a link's static descriptors."""
    profile = profile.lower()
    return [
        float(bandwidth_mbps),
        float(profile == "host"),
        float(profile == "fabric"),
        float(profile == "wan"),
    ]


def _probe_feature_row(row: dict[str, str]) -> list[float]:
    """Build the probe feature vector with zero-filled missing measurements and their indicators."""
    values = []
    missing = []
    for name in PROBE_CONTINUOUS_FEATURES:
        value = as_float(row.get(name))
        if math.isfinite(value):
            values.append(value)
            missing.append(0.0)
        else:
            values.append(0.0)
            missing.append(1.0)
    timeout = as_float(row.get("TimeoutFlag"))
    values.extend([timeout if math.isfinite(timeout) else 0.0, *missing, 0.0])
    return values


def _safe_mean(values: Sequence[float]) -> float:
    """Return the mean of the finite values, or zero when none are finite."""
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0
    return float(sum(finite_values) / len(finite_values))


def _safe_max(values: Sequence[float]) -> float:
    """Return the maximum of the finite values, or zero when none are finite."""
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0
    return float(max(finite_values))


def _row_feature_vector(row: dict[str, str], feature_names: Sequence[str]) -> list[float]:
    """Read the named features of a row as floats, replacing non-finite values with zero."""
    values = []
    for name in feature_names:
        value = as_float(row.get(name))
        values.append(value if math.isfinite(value) else 0.0)
    return values


def _aggregate_queue_rows(
    snapshot_rows: Sequence[dict[str, str]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Reduce the qdisc rows of a snapshot to one maximum per node and interface."""
    feature_names = ("Drops", "Overlimits", "Backlog_Bytes", "Backlog_Packets", "Requeues")
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in snapshot_rows:
        interface = row.get("Interface", "")
        if interface == "lo":
            continue
        key = (row.get("Node", ""), interface)
        values = grouped.setdefault(key, {name: 0.0 for name in feature_names})
        for name in feature_names:
            values[name] = max(values[name], _safe_max([values[name], as_float(row.get(name))]))
    return grouped


def load_probe_ids_from_manifest_rows(
    dataset_dir: Path, manifest_rows: Sequence[dict[str, str]]
) -> list[str]:
    """Return the sorted probe identifiers seen in the probe telemetry of the listed episodes."""
    probe_ids: set[str] = set()
    for row in manifest_rows:
        run_dir = resolve_run_dir_values(
            dataset_dir,
            run_id=int(row["run_id"]),
            log_dir=str(row.get("log_dir", "") or ""),
        )
        for ping_row in read_csv_rows(run_dir / "ping_stats.csv"):
            probe_id = f"{ping_row.get('Source', '')}->{ping_row.get('Destination', '')}"
            probe_ids.add(probe_id)
    return sorted(probe_ids)


def topology_from_rows(
    node_rows: Sequence[dict[str, str]],
    link_rows: Sequence[dict[str, str]],
    probe_ids: Sequence[str],
    candidate_node_ids: Sequence[str],
    candidate_edge_ids: Sequence[str],
) -> dict[str, object]:
    """Build the topology with its edge index, static descriptors, probe endpoints, and the catalog of root-cause candidates."""
    node_ids = [row["Node"] for row in node_rows]
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    directed_edges: list[list[int]] = [[], []]
    edge_ids: list[str] = []
    edge_endpoints: list[list[int]] = []
    edge_static = []
    edge_lookup: dict[str, int] = {}
    for row in link_rows:
        src = row["Source"]
        dst = row["Destination"]
        edge_id = row.get("EdgeId") or canonical_edge_id(src, dst)
        src_index = node_to_index[src]
        dst_index = node_to_index[dst]
        directed_edges[0].extend((src_index, dst_index))
        directed_edges[1].extend((dst_index, src_index))
        edge_lookup[edge_id] = len(edge_ids)
        edge_ids.append(edge_id)
        edge_endpoints.append([src_index, dst_index])
        edge_static.append(
            _static_edge_features(
                row.get("Profile", ""),
                as_float(row.get("BandwidthMbps")),
            )
        )
    node_static = [
        _static_node_features(row.get("NodeType", ""), row.get("Role", "")) for row in node_rows
    ]
    probe_pairs = []
    for probe_id in probe_ids:
        src, dst = probe_id.split("->", 1)
        probe_pairs.append([node_to_index[src], node_to_index[dst]])
    unknown_nodes = sorted(set(candidate_node_ids) - set(node_to_index))
    unknown_edges = sorted(set(candidate_edge_ids) - set(edge_lookup))
    if unknown_nodes or unknown_edges:
        raise ValueError(
            "candidate catalog references entities absent from the topology: "
            f"nodes={unknown_nodes}, edges={unknown_edges}"
        )
    candidate_node_ids = list(candidate_node_ids)
    candidate_edge_ids = list(candidate_edge_ids)
    candidate_ids = ["none", *candidate_node_ids, *candidate_edge_ids]
    candidate_kinds = [
        "none",
        *(["node"] * len(candidate_node_ids)),
        *(["edge"] * len(candidate_edge_ids)),
    ]
    return {
        "node_ids": node_ids,
        "node_to_index": node_to_index,
        "edge_index": directed_edges,
        "edge_ids": edge_ids,
        "edge_id_to_index": edge_lookup,
        "edge_endpoints": edge_endpoints,
        "node_static_features": node_static,
        "edge_static_features": edge_static,
        "probe_ids": list(probe_ids),
        "probe_endpoints": probe_pairs,
        "candidate_ids": candidate_ids,
        "candidate_kinds": candidate_kinds,
        "candidate_to_index": {label: index for index, label in enumerate(candidate_ids)},
        "candidate_node_indices": [node_to_index[node_id] for node_id in candidate_node_ids],
        "candidate_edge_indices": [edge_lookup[edge_id] for edge_id in candidate_edge_ids],
    }


def load_topology(
    run_dir: Path,
    probe_ids: Sequence[str],
    candidate_node_ids: Sequence[str],
    candidate_edge_ids: Sequence[str],
) -> dict[str, object]:
    """Build the topology from the node and link tables of an episode directory."""
    return topology_from_rows(
        read_csv_rows(run_dir / "topology_nodes.csv"),
        read_csv_rows(run_dir / "topology_links.csv"),
        probe_ids,
        candidate_node_ids,
        candidate_edge_ids,
    )


def load_episode_graph(run_dir: Path, topology: dict[str, object]) -> dict[str, object]:
    """Align every telemetry family on snapshot identifier, check complete snapshots, and build the episode tensors."""
    node_rows = read_csv_rows(run_dir / "node_stats.csv")
    interface_rows = read_csv_rows(run_dir / "interface_stats.csv")
    ping_rows = read_csv_rows(run_dir / "ping_stats.csv")
    route_rows = read_csv_rows(run_dir / "route_stats.csv")
    neighbor_rows = read_csv_rows(run_dir / "neighbor_stats.csv")
    queue_rows = read_csv_rows(run_dir / "queue_stats.csv")
    timing_rows = read_csv_rows(run_dir / "telemetry_timing.csv")
    if not interface_rows:
        raise ValueError(f"{run_dir} did not contain interface telemetry")
    if not timing_rows:
        raise ValueError(f"{run_dir} did not contain required telemetry_timing.csv rows")

    telemetry_families = {
        "node": node_rows,
        "interface": interface_rows,
        "ping": ping_rows,
        "route": route_rows,
        "neighbor": neighbor_rows,
        "queue": queue_rows,
    }
    for family_name, rows in telemetry_families.items():
        if not rows:
            raise ValueError(f"{run_dir} has no {family_name} telemetry rows")
        missing_snapshot_ids = sum(not bool(str(row.get("SnapshotId", "")).strip()) for row in rows)
        if missing_snapshot_ids:
            raise ValueError(
                f"{run_dir}/{family_name}_stats.csv has {missing_snapshot_ids} rows "
                "without required SnapshotId"
            )

    node_groups = _group_rows_by_snapshot(node_rows)
    interface_groups = _group_rows_by_snapshot(interface_rows)
    ping_groups = _group_rows_by_snapshot(ping_rows)
    route_groups = _group_rows_by_snapshot(route_rows)
    neighbor_groups = _group_rows_by_snapshot(neighbor_rows)
    queue_groups = _group_rows_by_snapshot(queue_rows)
    grouped_families = {
        "node": node_groups,
        "interface": interface_groups,
        "route": route_groups,
        "neighbor": neighbor_groups,
        "queue": queue_groups,
        "probe": ping_groups,
    }
    timing_by_snapshot: dict[str, dict[str, str]] = {}
    for row in timing_rows:
        snapshot_id = str(row.get("SnapshotId", "")).strip()
        if not snapshot_id:
            raise ValueError(f"{run_dir}/telemetry_timing.csv has a row without SnapshotId")
        if snapshot_id in timing_by_snapshot:
            raise ValueError(
                f"{run_dir}/telemetry_timing.csv has duplicate SnapshotId={snapshot_id}"
            )
        status = str(row.get("Status", "")).strip().lower()
        if status not in {"complete", "partial", "error", "skipped"}:
            raise ValueError(
                f"{run_dir}/telemetry_timing.csv has invalid status={status!r} "
                f"for SnapshotId={snapshot_id}"
            )
        timing_by_snapshot[snapshot_id] = row

    snapshot_keys = _ordered_snapshot_keys(timing_by_snapshot)
    timing_snapshot_ids = set(snapshot_keys)
    for family_name, groups in grouped_families.items():
        unknown = sorted(set(groups) - timing_snapshot_ids)
        if unknown:
            raise ValueError(
                f"{run_dir}/{family_name}_stats.csv contains SnapshotIds absent from "
                f"telemetry_timing.csv: {unknown[:10]}"
            )

    steps = len(snapshot_keys)
    node_ids = list(topology["node_ids"])
    node_to_index = dict(topology["node_to_index"])
    edge_ids = list(topology["edge_ids"])
    edge_lookup = dict(topology["edge_id_to_index"])
    probe_ids = list(topology["probe_ids"])
    probe_lookup = {probe_id: index for index, probe_id in enumerate(probe_ids)}
    node_static = torch.tensor(topology["node_static_features"], dtype=torch.float32)
    edge_static = torch.tensor(topology["edge_static_features"], dtype=torch.float32)

    link_rows = read_csv_rows(run_dir / "topology_links.csv")
    expected_nodes = set(node_ids)
    expected_interfaces = {
        (str(row["Source"]), str(row["SourceInterface"])) for row in link_rows
    } | {(str(row["Destination"]), str(row["DestinationInterface"])) for row in link_rows}
    run_probe_ids = {f"{row.get('Source', '')}->{row.get('Destination', '')}" for row in ping_rows}
    metadata = read_json(run_dir / "run_metadata.json")
    expected_probe_count = metadata.get("ping_pair_count")
    if expected_probe_count not in (None, "") and len(run_probe_ids) != int(expected_probe_count):
        raise ValueError(
            f"{run_dir}/ping_stats.csv exposes {len(run_probe_ids)} probe pairs; "
            f"run_metadata.json records {expected_probe_count}"
        )

    snapshot_complete: list[bool] = []
    snapshot_timestamps: dict[str, str] = {}
    for snapshot_key in snapshot_keys:
        timing_row = timing_by_snapshot[snapshot_key]
        complete = str(timing_row.get("Status", "")).strip().lower() == "complete"
        snapshot_complete.append(complete)
        if not complete:
            snapshot_timestamps[snapshot_key] = ""
            continue
        for family_name, groups in grouped_families.items():
            if not groups.get(snapshot_key):
                raise ValueError(
                    f"{run_dir} marks SnapshotId={snapshot_key} complete but "
                    f"{family_name} telemetry is missing"
                )
        family_timestamps = {
            str(row.get("Timestamp", ""))
            for groups in grouped_families.values()
            for row in groups[snapshot_key]
        }
        if len(family_timestamps) != 1 or "" in family_timestamps:
            raise ValueError(
                f"{run_dir} has inconsistent timestamps for complete SnapshotId={snapshot_key}"
            )
        timestamp = next(iter(family_timestamps))
        timing_timestamp = str(timing_row.get("ActualStartTimestamp", "")).strip()
        if timing_timestamp != timestamp:
            raise ValueError(
                f"{run_dir} telemetry timestamp does not match timing artifact for "
                f"SnapshotId={snapshot_key}"
            )
        snapshot_timestamps[snapshot_key] = timestamp

        entity_sets = {
            "node": {str(row.get("Node", "")) for row in node_groups[snapshot_key]},
            "route": {str(row.get("Node", "")) for row in route_groups[snapshot_key]},
            "neighbor": {str(row.get("Node", "")) for row in neighbor_groups[snapshot_key]},
        }
        for family_name, actual_nodes in entity_sets.items():
            if actual_nodes != expected_nodes:
                raise ValueError(
                    f"{run_dir} complete SnapshotId={snapshot_key} has incorrect "
                    f"{family_name} node set"
                )
        actual_interfaces = {
            (str(row.get("Node", "")), str(row.get("Interface", "")))
            for row in interface_groups[snapshot_key]
        }
        if actual_interfaces != expected_interfaces:
            raise ValueError(
                f"{run_dir} complete SnapshotId={snapshot_key} has incorrect interface set"
            )
        actual_queue_interfaces = {
            (str(row.get("Node", "")), str(row.get("Interface", "")))
            for row in queue_groups[snapshot_key]
        }
        if not expected_interfaces.issubset(actual_queue_interfaces):
            raise ValueError(
                f"{run_dir} complete SnapshotId={snapshot_key} is missing queue entities"
            )
        actual_probe_ids = {
            f"{row.get('Source', '')}->{row.get('Destination', '')}"
            for row in ping_groups[snapshot_key]
        }
        if actual_probe_ids != run_probe_ids:
            raise ValueError(
                f"{run_dir} complete SnapshotId={snapshot_key} has incorrect probe pair set"
            )

    node_dynamic = torch.zeros(
        (steps, len(node_ids), len(NODE_DYNAMIC_FEATURES)), dtype=torch.float32
    )
    edge_dynamic = torch.zeros(
        (steps, len(edge_ids), len(EDGE_DYNAMIC_FEATURES)), dtype=torch.float32
    )
    probe_features = torch.zeros((steps, len(probe_ids), len(PROBE_FEATURES)), dtype=torch.float32)
    probe_features[..., -1] = 1.0
    timestamps: list[str] = []
    snapshot_ids: list[str] = []

    link_by_edge_id = {
        row.get("EdgeId") or canonical_edge_id(row["Source"], row["Destination"]): row
        for row in link_rows
    }

    for step, snapshot_key in enumerate(snapshot_keys):
        snapshot_rows = interface_groups.get(snapshot_key, [])
        timestamp = snapshot_timestamps[snapshot_key]
        timestamps.append(timestamp)
        snapshot_ids.append(snapshot_key)
        interface_by_key = {
            (row.get("Node", ""), row.get("Interface", "")): row for row in snapshot_rows
        }
        route_by_node = {row.get("Node", ""): row for row in route_groups.get(snapshot_key, [])}
        neighbor_by_node = {
            row.get("Node", ""): row for row in neighbor_groups.get(snapshot_key, [])
        }
        queue_by_key = _aggregate_queue_rows(queue_groups.get(snapshot_key, []))

        for row in node_groups.get(snapshot_key, []):
            node_index = node_to_index.get(row.get("Node", ""))
            if node_index is None:
                continue
            route_row = route_by_node.get(row.get("Node", ""), {})
            neighbor_row = neighbor_by_node.get(row.get("Node", ""), {})
            node_dynamic[step, node_index] = torch.tensor(
                _row_feature_vector(
                    {**row, **route_row, **neighbor_row},
                    NODE_DYNAMIC_FEATURES,
                ),
                dtype=torch.float32,
            )

        for edge_id, edge_index in edge_lookup.items():
            link_row = link_by_edge_id[edge_id]
            src_row = interface_by_key.get(
                (link_row["Source"], link_row.get("SourceInterface", "")), {}
            )
            dst_row = interface_by_key.get(
                (link_row["Destination"], link_row.get("DestinationInterface", "")), {}
            )
            src_queue = queue_by_key.get(
                (link_row["Source"], link_row.get("SourceInterface", "")), {}
            )
            dst_queue = queue_by_key.get(
                (link_row["Destination"], link_row.get("DestinationInterface", "")), {}
            )
            tx_kbps = _safe_mean(
                [as_float(src_row.get("TX_KBPS")), as_float(dst_row.get("TX_KBPS"))]
            )
            rx_kbps = _safe_mean(
                [as_float(src_row.get("RX_KBPS")), as_float(dst_row.get("RX_KBPS"))]
            )
            tx_pps = _safe_mean(
                [
                    as_float(src_row.get("TX_PacketsPerSec")),
                    as_float(dst_row.get("TX_PacketsPerSec")),
                ]
            )
            rx_pps = _safe_mean(
                [
                    as_float(src_row.get("RX_PacketsPerSec")),
                    as_float(dst_row.get("RX_PacketsPerSec")),
                ]
            )
            tx_drops = _safe_mean(
                [as_float(src_row.get("TX_DropsPerSec")), as_float(dst_row.get("TX_DropsPerSec"))]
            )
            rx_drops = _safe_mean(
                [as_float(src_row.get("RX_DropsPerSec")), as_float(dst_row.get("RX_DropsPerSec"))]
            )
            tx_errs = _safe_mean(
                [as_float(src_row.get("TX_ErrorsPerSec")), as_float(dst_row.get("TX_ErrorsPerSec"))]
            )
            rx_errs = _safe_mean(
                [as_float(src_row.get("RX_ErrorsPerSec")), as_float(dst_row.get("RX_ErrorsPerSec"))]
            )
            capacity = max(1e-6, as_float(link_row.get("BandwidthMbps")) * 1000.0)
            utilization = max(tx_kbps, rx_kbps) / capacity
            queue_drops = _safe_max(
                [as_float(src_queue.get("Drops")), as_float(dst_queue.get("Drops"))]
            )
            queue_overlimits = _safe_max(
                [as_float(src_queue.get("Overlimits")), as_float(dst_queue.get("Overlimits"))]
            )
            queue_backlog_bytes = _safe_max(
                [as_float(src_queue.get("Backlog_Bytes")), as_float(dst_queue.get("Backlog_Bytes"))]
            )
            queue_backlog_packets = _safe_max(
                [
                    as_float(src_queue.get("Backlog_Packets")),
                    as_float(dst_queue.get("Backlog_Packets")),
                ]
            )
            queue_requeues = _safe_max(
                [as_float(src_queue.get("Requeues")), as_float(dst_queue.get("Requeues"))]
            )
            edge_dynamic[step, edge_index] = torch.tensor(
                [
                    tx_kbps,
                    rx_kbps,
                    tx_pps,
                    rx_pps,
                    tx_drops,
                    rx_drops,
                    tx_errs,
                    rx_errs,
                    utilization,
                    queue_drops,
                    queue_overlimits,
                    queue_backlog_bytes,
                    queue_backlog_packets,
                    queue_requeues,
                ],
                dtype=torch.float32,
            )

        for row in ping_groups.get(snapshot_key, []):
            probe_id = f"{row.get('Source', '')}->{row.get('Destination', '')}"
            probe_index = probe_lookup.get(probe_id)
            if probe_index is None:
                continue
            probe_features[step, probe_index] = torch.tensor(
                _probe_feature_row(row), dtype=torch.float32
            )

    node_x = torch.cat([node_dynamic, node_static.unsqueeze(0).expand(steps, -1, -1)], dim=-1)
    edge_x = torch.cat([edge_dynamic, edge_static.unsqueeze(0).expand(steps, -1, -1)], dim=-1)
    return {
        "timestamps": timestamps,
        "snapshot_ids": snapshot_ids,
        "snapshot_complete": snapshot_complete,
        "node_x": node_x,
        "edge_x": edge_x,
        "probe_x": probe_features,
        "target_node_y": node_dynamic,
        "target_edge_y": edge_dynamic,
        "target_probe_y": probe_features,
    }


def json_topology(topology: dict[str, object]) -> dict[str, object]:
    """Return the serializable fields of a topology without its derived lookup maps."""
    return {
        "node_ids": topology["node_ids"],
        "edge_index": topology["edge_index"],
        "edge_ids": topology["edge_ids"],
        "edge_endpoints": topology["edge_endpoints"],
        "node_static_features": topology["node_static_features"],
        "edge_static_features": topology["edge_static_features"],
        "probe_ids": topology["probe_ids"],
        "probe_endpoints": topology["probe_endpoints"],
        "candidate_ids": topology["candidate_ids"],
        "candidate_kinds": topology["candidate_kinds"],
        "candidate_node_indices": topology["candidate_node_indices"],
        "candidate_edge_indices": topology["candidate_edge_indices"],
    }


def hydrate_topology(topology: dict[str, Any]) -> dict[str, Any]:
    """Restore the lookup maps of a topology loaded from JSON, checking its required fields."""
    hydrated = dict(topology)
    required = {
        "node_ids",
        "edge_ids",
        "candidate_ids",
        "candidate_kinds",
        "candidate_node_indices",
        "candidate_edge_indices",
    }
    missing = sorted(required - hydrated.keys())
    if missing:
        raise ValueError(f"topology artifact is missing required fields: {missing}")
    hydrated["node_to_index"] = {
        str(node_id): index for index, node_id in enumerate(hydrated["node_ids"])
    }
    hydrated["edge_id_to_index"] = {
        str(edge_id): index for index, edge_id in enumerate(hydrated["edge_ids"])
    }
    hydrated["candidate_to_index"] = {
        str(candidate_id): index for index, candidate_id in enumerate(hydrated["candidate_ids"])
    }
    return hydrated
