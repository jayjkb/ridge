"""Audit Stage-1 node features against emulation-host telemetry before freezing the schema."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ridge.common.contracts import (
    HOST_WIDE_NODE_TELEMETRY_FIELDS,
    STAGE2_NODE_DYNAMIC_FEATURES,
    STAGE2_NODE_STATIC_FEATURES,
)
from ridge.common.io import finite_float, read_csv_rows, write_json

FIELDS = HOST_WIDE_NODE_TELEMETRY_FIELDS
EXCLUDED_FROM_SCHEMA = (*HOST_WIDE_NODE_TELEMETRY_FIELDS, "Role_Tor")


def _close_to_host(node_value: float, host_value: float) -> bool:
    """Return whether a node value matches the emulation-host value within one percent or 1e-6."""
    tolerance = max(1e-6, abs(host_value) * 0.01)
    return abs(node_value - host_value) <= tolerance


def audit_dataset(dataset_root: Path) -> dict[str, Any]:
    """Test the machine-level node fields for equality with the emulation host and constancy across nodes, listing those to exclude."""
    manifest = read_csv_rows(dataset_root / "manifest.csv", required=True)
    stats: dict[str, dict[str, Any]] = {
        field: {
            "snapshot_count": 0,
            "cross_node_constant_count": 0,
            "host_comparison_count": 0,
            "host_match_count": 0,
            "values": [],
        }
        for field in FIELDS
    }
    topology_roles: set[str] = set()

    for manifest_row in manifest:
        if str(manifest_row.get("status", "")).strip().lower() != "ok":
            continue
        run_id = int(manifest_row["run_id"])
        raw_log_dir = str(manifest_row.get("log_dir", "")).strip()
        relative = Path(raw_log_dir) if raw_log_dir else Path(f"run_{run_id:06d}")
        run_dir = relative if relative.is_absolute() else dataset_root / relative
        node_rows = read_csv_rows(run_dir / "node_stats.csv", required=True)
        host_rows = read_csv_rows(run_dir / "host_stats.csv", required=True)
        host_by_snapshot = {int(row["SnapshotId"]): row for row in host_rows}
        nodes_by_snapshot: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in node_rows:
            nodes_by_snapshot[int(row["SnapshotId"])].append(row)
        for snapshot_id, rows in nodes_by_snapshot.items():
            host_row = host_by_snapshot.get(snapshot_id)
            for field in FIELDS:
                node_values = [
                    value for row in rows if (value := finite_float(row.get(field))) is not None
                ]
                if not node_values:
                    continue
                field_stats = stats[field]
                field_stats["snapshot_count"] += 1
                field_stats["values"].extend(node_values)
                if max(node_values) - min(node_values) <= 1e-9:
                    field_stats["cross_node_constant_count"] += 1
                host_value = finite_float(host_row.get(field)) if host_row is not None else None
                if host_value is not None:
                    field_stats["host_comparison_count"] += len(node_values)
                    field_stats["host_match_count"] += sum(
                        _close_to_host(value, host_value) for value in node_values
                    )
        for row in read_csv_rows(run_dir / "topology_nodes.csv", required=True):
            topology_roles.add(str(row.get("Role", "")).strip().lower())

    channels: dict[str, Any] = {}
    removal_candidates: list[str] = []
    for field, field_stats in stats.items():
        values = field_stats.pop("values")
        snapshots = int(field_stats["snapshot_count"])
        comparisons = int(field_stats["host_comparison_count"])
        constant_fraction = (
            field_stats["cross_node_constant_count"] / snapshots if snapshots else None
        )
        host_match_fraction = field_stats["host_match_count"] / comparisons if comparisons else None
        removal_candidates.append(field)
        channels[field] = {
            **field_stats,
            "cross_node_constant_fraction": constant_fraction,
            "host_match_fraction": host_match_fraction,
            "global_min": min(values) if values else None,
            "global_max": max(values) if values else None,
            "recommendation": "remove_namespace_global",
        }

    role_tor_present = "tor" in topology_roles
    if not role_tor_present:
        removal_candidates.append("Role_Tor")
    return {
        "dataset_root": str(dataset_root),
        "channels": channels,
        "topology_roles": sorted(topology_roles),
        "role_tor_present": role_tor_present,
        "removal_candidates": sorted(set(removal_candidates)),
        "excluded_from_v2_feature_schema": list(EXCLUDED_FROM_SCHEMA),
        "v2_node_dynamic_features": list(STAGE2_NODE_DYNAMIC_FEATURES),
        "v2_node_static_features": list(STAGE2_NODE_STATIC_FEATURES),
    }


def main(argv: list[str] | None = None) -> int:
    """Audit the dataset, write the report beside it or to the given path, and print it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_dataset(args.dataset_root.resolve())
    output = args.output or args.dataset_root / "feature_schema_audit.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
