"""Data contracts shared across the RIDGE pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 2
STAGE1_ARTIFACT_TYPE = "ridge.stage1_dataset"
MAX_GENERATION_WORKERS = 16
STAGE1_SCHEDULING_POLICY = "monotonic_fixed_rate_fault_guard_v4"
STAGE1_PROBE_SCHEDULING_POLICY = "async_fixed_rate_prewarmed_v2"

FAULT_CATEGORY_NONE = "none"
FAULT_CATEGORY_DRAIN = "drain"
FAULT_CATEGORY_FIBER_CUT = "fiber_cut"
FAULT_CATEGORY_LINK_DEGRADATION = "link_degradation"
FAULT_CATEGORY_LINK_FLAP = "link_flap"
FAULT_CATEGORIES = (
    FAULT_CATEGORY_DRAIN,
    FAULT_CATEGORY_FIBER_CUT,
    FAULT_CATEGORY_LINK_DEGRADATION,
    FAULT_CATEGORY_LINK_FLAP,
)
FAULT_LABELS = (FAULT_CATEGORY_NONE, *FAULT_CATEGORIES)


def canonical_edge_id(src: str, dst: str) -> str:
    """Canonical undirected link identifier: sorted endpoints joined by '<->'."""
    ordered = sorted((src, dst))
    return f"{ordered[0]}<->{ordered[1]}"


RIDGE_EARLY_STOPPING_METRICS = (
    "fault_present_f1",
    "candidate_top1_accuracy",
    "candidate_top3_accuracy",
    "mrr",
    "category_macro_f1",
    "joint_top1_entity_and_category_accuracy",
)

TELEMETRY_FAMILIES = (
    "node",
    "interface",
    "queue",
    "route",
    "neighbor",
    "probe",
)

# Mininet creates network namespaces, not independent PID/mount namespaces.
# These /proc-derived fields therefore describe the emulation host and must
# never be presented to the model as node-local features.
HOST_WIDE_NODE_TELEMETRY_FIELDS = (
    "CPUPercent",
    "LoadAvg1",
    "LoadAvg5",
    "MemoryPercent",
    "ProcCount",
)

STAGE2_NODE_DYNAMIC_FEATURES = (
    "RouteCount",
    "DefaultRouteCount",
    "HostPrefixRouteCount",
    "KernelRouteCount",
    "StaticRouteCount",
    "OspfRouteCount",
    "NeighborCount",
    "ReachableCount",
    "StaleCount",
    "FailedCount",
)

STAGE2_NODE_STATIC_FEATURES = (
    "NodeType_Host",
    "NodeType_Infrastructure",
    "Role_Host",
    "Role_Spine",
)

RAW_TELEMETRY_UNITS = {
    "SnapshotId": "index",
    "Timestamp": "ISO-8601 UTC",
    "CPUPercent": "percent",
    "LoadAvg1": "runnable-process average",
    "LoadAvg5": "runnable-process average",
    "MemoryPercent": "percent",
    "ProcCount": "count",
    "TX_Bytes": "bytes",
    "RX_Bytes": "bytes",
    "TX_Packets": "count",
    "RX_Packets": "count",
    "TX_Errors": "count",
    "RX_Errors": "count",
    "TX_Drops": "count",
    "RX_Drops": "count",
    "TX_KBPS": "kilobits/second",
    "RX_KBPS": "kilobits/second",
    "TX_PacketsPerSec": "packets/second",
    "RX_PacketsPerSec": "packets/second",
    "TX_DropsPerSec": "packets/second",
    "RX_DropsPerSec": "packets/second",
    "TX_ErrorsPerSec": "errors/second",
    "RX_ErrorsPerSec": "errors/second",
    "Bytes": "bytes",
    "Packets": "count",
    "Drops": "count",
    "Overlimits": "count",
    "Backlog_Bytes": "bytes",
    "Backlog_Packets": "count",
    "Requeues": "count",
    "RouteCount": "count",
    "DefaultRouteCount": "count",
    "HostPrefixRouteCount": "count",
    "KernelRouteCount": "count",
    "StaticRouteCount": "count",
    "OspfRouteCount": "count",
    "NeighborCount": "count",
    "ReachableCount": "count",
    "StaleCount": "count",
    "FailedCount": "count",
    "MinRTT": "milliseconds",
    "AvgRTT": "milliseconds",
    "MaxRTT": "milliseconds",
    "MdevRTT": "milliseconds",
    "PacketLoss": "percent",
    "TimeoutFlag": "binary",
}

_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "n", "off", ""})


def parse_bool(value: object, *, field: str = "value") -> bool:
    """Parse a serialized boolean without Python's truthy-string ambiguity."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{field} must be a boolean value, got {value!r}")


def require_artifact_contract(
    payload: Mapping[str, Any],
    *,
    artifact_type: str,
    schema_version: int = SCHEMA_VERSION,
) -> None:
    """Reject unversioned or incompatible artifacts with an actionable error."""
    actual_version = payload.get("schema_version")
    actual_type = payload.get("artifact_type")
    if actual_version != schema_version:
        raise ValueError(
            f"Unsupported schema_version={actual_version!r}; expected {schema_version}. "
            "Rebuild this artifact with the current RIDGE pipeline."
        )
    if actual_type != artifact_type:
        raise ValueError(f"Expected artifact_type={artifact_type!r}, got {actual_type!r}")
