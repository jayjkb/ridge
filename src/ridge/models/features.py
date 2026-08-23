"""Feature vocabulary and the recorded feature schema for Stages 2-5."""

from __future__ import annotations

from ridge.common.contracts import (
    FAULT_LABELS,
    RAW_TELEMETRY_UNITS,
    STAGE2_NODE_DYNAMIC_FEATURES,
    STAGE2_NODE_STATIC_FEATURES,
)

NODE_DYNAMIC_FEATURES = STAGE2_NODE_DYNAMIC_FEATURES
NODE_STATIC_FEATURES = STAGE2_NODE_STATIC_FEATURES
EDGE_DYNAMIC_FEATURES = (
    "TX_KBPS",
    "RX_KBPS",
    "TX_PacketsPerSec",
    "RX_PacketsPerSec",
    "TX_DropsPerSec",
    "RX_DropsPerSec",
    "TX_ErrorsPerSec",
    "RX_ErrorsPerSec",
    "EstimatedUtilization", # Estimated link utilization fraction.
    "QueueDrops",
    "QueueOverlimits", # Times the queue exceeded its configured limit.
    "QueueBacklogBytes",
    "QueueBacklogPackets",
    "QueueRequeues", # Packets requeued for another transmit attempt.
)
EDGE_STATIC_FEATURES = (
    "ConfiguredBandwidth",
    "Profile_Host",
    "Profile_Fabric",
    "Profile_WAN",
)
PROBE_CONTINUOUS_FEATURES = (
    "AvgRTT",
    "MaxRTT",
    "MdevRTT", # RTT variation reported by mdev.
    "PacketLoss", # Percentage of lost probe packets in the range 0-100.
)
PROBE_BINARY_FEATURES = (
    "TimeoutFlag", # Whether the probe timed out.
    "AvgRTT_Missing", # Whether AvgRTT is missing.
    "MaxRTT_Missing", # Whether MaxRTT is missing.
    "MdevRTT_Missing", # Whether MdevRTT is missing.
    "PacketLoss_Missing",  # Whether PacketLoss is missing.
    "NoProbe", # Whether no probe sample was collected.
)
PROBE_FEATURES = (*PROBE_CONTINUOUS_FEATURES, *PROBE_BINARY_FEATURES)
FAULT_CATEGORY_LABELS = FAULT_LABELS
FAULT_CATEGORY_TO_INDEX = {label: index for index, label in enumerate(FAULT_CATEGORY_LABELS)}
CANDIDATE_TYPE_TO_INDEX = {"none": 0, "node": 1, "edge": 2}


def feature_schema() -> dict[str, object]:
    """Return the feature names, units, index ranges, and normalization eligibility recorded in artifacts."""
    node_input_features = [*NODE_DYNAMIC_FEATURES, *NODE_STATIC_FEATURES]
    edge_input_features = [*EDGE_DYNAMIC_FEATURES, *EDGE_STATIC_FEATURES]
    node_units = {
        **{name: RAW_TELEMETRY_UNITS[name] for name in NODE_DYNAMIC_FEATURES},
        **{name: "binary" for name in NODE_STATIC_FEATURES},
    }
    edge_units = {
        **{name: RAW_TELEMETRY_UNITS.get(name, "count") for name in EDGE_DYNAMIC_FEATURES},
        "EstimatedUtilization": "fraction",
        "QueueDrops": "count",
        "QueueOverlimits": "count",
        "QueueBacklogBytes": "bytes",
        "QueueBacklogPackets": "count",
        "QueueRequeues": "count",
        "ConfiguredBandwidth": "megabits/second",
        "Profile_Host": "binary",
        "Profile_Fabric": "binary",
        "Profile_WAN": "binary",
    }
    probe_units = {
        "AvgRTT": "milliseconds",
        "MaxRTT": "milliseconds",
        "MdevRTT": "milliseconds",
        "PacketLoss": "percent_0_100",
        "TimeoutFlag": "binary",
        "AvgRTT_Missing": "binary",
        "MaxRTT_Missing": "binary",
        "MdevRTT_Missing": "binary",
        "PacketLoss_Missing": "binary",
        "NoProbe": "binary",
    }
    return {
        "node_input_features": node_input_features,
        "node_target_features": list(NODE_DYNAMIC_FEATURES),
        "edge_input_features": edge_input_features,
        "edge_target_features": list(EDGE_DYNAMIC_FEATURES),
        "probe_input_features": list(PROBE_FEATURES),
        "probe_target_features": list(PROBE_FEATURES),
        "probe_continuous_feature_indices": list(range(len(PROBE_CONTINUOUS_FEATURES))),
        "probe_binary_feature_indices": list(
            range(len(PROBE_CONTINUOUS_FEATURES), len(PROBE_FEATURES))
        ),
        "feature_units": {
            "node": node_units,
            "edge": edge_units,
            "probe": probe_units,
        },
        "normalization_eligible": {
            "node_input": list(NODE_DYNAMIC_FEATURES),
            "node_target": list(NODE_DYNAMIC_FEATURES),
            "edge_input": list(EDGE_DYNAMIC_FEATURES),
            "edge_target": list(EDGE_DYNAMIC_FEATURES),
            "probe_input": list(PROBE_CONTINUOUS_FEATURES),
            "probe_target": list(PROBE_CONTINUOUS_FEATURES),
        },
        "missingness": {
            "snapshot": (
                "Windows spanning a non-complete or non-consecutive SnapshotId are excluded."
            ),
            "node": "Complete snapshots require every topology node; values are not imputed.",
            "edge": (
                "Complete snapshots require both interfaces of every topology edge; "
                "endpoint values are aggregated deterministically."
            ),
            "probe": (
                "Missing continuous values are zero-filled only with their explicit *_Missing "
                "indicator; an absent pair sets NoProbe."
            ),
        },
    }
