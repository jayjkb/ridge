"""Artifact contracts shared by the model pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = 2


class ArtifactValidationError(ValueError):
    """Raised when an artifact is missing required contract metadata."""


def artifact_metadata(artifact_type: str) -> dict[str, object]:
    """Build the required top-level metadata for a versioned artifact."""
    if not artifact_type:
        raise ValueError("artifact_type must be non-empty")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": artifact_type,
    }


def validate_artifact(
    payload: object,
    expected_type: str,
    *,
    source: str | Path,
) -> dict[str, Any]:
    """Validate and return a model artifact mapping."""
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{source} is not a JSON/object artifact")
    version = payload.get("schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"{source} uses unsupported schema_version={version!r}; "
            f"rebuild it with schema version {ARTIFACT_SCHEMA_VERSION}"
        )
    artifact_type = payload.get("artifact_type")
    if artifact_type != expected_type:
        raise ArtifactValidationError(
            f"{source} has artifact_type={artifact_type!r}; expected {expected_type!r}"
        )
    return payload


def require_matching_field(
    actual: object,
    expected: object,
    *,
    name: str,
    source: str | Path,
) -> None:
    """Reject an artifact pairing whose recorded semantics disagree."""
    if actual != expected:
        raise ArtifactValidationError(
            f"{source} has incompatible {name}={actual!r}; expected {expected!r}"
        )


TOPOLOGY_IDENTITY_FIELDS = (
    "node_ids",
    "edge_ids",
    "edge_index",
    "probe_ids",
    "candidate_ids",
    "candidate_kinds",
)


def topology_identity(topology: Mapping[str, object]) -> dict[str, object]:
    """Return the fields that make two topologies the same monitored network instance."""
    return {name: topology[name] for name in TOPOLOGY_IDENTITY_FIELDS if name in topology}


def require_compatible_normal_checkpoint(
    checkpoint: Mapping[str, object],
    normal_index: Mapping[str, object],
    *,
    source: str | Path,
    topology: Mapping[str, object] | None = None,
) -> None:
    """Reject a Stage-3 checkpoint that does not belong to this Stage-2 dataset."""
    for name in ("feature_schema", "history_len", "prediction_horizon"):
        require_matching_field(
            checkpoint.get(name),
            normal_index.get(name),
            name=name,
            source=source,
        )
    if topology is None:
        return
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ArtifactValidationError(f"{source} is missing model_config")
    checkpoint_topology = model_config.get("topology")
    if not isinstance(checkpoint_topology, dict):
        raise ArtifactValidationError(f"{source} is missing model_config.topology")
    require_matching_field(
        topology_identity(checkpoint_topology),
        topology_identity(topology),
        name="topology",
        source=source,
    )


RIDGE_COMPATIBILITY_FIELDS = (
    "residual_mode",
    "residual_history_len",
    "emulator_history_len",
    "prediction_horizon",
    "fault_category_labels",
)


def require_compatible_ridge_checkpoint(
    checkpoint: Mapping[str, object],
    residual_index: Mapping[str, object],
    *,
    source: str | Path,
    topology: Mapping[str, object] | None = None,
) -> None:
    """Reject a Stage-5 checkpoint that does not belong to this Stage-4 dataset.

    ``feature_schema`` is not compared: the Stage-4 index does not record one,
    and the sibling ``residual_feature_schema.json`` uses prefixed names that
    never match the checkpoint's base schema.
    """
    for name in RIDGE_COMPATIBILITY_FIELDS:
        require_matching_field(
            checkpoint.get(name),
            residual_index.get(name),
            name=name,
            source=source,
        )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ArtifactValidationError(f"{source} is missing model_config")
    labels = checkpoint.get("fault_category_labels")
    if isinstance(labels, list):
        require_matching_field(
            model_config.get("num_categories"),
            len(labels),
            name="model_config.num_categories",
            source=source,
        )
    if topology is None:
        return
    for name in ("candidate_ids", "candidate_kinds"):
        recorded = checkpoint.get(name)
        if recorded is None:
            continue
        require_matching_field(recorded, topology.get(name), name=name, source=source)


def require_valid_splits(
    splits: Mapping[str, object],
    *,
    source: str | Path,
    run_ids: Iterable[object] | None = None,
) -> None:
    """Reject split assignments that overlap, repeat an episode, or miss episodes."""
    members: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        values = splits.get(name)
        if not isinstance(values, list):
            raise ArtifactValidationError(f"{source} has a malformed {name!r} split")
        members[name] = {str(value) for value in values}
        if len(members[name]) != len(values):
            raise ArtifactValidationError(f"{source} repeats runs within the {name!r} split")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(members[left] & members[right])
        if shared:
            raise ArtifactValidationError(
                f"{source} assigns the same runs to {left!r} and {right!r}: {shared[:5]}"
            )
    if run_ids is None:
        return
    expected = {str(run_id) for run_id in run_ids}
    assigned = members["train"] | members["val"] | members["test"]
    if assigned != expected:
        raise ArtifactValidationError(
            f"{source} covers {len(assigned)} runs; the window index references {len(expected)}"
        )
