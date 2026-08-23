"""Residual-dataset construction and graph batch data structures."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from ridge.common.io import (
    read_csv_rows,
    read_json,
    write_csv_rows,
    write_json,
)
from ridge.models.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    artifact_metadata,
    require_compatible_normal_checkpoint,
    require_valid_splits,
    validate_artifact,
)
from ridge.models.dataset_io import (
    DEFAULT_SHARD_CACHE_SIZE,
    NORMAL_RUNS_DIRNAME,
    RESIDUAL_RUNS_DIRNAME,
    RESIDUAL_WINDOW_INDEX,
    apply_feature_stats,
    finalize_feature_stats_accumulator,
    group_rows_by_run,
    has_normal_run_shards,
    load_cached_run_shard,
    load_normal_index,
    load_residual_index,
    load_window_index,
    new_feature_stats_accumulator,
    parse_window_index_int,
    require_index_int,
    run_ids_from_splits,
    update_feature_stats_accumulator,
    validate_window_sample_count,
    window_index_error,
)
from ridge.models.episode_graph import (
    hydrate_topology,
    json_topology,
)
from ridge.models.features import (
    CANDIDATE_TYPE_TO_INDEX,
    EDGE_DYNAMIC_FEATURES,
    EDGE_STATIC_FEATURES,
    FAULT_CATEGORY_LABELS,
    FAULT_CATEGORY_TO_INDEX,
    NODE_DYNAMIC_FEATURES,
    NODE_STATIC_FEATURES,
    PROBE_CONTINUOUS_FEATURES,
    PROBE_FEATURES,
)
from ridge.models.normal_data import (
    episode_split_label,
    window_fault_state,
)
from ridge.models.normal_model import (
    NormalTemporalGraphEmulator,
    make_normal_model_from_checkpoint,
)

RESIDUAL_WINDOW_INDEX_COLUMNS = (
    "run_id",
    "history_len",
    "window_end_index",
    "snapshot_id",
    "timestamp",
)


def _residual_series_for_run(
    normal_shard: dict[str, object],
    model: NormalTemporalGraphEmulator | None,
    emulator_history_len: int,
    residual_mode: str,
    inference_batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Residualize one Stage-2 shard without retaining any other episode in memory."""
    node_x = normal_shard["node_x"]
    edge_x = normal_shard["edge_x"]
    probe_x = normal_shard["probe_x"]
    node_y = normal_shard["target_node_y"]
    edge_y = normal_shard["target_edge_y"]
    probe_y = normal_shard["target_probe_y"]
    steps = int(node_x.shape[0])
    if "eligible_prediction_indices" not in normal_shard:
        raise ValueError("normal shard is missing eligible_prediction_indices")
    target_indices = [int(index) for index in normal_shard["eligible_prediction_indices"]]
    timestamps = [str(normal_shard["timestamps"][index]) for index in target_indices]
    shard_snapshot_ids = list(normal_shard.get("snapshot_ids", [""] * steps))
    snapshot_ids = [str(shard_snapshot_ids[index]) for index in target_indices]

    if not target_indices:
        return {
            "timestamps": [],
            "snapshot_ids": [],
            "node_residual": node_y[:0].float(),
            "edge_residual": edge_y[:0].float(),
            "probe_residual": probe_y[:0].float(),
        }

    if residual_mode == "raw":
        return {
            "timestamps": timestamps,
            "snapshot_ids": snapshot_ids,
            "node_residual": node_y[target_indices].float(),
            "edge_residual": edge_y[target_indices].float(),
            "probe_residual": probe_y[target_indices].float(),
        }

    if model is None:
        raise ValueError("standardized residuals require a normal emulator")
    node_chunks: list[torch.Tensor] = []
    edge_chunks: list[torch.Tensor] = []
    probe_chunks: list[torch.Tensor] = []
    for chunk_start in range(0, len(target_indices), inference_batch_size):
        chunk_indices = target_indices[chunk_start : chunk_start + inference_batch_size]
        batch_node = torch.stack(
            [node_x[index - emulator_history_len : index] for index in chunk_indices]
        ).to(device)
        batch_edge = torch.stack(
            [edge_x[index - emulator_history_len : index] for index in chunk_indices]
        ).to(device)
        batch_probe = torch.stack(
            [probe_x[index - emulator_history_len : index] for index in chunk_indices]
        ).to(device)
        with torch.no_grad():
            outputs = model(batch_node, batch_edge, batch_probe)
        target_node = node_y[chunk_indices].to(device)
        target_edge = edge_y[chunk_indices].to(device)
        target_probe = probe_y[chunk_indices].to(device)
        node_chunks.append(
            (
                (target_node - outputs["node_mean"])
                / torch.exp(outputs["node_log_std"]).clamp_min(1e-6)
            )
            .detach()
            .cpu()
        )
        edge_chunks.append(
            (
                (target_edge - outputs["edge_mean"])
                / torch.exp(outputs["edge_log_std"]).clamp_min(1e-6)
            )
            .detach()
            .cpu()
        )
        probe_chunks.append(
            torch.cat(
                [
                    (target_probe[..., : len(PROBE_CONTINUOUS_FEATURES)] - outputs["probe_mean"])
                    / torch.exp(outputs["probe_log_std"]).clamp_min(1e-6),
                    target_probe[..., len(PROBE_CONTINUOUS_FEATURES) :]
                    - torch.sigmoid(outputs["probe_binary_logits"]),
                ],
                dim=-1,
            )
            .detach()
            .cpu()
        )
    return {
        "timestamps": timestamps,
        "snapshot_ids": snapshot_ids,
        "node_residual": torch.cat(node_chunks).float(),
        "edge_residual": torch.cat(edge_chunks).float(),
        "probe_residual": torch.cat(probe_chunks).float(),
    }


def _validate_fault_target(
    row: dict[str, str],
    topology: dict[str, object],
    fault_present: int,
) -> tuple[str, str, int]:
    """Return the root cause identifier, its candidate type, and catalog index, or the no-fault entry."""
    if not fault_present:
        return "none", "none", 0
    root_id = str(row.get("root_cause_id", "")).strip()
    root_kind = str(row.get("root_cause_kind", "")).strip()
    if root_id not in topology["candidate_to_index"]:
        raise ValueError(f"run_id={row.get('run_id')} has unknown root_cause_id={root_id!r}")
    candidate_index = int(topology["candidate_to_index"][root_id])
    expected_kind = str(topology["candidate_kinds"][candidate_index])
    if root_kind != expected_kind:
        raise ValueError(
            f"run_id={row.get('run_id')} root_cause_kind={root_kind!r} does not match "
            f"candidate kind {expected_kind!r} for {root_id!r}"
        )
    return root_id, root_kind, candidate_index


def build_residual_dataset(
    dataset_dir: Path,
    normal_data_dir: Path,
    normal_emulator_path: Path | None,
    output_dir: Path,
    residual_history_len: int,
    residual_mode: str,
    *,
    inference_batch_size: int = 256,
    device: str = "cpu",
) -> dict[str, object]:
    """Build a residual dataset while preserving the Stage-2 split lineage."""
    if residual_mode not in ("standardized", "raw"):
        raise ValueError("residual_mode must be standardized or raw")
    if residual_history_len < 1:
        raise ValueError("residual_history_len must be at least 1")
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be at least 1")
    if not has_normal_run_shards(normal_data_dir):
        raise ValueError("Residual builder requires sharded normal dataset artifacts")

    normal_index = load_normal_index(normal_data_dir)
    emulator_history_len = int(normal_index["history_len"])
    prediction_horizon = int(normal_index["prediction_horizon"])
    if prediction_horizon != 1:
        raise ValueError(
            "Residual construction requires a one-step normal-emulator prediction horizon. "
            f"Stage 2 records prediction_horizon={prediction_horizon}"
        )
    normal_splits = read_json(normal_data_dir / "normal_splits.json")
    splits = {
        name: [str(run_id) for run_id in normal_splits[name]] for name in ("train", "val", "test")
    }
    require_valid_splits(splits, source=normal_data_dir / "normal_splits.json")
    topology = hydrate_topology(read_json(normal_data_dir / "topology.json"))
    model: NormalTemporalGraphEmulator | None = None
    device_object = torch.device(device)
    if residual_mode == "standardized":
        if normal_emulator_path is None:
            raise ValueError("normal_emulator_path is required for residual_mode=standardized")
        checkpoint = torch.load(
            normal_emulator_path, map_location=device_object, weights_only=False
        )
        validate_artifact(checkpoint, "normal_checkpoint", source=normal_emulator_path)
        require_compatible_normal_checkpoint(
            checkpoint,
            normal_index,
            source=normal_emulator_path,
            topology=topology,
        )
        if int(checkpoint.get("history_len", -1)) != emulator_history_len:
            raise ValueError("normal checkpoint history_len does not match its Stage-2 dataset")
        if int(checkpoint.get("prediction_horizon", -1)) != prediction_horizon:
            raise ValueError(
                "normal checkpoint prediction_horizon does not match its Stage-2 dataset"
            )
        model = make_normal_model_from_checkpoint(checkpoint, device=device)

    stage2_run_ids = run_ids_from_splits(splits)
    manifest_by_run: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(dataset_dir / "manifest.csv"):
        run_id = str(row.get("run_id", ""))
        if run_id not in stage2_run_ids:
            continue
        if run_id in manifest_by_run:
            raise ValueError(f"manifest contains duplicate run_id={run_id}")
        if row.get("status", "ok") != "ok":
            raise ValueError(
                f"Stage-2 run_id={run_id} is no longer marked status=ok in the source manifest"
            )
        episode_split_label(row)
        manifest_by_run[run_id] = row
    missing_manifest_runs = sorted(stage2_run_ids - set(manifest_by_run))
    if missing_manifest_runs:
        raise ValueError(
            f"source manifest is missing Stage-2 run IDs: {missing_manifest_runs[:10]}"
        )
    manifest_rows = [manifest_by_run[run_id] for run_id in sorted(stage2_run_ids, key=int)]

    output_dir.mkdir(parents=True, exist_ok=True)
    provisional_dir = output_dir / ".provisional_residual_runs"
    provisional_dir.mkdir(parents=True, exist_ok=True)
    node_stats = new_feature_stats_accumulator(range(len(NODE_DYNAMIC_FEATURES)))
    edge_stats = new_feature_stats_accumulator(range(len(EDGE_DYNAMIC_FEATURES)))
    probe_stats = new_feature_stats_accumulator(range(len(PROBE_CONTINUOUS_FEATURES)))
    train_run_ids = set(splits["train"])

    for row in manifest_rows:
        run_id = str(row["run_id"])
        shard_path = normal_data_dir / NORMAL_RUNS_DIRNAME / f"run_{int(run_id):06d}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(
                f"Stage 2 normal shard missing for run_id={run_id}: expected {shard_path}"
            )
        normal_shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        validate_artifact(normal_shard, "normal_run_shard", source=shard_path)
        series = _residual_series_for_run(
            normal_shard,
            model,
            emulator_history_len,
            residual_mode,
            inference_batch_size,
            device_object,
        )
        torch.save(series, provisional_dir / f"run_{int(run_id):06d}.pt")
        if run_id in train_run_ids:
            update_feature_stats_accumulator(node_stats, series["node_residual"])
            update_feature_stats_accumulator(edge_stats, series["edge_residual"])
            update_feature_stats_accumulator(probe_stats, series["probe_residual"])

    residual_normalization = {
        "node_residual": finalize_feature_stats_accumulator(node_stats),
        "edge_residual": finalize_feature_stats_accumulator(edge_stats),
        "probe_residual": finalize_feature_stats_accumulator(probe_stats),
        "fitted_on": "train_split_residual_snapshots",
    }
    feature_prefix = "Residual_" if residual_mode == "standardized" else "Raw_"
    residual_feature_schema = {
        "node_input_features": [
            *(f"{feature_prefix}{name}" for name in NODE_DYNAMIC_FEATURES),
            *NODE_STATIC_FEATURES,
        ],
        "edge_input_features": [
            *(f"{feature_prefix}{name}" for name in EDGE_DYNAMIC_FEATURES),
            *EDGE_STATIC_FEATURES,
        ],
        "probe_input_features": [f"{feature_prefix}{name}" for name in PROBE_FEATURES],
    }
    residual_metadata = artifact_metadata("residual_dataset")

    residual_runs_dir = output_dir / RESIDUAL_RUNS_DIRNAME
    residual_runs_dir.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    window_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        run_id = str(row["run_id"])
        provisional_path = provisional_dir / f"run_{int(run_id):06d}.pt"
        series = torch.load(provisional_path, map_location="cpu", weights_only=False)
        timestamps = list(series["timestamps"])
        snapshot_ids = list(series["snapshot_ids"])
        node_residual = apply_feature_stats(
            series["node_residual"], residual_normalization["node_residual"]
        )
        edge_residual = apply_feature_stats(
            series["edge_residual"], residual_normalization["edge_residual"]
        )
        probe_residual = apply_feature_stats(
            series["probe_residual"], residual_normalization["probe_residual"]
        )
        torch.save(
            {
                **residual_metadata,
                "artifact_type": "residual_run_shard",
                "run_id": run_id,
                "timestamps": timestamps,
                "snapshot_ids": snapshot_ids,
                "node_residual": node_residual.float(),
                "edge_residual": edge_residual.float(),
                "probe_residual": probe_residual.float(),
            },
            residual_runs_dir / f"run_{int(run_id):06d}.pt",
        )
        provisional_path.unlink()
        if len(timestamps) < residual_history_len:
            continue
        for index in range(residual_history_len - 1, len(timestamps)):
            window_snapshot_ids = snapshot_ids[index - residual_history_len + 1 : index + 1]
            try:
                numeric_snapshot_ids = [int(str(value)) for value in window_snapshot_ids]
            except (TypeError, ValueError) as exc:
                raise ValueError("residual SnapshotId values must be integers") from exc
            if numeric_snapshot_ids != list(
                range(
                    numeric_snapshot_ids[0],
                    numeric_snapshot_ids[0] + residual_history_len,
                )
            ):
                continue
            timestamp = str(timestamps[index])
            fault_present, category_label = window_fault_state(timestamp, row)
            root_id, root_kind, root_candidate_index = _validate_fault_target(
                row, topology, fault_present
            )
            window_rows.append(
                {
                    "run_id": run_id,
                    "history_len": residual_history_len,
                    "window_end_index": index,
                    "snapshot_id": str(snapshot_ids[index]),
                    "timestamp": timestamp,
                    "root_candidate_index": root_candidate_index,
                    "fault_present": float(fault_present),
                    "fault_category_index": FAULT_CATEGORY_TO_INDEX[category_label],
                    "fault_category": category_label,
                    "root_cause_kind": root_kind,
                    "root_cause_id": root_id,
                }
            )
            sample_count += 1
    provisional_dir.rmdir()

    write_csv_rows(output_dir / RESIDUAL_WINDOW_INDEX, window_rows)
    write_json(
        output_dir / "residual_index.json",
        {
            **residual_metadata,
            "sample_count": sample_count,
            "fault_category_labels": list(FAULT_CATEGORY_LABELS),
            "residual_mode": residual_mode,
            "source_dataset": str(dataset_dir.resolve()),
            "source_normal_data_dir": str(normal_data_dir.resolve()),
            "normal_emulator_checkpoint": (
                str(Path(normal_emulator_path).resolve()) if normal_emulator_path else None
            ),
            "split_run_counts": {name: len(run_ids) for name, run_ids in splits.items()},
            "candidate_count": len(topology["candidate_ids"]),
            "emulator_history_len": emulator_history_len,
            "prediction_horizon": prediction_horizon,
            "residual_history_len": residual_history_len,
        },
    )
    write_json(output_dir / "residual_splits.json", splits)
    write_json(output_dir / "residual_normalization.json", residual_normalization)
    write_json(output_dir / "topology.json", json_topology(topology))
    write_json(
        output_dir / "candidate_index.json",
        {
            "candidate_ids": topology["candidate_ids"],
            "candidate_kinds": topology["candidate_kinds"],
        },
    )
    write_json(output_dir / "residual_feature_schema.json", residual_feature_schema)
    return {
        "sample_count": sample_count,
        "split_sizes": {name: len(run_ids) for name, run_ids in splits.items()},
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "emulator_history_len": emulator_history_len,
        "residual_history_len": residual_history_len,
    }


class ResidualWindowDataset(Dataset):
    """Dataset of residual histories served from per-episode shards, with static descriptors appended to node and edge inputs."""

    def __init__(
        self,
        data_dir: Path,
        run_ids: Sequence[str],
        topology: dict[str, Any],
        cache_size: int = DEFAULT_SHARD_CACHE_SIZE,
    ) -> None:
        self.data_dir = data_dir
        self.topology = topology
        self._index_path = data_dir / RESIDUAL_WINDOW_INDEX
        index = load_residual_index(data_dir)
        self._index_history_len = require_index_int(
            index, "residual_history_len", index_path=self._index_path
        )
        all_rows = load_window_index(self._index_path)
        validate_window_sample_count(index, all_rows, index_path=self._index_path)
        selected = set(str(run_id) for run_id in run_ids)
        self.rows = [row for row in all_rows if str(row["run_id"]) in selected]
        self._rows_by_run = group_rows_by_run(self.rows)
        self._validated_runs: set[str] = set()
        self.cache_size = max(0, int(cache_size))
        self._run_cache: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._node_static = torch.tensor(topology["node_static_features"], dtype=torch.float32)
        self._edge_static = torch.tensor(topology["edge_static_features"], dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of windows selected from the index."""
        return len(self.rows)

    def _validate_run_windows(self, run_id: str, shard: dict[str, object]) -> None:
        """Validate every selected residual window row for ``run_id`` against its shard."""
        index_path = self._index_path
        shard_run_id = str(shard.get("run_id"))
        if shard_run_id != run_id:
            raise window_index_error(
                index_path, run_id, "run_id", f"shard run_id {shard_run_id!r} does not match"
            )
        timestamps = shard["timestamps"]
        snapshot_ids = shard["snapshot_ids"]
        lengths = {
            key: int(shard[key].shape[0])
            for key in ("node_residual", "edge_residual", "probe_residual")
        }
        metadata_len = min(len(timestamps), len(snapshot_ids))
        for row in self._rows_by_run.get(run_id, ()):
            for column in RESIDUAL_WINDOW_INDEX_COLUMNS:
                if column not in row:
                    raise window_index_error(index_path, run_id, "missing_column", column)
            if str(row["run_id"]) != run_id:
                raise window_index_error(
                    index_path, run_id, "run_id", f"row run_id {row['run_id']!r} misfiled"
                )
            history_len = parse_window_index_int(
                row, "history_len", index_path=index_path, run_id=run_id
            )
            window_end_index = parse_window_index_int(
                row, "window_end_index", index_path=index_path, run_id=run_id
            )
            if history_len != self._index_history_len:
                raise window_index_error(
                    index_path,
                    run_id,
                    "history_len",
                    f"{history_len} != residual_index {self._index_history_len}",
                )
            window_start = window_end_index - history_len + 1
            if window_start < 0:
                raise window_index_error(
                    index_path,
                    run_id,
                    "window_start",
                    f"window_end_index {window_end_index} < history_len {history_len} - 1",
                )
            for key in ("node_residual", "edge_residual", "probe_residual"):
                if lengths[key] <= window_end_index:
                    raise window_index_error(
                        index_path,
                        run_id,
                        "residual_bounds",
                        f"{key} length {lengths[key]} <= window_end_index {window_end_index}",
                    )
            if metadata_len <= window_end_index:
                raise window_index_error(
                    index_path,
                    run_id,
                    "metadata_bounds",
                    f"metadata length {metadata_len} <= window_end_index {window_end_index}",
                )
            if str(snapshot_ids[window_end_index]) != str(row["snapshot_id"]):
                raise window_index_error(
                    index_path,
                    run_id,
                    "snapshot_id",
                    f"{row['snapshot_id']!r} != shard {snapshot_ids[window_end_index]!r}",
                )
            if str(timestamps[window_end_index]) != str(row["timestamp"]):
                raise window_index_error(
                    index_path,
                    run_id,
                    "timestamp",
                    f"{row['timestamp']!r} != shard {timestamps[window_end_index]!r}",
                )
            window_indices = range(window_start, window_end_index + 1)
            try:
                window_snapshot_ids = [int(str(snapshot_ids[i])) for i in window_indices]
            except ValueError as exc:
                raise window_index_error(
                    index_path,
                    run_id,
                    "snapshot_id_parse",
                    f"window [{window_start},{window_end_index}] has non-integer SnapshotIds",
                ) from exc
            if window_snapshot_ids != list(
                range(window_snapshot_ids[0], window_snapshot_ids[0] + len(window_snapshot_ids))
            ):
                raise window_index_error(
                    index_path,
                    run_id,
                    "snapshot_consecutive",
                    f"window [{window_start},{window_end_index}] spans non-consecutive SnapshotIds",
                )
        self._validated_runs.add(run_id)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one residual history with static descriptors appended and its window labels."""
        row = self.rows[index]
        run_id = str(row["run_id"])
        shard_path = self.data_dir / RESIDUAL_RUNS_DIRNAME / f"run_{int(run_id):06d}.pt"
        shard = load_cached_run_shard(
            cache=self._run_cache,
            cache_size=self.cache_size,
            validated_runs=self._validated_runs,
            validate_run_windows=self._validate_run_windows,
            run_id=run_id,
            shard_path=shard_path,
            artifact_type="residual_run_shard",
        )
        history_len = int(row["history_len"])
        window_end_index = int(row["window_end_index"])
        window_start = window_end_index - history_len + 1
        node_residual = shard["node_residual"][window_start : window_end_index + 1].clone()
        edge_residual = shard["edge_residual"][window_start : window_end_index + 1].clone()
        probe_residual = shard["probe_residual"][window_start : window_end_index + 1].clone()
        return {
            "run_id": run_id,
            "timestamp": str(row["timestamp"]),
            "node_x": torch.cat(
                [node_residual, self._node_static.unsqueeze(0).expand(history_len, -1, -1)], dim=-1
            ),
            "edge_x": torch.cat(
                [edge_residual, self._edge_static.unsqueeze(0).expand(history_len, -1, -1)], dim=-1
            ),
            "probe_x": probe_residual,
            "root_candidate_index": int(row["root_candidate_index"]),
            "fault_present": float(row["fault_present"]),
            "fault_category_index": int(row["fault_category_index"]),
            "fault_category": str(row["fault_category"]),
            "root_cause_kind": str(row["root_cause_kind"]),
            "root_cause_id": str(row["root_cause_id"]),
        }


@dataclass
class ResidualGraphSample:
    """One residual history with its topology tensors, catalog, and labels, as the RCA model consumes it."""

    run_id: str
    timestamp: str
    x_node: torch.Tensor
    x_edge: torch.Tensor
    x_probe: torch.Tensor
    edge_index: torch.Tensor
    edge_endpoints: torch.Tensor
    probe_endpoints: torch.Tensor
    candidate_ids: list[str]
    candidate_types: list[str]
    candidate_entity_indices: torch.Tensor
    candidate_label: int
    category_label: int
    fault_present: float
    fault_category: str
    root_cause_kind: str
    root_cause_id: str
    y_node: torch.Tensor
    y_edge: torch.Tensor
    y_probe: torch.Tensor


@dataclass
class ResidualGraphBatch:
    """Mini-batch of residual histories with entities concatenated and per-graph assignment vectors."""

    x_node: torch.Tensor
    x_edge: torch.Tensor
    x_probe: torch.Tensor
    edge_index: torch.Tensor
    edge_endpoints: torch.Tensor
    probe_endpoints: torch.Tensor
    node_batch: torch.Tensor
    edge_batch: torch.Tensor
    probe_batch: torch.Tensor
    candidate_ptr: torch.Tensor
    candidate_batch: torch.Tensor
    candidate_types: torch.Tensor
    candidate_entity_indices: torch.Tensor
    candidate_labels: torch.Tensor
    category_labels: torch.Tensor
    y_node: torch.Tensor
    y_edge: torch.Tensor
    y_probe: torch.Tensor
    run_id: list[str]
    timestamp: list[str]
    candidate_ids: list[list[str]]
    candidate_type_labels: list[list[str]]
    root_cause_kind: list[str]
    root_cause_id: list[str]
    fault_category_label: list[str]

    @property
    def num_graphs(self) -> int:
        """Return the number of windows in the batch."""
        return int(self.candidate_labels.shape[0])

    @property
    def total_nodes(self) -> int:
        """Return the number of nodes concatenated across the batch."""
        return int(self.x_node.shape[1])

    @property
    def total_edges(self) -> int:
        """Return the number of edges concatenated across the batch."""
        return int(self.x_edge.shape[1])

    @property
    def total_probes(self) -> int:
        """Return the number of probes concatenated across the batch."""
        return int(self.x_probe.shape[1])

    @property
    def total_candidates(self) -> int:
        """Return the number of candidates concatenated across the batch."""
        return int(self.candidate_types.shape[0])

    def to(self, device: torch.device | str) -> "ResidualGraphBatch":
        """Return a copy of the batch with every tensor moved to the device."""
        tensor_fields = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        payload = dict(self.__dict__)
        payload.update(tensor_fields)
        return ResidualGraphBatch(**payload)


class ResidualGraphWindowDataset(Dataset):
    """Dataset that wraps residual windows as graph samples carrying the topology and the catalog."""
    
    def __init__(
        self,
        data_dir: Path,
        run_ids: Sequence[str],
        topology: dict[str, Any],
        cache_size: int = DEFAULT_SHARD_CACHE_SIZE,
    ) -> None:
        self.window_dataset = ResidualWindowDataset(
            data_dir, run_ids, topology, cache_size=cache_size
        )
        self.rows = self.window_dataset.rows
        self.topology = topology
        candidate_ids = list(topology["candidate_ids"])
        candidate_kinds = list(topology["candidate_kinds"])
        self.candidate_ids = candidate_ids
        self.candidate_types = candidate_kinds
        self.candidate_entity_indices = torch.tensor(
            [
                -1,
                *[int(index) for index in topology["candidate_node_indices"]],
                *[int(index) for index in topology["candidate_edge_indices"]],
            ],
            dtype=torch.long,
        )
        if len(self.candidate_entity_indices) != len(self.candidate_ids):
            raise ValueError("candidate metadata length mismatch in topology")
        if len(self.candidate_types) != len(self.candidate_ids):
            raise ValueError("candidate kind metadata length mismatch in topology")
        unknown_kinds = sorted(set(self.candidate_types) - set(CANDIDATE_TYPE_TO_INDEX))
        if unknown_kinds:
            raise ValueError(f"topology contains unknown candidate kinds: {unknown_kinds}")
        self.edge_index = torch.tensor(topology["edge_index"], dtype=torch.long)
        self.edge_endpoints = torch.tensor(topology["edge_endpoints"], dtype=torch.long)
        self.probe_endpoints = torch.tensor(topology["probe_endpoints"], dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of windows in the wrapped dataset."""
        return len(self.window_dataset)

    def __getitem__(self, index: int) -> ResidualGraphSample:
        """Return one graph sample whose reconstruction targets are the last snapshot's dynamic residuals."""
        sample = self.window_dataset[index]
        node_recon_dim = len(NODE_DYNAMIC_FEATURES)
        edge_recon_dim = len(EDGE_DYNAMIC_FEATURES)
        return ResidualGraphSample(
            run_id=str(sample["run_id"]),
            timestamp=str(sample["timestamp"]),
            x_node=sample["node_x"],
            x_edge=sample["edge_x"],
            x_probe=sample["probe_x"],
            edge_index=self.edge_index,
            edge_endpoints=self.edge_endpoints,
            probe_endpoints=self.probe_endpoints,
            candidate_ids=list(self.candidate_ids),
            candidate_types=list(self.candidate_types),
            candidate_entity_indices=self.candidate_entity_indices.clone(),
            candidate_label=int(sample["root_candidate_index"]),
            category_label=int(sample["fault_category_index"]),
            fault_present=float(sample["fault_present"]),
            fault_category=str(sample["fault_category"]),
            root_cause_kind=str(sample["root_cause_kind"]),
            root_cause_id=str(sample["root_cause_id"]),
            y_node=sample["node_x"][-1, :, :node_recon_dim].clone(),
            y_edge=sample["edge_x"][-1, :, :edge_recon_dim].clone(),
            y_probe=sample["probe_x"][-1].clone(),
        )


def collate_residual_graph_batch(samples: Sequence[ResidualGraphSample]) -> ResidualGraphBatch:
    """Concatenate graph samples into one batch, offsetting node and edge indices per graph."""
    if not samples:
        raise ValueError("Cannot collate an empty residual graph batch")
    history_len = int(samples[0].x_node.shape[0])
    if any(int(sample.x_node.shape[0]) != history_len for sample in samples):
        raise ValueError("All residual graph samples in a batch must share history_len")

    node_chunks = []
    edge_chunks = []
    probe_chunks = []
    edge_index_chunks = []
    edge_endpoint_chunks = []
    probe_endpoint_chunks = []
    node_batch = []
    edge_batch = []
    probe_batch = []
    candidate_batch = []
    candidate_types = []
    candidate_entity_indices = []
    candidate_ptr = [0]
    y_node = []
    y_edge = []
    y_probe = []

    node_offset = 0
    edge_offset = 0
    for graph_index, sample in enumerate(samples):
        node_count = int(sample.x_node.shape[1])
        edge_count = int(sample.x_edge.shape[1])
        probe_count = int(sample.x_probe.shape[1])
        candidate_count = len(sample.candidate_ids)
        unknown_candidate_types = sorted(set(sample.candidate_types) - set(CANDIDATE_TYPE_TO_INDEX))
        if unknown_candidate_types:
            raise ValueError(
                f"sample {sample.run_id} contains unknown candidate types: "
                f"{unknown_candidate_types}"
            )

        node_chunks.append(sample.x_node)
        edge_chunks.append(sample.x_edge)
        probe_chunks.append(sample.x_probe)
        edge_index_chunks.append(sample.edge_index.long() + node_offset)
        edge_endpoint_chunks.append(sample.edge_endpoints.long() + node_offset)
        probe_endpoint_chunks.append(sample.probe_endpoints.long() + node_offset)
        node_batch.append(torch.full((node_count,), graph_index, dtype=torch.long))
        edge_batch.append(torch.full((edge_count,), graph_index, dtype=torch.long))
        probe_batch.append(torch.full((probe_count,), graph_index, dtype=torch.long))
        candidate_batch.append(torch.full((candidate_count,), graph_index, dtype=torch.long))
        candidate_types.append(
            torch.tensor(
                [
                    CANDIDATE_TYPE_TO_INDEX[candidate_type]
                    for candidate_type in sample.candidate_types
                ],
                dtype=torch.long,
            )
        )
        entity_indices = sample.candidate_entity_indices.long().clone()
        for candidate_index, candidate_type in enumerate(sample.candidate_types):
            if candidate_type == "node" and entity_indices[candidate_index] >= 0:
                entity_indices[candidate_index] += node_offset
            elif candidate_type == "edge" and entity_indices[candidate_index] >= 0:
                entity_indices[candidate_index] += edge_offset
            else:
                entity_indices[candidate_index] = -1
        candidate_entity_indices.append(entity_indices)
        candidate_ptr.append(candidate_ptr[-1] + candidate_count)
        y_node.append(sample.y_node)
        y_edge.append(sample.y_edge)
        y_probe.append(sample.y_probe)
        node_offset += node_count
        edge_offset += edge_count

    return ResidualGraphBatch(
        x_node=torch.cat(node_chunks, dim=1).float(),
        x_edge=torch.cat(edge_chunks, dim=1).float(),
        x_probe=torch.cat(probe_chunks, dim=1).float(),
        edge_index=torch.cat(edge_index_chunks, dim=1).long(),
        edge_endpoints=torch.cat(edge_endpoint_chunks, dim=0).long(),
        probe_endpoints=torch.cat(probe_endpoint_chunks, dim=0).long(),
        node_batch=torch.cat(node_batch).long(),
        edge_batch=torch.cat(edge_batch).long(),
        probe_batch=torch.cat(probe_batch).long(),
        candidate_ptr=torch.tensor(candidate_ptr, dtype=torch.long),
        candidate_batch=torch.cat(candidate_batch).long(),
        candidate_types=torch.cat(candidate_types).long(),
        candidate_entity_indices=torch.cat(candidate_entity_indices).long(),
        candidate_labels=torch.tensor(
            [sample.candidate_label for sample in samples], dtype=torch.long
        ),
        category_labels=torch.tensor(
            [sample.category_label for sample in samples], dtype=torch.long
        ),
        y_node=torch.cat(y_node, dim=0).float(),
        y_edge=torch.cat(y_edge, dim=0).float(),
        y_probe=torch.cat(y_probe, dim=0).float(),
        run_id=[sample.run_id for sample in samples],
        timestamp=[sample.timestamp for sample in samples],
        candidate_ids=[list(sample.candidate_ids) for sample in samples],
        candidate_type_labels=[list(sample.candidate_types) for sample in samples],
        root_cause_kind=[sample.root_cause_kind for sample in samples],
        root_cause_id=[sample.root_cause_id for sample in samples],
        fault_category_label=[sample.fault_category for sample in samples],
    )
