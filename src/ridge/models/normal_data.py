"""Stage-1 loading and Stage-2 normal-dataset construction."""

from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from ridge.common.contracts import parse_bool
from ridge.common.io import (
    read_csv_rows,
    read_json,
    write_csv_rows,
    write_json,
)
from ridge.io.stage1_dataset import load_generation_provenance, resolve_run_dir_values
from ridge.models.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    artifact_metadata,
)
from ridge.models.dataset_io import (
    DEFAULT_SHARD_CACHE_SIZE,
    NORMAL_RUNS_DIRNAME,
    NORMAL_WINDOW_INDEX,
    NORMAL_WINDOW_INDEX_COLUMNS,
    apply_feature_stats,
    finalize_feature_stats_accumulator,
    group_rows_by_run,
    iter_map_in_workers,
    load_cached_run_shard,
    load_normal_index,
    load_window_index,
    merge_feature_stats_accumulator,
    merge_normalization,
    new_feature_stats_accumulator,
    normalized_worker_count,
    parse_window_index_int,
    require_index_int,
    update_feature_stats_accumulator,
    validate_window_sample_count,
    window_index_error,
)
from ridge.models.episode_graph import (
    json_topology,
    load_episode_graph,
    load_probe_ids_from_manifest_rows,
    load_topology,
)
from ridge.models.features import (
    EDGE_DYNAMIC_FEATURES,
    FAULT_CATEGORY_TO_INDEX,
    NODE_DYNAMIC_FEATURES,
    PROBE_CONTINUOUS_FEATURES,
    feature_schema,
)


def _split_counts(size: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """Return the train and validation episode counts for one stratum, leaving at least one for test."""
    if size <= 1:
        return size, 0
    if size == 2:
        return 1, 0
    train_count = max(1, round(size * train_ratio))
    val_count = max(1, round(size * val_ratio))
    if train_count + val_count >= size:
        train_count = size - 2
        val_count = 1
    return int(train_count), int(val_count)


def stratified_split(
    rows: Sequence[dict[str, object]],
    seed: int,
    label_key: str = "split_label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """Assign episodes to train, validation, and test splits per label with a seeded shuffle."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[label_key])].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in sorted(grouped):
        members = list(grouped[label])
        rng.shuffle(members)
        train_count, val_count = _split_counts(len(members), train_ratio, val_ratio)
        splits["train"].extend(str(row["run_id"]) for row in members[:train_count])
        splits["val"].extend(
            str(row["run_id"]) for row in members[train_count : train_count + val_count]
        )
        splits["test"].extend(str(row["run_id"]) for row in members[train_count + val_count :])
    total_runs = sum(len(run_ids) for run_ids in splits.values())
    if total_runs >= 3 and not splits["val"] and len(splits["train"]) > 1:
        splits["val"].append(splits["train"].pop())
    if total_runs >= 3 and not splits["test"] and len(splits["train"]) > 1:
        splits["test"].append(splits["train"].pop())
    if total_runs >= 2 and not splits["test"] and splits["val"]:
        splits["test"].append(splits["val"][-1])
    splits["test"] = list(dict.fromkeys(splits["test"]))
    splits["val"] = [run_id for run_id in splits["val"] if run_id not in set(splits["test"])]
    for run_ids in splits.values():
        rng.shuffle(run_ids)
    return splits


def episode_split_label(row: dict[str, str]) -> str:
    """Return the stratification label of an episode, healthy for the no-fault category."""
    fault_category = str(row.get("fault_category", "none") or "none")
    if fault_category not in FAULT_CATEGORY_TO_INDEX:
        raise ValueError(
            f"run_id={row.get('run_id', 'unknown')} has unknown fault_category={fault_category!r}"
        )
    if fault_category == "none":
        return "healthy"
    return fault_category


def _window_is_prefault(timestamp: str, row: dict[str, str]) -> bool:
    """Return whether a timestamp precedes the fault onset of the episode, or True without a fault."""
    fault_start = row.get("fault_start_ts", "")
    if not fault_start:
        return True
    return timestamp < fault_start


def _window_has_complete_consecutive_snapshots(
    episode: dict[str, object],
    target_index: int,
    history_len: int,
    prediction_horizon: int,
) -> bool:
    """Reject windows that would bridge an explicitly missing telemetry slot."""
    prediction_index = target_index + prediction_horizon - 1
    window_indices = list(range(target_index - history_len, prediction_index + 1))
    complete = list(episode["snapshot_complete"])
    if not all(bool(complete[index]) for index in window_indices):
        return False
    try:
        snapshot_ids = [int(str(episode["snapshot_ids"][index])) for index in window_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Stage-1 SnapshotId values must be monotonically increasing integers"
        ) from exc
    return snapshot_ids == list(range(snapshot_ids[0], snapshot_ids[0] + len(snapshot_ids)))


def window_fault_state(timestamp: str, row: dict[str, str]) -> tuple[int, str]:
    """Return the fault-present flag and fault category of an episode at a timestamp."""
    fault_category = str(row.get("fault_category", "none") or "none")
    if fault_category not in FAULT_CATEGORY_TO_INDEX:
        raise ValueError(
            f"run_id={row.get('run_id', 'unknown')} has unknown fault_category={fault_category!r}"
        )
    if fault_category == "none":
        return 0, "none"
    fault_start = row.get("fault_start_ts", "")
    fault_end = row.get("fault_end_ts", "")
    if fault_start and timestamp < fault_start:
        return 0, "none"
    if fault_end and timestamp >= fault_end:
        return 0, "none"
    return 1, fault_category


def _eligible_normal_indices(
    episode: dict[str, object],
    row: dict[str, str],
    history_len: int,
    prediction_horizon: int,
) -> tuple[list[int], list[int]]:
    """Return unique input/target snapshots used by eligible normal windows."""
    steps = int(episode["node_x"].shape[0])
    input_indices: set[int] = set()
    target_indices: set[int] = set()
    for target_index in range(history_len, steps - prediction_horizon + 1):
        prediction_index = target_index + prediction_horizon - 1
        if not _window_has_complete_consecutive_snapshots(
            episode,
            target_index,
            history_len,
            prediction_horizon,
        ):
            continue
        target_timestamp = str(episode["timestamps"][prediction_index])
        if row.get("fault_category", "none") != "none" and not _window_is_prefault(
            target_timestamp, row
        ):
            continue
        input_indices.update(range(target_index - history_len, target_index))
        target_indices.add(prediction_index)
    return sorted(input_indices), sorted(target_indices)


def _episode_stats_payload(
    episode: dict[str, object],
    input_indices: Sequence[int],
    target_indices: Sequence[int],
) -> dict[str, dict[str, object]]:
    """Accumulate input and target statistics of one episode over its eligible snapshots as lists."""
    node_input = new_feature_stats_accumulator(range(len(NODE_DYNAMIC_FEATURES)))
    edge_input = new_feature_stats_accumulator(range(len(EDGE_DYNAMIC_FEATURES)))
    probe_input = new_feature_stats_accumulator(range(len(PROBE_CONTINUOUS_FEATURES)))
    node_target = new_feature_stats_accumulator(range(len(NODE_DYNAMIC_FEATURES)))
    edge_target = new_feature_stats_accumulator(range(len(EDGE_DYNAMIC_FEATURES)))
    probe_target = new_feature_stats_accumulator(range(len(PROBE_CONTINUOUS_FEATURES)))
    update_feature_stats_accumulator(node_input, episode["node_x"][list(input_indices)])
    update_feature_stats_accumulator(edge_input, episode["edge_x"][list(input_indices)])
    update_feature_stats_accumulator(probe_input, episode["probe_x"][list(input_indices)])
    update_feature_stats_accumulator(node_target, episode["target_node_y"][list(target_indices)])
    update_feature_stats_accumulator(edge_target, episode["target_edge_y"][list(target_indices)])
    update_feature_stats_accumulator(probe_target, episode["target_probe_y"][list(target_indices)])
    payload = {
        "node_input": node_input,
        "edge_input": edge_input,
        "probe_input": probe_input,
        "node_target": node_target,
        "edge_target": edge_target,
        "probe_target": probe_target,
    }
    for stats in payload.values():
        if stats["sum"] is not None:
            stats["sum"] = stats["sum"].tolist()
        if stats["sum_sq"] is not None:
            stats["sum_sq"] = stats["sum_sq"].tolist()
    return payload


def _merge_episode_stats(
    node_input_stats: dict[str, object],
    edge_input_stats: dict[str, object],
    probe_input_stats: dict[str, object],
    node_target_stats: dict[str, object],
    edge_target_stats: dict[str, object],
    probe_target_stats: dict[str, object],
    partial_stats: dict[str, dict[str, object]],
) -> None:
    """Merge one episode's partial statistics into the six dataset accumulators."""
    merge_feature_stats_accumulator(node_input_stats, partial_stats["node_input"])
    merge_feature_stats_accumulator(edge_input_stats, partial_stats["edge_input"])
    merge_feature_stats_accumulator(probe_input_stats, partial_stats["probe_input"])
    merge_feature_stats_accumulator(node_target_stats, partial_stats["node_target"])
    merge_feature_stats_accumulator(edge_target_stats, partial_stats["edge_target"])
    merge_feature_stats_accumulator(probe_target_stats, partial_stats["probe_target"])


def _load_episode_stats_build_job(
    job: tuple[Path, dict[str, str], dict[str, object], bool, int, int, Path],
) -> tuple[str, dict[str, dict[str, object]] | None, list[dict[str, object]]]:
    """Load one episode's tensors, save the provisional shard, and return its statistics and window rows."""
    (
        dataset_dir,
        row,
        topology,
        include_stats,
        history_len,
        prediction_horizon,
        provisional_dir,
    ) = job
    run_id = str(row["run_id"])
    episode = load_episode_graph(
        resolve_run_dir_values(
            dataset_dir,
            run_id=int(row["run_id"]),
            log_dir=str(row.get("log_dir", "") or ""),
        ),
        topology,
    )
    eligible_prediction_indices = [
        target_index + prediction_horizon - 1
        for target_index in range(
            history_len,
            int(episode["node_x"].shape[0]) - prediction_horizon + 1,
        )
        if _window_has_complete_consecutive_snapshots(
            episode,
            target_index,
            history_len,
            prediction_horizon,
        )
    ]
    torch.save(
        {
            "run_id": run_id,
            "timestamps": list(episode["timestamps"]),
            "snapshot_ids": list(episode["snapshot_ids"]),
            "snapshot_complete": list(episode["snapshot_complete"]),
            "eligible_prediction_indices": eligible_prediction_indices,
            "node_x": episode["node_x"],
            "edge_x": episode["edge_x"],
            "probe_x": episode["probe_x"],
            "target_node_y": episode["target_node_y"],
            "target_edge_y": episode["target_edge_y"],
            "target_probe_y": episode["target_probe_y"],
        },
        provisional_dir / f"run_{int(run_id):06d}.pt",
    )
    input_indices, target_indices = _eligible_normal_indices(
        episode, row, history_len, prediction_horizon
    )
    window_rows = []
    for target_index in range(
        history_len, int(episode["node_x"].shape[0]) - prediction_horizon + 1
    ):
        prediction_index = target_index + prediction_horizon - 1
        if not _window_has_complete_consecutive_snapshots(
            episode,
            target_index,
            history_len,
            prediction_horizon,
        ):
            continue
        target_timestamp = str(episode["timestamps"][prediction_index])
        if row.get("fault_category", "none") != "none" and not _window_is_prefault(
            target_timestamp, row
        ):
            continue
        window_rows.append(
            {
                "run_id": run_id,
                "history_len": history_len,
                "target_index": target_index,
                "prediction_index": prediction_index,
                "target_snapshot_id": str(episode["snapshot_ids"][prediction_index]),
                "target_timestamp": target_timestamp,
            }
        )
    stats = (
        _episode_stats_payload(episode, input_indices, target_indices) if include_stats else None
    )
    return run_id, stats, window_rows


def _build_normal_shard_job(
    job: tuple[
        str,
        Path,
        Path,
        dict[str, object],
    ],
) -> str:
    """Normalize one provisional shard, write the final shard, and delete the provisional file."""
    (
        run_id,
        provisional_dir,
        normal_runs_dir,
        normalization,
    ) = job
    episode = torch.load(
        provisional_dir / f"run_{int(run_id):06d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    shard = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "normal_run_shard",
        "run_id": run_id,
        "timestamps": list(episode["timestamps"]),
        "snapshot_ids": list(episode["snapshot_ids"]),
        "snapshot_complete": list(episode["snapshot_complete"]),
        "eligible_prediction_indices": list(episode["eligible_prediction_indices"]),
        "node_x": apply_feature_stats(episode["node_x"], normalization["node_input"]),
        "edge_x": apply_feature_stats(episode["edge_x"], normalization["edge_input"]),
        "probe_x": apply_feature_stats(episode["probe_x"], normalization["probe_input"]),
        "target_node_y": apply_feature_stats(
            episode["target_node_y"], normalization["node_target"]
        ),
        "target_edge_y": apply_feature_stats(
            episode["target_edge_y"], normalization["edge_target"]
        ),
        "target_probe_y": apply_feature_stats(
            episode["target_probe_y"], normalization["probe_target"]
        ),
    }
    torch.save(shard, normal_runs_dir / f"run_{int(run_id):06d}.pt")
    (provisional_dir / f"run_{int(run_id):06d}.pt").unlink()
    return run_id


def _eligible_normal_manifest_rows(
    dataset_dir: Path,
    require_baseline_health_pass: bool,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return the successful manifest rows, optionally restricted to episodes that passed the validity check, with counts."""
    all_rows = read_csv_rows(dataset_dir / "manifest.csv")
    ok_rows = [row for row in all_rows if row.get("status", "ok") == "ok"]
    eligible_rows = list(ok_rows)
    excluded_for_baseline_health = 0
    if require_baseline_health_pass:
        eligible_rows = []
        for row in ok_rows:
            if parse_bool(row.get("baseline_health_pass")):
                eligible_rows.append(row)
            else:
                excluded_for_baseline_health += 1
    counts = {
        "manifest_rows_total": len(all_rows),
        "manifest_rows_ok": len(ok_rows),
        "manifest_rows_used": len(eligible_rows),
        "manifest_rows_excluded_for_baseline_health": excluded_for_baseline_health,
    }
    return eligible_rows, counts


def _collect_candidate_lists(
    payload: object,
    node_ids: set[str],
    edge_ids: set[str],
) -> None:
    """Gather drain and link fault target identifiers from a provenance payload recursively."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "drain_node_fault_targets" and isinstance(value, list):
                node_ids.update(str(item) for item in value)
            elif key in {
                "fiber_cut_edge_fault_targets",
                "link_degradation_edge_fault_targets",
                "link_flap_edge_fault_targets",
            } and isinstance(value, list):
                edge_ids.update(str(item) for item in value)
            else:
                _collect_candidate_lists(value, node_ids, edge_ids)
    elif isinstance(payload, list):
        for item in payload:
            _collect_candidate_lists(item, node_ids, edge_ids)


def _source_candidate_catalog(
    dataset_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
) -> tuple[list[str], list[str]]:
    """Resolve the RCA catalog from source provenance and observed labels."""
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    provenance_path = dataset_dir / "generation_provenance.json"
    if provenance_path.exists():
        _collect_candidate_lists(read_json(provenance_path), node_ids, edge_ids)
    for row in manifest_rows:
        category = episode_split_label(row)
        if category == "healthy":
            continue
        root_id = str(row.get("root_cause_id", "")).strip()
        root_kind = str(row.get("root_cause_kind", "")).strip()
        if not root_id:
            raise ValueError(f"run_id={row.get('run_id')} is faulty but has no root_cause_id")
        if root_kind == "node":
            node_ids.add(root_id)
        elif root_kind == "edge":
            edge_ids.add(root_id)
        else:
            raise ValueError(
                f"run_id={row.get('run_id')} has unsupported root_cause_kind={root_kind!r}"
            )
    return sorted(node_ids), sorted(edge_ids)


def build_normal_emulator_dataset(
    dataset_dir: Path,
    output_dir: Path,
    history_len: int,
    prediction_horizon: int,
    seed: int,
    workers: int | None = None,
    require_baseline_health_pass: bool = True,
) -> dict[str, object]:
    """Build the Stage-2 windows, fit normalization on training episodes only, and write the shards, splits, and index."""
    if history_len < 1:
        raise ValueError("history_len must be at least 1")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be at least 1")
    stage1_provenance = load_generation_provenance(dataset_dir, require_contract=True)
    manifest_rows, eligibility_counts = _eligible_normal_manifest_rows(
        dataset_dir,
        require_baseline_health_pass=require_baseline_health_pass,
    )
    if not manifest_rows:
        requirement = " and baseline_health_pass == true" if require_baseline_health_pass else ""
        raise ValueError(
            f"No eligible manifest rows found in {dataset_dir} (requires status == ok{requirement})"
        )
    probe_ids = load_probe_ids_from_manifest_rows(dataset_dir, manifest_rows)
    candidate_node_ids, candidate_edge_ids = _source_candidate_catalog(dataset_dir, manifest_rows)
    topology = load_topology(
        resolve_run_dir_values(
            dataset_dir,
            run_id=int(manifest_rows[0]["run_id"]),
            log_dir=str(manifest_rows[0].get("log_dir", "") or ""),
        ),
        probe_ids,
        candidate_node_ids,
        candidate_edge_ids,
    )
    index_rows: list[dict[str, object]] = []
    node_input_stats = new_feature_stats_accumulator(range(len(NODE_DYNAMIC_FEATURES)))
    edge_input_stats = new_feature_stats_accumulator(range(len(EDGE_DYNAMIC_FEATURES)))
    probe_input_stats = new_feature_stats_accumulator(range(len(PROBE_CONTINUOUS_FEATURES)))
    node_target_stats = new_feature_stats_accumulator(range(len(NODE_DYNAMIC_FEATURES)))
    edge_target_stats = new_feature_stats_accumulator(range(len(EDGE_DYNAMIC_FEATURES)))
    probe_target_stats = new_feature_stats_accumulator(range(len(PROBE_CONTINUOUS_FEATURES)))

    for row in manifest_rows:
        index_rows.append(
            {
                "run_id": str(row["run_id"]),
                "split_label": episode_split_label(row),
                "root_cause_kind": row.get("root_cause_kind", ""),
                "fault_category": row.get("fault_category", "none"),
            }
        )

    splits = stratified_split(index_rows, seed, label_key="split_label")
    train_run_ids = set(splits["train"])
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_runs_dir = output_dir / NORMAL_RUNS_DIRNAME
    normal_runs_dir.mkdir(parents=True, exist_ok=True)
    provisional_dir = output_dir / ".provisional_normal_runs"
    provisional_dir.mkdir(parents=True, exist_ok=True)
    worker_count = normalized_worker_count(workers)
    window_rows: list[dict[str, object]] = []

    stats_jobs = [
        (
            dataset_dir,
            row,
            topology,
            str(row["run_id"]) in train_run_ids,
            history_len,
            prediction_horizon,
            provisional_dir,
        )
        for row in manifest_rows
    ]
    for _run_id, partial_stats, run_window_rows in iter_map_in_workers(
        _load_episode_stats_build_job, stats_jobs, worker_count
    ):
        window_rows.extend(run_window_rows)
        if partial_stats is not None:
            _merge_episode_stats(
                node_input_stats,
                edge_input_stats,
                probe_input_stats,
                node_target_stats,
                edge_target_stats,
                probe_target_stats,
                partial_stats,
            )

    normalization = merge_normalization(
        finalize_feature_stats_accumulator(node_input_stats),
        finalize_feature_stats_accumulator(edge_input_stats),
        finalize_feature_stats_accumulator(probe_input_stats),
        finalize_feature_stats_accumulator(node_target_stats),
        finalize_feature_stats_accumulator(edge_target_stats),
        finalize_feature_stats_accumulator(probe_target_stats),
    )
    normalization["fitted_on"] = "train_split_eligible_normal_windows"
    empty_stats = [
        name
        for name, stats in normalization.items()
        if isinstance(stats, dict) and stats.get("feature_indices") and not stats.get("mean")
    ]
    if empty_stats:
        raise ValueError(
            "training split contains no eligible normal observations for: " + ", ".join(empty_stats)
        )

    topology_payload = json_topology(topology)
    normal_metadata = artifact_metadata("normal_dataset")

    sample_count = len(window_rows)
    shard_jobs = [
        (
            str(row["run_id"]),
            provisional_dir,
            normal_runs_dir,
            normalization,
        )
        for row in manifest_rows
    ]
    for _run_id in iter_map_in_workers(_build_normal_shard_job, shard_jobs, worker_count):
        pass
    provisional_dir.rmdir()

    window_rows.sort(
        key=lambda row: (
            int(str(row["run_id"])),
            int(row["target_index"]),
            int(row["prediction_index"]),
        )
    )
    write_csv_rows(output_dir / NORMAL_WINDOW_INDEX, window_rows)
    write_json(
        output_dir / "normal_index.json",
        {
            **normal_metadata,
            "sample_count": sample_count,
            "feature_schema": feature_schema(),
            "source_dataset": str(dataset_dir.resolve()),
            "source_stage1_generated_at_utc": stage1_provenance.get("generated_at_utc"),
            "source_stage1_source_code_commit": stage1_provenance.get("source_code_commit"),
            "source_manifest_run_count": len(manifest_rows),
            "split_run_counts": {name: len(run_ids) for name, run_ids in splits.items()},
            "history_len": history_len,
            "prediction_horizon": prediction_horizon,
            "normalization_scope": normalization["fitted_on"],
            "snapshot_join_key": "SnapshotId",
            "build_seed": seed,
        },
    )
    write_json(output_dir / "normal_splits.json", splits)
    write_json(output_dir / "normalization.json", normalization)
    write_json(output_dir / "topology.json", topology_payload)
    return {
        "sample_count": sample_count,
        "split_sizes": {name: len(run_ids) for name, run_ids in splits.items()},
        **eligibility_counts,
        "require_baseline_health_pass": require_baseline_health_pass,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
    }


class NormalWindowDataset(Dataset):
    """Dataset of Stage-2 windows served from per-episode shards with lazy validation and an LRU cache."""
    
    def __init__(
        self,
        data_dir: Path,
        run_ids: Sequence[str],
        cache_size: int = DEFAULT_SHARD_CACHE_SIZE,
    ) -> None:
        self.data_dir = data_dir
        self._index_path = data_dir / NORMAL_WINDOW_INDEX
        index = load_normal_index(data_dir)
        self._index_history_len = require_index_int(
            index, "history_len", index_path=data_dir / "normal_index.json"
        )
        self._index_prediction_horizon = require_index_int(
            index, "prediction_horizon", index_path=data_dir / "normal_index.json"
        )
        all_rows = load_window_index(self._index_path)
        validate_window_sample_count(index, all_rows, index_path=self._index_path)
        selected = set(str(run_id) for run_id in run_ids)
        self.rows = [row for row in all_rows if str(row["run_id"]) in selected]
        self._rows_by_run = group_rows_by_run(self.rows)
        self._validated_runs: set[str] = set()
        self.cache_size = max(0, int(cache_size))
        self._run_cache: OrderedDict[str, dict[str, object]] = OrderedDict()

    def __len__(self) -> int:
        """Return the number of windows selected from the index."""
        return len(self.rows)

    def _validate_run_windows(self, run_id: str, shard: dict[str, object]) -> None:
        """Validate every selected window row for ``run_id`` against its shard."""
        index_path = self._index_path
        shard_run_id = str(shard.get("run_id"))
        if shard_run_id != run_id:
            raise window_index_error(
                index_path, run_id, "run_id", f"shard run_id {shard_run_id!r} does not match"
            )
        timestamps = shard["timestamps"]
        snapshot_ids = shard["snapshot_ids"]
        snapshot_complete = shard["snapshot_complete"]
        eligible = {int(value) for value in shard["eligible_prediction_indices"]}
        lengths = {
            key: int(shard[key].shape[0])
            for key in (
                "node_x",
                "edge_x",
                "probe_x",
                "target_node_y",
                "target_edge_y",
                "target_probe_y",
            )
        }
        metadata_len = min(len(timestamps), len(snapshot_ids), len(snapshot_complete))
        for row in self._rows_by_run.get(run_id, ()):
            for column in NORMAL_WINDOW_INDEX_COLUMNS:
                if column not in row:
                    raise window_index_error(index_path, run_id, "missing_column", column)
            if str(row["run_id"]) != run_id:
                raise window_index_error(
                    index_path, run_id, "run_id", f"row run_id {row['run_id']!r} misfiled"
                )
            history_len = parse_window_index_int(
                row, "history_len", index_path=index_path, run_id=run_id
            )
            target_index = parse_window_index_int(
                row, "target_index", index_path=index_path, run_id=run_id
            )
            prediction_index = parse_window_index_int(
                row, "prediction_index", index_path=index_path, run_id=run_id
            )
            if history_len != self._index_history_len:
                raise window_index_error(
                    index_path,
                    run_id,
                    "history_len",
                    f"{history_len} != normal_index {self._index_history_len}",
                )
            expected_prediction = target_index + self._index_prediction_horizon - 1
            if prediction_index != expected_prediction:
                raise window_index_error(
                    index_path,
                    run_id,
                    "prediction_index",
                    f"{prediction_index} != target_index+horizon-1 ({expected_prediction})",
                )
            window_start = target_index - history_len
            if window_start < 0:
                raise window_index_error(
                    index_path,
                    run_id,
                    "window_start",
                    f"target_index {target_index} < history_len {history_len}",
                )
            for key in ("node_x", "edge_x", "probe_x"):
                if lengths[key] < target_index:
                    raise window_index_error(
                        index_path,
                        run_id,
                        "input_bounds",
                        f"{key} length {lengths[key]} < target_index {target_index}",
                    )
            for key in ("target_node_y", "target_edge_y", "target_probe_y"):
                if lengths[key] <= prediction_index:
                    raise window_index_error(
                        index_path,
                        run_id,
                        "target_bounds",
                        f"{key} length {lengths[key]} <= prediction_index {prediction_index}",
                    )
            if metadata_len <= prediction_index:
                raise window_index_error(
                    index_path,
                    run_id,
                    "metadata_bounds",
                    f"metadata length {metadata_len} <= prediction_index {prediction_index}",
                )
            if str(snapshot_ids[prediction_index]) != str(row["target_snapshot_id"]):
                raise window_index_error(
                    index_path,
                    run_id,
                    "target_snapshot_id",
                    f"{row['target_snapshot_id']!r} != shard {snapshot_ids[prediction_index]!r}",
                )
            if str(timestamps[prediction_index]) != str(row["target_timestamp"]):
                raise window_index_error(
                    index_path,
                    run_id,
                    "target_timestamp",
                    f"{row['target_timestamp']!r} != shard {timestamps[prediction_index]!r}",
                )
            if prediction_index not in eligible:
                raise window_index_error(
                    index_path,
                    run_id,
                    "eligible_prediction_indices",
                    f"prediction_index {prediction_index} is not eligible",
                )
            window_indices = range(window_start, prediction_index + 1)
            if not all(bool(snapshot_complete[i]) for i in window_indices):
                raise window_index_error(
                    index_path,
                    run_id,
                    "snapshot_complete",
                    f"window [{window_start},{prediction_index}] spans an incomplete snapshot",
                )
            try:
                window_snapshot_ids = [int(str(snapshot_ids[i])) for i in window_indices]
            except ValueError as exc:
                raise window_index_error(
                    index_path,
                    run_id,
                    "snapshot_id_parse",
                    f"window [{window_start},{prediction_index}] has non-integer SnapshotIds",
                ) from exc
            if window_snapshot_ids != list(
                range(window_snapshot_ids[0], window_snapshot_ids[0] + len(window_snapshot_ids))
            ):
                raise window_index_error(
                    index_path,
                    run_id,
                    "snapshot_consecutive",
                    f"window [{window_start},{prediction_index}] spans non-consecutive SnapshotIds",
                )
        self._validated_runs.add(run_id)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one window's normalized history tensors and its one-step targets, cloned from the cached shard."""
        row = self.rows[index]
        run_id = str(row["run_id"])
        shard = load_cached_run_shard(
            cache=self._run_cache,
            cache_size=self.cache_size,
            validated_runs=self._validated_runs,
            validate_run_windows=self._validate_run_windows,
            run_id=run_id,
            shard_path=self.data_dir / NORMAL_RUNS_DIRNAME / f"run_{int(run_id):06d}.pt",
            artifact_type="normal_run_shard",
        )
        target_index = int(row["target_index"])
        prediction_index = int(row["prediction_index"])
        history_len = int(row["history_len"])
        window_start = target_index - history_len
        return {
            "run_id": run_id,
            "target_timestamp": str(row["target_timestamp"]),
            "node_x": shard["node_x"][window_start:target_index].clone(),
            "edge_x": shard["edge_x"][window_start:target_index].clone(),
            "probe_x": shard["probe_x"][window_start:target_index].clone(),
            "target_node_y": shard["target_node_y"][prediction_index].clone(),
            "target_edge_y": shard["target_edge_y"][prediction_index].clone(),
            "target_probe_y": shard["target_probe_y"][prediction_index].clone(),
        }


def collate_normal_batch(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    """Stack the windows of a batch into float tensors and collect their episode identifiers."""
    return {
        "node_x": torch.stack([sample["node_x"] for sample in batch]).float(),
        "edge_x": torch.stack([sample["edge_x"] for sample in batch]).float(),
        "probe_x": torch.stack([sample["probe_x"] for sample in batch]).float(),
        "target_node_y": torch.stack([sample["target_node_y"] for sample in batch]).float(),
        "target_edge_y": torch.stack([sample["target_edge_y"] for sample in batch]).float(),
        "target_probe_y": torch.stack([sample["target_probe_y"] for sample in batch]).float(),
        "run_id": [str(sample["run_id"]) for sample in batch],
    }
