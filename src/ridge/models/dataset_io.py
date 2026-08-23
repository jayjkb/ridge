"""Shard, window-index and normalization access shared by Stages 2-5."""

from __future__ import annotations

import multiprocessing
import os
import random
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ridge.common.io import read_csv_rows, read_json
from ridge.models.artifacts import ArtifactValidationError, validate_artifact

NORMAL_RUNS_DIRNAME = "normal_runs"
RESIDUAL_RUNS_DIRNAME = "residual_runs"
NORMAL_WINDOW_INDEX = "normal_window_index.csv"
RESIDUAL_WINDOW_INDEX = "residual_window_index.csv"
DEFAULT_BUILD_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_TRAIN_DATALOADER_WORKERS = max(0, min(8, (os.cpu_count() or 1) // 2))
DEFAULT_TRAIN_PREFETCH_FACTOR = 2
DEFAULT_SHARD_CACHE_SIZE = 2


def set_seed(seed: int) -> None:
    """Seed Python, PyTorch, and CUDA random number generators."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_aware_loader(
    dataset: Dataset,
    *,
    collate_fn: Callable[..., object],
    batch_size: int,
    seed: int,
    workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    shuffle: bool,
) -> DataLoader:
    """Assemble the DataLoader configuration shared by the Stage 3 and 5 trainers."""
    workers = max(0, int(workers))
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "num_workers": workers,
    }
    if shuffle:
        kwargs["sampler"] = RunAwareSampler(dataset.rows, seed=seed)
    if workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(dataset, **kwargs)


def load_cached_run_shard(
    *,
    cache: OrderedDict[str, dict[str, object]],
    cache_size: int,
    validated_runs: set[str],
    validate_run_windows: Callable[[str, dict[str, object]], None],
    run_id: str,
    shard_path: Path,
    artifact_type: str,
) -> dict[str, object]:
    """Load an episode shard through the shared validate-then-LRU-cache protocol."""
    shard = cache.pop(run_id, None)
    if shard is None:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        validate_artifact(shard, artifact_type, source=shard_path)
    if run_id not in validated_runs:
        validate_run_windows(run_id, shard)
    if cache_size:
        cache[run_id] = shard
        while len(cache) > cache_size:
            cache.popitem(last=False)
    return shard


def new_feature_stats_accumulator(continuous_indices: Sequence[int]) -> dict[str, object]:
    """Return an empty running-sum accumulator for the given numerical feature indices."""
    return {
        "feature_indices": list(continuous_indices),
        "sum": None,
        "sum_sq": None,
        "count": 0,
    }


def update_feature_stats_accumulator(accumulator: dict[str, object], tensor: torch.Tensor) -> None:
    """Add the sums and squared sums of the selected features over every row of a tensor."""
    indices = list(accumulator["feature_indices"])
    if not indices:
        return
    matrix = tensor[..., indices].reshape(-1, len(indices)).float()
    if matrix.numel() == 0:
        return
    total = matrix.sum(dim=0)
    total_sq = (matrix * matrix).sum(dim=0)
    accumulator["sum"] = total if accumulator["sum"] is None else accumulator["sum"] + total
    accumulator["sum_sq"] = (
        total_sq if accumulator["sum_sq"] is None else accumulator["sum_sq"] + total_sq
    )
    accumulator["count"] = int(accumulator["count"]) + int(matrix.shape[0])


def merge_feature_stats_accumulator(
    accumulator: dict[str, object], partial: dict[str, object]
) -> None:
    """Add the sums and count of a partial accumulator, accepting list-valued sums from workers."""
    indices = list(accumulator["feature_indices"])
    if not indices:
        return
    count = int(partial.get("count", 0))
    partial_sum = partial.get("sum")
    partial_sum_sq = partial.get("sum_sq")
    if count <= 0 or partial_sum is None or partial_sum_sq is None:
        return
    if not isinstance(partial_sum, torch.Tensor):
        partial_sum = torch.tensor(partial_sum, dtype=torch.float32)
    if not isinstance(partial_sum_sq, torch.Tensor):
        partial_sum_sq = torch.tensor(partial_sum_sq, dtype=torch.float32)
    accumulator["sum"] = (
        partial_sum if accumulator["sum"] is None else accumulator["sum"] + partial_sum
    )
    accumulator["sum_sq"] = (
        partial_sum_sq if accumulator["sum_sq"] is None else accumulator["sum_sq"] + partial_sum_sq
    )
    accumulator["count"] = int(accumulator["count"]) + count


def finalize_feature_stats_accumulator(accumulator: dict[str, object]) -> dict[str, list[float]]:
    """Convert the running sums into per-feature mean and standard deviation with a floor of 1e-6."""
    indices = list(accumulator["feature_indices"])
    count = int(accumulator["count"])
    if not indices or count <= 0 or accumulator["sum"] is None or accumulator["sum_sq"] is None:
        return {"feature_indices": indices, "mean": [], "std": []}
    mean = accumulator["sum"] / count
    variance = (accumulator["sum_sq"] / count) - (mean * mean)
    std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return {"feature_indices": indices, "mean": mean.tolist(), "std": std.tolist()}


def apply_feature_stats(tensor: torch.Tensor, stats: dict[str, list[float]]) -> torch.Tensor:
    """Standardize the selected features of a tensor with the fitted mean and standard deviation."""
    indices = list(stats.get("feature_indices", []))
    if not indices:
        return tensor.clone().float()
    normalized = tensor.clone().float()
    mean = torch.tensor(stats["mean"], dtype=normalized.dtype)
    std = torch.tensor(stats["std"], dtype=normalized.dtype)
    normalized[..., indices] = (normalized[..., indices] - mean) / std
    return normalized


def merge_normalization(
    node_input: dict[str, list[float]],
    edge_input: dict[str, list[float]],
    probe_input: dict[str, list[float]],
    node_target: dict[str, list[float]],
    edge_target: dict[str, list[float]],
    probe_target: dict[str, list[float]],
) -> dict[str, object]:
    """Collect the six input and target statistics blocks into one normalization dictionary."""
    return {
        "node_input": node_input,
        "edge_input": edge_input,
        "probe_input": probe_input,
        "node_target": node_target,
        "edge_target": edge_target,
        "probe_target": probe_target,
    }


def normalized_worker_count(workers: int | None) -> int:
    """Return the requested worker count, at least one, or the default when unset."""
    if workers is None:
        return DEFAULT_BUILD_WORKERS
    return max(1, int(workers))


def iter_map_in_workers(function: Any, items: Sequence[Any], workers: int) -> Iterable[Any]:
    """Yield the results of a function over items, using spawned worker processes when more than one is requested."""
    if workers <= 1 or len(items) <= 1:
        for item in items:
            yield function(item)
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [(executor.submit(function, item), item) for item in items]
        for future, job in futures:
            try:
                yield future.result()
            except Exception as exc:  # pragma: no cover - exercised via caller tests
                run_id = "unknown"
                if isinstance(job, tuple) and job:
                    if isinstance(job[0], str):
                        run_id = job[0]
                    elif len(job) > 1 and isinstance(job[1], dict):
                        run_id = str(job[1].get("run_id", "unknown"))
                raise RuntimeError(f"build-normal worker failed for run_id={run_id}") from exc


def has_normal_run_shards(data_dir: Path) -> bool:
    """Return whether a directory holds a Stage-2 window index and episode shards."""
    return (data_dir / NORMAL_WINDOW_INDEX).exists() and (data_dir / NORMAL_RUNS_DIRNAME).is_dir()


def has_residual_run_shards(data_dir: Path) -> bool:
    """Return whether a directory holds a Stage-4 window index and episode shards."""
    return (data_dir / RESIDUAL_WINDOW_INDEX).exists() and (
        data_dir / RESIDUAL_RUNS_DIRNAME
    ).is_dir()


def load_normal_index(data_dir: Path) -> dict[str, Any]:
    """Read normal_index.json and validate its artifact contract."""
    path = data_dir / "normal_index.json"
    return validate_artifact(read_json(path), "normal_dataset", source=path)


def load_residual_index(data_dir: Path) -> dict[str, Any]:
    """Read residual_index.json and validate its artifact contract."""
    path = data_dir / "residual_index.json"
    return validate_artifact(read_json(path), "residual_dataset", source=path)


def run_ids_from_splits(splits: dict[str, list[str]]) -> set[str]:
    """Return the episode identifiers of the train, validation, and test splits as strings."""
    run_ids: set[str] = set()
    for split_name in ("train", "val", "test"):
        run_ids.update(str(run_id) for run_id in splits.get(split_name, []))
    return run_ids


def load_window_index(path: Path) -> list[dict[str, str]]:
    """Read a window index CSV into string-valued rows."""
    return read_csv_rows(path)


NORMAL_WINDOW_INDEX_COLUMNS = (
    "run_id",
    "history_len",
    "target_index",
    "prediction_index",
    "target_snapshot_id",
    "target_timestamp",
)


def window_index_error(
    index_path: Path, run_id: str, field: str, detail: str
) -> ArtifactValidationError:
    """Build a uniform window-index validation error naming the file, episode, and field."""
    return ArtifactValidationError(f"{index_path}: run {run_id}: {field}: {detail}")


def parse_window_index_int(
    row: dict[str, str], column: str, *, index_path: Path, run_id: str
) -> int:
    """Parse an integer column of a window-index row, raising the uniform validation error."""
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise window_index_error(
            index_path, run_id, "integer_parse", f"column {column!r}={row.get(column)!r}"
        ) from exc


def require_index_int(index: dict[str, Any], key: str, *, index_path: Path) -> int:
    """Return an integer field of a dataset index, raising ArtifactValidationError when absent or malformed."""
    try:
        return int(index[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"{index_path}: window index is missing an integer {key!r}"
        ) from exc


def validate_window_sample_count(
    index: dict[str, Any], all_rows: Sequence[dict[str, str]], *, index_path: Path
) -> None:
    """Root ``sample_count`` must equal the number of rows in the window index CSV."""
    if "sample_count" not in index:
        return
    declared = require_index_int(index, "sample_count", index_path=index_path)
    if declared != len(all_rows):
        raise ArtifactValidationError(
            f"{index_path}: sample_count {declared} does not match "
            f"{len(all_rows)} window-index rows"
        )


def group_rows_by_run(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group window-index rows by episode identifier in first-seen order."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_id"]), []).append(row)
    return grouped


class RunAwareSampler(Sampler[int]):
    """Shuffle episodes and their windows while keeping each episode contiguous."""

    def __init__(self, rows: Sequence[dict[str, str]], *, seed: int) -> None:
        self.seed = int(seed)
        self._epoch = 0
        grouped: OrderedDict[str, list[int]] = OrderedDict()
        for index, row in enumerate(rows):
            grouped.setdefault(str(row["run_id"]), []).append(index)
        self._indices_by_run = grouped
        self._length = sum(len(indices) for indices in grouped.values())

    def __len__(self) -> int:
        """Return the number of windows the sampler yields per epoch."""
        return self._length

    def __iter__(self):
        """Yield window indices with episodes shuffled and windows shuffled within each episode, reseeded per epoch."""
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        run_ids = list(self._indices_by_run)
        rng.shuffle(run_ids)
        for run_id in run_ids:
            indices = list(self._indices_by_run[run_id])
            rng.shuffle(indices)
            yield from indices
