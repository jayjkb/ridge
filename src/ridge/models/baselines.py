"""Learning-free baselines for the RIDGE thesis comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from ridge.common.io import (
    read_json,
    write_json,
)
from ridge.models.artifacts import artifact_metadata
from ridge.models.dataset_io import (
    RESIDUAL_RUNS_DIRNAME,
    RESIDUAL_WINDOW_INDEX,
    has_residual_run_shards,
    load_residual_index,
    load_window_index,
)
from ridge.models.episode_graph import (
    hydrate_topology,
)
from ridge.models.features import (
    EDGE_DYNAMIC_FEATURES,
    NODE_DYNAMIC_FEATURES,
)
from ridge.models.ridge_model import rank_candidates


def _split_window_scores(
    data_dir: Path,
    rows: Sequence[dict[str, str]],
    topology: dict[str, object],
) -> dict[str, object]:
    """Score every window of one split with per-candidate mean absolute residuals."""
    candidate_node_indices = torch.tensor(
        [int(i) for i in topology["candidate_node_indices"]], dtype=torch.long
    )
    candidate_edge_indices = torch.tensor(
        [int(i) for i in topology["candidate_edge_indices"]], dtype=torch.long
    )
    entity_scores = []
    labels = []
    kinds = []
    shard: dict[str, torch.Tensor] | None = None
    shard_run_id: str | None = None
    for row in rows:
        run_id = str(row["run_id"])
        if run_id != shard_run_id:
            shard = torch.load(
                data_dir / RESIDUAL_RUNS_DIRNAME / f"run_{int(run_id):06d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            shard_run_id = run_id
        history_len = int(row["history_len"])
        end = int(row["window_end_index"])
        start = end - history_len + 1
        node_window = shard["node_residual"][start : end + 1, :, : len(NODE_DYNAMIC_FEATURES)]
        edge_window = shard["edge_residual"][start : end + 1, :, : len(EDGE_DYNAMIC_FEATURES)]
        node_scores = node_window.abs().mean(dim=(0, 2))
        edge_scores = edge_window.abs().mean(dim=(0, 2))
        entity_scores.append(
            torch.cat([node_scores[candidate_node_indices], edge_scores[candidate_edge_indices]])
        )
        labels.append(int(row["root_candidate_index"]))
        kinds.append(str(row["root_cause_kind"]))
    return {
        "entity_scores": torch.stack(entity_scores) if entity_scores else torch.zeros((0, 0)),
        "labels": torch.tensor(labels, dtype=torch.long),
        "kinds": kinds,
    }


def _calibrate_threshold(entity_scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Choose the exact F1-optimal entity-wins-equality threshold.

    Scores are swept once in descending order.  This evaluates every distinct
    decision boundary without an approximate quantile cap or an O(N^2) mask per
    candidate threshold.
    """
    if entity_scores.ndim != 2 or entity_scores.shape[0] == 0:
        raise ValueError("threshold calibration requires a non-empty validation split")
    max_scores = entity_scores.max(dim=1).values
    fault_truth = labels != 0
    order = torch.argsort(max_scores, descending=True, stable=True)
    sorted_scores = max_scores[order]
    sorted_truth = fault_truth[order]
    positive_total = int(fault_truth.sum())
    best_threshold = float(
        torch.nextafter(
            sorted_scores[0],
            torch.full_like(sorted_scores[0], torch.inf),
        )
    )
    best_f1 = 0.0
    best_precision = 0.0
    tp = 0
    fp = 0
    index = 0
    while index < sorted_scores.numel():
        score = sorted_scores[index]
        end = index + 1
        while end < sorted_scores.numel() and bool(sorted_scores[end] == score):
            end += 1
        group_truth = sorted_truth[index:end]
        tp += int(group_truth.sum())
        fp += int((~group_truth).sum())
        fn = positive_total - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        threshold = float(score)
        if (f1, precision, threshold) > (best_f1, best_precision, best_threshold):
            best_f1 = f1
            best_precision = precision
            best_threshold = threshold
        index = end
    return best_threshold, best_f1


def _threshold_metrics(
    entity_scores: torch.Tensor,
    labels: torch.Tensor,
    kinds: Sequence[str],
    threshold: float,
) -> dict[str, object]:
    """Candidate ranking and derived detection metrics from pseudo-logits.

    The `none` candidate receives the calibrated threshold as its score, so the
    top-ranked decision reproduces the threshold rule and the ranking metrics are
    directly comparable with the learned model's evaluation output.
    """
    threshold_tensor = torch.tensor(threshold, dtype=entity_scores.dtype)
    none_score = torch.nextafter(threshold_tensor, torch.full_like(threshold_tensor, -torch.inf))
    none_column = none_score.expand(entity_scores.shape[0], 1)
    logits = torch.cat([none_column, entity_scores], dim=1)
    order = rank_candidates(logits)
    predictions = order[:, 0]
    ranks = []
    node_correct = 0
    edge_correct = 0
    node_total = 0
    edge_total = 0
    for index, target in enumerate(labels.tolist()):
        rank = int((order[index] == target).nonzero(as_tuple=False)[0, 0]) + 1
        ranks.append(rank)
        if kinds[index] == "node":
            node_total += 1
            node_correct += int(predictions[index] == target)
        elif kinds[index] == "edge":
            edge_total += 1
            edge_correct += int(predictions[index] == target)
    top3 = order[:, : min(3, logits.shape[1])]
    fault_truth = labels != 0
    fault_pred = predictions != 0
    tp = int((fault_pred & fault_truth).sum())
    fp = int((fault_pred & ~fault_truth).sum())
    fn = int((~fault_pred & fault_truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = max(1, labels.shape[0])
    return {
        "candidate_top1_accuracy": float((predictions == labels).float().mean())
        if labels.numel()
        else 0.0,
        "candidate_top3_accuracy": float((top3 == labels.unsqueeze(1)).any(dim=1).float().mean())
        if labels.numel()
        else 0.0,
        "mean_rank": float(sum(ranks) / len(ranks)) if ranks else 0.0,
        "mrr": float(sum(1.0 / rank for rank in ranks) / len(ranks)) if ranks else 0.0,
        "node_candidate_accuracy": float(node_correct / node_total) if node_total else 0.0,
        "edge_candidate_accuracy": float(edge_correct / edge_total) if edge_total else 0.0,
        "fault_present_precision": precision,
        "fault_present_recall": recall,
        "fault_present_f1": f1,
        "sample_count": int(total) if labels.numel() else 0,
    }


def evaluate_threshold_baseline(
    data_dir: Path, output_path: Path | None = None
) -> dict[str, object]:
    """Evaluate the learning-free threshold-ranking baseline on a residual dataset.

    Each candidate entity is scored by the mean absolute value of its own
    residual features over the window. The no-fault threshold is calibrated on
    the validation split only. The test split is used exactly once.
    """
    if not has_residual_run_shards(data_dir):
        raise ValueError(f"{data_dir} must contain sharded residual dataset artifacts")
    residual_index = load_residual_index(data_dir)
    splits = read_json(data_dir / "residual_splits.json")
    topology = hydrate_topology(read_json(data_dir / "topology.json"))
    rows = load_window_index(data_dir / RESIDUAL_WINDOW_INDEX)
    rows_by_split = {
        split: [row for row in rows if str(row["run_id"]) in {str(run_id) for run_id in run_ids}]
        for split, run_ids in splits.items()
    }
    val = _split_window_scores(data_dir, rows_by_split["val"], topology)
    threshold, val_f1 = _calibrate_threshold(val["entity_scores"], val["labels"])
    test = _split_window_scores(data_dir, rows_by_split["test"], topology)
    metrics = _threshold_metrics(test["entity_scores"], test["labels"], test["kinds"], threshold)
    summary = {
        **artifact_metadata("threshold_evaluation"),
        "baseline": "threshold_residual_ranking",
        "data_dir": str(data_dir),
        "residual_mode": residual_index.get("residual_mode"),
        "threshold": threshold,
        "val_fault_present_f1": val_f1,
        "test_metrics": metrics,
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary
