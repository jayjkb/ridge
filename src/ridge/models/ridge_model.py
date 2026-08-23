"""RIDGE model, objectives, training, and evaluation."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ridge.common.contracts import RIDGE_EARLY_STOPPING_METRICS
from ridge.common.io import (
    read_json,
    write_csv_rows,
    write_json,
)
from ridge.models.artifacts import (
    artifact_metadata,
    require_compatible_ridge_checkpoint,
    require_valid_splits,
    validate_artifact,
)
from ridge.models.dataset_io import (
    DEFAULT_SHARD_CACHE_SIZE,
    DEFAULT_TRAIN_PREFETCH_FACTOR,
    RESIDUAL_WINDOW_INDEX,
    build_run_aware_loader,
    has_residual_run_shards,
    load_residual_index,
    load_window_index,
    set_seed,
)
from ridge.models.episode_graph import (
    hydrate_topology,
)
from ridge.models.features import (
    CANDIDATE_TYPE_TO_INDEX,
    EDGE_DYNAMIC_FEATURES,
    EDGE_STATIC_FEATURES,
    FAULT_CATEGORY_LABELS,
    NODE_DYNAMIC_FEATURES,
    NODE_STATIC_FEATURES,
    PROBE_FEATURES,
    feature_schema,
)
from ridge.models.residual_data import (
    ResidualGraphBatch,
    ResidualGraphWindowDataset,
    collate_residual_graph_batch,
)


class GraphSAGEFlatLayer(nn.Module):
    """Mean-aggregation GraphSAGE layer over one flat snapshot of a concatenated batch, safe on empty graphs."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Combine each node's own projection with the projected mean of its neighbors."""
        if x.shape[0] == 0:
            return x.new_zeros((0, self.self_linear.out_features))
        src, dst = edge_index
        neighbor_sum = x.new_zeros(x.shape)
        if src.numel():
            neighbor_sum.index_add_(0, dst, x[src])
        degrees = x.new_zeros((x.shape[0],))
        if dst.numel():
            degrees.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        neighbor_mean = neighbor_sum / degrees.clamp_min(1.0).unsqueeze(-1)
        return self.self_linear(x) + self.neighbor_linear(neighbor_mean)


def _segment_mean(values: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Average rows by their graph assignment, returning zeros for graphs without rows."""
    if values.shape[0] == 0:
        return values.new_zeros((num_graphs, values.shape[-1]))
    pooled = values.new_zeros((num_graphs, values.shape[-1]))
    pooled.index_add_(0, batch_index, values)
    counts = values.new_zeros((num_graphs,))
    counts.index_add_(0, batch_index, torch.ones_like(batch_index, dtype=values.dtype))
    return pooled / counts.clamp_min(1.0).unsqueeze(-1)


class RIDGE(nn.Module):
    """RCA model that encodes residual histories, scores the catalog candidates of each graph, and classifies the fault category."""

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        probe_input_dim: int,
        node_recon_dim: int,
        edge_recon_dim: int,
        probe_recon_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_categories: int = len(FAULT_CATEGORY_LABELS),
        use_graph_structure: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_graph_structure = use_graph_structure
        if use_graph_structure:
            self.sage_layers = nn.ModuleList(
                [
                    GraphSAGEFlatLayer(node_input_dim, hidden_dim),
                    GraphSAGEFlatLayer(hidden_dim, hidden_dim),
                ]
            )
            edge_projection_input = hidden_dim * 3 + edge_input_dim
            probe_projection_input = hidden_dim * 2 + probe_input_dim
            edge_scorer_input = hidden_dim * 6
        else:
            # Entities are encoded from their own features only, with no message passing and no endpoint conditioning.
            self.node_layers = nn.ModuleList(
                [nn.Linear(node_input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)]
            )
            edge_projection_input = edge_input_dim
            probe_projection_input = probe_input_dim
            edge_scorer_input = hidden_dim * 4
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_projection_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.probe_projection = nn.Sequential(
            nn.Linear(probe_projection_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.node_gru = nn.GRU(hidden_dim, hidden_dim)
        self.edge_gru = nn.GRU(hidden_dim, hidden_dim)
        self.probe_gru = nn.GRU(hidden_dim, hidden_dim)
        self.none_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.node_candidate_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_candidate_scorer = nn.Sequential(
            nn.Linear(edge_scorer_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_categories)
        )
        self.node_reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, node_recon_dim)
        )
        self.edge_reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, edge_recon_dim)
        )
        self.probe_reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, probe_recon_dim)
        )

    def _encode_nodes(self, batch: ResidualGraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode every residual snapshot's nodes, with or without message passing, and summarize each node over time."""
        steps = int(batch.x_node.shape[0])
        snapshots = []
        for step in range(steps):
            h = batch.x_node[step]
            if self.use_graph_structure:
                for layer in self.sage_layers:
                    h = self.dropout(torch.relu(layer(h, batch.edge_index)))
            else:
                for layer in self.node_layers:
                    h = self.dropout(torch.relu(layer(h)))
            snapshots.append(h)
        sequence = torch.stack(snapshots, dim=0)
        if sequence.shape[1] == 0:
            return sequence, sequence.new_zeros((0, self.hidden_dim))
        _, hidden = self.node_gru(sequence)
        return sequence, hidden.squeeze(0)

    def _encode_entity_sequence(self, sequence: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
        """Run one GRU over each entity's snapshot sequence and return its final hidden state."""
        if sequence.shape[1] == 0:
            return sequence.new_zeros((0, self.hidden_dim))
        _, hidden = gru(sequence)
        return hidden.squeeze(0)

    def forward(self, batch: ResidualGraphBatch) -> dict[str, torch.Tensor]:
        """Return the candidate scores, fault-category logits, and residual reconstructions of a batch."""
        node_steps, node_h = self._encode_nodes(batch)
        src = batch.edge_endpoints[:, 0]
        dst = batch.edge_endpoints[:, 1]
        if batch.x_edge.shape[1] == 0:
            edge_step_h = batch.x_edge.new_zeros((batch.x_edge.shape[0], 0, self.hidden_dim))
        elif self.use_graph_structure:
            src_spatial = node_steps[:, src, :]
            dst_spatial = node_steps[:, dst, :]
            edge_step_h = self.edge_projection(
                torch.cat(
                    [src_spatial, dst_spatial, (src_spatial - dst_spatial).abs(), batch.x_edge],
                    dim=-1,
                )
            )
        else:
            edge_step_h = self.edge_projection(batch.x_edge)
        edge_h = self._encode_entity_sequence(edge_step_h, self.edge_gru)

        probe_src = batch.probe_endpoints[:, 0]
        probe_dst = batch.probe_endpoints[:, 1]
        if batch.x_probe.shape[1] == 0:
            probe_step_h = batch.x_probe.new_zeros((batch.x_probe.shape[0], 0, self.hidden_dim))
        elif self.use_graph_structure:
            probe_step_h = self.probe_projection(
                torch.cat(
                    [node_steps[:, probe_src, :], node_steps[:, probe_dst, :], batch.x_probe],
                    dim=-1,
                )
            )
        else:
            probe_step_h = self.probe_projection(batch.x_probe)
        probe_h = self._encode_entity_sequence(probe_step_h, self.probe_gru)

        graph_h = torch.cat(
            [
                _segment_mean(node_h, batch.node_batch, batch.num_graphs),
                _segment_mean(edge_h, batch.edge_batch, batch.num_graphs),
                _segment_mean(probe_h, batch.probe_batch, batch.num_graphs),
            ],
            dim=-1,
        )

        none_logits = self.none_scorer(graph_h).squeeze(-1)
        candidate_logits_tensor = graph_h.new_empty((batch.total_candidates,))
        none_mask = batch.candidate_types == CANDIDATE_TYPE_TO_INDEX["none"]
        node_mask = batch.candidate_types == CANDIDATE_TYPE_TO_INDEX["node"]
        edge_mask = batch.candidate_types == CANDIDATE_TYPE_TO_INDEX["edge"]
        if none_mask.any():
            candidate_logits_tensor[none_mask] = none_logits[batch.candidate_batch[none_mask]]
        if node_mask.any():
            entity_indices = batch.candidate_entity_indices[node_mask]
            graph_indices = batch.candidate_batch[node_mask]
            node_inputs = torch.cat([node_h[entity_indices], graph_h[graph_indices]], dim=-1)
            candidate_logits_tensor[node_mask] = self.node_candidate_scorer(node_inputs).squeeze(-1)
        if edge_mask.any():
            entity_indices = batch.candidate_entity_indices[edge_mask]
            graph_indices = batch.candidate_batch[edge_mask]
            if self.use_graph_structure:
                endpoints = batch.edge_endpoints[entity_indices]
                edge_inputs = torch.cat(
                    [
                        edge_h[entity_indices],
                        node_h[endpoints[:, 0]],
                        node_h[endpoints[:, 1]],
                        graph_h[graph_indices],
                    ],
                    dim=-1,
                )
            else:
                edge_inputs = torch.cat([edge_h[entity_indices], graph_h[graph_indices]], dim=-1)
            candidate_logits_tensor[edge_mask] = self.edge_candidate_scorer(edge_inputs).squeeze(-1)
        return {
            "candidate_logits": candidate_logits_tensor,
            "candidate_ptr": batch.candidate_ptr,
            "category_logits": self.category_head(graph_h),
            "node_recon": self.node_reconstruction_head(node_h),
            "edge_recon": self.edge_reconstruction_head(edge_h),
            "probe_recon": self.probe_reconstruction_head(probe_h),
        }


@dataclass
class RidgeTrainConfig:
    """Training settings of the RCA model, including the reconstruction coefficient and early-stopping metric."""

    data_dir: str
    output_dir: str
    seed: int = 42
    epochs: int = 60
    batch_size: int = 32
    hidden_dim: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    device: str = "cpu"
    lambda_recon: float = 0.5
    early_stopping_metric: str = "fault_present_f1"
    use_graph_structure: bool = True
    workers: int = 0
    prefetch_factor: int = DEFAULT_TRAIN_PREFETCH_FACTOR
    persistent_workers: bool = True
    shard_cache_size: int = DEFAULT_SHARD_CACHE_SIZE


def _macro_f1(
    predictions: Sequence[int],
    targets: Sequence[int],
    labels: Sequence[int],
    label_names: Sequence[str] | None = None,
) -> tuple[float, list[dict[str, object]]]:
    """Return the macro-averaged F1 over the labels with per-class precision, recall, F1, and support."""
    per_class = []
    f1_values = []
    resolved_label_names = (
        list(label_names) if label_names is not None else [str(label) for label in labels]
    )
    for label in labels:
        tp = sum(
            int(pred == label and target == label) for pred, target in zip(predictions, targets)
        )
        fp = sum(
            int(pred == label and target != label) for pred, target in zip(predictions, targets)
        )
        fn = sum(
            int(pred != label and target == label) for pred, target in zip(predictions, targets)
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "label": label,
                "label_name": resolved_label_names[int(label)]
                if 0 <= int(label) < len(resolved_label_names)
                else str(label),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
            }
        )
    return (sum(f1_values) / len(f1_values) if f1_values else 0.0), per_class


def _inverse_frequency_weights(
    samples: Sequence[dict[str, object]],
    run_ids: Sequence[str],
    label_key: str,
    class_count: int,
) -> torch.Tensor:
    """Return mean-one inverse-frequency class weights from the labels of the selected episodes."""
    counts = torch.zeros(class_count, dtype=torch.float32)
    selected_run_ids = set(str(run_id) for run_id in run_ids)
    for sample in samples:
        if str(sample["run_id"]) not in selected_run_ids:
            continue
        counts[int(sample[label_key])] += 1.0
    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = counts[nonzero].sum() / counts[nonzero]
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return weights


def _graph_candidate_cross_entropy(
    candidate_logits: torch.Tensor,
    candidate_ptr: torch.Tensor,
    candidate_labels: torch.Tensor,
) -> torch.Tensor:
    """Average the cross-entropy of each graph's softmax over its own catalog."""
    losses = []
    for graph_index in range(int(candidate_labels.shape[0])):
        start = int(candidate_ptr[graph_index])
        end = int(candidate_ptr[graph_index + 1])
        losses.append(
            F.cross_entropy(
                candidate_logits[start:end].unsqueeze(0), candidate_labels[graph_index].unsqueeze(0)
            )
        )
    return torch.stack(losses).mean()


def _mse_loss_or_zero(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean squared error, or a differentiable zero when either tensor is empty."""
    if prediction.numel() == 0 or target.numel() == 0:
        return prediction.sum() * 0.0
    return F.mse_loss(prediction, target)


def _l1_loss_or_zero(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean absolute error, or a differentiable zero when either tensor is empty."""
    if prediction.numel() == 0 or target.numel() == 0:
        return prediction.sum() * 0.0
    return F.l1_loss(prediction, target)


def _ridge_graph_loss(
    outputs: dict[str, torch.Tensor],
    batch: ResidualGraphBatch,
    lambda_recon: float,
    category_class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine the root-cause, weighted fault-category, and scaled reconstruction losses and report each."""
    rank_loss = _graph_candidate_cross_entropy(
        outputs["candidate_logits"], batch.candidate_ptr, batch.candidate_labels
    )
    type_loss = F.cross_entropy(
        outputs["category_logits"], batch.category_labels, weight=category_class_weights
    )
    node_recon_loss = _mse_loss_or_zero(outputs["node_recon"], batch.y_node)
    edge_recon_loss = _mse_loss_or_zero(outputs["edge_recon"], batch.y_edge)
    probe_recon_loss = _mse_loss_or_zero(outputs["probe_recon"], batch.y_probe)
    recon_loss = node_recon_loss + edge_recon_loss + probe_recon_loss
    total = rank_loss + type_loss + lambda_recon * recon_loss
    return total, {
        "rank_loss": float(rank_loss.detach()),
        "type_loss": float(type_loss.detach()),
        "recon_loss": float(recon_loss.detach()),
        "node_recon_loss": float(node_recon_loss.detach()),
        "edge_recon_loss": float(edge_recon_loss.detach()),
        "probe_recon_loss": float(probe_recon_loss.detach()),
    }


def _graph_logits_slice(result: dict[str, object], index: int) -> torch.Tensor:
    """Return the candidate scores of one window from the concatenated epoch outputs."""
    candidate_ptr = result["candidate_ptr"]
    start = int(candidate_ptr[index])
    end = int(candidate_ptr[index + 1])
    return result["candidate_logits"][start:end]


def rank_candidates(logits: torch.Tensor) -> torch.Tensor:
    """Order candidates by decreasing score, breaking exact ties by catalog position."""
    return torch.argsort(logits, dim=-1, descending=True, stable=True)


def _candidate_ranking_metrics_graph(result: dict[str, object]) -> dict[str, object]:
    """Compute top-1 and top-3 accuracy, mean rank, MRR, and device and link top-1 accuracy over the windows."""
    targets = result["targets"]
    kinds = result["root_cause_kind"]
    top1_correct = []
    top3_correct = []
    ranks = []
    node_correct = 0
    edge_correct = 0
    node_total = 0
    edge_total = 0
    for index, target_tensor in enumerate(targets.tolist()):
        logits = _graph_logits_slice(result, index)
        target = int(target_tensor)
        predictions = rank_candidates(logits)
        pred_top1 = int(predictions[0])
        rank = int((predictions == target).nonzero(as_tuple=False)[0, 0]) + 1
        ranks.append(rank)
        top1_correct.append(pred_top1 == target)
        top3_correct.append(target in predictions[: min(3, logits.shape[0])].tolist())
        if kinds[index] == "node":
            node_total += 1
            node_correct += int(pred_top1 == target)
        elif kinds[index] == "edge":
            edge_total += 1
            edge_correct += int(pred_top1 == target)
    total = max(1, len(ranks))
    return {
        "candidate_top1_accuracy": sum(int(value) for value in top1_correct) / total,
        "candidate_top3_accuracy": sum(int(value) for value in top3_correct) / total,
        "mean_rank": float(sum(ranks) / total) if ranks else 0.0,
        "mrr": float(sum(1.0 / rank for rank in ranks) / total) if ranks else 0.0,
        "node_candidate_accuracy": float(node_correct / node_total) if node_total else 0.0,
        "edge_candidate_accuracy": float(edge_correct / edge_total) if edge_total else 0.0,
    }


def ridge_graph_metrics(result: dict[str, object]) -> dict[str, object]:
    """Compute the ranking, fault-presence, fault-category, joint, and reconstruction metrics of an epoch."""
    targets = result["targets"]
    category_logits = result["category_logits"]
    category_targets = result["fault_category_targets"]
    category_predictions = category_logits.argmax(dim=1).tolist()
    category_targets_list = category_targets.tolist()
    category_macro_f1, per_class = _macro_f1(
        category_predictions,
        category_targets_list,
        list(range(len(FAULT_CATEGORY_LABELS))),
        list(FAULT_CATEGORY_LABELS),
    )
    pred_top1 = []
    for index in range(int(targets.shape[0])):
        pred_top1.append(int(rank_candidates(_graph_logits_slice(result, index))[0]))
    fault_truth = [int(target) != 0 for target in targets.tolist()]
    fault_pred = [prediction != 0 for prediction in pred_top1]
    tp = sum(int(pred and truth) for pred, truth in zip(fault_pred, fault_truth))
    fp = sum(int(pred and not truth) for pred, truth in zip(fault_pred, fault_truth))
    fn = sum(int((not pred) and truth) for pred, truth in zip(fault_pred, fault_truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fault_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ranking = _candidate_ranking_metrics_graph(result)
    node_recon_mse = float(_mse_loss_or_zero(result["node_recon"], result["y_node"]))
    edge_recon_mse = float(_mse_loss_or_zero(result["edge_recon"], result["y_edge"]))
    probe_recon_mse = float(_mse_loss_or_zero(result["probe_recon"], result["y_probe"]))
    node_recon_mae = float(_l1_loss_or_zero(result["node_recon"], result["y_node"]))
    edge_recon_mae = float(_l1_loss_or_zero(result["edge_recon"], result["y_edge"]))
    probe_recon_mae = float(_l1_loss_or_zero(result["probe_recon"], result["y_probe"]))
    joint_top1 = 0
    for index, target in enumerate(targets.tolist()):
        joint_top1 += int(
            pred_top1[index] == int(target)
            and category_predictions[index] == category_targets_list[index]
        )
    denominator = max(1, len(category_predictions))
    return {
        **ranking,
        "fault_present_precision": precision,
        "fault_present_recall": recall,
        "fault_present_f1": fault_f1,
        "category_accuracy": sum(
            int(pred == target) for pred, target in zip(category_predictions, category_targets_list)
        )
        / denominator,
        "category_macro_f1": category_macro_f1,
        "category_per_class_f1": per_class,
        "joint_top1_entity_and_category_accuracy": joint_top1 / denominator,
        "residual_reconstruction_mse": node_recon_mse + edge_recon_mse + probe_recon_mse,
        "residual_reconstruction_mae": node_recon_mae + edge_recon_mae + probe_recon_mae,
        "node_residual_reconstruction_mse": node_recon_mse,
        "edge_residual_reconstruction_mse": edge_recon_mse,
        "probe_residual_reconstruction_mse": probe_recon_mse,
    }


def _run_ridge_graph_epoch(
    model: RIDGE,
    loader: DataLoader,
    device: torch.device,
    lambda_recon: float,
    optimizer: torch.optim.Optimizer | None = None,
    category_class_weights: torch.Tensor | None = None,
    *,
    collect_outputs: bool = True,
) -> dict[str, object]:
    """Run one pass over a loader, stepping the optimizer when given, and collect the outputs needed for metrics."""
    model.train(optimizer is not None)
    losses = []
    outputs = defaultdict(list)
    candidate_ptr = [0]
    for batch in loader:
        device_batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        model_output = model(device_batch)
        loss, _metrics = _ridge_graph_loss(
            model_output,
            device_batch,
            lambda_recon,
            category_class_weights,
        )
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
        if not collect_outputs:
            continue
        outputs["candidate_logits"].append(model_output["candidate_logits"].detach().cpu())
        ptr_offset = candidate_ptr[-1]
        candidate_ptr.extend(
            [
                ptr_offset + int(value)
                for value in model_output["candidate_ptr"].detach().cpu()[1:].tolist()
            ]
        )
        outputs["category_logits"].append(model_output["category_logits"].detach().cpu())
        outputs["node_recon"].append(model_output["node_recon"].detach().cpu())
        outputs["edge_recon"].append(model_output["edge_recon"].detach().cpu())
        outputs["probe_recon"].append(model_output["probe_recon"].detach().cpu())
        outputs["targets"].append(device_batch.candidate_labels.detach().cpu())
        outputs["fault_category_targets"].append(device_batch.category_labels.detach().cpu())
        outputs["y_node"].append(device_batch.y_node.detach().cpu())
        outputs["y_edge"].append(device_batch.y_edge.detach().cpu())
        outputs["y_probe"].append(device_batch.y_probe.detach().cpu())
        for graph_index in range(batch.num_graphs):
            node_mask = device_batch.node_batch == graph_index
            edge_mask = device_batch.edge_batch == graph_index
            probe_mask = device_batch.probe_batch == graph_index
            node_error = _mse_loss_or_zero(
                model_output["node_recon"][node_mask], device_batch.y_node[node_mask]
            )
            edge_error = _mse_loss_or_zero(
                model_output["edge_recon"][edge_mask], device_batch.y_edge[edge_mask]
            )
            probe_error = _mse_loss_or_zero(
                model_output["probe_recon"][probe_mask], device_batch.y_probe[probe_mask]
            )
            residual_norm = torch.cat(
                [
                    device_batch.y_node[node_mask].reshape(-1),
                    device_batch.y_edge[edge_mask].reshape(-1),
                    device_batch.y_probe[probe_mask].reshape(-1),
                ]
            ).norm()
            outputs["reconstruction_error"].append(
                float((node_error + edge_error + probe_error).detach().cpu())
            )
            outputs["residual_norm"].append(float(residual_norm.detach().cpu()))
        outputs["run_id"].extend(batch.run_id)
        outputs["timestamp"].extend(batch.timestamp)
        outputs["fault_category_label"].extend(batch.fault_category_label)
        outputs["root_cause_kind"].extend(batch.root_cause_kind)
        outputs["root_cause_id"].extend(batch.root_cause_id)
        outputs["candidate_ids"].extend(batch.candidate_ids)
        outputs["candidate_type_labels"].extend(batch.candidate_type_labels)
    if not losses:
        raise ValueError("RIDGE graph loader contained no samples")
    if not collect_outputs:
        return {"loss": sum(losses) / len(losses)}
    return {
        "loss": sum(losses) / len(losses),
        "candidate_logits": torch.cat(outputs["candidate_logits"]),
        "candidate_ptr": torch.tensor(candidate_ptr, dtype=torch.long),
        "category_logits": torch.cat(outputs["category_logits"]),
        "node_recon": torch.cat(outputs["node_recon"]),
        "edge_recon": torch.cat(outputs["edge_recon"]),
        "probe_recon": torch.cat(outputs["probe_recon"]),
        "targets": torch.cat(outputs["targets"]),
        "fault_category_targets": torch.cat(outputs["fault_category_targets"]),
        "y_node": torch.cat(outputs["y_node"]),
        "y_edge": torch.cat(outputs["y_edge"]),
        "y_probe": torch.cat(outputs["y_probe"]),
        "run_id": list(outputs["run_id"]),
        "timestamp": list(outputs["timestamp"]),
        "fault_category_label": list(outputs["fault_category_label"]),
        "root_cause_kind": list(outputs["root_cause_kind"]),
        "root_cause_id": list(outputs["root_cause_id"]),
        "candidate_ids": list(outputs["candidate_ids"]),
        "candidate_type_labels": list(outputs["candidate_type_labels"]),
        "reconstruction_error": list(outputs["reconstruction_error"]),
        "residual_norm": list(outputs["residual_norm"]),
    }


def _ridge_loader(
    data_dir: Path,
    run_ids: Sequence[str],
    topology: dict[str, Any],
    config: RidgeTrainConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    """Build the data loader of residual graph windows for the given episodes."""
    dataset = ResidualGraphWindowDataset(
        data_dir,
        run_ids,
        topology,
        cache_size=config.shard_cache_size,
    )
    return build_run_aware_loader(
        dataset,
        collate_fn=collate_residual_graph_batch,
        batch_size=config.batch_size,
        seed=config.seed,
        workers=config.workers,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=shuffle,
    )


def train_ridge(config: RidgeTrainConfig) -> dict[str, object]:
    """Train the RCA model with early stopping on a validation metric and save the best checkpoint with its metrics."""
    set_seed(config.seed)
    if config.early_stopping_metric not in RIDGE_EARLY_STOPPING_METRICS:
        raise ValueError(
            f"early_stopping_metric must be one of {RIDGE_EARLY_STOPPING_METRICS}; "
            f"got {config.early_stopping_metric!r}"
        )
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not has_residual_run_shards(data_dir):
        raise ValueError(f"{data_dir} must contain sharded residual dataset artifacts")
    residual_index = load_residual_index(data_dir)
    splits = read_json(data_dir / "residual_splits.json")
    topology = hydrate_topology(read_json(data_dir / "topology.json"))
    rows = load_window_index(data_dir / RESIDUAL_WINDOW_INDEX)
    require_valid_splits(
        splits,
        source=data_dir / "residual_splits.json",
        run_ids={str(row["run_id"]) for row in rows},
    )
    loaders = {
        split: _ridge_loader(
            data_dir,
            splits[split],
            topology,
            config,
            shuffle=(split == "train"),
        )
        for split in ("train", "val")
    }
    category_class_weights = _inverse_frequency_weights(
        rows, splits["train"], "fault_category_index", len(FAULT_CATEGORY_LABELS)
    ).to(config.device)
    model = RIDGE(
        node_input_dim=len(NODE_DYNAMIC_FEATURES) + len(NODE_STATIC_FEATURES),
        edge_input_dim=len(EDGE_DYNAMIC_FEATURES) + len(EDGE_STATIC_FEATURES),
        probe_input_dim=len(PROBE_FEATURES),
        node_recon_dim=len(NODE_DYNAMIC_FEATURES),
        edge_recon_dim=len(EDGE_DYNAMIC_FEATURES),
        probe_recon_dim=len(PROBE_FEATURES),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_categories=len(FAULT_CATEGORY_LABELS),
        use_graph_structure=config.use_graph_structure,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best_score = -1.0
    best_top3 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    patience = 0

    for epoch in range(config.epochs):
        train_result = _run_ridge_graph_epoch(
            model,
            loaders["train"],
            torch.device(config.device),
            config.lambda_recon,
            optimizer,
            category_class_weights,
            collect_outputs=False,
        )
        with torch.no_grad():
            val_result = _run_ridge_graph_epoch(
                model,
                loaders["val"],
                torch.device(config.device),
                config.lambda_recon,
                None,
                category_class_weights,
            )
        val_metrics = ridge_graph_metrics(val_result)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "val_fault_present_f1": val_metrics["fault_present_f1"],
                "val_top3_candidate_accuracy": val_metrics["candidate_top3_accuracy"],
                "val_mrr": val_metrics["mrr"],
            }
        )
        score = float(val_metrics[config.early_stopping_metric])
        top3_score = float(val_metrics["candidate_top3_accuracy"])
        if score > best_score or (math.isclose(score, best_score) and top3_score > best_top3):
            best_score = score
            best_top3 = top3_score
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model_config = {
        "node_input_dim": len(NODE_DYNAMIC_FEATURES) + len(NODE_STATIC_FEATURES),
        "edge_input_dim": len(EDGE_DYNAMIC_FEATURES) + len(EDGE_STATIC_FEATURES),
        "probe_input_dim": len(PROBE_FEATURES),
        "node_recon_dim": len(NODE_DYNAMIC_FEATURES),
        "edge_recon_dim": len(EDGE_DYNAMIC_FEATURES),
        "probe_recon_dim": len(PROBE_FEATURES),
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "num_categories": len(FAULT_CATEGORY_LABELS),
        "use_graph_structure": config.use_graph_structure,
    }
    checkpoint = {
        **artifact_metadata("ridge_checkpoint"),
        "model_type": "ridge",
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "feature_schema": feature_schema(),
        "fault_category_labels": list(FAULT_CATEGORY_LABELS),
        "source_residual_data_dir": str(data_dir.resolve()),
        "candidate_ids": list(topology["candidate_ids"]),
        "candidate_kinds": list(topology["candidate_kinds"]),
        "residual_mode": residual_index["residual_mode"],
        "emulator_history_len": residual_index["emulator_history_len"],
        "prediction_horizon": residual_index["prediction_horizon"],
        "residual_history_len": residual_index["residual_history_len"],
        "lambda_recon": config.lambda_recon,
    }
    checkpoint_path = output_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    write_json(
        output_dir / "metrics.json",
        {
            "best_val_metric_name": config.early_stopping_metric,
            "best_val_metric_value": best_score,
            "best_val_top3_candidate_accuracy": best_top3,
            "category_class_weights": category_class_weights.detach().cpu().tolist(),
        },
    )
    write_csv_rows(output_dir / "training_history.csv", history)
    return {
        "checkpoint": str(checkpoint_path),
        "best_val_metric_name": config.early_stopping_metric,
        "best_val_metric_value": best_score,
        "best_val_top3_candidate_accuracy": best_top3,
    }


def evaluate_ridge_checkpoint(
    data_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 64,
    device: str = "cpu",
    top_k: int = 3,
    prediction_output_path: Path | None = None,
) -> dict[str, object]:
    """Validate a checkpoint against the residual dataset and evaluate it once on the test split."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not has_residual_run_shards(data_dir):
        raise ValueError(f"{data_dir} must contain sharded residual dataset artifacts")
    residual_index = load_residual_index(data_dir)
    splits = read_json(data_dir / "residual_splits.json")
    topology = hydrate_topology(read_json(data_dir / "topology.json"))
    candidate_count = len(topology["candidate_ids"])
    if top_k > candidate_count:
        raise ValueError(
            f"top_k={top_k} exceeds the residual dataset candidate count ({candidate_count})"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_artifact(checkpoint, "ridge_checkpoint", source=checkpoint_path)
    require_valid_splits(splits, source=data_dir / "residual_splits.json")
    require_compatible_ridge_checkpoint(
        checkpoint,
        residual_index,
        source=checkpoint_path,
        topology=topology,
    )
    return _evaluate_ridge_graph_checkpoint(
        data_dir,
        checkpoint,
        checkpoint_path,
        splits,
        topology,
        residual_index,
        batch_size,
        device,
        top_k,
        prediction_output_path,
    )


def _evaluate_ridge_graph_checkpoint(
    data_dir: Path,
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    splits: dict[str, list[str]],
    topology: dict[str, Any],
    residual_index: dict[str, Any],
    batch_size: int,
    device: str,
    top_k: int,
    prediction_output_path: Path | None,
) -> dict[str, object]:
    """Rebuild the model, score the test windows, time the inference, and export per-window predictions."""
    config = checkpoint["model_config"]
    model = RIDGE(
        node_input_dim=int(config["node_input_dim"]),
        edge_input_dim=int(config["edge_input_dim"]),
        probe_input_dim=int(config["probe_input_dim"]),
        node_recon_dim=int(config["node_recon_dim"]),
        edge_recon_dim=int(config["edge_recon_dim"]),
        probe_recon_dim=int(config["probe_recon_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        num_categories=int(config["num_categories"]),
        use_graph_structure=bool(config["use_graph_structure"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(
        ResidualGraphWindowDataset(data_dir, splits["test"], topology),
        batch_size=batch_size,
        collate_fn=collate_residual_graph_batch,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    start_time = time.perf_counter()
    with torch.no_grad():
        result = _run_ridge_graph_epoch(
            model,
            loader,
            torch.device(device),
            float(checkpoint.get("lambda_recon", RidgeTrainConfig.lambda_recon)),
        )
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    elapsed_seconds = time.perf_counter() - start_time
    metrics = ridge_graph_metrics(result)
    predictions_path = (
        prediction_output_path
        if prediction_output_path is not None
        else checkpoint_path.parent / "per_sample_ridge_predictions.csv"
    )
    _write_per_sample_graph_predictions(
        checkpoint.get("fault_category_labels", list(FAULT_CATEGORY_LABELS)),
        result,
        predictions_path,
        top_k,
    )
    window_count = int(result["targets"].shape[0])
    inference = {
        "total_seconds": elapsed_seconds,
        "windows": window_count,
        "windows_per_second": window_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "mean_latency_ms_per_window": (elapsed_seconds / window_count * 1000.0)
        if window_count
        else 0.0,
        "batch_size": batch_size,
        "device": device,
    }
    return {
        **artifact_metadata("ridge_evaluation"),
        "checkpoint": str(checkpoint_path),
        "prediction_export": str(predictions_path),
        "residual_mode": residual_index.get("residual_mode"),
        "split": "test",
        "top_k": top_k,
        "candidate_count": len(topology["candidate_ids"]),
        "test_metrics": metrics,
        "inference": inference,
    }


def _write_per_sample_graph_predictions(
    category_labels: Sequence[str],
    result: dict[str, object],
    output_path: Path,
    top_k: int,
) -> None:
    """Write one CSV row per test window with its true labels and ranked top-k candidates."""
    category_logits = result["category_logits"]
    targets = result["targets"]
    resolved_category_labels = (
        list(category_labels) if category_labels else list(FAULT_CATEGORY_LABELS)
    )
    rows = []
    for index in range(int(targets.shape[0])):
        logits = _graph_logits_slice(result, index)
        top_order = rank_candidates(logits)[: min(top_k, logits.shape[0])]
        candidate_ids = result["candidate_ids"][index]
        candidate_types = result["candidate_type_labels"][index]
        prediction_indices = [int(candidate_index) for candidate_index in top_order.tolist()]
        candidate_predictions = [
            candidate_ids[candidate_index] for candidate_index in prediction_indices
        ]
        candidate_prediction_types = [
            candidate_types[candidate_index] for candidate_index in prediction_indices
        ]
        true_candidate_index = int(targets[index])
        output_row = {
            "run_id": result["run_id"][index],
            "timestamp": result["timestamp"][index],
            "true_candidate_local_index": true_candidate_index,
            "true_candidate": candidate_ids[true_candidate_index],
            "true_candidate_kind": candidate_types[true_candidate_index],
            "true_category": result["fault_category_label"][index],
            "pred_category": resolved_category_labels[int(category_logits[index].argmax())],
            "candidate_top1_correct": int(prediction_indices[0] == true_candidate_index),
            "candidate_top3_correct": int(true_candidate_index in prediction_indices[:3]),
            "residual_norm": result["residual_norm"][index],
            "reconstruction_error": result["reconstruction_error"][index],
        }
        for rank, (candidate_index, candidate_id, candidate_kind) in enumerate(
            zip(prediction_indices, candidate_predictions, candidate_prediction_types), start=1
        ):
            output_row[f"pred_top{rank}_local_index"] = candidate_index
            output_row[f"pred_top{rank}"] = candidate_id
            output_row[f"pred_top{rank}_kind"] = candidate_kind
        rows.append(output_row)
    write_csv_rows(output_path, rows)
