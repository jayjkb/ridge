"""Normal-emulator architecture, training, and evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ridge.common.io import (
    read_json,
    write_csv_rows,
    write_json,
)
from ridge.models.artifacts import (
    artifact_metadata,
    require_compatible_normal_checkpoint,
    require_valid_splits,
    validate_artifact,
)
from ridge.models.dataset_io import (
    DEFAULT_SHARD_CACHE_SIZE,
    DEFAULT_TRAIN_DATALOADER_WORKERS,
    DEFAULT_TRAIN_PREFETCH_FACTOR,
    build_run_aware_loader,
    has_normal_run_shards,
    load_normal_index,
    set_seed,
)
from ridge.models.episode_graph import (
    hydrate_topology,
)
from ridge.models.features import (
    EDGE_DYNAMIC_FEATURES,
    EDGE_STATIC_FEATURES,
    NODE_DYNAMIC_FEATURES,
    NODE_STATIC_FEATURES,
    PROBE_BINARY_FEATURES,
    PROBE_CONTINUOUS_FEATURES,
    PROBE_FEATURES,
    feature_schema,
)
from ridge.models.normal_data import (
    NormalWindowDataset,
    collate_normal_batch,
)


class GraphSAGELayer(nn.Module):
    """Mean-aggregation GraphSAGE layer over a batch of snapshots laid out as [batch*steps, nodes, features]."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Combine each node's own projection with the projected mean of its neighbors."""
        src, dst = edge_index
        messages = x[:, src, :]
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(1, dst, messages)
        degrees = torch.zeros((x.shape[1],), dtype=x.dtype, device=x.device)
        degrees.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        neighbor_mean = neighbor_sum / degrees.clamp_min(1.0).view(1, -1, 1)
        return self.self_linear(x) + self.neighbor_linear(neighbor_mean)


class MultiBranchTemporalEncoder(nn.Module):
    """Encoder with GraphSAGE node layers, endpoint-conditioned edge and probe projections, and one GRU per entity family."""

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        probe_input_dim: int,
        edge_index: torch.Tensor,
        edge_endpoints: torch.Tensor,
        probe_endpoints: torch.Tensor,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_endpoints", edge_endpoints.long())
        self.register_buffer("probe_endpoints", probe_endpoints.long())
        self.sage_layers = nn.ModuleList(
            [GraphSAGELayer(node_input_dim, hidden_dim), GraphSAGELayer(hidden_dim, hidden_dim)]
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.probe_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + probe_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.node_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.edge_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.probe_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def encode_snapshots(self, node_x: torch.Tensor) -> torch.Tensor:
        """Apply the GraphSAGE layers to every snapshot of the batch and return the node representations."""
        batch_size, steps, node_count = node_x.shape[:3]
        h = node_x.reshape(batch_size * steps, node_count, node_x.shape[-1])
        for layer in self.sage_layers:
            h = self.dropout(torch.relu(layer(h, self.edge_index)))
        return h.reshape(batch_size, steps, node_count, self.hidden_dim)

    def _encode_entity_sequence(self, sequence: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
        """Run one GRU over each entity's snapshot sequence and return its final hidden state."""
        batch_size, steps, entity_count = sequence.shape[:3]
        flattened = sequence.permute(0, 2, 1, 3).reshape(
            batch_size * entity_count, steps, sequence.shape[-1]
        )
        _, hidden = gru(flattened)
        return hidden.squeeze(0).reshape(batch_size, entity_count, self.hidden_dim)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_x: torch.Tensor,
        probe_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the temporal summary of every node, edge, and probe over the input history."""
        node_spatial = self.encode_snapshots(node_x)
        node_h = self._encode_entity_sequence(node_spatial, self.node_gru)

        src = self.edge_endpoints[:, 0]
        dst = self.edge_endpoints[:, 1]
        src_spatial = node_spatial[:, :, src, :]
        dst_spatial = node_spatial[:, :, dst, :]
        edge_step_h = self.edge_projection(
            torch.cat([src_spatial, dst_spatial, (src_spatial - dst_spatial).abs(), edge_x], dim=-1)
        )
        edge_h = self._encode_entity_sequence(edge_step_h, self.edge_gru)

        probe_src = self.probe_endpoints[:, 0]
        probe_dst = self.probe_endpoints[:, 1]
        probe_step_h = self.probe_projection(
            torch.cat(
                [
                    node_spatial[:, :, probe_src, :],
                    node_spatial[:, :, probe_dst, :],
                    probe_x,
                ],
                dim=-1,
            )
        )
        probe_h = self._encode_entity_sequence(probe_step_h, self.probe_gru)
        return {
            "node_h": node_h,
            "edge_h": edge_h,
            "probe_h": probe_h,
        }


class NormalTemporalGraphEmulator(nn.Module):
    """Normal-behavior emulator that forecasts a mean and log standard deviation per numerical feature and a logit per binary probe feature."""

    def __init__(
        self,
        topology: dict[str, Any],
        node_input_dim: int,
        edge_input_dim: int,
        probe_input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MultiBranchTemporalEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            probe_input_dim=probe_input_dim,
            edge_index=torch.tensor(topology["edge_index"], dtype=torch.long),
            edge_endpoints=torch.tensor(topology["edge_endpoints"], dtype=torch.long),
            probe_endpoints=torch.tensor(topology["probe_endpoints"], dtype=torch.long),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.node_mean = nn.Linear(hidden_dim, len(NODE_DYNAMIC_FEATURES))
        self.node_log_std = nn.Linear(hidden_dim, len(NODE_DYNAMIC_FEATURES))
        self.edge_mean = nn.Linear(hidden_dim, len(EDGE_DYNAMIC_FEATURES))
        self.edge_log_std = nn.Linear(hidden_dim, len(EDGE_DYNAMIC_FEATURES))
        self.probe_mean = nn.Linear(hidden_dim, len(PROBE_CONTINUOUS_FEATURES))
        self.probe_log_std = nn.Linear(hidden_dim, len(PROBE_CONTINUOUS_FEATURES))
        self.probe_binary_logits = nn.Linear(hidden_dim, len(PROBE_BINARY_FEATURES))

    def forward(
        self, node_x: torch.Tensor, edge_x: torch.Tensor, probe_x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Encode the history and return the forecast distribution parameters with log standard deviations clipped to [-5, 3]."""
        encoded = self.encoder(node_x, edge_x, probe_x)
        return {
            "node_mean": self.node_mean(encoded["node_h"]),
            "node_log_std": self.node_log_std(encoded["node_h"]).clamp(-5.0, 3.0),
            "edge_mean": self.edge_mean(encoded["edge_h"]),
            "edge_log_std": self.edge_log_std(encoded["edge_h"]).clamp(-5.0, 3.0),
            "probe_mean": self.probe_mean(encoded["probe_h"]),
            "probe_log_std": self.probe_log_std(encoded["probe_h"]).clamp(-5.0, 3.0),
            "probe_binary_logits": self.probe_binary_logits(encoded["probe_h"]),
        }


def _gaussian_nll(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean Gaussian negative log-likelihood of the targets under the predicted mean and log standard deviation."""
    inv_var = torch.exp(-2.0 * log_std)
    return 0.5 * (((target - mean) ** 2) * inv_var + 2.0 * log_std).mean()


def normal_emulator_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum the unweighted mean negative log-likelihoods of the node, edge, numerical probe, and binary probe families."""
    node_loss = _gaussian_nll(outputs["node_mean"], outputs["node_log_std"], batch["target_node_y"])
    edge_loss = _gaussian_nll(outputs["edge_mean"], outputs["edge_log_std"], batch["target_edge_y"])
    probe_cont_target = batch["target_probe_y"][..., : len(PROBE_CONTINUOUS_FEATURES)]
    probe_bin_target = batch["target_probe_y"][..., len(PROBE_CONTINUOUS_FEATURES) :]
    probe_cont_loss = _gaussian_nll(
        outputs["probe_mean"], outputs["probe_log_std"], probe_cont_target
    )
    probe_bin_loss = F.binary_cross_entropy_with_logits(
        outputs["probe_binary_logits"], probe_bin_target
    )
    total = node_loss + edge_loss + probe_cont_loss + probe_bin_loss
    return total, {
        "node_loss": float(node_loss.detach()),
        "edge_loss": float(edge_loss.detach()),
        "probe_cont_loss": float(probe_cont_loss.detach()),
        "probe_bin_loss": float(probe_bin_loss.detach()),
    }


@dataclass
class NormalTrainConfig:
    """Training settings of the emulator, from data paths to optimizer, early stopping, and loader options."""
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
    workers: int = DEFAULT_TRAIN_DATALOADER_WORKERS
    prefetch_factor: int = DEFAULT_TRAIN_PREFETCH_FACTOR
    persistent_workers: bool = True
    shard_cache_size: int = DEFAULT_SHARD_CACHE_SIZE
    torch_threads: int | None = None
    torch_interop_threads: int | None = None


def _configure_torch_cpu_threads(config: NormalTrainConfig) -> None:
    """Apply the configured PyTorch thread counts when they are set."""
    if config.torch_threads is not None:
        torch.set_num_threads(max(1, int(config.torch_threads)))
    if config.torch_interop_threads is not None:
        torch.set_num_interop_threads(max(1, int(config.torch_interop_threads)))


def _normal_loader(
    data_dir: Path,
    run_ids: list[str],
    config: NormalTrainConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    """Build the data loader for the Stage-2 windows of the given episodes."""
    dataset = NormalWindowDataset(data_dir, run_ids, cache_size=config.shard_cache_size)
    return build_run_aware_loader(
        dataset,
        collate_fn=collate_normal_batch,
        batch_size=config.batch_size,
        seed=config.seed,
        workers=config.workers,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=shuffle,
    )


def train_normal_emulator(config: NormalTrainConfig) -> dict[str, object]:
    """Train the emulator with early stopping on validation loss and save the best checkpoint with its metrics."""
    set_seed(config.seed)
    _configure_torch_cpu_threads(config)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not has_normal_run_shards(data_dir):
        raise ValueError(f"{data_dir} must contain sharded normal dataset artifacts")
    normal_index = load_normal_index(data_dir)
    splits = read_json(data_dir / "normal_splits.json")
    topology = hydrate_topology(read_json(data_dir / "topology.json"))
    normalization = read_json(data_dir / "normalization.json")
    require_valid_splits(splits, source=data_dir / "normal_splits.json")
    loaders = {
        split: _normal_loader(
            data_dir,
            splits[split],
            config,
            shuffle=(split == "train"),
        )
        for split in ("train", "val")
    }
    model = NormalTemporalGraphEmulator(
        topology,
        node_input_dim=len(NODE_DYNAMIC_FEATURES) + len(NODE_STATIC_FEATURES),
        edge_input_dim=len(EDGE_DYNAMIC_FEATURES) + len(EDGE_STATIC_FEATURES),
        probe_input_dim=len(PROBE_FEATURES),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    patience = 0

    for epoch in range(config.epochs):
        train_result = _run_normal_epoch(
            model, loaders["train"], torch.device(config.device), optimizer
        )
        with torch.no_grad():
            val_result = _run_normal_epoch(model, loaders["val"], torch.device(config.device))
        history.append(
            {
                "epoch": epoch,
                **train_result["metrics"],
                **{f"val_{k}": v for k, v in val_result["metrics"].items()},
            }
        )
        if val_result["loss"] < best_loss:
            best_loss = float(val_result["loss"])
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
        "topology": topology,
        "node_input_dim": len(NODE_DYNAMIC_FEATURES) + len(NODE_STATIC_FEATURES),
        "edge_input_dim": len(EDGE_DYNAMIC_FEATURES) + len(EDGE_STATIC_FEATURES),
        "probe_input_dim": len(PROBE_FEATURES),
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
    }
    checkpoint = {
        **artifact_metadata("normal_checkpoint"),
        "model_type": "normal_emulator",
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "normalization": normalization,
        "feature_schema": feature_schema(),
        "source_normal_data_dir": str(data_dir.resolve()),
        "history_len": normal_index["history_len"],
        "prediction_horizon": normal_index["prediction_horizon"],
    }
    checkpoint_path = output_dir / "best_normal_emulator.pt"
    torch.save(checkpoint, checkpoint_path)
    write_json(output_dir / "normal_metrics.json", {"best_val_loss": best_loss})
    write_csv_rows(output_dir / "normal_training_history.csv", history)
    write_json(output_dir / "normal_feature_schema.json", feature_schema())
    return {"checkpoint": str(checkpoint_path), "best_val_loss": best_loss}


def _run_normal_epoch(
    model: NormalTemporalGraphEmulator,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    """Run one pass over a loader, stepping the optimizer when given, and return the mean losses."""
    model.train(optimizer is not None)
    losses = []
    metrics_accumulator = defaultdict(float)
    steps = 0
    for batch in loader:
        device_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(device_batch["node_x"], device_batch["edge_x"], device_batch["probe_x"])
        loss, metrics = normal_emulator_loss(outputs, device_batch)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
        for key, value in metrics.items():
            metrics_accumulator[key] += float(value)
        steps += 1
    if not steps:
        raise ValueError("Normal emulator loader contained no samples")
    metrics = {"loss": sum(losses) / len(losses)}
    for key, value in metrics_accumulator.items():
        metrics[key] = value / steps
    return {"loss": metrics["loss"], "metrics": metrics}


def make_normal_model_from_checkpoint(
    checkpoint: dict[str, object], device: str = "cpu"
) -> NormalTemporalGraphEmulator:
    """Rebuild the emulator from a checkpoint's configuration and weights in evaluation mode."""
    config = checkpoint["model_config"]
    model = NormalTemporalGraphEmulator(
        topology=config["topology"],
        node_input_dim=int(config["node_input_dim"]),
        edge_input_dim=int(config["edge_input_dim"]),
        probe_input_dim=int(config["probe_input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_normal_checkpoint(
    data_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 64,
    device: str = "cpu",
    output_path: Path | None = None,
) -> dict[str, object]:
    """Forecasting-quality evaluation of the normal emulator on the test split.

    Reports, per continuous output family, the RMSE and MAE of the mean forecast, the Gaussian negative log-likelihood, and the empirical coverage
    of the central two-standard-deviation prediction interval, plus the Bernoulli cross-entropy of the binary probe features.
    """
    if not has_normal_run_shards(data_dir):
        raise ValueError(f"{data_dir} must contain sharded normal dataset artifacts")
    normal_index = load_normal_index(data_dir)
    splits = read_json(data_dir / "normal_splits.json")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_artifact(checkpoint, "normal_checkpoint", source=checkpoint_path)
    require_valid_splits(splits, source=data_dir / "normal_splits.json")
    require_compatible_normal_checkpoint(
        checkpoint,
        normal_index,
        source=checkpoint_path,
        topology=read_json(data_dir / "topology.json"),
    )
    model = make_normal_model_from_checkpoint(checkpoint, device=device)
    loader = DataLoader(
        NormalWindowDataset(data_dir, splits["test"]),
        batch_size=batch_size,
        collate_fn=collate_normal_batch,
    )
    families = {
        name: {
            "squared_error": 0.0,
            "absolute_error": 0.0,
            "within_2sigma": 0.0,
            "nll": 0.0,
            "count": 0.0,
        }
        for name in ("node", "edge", "probe_cont")
    }
    bce_sum = 0.0
    bce_count = 0.0
    with torch.no_grad():
        for batch in loader:
            device_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = model(device_batch["node_x"], device_batch["edge_x"], device_batch["probe_x"])
            probe_cont_target = device_batch["target_probe_y"][
                ..., : len(PROBE_CONTINUOUS_FEATURES)
            ]
            probe_bin_target = device_batch["target_probe_y"][..., len(PROBE_CONTINUOUS_FEATURES) :]
            for name, mean, log_std, target in (
                (
                    "node",
                    outputs["node_mean"],
                    outputs["node_log_std"],
                    device_batch["target_node_y"],
                ),
                (
                    "edge",
                    outputs["edge_mean"],
                    outputs["edge_log_std"],
                    device_batch["target_edge_y"],
                ),
                ("probe_cont", outputs["probe_mean"], outputs["probe_log_std"], probe_cont_target),
            ):
                accumulator = families[name]
                error = target - mean
                accumulator["squared_error"] += float(error.pow(2).sum())
                accumulator["absolute_error"] += float(error.abs().sum())
                accumulator["within_2sigma"] += float(
                    (error.abs() <= 2.0 * torch.exp(log_std)).sum()
                )
                accumulator["nll"] += float(
                    (0.5 * (error.pow(2) * torch.exp(-2.0 * log_std) + 2.0 * log_std)).sum()
                )
                accumulator["count"] += float(target.numel())
            bce_sum += float(
                F.binary_cross_entropy_with_logits(
                    outputs["probe_binary_logits"], probe_bin_target, reduction="sum"
                )
            )
            bce_count += float(probe_bin_target.numel())
    metrics: dict[str, float] = {}
    for name, accumulator in families.items():
        count = max(accumulator["count"], 1.0)
        metrics[f"{name}_rmse"] = math.sqrt(accumulator["squared_error"] / count)
        metrics[f"{name}_mae"] = accumulator["absolute_error"] / count
        metrics[f"{name}_coverage_2sigma"] = accumulator["within_2sigma"] / count
        metrics[f"{name}_nll"] = accumulator["nll"] / count
    metrics["probe_bin_bce"] = bce_sum / max(bce_count, 1.0)
    summary = {
        **artifact_metadata("normal_evaluation"),
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir.resolve()),
        "split": "test",
        "test_metrics": metrics,
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary
