"""Command-line interface for the RIDGE research pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ridge.common.contracts import (
    MAX_GENERATION_WORKERS,
    RIDGE_EARLY_STOPPING_METRICS,
)
from ridge.common.io import write_json


def default_build_workers() -> int:
    """Return the default number of dataset build workers, at most eight."""
    return max(1, min(8, os.cpu_count() or 1))


def default_train_loader_workers() -> int:
    """Return the default number of data loader workers, half the CPUs and at most eight."""
    return max(0, min(8, (os.cpu_count() or 1) // 2))


def generation_worker_count(value: str) -> int:
    """Parse the generation worker count and reject values outside the certified range."""
    workers = int(value)
    if not 1 <= workers <= MAX_GENERATION_WORKERS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_GENERATION_WORKERS}, inclusive"
        )
    return workers


def _add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the Stage-1 generation arguments to a subcommand parser."""
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=generation_worker_count,
        required=True,
        help="Explicit concurrent Mininet process count (certified maximum: 16).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--burst-mean-gap-sec", type=float, default=None)
    parser.add_argument("--traffic-flow-min", type=int, default=4)
    parser.add_argument("--traffic-flow-max", type=int, default=6)
    parser.add_argument("--ping-pair-min", type=int, default=4)
    parser.add_argument("--ping-pair-max", type=int, default=6)
    parser.add_argument("--probe-packets", type=int, default=1)
    parser.add_argument("--probe-timeout-sec", type=float, default=1.0)
    parser.add_argument("--probe-cadence-sec", type=float, default=1.0)
    parser.add_argument(
        "--warmup-sec",
        type=int,
        default=15,
        help=(
            "Nominal warmup period. Deterministic episode plans sample from "
            "[max(0, value - 10), value + 20] seconds."
        ),
    )
    parser.add_argument("--fault-start-offset-min-sec", type=int, default=0)
    parser.add_argument("--fault-start-offset-max-sec", type=int, default=0)
    parser.add_argument("--fault-duration-min-sec", type=int, default=60)
    parser.add_argument("--fault-duration-max-sec", type=int, default=60)
    parser.add_argument("--min-training-runs", type=int, default=500)
    parser.add_argument("--fault-fraction", type=float, default=0.5)
    parser.add_argument(
        "--fault-category",
        default="mixed",
        choices=("mixed", "drain", "fiber_cut", "link_degradation", "link_flap"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drain-ramp-steps", type=int, default=5)
    parser.add_argument("--drain-phase-ratio-ramp-down", type=float, default=0.25)
    parser.add_argument("--drain-phase-ratio-link-down", type=float, default=0.15)
    parser.add_argument("--drain-phase-ratio-hold-down", type=float, default=0.20)
    parser.add_argument("--drain-phase-ratio-ramp-up", type=float, default=0.40)
    parser.add_argument(
        "--max-episode-retries",
        type=int,
        default=3,
        help=(
            "Rerun an episode whose timing or health check fails, up to this many "
            "times, before it counts as a hard failure. The retried episode reuses "
            "the identical deterministic specification. 0 disables retries."
        ),
    )
    parser.add_argument(
        "--max-total-retries",
        type=int,
        default=-1,
        help=(
            "Global cap on retries across the whole generation, which halts generation "
            "when exhausted. Negative auto-sizes to max(10, ceil(0.02 * --runs))."
        ),
    )


def _add_common_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the training arguments shared by the emulator and RCA model subcommands."""
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=default_train_loader_workers())
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--shard-cache-size", type=int, default=2)


def _add_common_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the evaluation arguments shared by the checkpoint evaluation subcommands."""
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the unified parser without importing execution dependencies."""
    parser = argparse.ArgumentParser(
        prog="ridge",
        description="RIDGE dataset, training, and evaluation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a Stage-1 dataset")
    _add_generate_arguments(generate)

    build_normal = subparsers.add_parser(
        "build-normal",
        help="Build the Stage-2 normal-emulator dataset",
    )
    build_normal.add_argument("--dataset-dir", type=Path, required=True)
    build_normal.add_argument("--output-dir", type=Path, required=True)
    build_normal.add_argument("--history-len", type=int, default=6)
    build_normal.add_argument("--prediction-horizon", type=int, default=1)
    build_normal.add_argument("--seed", type=int, default=42)
    build_normal.add_argument("--workers", type=int, default=default_build_workers())
    build_normal.add_argument(
        "--require-baseline-health-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    train_normal = subparsers.add_parser(
        "train-normal",
        help="Train the Stage-3 normal emulator",
    )
    _add_common_training_arguments(train_normal)
    train_normal.add_argument("--torch-threads", type=int, default=None)
    train_normal.add_argument("--torch-interop-threads", type=int, default=None)

    build_residual = subparsers.add_parser(
        "build-residual",
        help="Build the Stage-4 residual dataset",
    )
    build_residual.add_argument("--dataset-dir", type=Path, required=True)
    build_residual.add_argument("--normal-data-dir", type=Path, required=True)
    build_residual.add_argument(
        "--normal-emulator",
        type=Path,
        default=None,
        help="Required when --residual-mode=standardized.",
    )
    build_residual.add_argument("--output-dir", type=Path, required=True)
    build_residual.add_argument("--residual-history-len", type=int, default=6)
    build_residual.add_argument(
        "--residual-mode",
        default="standardized",
        choices=("standardized", "raw"),
    )
    build_residual.add_argument("--inference-batch-size", type=int, default=256)
    build_residual.add_argument("--device", default="cpu")

    train_ridge = subparsers.add_parser(
        "train-ridge",
        help="Train the Stage-5 RIDGE model",
    )
    _add_common_training_arguments(train_ridge)
    train_ridge.add_argument("--lambda-recon", type=float, default=0.5)
    train_ridge.add_argument(
        "--early-stopping-metric",
        default="fault_present_f1",
        choices=RIDGE_EARLY_STOPPING_METRICS,
    )
    train_ridge.add_argument(
        "--no-graph",
        action="store_true",
        help="Train the no-graph model, without message passing or endpoint conditioning.",
    )

    evaluate_normal = subparsers.add_parser(
        "evaluate-normal",
        help="Evaluate Stage-3 forecasting quality on the test split",
    )
    _add_common_evaluation_arguments(evaluate_normal)

    evaluate_ridge = subparsers.add_parser(
        "evaluate-ridge",
        help="Evaluate a Stage-5 checkpoint on the test split",
    )
    _add_common_evaluation_arguments(evaluate_ridge)
    evaluate_ridge.add_argument("--top-k", type=int, default=3)

    evaluate_threshold = subparsers.add_parser(
        "evaluate-threshold",
        help="Evaluate the threshold-ranking baseline on the test split",
    )
    evaluate_threshold.add_argument("--data-dir", type=Path, required=True)
    evaluate_threshold.add_argument("--output", type=Path)

    return parser


def _emit_summary(summary: dict[str, Any], output: Path | None = None) -> int:
    """Print a summary as JSON, write it to the output path when given, and return exit code zero."""
    if output is not None:
        write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse and execute one pipeline command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        from ridge.pipeline.generate import run_from_args

        return int(run_from_args(args))

    if args.command == "build-normal":
        from ridge.models.normal_data import build_normal_emulator_dataset

        summary = build_normal_emulator_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            history_len=args.history_len,
            prediction_horizon=args.prediction_horizon,
            seed=args.seed,
            workers=args.workers,
            require_baseline_health_pass=args.require_baseline_health_pass,
        )
        return _emit_summary(summary)

    if args.command == "train-normal":
        from ridge.models.normal_model import NormalTrainConfig, train_normal_emulator

        summary = train_normal_emulator(
            NormalTrainConfig(
                data_dir=str(args.data_dir),
                output_dir=str(args.output_dir),
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                device=args.device,
                workers=args.workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                shard_cache_size=args.shard_cache_size,
                torch_threads=args.torch_threads,
                torch_interop_threads=args.torch_interop_threads,
            )
        )
        return _emit_summary(summary)

    if args.command == "build-residual":
        from ridge.models.residual_data import build_residual_dataset

        summary = build_residual_dataset(
            dataset_dir=args.dataset_dir,
            normal_data_dir=args.normal_data_dir,
            normal_emulator_path=args.normal_emulator,
            output_dir=args.output_dir,
            residual_history_len=args.residual_history_len,
            residual_mode=args.residual_mode,
            inference_batch_size=args.inference_batch_size,
            device=args.device,
        )
        return _emit_summary(summary)

    if args.command == "train-ridge":
        from ridge.models.ridge_model import RidgeTrainConfig, train_ridge

        summary = train_ridge(
            RidgeTrainConfig(
                data_dir=str(args.data_dir),
                output_dir=str(args.output_dir),
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                device=args.device,
                lambda_recon=args.lambda_recon,
                early_stopping_metric=args.early_stopping_metric,
                use_graph_structure=not args.no_graph,
                workers=args.workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                shard_cache_size=args.shard_cache_size,
            )
        )
        return _emit_summary(summary)

    if args.command == "evaluate-normal":
        from ridge.models.normal_model import evaluate_normal_checkpoint

        summary = evaluate_normal_checkpoint(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            output_path=args.output,
        )
        return _emit_summary(summary)

    if args.command == "evaluate-ridge":
        from ridge.models.ridge_model import evaluate_ridge_checkpoint

        summary = evaluate_ridge_checkpoint(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            top_k=args.top_k,
            prediction_output_path=(
                args.output.parent / "per_sample_ridge_predictions.csv"
                if args.output is not None
                else None
            ),
        )
        return _emit_summary(summary, args.output)

    if args.command == "evaluate-threshold":
        from ridge.models.baselines import evaluate_threshold_baseline

        summary = evaluate_threshold_baseline(args.data_dir, output_path=args.output)
        return _emit_summary(summary)

    raise AssertionError(f"unhandled command: {args.command}")
