"""Streamlit explorer for RIDGE Stage 3 normal-emulator training outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from apps.common import (
    cached_csv,
    cached_json,
    format_number,
    is_artifact_dataset_root,
    render_artifact_inspector_entries,
    render_metric_notes,
    require_artifact_payload,
    split_run_counts,
    topology_counts,
)
from ridge.common.io import read_json_object

REQUIRED_STAGE3_FILES = (
    "best_normal_emulator.pt",
    "normal_metrics.json",
    "normal_training_history.csv",
    "normal_feature_schema.json",
)

REQUIRED_STAGE2_FILES = (
    "normal_window_index.csv",
    "normal_index.json",
    "normal_splits.json",
    "topology.json",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse the model root, optional Stage-2 root, and optional evaluation path, ignoring the options Streamlit adds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root", type=Path, default=None, help="Path to Stage 3 model output directory"
    )
    parser.add_argument(
        "--stage2-root", type=Path, default=None, help="Optional paired Stage 2 dataset root"
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Optional JSON produced by `evaluate-normal`.",
    )
    return parser.parse_known_args()[0]


def is_stage3_model_root(path: Path) -> bool:
    """Return whether a path holds every required Stage-3 output file."""
    return (
        path.exists()
        and path.is_dir()
        and all((path / filename).exists() for filename in REQUIRED_STAGE3_FILES)
    )


def is_stage2_dataset_root(path: Path) -> bool:
    """Return whether a path holds a complete Stage-2 dataset with a valid index artifact."""
    return is_artifact_dataset_root(
        path,
        runs_dirname="normal_runs",
        required_files=REQUIRED_STAGE2_FILES,
        index_filename="normal_index.json",
        artifact_type="normal_dataset",
    )


@st.cache_resource(show_spinner=False)
def cached_checkpoint(model_root: str) -> dict[str, Any]:
    """Load the best emulator checkpoint and validate its artifact envelope, cached per session."""
    checkpoint_path = Path(model_root) / "best_normal_emulator.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return require_artifact_payload(payload, "normal_checkpoint", source=checkpoint_path)


def _fmt_loss(value: Any) -> str:
    """Format a loss with four decimals, or n/a when it is not numeric."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def build_overview_metrics(
    metrics: dict[str, Any],
    history: pd.DataFrame,
    checkpoint: dict[str, Any],
    stage2_root: Path | None,
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """Build the overview cards and their notes from the metrics, history, and checkpoint."""
    model_config = checkpoint.get("model_config", {})
    topology = model_config.get("topology", {}) if isinstance(model_config, dict) else {}
    topo_counts = topology_counts(topology if isinstance(topology, dict) else {})
    epochs_logged = int(len(history))
    val_loss = (
        pd.to_numeric(history["val_loss"], errors="coerce").dropna()
        if "val_loss" in history.columns
        else pd.Series(dtype=float)
    )
    best_epoch = "n/a"
    if not val_loss.empty:
        best_row = history.loc[val_loss.idxmin()]
        best_epoch = str(int(best_row.get("epoch", val_loss.idxmin())))
    train_loss = (
        pd.to_numeric(history["loss"], errors="coerce").dropna()
        if "loss" in history.columns
        else pd.Series(dtype=float)
    )

    cards = [
        ("Best Val Loss", _fmt_loss(metrics.get("best_val_loss"))),
        ("Best Epoch", best_epoch),
        ("Epochs Logged", str(epochs_logged)),
        ("Final Train Loss", _fmt_loss(train_loss.iloc[-1] if not train_loss.empty else None)),
        ("Hidden Dim", str(model_config.get("hidden_dim", "n/a"))),
        ("Dropout", format_number(model_config.get("dropout"))),
        ("Nodes", str(topo_counts["node_count"])),
        ("Edges", str(topo_counts["edge_count"])),
        ("Probes", str(topo_counts["probe_count"])),
    ]
    notes = [
        {
            "Metric": "Best Val Loss",
            "Meaning": "Lowest validation loss reached during training. This checkpoint is restored when training ends.",
        },
        {"Metric": "Best Epoch", "Meaning": "Epoch with the lowest logged validation loss."},
        {
            "Metric": "Epochs Logged",
            "Meaning": "Number of epochs recorded in `normal_training_history.csv`.",
        },
        {
            "Metric": "Final Train Loss",
            "Meaning": "Training loss in the final logged epoch; this need not be the restored checkpoint epoch.",
        },
        {
            "Metric": "Hidden Dim",
            "Meaning": "Main hidden width used across the temporal encoder and prediction heads.",
        },
        {
            "Metric": "Dropout",
            "Meaning": "Dropout rate used for regularization inside the encoder.",
        },
        {"Metric": "Nodes", "Meaning": "Graph nodes per timestep that the normal emulator models."},
        {"Metric": "Edges", "Meaning": "Graph edges per timestep that the normal emulator models."},
        {
            "Metric": "Probes",
            "Meaning": "End-to-end probe paths included in the model input and predictions.",
        },
    ]

    if stage2_root is not None and is_stage2_dataset_root(stage2_root):
        splits = cached_json(str(stage2_root), "normal_splits.json")
        window_index = cached_csv(str(stage2_root), "normal_window_index.csv")
        counts = split_run_counts(splits)
        cards.extend(
            [
                ("Train Runs", str(counts["train"])),
                ("Val Runs", str(counts["val"])),
                ("Test Runs", str(counts["test"])),
                ("Total Windows", str(len(window_index))),
            ]
        )
        notes.extend(
            [
                {
                    "Metric": "Train Runs",
                    "Meaning": "Stage 2 runs used to fit normalization and optimize the normal emulator.",
                },
                {
                    "Metric": "Val Runs",
                    "Meaning": "Stage 2 runs reserved for checkpoint selection and early stopping.",
                },
                {
                    "Metric": "Test Runs",
                    "Meaning": "Stage 2 runs reserved for the separate `evaluate-normal` command; the trainer never evaluates them.",
                },
                {
                    "Metric": "Total Windows",
                    "Meaning": "Number of Stage 2 supervised windows available across all splits.",
                },
            ]
        )
    return cards, notes


def render_training_overview(
    metrics: dict[str, Any],
    history: pd.DataFrame,
    checkpoint: dict[str, Any],
    stage2_root: Path | None,
) -> None:
    """Show the training overview cards with a caption that they hold validation diagnostics only."""
    st.subheader("Training Overview")
    st.caption(
        "Stage 3 trains the normal emulator to predict the next graph snapshot under healthy behavior. "
        "These headline numbers summarize training duration, final quality, and the model size/context."
    )
    cards, notes = build_overview_metrics(metrics, history, checkpoint, stage2_root)
    columns = st.columns(4)
    for index, (label, value) in enumerate(cards):
        columns[index % len(columns)].metric(label, value)
    render_metric_notes(notes)
    st.caption(
        "Training artifacts contain validation diagnostics only. Test quality is loaded from a separate `evaluate-normal` result when provided."
    )


def render_loss_curves(history: pd.DataFrame) -> None:
    """Show the total and per-family training and validation loss curves by epoch."""
    st.subheader("Loss Curves")
    st.caption(
        "The total loss is the sum of node, edge, probe-continuous, and probe-binary terms. "
        "Node, edge, and probe-continuous components use Gaussian NLL, while the probe-binary term uses BCE-with-logits."
    )
    curve = history.copy()
    curve["epoch"] = pd.to_numeric(curve["epoch"], errors="coerce")

    total_curve = curve.melt(
        id_vars=["epoch"],
        value_vars=["loss", "val_loss"],
        var_name="split",
        value_name="loss_value",
    )
    total_curve["split"] = total_curve["split"].map({"loss": "train", "val_loss": "validation"})
    st.plotly_chart(
        px.line(
            total_curve,
            x="epoch",
            y="loss_value",
            color="split",
            markers=True,
            title="Train vs Validation Total Loss",
            labels={"epoch": "Epoch", "loss_value": "Loss"},
        ),
        width="stretch",
    )
    st.caption(
        "For the Gaussian NLL terms, more negative values are generally better. Probe-binary loss stays positive because it is a BCE term."
    )

    component_curve = curve.melt(
        id_vars=["epoch"],
        value_vars=[
            "node_loss",
            "val_node_loss",
            "edge_loss",
            "val_edge_loss",
            "probe_cont_loss",
            "val_probe_cont_loss",
            "probe_bin_loss",
            "val_probe_bin_loss",
        ],
        var_name="metric",
        value_name="value",
    )
    component_curve["split"] = component_curve["metric"].map(
        lambda value: "validation" if str(value).startswith("val_") else "train"
    )
    component_curve["component"] = component_curve["metric"].map(
        lambda value: str(value).removeprefix("val_").removesuffix("_loss")
    )
    component_curve["component"] = component_curve["component"].replace(
        {
            "node": "node",
            "edge": "edge",
            "probe_cont": "probe continuous",
            "probe_bin": "probe binary",
        }
    )
    st.plotly_chart(
        px.line(
            component_curve,
            x="epoch",
            y="value",
            color="split",
            facet_col="component",
            facet_col_wrap=2,
            markers=True,
            title="Component Losses by Epoch",
            labels={"epoch": "Epoch", "value": "Loss", "component": "Component"},
        ),
        width="stretch",
    )


def render_validation_metrics(metrics: dict[str, Any], history: pd.DataFrame) -> None:
    """Show the validation loss used for checkpoint selection and its best epoch."""
    st.subheader("Checkpoint-Selection Diagnostics")
    st.caption(
        "These values come from the validation split used for checkpoint selection. "
        "They are not held-out test results."
    )
    if history.empty or "val_loss" not in history.columns:
        st.info("No validation history was found.")
        return

    val_loss = pd.to_numeric(history["val_loss"], errors="coerce")
    if val_loss.dropna().empty:
        st.info("The validation-loss column contains no numeric values.")
        return
    best = history.loc[val_loss.idxmin()]
    metric_columns = [
        ("total loss", "val_loss"),
        ("node loss", "val_node_loss"),
        ("edge loss", "val_edge_loss"),
        ("probe continuous loss", "val_probe_cont_loss"),
        ("probe binary loss", "val_probe_bin_loss"),
    ]
    rows = [
        {"metric": label, "value": pd.to_numeric(best.get(column), errors="coerce")}
        for label, column in metric_columns
        if column in history.columns
    ]
    columns = st.columns(max(1, len(rows)))
    for column, row in zip(columns, rows):
        column.metric(str(row["metric"]).title(), _fmt_loss(row["value"]))
    if rows:
        st.plotly_chart(
            px.bar(pd.DataFrame(rows), x="metric", y="value", title="Best-Epoch Validation Losses"),
            width="stretch",
        )
    st.caption(
        f"Best logged epoch: {int(best.get('epoch', val_loss.idxmin()))}. "
        f"Persisted best validation loss: {_fmt_loss(metrics.get('best_val_loss'))}."
    )


def render_test_evaluation(evaluation: dict[str, Any] | None) -> None:
    """Show the held-out forecast metrics from an evaluate-normal result when one is given."""
    st.subheader("Held-Out Forecast Evaluation")
    if evaluation is None:
        st.info(
            "No `evaluate-normal` JSON was provided. Pass `--evaluation <path>` to inspect "
            "held-out RMSE, MAE, NLL, coverage, and probe-binary BCE."
        )
        return
    test_metrics = evaluation.get("test_metrics", {})
    st.caption("This section is loaded from a separate test-split evaluation artifact.")
    st.json(test_metrics, expanded=False)


def render_model_context(checkpoint: dict[str, Any], feature_schema: dict[str, Any]) -> None:
    """Show the feature group sizes and the topology counts stored in the checkpoint."""
    st.subheader("Model / Schema Context")
    st.caption(
        "This section summarizes what the Stage 3 model was trained to see and predict: feature group sizes plus the fixed graph topology carried in the checkpoint."
    )
    schema_rows = [
        {
            "feature_group": "node input",
            "feature_count": len(feature_schema.get("node_input_features", [])),
        },
        {
            "feature_group": "node target",
            "feature_count": len(feature_schema.get("node_target_features", [])),
        },
        {
            "feature_group": "edge input",
            "feature_count": len(feature_schema.get("edge_input_features", [])),
        },
        {
            "feature_group": "edge target",
            "feature_count": len(feature_schema.get("edge_target_features", [])),
        },
        {
            "feature_group": "probe input",
            "feature_count": len(feature_schema.get("probe_input_features", [])),
        },
        {
            "feature_group": "probe target",
            "feature_count": len(feature_schema.get("probe_target_features", [])),
        },
    ]
    schema_frame = pd.DataFrame(schema_rows)
    schema_cols = st.columns(6)
    for column, row in zip(schema_cols, schema_rows):
        column.metric(str(row["feature_group"]).title(), int(row["feature_count"]))
    st.dataframe(schema_frame, width="stretch", hide_index=True)

    model_config = checkpoint.get("model_config", {})
    topology = model_config.get("topology", {}) if isinstance(model_config, dict) else {}
    counts = topology_counts(topology if isinstance(topology, dict) else {})
    topo_cols = st.columns(4)
    topo_cols[0].metric("Node Count", counts["node_count"])
    topo_cols[1].metric("Edge Count", counts["edge_count"])
    topo_cols[2].metric("Probe Count", counts["probe_count"])
    topo_cols[3].metric("Candidate Count", counts["candidate_count"])
    st.caption(
        "Candidate count is included when the saved topology carries forward the downstream localization label space."
    )


def render_artifact_inspector(
    model_root: Path,
    metrics: dict[str, Any],
    feature_schema: dict[str, Any],
    checkpoint: dict[str, Any],
    stage2_root: Path | None,
) -> None:
    """Show the Stage-3 artifacts and the checkpoint configuration in collapsed expanders."""
    st.subheader("Artifact Inspector")
    st.caption(
        "These raw artifacts back the summaries above and are useful for quick verification or debugging."
    )

    artifacts: list[tuple[str, dict[str, Any]]] = [
        ("normal_metrics.json", metrics),
        ("normal_feature_schema.json", feature_schema),
    ]
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, dict):
        artifacts.append(("checkpoint.model_config", model_config))
        topology = model_config.get("topology")
        if isinstance(topology, dict):
            artifacts.append(("checkpoint.topology", topology))

    if stage2_root is not None and is_stage2_dataset_root(stage2_root):
        artifacts.append(
            ("stage2.normal_splits.json", cached_json(str(stage2_root), "normal_splits.json"))
        )
        artifacts.append(("stage2.topology.json", cached_json(str(stage2_root), "topology.json")))

    render_artifact_inspector_entries(artifacts)

    st.caption(f"Model root: {model_root}")
    if stage2_root is not None and is_stage2_dataset_root(stage2_root):
        st.caption(f"Paired Stage 2 dataset: {stage2_root}")


def main() -> None:
    """Validate the model root, load the Stage-3 artifacts, and show every section."""
    st.set_page_config(page_title="RIDGE Stage 3 Training Explorer", layout="wide")
    args = parse_args()
    model_root = args.model_root.expanduser().resolve() if args.model_root else None
    if model_root is None or not is_stage3_model_root(model_root):
        st.error(f"Model root is not a valid Stage 3 output directory: {model_root}")
        st.stop()

    stage2_root = args.stage2_root.expanduser().resolve() if args.stage2_root else None
    if stage2_root is not None and not is_stage2_dataset_root(stage2_root):
        st.sidebar.warning(f"Ignoring invalid Stage 2 dataset root: {stage2_root}")
        stage2_root = None

    st.title("RIDGE Stage 3 Normal Emulator Training Explorer")
    st.caption(f"Model root: {model_root}")

    try:
        metrics = cached_json(str(model_root), "normal_metrics.json")
        history = cached_csv(str(model_root), "normal_training_history.csv")
        feature_schema = cached_json(str(model_root), "normal_feature_schema.json")
        checkpoint = cached_checkpoint(str(model_root))
        evaluation = (
            read_json_object(Path(args.evaluation.parent) / args.evaluation.name)
            if args.evaluation
            else None
        )
    except Exception as exc:
        st.error(f"Failed to load Stage 3 artifacts: {exc}")
        st.stop()

    render_training_overview(metrics, history, checkpoint, stage2_root)
    render_loss_curves(history)
    render_validation_metrics(metrics, history)
    render_test_evaluation(evaluation)
    render_model_context(checkpoint, feature_schema)
    render_artifact_inspector(model_root, metrics, feature_schema, checkpoint, stage2_root)


if __name__ == "__main__":
    main()
