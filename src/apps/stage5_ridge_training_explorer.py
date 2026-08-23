"""Streamlit explorer for RIDGE Stage 5 training outputs."""

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
    cached_residual_window_index,
    format_integer,
    format_number,
    format_percent,
    is_artifact_dataset_root,
    render_artifact_inspector_entries,
    render_metric_notes,
    require_artifact_payload,
    split_run_counts,
    split_window_counts,
    topology_counts,
)

REQUIRED_STAGE5_FILES = (
    "best.pt",
    "metrics.json",
    "training_history.csv",
)

REQUIRED_STAGE4_FILES = (
    "residual_window_index.csv",
    "residual_index.json",
    "residual_splits.json",
    "candidate_index.json",
    "topology.json",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse the model root and optional Stage-4 root, ignoring the options Streamlit adds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root", type=Path, default=None, help="Path to Stage 5 model output directory"
    )
    parser.add_argument(
        "--stage4-root", type=Path, default=None, help="Optional paired Stage 4 dataset root"
    )
    return parser.parse_known_args()[0]


def is_stage5_model_root(path: Path) -> bool:
    """Return whether a path holds every required Stage-5 output file."""
    return (
        path.exists()
        and path.is_dir()
        and all((path / filename).exists() for filename in REQUIRED_STAGE5_FILES)
    )


def is_stage4_dataset_root(path: Path) -> bool:
    """Return whether a path holds a complete Stage-4 dataset with a valid index artifact."""
    return is_artifact_dataset_root(
        path,
        runs_dirname="residual_runs",
        required_files=REQUIRED_STAGE4_FILES,
        index_filename="residual_index.json",
        artifact_type="residual_dataset",
    )


@st.cache_resource(show_spinner=False)
def cached_checkpoint(model_root: str) -> dict[str, Any]:
    """Load the best RCA model checkpoint and validate its artifact envelope, cached per session."""
    checkpoint_path = Path(model_root) / "best.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return require_artifact_payload(payload, "ridge_checkpoint", source=checkpoint_path)


def ridge_topology_counts(topology: dict[str, Any]) -> dict[str, int]:
    """Return the topology counts with the number of fault categories."""
    return {
        **topology_counts(topology),
        "category_count": len(topology.get("fault_category_labels", [])),
    }


def build_overview_metrics(
    metrics: dict[str, Any],
    history: pd.DataFrame,
    checkpoint: dict[str, Any],
    stage4_root: Path | None,
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """Build the overview cards and their notes from the metrics, history, checkpoint, and optional Stage-4 context."""
    model_config = checkpoint.get("model_config", {})
    topology: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    residual_index: dict[str, Any] = {}
    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        topology = cached_json(str(stage4_root), "topology.json")
        candidates = cached_json(str(stage4_root), "candidate_index.json")
        residual_index_path = stage4_root / "residual_index.json"
        if residual_index_path.is_file():
            residual_index = cached_json(str(stage4_root), "residual_index.json")
    topology = {
        **topology,
        "candidate_ids": candidates.get("candidate_ids", topology.get("candidate_ids", [])),
        "fault_category_labels": checkpoint.get("fault_category_labels", []),
    }
    counts = ridge_topology_counts(topology)
    cards = [
        (
            "Best Val Metric",
            f"{metrics.get('best_val_metric_name', 'n/a')}: {format_number(metrics.get('best_val_metric_value'))}",
        ),
        ("Best Val Top-3", format_percent(metrics.get("best_val_top3_candidate_accuracy"))),
        ("Epochs Logged", format_integer(len(history))),
        ("Hidden Dim", format_integer(model_config.get("hidden_dim"))),
        ("Dropout", format_number(model_config.get("dropout"))),
        (
            "Graph Structure",
            "enabled" if model_config.get("use_graph_structure", True) else "disabled",
        ),
        ("Input Mode", str(residual_index.get("residual_mode", "unknown"))),
        ("Nodes", format_integer(counts["node_count"])),
        ("Edges", format_integer(counts["edge_count"])),
        ("Probes", format_integer(counts["probe_count"])),
        ("Candidates", format_integer(counts["candidate_count"])),
        ("Categories", format_integer(counts["category_count"])),
    ]
    notes = [
        {
            "Metric": "Best Val Metric",
            "Meaning": "Metric used for checkpoint selection during early stopping.",
        },
        {
            "Metric": "Best Val Top-3",
            "Meaning": "Validation candidate top-3 accuracy used as the tie-breaker when the main validation metric is tied.",
        },
        {
            "Metric": "Graph Structure",
            "Meaning": "Whether GraphSAGE and endpoint conditioning were enabled for this checkpoint.",
        },
        {
            "Metric": "Input Mode",
            "Meaning": "Whether the paired Stage 4 dataset contains standardized residuals or normalized raw observations.",
        },
        {
            "Metric": "Candidates",
            "Meaning": "Root-cause label space including the healthy `none` class.",
        },
        {
            "Metric": "Categories",
            "Meaning": "Coarse fault-category label count stored in the checkpoint.",
        },
    ]

    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        splits = cached_json(str(stage4_root), "residual_splits.json")
        window_index = cached_residual_window_index(str(stage4_root))
        run_counts = split_run_counts(splits)
        cards.extend(
            [
                ("Train Runs", format_integer(run_counts.get("train", 0))),
                ("Val Runs", format_integer(run_counts.get("val", 0))),
                ("Test Runs", format_integer(run_counts.get("test", 0))),
                ("Residual Windows", format_integer(len(window_index))),
            ]
        )
        notes.extend(
            [
                {
                    "Metric": "Train / Val / Test Runs",
                    "Meaning": "Stage 4 runs assigned to each split and reused for Stage 5 training, model selection, and held-out evaluation.",
                },
                {
                    "Metric": "Residual Windows",
                    "Meaning": "Number of supervised Stage 5 windows available from the paired Stage 4 dataset.",
                },
            ]
        )
    return cards, notes


def render_training_overview(
    metrics: dict[str, Any],
    history: pd.DataFrame,
    checkpoint: dict[str, Any],
    stage4_root: Path | None,
) -> None:
    """Show the training overview cards with a caption that held-out results belong to Stage 6."""
    st.subheader("Training Overview")
    st.caption(
        "Stage 5 trains the RCA model. These values describe validation-based checkpoint "
        "selection and model context. Held-out results belong to Stage 6."
    )
    cards, notes = build_overview_metrics(metrics, history, checkpoint, stage4_root)
    columns = st.columns(4)
    for index, (label, value) in enumerate(cards):
        columns[index % len(columns)].metric(label, value)
    render_metric_notes(notes)


def render_training_curves(history: pd.DataFrame) -> None:
    """Show the training and validation loss curves and the validation RCA metrics by epoch."""
    st.subheader("Training Curves")
    st.caption(
        "The saved history tracks total train and validation loss plus the validation metrics used to monitor RCA quality while training."
    )
    curve = history.copy()
    curve["epoch"] = pd.to_numeric(curve["epoch"], errors="coerce")

    loss_curve = curve.melt(
        id_vars=["epoch"],
        value_vars=["train_loss", "val_loss"],
        var_name="split",
        value_name="loss_value",
    )
    loss_curve["split"] = loss_curve["split"].map({"train_loss": "train", "val_loss": "validation"})
    st.plotly_chart(
        px.line(
            loss_curve,
            x="epoch",
            y="loss_value",
            color="split",
            markers=True,
            title="Train vs Validation Loss",
            labels={"epoch": "Epoch", "loss_value": "Loss"},
        ),
        width="stretch",
    )

    metric_curve = curve.melt(
        id_vars=["epoch"],
        value_vars=["val_fault_present_f1", "val_top3_candidate_accuracy", "val_mrr"],
        var_name="metric",
        value_name="value",
    )
    metric_curve["metric"] = metric_curve["metric"].replace(
        {
            "val_fault_present_f1": "validation fault-present F1",
            "val_top3_candidate_accuracy": "validation top-3 candidate accuracy",
            "val_mrr": "validation MRR",
        }
    )
    st.plotly_chart(
        px.line(
            metric_curve,
            x="epoch",
            y="value",
            color="metric",
            markers=True,
            title="Validation RCA Metrics by Epoch",
            labels={"epoch": "Epoch", "value": "Metric Value"},
        ),
        width="stretch",
    )


def render_checkpoint_selection(metrics: dict[str, Any], history: pd.DataFrame) -> None:
    """Show the selection metric, its best value, the top-3 tie-break, and the epochs logged."""
    st.subheader("Checkpoint-Selection Summary")
    st.caption(
        "The trainer writes validation metrics only. Test metrics are produced separately "
        "by Stage 6 after the checkpoint is fixed."
    )
    highlights = [
        ("Selection Metric", str(metrics.get("best_val_metric_name", "n/a"))),
        ("Best Value", format_number(metrics.get("best_val_metric_value"))),
        ("Top-3 Tie-break", format_percent(metrics.get("best_val_top3_candidate_accuracy"))),
        ("Epochs Logged", format_integer(len(history))),
    ]
    columns = st.columns(len(highlights))
    for column, (label, value) in zip(columns, highlights):
        column.metric(label, value)

    available = [
        column
        for column in (
            "val_fault_present_f1",
            "val_top3_candidate_accuracy",
            "val_mrr",
        )
        if column in history.columns
    ]
    if available:
        final = history.tail(1)[["epoch", *available]].melt(
            id_vars=["epoch"], var_name="metric", value_name="final_value"
        )
        st.dataframe(final, width="stretch", hide_index=True)


def render_label_context(
    metrics: dict[str, Any], checkpoint: dict[str, Any], stage4_root: Path | None
) -> None:
    """Show the candidate and fault-category label spaces, the class weights, and the paired Stage-4 splits."""
    st.subheader("Label Space / Training Context")
    st.caption(
        "This section shows the candidate and category label spaces the model was trained on, plus class weighting and optional paired Stage 4 split context."
    )
    candidate_ids: list[Any] = []
    candidate_kinds: list[Any] = []
    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        candidate_index = cached_json(str(stage4_root), "candidate_index.json")
        topology = cached_json(str(stage4_root), "topology.json")
        candidate_ids = list(candidate_index.get("candidate_ids", []))
        candidate_kinds = list(topology.get("candidate_kinds", []))
    category_labels = checkpoint.get("fault_category_labels", [])
    category_weights = metrics.get("category_class_weights", [])

    candidate_frame = pd.DataFrame(
        {
            "candidate_index": list(range(len(candidate_ids))),
            "candidate_id": [str(candidate_id) for candidate_id in candidate_ids],
            "candidate_kind": [
                str(candidate_kinds[index]) if index < len(candidate_kinds) else "unknown"
                for index in range(len(candidate_ids))
            ],
        }
    )
    category_frame = pd.DataFrame(
        {
            "category_index": list(range(len(category_labels))),
            "category_label": [str(label) for label in category_labels],
            "class_weight": [float(weight) for weight in category_weights]
            if category_weights
            else [float("nan")] * len(category_labels),
        }
    )

    left, right = st.columns(2)
    with left:
        st.dataframe(candidate_frame, width="stretch", hide_index=True)
        if candidate_frame.empty:
            st.info("Pair a Stage 4 dataset to inspect the candidate catalog.")
    with right:
        st.dataframe(category_frame, width="stretch", hide_index=True)
        if not category_frame.empty:
            st.plotly_chart(
                px.bar(
                    category_frame,
                    x="category_label",
                    y="class_weight",
                    title="Category Class Weights",
                ),
                width="stretch",
            )

    if stage4_root is None or not is_stage4_dataset_root(stage4_root):
        st.caption(
            "Add a paired Stage 4 dataset to see split counts and label balance from the exact residual corpus used for training."
        )
        return

    splits = cached_json(str(stage4_root), "residual_splits.json")
    window_index = cached_residual_window_index(str(stage4_root))
    run_counts = split_run_counts(splits)
    split_windows = split_window_counts(window_index, splits)

    split_cols = st.columns(3)
    split_cols[0].metric("Train Runs", format_integer(run_counts.get("train", 0)))
    split_cols[1].metric("Val Runs", format_integer(run_counts.get("val", 0)))
    split_cols[2].metric("Test Runs", format_integer(run_counts.get("test", 0)))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(split_windows, x="split", y="window_count", title="Stage 4 Windows by Split"),
            width="stretch",
        )
    with right:
        label_rows = [
            {
                "label_family": "fault present",
                "label": "faulty",
                "count": int((window_index["fault_present"] > 0).sum()),
            },
            {
                "label_family": "fault present",
                "label": "none",
                "count": int((window_index["fault_present"] <= 0).sum()),
            },
        ]
        if "fault_category" in window_index.columns:
            for label, count in (
                window_index["fault_category"].fillna("unknown").value_counts().items()
            ):
                label_rows.append(
                    {"label_family": "fault category", "label": str(label), "count": int(count)}
                )
        label_frame = pd.DataFrame(label_rows)
        st.plotly_chart(
            px.bar(
                label_frame,
                x="label",
                y="count",
                color="label_family",
                title="Stage 4 Label Balance",
            ),
            width="stretch",
        )

    st.caption(
        "Candidates comprise `none`, node causes, and edge causes. All are ranked and weighted by the same candidate head."
    )


def render_artifact_inspector(
    model_root: Path, metrics: dict[str, Any], checkpoint: dict[str, Any], stage4_root: Path | None
) -> None:
    """Show the Stage-5 artifacts and the checkpoint configuration in collapsed expanders."""
    st.subheader("Artifact Inspector")
    st.caption(
        "These raw artifacts back the summaries above and are useful for quick verification."
    )

    artifacts: list[tuple[str, dict[str, Any]]] = [
        ("metrics.json", metrics),
    ]
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, dict):
        artifacts.append(("checkpoint.model_config", model_config))

    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        artifacts.append(
            ("stage4.residual_splits.json", cached_json(str(stage4_root), "residual_splits.json"))
        )
        artifacts.append(
            ("stage4.candidate_index.json", cached_json(str(stage4_root), "candidate_index.json"))
        )
        artifacts.append(("stage4.topology.json", cached_json(str(stage4_root), "topology.json")))

    render_artifact_inspector_entries(artifacts)

    st.caption(f"Model root: {model_root}")
    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        st.caption(f"Paired Stage 4 dataset: {stage4_root}")


def main() -> None:
    """Validate the model root, load the Stage-5 artifacts, and show every section."""
    st.set_page_config(page_title="RIDGE Stage 5 Training Explorer", layout="wide")
    args = parse_args()
    model_root = args.model_root.expanduser().resolve() if args.model_root else None
    if model_root is None or not is_stage5_model_root(model_root):
        st.error(f"Model root is not a valid Stage 5 output directory: {model_root}")
        st.stop()

    stage4_root = args.stage4_root.expanduser().resolve() if args.stage4_root else None
    if stage4_root is not None and not is_stage4_dataset_root(stage4_root):
        st.sidebar.warning(f"Ignoring invalid Stage 4 dataset root: {stage4_root}")
        stage4_root = None

    st.title("RIDGE Stage 5 Training Explorer")
    st.caption(f"Model root: {model_root}")

    try:
        metrics = cached_json(str(model_root), "metrics.json")
        history = cached_csv(str(model_root), "training_history.csv")
        checkpoint = cached_checkpoint(str(model_root))
    except Exception as exc:
        st.error(f"Failed to load Stage 5 artifacts: {exc}")
        st.stop()

    render_training_overview(metrics, history, checkpoint, stage4_root)
    render_training_curves(history)
    render_checkpoint_selection(metrics, history)
    render_label_context(metrics, checkpoint, stage4_root)
    render_artifact_inspector(model_root, metrics, checkpoint, stage4_root)


if __name__ == "__main__":
    main()
