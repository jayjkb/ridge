"""Streamlit explorer for RIDGE Stage 2 normal-emulator datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from apps.common import (
    cached_json,
    is_artifact_dataset_root,
    load_csv,
    render_artifact_inspector_entries,
    render_metric_notes,
    split_run_counts_frame,
    split_window_counts,
    topology_counts,
    window_count_summary,
)
from ridge.common.io import read_json_object

REQUIRED_STAGE2_FILES = (
    "normal_window_index.csv",
    "normal_index.json",
    "normal_splits.json",
    "normalization.json",
    "topology.json",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse the dataset root argument, ignoring the options Streamlit adds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=None, help="Path to Stage 2 dataset root"
    )
    return parser.parse_known_args()[0]


def is_stage2_dataset_root(path: Path) -> bool:
    """Return whether a path holds a complete Stage-2 dataset with a valid index artifact."""
    return is_artifact_dataset_root(
        path,
        runs_dirname="normal_runs",
        required_files=REQUIRED_STAGE2_FILES,
        index_filename="normal_index.json",
        artifact_type="normal_dataset",
    )


@st.cache_data(show_spinner=False)
def cached_window_index(dataset_root: str) -> pd.DataFrame:
    """Load the Stage-2 window index with string episode identifiers, cached per session."""
    frame = load_csv(dataset_root, "normal_window_index.csv")
    if "run_id" in frame.columns:
        frame["run_id"] = frame["run_id"].astype(str)
    return frame


@st.cache_data(show_spinner=False)
def cached_source_manifest(dataset_root: str) -> pd.DataFrame:
    """Load the source Stage-1 manifest with an episode label per fault category, or an empty frame, cached per session."""
    manifest_path = infer_source_manifest_path(Path(dataset_root))
    if manifest_path is None:
        return pd.DataFrame(columns=["run_id", "fault_category", "run_label"])
    frame = pd.read_csv(manifest_path)
    if "run_id" not in frame.columns:
        return pd.DataFrame(columns=["run_id", "fault_category", "run_label"])
    frame["run_id"] = frame["run_id"].astype(str)
    if "fault_category" not in frame.columns:
        frame["fault_category"] = "unknown"
    frame["fault_category"] = frame["fault_category"].fillna("unknown").astype(str)
    frame["run_label"] = frame["fault_category"].map(
        lambda value: "healthy" if value == "none" else value
    )
    return (
        frame[["run_id", "fault_category", "run_label"]]
        .drop_duplicates(subset=["run_id"])
        .reset_index(drop=True)
    )


def infer_source_manifest_path(dataset_root: Path) -> Path | None:
    """Locate the source Stage-1 manifest from the recorded dataset path, a naming convention, or the dataset root."""
    candidates: list[Path] = []
    index_path = dataset_root / "normal_index.json"
    if index_path.is_file():
        normal_index = read_json_object(Path(dataset_root) / "normal_index.json")
        source_dataset = normal_index.get("source_dataset")
        if source_dataset:
            source_root = Path(str(source_dataset)).expanduser()
            candidates.append(source_root / "manifest.csv")
            if not source_root.is_absolute():
                candidates.append(REPO_ROOT / source_root / "manifest.csv")
    if dataset_root.name.endswith("_normal"):
        candidates.append(
            dataset_root.with_name(dataset_root.name[: -len("_normal")]) / "manifest.csv"
        )
    candidates.append(dataset_root / "manifest.csv")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def windows_per_run(window_index: pd.DataFrame) -> pd.DataFrame:
    """Count the windows of each episode, most windows first."""
    if window_index.empty:
        return pd.DataFrame(columns=["run_id", "window_count"])
    counts = (
        window_index.groupby(window_index["run_id"].astype(str), as_index=False)
        .size()
        .rename(columns={"size": "window_count"})
        .sort_values(["window_count", "run_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return counts


def windows_per_run_with_labels(
    window_index: pd.DataFrame, source_manifest: pd.DataFrame | None
) -> pd.DataFrame:
    """Attach the episode label from the source manifest to the per-episode window counts."""
    counts = windows_per_run(window_index)
    if counts.empty:
        return counts
    if source_manifest is None or source_manifest.empty:
        counts["run_label"] = "unknown"
        return counts
    labels = source_manifest[["run_id", "run_label"]].copy()
    labels["run_id"] = labels["run_id"].astype(str)
    merged = counts.merge(labels, on="run_id", how="left")
    merged["run_label"] = merged["run_label"].fillna("unknown")
    return merged


def feature_schema_counts(normal_index: dict[str, Any]) -> pd.DataFrame:
    """Return the feature count of every input and target group in the saved schema."""
    schema = normal_index.get("feature_schema", {})
    if not isinstance(schema, dict):
        schema = {}
    rows = [
        {
            "feature_group": "node_input",
            "feature_count": len(schema.get("node_input_features", [])),
        },
        {
            "feature_group": "node_target",
            "feature_count": len(schema.get("node_target_features", [])),
        },
        {
            "feature_group": "edge_input",
            "feature_count": len(schema.get("edge_input_features", [])),
        },
        {
            "feature_group": "edge_target",
            "feature_count": len(schema.get("edge_target_features", [])),
        },
        {
            "feature_group": "probe_input",
            "feature_count": len(schema.get("probe_input_features", [])),
        },
        {
            "feature_group": "probe_target",
            "feature_count": len(schema.get("probe_target_features", [])),
        },
    ]
    return pd.DataFrame(rows)


def normalization_counts(normalization: dict[str, Any]) -> pd.DataFrame:
    """Return the number of normalized features per input and target group."""
    rows = []
    for name in (
        "node_input",
        "edge_input",
        "probe_input",
        "node_target",
        "edge_target",
        "probe_target",
    ):
        stats = normalization.get(name, {})
        feature_indices = stats.get("feature_indices", []) if isinstance(stats, dict) else []
        rows.append({"feature_group": name, "normalized_feature_count": len(feature_indices)})
    return pd.DataFrame(rows)


def render_metrics(
    splits: dict[str, list[str]],
    window_index: pd.DataFrame,
    normal_index: dict[str, Any],
    topology: dict[str, Any],
) -> None:
    """Show window, episode, and topology counts of the dataset with their explanations."""
    run_counts = split_run_counts_frame(splits).set_index("split")["run_count"].to_dict()
    topo_counts = topology_counts(topology)
    sample_count = int(normal_index.get("sample_count", len(window_index)))

    st.subheader("Dataset Overview")
    st.caption(
        "Stage 2 converts successful Stage 1 runs into supervised forecasting windows for the normal emulator. "
        "These headline numbers show how much usable data was produced and the graph size seen by the model."
    )

    cols = st.columns(8)
    cols[0].metric("Total Runs", sum(run_counts.values()))
    cols[1].metric("Total Windows", sample_count)
    cols[2].metric("Train Runs", int(run_counts.get("train", 0)))
    cols[3].metric("Val Runs", int(run_counts.get("val", 0)))
    cols[4].metric("Test Runs", int(run_counts.get("test", 0)))
    cols[5].metric("Nodes", topo_counts["node_count"])
    cols[6].metric("Edges", topo_counts["edge_count"])
    cols[7].metric("Probes", topo_counts["probe_count"])

    render_metric_notes(
        [
            {
                "Metric": "Total Runs",
                "Meaning": "Number of Stage 2 runs assigned to train, validation, or test splits.",
            },
            {
                "Metric": "Total Windows",
                "Meaning": "Total supervised samples written to the lazy window index and used for training or evaluation.",
            },
            {
                "Metric": "Train Runs",
                "Meaning": "Runs used to fit normalization statistics and train the normal emulator.",
            },
            {
                "Metric": "Val Runs",
                "Meaning": "Runs reserved for model selection and early stopping checks.",
            },
            {"Metric": "Test Runs", "Meaning": "Runs held out for final evaluation only."},
            {
                "Metric": "Nodes",
                "Meaning": "Graph nodes per timestep, such as hosts and infrastructure devices.",
            },
            {
                "Metric": "Edges",
                "Meaning": "Physical or logical links represented in each graph snapshot.",
            },
            {
                "Metric": "Probes",
                "Meaning": "End-to-end probe paths whose RTT and loss features are part of the model input.",
            },
        ]
    )


def render_charts(
    window_index: pd.DataFrame, splits: dict[str, list[str]], source_manifest: pd.DataFrame
) -> None:
    """Show episode and window counts per split and the per-episode window distribution by label."""
    run_counts = split_run_counts_frame(splits)
    window_counts = split_window_counts(window_index, splits)
    by_run = windows_per_run_with_labels(window_index, source_manifest)
    summary = window_count_summary(by_run)

    st.subheader("Split and Window Coverage")
    st.caption(
        "These charts show how the dataset is partitioned and whether some runs contribute many more windows than others. "
        "That helps us spot imbalance before model training."
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(run_counts, x="split", y="run_count", title="Run Counts by Split"),
            width="stretch",
        )
        st.caption(
            "Each bar counts how many full runs were assigned to a split. Splits happen at the run level, not the window level."
        )
    with right:
        st.plotly_chart(
            px.bar(window_counts, x="split", y="window_count", title="Window Counts by Split"),
            width="stretch",
        )
        st.caption(
            "Each bar counts supervised history-to-target windows available inside that split after Stage 2 filtering."
        )

    stats = st.columns(4)
    stats[0].metric("Min Windows / Run", summary["min"])
    stats[1].metric("Median Windows / Run", summary["median"])
    stats[2].metric("P90 Windows / Run", summary["p90"])
    stats[3].metric("Max Windows / Run", summary["max"])

    color = (
        "run_label" if "run_label" in by_run.columns and by_run["run_label"].nunique() > 1 else None
    )
    st.plotly_chart(
        px.histogram(
            by_run,
            x="window_count",
            color=color,
            barmode="overlay",
            category_orders={
                "run_label": [
                    "healthy",
                    "drain",
                    "fiber_cut",
                    "link_degradation",
                    "link_flap",
                    "unknown",
                ]
            },
            title="Distribution of Windows per Run",
        ),
        width="stretch",
    )
    if not by_run.empty:
        outliers = pd.concat(
            [
                by_run.nsmallest(10, "window_count").assign(extreme="lowest"),
                by_run.nlargest(10, "window_count").assign(extreme="highest"),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["run_id"])
        st.dataframe(outliers, width="stretch", hide_index=True)
    if color == "run_label":
        st.caption(
            "Runs are colored by Stage 1 fault label. Healthy runs usually keep almost the full episode, while faulty runs often contribute fewer windows because Stage 2 only keeps pre-fault prediction targets."
        )
    else:
        st.caption(
            "This distribution shows how many usable windows each run contributes. Larger variation here can indicate that some runs dominate the training signal. Fault labels were not available from the current Stage 2 artifacts, so this view is uncolored."
        )


def render_schema(normal_index: dict[str, Any], normalization: dict[str, Any]) -> None:
    """Show the feature counts of the saved schema and of the normalization statistics."""
    st.subheader("Feature Schema")
    st.caption(
        "Stage 2 packages three input modalities for the normal emulator: node features, edge features, and probe features. "
        "Input counts describe what the model sees, while target counts describe what it predicts one step ahead."
    )
    schema_counts = feature_schema_counts(normal_index)
    cols = st.columns(6)
    for col, (_, row) in zip(cols, schema_counts.iterrows()):
        col.metric(str(row["feature_group"]).replace("_", " ").title(), int(row["feature_count"]))

    st.dataframe(schema_counts, width="stretch", hide_index=True)
    st.caption(
        "Feature counts are taken from `normal_index.json` and reflect the saved Stage 2 schema, not inferred tensor shapes."
    )

    st.subheader("Normalization Summary")
    st.caption(
        "Normalization is fit on training runs only. The counts below show how many continuous features in each tensor group are standardized."
    )
    norm_counts = normalization_counts(normalization)
    st.dataframe(norm_counts, width="stretch", hide_index=True)


def render_topology(topology: dict[str, Any]) -> None:
    """Show the node, edge, probe, and candidate counts of the topology."""
    st.subheader("Topology Summary")
    st.caption(
        "This is the fixed graph structure shared across Stage 2 samples. Candidate count refers to the localization label space carried forward in the pipeline."
    )
    counts = topology_counts(topology)
    cols = st.columns(4)
    cols[0].metric("Node Count", counts["node_count"])
    cols[1].metric("Edge Count", counts["edge_count"])
    cols[2].metric("Probe Count", counts["probe_count"])
    cols[3].metric("Candidate Count", counts["candidate_count"])
    st.caption(
        "Candidate IDs include the healthy `none` label plus the node and edge candidates used later for root-cause prediction."
    )


def render_raw_artifacts(dataset_root: Path) -> None:
    """Show the Stage-2 JSON artifacts in collapsed expanders."""
    st.subheader("Artifact Inspector")
    st.caption(
        "These raw files are the exact Stage 2 artifacts behind the summaries above, useful when you want to verify counts or inspect the saved metadata directly."
    )
    render_artifact_inspector_entries(
        [
            (filename, cached_json(str(dataset_root), filename))
            for filename in (
                "normal_index.json",
                "normal_splits.json",
                "normalization.json",
                "topology.json",
            )
        ]
    )


def main() -> None:
    """Validate the dataset root, load the Stage-2 artifacts, and show every section."""
    st.set_page_config(page_title="RIDGE Stage 2 Dataset Explorer", layout="wide")
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None
    if dataset_root is None or not is_stage2_dataset_root(dataset_root):
        st.error(f"Dataset root is not a valid Stage 2 dataset: {dataset_root}")
        st.stop()

    st.title("RIDGE Stage 2 Normal-Emulator Dataset Explorer")
    st.caption(f"Dataset root: {dataset_root}")

    try:
        window_index = cached_window_index(str(dataset_root))
        normal_index = cached_json(str(dataset_root), "normal_index.json")
        splits = cached_json(str(dataset_root), "normal_splits.json")
        normalization = cached_json(str(dataset_root), "normalization.json")
        topology = cached_json(str(dataset_root), "topology.json")
        source_manifest = cached_source_manifest(str(dataset_root))
    except Exception as exc:
        st.error(f"Failed to load Stage 2 dataset: {exc}")
        st.stop()

    render_metrics(splits, window_index, normal_index, topology)
    render_charts(window_index, splits, source_manifest)
    render_schema(normal_index, normalization)
    render_topology(topology)
    render_raw_artifacts(dataset_root)


if __name__ == "__main__":
    main()
