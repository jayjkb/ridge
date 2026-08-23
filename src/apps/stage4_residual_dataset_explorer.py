"""Streamlit explorer for RIDGE Stage 4 residual datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from apps.common import (
    cached_json,
    cached_residual_window_index,
    format_percent,
    is_artifact_dataset_root,
    render_artifact_inspector_entries,
    render_metric_notes,
    split_run_counts_frame,
    split_window_counts,
    topology_counts,
    window_count_summary,
)

REQUIRED_STAGE4_FILES = (
    "residual_window_index.csv",
    "residual_index.json",
    "residual_splits.json",
    "residual_normalization.json",
    "residual_feature_schema.json",
    "candidate_index.json",
    "topology.json",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse the dataset root argument, ignoring the options Streamlit adds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=None, help="Path to Stage 4 residual dataset root"
    )
    return parser.parse_known_args()[0]


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
def cached_run_shard(dataset_root: str, run_id: str) -> dict[str, Any]:
    """Load the residual shard of one episode, cached per session."""
    return torch.load(
        Path(dataset_root) / "residual_runs" / f"run_{int(run_id):06d}.pt",
        map_location="cpu",
        weights_only=False,
    )


def windows_per_run(window_index: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    """Count the windows of each episode with its fault label and split, most windows first."""
    if window_index.empty:
        return pd.DataFrame(columns=["run_id", "window_count", "fault_label", "split"])
    split_map: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for run_id in splits.get(split_name, []):
            split_map[str(run_id)] = split_name
    grouped = (
        window_index.groupby("run_id", as_index=False)
        .agg(
            window_count=("run_id", "size"),
            fault_windows=("fault_present", "sum"),
        )
        .sort_values(["window_count", "run_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    grouped["fault_label"] = grouped["fault_windows"].map(
        lambda value: "faulty" if float(value) > 0 else "healthy"
    )
    grouped["split"] = grouped["run_id"].map(split_map).fillna("unassigned")
    return grouped


def residual_normalization_counts(normalization: dict[str, Any]) -> pd.DataFrame:
    """Return the number of renormalized features per residual family."""
    rows = []
    for name in ("node_residual", "edge_residual", "probe_residual"):
        stats = normalization.get(name, {})
        rows.append(
            {
                "feature_group": name,
                "normalized_feature_count": len(stats.get("feature_indices", []))
                if isinstance(stats, dict)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def residual_topology_counts(
    topology: dict[str, Any], candidates: dict[str, Any]
) -> dict[str, int]:
    """Return the topology counts with the candidate count taken from the candidate index."""
    return {
        **topology_counts(topology),
        "candidate_count": len(candidates.get("candidate_ids", [])),
    }


def build_overview_cards(
    residual_index: dict[str, Any],
    splits: dict[str, list[str]],
    topology: dict[str, Any],
    candidates: dict[str, Any],
    window_index: pd.DataFrame,
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """Build the overview cards and their notes from the index, splits, topology, and window index."""
    run_counts = split_run_counts_frame(splits).set_index("split")["run_count"].to_dict()
    topo_counts = residual_topology_counts(topology, candidates)
    residual_history_len = residual_index.get("residual_history_len")
    if residual_history_len is None and "history_len" in window_index.columns:
        populated = window_index["history_len"].dropna()
        residual_history_len = int(populated.iloc[0]) if not populated.empty else None
    fault_window_ratio = (
        float(window_index["fault_present"].mean()) if not window_index.empty else 0.0
    )

    cards = [
        ("Input Mode", str(residual_index.get("residual_mode", "unknown"))),
        ("Total Runs", str(sum(run_counts.values()))),
        ("Stage 4 Windows", str(int(residual_index.get("sample_count", len(window_index))))),
        ("Train Runs", str(int(run_counts.get("train", 0)))),
        ("Val Runs", str(int(run_counts.get("val", 0)))),
        ("Test Runs", str(int(run_counts.get("test", 0)))),
        ("Nodes", str(topo_counts["node_count"])),
        ("Edges", str(topo_counts["edge_count"])),
        ("Probes", str(topo_counts["probe_count"])),
        ("Candidates", str(topo_counts["candidate_count"])),
        ("Emulator History", str(residual_index.get("emulator_history_len", "n/a"))),
        ("Prediction Horizon", str(residual_index.get("prediction_horizon", "n/a"))),
        ("Residual History", str(residual_history_len or "n/a")),
        ("Fault Windows %", format_percent(fault_window_ratio)),
    ]
    notes = [
        {
            "Metric": "Input Mode",
            "Meaning": "`standardized` uses emulator residuals. `raw` uses normalized observations for the comparison arm.",
        },
        {
            "Metric": "Stage 4 Windows",
            "Meaning": "Supervised Stage 5 histories ending at labeled timestamps.",
        },
        {
            "Metric": "Fault Windows %",
            "Meaning": "Share of Stage 4 windows whose endpoint timestamp falls inside the injected fault interval.",
        },
        {
            "Metric": "Emulator History",
            "Meaning": "Stage 3 lookback used to produce each one-step forecast.",
        },
        {
            "Metric": "Prediction Horizon",
            "Meaning": "Forecast horizon inherited from Stage 2. residualization requires one step.",
        },
        {
            "Metric": "Residual History",
            "Meaning": "Stage 4 timesteps per supervised Stage 5 sample.",
        },
        {
            "Metric": "Candidates",
            "Meaning": "Root-cause label space carried into Stage 5, including the healthy `none` label.",
        },
        {
            "Metric": "Nodes / Edges / Probes",
            "Meaning": "Fixed graph entities whose residual channels are saved at every timestep.",
        },
    ]
    return cards, notes


def render_overview(
    residual_index: dict[str, Any],
    splits: dict[str, list[str]],
    topology: dict[str, Any],
    candidates: dict[str, Any],
    window_index: pd.DataFrame,
) -> None:
    """Show the overview cards with a caption that names the residual mode of the dataset."""
    mode = str(residual_index.get("residual_mode", "unknown"))
    st.subheader("Stage 4 Dataset Overview")
    if mode == "raw":
        st.caption(
            "This comparison dataset stores normalized raw observations; it does not use "
            "normal-emulator residualization."
        )
    else:
        st.caption(
            "This dataset stores standardized deviations from the normal emulator's forecast."
        )
    cards, notes = build_overview_cards(residual_index, splits, topology, candidates, window_index)
    columns = st.columns(4)
    for index, (label, value) in enumerate(cards):
        columns[index % len(columns)].metric(label, value)
    render_metric_notes(notes)


def render_coverage(window_index: pd.DataFrame, splits: dict[str, list[str]]) -> None:
    """Show episode and window counts per split and the per-episode window distribution by fault label."""
    st.subheader("Split and Label Coverage")
    st.caption(
        "These views show how Stage 4 supervision is distributed across splits, runs, and labels so we can spot imbalance before training the RCA model."
    )

    run_counts = split_run_counts_frame(splits)
    window_counts = split_window_counts(window_index, splits)
    by_run = windows_per_run(window_index, splits)
    summary = window_count_summary(by_run)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(run_counts, x="split", y="run_count", title="Run Counts by Split"),
            width="stretch",
        )
        st.caption(
            "Runs are split at the episode level, so every residual timestep from a run stays in the same partition."
        )
    with right:
        st.plotly_chart(
            px.bar(window_counts, x="split", y="window_count", title="Window Counts by Split"),
            width="stretch",
        )
        st.caption(
            "Window counts reflect the number of labeled history windows Stage 5 can train on in each split."
        )

    stats = st.columns(4)
    stats[0].metric("Min Windows / Run", summary["min"])
    stats[1].metric("Median Windows / Run", summary["median"])
    stats[2].metric("P90 Windows / Run", summary["p90"])
    stats[3].metric("Max Windows / Run", summary["max"])

    label_cols = st.columns(2)
    with label_cols[0]:
        category_counts = (
            window_index["fault_category"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("fault_category")
            .reset_index(name="count")
        )
        st.plotly_chart(
            px.bar(
                category_counts, x="fault_category", y="count", title="Fault Category Distribution"
            ),
            width="stretch",
        )
    with label_cols[1]:
        cause_counts = (
            window_index["root_cause_id"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("root_cause_id")
            .reset_index(name="count")
        )
        st.plotly_chart(
            px.bar(cause_counts, x="root_cause_id", y="count", title="Root Cause Distribution"),
            width="stretch",
        )

    st.plotly_chart(
        px.histogram(
            by_run,
            x="window_count",
            color="fault_label",
            barmode="overlay",
            title="Distribution of Windows per Run",
            category_orders={"fault_label": ["healthy", "faulty"]},
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
    st.caption(
        "Healthy-colored runs have no fault-active windows. Faulty-colored runs include both fault-active and non-fault windows because Stage 4 labels by the window endpoint timestamp."
    )


def render_schema(
    feature_schema: dict[str, Any],
    normalization: dict[str, Any],
) -> None:
    """Show the input feature names per family and the renormalization counts."""
    st.subheader("Input Schema and Normalization")
    st.caption(
        "Counts come from the saved artifact schema so the explorer remains tied to the "
        "dataset that was actually built."
    )
    node_names = list(feature_schema.get("node_input_features", []))
    edge_names = list(feature_schema.get("edge_input_features", []))
    probe_names = list(feature_schema.get("probe_input_features", []))
    node_dynamic = [name for name in node_names if str(name).startswith(("Residual_", "Raw_"))]
    edge_dynamic = [name for name in edge_names if str(name).startswith(("Residual_", "Raw_"))]
    schema_rows = [
        {"feature_group": "node dynamic", "saved_feature_count": len(node_dynamic)},
        {"feature_group": "edge dynamic", "saved_feature_count": len(edge_dynamic)},
        {"feature_group": "probe dynamic", "saved_feature_count": len(probe_names)},
        {"feature_group": "node Stage 5 input", "saved_feature_count": len(node_names)},
        {"feature_group": "edge Stage 5 input", "saved_feature_count": len(edge_names)},
        {"feature_group": "probe Stage 5 input", "saved_feature_count": len(probe_names)},
    ]
    st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

    norm_counts = residual_normalization_counts(normalization)
    st.dataframe(norm_counts, width="stretch", hide_index=True)
    st.caption(
        "Normalization statistics are fitted on training runs only. Binary probe channels "
        "remain unstandardized when the saved normalization indices exclude them."
    )


def run_rows(window_index: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Return the window-index rows of one episode ordered by window end."""
    rows = window_index[window_index["run_id"] == str(run_id)].copy()
    return rows.sort_values("window_end_index").reset_index(drop=True)


def summarize_run(shard: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    """Return the mean absolute residual per family at every snapshot of an episode, flagging fault-active snapshots."""
    timestamps = pd.to_datetime(pd.Series(shard["timestamps"]), errors="coerce", utc=True)
    node = shard["node_residual"].abs().mean(dim=(1, 2)).numpy()
    edge = shard["edge_residual"].abs().mean(dim=(1, 2)).numpy()
    probe = shard["probe_residual"].abs().mean(dim=(1, 2)).numpy()
    frame = pd.DataFrame(
        {
            "timestep_index": range(len(timestamps)),
            "timestamp": timestamps,
            "mean_abs_node_residual": node,
            "mean_abs_edge_residual": edge,
            "mean_abs_probe_residual": probe,
            "max_group_magnitude": pd.DataFrame({"node": node, "edge": edge, "probe": probe}).max(
                axis=1
            ),
        }
    )
    fault_indices = (
        rows.loc[rows["fault_present"] > 0.0, "window_end_index"].dropna().astype(int).tolist()
    )
    frame["fault_active"] = frame["timestep_index"].isin(fault_indices)
    return frame


def infer_fault_window(frame: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the first and last fault-active timestamps of an episode, or None."""
    active = frame[frame["fault_active"]]
    if active.empty:
        return None, None
    start = active["timestamp"].dropna().min()
    end = active["timestamp"].dropna().max()
    return (start if pd.notna(start) else None, end if pd.notna(end) else None)


def add_fault_window(fig: go.Figure, start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
    """Shade the fault window on a figure and label it."""
    if start is None or end is None:
        return
    fig.add_vrect(
        x0=start.to_pydatetime(),
        x1=end.to_pydatetime(),
        fillcolor="#ffb703",
        opacity=0.15,
        line_width=0,
    )
    fig.add_annotation(
        x=start.to_pydatetime(),
        y=1,
        xref="x",
        yref="paper",
        text="fault-active window",
        showarrow=False,
        yshift=10,
        font=dict(color="#c77d00"),
    )


def top_surprises(
    shard: dict[str, Any],
    topology: dict[str, Any],
    feature_schema: dict[str, Any],
    timestep_index: int,
    top_k: int = 12,
) -> pd.DataFrame:
    """Return the entity and feature pairs with the largest residual magnitude at one snapshot."""
    rows: list[dict[str, object]] = []
    node_tensor = shard["node_residual"][timestep_index]
    edge_tensor = shard["edge_residual"][timestep_index]
    probe_tensor = shard["probe_residual"][timestep_index]

    def feature_names(family: str, width: int) -> list[str]:
        """Return the saved feature names of a family without their prefix, padded to the tensor width."""
        saved = list(feature_schema.get(f"{family}_input_features", []))[:width]
        names = [str(name).removeprefix("Residual_").removeprefix("Raw_") for name in saved]
        return [*names, *(f"feature_{index}" for index in range(len(names), width))]

    node_features = feature_names("node", int(node_tensor.shape[-1]))
    edge_features = feature_names("edge", int(edge_tensor.shape[-1]))
    probe_features = feature_names("probe", int(probe_tensor.shape[-1]))

    for entity_index, entity_id in enumerate(topology.get("node_ids", [])):
        for feature_index, feature_name in enumerate(node_features):
            value = float(node_tensor[entity_index, feature_index])
            rows.append(
                {
                    "entity_type": "node",
                    "entity_id": entity_id,
                    "feature": feature_name,
                    "residual": value,
                    "abs_residual": abs(value),
                }
            )

    for entity_index, entity_id in enumerate(topology.get("edge_ids", [])):
        for feature_index, feature_name in enumerate(edge_features):
            value = float(edge_tensor[entity_index, feature_index])
            rows.append(
                {
                    "entity_type": "edge",
                    "entity_id": entity_id,
                    "feature": feature_name,
                    "residual": value,
                    "abs_residual": abs(value),
                }
            )

    for entity_index, entity_id in enumerate(topology.get("probe_ids", [])):
        for feature_index, feature_name in enumerate(probe_features):
            value = float(probe_tensor[entity_index, feature_index])
            rows.append(
                {
                    "entity_type": "probe",
                    "entity_id": entity_id,
                    "feature": feature_name,
                    "residual": value,
                    "abs_residual": abs(value),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_residual", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def render_run_drilldown(
    dataset_root: Path,
    window_index: pd.DataFrame,
    topology: dict[str, Any],
    feature_schema: dict[str, Any],
    residual_mode: str,
) -> None:
    """Show one episode's residual magnitude over time and the largest residuals at a chosen snapshot."""
    st.subheader("Run Drilldown")
    st.caption(
        "This view shows per-family input magnitude around fault periods and the largest "
        "entity/channel values at a selected timestep."
    )

    available_runs = sorted(window_index["run_id"].astype(str).unique(), key=int)
    default_run_index = 0
    faulty_runs = [
        run_id
        for run_id in available_runs
        if float(run_rows(window_index, run_id)["fault_present"].sum()) > 0
    ]
    if faulty_runs:
        default_run_index = available_runs.index(faulty_runs[0])
    selected_run = st.selectbox("Select run", available_runs, index=default_run_index)

    rows = run_rows(window_index, selected_run)
    shard = cached_run_shard(str(dataset_root), selected_run)
    frame = summarize_run(shard, rows)
    fault_start, fault_end = infer_fault_window(frame)

    left, right = st.columns((3, 1))
    with left:
        long_frame = frame.melt(
            id_vars=["timestep_index", "timestamp", "fault_active"],
            value_vars=[
                "mean_abs_node_residual",
                "mean_abs_edge_residual",
                "mean_abs_probe_residual",
            ],
            var_name="metric",
            value_name="value",
        )
        long_frame["metric"] = long_frame["metric"].map(
            {
                "mean_abs_node_residual": "node",
                "mean_abs_edge_residual": "edge",
                "mean_abs_probe_residual": "probe",
            }
        )
        fig = px.line(
            long_frame,
            x="timestamp",
            y="value",
            color="metric",
            markers=True,
            title=f"Stage 4 Input Magnitude Over Time: run {selected_run}",
            labels={"value": "Mean absolute input", "metric": "Group"},
        )
        add_fault_window(fig, fault_start, fault_end)
        st.plotly_chart(fig, width="stretch")

    with right:
        fault_windows = int((rows["fault_present"] > 0.0).sum())
        st.metric("Input Timesteps", len(frame))
        st.metric("Input Mode", residual_mode)
        st.metric("Fault Windows", fault_windows)
        st.metric(
            "Fault Category",
            rows.loc[rows["fault_present"] > 0.0, "fault_category"].iloc[0]
            if fault_windows
            else "none",
        )
        st.metric(
            "Root Cause",
            rows.loc[rows["fault_present"] > 0.0, "root_cause_id"].iloc[0]
            if fault_windows
            else "none",
        )

    peak_index = int(frame["max_group_magnitude"].idxmax()) if not frame.empty else 0
    timestep_index = st.slider(
        "Selected timestep index", min_value=0, max_value=max(0, len(frame) - 1), value=peak_index
    )
    selected_row = frame.iloc[timestep_index]
    st.caption(
        f"Selected timestamp: {selected_row['timestamp']} | fault active: {'yes' if bool(selected_row['fault_active']) else 'no'}"
    )
    surprise_rows = top_surprises(shard, topology, feature_schema, timestep_index)
    st.dataframe(surprise_rows, width="stretch", hide_index=True)
    st.caption(
        "Rows are ranked by absolute input magnitude. In standardized mode this is "
        "surprise relative to the emulator. In raw mode it is normalized observation magnitude."
    )


def render_artifact_inspector(
    dataset_root: Path,
    residual_index: dict[str, Any],
    splits: dict[str, Any],
    normalization: dict[str, Any],
    feature_schema: dict[str, Any],
    candidates: dict[str, Any],
    topology: dict[str, Any],
) -> None:
    """Show the Stage-4 JSON artifacts in collapsed expanders."""
    st.subheader("Artifact Inspector")
    st.caption(
        "These raw files back the summaries above and are useful for quick verification or debugging."
    )
    artifacts = [
        ("residual_index.json", residual_index),
        ("residual_splits.json", splits),
        ("residual_normalization.json", normalization),
        ("residual_feature_schema.json", feature_schema),
        ("candidate_index.json", candidates),
        ("topology.json", topology),
    ]
    render_artifact_inspector_entries(artifacts)
    st.caption(f"Dataset root: {dataset_root}")


def main() -> None:
    """Validate the dataset root, load the Stage-4 artifacts, and show every section."""
    st.set_page_config(page_title="RIDGE Stage 4 Dataset Explorer", layout="wide")
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None
    if dataset_root is None or not is_stage4_dataset_root(dataset_root):
        st.error(f"Dataset root is not a valid Stage 4 residual dataset: {dataset_root}")
        st.stop()

    st.title("RIDGE Stage 4 Dataset Explorer")
    st.caption(f"Dataset root: {dataset_root}")

    try:
        window_index = cached_residual_window_index(str(dataset_root))
        residual_index = cached_json(str(dataset_root), "residual_index.json")
        splits = cached_json(str(dataset_root), "residual_splits.json")
        normalization = cached_json(str(dataset_root), "residual_normalization.json")
        feature_schema = cached_json(str(dataset_root), "residual_feature_schema.json")
        candidates = cached_json(str(dataset_root), "candidate_index.json")
        topology = cached_json(str(dataset_root), "topology.json")
    except Exception as exc:
        st.error(f"Failed to load Stage 4 artifacts: {exc}")
        st.stop()

    render_overview(residual_index, splits, topology, candidates, window_index)
    render_coverage(window_index, splits)
    render_schema(feature_schema, normalization)
    render_run_drilldown(
        dataset_root,
        window_index,
        topology,
        feature_schema,
        str(residual_index.get("residual_mode", "unknown")),
    )
    render_artifact_inspector(
        dataset_root, residual_index, splits, normalization, feature_schema, candidates, topology
    )


if __name__ == "__main__":
    main()
