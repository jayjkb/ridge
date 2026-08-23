"""Streamlit explorer for RIDGE Stage 1 raw datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.common import (
    json_artifact_matches,
    render_metric_notes,
)
from ridge.common.contracts import STAGE1_ARTIFACT_TYPE
from ridge.io.stage1_dataset import (
    RunArtifacts,
    load_manifest,
    load_realism_artifacts,
    load_run_artifacts,
    resolve_run_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse the dataset root argument, ignoring the options Streamlit adds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=None, help="Path to Stage 1 dataset root"
    )
    return parser.parse_known_args()[0]


def is_stage1_dataset_root(path: Path) -> bool:
    """Return whether a path holds a manifest and a valid Stage-1 provenance artifact."""
    return bool(
        path.is_dir()
        and (path / "manifest.csv").is_file()
        and json_artifact_matches(
            path / "generation_provenance.json",
            STAGE1_ARTIFACT_TYPE,
        )
    )


def _readable_count(series: pd.Series) -> str:
    """Return the integer sum of a series as text, or zero when it is empty."""
    return f"{int(series.sum())}" if len(series) else "0"


def _has_col(frame: pd.DataFrame, col: str) -> bool:
    """Return whether a data frame has the named column."""
    return col in frame.columns


def _to_numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    """Return a column coerced to numbers, or an empty float series when absent."""
    if col not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _to_timestamp_series(frame: pd.DataFrame, col: str = "Timestamp") -> pd.Series:
    """Return a column parsed as UTC timestamps, or an empty series when absent."""
    if col not in frame.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[col], errors="coerce", utc=True)


def _fault_window(artifacts: RunArtifacts) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the fault onset and end timestamps of an episode, None when unset."""
    start = pd.to_datetime(
        artifacts.run_metadata.get("fault_start_ts", ""), errors="coerce", utc=True
    )
    end = pd.to_datetime(artifacts.run_metadata.get("fault_end_ts", ""), errors="coerce", utc=True)
    return (start if pd.notna(start) else None, end if pd.notna(end) else None)


def _add_fault_markers(
    fig: go.Figure, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> None:
    """Draw labelled vertical lines at the fault onset and end of a figure."""
    if start is not None:
        start_dt = start.to_pydatetime()
        fig.add_vline(x=start_dt, line_color="#ff7f0e", line_dash="dash", line_width=1.5)
        fig.add_annotation(
            x=start_dt,
            y=1,
            xref="x",
            yref="paper",
            text="fault_start",
            showarrow=False,
            yshift=10,
            font=dict(color="#ff7f0e"),
        )
    if end is not None:
        end_dt = end.to_pydatetime()
        fig.add_vline(x=end_dt, line_color="#1d3557", line_dash="dash", line_width=1.5)
        fig.add_annotation(
            x=end_dt,
            y=1,
            xref="x",
            yref="paper",
            text="fault_end",
            showarrow=False,
            yshift=10,
            font=dict(color="#1d3557"),
        )


@st.cache_data(show_spinner=False)
def cached_manifest(dataset_root: str) -> pd.DataFrame:
    """Load the manifest of a dataset root, cached per session."""
    return load_manifest(Path(dataset_root))


@st.cache_data(show_spinner=False)
def cached_realism(dataset_root: str) -> tuple[dict, dict]:
    """Load the realism summary and checks of a dataset root, cached per session."""
    return load_realism_artifacts(Path(dataset_root))


@st.cache_data(show_spinner=False)
def cached_run_artifacts(run_dir: str) -> RunArtifacts:
    """Load the artifacts of an episode directory, cached per session."""
    return load_run_artifacts(Path(run_dir))


def render_dataset_summary(manifest: pd.DataFrame) -> None:
    """Show episode counts, status, and elapsed-time cards with their explanations."""
    st.subheader("Dataset Summary")
    healthy = manifest.get("is_healthy_baseline", pd.Series(dtype=float)).fillna(0).astype(float)
    faulty = manifest.get("is_fault_episode", pd.Series(dtype=float)).fillna(0).astype(float)
    elapsed = _to_numeric_series(manifest, "elapsed_sec")
    status_ok = (
        manifest["status"].astype(str) == "ok"
        if _has_col(manifest, "status")
        else pd.Series(dtype=bool)
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Runs", len(manifest))
    col2.metric("Healthy Runs", _readable_count(healthy))
    col3.metric("Fault Runs", _readable_count(faulty))
    col4.metric(
        "Mean Elapsed Sec", f"{elapsed.dropna().mean():.2f}" if elapsed.notna().any() else "n/a"
    )
    ok_ratio = status_ok.mean() if len(status_ok) else 0.0
    col5.metric("Status OK %", f"{ok_ratio * 100:.1f}%")
    render_metric_notes(
        [
            {
                "Metric": "Total Runs",
                "Meaning": "How many Stage 1 runs are listed in the manifest.",
            },
            {
                "Metric": "Healthy Runs",
                "Meaning": "Runs labeled as healthy baseline episodes with no injected fault.",
            },
            {
                "Metric": "Fault Runs",
                "Meaning": "Runs labeled as fault episodes with an injected failure or anomaly.",
            },
            {
                "Metric": "Mean Elapsed Sec",
                "Meaning": "Average wall-clock runtime of a Stage 1 run.",
            },
            {
                "Metric": "Status OK %",
                "Meaning": "Share of runs whose final status was recorded as `ok`.",
            },
        ]
    )

    left, right = st.columns(2)
    with left:
        if _has_col(manifest, "fault_category"):
            st.plotly_chart(
                px.histogram(manifest, x="fault_category", title="Fault Category"),
                width="stretch",
            )
        if _has_col(manifest, "root_cause_kind"):
            st.plotly_chart(
                px.histogram(manifest, x="root_cause_kind", title="Root Cause Kind"),
                width="stretch",
            )
    with right:
        if _has_col(manifest, "fault_target"):
            st.plotly_chart(
                px.histogram(manifest, x="fault_target", title="Fault Target Frequency"),
                width="stretch",
            )
        if _has_col(manifest, "returncode"):
            rc_counts = (
                _to_numeric_series(manifest, "returncode")
                .fillna(-1)
                .astype(int)
                .value_counts()
                .sort_index()
            )
            st.plotly_chart(
                px.bar(
                    x=rc_counts.index.astype(str),
                    y=rc_counts.values,
                    labels={"x": "returncode", "y": "count"},
                    title="Returncode Distribution",
                ),
                width="stretch",
            )


def render_manifest_extended(manifest: pd.DataFrame) -> None:
    """Show the validity check pass rate, baseline probe statistics, and further manifest columns."""
    st.subheader("Manifest Details")

    baseline_pass = manifest.get("baseline_health_pass", pd.Series(dtype=bool))
    if len(baseline_pass):
        baseline_pass = baseline_pass.astype(str).str.lower().isin({"1", "true", "yes"})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline Health Pass %", f"{baseline_pass.mean() * 100:.1f}%")
        c2.metric(
            "Mean Baseline Loss %",
            f"{_to_numeric_series(manifest, 'baseline_mean_packet_loss_pct').dropna().mean():.3f}",
        )
        c3.metric(
            "Mean Baseline Timeout Ratio",
            f"{_to_numeric_series(manifest, 'baseline_timeout_ratio').dropna().mean():.3f}",
        )
        c4.metric(
            "Mean Baseline p95 RTT (ms)",
            f"{_to_numeric_series(manifest, 'baseline_p95_rtt_ms').dropna().mean():.2f}",
        )
        render_metric_notes(
            [
                {
                    "Metric": "Baseline Health Pass %",
                    "Meaning": "Share of runs whose baseline health check passed before the main episode.",
                },
                {
                    "Metric": "Mean Baseline Loss %",
                    "Meaning": "Average packet-loss percentage seen during the baseline check.",
                },
                {
                    "Metric": "Mean Baseline Timeout Ratio",
                    "Meaning": "Average fraction of baseline probes that timed out.",
                },
                {
                    "Metric": "Mean Baseline p95 RTT (ms)",
                    "Meaning": "Average 95th-percentile ping latency during the baseline check.",
                },
            ]
        )

    l1, l2 = st.columns(2)
    with l1:
        if _has_col(manifest, "routing_mode_requested") and _has_col(
            manifest, "routing_mode_effective"
        ):
            st.plotly_chart(
                px.histogram(
                    manifest,
                    x="routing_mode_requested",
                    color="routing_mode_effective",
                    title="Routing Mode Requested vs Effective",
                ),
                width="stretch",
            )
        if _has_col(manifest, "routing_mode_note"):
            note_counts = (
                manifest["routing_mode_note"]
                .fillna("")
                .astype(str)
                .replace("", "none")
                .value_counts()
            )
            st.plotly_chart(
                px.bar(
                    x=note_counts.index,
                    y=note_counts.values,
                    labels={"x": "routing_mode_note", "y": "count"},
                    title="Routing Mode Notes",
                ),
                width="stretch",
            )
    with l2:
        if _has_col(manifest, "traffic_flow_count") and _has_col(manifest, "ping_pair_count"):
            traffic_df = pd.DataFrame(
                {
                    "traffic_flow_count": _to_numeric_series(manifest, "traffic_flow_count"),
                    "ping_pair_count": _to_numeric_series(manifest, "ping_pair_count"),
                    "fault_category": manifest.get(
                        "fault_category", pd.Series(["unknown"] * len(manifest))
                    ).astype(str),
                }
            ).dropna(subset=["traffic_flow_count", "ping_pair_count"])
            if not traffic_df.empty:
                st.plotly_chart(
                    px.scatter(
                        traffic_df,
                        x="traffic_flow_count",
                        y="ping_pair_count",
                        color="fault_category",
                        title="Traffic Flows vs Ping Pairs",
                    ),
                    width="stretch",
                )
        if _has_col(manifest, "traffic_profile_id"):
            st.plotly_chart(
                px.histogram(
                    manifest, x="traffic_profile_id", title="Traffic Profile Distribution"
                ),
                width="stretch",
            )


def render_realism(dataset_root: Path, manifest: pd.DataFrame) -> None:
    """Show traffic flow, probe pair, and congestion burst statistics with the realism checks."""
    st.subheader("Traffic Realism")
    summary, checks = cached_realism(str(dataset_root))
    cols = st.columns(4)
    cols[0].metric(
        "Avg Flow Count",
        f"{manifest.get('traffic_flow_count', pd.Series(dtype=float)).dropna().mean():.2f}"
        if "traffic_flow_count" in manifest
        else "n/a",
    )
    cols[1].metric(
        "Avg Ping Pair Count",
        f"{manifest.get('ping_pair_count', pd.Series(dtype=float)).dropna().mean():.2f}"
        if "ping_pair_count" in manifest
        else "n/a",
    )
    cols[2].metric(
        "Traffic Profiles",
        str(manifest.get("traffic_profile_id", pd.Series(dtype=object)).nunique())
        if "traffic_profile_id" in manifest
        else "n/a",
    )
    cols[3].metric(
        "Baseline Pass Rate",
        f"{float(summary.get('baseline_health_pass_rate', 0.0)) * 100:.1f}%" if summary else "n/a",
    )
    render_metric_notes(
        [
            {
                "Metric": "Avg Flow Count",
                "Meaning": "Average number of traffic flows configured per run.",
            },
            {
                "Metric": "Avg Ping Pair Count",
                "Meaning": "Average number of end-to-end ping probe pairs per run.",
            },
            {
                "Metric": "Traffic Profiles",
                "Meaning": "How many distinct traffic profile IDs appear in the dataset.",
            },
            {
                "Metric": "Baseline Pass Rate",
                "Meaning": "Share of runs that passed the baseline health check in the realism summary.",
            },
        ]
    )

    st.caption("Realism Checks")
    check_items = [
        ("Flow variability", "flow_variability_non_trivial"),
        ("Probe coverage", "probe_coverage_non_trivial"),
        ("Fault labels", "single_fault_label_integrity"),
        ("Burst traffic", "burst_presence"),
        ("L3 scope only", "l3_scope_only"),
    ]
    check_cols = st.columns(len(check_items))
    for col, (label, key) in zip(check_cols, check_items):
        value = checks.get(key, None)
        if value is True:
            col.metric(label, "PASS")
        elif value is False:
            col.metric(label, "FAIL")
        else:
            col.metric(label, "n/a")
    render_metric_notes(
        [
            {
                "Metric": "Flow variability",
                "Meaning": "Checks that traffic flow counts or patterns vary across runs.",
            },
            {
                "Metric": "Probe coverage",
                "Meaning": "Checks that ping probes cover a meaningful set of source-destination pairs.",
            },
            {
                "Metric": "Fault labels",
                "Meaning": "Checks that each run has a clean, single fault label.",
            },
            {
                "Metric": "Burst traffic",
                "Meaning": "Checks whether bursty traffic behavior is present in the dataset.",
            },
            {
                "Metric": "L3 scope only",
                "Meaning": "Checks that the generated faults stay within the intended Layer 3 scope.",
            },
        ]
    )


def _graph_figure(nodes: pd.DataFrame, links: pd.DataFrame) -> go.Figure:
    """Draw the topology as a spring-layout graph with link attributes on hover."""
    graph = nx.Graph()
    for _, row in nodes.iterrows():
        graph.add_node(str(row["Node"]), role=str(row.get("Role", "unknown")))

    for _, row in links.iterrows():
        graph.add_edge(
            str(row["Source"]),
            str(row["Destination"]),
            profile=str(row.get("Profile", "")),
            bw=row.get("BandwidthMbps", ""),
            delay=row.get("Delay", ""),
            loss=row.get("LossPercent", ""),
        )

    pos = nx.spring_layout(graph, seed=42)
    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_text: list[str] = []
    for src, dst, attrs in graph.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(
            f"{src} ↔ {dst}<br>profile={attrs.get('profile')}<br>bw={attrs.get('bw')} Mbps"
            f"<br>delay={attrs.get('delay')}<br>loss={attrs.get('loss')}"
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color="#7f8ea3"),
        hoverinfo="none",
        mode="lines",
    )

    role_color = {"host": "#2a9d8f", "tor": "#e76f51", "spine": "#264653"}
    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_colors: list[str] = []
    for name, attrs in graph.nodes(data=True):
        x, y = pos[name]
        role = str(attrs.get("role", "unknown"))
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{name}<br>role={role}")
        node_colors.append(role_color.get(role, "#888888"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_text,
        text=[n for n in graph.nodes()],
        textposition="top center",
        marker=dict(size=13, color=node_colors, line=dict(width=1, color="#ffffff")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Network Topology",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    return fig


def _require_columns(frame: pd.DataFrame, required: set[str], frame_name: str) -> None:
    """Raise ValueError naming the columns a frame lacks."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {', '.join(missing)}")


def _compute_link_utilization(links: pd.DataFrame, iface: pd.DataFrame) -> pd.DataFrame:
    """Compute per-link utilization over time from the busier direction of each endpoint interface."""
    _require_columns(
        links,
        {
            "EdgeId",
            "Source",
            "Destination",
            "SourceInterface",
            "DestinationInterface",
            "BandwidthMbps",
        },
        "topology_links",
    )
    _require_columns(
        iface,
        {"Timestamp", "Node", "Interface", "TX_KBPS", "RX_KBPS"},
        "interface_stats",
    )

    links_df = links.copy()
    iface_df = iface.copy()
    links_df["BandwidthMbps"] = pd.to_numeric(links_df["BandwidthMbps"], errors="coerce")
    links_df = links_df[links_df["BandwidthMbps"] > 0].copy()
    if links_df.empty:
        raise ValueError("topology_links has no rows with valid positive BandwidthMbps.")

    iface_df["TX_KBPS"] = pd.to_numeric(iface_df["TX_KBPS"], errors="coerce")
    iface_df["RX_KBPS"] = pd.to_numeric(iface_df["RX_KBPS"], errors="coerce")
    iface_df["Timestamp"] = pd.to_datetime(iface_df["Timestamp"], errors="coerce", utc=True)
    iface_df = iface_df.dropna(subset=["Timestamp"]).copy()
    iface_df["endpoint_peak_kbps"] = iface_df[["TX_KBPS", "RX_KBPS"]].max(axis=1, skipna=True)

    endpoint_a = links_df[["EdgeId", "BandwidthMbps", "Source", "SourceInterface"]].rename(
        columns={"Source": "Node", "SourceInterface": "Interface"}
    )
    endpoint_a["EndpointSide"] = "source"
    endpoint_b = links_df[
        ["EdgeId", "BandwidthMbps", "Destination", "DestinationInterface"]
    ].rename(columns={"Destination": "Node", "DestinationInterface": "Interface"})
    endpoint_b["EndpointSide"] = "destination"
    endpoints = pd.concat([endpoint_a, endpoint_b], ignore_index=True)

    joined = endpoints.merge(
        iface_df[["Timestamp", "Node", "Interface", "endpoint_peak_kbps"]],
        on=["Node", "Interface"],
        how="inner",
    )
    if joined.empty:
        raise ValueError(
            "No link/interface stats join matches found. "
            "Verify topology_links SourceInterface/DestinationInterface and interface_stats Node/Interface naming."
        )

    joined["endpoint_util_pct_raw"] = (
        100.0 * joined["endpoint_peak_kbps"] / (joined["BandwidthMbps"] * 1000.0)
    )
    per_link = (
        joined.groupby(["Timestamp", "EdgeId"], as_index=False)
        .agg(
            link_util_pct_raw=("endpoint_util_pct_raw", "mean"),
            endpoint_samples=("EndpointSide", "nunique"),
        )
        .sort_values(["Timestamp", "EdgeId"])
    )
    per_link["is_partial"] = per_link["endpoint_samples"] < 2
    per_link["link_util_pct"] = per_link["link_util_pct_raw"].clip(lower=0.0, upper=100.0)
    return per_link


def _render_link_utilization(
    artifacts: RunArtifacts,
    time_window: tuple[pd.Timestamp, pd.Timestamp] | None,
    fault_start: pd.Timestamp | None,
    fault_end: pd.Timestamp | None,
) -> None:
    """Show peak and mean link utilization cards and the utilization time series with fault markers."""
    st.subheader("Link Utilization")
    util = _compute_link_utilization(artifacts.topology_links, artifacts.interface_stats)
    if util.empty:
        st.warning(
            "Link utilization could not be computed because no usable time-series rows were produced after filtering."
        )
        return

    peak_row = util.loc[util["link_util_pct"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Link Utilization %", f"{util['link_util_pct'].max():.2f}%")
    c2.metric("Mean Link Utilization %", f"{util['link_util_pct'].mean():.2f}%")
    c3.metric(
        "Links >80% At Least Once",
        f"{int((util.groupby('EdgeId')['link_util_pct'].max() > 80.0).sum())}",
    )
    c4.metric("Timestamp Of Global Peak", str(peak_row["Timestamp"]))
    render_metric_notes(
        [
            {
                "Metric": "Peak Link Utilization %",
                "Meaning": "Highest observed link load, relative to configured link bandwidth.",
            },
            {
                "Metric": "Mean Link Utilization %",
                "Meaning": "Average link load across all links and timestamps.",
            },
            {
                "Metric": "Links >80% At Least Once",
                "Meaning": "Number of links that crossed 80% utilization at any point.",
            },
            {
                "Metric": "Timestamp Of Global Peak",
                "Meaning": "Time when the highest single link utilization was observed.",
            },
        ]
    )
    if bool(util["is_partial"].any()):
        st.caption(
            "Some samples were computed from one endpoint only due to missing counterpart interface rows."
        )

    agg = (
        util.groupby("Timestamp", as_index=False)
        .agg(
            mean_util_pct=("link_util_pct", "mean"),
            p95_util_pct=("link_util_pct", lambda s: float(s.quantile(0.95))),
            max_util_pct=("link_util_pct", "max"),
        )
        .sort_values("Timestamp")
    )
    agg_fig = px.line(
        agg,
        x="Timestamp",
        y=["mean_util_pct", "p95_util_pct", "max_util_pct"],
        title="Aggregate Link Utilization Over Time (%)",
    )
    _add_fault_markers(agg_fig, fault_start, fault_end)
    if time_window is not None:
        start_ts, end_ts = time_window
        agg_fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
    st.plotly_chart(agg_fig, width="stretch")

    top_n = st.slider("Top busy links (N)", min_value=1, max_value=12, value=5, step=1)
    top_edges = (
        util.groupby("EdgeId", as_index=False)["link_util_pct"]
        .max()
        .sort_values("link_util_pct", ascending=False)
        .head(top_n)["EdgeId"]
        .tolist()
    )
    top_df = util[util["EdgeId"].isin(top_edges)].copy()
    top_fig = px.line(
        top_df,
        x="Timestamp",
        y="link_util_pct",
        color="EdgeId",
        title=f"Top {top_n} Busiest Links (%)",
    )
    _add_fault_markers(top_fig, fault_start, fault_end)
    if time_window is not None:
        start_ts, end_ts = time_window
        top_fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
    st.plotly_chart(top_fig, width="stretch")

    all_edges = sorted(util["EdgeId"].astype(str).unique().tolist())
    selected_edge = st.selectbox("Link drilldown", all_edges, index=0)
    edge_df = util[util["EdgeId"].astype(str) == selected_edge].copy()
    edge_fig = px.line(
        edge_df,
        x="Timestamp",
        y=["link_util_pct", "link_util_pct_raw"],
        title=f"Link Utilization Detail: {selected_edge}",
    )
    _add_fault_markers(edge_fig, fault_start, fault_end)
    if time_window is not None:
        start_ts, end_ts = time_window
        edge_fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
    st.plotly_chart(edge_fig, width="stretch")


def _render_fault_timeline(
    artifacts: RunArtifacts, time_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
) -> None:
    """Show every fault log action on a timeline with its parameters on hover."""
    st.subheader("Fault Timeline")
    fig = go.Figure()
    fault_df = artifacts.fault_log.copy()
    if not fault_df.empty and "Timestamp" in fault_df.columns:
        fault_df["Timestamp"] = pd.to_datetime(fault_df["Timestamp"], errors="coerce", utc=True)
        fault_df = fault_df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        if not fault_df.empty:
            hover_columns = [
                column
                for column in ("Action", "Target", "Parameters", "FaultType", "FaultCategory")
                if column in fault_df.columns
            ]
            hover_text = (
                fault_df[hover_columns]
                .fillna("")
                .apply(
                    lambda row: "<br>".join(
                        f"{column}: {row[column]}" for column in hover_columns if str(row[column])
                    ),
                    axis=1,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fault_df["Timestamp"],
                    y=["fault action"] * len(fault_df),
                    mode="markers",
                    marker=dict(size=9, color="#d62828", symbol="diamond"),
                    text=hover_text,
                    hovertemplate="%{x}<br>%{text}<extra></extra>",
                    name="fault log",
                )
            )

    # Keep a stable x-axis domain even when there are no timeline points.
    if time_window is not None:
        start_ts, end_ts = time_window
        fig.add_trace(
            go.Scatter(
                x=[start_ts.to_pydatetime(), end_ts.to_pydatetime()],
                y=[0, 0],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    start, end = _fault_window(artifacts)
    _add_fault_markers(fig, start, end)

    # Visual baseline so timeline markers have an explicit horizontal axis.
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        xref="paper",
        yref="y",
        line=dict(color="#5e6472", width=1),
    )

    fig.update_layout(
        height=180,
        xaxis_title=None,
        yaxis_title=None,
        yaxis=dict(title=None),
        margin=dict(l=20, r=20, t=20, b=10),
    )
    if time_window is not None:
        start_ts, end_ts = time_window
        fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
    st.plotly_chart(fig, width="stretch")


def _render_route_stats(
    artifacts: RunArtifacts,
    time_window: tuple[pd.Timestamp, pd.Timestamp] | None,
    fault_start: pd.Timestamp | None,
    fault_end: pd.Timestamp | None,
) -> None:
    """Show route counts by origin over time with fault markers."""
    st.subheader("Route Health")
    route = artifacts.route_stats.copy()
    if route.empty:
        st.info("No route_stats.csv found for this run.")
        return
    if "Timestamp" in route.columns:
        route = route.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    route["RouteCount"] = _to_numeric_series(route, "RouteCount")
    route["OspfRouteCount"] = _to_numeric_series(route, "OspfRouteCount")
    route["KernelRouteCount"] = _to_numeric_series(route, "KernelRouteCount")
    route["StaticRouteCount"] = _to_numeric_series(route, "StaticRouteCount")
    route["HostPrefixRouteCount"] = _to_numeric_series(route, "HostPrefixRouteCount")

    node_options = (
        sorted(route["Node"].astype(str).unique().tolist()) if _has_col(route, "Node") else []
    )
    selected_nodes = st.multiselect("Route nodes", node_options, default=node_options)
    if selected_nodes and _has_col(route, "Node"):
        route = route[route["Node"].astype(str).isin(selected_nodes)]
    if route.empty:
        st.warning("No routing samples for selected node filter.")
        return

    route_churn = pd.Series(dtype=float)
    if _has_col(route, "Node"):
        route_churn = route.groupby("Node")["RouteCount"].agg(lambda s: s.max() - s.min())
    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Mean Route Count",
        f"{route['RouteCount'].dropna().mean():.2f}"
        if route["RouteCount"].notna().any()
        else "n/a",
    )
    k2.metric(
        "Mean OSPF Route Count",
        f"{route['OspfRouteCount'].dropna().mean():.2f}"
        if route["OspfRouteCount"].notna().any()
        else "n/a",
    )
    k3.metric(
        "Nodes With Route Churn", f"{int((route_churn > 0).sum())}" if len(route_churn) else "0"
    )
    render_metric_notes(
        [
            {
                "Metric": "Mean Route Count",
                "Meaning": "Average total number of routing-table entries per sampled node and time.",
            },
            {
                "Metric": "Mean OSPF Route Count",
                "Meaning": "Average number of OSPF-learned routes per sampled node and time.",
            },
            {
                "Metric": "Nodes With Route Churn",
                "Meaning": "How many nodes changed their route count during the selected time span.",
            },
        ]
    )

    agg = route.groupby("Timestamp", as_index=False)[["RouteCount", "OspfRouteCount"]].mean(
        numeric_only=True
    )
    agg_fig = px.line(
        agg,
        x="Timestamp",
        y=["RouteCount", "OspfRouteCount"],
        title="Aggregate Routing Table Size Over Time",
    )
    _add_fault_markers(agg_fig, fault_start, fault_end)
    if time_window is not None:
        start_ts, end_ts = time_window
        agg_fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
    st.plotly_chart(agg_fig, width="stretch")

    if _has_col(route, "Node"):
        heat = route.pivot_table(
            index="Node", columns="Timestamp", values="RouteCount", aggfunc="mean"
        )
        if not heat.empty:
            st.plotly_chart(
                px.imshow(
                    heat,
                    aspect="auto",
                    title="Route Count Heatmap (Node x Time)",
                    labels={"x": "Timestamp", "y": "Node", "color": "RouteCount"},
                ),
                width="stretch",
            )

        focus_node = st.selectbox(
            "Route node drilldown", sorted(route["Node"].astype(str).unique().tolist()), index=0
        )
        node_route = route[route["Node"].astype(str) == focus_node].copy()
        if not node_route.empty:
            node_fig = px.line(
                node_route,
                x="Timestamp",
                y=[
                    "RouteCount",
                    "OspfRouteCount",
                    "KernelRouteCount",
                    "StaticRouteCount",
                    "HostPrefixRouteCount",
                ],
                title=f"Routing Counters for {focus_node}",
            )
            _add_fault_markers(node_fig, fault_start, fault_end)
            if time_window is not None:
                start_ts, end_ts = time_window
                node_fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
            st.plotly_chart(node_fig, width="stretch")


def _render_queue_stats(
    artifacts: RunArtifacts,
    time_window: tuple[pd.Timestamp, pd.Timestamp] | None,
    fault_start: pd.Timestamp | None,
    fault_end: pd.Timestamp | None,
) -> None:
    """Show queue drops, backlog, and overlimits over time with fault markers."""
    st.subheader("Queue Dynamics")
    queue = artifacts.queue_stats.copy()
    if queue.empty:
        st.info("No queue_stats.csv found for this run.")
        return
    if "Timestamp" in queue.columns:
        queue = queue.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    for col in [
        "Bytes",
        "Packets",
        "Drops",
        "Overlimits",
        "Backlog_Bytes",
        "Backlog_Packets",
        "Requeues",
    ]:
        queue[col] = _to_numeric_series(queue, col)

    nodes = sorted(queue["Node"].astype(str).unique().tolist()) if _has_col(queue, "Node") else []
    qdiscs = (
        sorted(queue["Qdisc"].astype(str).unique().tolist()) if _has_col(queue, "Qdisc") else []
    )
    selected_nodes = st.multiselect("Queue nodes", nodes, default=nodes)
    selected_qdiscs = st.multiselect("Qdisc", qdiscs, default=qdiscs)
    if selected_nodes and _has_col(queue, "Node"):
        queue = queue[queue["Node"].astype(str).isin(selected_nodes)]
    if selected_qdiscs and _has_col(queue, "Qdisc"):
        queue = queue[queue["Qdisc"].astype(str).isin(selected_qdiscs)]
    if queue.empty:
        st.warning("No queue samples for selected filters.")
        return

    identity = [column for column in ("Node", "Interface", "Qdisc") if column in queue.columns]
    latest = (
        queue.sort_values("Timestamp").groupby(identity, as_index=False).tail(1)
        if identity
        else queue.tail(1)
    )
    interface_identity = [column for column in ("Node", "Interface") if column in latest.columns]
    final_counters = (
        latest.groupby(interface_identity, as_index=False)[["Drops", "Overlimits"]].max()
        if interface_identity
        else latest
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Drops", f"{int(final_counters['Drops'].fillna(0).sum())}")
    c2.metric("Final Overlimits", f"{int(final_counters['Overlimits'].fillna(0).sum())}")
    c3.metric("Peak Backlog (Bytes)", f"{int(queue['Backlog_Bytes'].fillna(0).max())}")
    c4.metric("Peak Backlog (Packets)", f"{int(queue['Backlog_Packets'].fillna(0).max())}")
    render_metric_notes(
        [
            {
                "Metric": "Final Drops",
                "Meaning": "Final cumulative drop counters, summed once per selected interface.",
            },
            {
                "Metric": "Final Overlimits",
                "Meaning": "Final cumulative overlimit counters, summed once per selected interface.",
            },
            {
                "Metric": "Peak Backlog (Bytes)",
                "Meaning": "Largest queued byte backlog seen at any sampled time.",
            },
            {
                "Metric": "Peak Backlog (Packets)",
                "Meaning": "Largest queued packet backlog seen at any sampled time.",
            },
        ]
    )

    agg = queue.groupby("Timestamp", as_index=False)[
        ["Drops", "Overlimits", "Backlog_Bytes", "Backlog_Packets"]
    ].sum(numeric_only=True)
    backlog_fig = px.line(
        agg,
        x="Timestamp",
        y=["Backlog_Bytes", "Backlog_Packets"],
        title="Aggregate Queue Backlog Over Time",
    )
    drops_fig = px.line(
        agg,
        x="Timestamp",
        y=["Drops", "Overlimits"],
        title="Aggregate Queue Drops/Overlimits Over Time",
    )
    for fig in (backlog_fig, drops_fig):
        _add_fault_markers(fig, fault_start, fault_end)
        if time_window is not None:
            start_ts, end_ts = time_window
            fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
    st.plotly_chart(backlog_fig, width="stretch")
    st.plotly_chart(drops_fig, width="stretch")

    if _has_col(queue, "Interface"):
        offenders = (
            final_counters.groupby("Interface", as_index=False)[["Drops", "Overlimits"]]
            .sum(numeric_only=True)
            .sort_values(["Drops", "Overlimits"], ascending=False)
            .head(12)
        )
        if not offenders.empty:
            offenders_long = offenders.melt(
                id_vars=["Interface"],
                value_vars=["Drops", "Overlimits"],
                var_name="Metric",
                value_name="Count",
            )
            st.plotly_chart(
                px.bar(
                    offenders_long,
                    x="Interface",
                    y="Count",
                    color="Metric",
                    barmode="group",
                    title="Top Interfaces by Drops/Overlimits",
                ),
                width="stretch",
            )

    if _has_col(queue, "Qdisc"):
        by_qdisc = queue.groupby(["Timestamp", "Qdisc"], as_index=False)["Backlog_Bytes"].sum(
            numeric_only=True
        )
        st.plotly_chart(
            px.area(
                by_qdisc,
                x="Timestamp",
                y="Backlog_Bytes",
                color="Qdisc",
                title="Backlog Split by Qdisc",
            ),
            width="stretch",
        )


def render_run_detail(artifacts: RunArtifacts) -> None:
    """Show the topology and the per-episode link, fault, route, and queue sections over a shared time window."""
    st.subheader("Run Analysis")
    st.plotly_chart(
        _graph_figure(artifacts.topology_nodes, artifacts.topology_links), width="stretch"
    )

    iface = artifacts.interface_stats.copy()
    ping = artifacts.ping_stats.copy()
    fault = artifacts.fault_log.copy()

    window_candidates: list[pd.Series] = [
        _to_timestamp_series(iface),
        _to_timestamp_series(ping),
        _to_timestamp_series(fault),
    ]
    window_candidates = [
        series.dropna() for series in window_candidates if not series.dropna().empty
    ]
    time_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if window_candidates:
        combined = pd.concat(window_candidates, ignore_index=True)
        time_window = (combined.min(), combined.max())
    fault_start, fault_end = _fault_window(artifacts)

    if not iface.empty:
        agg_iface = (
            iface.groupby("Timestamp", as_index=False)[
                [
                    "TX_KBPS",
                    "RX_KBPS",
                    "TX_PacketsPerSec",
                    "RX_PacketsPerSec",
                    "TX_DropsPerSec",
                    "RX_DropsPerSec",
                    "TX_ErrorsPerSec",
                    "RX_ErrorsPerSec",
                ]
            ]
            .sum(numeric_only=True)
            .sort_values("Timestamp")
        )
        throughput_fig = px.line(
            agg_iface,
            x="Timestamp",
            y=["TX_KBPS", "RX_KBPS"],
            title="Aggregate Throughput (KBPS)",
        )
        packet_rate_fig = px.line(
            agg_iface,
            x="Timestamp",
            y=["TX_PacketsPerSec", "RX_PacketsPerSec"],
            title="Aggregate Packet Rate",
        )
        drops_errors_fig = px.line(
            agg_iface,
            x="Timestamp",
            y=["TX_DropsPerSec", "RX_DropsPerSec", "TX_ErrorsPerSec", "RX_ErrorsPerSec"],
            title="Aggregate Drops/Errors Per Sec",
        )
        if time_window is not None:
            start_ts, end_ts = time_window
            for fig in [throughput_fig, packet_rate_fig, drops_errors_fig]:
                fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
        for fig in [throughput_fig, packet_rate_fig, drops_errors_fig]:
            _add_fault_markers(fig, fault_start, fault_end)
        st.plotly_chart(throughput_fig, width="stretch")
        st.plotly_chart(packet_rate_fig, width="stretch")
        st.plotly_chart(drops_errors_fig, width="stretch")
        _render_link_utilization(artifacts, time_window, fault_start, fault_end)

    if not ping.empty:
        ping["Pair"] = ping["Source"].astype(str) + "→" + ping["Destination"].astype(str)
        ping_rtt_fig = px.line(
            ping, x="Timestamp", y="AvgRTT", color="Pair", title="Ping AvgRTT by Pair"
        )
        ping_loss_fig = px.line(
            ping, x="Timestamp", y=["PacketLoss", "TimeoutFlag"], title="Ping Loss/Timeout"
        )
        if time_window is not None:
            start_ts, end_ts = time_window
            ping_rtt_fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
            ping_loss_fig.update_xaxes(range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()])
        _add_fault_markers(ping_rtt_fig, fault_start, fault_end)
        _add_fault_markers(ping_loss_fig, fault_start, fault_end)
        st.plotly_chart(ping_rtt_fig, width="stretch")
        st.plotly_chart(ping_loss_fig, width="stretch")

    _render_fault_timeline(artifacts, time_window=time_window)
    with st.expander("Routing Metrics", expanded=False):
        _render_route_stats(artifacts, time_window, fault_start, fault_end)
    with st.expander("Queue Metrics", expanded=False):
        _render_queue_stats(artifacts, time_window, fault_start, fault_end)

    st.subheader("Raw Data Preview")
    choice = st.selectbox(
        "Table",
        [
            "host_stats",
            "node_stats",
            "interface_stats",
            "ping_stats",
            "fault_log",
            "route_stats",
            "queue_stats",
            "telemetry_timing",
            "probe_timing",
            "fault_timing",
            "topology_links",
            "topology_nodes",
        ],
    )
    preview = {
        "host_stats": artifacts.host_stats,
        "node_stats": artifacts.node_stats,
        "interface_stats": artifacts.interface_stats,
        "ping_stats": artifacts.ping_stats,
        "fault_log": artifacts.fault_log,
        "route_stats": artifacts.route_stats,
        "queue_stats": artifacts.queue_stats,
        "telemetry_timing": artifacts.telemetry_timing,
        "probe_timing": artifacts.probe_timing,
        "fault_timing": artifacts.fault_timing,
        "topology_links": artifacts.topology_links,
        "topology_nodes": artifacts.topology_nodes,
    }[choice]
    st.dataframe(preview.head(250), width="stretch")


def main() -> None:
    """Validate the dataset root, show the dataset sections, and drill into one selected episode."""
    st.set_page_config(page_title="RIDGE Stage 1 Dataset Explorer", layout="wide")
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None
    if dataset_root is None or not is_stage1_dataset_root(dataset_root):
        st.error(f"Dataset root is not a valid Stage 1 artifact: {dataset_root}")
        st.stop()

    st.title("RIDGE Stage 1 Raw Dataset Explorer")
    st.caption(f"Dataset root: {dataset_root}")

    try:
        manifest = cached_manifest(str(dataset_root))
    except Exception as exc:
        st.error(f"Failed to load Stage 1 dataset: {exc}")
        st.stop()

    render_dataset_summary(manifest)
    render_manifest_extended(manifest)
    render_realism(dataset_root, manifest)

    st.subheader("Run Selector")
    default_statuses = (
        sorted(manifest["status"].astype(str).unique().tolist())
        if _has_col(manifest, "status")
        else []
    )
    status_filter = st.multiselect("Status", default_statuses, default=default_statuses)
    filtered = (
        manifest[manifest["status"].astype(str).isin(status_filter)].copy()
        if _has_col(manifest, "status")
        else manifest.copy()
    )
    if filtered.empty:
        st.warning("No runs match the current filter.")
        st.stop()

    run_id = st.selectbox("run_id", filtered["run_id"].tolist(), index=0)
    run_row = filtered[filtered["run_id"] == run_id].iloc[0]

    try:
        run_dir = resolve_run_dir(dataset_root, run_row)
        artifacts = cached_run_artifacts(str(run_dir))
    except Exception as exc:
        st.error(f"Failed to load run artifacts for run_id={run_id}: {exc}")
        st.stop()

    st.caption(f"Run directory: {artifacts.run_dir}")
    with st.expander("run_metadata.json", expanded=False):
        st.code(json.dumps(artifacts.run_metadata, indent=2), language="json")

    render_run_detail(artifacts)


if __name__ == "__main__":
    main()
