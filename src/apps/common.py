"""Helpers shared by the six Streamlit explorers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ridge.common.contracts import require_artifact_contract
from ridge.common.io import read_json_object


def load_csv(root: str | Path, filename: str) -> pd.DataFrame:
    """Read a CSV file under a root directory into a data frame."""
    return pd.read_csv(Path(root) / filename)


def require_artifact_payload(
    payload: object,
    artifact_type: str,
    *,
    source: str | Path,
) -> dict[str, Any]:
    """Validate the common artifact envelope used by explorer inputs."""
    if not isinstance(payload, dict):
        raise ValueError(f"{source} did not contain an artifact object")
    require_artifact_contract(payload, artifact_type=artifact_type)
    return payload


def json_artifact_matches(path: Path, artifact_type: str) -> bool:
    """Return whether a JSON file carries the expected artifact envelope."""
    try:
        payload = read_json_object(Path(path.parent) / path.name)
        require_artifact_payload(payload, artifact_type, source=path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def format_number(value: Any, *, digits: int = 3) -> str:
    """Format a value with fixed decimals, or n/a when it is not numeric."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def format_integer(value: Any) -> str:
    """Format a value as an integer, or n/a when it is not numeric."""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def format_percent(value: Any) -> str:
    """Format a fraction as a percentage with one decimal, or n/a when it is not numeric."""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def is_artifact_dataset_root(
    path: Path,
    *,
    runs_dirname: str,
    required_files: tuple[str, ...],
    index_filename: str,
    artifact_type: str,
) -> bool:
    """Shared dataset-root check: episode shard directory, required files, validated index artifact."""
    if not path.exists() or not path.is_dir():
        return False
    if not (path / runs_dirname).is_dir():
        return False
    return all((path / filename).exists() for filename in required_files) and json_artifact_matches(
        path / index_filename, artifact_type
    )


@st.cache_data(show_spinner=False)
def cached_json(root: str, filename: str) -> dict[str, Any]:
    """Read a JSON object under a root directory, cached per session."""
    return read_json_object(Path(root) / filename)


@st.cache_data(show_spinner=False)
def cached_csv(root: str, filename: str) -> pd.DataFrame:
    """Read a CSV file under a root directory, cached per session."""
    return load_csv(root, filename)


@st.cache_data(show_spinner=False)
def cached_residual_window_index(stage4_root: str) -> pd.DataFrame:
    """Read the Stage-4 window index with typed identifier, index, label, and timestamp columns, cached per session."""
    frame = load_csv(stage4_root, "residual_window_index.csv")
    if "run_id" in frame.columns:
        frame["run_id"] = frame["run_id"].astype(str)
    if "window_end_index" in frame.columns:
        frame["window_end_index"] = pd.to_numeric(
            frame["window_end_index"], errors="coerce"
        ).astype("Int64")
    if "history_len" in frame.columns:
        frame["history_len"] = pd.to_numeric(frame["history_len"], errors="coerce").astype("Int64")
    if "fault_present" in frame.columns:
        frame["fault_present"] = pd.to_numeric(frame["fault_present"], errors="coerce").fillna(0.0)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def split_run_counts(splits: dict[str, list[str]]) -> dict[str, int]:
    """Return the number of episodes in each of the train, validation, and test splits."""
    return {
        split: len([str(run_id) for run_id in splits.get(split, [])])
        for split in ("train", "val", "test")
    }


def split_run_counts_frame(splits: dict[str, list[str]]) -> pd.DataFrame:
    """Return the episode count per split as a two-column data frame."""
    return pd.DataFrame(
        [{"split": split, "run_count": count} for split, count in split_run_counts(splits).items()]
    )


def split_window_counts(window_index: pd.DataFrame, splits: dict[str, list[str]]) -> pd.DataFrame:
    """Count the windows of the index that fall in each split, with zero rows for empty splits."""
    split_map: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for run_id in splits.get(split_name, []):
            split_map[str(run_id)] = split_name

    if window_index.empty:
        return pd.DataFrame(
            [{"split": split_name, "window_count": 0} for split_name in ("train", "val", "test")]
        )

    enriched = window_index.copy()
    enriched["split"] = enriched["run_id"].astype(str).map(split_map).fillna("unassigned")
    counts = (
        enriched[enriched["split"].isin(("train", "val", "test"))]
        .groupby("split", as_index=False)
        .size()
        .rename(columns={"size": "window_count"})
    )
    return counts.set_index("split").reindex(["train", "val", "test"], fill_value=0).reset_index()


def topology_counts(topology: dict[str, Any]) -> dict[str, int]:
    """Return the numbers of nodes, edges, probes, and candidates in a topology."""
    return {
        "node_count": len(topology.get("node_ids", [])),
        "edge_count": len(topology.get("edge_ids", [])),
        "probe_count": len(topology.get("probe_ids", [])),
        "candidate_count": len(topology.get("candidate_ids", [])),
    }


def window_count_summary(by_run: pd.DataFrame) -> dict[str, int]:
    """Return the minimum, median, 90th percentile, and maximum of the per-episode window counts."""
    if by_run.empty:
        return {"min": 0, "median": 0, "p90": 0, "max": 0}
    series = pd.to_numeric(by_run["window_count"], errors="coerce").fillna(0)
    return {
        "min": int(series.min()),
        "median": int(series.median()),
        "p90": int(series.quantile(0.9)),
        "max": int(series.max()),
    }


def render_metric_notes(rows: list[dict[str, str]]) -> None:
    """Show a table of metric explanations without an index column."""
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_artifact_inspector_entries(artifacts: list[tuple[str, dict[str, Any]]]) -> None:
    """Show each artifact payload as formatted JSON inside a collapsed expander."""
    for label, payload in artifacts:
        with st.expander(label, expanded=False):
            st.code(json.dumps(payload, indent=2), language="json")
