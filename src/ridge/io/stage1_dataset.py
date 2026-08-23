"""Readers for the manifest, provenance, and per-episode artifacts of a generated Stage-1 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ridge.common.contracts import STAGE1_ARTIFACT_TYPE, require_artifact_contract
from ridge.common.io import read_json_object


@dataclass(frozen=True)
class RunArtifacts:
    """Every CSV table and the metadata of one episode directory, loaded as data frames."""
    
    run_dir: Path
    run_metadata: dict[str, Any]
    topology_nodes: pd.DataFrame
    topology_links: pd.DataFrame
    host_stats: pd.DataFrame
    node_stats: pd.DataFrame
    interface_stats: pd.DataFrame
    ping_stats: pd.DataFrame
    fault_log: pd.DataFrame
    route_stats: pd.DataFrame
    queue_stats: pd.DataFrame
    neighbor_stats: pd.DataFrame
    telemetry_timing: pd.DataFrame
    probe_timing: pd.DataFrame
    fault_timing: pd.DataFrame


_MANIFEST_REQUIRED_COLUMNS = {
    "run_id",
    "fault_category",
    "root_cause_kind",
    "fault_target",
    "status",
    "returncode",
    "elapsed_sec",
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file the pipeline guarantees, raising FileNotFoundError when it is absent."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def validate_dataset_root(dataset_root: Path) -> None:
    """Raise FileNotFoundError unless the dataset root is a directory holding a manifest."""
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist or is not a directory: {dataset_root}"
        )
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing required manifest file: {manifest_path}")


def load_manifest(dataset_root: Path) -> pd.DataFrame:
    """Load manifest.csv with typed identifier, timestamp, and numeric columns, sorted by episode."""
    validate_dataset_root(dataset_root)
    manifest = pd.read_csv(dataset_root / "manifest.csv")
    missing = _MANIFEST_REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"manifest.csv is missing required columns: {missing_str}")

    manifest["run_id"] = pd.to_numeric(manifest["run_id"], errors="raise").astype(int)

    for col in ("fault_start_ts", "fault_end_ts", "started_at", "finished_at"):
        if col in manifest.columns:
            manifest[col] = pd.to_datetime(manifest[col], errors="coerce", utc=True)

    if "elapsed_sec" in manifest.columns:
        manifest["elapsed_sec"] = pd.to_numeric(manifest["elapsed_sec"], errors="coerce")
    if "returncode" in manifest.columns:
        manifest["returncode"] = pd.to_numeric(manifest["returncode"], errors="coerce")

    return manifest.sort_values("run_id").reset_index(drop=True)


def load_generation_provenance(
    dataset_root: Path, *, require_contract: bool = False
) -> dict[str, Any]:
    """Load Stage-1 provenance, optionally enforcing the artifact contract."""
    payload = read_json_object(dataset_root / "generation_provenance.json", missing_ok=True)
    if require_contract:
        require_artifact_contract(payload, artifact_type=STAGE1_ARTIFACT_TYPE)
    return payload


def resolve_run_dir(dataset_root: Path, run_row: pd.Series) -> Path:
    """Resolve the episode directory of one manifest row."""
    return resolve_run_dir_values(
        dataset_root,
        run_id=int(run_row["run_id"]),
        log_dir=str(run_row.get("log_dir", "") or ""),
    )


def resolve_run_dir_values(dataset_root: Path, *, run_id: int, log_dir: str = "") -> Path:
    """Resolve an episode directory from portable manifest values."""
    log_dir = log_dir.strip()
    candidates: list[Path] = []
    if log_dir:
        log_path = Path(log_dir)
        candidates.append(log_path if log_path.is_absolute() else dataset_root / log_path)
    candidates.append(dataset_root / f"run_{run_id:06d}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Unable to resolve run directory for run_id={run_id}; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _parse_timestamp_column(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Parse a timestamp column as UTC when present, coercing unparsable values to NaT."""
    if timestamp_column in frame.columns:
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    return frame


def _optional_csv(run_dir: Path, filename: str, *, timestamp_column: str) -> pd.DataFrame:
    """Read an optional episode CSV, returning an empty frame when the file is absent."""
    path = run_dir / filename
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return _parse_timestamp_column(frame, timestamp_column)


def load_run_artifacts(run_dir: Path) -> RunArtifacts:
    """Load every telemetry, fault, and timing table of an episode directory with its metadata."""
    topology_nodes = _read_csv(run_dir / "topology_nodes.csv")
    topology_links = _read_csv(run_dir / "topology_links.csv")
    host_stats = _optional_csv(run_dir, "host_stats.csv", timestamp_column="Timestamp")
    node_stats = _optional_csv(run_dir, "node_stats.csv", timestamp_column="Timestamp")
    interface_stats = _parse_timestamp_column(
        _read_csv(run_dir / "interface_stats.csv"), "Timestamp"
    )
    ping_stats = _parse_timestamp_column(_read_csv(run_dir / "ping_stats.csv"), "Timestamp")
    fault_log = _optional_csv(run_dir, "fault_log.csv", timestamp_column="Timestamp")
    route_stats = _optional_csv(run_dir, "route_stats.csv", timestamp_column="Timestamp")
    queue_stats = _optional_csv(run_dir, "queue_stats.csv", timestamp_column="Timestamp")
    neighbor_stats = _optional_csv(run_dir, "neighbor_stats.csv", timestamp_column="Timestamp")
    telemetry_timing = _optional_csv(
        run_dir, "telemetry_timing.csv", timestamp_column="ActualStartTimestamp"
    )
    probe_timing = _optional_csv(run_dir, "probe_timing.csv", timestamp_column="Timestamp")
    fault_timing = _optional_csv(run_dir, "fault_timing.csv", timestamp_column="Timestamp")

    run_metadata = read_json_object(run_dir / "run_metadata.json")

    return RunArtifacts(
        run_dir=run_dir,
        run_metadata=run_metadata,
        topology_nodes=topology_nodes,
        topology_links=topology_links,
        host_stats=host_stats,
        node_stats=node_stats,
        interface_stats=interface_stats,
        ping_stats=ping_stats,
        fault_log=fault_log,
        route_stats=route_stats,
        queue_stats=queue_stats,
        neighbor_stats=neighbor_stats,
        telemetry_timing=telemetry_timing,
        probe_timing=probe_timing,
        fault_timing=fault_timing,
    )


def load_realism_artifacts(dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the realism summary and the realism checks of a dataset."""
    return read_json_object(dataset_root / "realism_summary.json"), read_json_object(
        dataset_root / "realism_checks.json"
    )
