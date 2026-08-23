"""Quantify how Stage 1 faults manifest in telemetry for thesis reporting.

Example:
    python src/analysis/stage1_fault_effects.py \
      --dataset-root /data/stage1-dataset \
      --workers 8
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from analysis.common import (
    configure_logging,
    format_float,
    parse_optional_bool,
    print_section,
    progress_interval,
    safe_ratio,
)
from ridge.common.io import write_json
from ridge.io.stage1_dataset import (
    load_manifest,
    load_run_artifacts,
    resolve_run_dir,
    validate_dataset_root,
)

DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_BOOTSTRAP_SAMPLES = 400
LOGGER = logging.getLogger("stage1_fault_effects")


@dataclass(frozen=True)
class RunFaultEffect:
    """Per-episode contrast of probe, queue, interface, and routing telemetry before and during the fault window."""
    
    run_id: int
    category: str
    fault_target: str
    root_cause_kind: str
    baseline_health_pass: bool | None
    has_fault_log_rows: bool
    fault_window_valid: bool
    rtt_baseline_ms: float | None
    rtt_fault_ms: float | None
    rtt_delta_ms: float | None
    rtt_ratio: float | None
    rtt_effect_z: float | None
    loss_baseline_pct: float | None
    loss_fault_pct: float | None
    loss_delta_pct: float | None
    timeout_baseline_ratio: float | None
    timeout_fault_ratio: float | None
    timeout_delta_ratio: float | None
    queue_backlog_baseline_bytes: float | None
    queue_backlog_fault_bytes: float | None
    queue_backlog_delta_bytes: float | None
    queue_backlog_ratio: float | None
    interface_drop_baseline_rate: float | None
    interface_drop_fault_rate: float | None
    interface_drop_delta_rate: float | None
    route_count_baseline: float | None
    route_count_fault: float | None
    route_count_abs_delta: float | None
    ospf_route_baseline: float | None
    ospf_route_fault: float | None
    ospf_route_abs_delta: float | None
    probe_observable: bool | None
    queue_observable: bool | None
    route_observable: bool | None
    any_observable: bool | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse the dataset root, worker count, output path, cohort filters, bootstrap samples, and log level."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="Path to the Stage 1 dataset root."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Worker threads for per-run analysis (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <dataset-root>/analysis_stage1_fault_effects.json.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Bootstrap resamples per aggregate statistic (default: {DEFAULT_BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--require-baseline-pass",
        action="store_true",
        help="Restrict the analysis to runs whose documented baseline-health gate passed.",
    )
    parser.add_argument(
        "--allow-missing-fault-log",
        action="store_true",
        help="Include faulty runs even when fault_log.csv is empty.",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help=f"Progress logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def _series_values(
    frame: pd.DataFrame,
    column: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Return the numeric values of a column within an optional half-open time window."""
    if frame.empty or "Timestamp" not in frame.columns or column not in frame.columns:
        return pd.Series(dtype=float)
    timestamps = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = timestamps.notna()
    if start is not None:
        mask &= timestamps >= start
    if end is not None:
        mask &= timestamps < end
    return values[mask].dropna()


def _mean(values: pd.Series) -> float | None:
    """Return the mean of a series, or None when it is empty."""
    if values.empty:
        return None
    return float(values.mean())


def _std(values: pd.Series) -> float | None:
    """Return the sample standard deviation of a series, or None below two values."""
    if len(values) < 2:
        return None
    value = float(values.std(ddof=1))
    return None if math.isnan(value) else value


def _max(values: pd.Series) -> float | None:
    """Return the maximum of a series, or None when it is empty."""
    if values.empty:
        return None
    return float(values.max())


def _effect_z(
    baseline: pd.Series, baseline_mean: float | None, fault_mean: float | None
) -> float | None:
    """Return the fault-window mean shift in units of the baseline standard deviation, or None."""
    baseline_std = _std(baseline)
    if baseline_mean is None or fault_mean is None or baseline_std is None or baseline_std <= 0:
        return None
    return float((fault_mean - baseline_mean) / baseline_std)


def _interface_drop_series(
    interface_stats: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Return the larger per-interface drop rate within an optional half-open time window."""
    if interface_stats.empty or "Timestamp" not in interface_stats.columns:
        return pd.Series(dtype=float)
    timestamps = pd.to_datetime(interface_stats["Timestamp"], errors="coerce", utc=True)
    tx = (
        pd.to_numeric(interface_stats.get("TX_DropsPerSec"), errors="coerce")
        if "TX_DropsPerSec" in interface_stats.columns
        else pd.Series(0.0, index=interface_stats.index)
    )
    rx = (
        pd.to_numeric(interface_stats.get("RX_DropsPerSec"), errors="coerce")
        if "RX_DropsPerSec" in interface_stats.columns
        else pd.Series(0.0, index=interface_stats.index)
    )
    values = pd.concat([tx, rx], axis=1).max(axis=1)
    mask = timestamps.notna()
    if start is not None:
        mask &= timestamps >= start
    if end is not None:
        mask &= timestamps < end
    return values[mask].dropna()


def analyze_fault_run(run_row: pd.Series, dataset_root: Path) -> RunFaultEffect:
    """Contrast one faulty episode's telemetry before and during its fault window, returning None fields where not computable."""
    run_id = int(run_row["run_id"])
    category = str(run_row.get("fault_category", "") or "")
    fault_target = str(run_row.get("fault_target", "") or "")
    root_cause_kind = str(run_row.get("root_cause_kind", "") or "")
    baseline_health_pass = parse_optional_bool(run_row.get("baseline_health_pass"))

    try:
        run_dir = resolve_run_dir(dataset_root, run_row)
        artifacts = load_run_artifacts(run_dir)
    except Exception as exc:
        return RunFaultEffect(
            run_id=run_id,
            category=category,
            fault_target=fault_target,
            root_cause_kind=root_cause_kind,
            baseline_health_pass=baseline_health_pass,
            has_fault_log_rows=False,
            fault_window_valid=False,
            rtt_baseline_ms=None,
            rtt_fault_ms=None,
            rtt_delta_ms=None,
            rtt_ratio=None,
            rtt_effect_z=None,
            loss_baseline_pct=None,
            loss_fault_pct=None,
            loss_delta_pct=None,
            timeout_baseline_ratio=None,
            timeout_fault_ratio=None,
            timeout_delta_ratio=None,
            queue_backlog_baseline_bytes=None,
            queue_backlog_fault_bytes=None,
            queue_backlog_delta_bytes=None,
            queue_backlog_ratio=None,
            interface_drop_baseline_rate=None,
            interface_drop_fault_rate=None,
            interface_drop_delta_rate=None,
            route_count_baseline=None,
            route_count_fault=None,
            route_count_abs_delta=None,
            ospf_route_baseline=None,
            ospf_route_fault=None,
            ospf_route_abs_delta=None,
            probe_observable=None,
            queue_observable=None,
            route_observable=None,
            any_observable=None,
            error=str(exc),
        )

    fault_start = pd.to_datetime(run_row.get("fault_start_ts"), errors="coerce", utc=True)
    fault_end = pd.to_datetime(run_row.get("fault_end_ts"), errors="coerce", utc=True)
    fault_window_valid = bool(
        pd.notna(fault_start) and pd.notna(fault_end) and fault_end >= fault_start
    )
    if not fault_window_valid:
        return RunFaultEffect(
            run_id=run_id,
            category=category,
            fault_target=fault_target,
            root_cause_kind=root_cause_kind,
            baseline_health_pass=baseline_health_pass,
            has_fault_log_rows=not artifacts.fault_log.empty,
            fault_window_valid=False,
            rtt_baseline_ms=None,
            rtt_fault_ms=None,
            rtt_delta_ms=None,
            rtt_ratio=None,
            rtt_effect_z=None,
            loss_baseline_pct=None,
            loss_fault_pct=None,
            loss_delta_pct=None,
            timeout_baseline_ratio=None,
            timeout_fault_ratio=None,
            timeout_delta_ratio=None,
            queue_backlog_baseline_bytes=None,
            queue_backlog_fault_bytes=None,
            queue_backlog_delta_bytes=None,
            queue_backlog_ratio=None,
            interface_drop_baseline_rate=None,
            interface_drop_fault_rate=None,
            interface_drop_delta_rate=None,
            route_count_baseline=None,
            route_count_fault=None,
            route_count_abs_delta=None,
            ospf_route_baseline=None,
            ospf_route_fault=None,
            ospf_route_abs_delta=None,
            probe_observable=None,
            queue_observable=None,
            route_observable=None,
            any_observable=None,
            error=None,
        )

    ping = artifacts.ping_stats
    baseline_rtt = _series_values(ping, "AvgRTT", end=fault_start)
    fault_rtt = _series_values(ping, "AvgRTT", start=fault_start, end=fault_end)
    baseline_loss = _series_values(ping, "PacketLoss", end=fault_start)
    fault_loss = _series_values(ping, "PacketLoss", start=fault_start, end=fault_end)
    baseline_timeout = _series_values(ping, "TimeoutFlag", end=fault_start)
    fault_timeout = _series_values(ping, "TimeoutFlag", start=fault_start, end=fault_end)

    rtt_baseline_ms = _mean(baseline_rtt)
    rtt_fault_ms = _mean(fault_rtt)
    loss_baseline_pct = _mean(baseline_loss)
    loss_fault_pct = _mean(fault_loss)
    timeout_baseline_ratio = _mean(baseline_timeout)
    timeout_fault_ratio = _mean(fault_timeout)

    queue = artifacts.queue_stats
    queue_backlog_baseline = _series_values(queue, "Backlog_Bytes", end=fault_start)
    queue_backlog_fault = _series_values(queue, "Backlog_Bytes", start=fault_start, end=fault_end)
    queue_backlog_baseline_bytes = _max(queue_backlog_baseline)
    queue_backlog_fault_bytes = _max(queue_backlog_fault)

    interface_baseline = _interface_drop_series(artifacts.interface_stats, end=fault_start)
    interface_fault = _interface_drop_series(
        artifacts.interface_stats, start=fault_start, end=fault_end
    )
    interface_drop_baseline_rate = _mean(interface_baseline)
    interface_drop_fault_rate = _mean(interface_fault)

    route = artifacts.route_stats
    route_count_baseline = _mean(_series_values(route, "RouteCount", end=fault_start))
    route_count_fault = _mean(_series_values(route, "RouteCount", start=fault_start, end=fault_end))
    ospf_route_baseline = _mean(_series_values(route, "OspfRouteCount", end=fault_start))
    ospf_route_fault = _mean(
        _series_values(route, "OspfRouteCount", start=fault_start, end=fault_end)
    )

    rtt_delta_ms = (
        None
        if rtt_baseline_ms is None or rtt_fault_ms is None
        else float(rtt_fault_ms - rtt_baseline_ms)
    )
    loss_delta_pct = (
        None
        if loss_baseline_pct is None or loss_fault_pct is None
        else float(loss_fault_pct - loss_baseline_pct)
    )
    timeout_delta_ratio = (
        None
        if timeout_baseline_ratio is None or timeout_fault_ratio is None
        else float(timeout_fault_ratio - timeout_baseline_ratio)
    )
    queue_backlog_delta_bytes = (
        None
        if queue_backlog_baseline_bytes is None or queue_backlog_fault_bytes is None
        else float(queue_backlog_fault_bytes - queue_backlog_baseline_bytes)
    )
    interface_drop_delta_rate = (
        None
        if interface_drop_baseline_rate is None or interface_drop_fault_rate is None
        else float(interface_drop_fault_rate - interface_drop_baseline_rate)
    )
    route_count_abs_delta = (
        None
        if route_count_baseline is None or route_count_fault is None
        else float(abs(route_count_fault - route_count_baseline))
    )
    ospf_route_abs_delta = (
        None
        if ospf_route_baseline is None or ospf_route_fault is None
        else float(abs(ospf_route_fault - ospf_route_baseline))
    )

    rtt_ratio = safe_ratio(rtt_fault_ms, rtt_baseline_ms)
    queue_backlog_ratio = safe_ratio(queue_backlog_fault_bytes, queue_backlog_baseline_bytes)
    rtt_effect_z = _effect_z(baseline_rtt, rtt_baseline_ms, rtt_fault_ms)

    # Heuristics are intentionally simple. Just to check whether the injected fault left a visible symptom in each family
    probe_observable = None
    if any(value is not None for value in (rtt_ratio, loss_delta_pct, timeout_fault_ratio)):
        probe_observable = bool(
            (rtt_ratio is not None and rtt_ratio >= 1.25)
            or (loss_delta_pct is not None and loss_delta_pct >= 1.0)
            or (timeout_delta_ratio is not None and timeout_delta_ratio > 0.0)
        )
    queue_observable = None
    if queue_backlog_baseline_bytes is not None and queue_backlog_fault_bytes is not None:
        queue_observable = bool(
            (queue_backlog_baseline_bytes == 0 and queue_backlog_fault_bytes > 0)
            or (queue_backlog_ratio is not None and queue_backlog_ratio >= 1.25)
        )
    route_observable = (
        None
        if route_count_abs_delta is None and ospf_route_abs_delta is None
        else bool((route_count_abs_delta or 0.0) > 0.0 or (ospf_route_abs_delta or 0.0) > 0.0)
    )
    evidence = [probe_observable, queue_observable, route_observable]
    any_observable = (
        None
        if not any(value is not None for value in evidence)
        else bool(any(value is True for value in evidence))
    )

    return RunFaultEffect(
        run_id=run_id,
        category=category,
        fault_target=fault_target,
        root_cause_kind=root_cause_kind,
        baseline_health_pass=baseline_health_pass,
        has_fault_log_rows=not artifacts.fault_log.empty,
        fault_window_valid=True,
        rtt_baseline_ms=rtt_baseline_ms,
        rtt_fault_ms=rtt_fault_ms,
        rtt_delta_ms=rtt_delta_ms,
        rtt_ratio=rtt_ratio,
        rtt_effect_z=rtt_effect_z,
        loss_baseline_pct=loss_baseline_pct,
        loss_fault_pct=loss_fault_pct,
        loss_delta_pct=loss_delta_pct,
        timeout_baseline_ratio=timeout_baseline_ratio,
        timeout_fault_ratio=timeout_fault_ratio,
        timeout_delta_ratio=timeout_delta_ratio,
        queue_backlog_baseline_bytes=queue_backlog_baseline_bytes,
        queue_backlog_fault_bytes=queue_backlog_fault_bytes,
        queue_backlog_delta_bytes=queue_backlog_delta_bytes,
        queue_backlog_ratio=queue_backlog_ratio,
        interface_drop_baseline_rate=interface_drop_baseline_rate,
        interface_drop_fault_rate=interface_drop_fault_rate,
        interface_drop_delta_rate=interface_drop_delta_rate,
        route_count_baseline=route_count_baseline,
        route_count_fault=route_count_fault,
        route_count_abs_delta=route_count_abs_delta,
        ospf_route_baseline=ospf_route_baseline,
        ospf_route_fault=ospf_route_fault,
        ospf_route_abs_delta=ospf_route_abs_delta,
        probe_observable=probe_observable,
        queue_observable=queue_observable,
        route_observable=route_observable,
        any_observable=any_observable,
        error=None,
    )


def _bootstrap_ci(
    values: list[float], stat_fn: Callable[[list[float]], float], samples: int, seed: int
) -> dict[str, float] | None:
    """Return a seeded bootstrap 95 percent confidence interval of a statistic over the values."""
    if not values:
        return None
    rng = random.Random(seed)
    if len(values) == 1:
        value = stat_fn(values)
        return {"low": value, "high": value}
    stats: list[float] = []
    for _ in range(max(1, samples)):
        resample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        stats.append(stat_fn(resample))
    stats.sort()
    low_index = max(0, int(0.025 * (len(stats) - 1)))
    high_index = min(len(stats) - 1, int(0.975 * (len(stats) - 1)))
    return {"low": float(stats[low_index]), "high": float(stats[high_index])}


def _metric_summary(
    records: list[RunFaultEffect],
    field: str,
    *,
    positive_if: Callable[[float], bool],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize a per-episode metric by median, mean, positive rate, and their bootstrap intervals."""
    values = [getattr(record, field) for record in records if getattr(record, field) is not None]
    values = [float(value) for value in values]
    if not values:
        return {"n": 0}
    median = float(pd.Series(values).median())
    mean = float(pd.Series(values).mean())
    positive_rate = float(sum(1 for value in values if positive_if(value)) / len(values))
    return {
        "n": len(values),
        "median": median,
        "mean": mean,
        "positive_rate": positive_rate,
        "median_ci_95": _bootstrap_ci(
            values, lambda sample: float(pd.Series(sample).median()), bootstrap_samples, seed
        ),
        "positive_rate_ci_95": _bootstrap_ci(
            [1.0 if positive_if(value) else 0.0 for value in values],
            lambda sample: float(sum(sample) / len(sample)),
            bootstrap_samples,
            seed + 1000,
        ),
    }


def _observability_summary(
    records: list[RunFaultEffect], field: str, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    """Summarize a per-episode observability flag by its rate and bootstrap interval."""
    values = [getattr(record, field) for record in records if getattr(record, field) is not None]
    numeric = [1.0 if value else 0.0 for value in values]
    if not numeric:
        return {"n": 0}
    rate = float(sum(numeric) / len(numeric))
    return {
        "n": len(numeric),
        "rate": rate,
        "rate_ci_95": _bootstrap_ci(
            numeric, lambda sample: float(sum(sample) / len(sample)), bootstrap_samples, seed
        ),
    }


def _group_summary(
    records: list[RunFaultEffect], bootstrap_samples: int, seed_base: int
) -> dict[str, Any]:
    """Summarize every effect metric and observability flag over a cohort of episodes with seeded bootstraps."""
    return {
        "run_count": len(records),
        "baseline_pass_rate": None
        if not records
        else float(
            sum(1 for record in records if record.baseline_health_pass is True) / len(records)
        ),
        "rtt_ratio": _metric_summary(
            records,
            "rtt_ratio",
            positive_if=lambda value: value > 1.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 1,
        ),
        "rtt_delta_ms": _metric_summary(
            records,
            "rtt_delta_ms",
            positive_if=lambda value: value > 0.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 2,
        ),
        "rtt_effect_z": _metric_summary(
            records,
            "rtt_effect_z",
            positive_if=lambda value: value > 0.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 3,
        ),
        "loss_delta_pct": _metric_summary(
            records,
            "loss_delta_pct",
            positive_if=lambda value: value > 0.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 4,
        ),
        "timeout_delta_ratio": _metric_summary(
            records,
            "timeout_delta_ratio",
            positive_if=lambda value: value > 0.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 5,
        ),
        "queue_backlog_ratio": _metric_summary(
            records,
            "queue_backlog_ratio",
            positive_if=lambda value: value > 1.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 6,
        ),
        "interface_drop_delta_rate": _metric_summary(
            records,
            "interface_drop_delta_rate",
            positive_if=lambda value: value > 0.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 7,
        ),
        "route_count_abs_delta": _metric_summary(
            records,
            "route_count_abs_delta",
            positive_if=lambda value: value > 0.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 8,
        ),
        "ospf_route_abs_delta": _metric_summary(
            records,
            "ospf_route_abs_delta",
            positive_if=lambda value: value > 0.0,
            bootstrap_samples=bootstrap_samples,
            seed=seed_base + 9,
        ),
        "probe_observable": _observability_summary(
            records, "probe_observable", bootstrap_samples, seed_base + 10
        ),
        "queue_observable": _observability_summary(
            records, "queue_observable", bootstrap_samples, seed_base + 11
        ),
        "route_observable": _observability_summary(
            records, "route_observable", bootstrap_samples, seed_base + 12
        ),
        "any_observable": _observability_summary(
            records, "any_observable", bootstrap_samples, seed_base + 13
        ),
    }


def _format_metric(summary: dict[str, Any], unit: str = "") -> str:
    """Format a metric summary as its median with the 95 percent confidence interval, or n/a."""
    if summary.get("n", 0) == 0:
        return "n/a"
    median = format_float(summary.get("median"))
    ci = summary.get("median_ci_95")
    if ci is None:
        return f"{median}{unit}"
    return f"{median}{unit} (95% CI {format_float(ci['low'])}-{format_float(ci['high'])}{unit})"


def main() -> None:
    """Analyze every faulty episode in a thread pool, aggregate the effects overall and by category and target, and write the report."""
    args = parse_args()
    configure_logging(args.log_level)
    LOGGER.info("Starting Stage 1 fault-effects analysis")
    LOGGER.info("Dataset root: %s", args.dataset_root)
    validate_dataset_root(args.dataset_root)
    output_json = args.output_json or (args.dataset_root / "analysis_stage1_fault_effects.json")

    LOGGER.info("Loading manifest")
    manifest = load_manifest(args.dataset_root)
    if args.max_runs is not None:
        manifest = manifest.head(args.max_runs).copy()
        LOGGER.info("Applying max-runs cap: %d", args.max_runs)

    faulty_ok = manifest.loc[
        (manifest["status"].astype(str) == "ok")
        & (manifest["fault_category"].fillna("").astype(str) != "")
        & (manifest["fault_category"].astype(str) != "none")
    ].copy()
    LOGGER.info("Candidate faulty successful runs: %d", len(faulty_ok))

    rows = [row for _, row in faulty_ok.iterrows()]
    records: list[RunFaultEffect] = []
    progress_every = progress_interval(len(rows))
    worker_count = max(1, int(args.workers))
    LOGGER.info("Analyzing %d runs with %d worker(s)", len(rows), worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(analyze_fault_run, row, args.dataset_root): int(row["run_id"])
            for row in rows
        }
        completed = 0
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                LOGGER.exception("Unexpected failure while analyzing run_id=%s: %s", run_id, exc)
                raise
            records.append(record)
            completed += 1
            if record.error is not None:
                LOGGER.warning("Run %06d analyzed with error: %s", record.run_id, record.error)
            if completed % progress_every == 0 or completed == len(rows):
                LOGGER.info("Completed %d/%d runs", completed, len(rows))
    records.sort(key=lambda record: record.run_id)

    analyzed = [record for record in records if record.error is None and record.fault_window_valid]
    if not args.allow_missing_fault_log:
        analyzed = [record for record in analyzed if record.has_fault_log_rows]
    if args.require_baseline_pass:
        analyzed = [record for record in analyzed if record.baseline_health_pass is True]
    LOGGER.info("Runs retained for effect aggregation: %d", len(analyzed))

    overall = _group_summary(analyzed, args.bootstrap_samples, 100)
    by_category = {
        category: _group_summary(
            [record for record in analyzed if record.category == category],
            args.bootstrap_samples,
            1000 + index * 100,
        )
        for index, category in enumerate(sorted({record.category for record in analyzed}))
    }
    by_target = {
        target: _group_summary(
            [record for record in analyzed if record.fault_target == target],
            args.bootstrap_samples,
            5000 + index * 100,
        )
        for index, target in enumerate(sorted({record.fault_target for record in analyzed}))
    }

    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "workers": int(args.workers),
            "max_runs": args.max_runs,
            "bootstrap_samples": int(args.bootstrap_samples),
            "require_baseline_pass": bool(args.require_baseline_pass),
            "allow_missing_fault_log": bool(args.allow_missing_fault_log),
            "output_json": str(output_json),
        },
        "cohort_summary": {
            "candidate_faulty_ok_runs": int(len(faulty_ok)),
            "analyzed_runs": int(len(analyzed)),
            "excluded_missing_fault_log": int(
                sum(
                    1
                    for record in records
                    if record.error is None
                    and record.fault_window_valid
                    and not record.has_fault_log_rows
                )
            ),
            "excluded_baseline_failures": int(
                sum(
                    1
                    for record in records
                    if record.error is None
                    and record.fault_window_valid
                    and record.baseline_health_pass is not True
                )
            ),
            "run_errors": int(sum(1 for record in records if record.error is not None)),
        },
        "overall": overall,
        "by_category": by_category,
        "by_target": by_target,
        "per_run": [asdict(record) for record in records],
    }
    LOGGER.info("Writing JSON report to %s", output_json)
    write_json(output_json, report)

    print_section(
        "Cohort",
        [
            f"candidate faulty ok runs={len(faulty_ok)}, analyzed={len(analyzed)}",
            f"excluded missing fault logs={report['cohort_summary']['excluded_missing_fault_log']}, excluded baseline failures={report['cohort_summary']['excluded_baseline_failures']}",
            f"run errors={report['cohort_summary']['run_errors']}",
        ],
    )
    print_section(
        "Probe Effects",
        [
            f"RTT ratio={_format_metric(overall['rtt_ratio'], 'x')}",
            f"RTT delta={_format_metric(overall['rtt_delta_ms'], ' ms')}",
            f"loss delta={_format_metric(overall['loss_delta_pct'], ' pct-pts')}",
            f"timeout delta={_format_metric(overall['timeout_delta_ratio'])}",
        ],
    )
    print_section(
        "System Effects",
        [
            f"queue backlog ratio={_format_metric(overall['queue_backlog_ratio'], 'x')}",
            f"interface drop delta={_format_metric(overall['interface_drop_delta_rate'])}",
            f"route abs delta={_format_metric(overall['route_count_abs_delta'])}",
            f"OSPF route abs delta={_format_metric(overall['ospf_route_abs_delta'])}",
        ],
    )
    print_section(
        "Observability",
        [
            f"probe observable rate={format_float(overall['probe_observable'].get('rate') if overall['probe_observable'].get('n', 0) else None)}",
            f"queue observable rate={format_float(overall['queue_observable'].get('rate') if overall['queue_observable'].get('n', 0) else None)}",
            f"route observable rate={format_float(overall['route_observable'].get('rate') if overall['route_observable'].get('n', 0) else None)}",
            f"any-signal observable rate={format_float(overall['any_observable'].get('rate') if overall['any_observable'].get('n', 0) else None)}",
        ],
    )
    category_lines = []
    for category, summary in by_category.items():
        category_lines.append(
            f"{category}: n={summary['run_count']}, RTT ratio={_format_metric(summary['rtt_ratio'], 'x')}, "
            f"loss delta={_format_metric(summary['loss_delta_pct'], ' pct-pts')}, any observable={format_float(summary['any_observable'].get('rate') if summary['any_observable'].get('n', 0) else None)}"
        )
    print_section("By Category", category_lines or ["n/a"])
    print_section("Report", [f"json report={output_json}"])
    LOGGER.info("Stage 1 fault-effects analysis finished")


if __name__ == "__main__":
    main()
