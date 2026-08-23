"""Summarize whether Stage 1 faults are observable enough for RCA.

The tool writes CSV/JSON tables plus self-contained Plotly HTML figures. 
It reuses the public fault-window analysis from ``stage1_fault_effects`` so window boundaries and observability rules cannot drift between reports.

Example:
    python src/analysis/stage1_rca_readiness.py \
      --dataset-root /data/stage1-dataset \
      --workers 8 \
      --require-baseline-pass
"""

from __future__ import annotations

import argparse
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.stage1_fault_effects import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_WORKERS,
    analyze_fault_run,
    configure_logging,
    parse_optional_bool,
    progress_interval,
)
from analysis.stage1_fault_effects import (
    LOGGER as FAULT_EFFECT_LOGGER,
)
from ridge.common.io import write_json
from ridge.io.stage1_dataset import load_manifest, validate_dataset_root

LOGGER = logging.getLogger("stage1_rca_readiness")
RUN_SUMMARY_COLUMNS = [
    "run_id",
    "fault_category",
    "fault_target",
    "root_cause_kind",
    "baseline_health_pass",
    "rtt_ratio",
    "rtt_delta_ms",
    "loss_delta_pct",
    "timeout_delta_ratio",
    "route_count_abs_delta",
    "ospf_route_abs_delta",
    "queue_backlog_delta_bytes",
    "interface_drop_delta_rate",
    "probe_observable",
    "queue_observable",
    "route_observable",
    "any_observable",
]
MAIN_EFFECT_METRICS = [
    "rtt_ratio",
    "timeout_delta_ratio",
    "route_count_abs_delta",
    "queue_backlog_delta_bytes",
]
OBSERVABILITY_FLAGS = [
    "probe_observable",
    "queue_observable",
    "route_observable",
    "any_observable",
]


def parse_args() -> argparse.Namespace:
    """Parse the dataset root, output directory, worker count, cohort filter, episode cap, and log level."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/analysis_rca_readiness.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--require-baseline-pass",
        action="store_true",
        help="Analyze only runs whose baseline-health gate passed.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def _safe_label(value: object) -> str:
    """Return a value as a stripped label, or unknown when it is blank."""
    text = "" if value is None else str(value).strip()
    return text or "unknown"


def _rate(series: pd.Series) -> float:
    """Return the mean of a numeric or boolean series, or NaN when it has no values."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _select_faulty_runs(manifest: pd.DataFrame, require_baseline_pass: bool) -> pd.DataFrame:
    """Return the successful faulty episodes of the manifest, optionally only those that passed the validity check."""
    categories = manifest["fault_category"].fillna("").astype(str)
    selected = manifest.loc[
        (manifest["status"].astype(str) == "ok") & ~categories.isin(("", "none"))
    ].copy()
    if require_baseline_pass and "baseline_health_pass" in selected.columns:
        passes = selected["baseline_health_pass"].map(parse_optional_bool)
        selected = selected.loc[passes == True].copy()  # noqa: E712
    return selected.sort_values("run_id").reset_index(drop=True)


def _analyze_runs(rows: pd.DataFrame, dataset_root: Path, workers: int) -> pd.DataFrame:
    """Run the fault-effect analysis over the episodes in a thread pool and return one row per episode."""
    if rows.empty:
        return pd.DataFrame(columns=[*RUN_SUMMARY_COLUMNS, "error"])

    records: list[dict[str, Any]] = []
    completed = 0
    report_every = progress_interval(len(rows))
    FAULT_EFFECT_LOGGER.disabled = True
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(analyze_fault_run, row, dataset_root) for _, row in rows.iterrows()
            ]
            for future in as_completed(futures):
                record = asdict(future.result())
                record["fault_category"] = record.pop("category")
                records.append(record)
                completed += 1
                if completed % report_every == 0 or completed == len(rows):
                    LOGGER.info("Analyzed %d/%d faulty runs", completed, len(rows))
    finally:
        FAULT_EFFECT_LOGGER.disabled = False

    return pd.DataFrame(records).sort_values("run_id").reset_index(drop=True)


def _valid_run_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the episodes with a valid fault window and no analysis error, restricted to the summary columns."""
    if frame.empty:
        return pd.DataFrame(columns=RUN_SUMMARY_COLUMNS)
    valid = (frame["fault_window_valid"] == True) & frame["error"].isna()  # noqa: E712
    return frame.loc[valid, RUN_SUMMARY_COLUMNS].sort_values("run_id").reset_index(drop=True)


def _quantiles(series: pd.Series) -> tuple[float, float, float]:
    """Return the first quartile, median, and third quartile of a series, or NaN values."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return (float("nan"),) * 3
    return tuple(float(values.quantile(q)) for q in (0.25, 0.5, 0.75))


def _category_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize the effect metrics and observability rates per fault category."""
    rows: list[dict[str, Any]] = []
    for category, group in run_summary.groupby("fault_category", dropna=False):
        row: dict[str, Any] = {
            "fault_category": _safe_label(category),
            "run_count": len(group),
        }
        for metric in MAIN_EFFECT_METRICS:
            q1, median, q3 = _quantiles(group[metric])
            row.update(
                {
                    f"{metric}_q1": q1,
                    f"{metric}_median": median,
                    f"{metric}_q3": q3,
                    f"{metric}_iqr": q3 - q1,
                }
            )
        for flag in OBSERVABILITY_FLAGS:
            row[f"{flag}_rate"] = _rate(group[flag])
        rows.append(row)
    return pd.DataFrame(rows)


def _target_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize the observability rates and episode count per fault target."""
    rows: list[dict[str, Any]] = []
    for target, group in run_summary.groupby("fault_target", dropna=False):
        rows.append(
            {
                "fault_target": _safe_label(target),
                "fault_category": ",".join(
                    sorted({_safe_label(value) for value in group["fault_category"]})
                ),
                "run_count": len(group),
                "any_observable_rate": _rate(group["any_observable"]),
                "median_rtt_ratio": pd.to_numeric(group["rtt_ratio"], errors="coerce").median(),
                "median_route_count_abs_delta": pd.to_numeric(
                    group["route_count_abs_delta"], errors="coerce"
                ).median(),
            }
        )
    return pd.DataFrame(rows)


def _kind_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize the observability rates and episode count per candidate type."""
    rows: list[dict[str, Any]] = []
    for kind, group in run_summary.groupby("root_cause_kind", dropna=False):
        row = {"root_cause_kind": _safe_label(kind), "run_count": len(group)}
        row.update({f"{flag}_rate": _rate(group[flag]) for flag in OBSERVABILITY_FLAGS})
        rows.append(row)
    return pd.DataFrame(rows)


def _json_ready(value: Any) -> Any:
    """Convert nested values for JSON, replacing non-finite floats with None."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary_payload(run_summary: pd.DataFrame, target_summary: pd.DataFrame) -> dict[str, Any]:
    """Build the JSON summary with episode counts, overall observability rates, and the most and least visible targets."""
    if run_summary.empty:
        return {
            "run_count": 0,
            "counts_by_fault_category": {},
            "counts_by_root_cause_kind": {},
            "overall_observable_rates": {flag: None for flag in OBSERVABILITY_FLAGS},
            "top_5_most_visible_targets": [],
            "bottom_5_least_visible_targets": [],
        }
    category_counts = run_summary["fault_category"].value_counts(dropna=False).to_dict()
    kind_counts = run_summary["root_cause_kind"].value_counts(dropna=False).to_dict()
    sorted_targets = (
        target_summary.sort_values(
            ["any_observable_rate", "run_count", "fault_target"],
            ascending=[False, False, True],
        )
        if not target_summary.empty
        else target_summary
    )
    return _json_ready(
        {
            "run_count": len(run_summary),
            "counts_by_fault_category": {
                _safe_label(key): int(value) for key, value in category_counts.items()
            },
            "counts_by_root_cause_kind": {
                _safe_label(key): int(value) for key, value in kind_counts.items()
            },
            "overall_observable_rates": {
                flag: _rate(run_summary[flag]) for flag in OBSERVABILITY_FLAGS
            },
            "top_5_most_visible_targets": sorted_targets.head(5).to_dict(orient="records"),
            "bottom_5_least_visible_targets": sorted_targets.tail(5)
            .iloc[::-1]
            .to_dict(orient="records"),
        }
    )


def _empty_figure(title: str, message: str) -> go.Figure:
    """Return a figure that shows only a message, for cohorts without valid episodes."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5)
    figure.update_layout(title=title, xaxis_visible=False, yaxis_visible=False)
    return figure


def _write_figures(
    run_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    kind_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the fault signature, target observability, and candidate type figures as standalone HTML."""
    if run_summary.empty:
        signatures = _empty_figure("Fault Category Signatures", "No valid faulty runs")
    else:
        long = run_summary.melt(
            id_vars=["fault_category"],
            value_vars=MAIN_EFFECT_METRICS,
            var_name="metric",
            value_name="value",
        )
        signatures = px.box(
            long,
            x="fault_category",
            y="value",
            color="fault_category",
            facet_col="metric",
            facet_col_wrap=2,
            points=False,
            title="Fault-window effects by category",
        )
        signatures.update_yaxes(matches=None)

    if target_summary.empty:
        targets = _empty_figure("Target Visibility", "No target summaries")
    else:
        visible = target_summary.sort_values("any_observable_rate").tail(20)
        targets = px.bar(
            visible,
            x="any_observable_rate",
            y="fault_target",
            color="fault_category",
            orientation="h",
            range_x=[0, 1],
            title="Share of runs with an observable symptom",
        )

    if kind_summary.empty:
        kinds = _empty_figure("Root-cause Kind Visibility", "No kind summaries")
    else:
        long = kind_summary.melt(
            id_vars=["root_cause_kind"],
            value_vars=[f"{flag}_rate" for flag in OBSERVABILITY_FLAGS],
            var_name="signal",
            value_name="observable_rate",
        )
        kinds = px.bar(
            long,
            x="root_cause_kind",
            y="observable_rate",
            color="signal",
            barmode="group",
            range_y=[0, 1],
            title="Observability by root-cause kind",
        )

    for filename, figure in (
        ("figure_1_category_signatures.html", signatures),
        ("figure_2_target_visibility.html", targets),
        ("figure_3_root_cause_kind_visibility.html", kinds),
    ):
        figure.write_html(output_dir / filename, include_plotlyjs=True)


def main() -> None:
    """Select the faulty episodes, analyze them, and write the CSV tables, JSON summary, and figures."""
    args = parse_args()
    configure_logging(args.log_level)
    validate_dataset_root(args.dataset_root)
    output_dir = args.output_dir or (args.dataset_root / "analysis_rca_readiness")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.dataset_root)
    if args.max_runs is not None:
        manifest = manifest.head(args.max_runs).copy()
    selected = _select_faulty_runs(manifest, args.require_baseline_pass)
    LOGGER.info("Selected %d successful faulty runs", len(selected))

    analyzed = _analyze_runs(selected, args.dataset_root, args.workers)
    run_summary = _valid_run_summary(analyzed)
    category_summary = _category_summary(run_summary)
    target_summary = _target_summary(run_summary)
    kind_summary = _kind_summary(run_summary)

    run_summary.to_csv(output_dir / "run_rca_summary.csv", index=False)
    category_summary.to_csv(output_dir / "category_summary.csv", index=False)
    target_summary.to_csv(output_dir / "target_summary.csv", index=False)
    kind_summary.to_csv(output_dir / "root_cause_kind_summary.csv", index=False)
    write_json(output_dir / "summary.json", _summary_payload(run_summary, target_summary))
    _write_figures(run_summary, target_summary, kind_summary, output_dir)
    LOGGER.info("Finished Stage 1 RCA-readiness analysis in %s", output_dir)


if __name__ == "__main__":
    main()
