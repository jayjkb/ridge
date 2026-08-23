"""Streamlit explorer for RIDGE Stage 6 evaluation outputs."""

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
    json_artifact_matches,
    render_artifact_inspector_entries,
    render_metric_notes,
    require_artifact_payload,
    split_run_counts,
    split_window_counts,
)

SUMMARY_FILENAMES = (
    "eval_summary.json",
    "test_evaluation.json",
)

REQUIRED_STAGE6_FILES = ("per_sample_ridge_predictions.csv",)

REQUIRED_STAGE4_FILES = (
    "residual_window_index.csv",
    "residual_index.json",
    "residual_splits.json",
    "candidate_index.json",
    "topology.json",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse the evaluation root and optional Stage-4 root, ignoring the options Streamlit adds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root", type=Path, default=None, help="Path to Stage 6 evaluation output directory"
    )
    parser.add_argument(
        "--stage4-root", type=Path, default=None, help="Optional paired Stage 4 dataset root"
    )
    return parser.parse_known_args()[0]


def resolve_summary_filename(path: Path) -> str | None:
    """Return the first summary filename carrying a valid ridge_evaluation artifact."""
    for filename in SUMMARY_FILENAMES:
        candidate = path / filename
        if candidate.is_file() and json_artifact_matches(candidate, "ridge_evaluation"):
            return filename
    return None


def is_stage6_model_root(path: Path) -> bool:
    """Return whether a path holds the prediction export and a valid evaluation summary."""
    return bool(
        path.exists()
        and path.is_dir()
        and all((path / filename).exists() for filename in REQUIRED_STAGE6_FILES)
        and resolve_summary_filename(path) is not None
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
def cached_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Load the RCA model checkpoint named by the summary and validate its artifact envelope, cached per session."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return require_artifact_payload(payload, "ridge_checkpoint", source=path)


def resolve_checkpoint_path(evaluation_root: Path, summary: dict[str, Any]) -> Path:
    """Resolve the Stage 5 checkpoint referenced by a Stage 6 summary."""
    raw_path = summary.get("checkpoint")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("the evaluation summary does not record its Stage 5 checkpoint")
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (evaluation_root / candidate).resolve()


def split_window_count_map(
    window_index: pd.DataFrame, splits: dict[str, list[str]]
) -> dict[str, int]:
    """Return the window count of each split as a dictionary."""
    counts = split_window_counts(window_index, splits)
    return counts.set_index("split")["window_count"].to_dict()


def render_overview(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
    checkpoint: dict[str, Any],
    stage4_root: Path | None,
) -> None:
    """Show the held-out ranking, detection, and category cards with the evaluated dataset context."""
    st.subheader("Evaluation Overview")
    st.caption(
        "Stage 6 evaluates the fixed Stage 5 checkpoint on held-out Stage 4 windows only. "
        "These cards summarize exact root-cause accuracy, shortlist quality, derived fault detection, and category quality."
    )
    test_metrics = summary.get("test_metrics", {})
    candidate_ids: list[Any] = []
    residual_mode = "unknown"
    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        candidate_ids = list(
            cached_json(str(stage4_root), "candidate_index.json").get("candidate_ids", [])
        )
        residual_index_path = stage4_root / "residual_index.json"
        if residual_index_path.is_file():
            residual_mode = str(
                cached_json(str(stage4_root), "residual_index.json").get("residual_mode", "unknown")
            )
    category_labels = checkpoint.get("fault_category_labels", [])
    model_config = checkpoint.get("model_config", {})
    inference = summary.get("inference", {})
    model_config = model_config if isinstance(model_config, dict) else {}
    inference = inference if isinstance(inference, dict) else {}

    cards = [
        ("Test Windows", format_integer(len(predictions))),
        ("Candidate Top-1", format_percent(test_metrics.get("candidate_top1_accuracy"))),
        ("Candidate Top-3", format_percent(test_metrics.get("candidate_top3_accuracy"))),
        ("Fault F1", format_number(test_metrics.get("fault_present_f1"))),
        ("Category Macro-F1", format_number(test_metrics.get("category_macro_f1"))),
        (
            "Joint Top-1",
            format_percent(test_metrics.get("joint_top1_entity_and_category_accuracy")),
        ),
        ("MRR", format_number(test_metrics.get("mrr"))),
        ("Mean Rank", format_number(test_metrics.get("mean_rank"))),
        ("Candidates", format_integer(summary.get("candidate_count", len(candidate_ids)))),
        ("Categories", format_integer(len(category_labels))),
        ("Top-k Export", format_integer(summary.get("top_k"))),
        (
            "Input Mode",
            residual_mode
            if residual_mode != "unknown"
            else str(checkpoint.get("residual_mode", "unknown")),
        ),
        (
            "Graph Structure",
            "enabled" if model_config.get("use_graph_structure", True) else "disabled",
        ),
        ("Throughput", f"{format_number(inference.get('windows_per_second'), digits=1)} windows/s"),
        (
            "Latency",
            f"{format_number(inference.get('mean_latency_ms_per_window'), digits=2)} ms/window",
        ),
    ]

    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        splits = cached_json(str(stage4_root), "residual_splits.json")
        window_index = cached_residual_window_index(str(stage4_root))
        run_counts = split_run_counts(splits)
        test_windows = split_window_count_map(window_index, splits)
        cards.extend(
            [
                ("Test Runs", format_integer(run_counts.get("test", 0))),
                ("Stage 4 Test Windows", format_integer(test_windows.get("test", 0))),
            ]
        )

    columns = st.columns(4)
    for index, (label, value) in enumerate(cards):
        columns[index % len(columns)].metric(label, value)

    render_metric_notes(
        [
            {
                "Metric": "Candidate Top-1",
                "Meaning": "Exact-match root-cause accuracy on the held-out test windows.",
            },
            {
                "Metric": "Candidate Top-3",
                "Meaning": "Whether the true root cause appears anywhere in the top-3 shortlist.",
            },
            {
                "Metric": "MRR / Mean Rank",
                "Meaning": "How high the true candidate tends to appear in the ranking. Better models move the truth closer to rank 1.",
            },
            {
                "Metric": "Fault F1",
                "Meaning": "Derived fault detection quality computed by treating candidate `none` as fault-absent.",
            },
            {
                "Metric": "Category Macro-F1",
                "Meaning": "Balanced coarse fault-category quality across categories instead of letting the majority class dominate.",
            },
            {
                "Metric": "Joint Top-1",
                "Meaning": "Stricter metric that requires both the root-cause candidate and the coarse category to be correct.",
            },
            {
                "Metric": "Top-k Export",
                "Meaning": "How many ranked candidates were exported into the per-sample CSV. Aggregate metrics still stay fixed to top-1 and top-3.",
            },
            {
                "Metric": "Throughput / Latency",
                "Meaning": "End-to-end evaluation timing from the saved inference block, including data loading and diagnostics.",
            },
        ]
    )


def render_metric_groups(summary: dict[str, Any]) -> None:
    """Show bounded scores, ranks, and reconstruction errors as separate bar charts."""
    st.subheader("Metric Groups")
    st.caption(
        "Bounded scores, ranks, and reconstruction errors use separate axes so their "
        "scales do not imply a meaningful numerical comparison."
    )
    test_metrics = summary.get("test_metrics", {})

    bounded_frame = pd.DataFrame(
        [
            {"metric": label, "value": float(test_metrics.get(key, float("nan")))}
            for label, key in (
                ("top-1", "candidate_top1_accuracy"),
                ("top-3", "candidate_top3_accuracy"),
                ("MRR", "mrr"),
                ("fault precision", "fault_present_precision"),
                ("fault recall", "fault_present_recall"),
                ("fault F1", "fault_present_f1"),
                ("category accuracy", "category_accuracy"),
                ("category macro-F1", "category_macro_f1"),
                ("joint top-1", "joint_top1_entity_and_category_accuracy"),
            )
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "mean rank", "value": test_metrics.get("mean_rank")},
            {
                "diagnostic": "reconstruction MSE",
                "value": test_metrics.get("residual_reconstruction_mse"),
            },
            {
                "diagnostic": "reconstruction MAE",
                "value": test_metrics.get("residual_reconstruction_mae"),
            },
        ]
    )

    left, right = st.columns((3, 2))
    with left:
        st.plotly_chart(
            px.bar(
                bounded_frame,
                x="metric",
                y="value",
                range_y=[0, 1],
                title="Bounded Accuracy, F1, and Ranking Scores",
            ),
            width="stretch",
        )
    with right:
        st.dataframe(diagnostics, width="stretch", hide_index=True)


def _category_label_map(summary: dict[str, Any], checkpoint: dict[str, Any]) -> dict[int, str]:
    """Return the fault-category index to name map from the checkpoint, or from the per-class metrics."""
    labels = checkpoint.get("fault_category_labels", [])
    if labels:
        return {index: str(label) for index, label in enumerate(labels)}

    per_class = summary.get("test_metrics", {}).get("category_per_class_f1", [])
    return {
        int(row.get("label", index)): str(
            row.get("label_name", f"class_{int(row.get('label', index))}")
        )
        for index, row in enumerate(per_class)
    }


def render_error_summary(
    summary: dict[str, Any], predictions: pd.DataFrame, checkpoint: dict[str, Any]
) -> None:
    """Show per-category F1, how often the top-3 shortlist rescues a top-1 miss, and the most predicted candidates."""
    st.subheader("Per-Class and Error Summary")
    st.caption(
        "These summaries connect the aggregate scores to concrete model behavior: which categories are weaker, how often top-3 rescues a top-1 miss, and which candidates dominate predictions."
    )

    test_metrics = summary.get("test_metrics", {})
    label_map = _category_label_map(summary, checkpoint)
    per_class = test_metrics.get("category_per_class_f1", [])
    per_class_frame = pd.DataFrame(
        [
            {
                "category_index": int(row.get("label", -1)),
                "category_label": label_map.get(
                    int(row.get("label", -1)), str(row.get("label", "unknown"))
                ),
                "precision": float(row.get("precision", float("nan"))),
                "recall": float(row.get("recall", float("nan"))),
                "f1": float(row.get("f1", float("nan"))),
                "support": int(row.get("support", 0)),
            }
            for row in per_class
        ]
    )

    top1 = pd.to_numeric(
        predictions.get("candidate_top1_correct", pd.Series(index=predictions.index, dtype=float)),
        errors="coerce",
    ).fillna(0)
    top1_correct = int(top1.sum())
    exported_top_k = int(summary.get("top_k", 0) or 0)
    has_top3 = exported_top_k >= 3 and "candidate_top3_correct" in predictions.columns
    top3 = (
        pd.to_numeric(predictions["candidate_top3_correct"], errors="coerce").fillna(0)
        if has_top3
        else pd.Series(index=predictions.index, dtype=float)
    )
    top3_correct = int(top3.sum()) if has_top3 else None
    rescued = int(((top1 == 0) & (top3 == 1)).sum()) if has_top3 else None
    true_category = predictions.get(
        "true_category", pd.Series(index=predictions.index, dtype=object)
    ).astype(str)
    pred_category = predictions.get(
        "pred_category", pd.Series(index=predictions.index, dtype=object)
    ).astype(str)
    category_correct = int((true_category == pred_category).sum())

    columns = st.columns(4)
    columns[0].metric("Top-1 Correct", format_integer(top1_correct))
    columns[1].metric("Exported Top-3 Correct", format_integer(top3_correct))
    columns[2].metric("Exported Top-3 Rescued", format_integer(rescued))
    columns[3].metric("Category Correct", format_integer(category_correct))

    left, right = st.columns(2)
    with left:
        if not per_class_frame.empty:
            st.dataframe(per_class_frame, width="stretch", hide_index=True)
            st.plotly_chart(
                px.bar(per_class_frame, x="category_label", y="f1", title="Category Per-Class F1"),
                width="stretch",
            )
        else:
            st.info("No per-class category metrics were found in the evaluation summary.")
    with right:
        pred_counts = (
            predictions.get("pred_top1", pd.Series(dtype=object))
            .astype(str)
            .value_counts()
            .rename_axis("candidate_id")
            .reset_index(name="prediction_count")
        )
        if not pred_counts.empty:
            st.plotly_chart(
                px.bar(
                    pred_counts.head(12),
                    x="candidate_id",
                    y="prediction_count",
                    title="Most Frequent Top-1 Predictions",
                ),
                width="stretch",
            )
        error_breakdown = pd.DataFrame(
            [
                {"outcome": "top-1 correct", "count": top1_correct},
                {"outcome": "top-3 rescued", "count": rescued},
                {
                    "outcome": "outside top-3",
                    "count": max(0, len(predictions) - top3_correct)
                    if top3_correct is not None
                    else None,
                },
            ]
        )
        st.plotly_chart(
            px.bar(error_breakdown, x="outcome", y="count", title="Shortlist Outcome Breakdown"),
            width="stretch",
        )


def render_audit_table(predictions: pd.DataFrame) -> None:
    """Show a filterable table of test windows, defaulting to top-1 mistakes."""
    st.subheader("Prediction Audit Table")
    st.caption(
        "Use this table to inspect concrete correct and incorrect test windows without leaving the dashboard."
    )

    mistakes_only = st.checkbox("Show only top-1 mistakes", value=True)
    available_categories = sorted(
        predictions.get("true_category", pd.Series(dtype=object)).astype(str).unique().tolist()
    )
    category_choice = st.selectbox(
        "Filter by true category", ["All"] + available_categories, index=0
    )

    frame = predictions.copy()
    if mistakes_only and "candidate_top1_correct" in frame.columns:
        frame = frame[
            pd.to_numeric(frame["candidate_top1_correct"], errors="coerce").fillna(0) == 0
        ]
    if category_choice != "All" and "true_category" in frame.columns:
        frame = frame[frame["true_category"].astype(str) == category_choice]

    if "reconstruction_error" in frame.columns:
        frame = frame.sort_values("reconstruction_error", ascending=False)

    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption(f"Rows shown: {len(frame)}")


def render_artifact_context(
    model_root: Path,
    summary: dict[str, Any],
    summary_filename: str,
    checkpoint: dict[str, Any],
    stage4_root: Path | None,
) -> None:
    """Show the evaluation summary, checkpoint configuration, and paired Stage-4 artifacts in collapsed expanders."""
    st.subheader("Artifact Context")
    st.caption(
        "These raw artifacts provide quick provenance checks for the evaluation results shown above."
    )

    artifacts: list[tuple[str, dict[str, Any]]] = [
        (summary_filename, summary),
    ]
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, dict):
        artifacts.append(("checkpoint.model_config", model_config))

    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        artifacts.append(
            ("stage4.residual_splits.json", cached_json(str(stage4_root), "residual_splits.json"))
        )
        if (stage4_root / "residual_index.json").is_file():
            artifacts.append(
                ("stage4.residual_index.json", cached_json(str(stage4_root), "residual_index.json"))
            )
        artifacts.append(
            ("stage4.candidate_index.json", cached_json(str(stage4_root), "candidate_index.json"))
        )

    render_artifact_inspector_entries(artifacts)

    st.caption(f"Evaluation root: {model_root}")
    if stage4_root is not None and is_stage4_dataset_root(stage4_root):
        st.caption(f"Paired Stage 4 dataset: {stage4_root}")


def main() -> None:
    """Validate the evaluation root, load the Stage-6 artifacts, and show every section."""
    st.set_page_config(page_title="RIDGE Stage 6 Evaluation Explorer", layout="wide")
    args = parse_args()
    model_root = args.model_root.expanduser().resolve() if args.model_root else None
    if model_root is None or not is_stage6_model_root(model_root):
        st.error(f"Model root is not a valid Stage 6 evaluation directory: {model_root}")
        st.stop()

    stage4_root = args.stage4_root.expanduser().resolve() if args.stage4_root else None
    if stage4_root is not None and not is_stage4_dataset_root(stage4_root):
        st.sidebar.warning(f"Ignoring invalid Stage 4 dataset root: {stage4_root}")
        stage4_root = None

    st.title("RIDGE Stage 6 Evaluation Explorer")
    st.caption(f"Evaluation root: {model_root}")

    try:
        summary_filename = resolve_summary_filename(model_root)
        if summary_filename is None:
            raise ValueError(
                "no evaluation summary with artifact type ridge_evaluation was found "
                f"(expected one of {', '.join(SUMMARY_FILENAMES)})"
            )
        summary = cached_json(str(model_root), summary_filename)
        predictions = cached_csv(str(model_root), "per_sample_ridge_predictions.csv")
        checkpoint_path = resolve_checkpoint_path(model_root, summary)
        checkpoint = cached_checkpoint(str(checkpoint_path))
    except Exception as exc:
        st.error(f"Failed to load Stage 6 artifacts: {exc}")
        st.stop()

    st.caption(f"Evaluation summary: {summary_filename}")

    render_overview(summary, predictions, checkpoint, stage4_root)
    render_metric_groups(summary)
    render_error_summary(summary, predictions, checkpoint)
    render_audit_table(predictions)
    render_artifact_context(model_root, summary, summary_filename, checkpoint, stage4_root)


if __name__ == "__main__":
    main()
