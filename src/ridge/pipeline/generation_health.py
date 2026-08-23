"""Resource and cadence gates for long-running Stage-1 generation jobs."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ridge.common.io import percentile
from ridge.io.stage1_timing import evaluate_run_timing


def _cpu_counters() -> tuple[int, int, int] | None:
    """Return total, idle, and iowait jiffies from /proc/stat, or None when unreadable."""
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
    except (OSError, ValueError, IndexError):
        return None
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    iowait = values[4] if len(values) > 4 else 0
    return total, idle, iowait


def _memory_used_fraction() -> float | None:
    """Return the used memory fraction from /proc/meminfo, or None when unreadable."""
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw_value = line.split(":", 1)
            values[key] = int(raw_value.strip().split()[0])
        return 1.0 - values["MemAvailable"] / values["MemTotal"]
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return None


def _io_pressure_avg10() -> float | None:
    """Return the ten-second IO pressure fraction from /proc/pressure/io, or None when unavailable."""
    try:
        first_line = Path("/proc/pressure/io").read_text(encoding="utf-8").splitlines()[0]
        fields = dict(field.split("=", 1) for field in first_line.split()[1:])
        return float(fields["avg10"]) / 100.0
    except (OSError, ValueError, KeyError, IndexError):
        return None


class HostResourceMonitor:
    """Sample emulation-host pressure without adding a third-party monitoring dependency."""

    def __init__(self, output: Path, *, interval_sec: float = 1.0) -> None:
        self.output = output
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._active_worker_lanes = 0
        self._active_simulators = 0
        self._process_cpu_affinity = sorted(os.sched_getaffinity(0))
        self._online_cpu_count = os.cpu_count() or len(self._process_cpu_affinity)
        self._samples: list[dict[str, Any]] = []

    def worker_started(self) -> None:
        """Increase the count of active worker lanes by one."""
        with self._lock:
            self._active_worker_lanes += 1

    def worker_finished(self) -> None:
        """Decrease the count of active worker lanes by one, floored at zero."""
        with self._lock:
            self._active_worker_lanes = max(0, self._active_worker_lanes - 1)

    def simulator_started(self) -> None:
        """Increase the count of running per-episode generators by one."""
        with self._lock:
            self._active_simulators += 1

    def simulator_finished(self) -> None:
        """Decrease the count of running per-episode generators by one, min is zero."""
        with self._lock:
            self._active_simulators = max(0, self._active_simulators - 1)

    def start(self) -> None:
        """Start the daemon thread that samples the emulation host."""
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="stage1-resource-monitor", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        """Sample CPU, memory, and IO pressure every interval and append each sample to the output file."""
        previous_cpu = _cpu_counters()
        with self.output.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                current_cpu = _cpu_counters()
                cpu_used: float | None = None
                io_wait: float | None = None
                if previous_cpu is not None and current_cpu is not None:
                    total_delta = current_cpu[0] - previous_cpu[0]
                    if total_delta > 0:
                        cpu_used = 1.0 - (current_cpu[1] - previous_cpu[1]) / total_delta
                        io_wait = (current_cpu[2] - previous_cpu[2]) / total_delta
                previous_cpu = current_cpu
                with self._lock:
                    active_worker_lanes = self._active_worker_lanes
                    active_simulators = self._active_simulators
                sample = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "monotonic_sec": time.monotonic(),
                    "active_simulators": active_simulators,
                    "active_worker_lanes": active_worker_lanes,
                    "load_1m": os.getloadavg()[0],
                    "cpu_used_fraction": cpu_used,
                    "memory_used_fraction": _memory_used_fraction(),
                    "cpu_iowait_fraction": io_wait,
                    "io_pressure_some_avg10_fraction": _io_pressure_avg10(),
                }
                with self._lock:
                    self._samples.append(sample)
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
                self._stop.wait(self.interval_sec)

    def summary(self) -> dict[str, Any]:
        """Summarize the samples into percentiles and the headroom pass flag."""
        with self._lock:
            samples = list(self._samples)

        def finite(key: str) -> list[float]:
            """Return the finite values of one sample field."""
            return [
                float(sample[key])
                for sample in samples
                if sample.get(key) is not None and math.isfinite(float(sample[key]))
            ]

        cpu = finite("cpu_used_fraction")
        memory = finite("memory_used_fraction")
        iowait = finite("cpu_iowait_fraction")
        pressure = finite("io_pressure_some_avg10_fraction")
        cpu_p95 = percentile(cpu, 0.95)
        memory_max = max(memory, default=None)
        io_pressure_p95 = percentile(pressure, 0.95)
        summary = {
            "sample_count": len(samples),
            "max_active_simulators": max(
                (int(sample.get("active_simulators", 0)) for sample in samples), default=0
            ),
            "max_active_worker_lanes": max(
                (int(sample.get("active_worker_lanes", 0)) for sample in samples), default=0
            ),
            "process_cpu_affinity": self._process_cpu_affinity,
            "online_cpu_count": self._online_cpu_count,
            "reserved_cpu_count": max(0, self._online_cpu_count - len(self._process_cpu_affinity)),
            "cpu_used_p95": cpu_p95,
            "memory_used_max": memory_max,
            "cpu_iowait_p95": percentile(iowait, 0.95),
            "io_pressure_some_avg10_p95": io_pressure_p95,
        }
        summary["headroom_passed"] = bool(
            cpu_p95 is not None
            and memory_max is not None
            and cpu_p95 <= 0.80
            and memory_max <= 0.80
            and (io_pressure_p95 is None or io_pressure_p95 <= 0.80)
        )
        return summary

    def stop(self) -> dict[str, Any]:
        """Stop the sampling thread and return the summary."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_sec * 2))
        return self.summary()


class EpisodeRetryBudget:
    """Bound how many transient per-episode retries a generation may spend."""

    def __init__(self, *, max_per_episode: int, max_total: int) -> None:
        self.max_per_episode = max(0, int(max_per_episode))
        self.max_total = max(0, int(max_total))
        self._lock = threading.Lock()
        self._total_used = 0
        self._attempts_by_run: dict[int, int] = {}

    def try_consume(self, run_id: int) -> bool:
        """Reserve one retry for run_id if per-episode and global budgets remain."""
        with self._lock:
            used_here = self._attempts_by_run.get(run_id, 0)
            if used_here >= self.max_per_episode or self._total_used >= self.max_total:
                return False
            self._attempts_by_run[run_id] = used_here + 1
            self._total_used += 1
            return True

    def summary(self) -> dict[str, Any]:
        """Return the budget limits, retries used, and per-episode retry counts."""
        with self._lock:
            return {
                "max_per_episode": self.max_per_episode,
                "max_total": self.max_total,
                "total_retries_used": self._total_used,
                "global_budget_exhausted": self._total_used >= self.max_total,
                "retried_runs": {
                    str(run_id): count for run_id, count in sorted(self._attempts_by_run.items())
                },
            }


class GenerationGate:
    """Stop scheduling new episodes when rolling cadence checks fail."""

    def __init__(self, output: Path, *, duration_sec: float, interval_sec: float) -> None:
        self.output = output
        self.duration_sec = duration_sec
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._completed = 0
        self._failed_runs = 0
        self._timing_reports: list[dict[str, Any]] = []

    def record_episode(
        self,
        row: dict[str, object],
        *,
        resource_monitor: HostResourceMonitor | None = None,
        timing_report: dict[str, Any] | None = None,
    ) -> None:
        """Record one finished episode and, at the rolling checkpoints, write the summary and halt on failure."""
        with self._lock:
            self._completed += 1
            if row.get("status") != "ok":
                self._failed_runs += 1
            else:
                report = (
                    timing_report
                    if timing_report is not None
                    else evaluate_run_timing(
                        Path(str(row["log_dir"])),
                        duration_sec=self.duration_sec,
                        interval_sec=self.interval_sec,
                    )
                )
                self._timing_reports.append(report)
            if self._completed == 100 or self._completed % 250 == 0:
                resource_summary = (
                    resource_monitor.summary() if resource_monitor is not None else None
                )
                summary = self._summary(resource_summary)
                self._write(summary)
                if not summary["passed"]:
                    self.stop_event.set()

    def _summary(self, resource_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Aggregate episode counts and timing completeness into a summary whose pass flag also requires headroom."""
        total_expected = sum(report["expected_snapshot_count"] for report in self._timing_reports)
        total_complete = sum(report["completed_snapshot_count"] for report in self._timing_reports)
        summary = {
            "completed_episode_count": self._completed,
            "failed_episode_count": self._failed_runs,
            "timing_report_count": len(self._timing_reports),
            "aggregate_completeness": total_complete / total_expected if total_expected else 0.0,
            "failed_timing_run_count": sum(
                1 for report in self._timing_reports if not report["passed"]
            ),
            "passed": bool(self._timing_reports)
            and self._failed_runs == 0
            and all(report["passed"] for report in self._timing_reports),
        }
        if resource_summary is not None:
            summary["resource_summary"] = resource_summary
            summary["passed"] = bool(summary["passed"] and resource_summary.get("headroom_passed"))
        return summary

    def _write(self, summary: dict[str, Any]) -> None:
        """Write the summary as indented JSON to the gate output file."""
        self.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def finalize(
        self,
        resource_summary: dict[str, Any],
        *,
        certification_checks: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Write the final summary with headroom, certification checks, and whether generation was halted."""
        with self._lock:
            summary = self._summary(resource_summary)
            if certification_checks is not None:
                summary["certification_checks"] = dict(certification_checks)
                summary["passed"] = bool(summary["passed"] and all(certification_checks.values()))
            summary["generation_halted"] = self.stop_event.is_set()
            self._write(summary)
            return summary
