"""Directory preparation and cleanup of FRRouting, D-ITG, Mininet processes, and leftover links for one Mininet instance."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

from ridge.sim.topology import runtime_name, topology_spec

FRR_RUNTIME_ROOT = Path("/tmp/ridge_frr")


def cleanup_mininet_state(log: Callable[[str], None], *, banner: str) -> None:
    """Clean stale FRR runtime and Mininet state, logging through the caller's sink."""
    cleanup_stale_frr_runtime(log)
    log(banner)
    subprocess.run(["mn", "-c"], check=True)


def prepare_empty_directory(path: Path, *, kind: str) -> None:
    """Create ``path`` fresh, refusing to reuse a non-empty directory."""
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{kind} path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"{kind} directory must be empty: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)


def frr_runtime_dir(instance_id: str, pid: int | None = None) -> Path:
    """Return the FRRouting runtime directory of a Mininet instance and process."""
    runtime_instance = instance_id or "default"
    return FRR_RUNTIME_ROOT / f"{runtime_instance}-p{pid or os.getpid()}"


def _pid_values(runtime_dir: Path) -> list[int]:
    """Return the positive process identifiers read from every pid file under a runtime directory."""
    pids: list[int] = []
    for path in runtime_dir.glob("**/*.pid"):
        try:
            value = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if value > 0:
            pids.append(value)
    return sorted(set(pids))


def _pid_alive(pid: int) -> bool:
    """Return whether a process with the identifier exists."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pid_matches_frr_runtime(pid: int, runtime_dir: Path) -> bool:
    """Guard privileged cleanup against stale PID files and PID reuse."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    arguments = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    if not arguments or Path(arguments[0]).name not in {"zebra", "ospfd"}:
        return False
    runtime_selector = str(runtime_dir)
    return any(runtime_selector in argument for argument in arguments[1:])


def _signal_pid(pid: int, signum: int) -> bool:
    """Send a signal to a process and return whether the call succeeded."""
    try:
        os.kill(pid, signum)
    except OSError:
        return False
    return True


def cleanup_frr_runtime_dir(runtime_dir: Path, log: Callable[[str], None]) -> int:
    """Terminate the FRRouting daemons recorded in a runtime directory and remove the directory."""
    if not runtime_dir.exists():
        return 0
    recorded_pids = _pid_values(runtime_dir)
    pids = [pid for pid in recorded_pids if _pid_matches_frr_runtime(pid, runtime_dir)]
    skipped = len(recorded_pids) - len(pids)
    signaled = 0
    for pid in pids:
        if _signal_pid(pid, signal.SIGTERM):
            signaled += 1
    if signaled:
        time.sleep(0.5)
    for pid in pids:
        if _pid_alive(pid):
            _signal_pid(pid, signal.SIGKILL)
    shutil.rmtree(runtime_dir, ignore_errors=True)
    log(f"removed FRR runtime {runtime_dir} pids={len(pids)} skipped_unowned_pids={skipped}")
    return len(pids)


def _expected_runtime_prefixes(instance_id: str) -> list[str]:
    """Return the runtime names of every node of the instance."""
    spec = topology_spec()
    return [runtime_name(str(name), instance_id) for name in [*spec["switches"], *spec["hosts"]]]


def _run_quiet(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command with its output discarded and return the completed process."""
    return subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, text=True
    )


def _cleanup_instance_ditg(run_dir: Path, log: Callable[[str], None]) -> int:
    """Terminate, then kill, the D-ITG processes whose command line names the episode directory."""
    run_dir_pattern = re.escape(str(run_dir))
    patterns = [
        rf"ITGSend.*{run_dir_pattern}",
        rf"ITGRecv.*{run_dir_pattern}",
    ]
    attempts = 0
    term_matched = False
    for signame in ("-TERM", "-KILL"):
        for pattern in patterns:
            result = _run_quiet(["pkill", signame, "-f", pattern])
            if signame == "-TERM" and result.returncode == 0:
                term_matched = True
            attempts += 1
        if signame == "-TERM" and term_matched:
            time.sleep(0.5)
    log(f"ran D-ITG run-dir cleanup selectors for {run_dir}")
    return attempts


def _cleanup_instance_mininet_nodes(instance_id: str, log: Callable[[str], None]) -> int:
    """Terminate, then kill, the Mininet node processes of the instance by runtime name."""
    prefixes = _expected_runtime_prefixes(instance_id)
    attempts = 0
    term_matched = False
    for signame in ("-TERM", "-KILL"):
        for prefix in prefixes:
            result = _run_quiet(["pkill", signame, "-f", f"mininet:{prefix}"])
            if signame == "-TERM" and result.returncode == 0:
                term_matched = True
            attempts += 1
        if signame == "-TERM" and term_matched:
            time.sleep(0.5)
    log(f"ran Mininet node cleanup selectors for instance_id={instance_id}")
    return attempts


def _cleanup_instance_links(instance_id: str, log: Callable[[str], None]) -> int:
    """Delete leftover virtual interfaces whose names carry the instance's runtime prefixes."""
    prefixes = tuple(f"{prefix}-eth" for prefix in _expected_runtime_prefixes(instance_id))
    result = subprocess.run(
        ["ip", "-o", "link", "show"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        log(f"failed to list links for instance_id={instance_id}")
        return 0
    deleted = 0
    for line in result.stdout.splitlines():
        match = re.match(r"\d+:\s+([^:@]+)", line)
        if not match:
            continue
        interface = match.group(1)
        if not interface.startswith(prefixes):
            continue
        _run_quiet(["ip", "link", "delete", interface])
        deleted += 1
    if deleted:
        log(f"deleted {deleted} links for instance_id={instance_id}")
    return deleted


def cleanup_instance_runtime(
    instance_id: str, run_dir: Path, log: Callable[[str], None]
) -> dict[str, int]:
    """Remove every FRRouting, D-ITG, Mininet process, and link left by one instance and count what was cleaned."""
    if not instance_id or not instance_id.isalnum():
        raise ValueError("cleanup_instance_runtime requires an alphanumeric instance_id")
    if not run_dir:
        raise ValueError("cleanup_instance_runtime requires a run_dir selector")

    frr_pids = 0
    frr_dirs = 0
    if FRR_RUNTIME_ROOT.exists():
        for entry in FRR_RUNTIME_ROOT.glob(f"{instance_id}-p*"):
            if not entry.is_dir():
                continue
            frr_pids += cleanup_frr_runtime_dir(entry, log)
            frr_dirs += 1
    return {
        "frr_runtime_dirs": frr_dirs,
        "frr_pids": frr_pids,
        "ditg_cleanup_selectors": _cleanup_instance_ditg(run_dir, log),
        "mininet_node_cleanup_selectors": _cleanup_instance_mininet_nodes(instance_id, log),
        "links_deleted": _cleanup_instance_links(instance_id, log),
    }


def cleanup_stale_frr_runtime(log: Callable[[str], None]) -> int:
    """Remove FRRouting runtime directories whose owning process no longer exists."""
    if not FRR_RUNTIME_ROOT.exists():
        return 0
    cleaned = 0
    for entry in FRR_RUNTIME_ROOT.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        pid = None
        if "-p" in name:
            suffix = name.rsplit("-p", 1)[-1]
            if suffix.isdigit():
                pid = int(suffix)
        stale = False
        if pid is None:
            stale = True
        else:
            try:
                os.kill(pid, 0)
            except OSError:
                stale = True
        if stale:
            cleanup_frr_runtime_dir(entry, log)
            cleaned += 1
    if cleaned:
        noun = "directory" if cleaned == 1 else "directories"
        log(f"removed {cleaned} stale FRR runtime {noun}")
    return cleaned
