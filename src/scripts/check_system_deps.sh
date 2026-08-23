#!/usr/bin/env bash
set -euo pipefail

missing=0
for bin in mn ovs-vsctl tc ping ITGSend ITGRecv iperf3 zebra ospfd vtysh; do
  if command -v "$bin" >/dev/null 2>&1; then
    printf '%-12s %s\n' "$bin" "$(command -v "$bin")"
  else
    printf '%-12s %s\n' "$bin" "MISSING"
    missing=1
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Configured PYTHON_BIN is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 1
  fi
elif [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys

expected = (3, 12, 3)
version_ok = sys.version_info[:3] == expected
suffix = "" if version_ok else " (expected 3.12.3)"
print(f"{'python':<12} {sys.version.split()[0]}{suffix}")

missing = not version_ok
modules = ["mininet", "torch", "pandas", "streamlit", "plotly", "networkx"]
for name in modules:
    try:
        __import__(name)
        print(f"{name:<12} OK")
    except Exception:
        print(f"{name:<12} MISSING")
        missing = True

raise SystemExit(1 if missing else 0)
PY
then
  missing=1
fi

exit "$missing"
