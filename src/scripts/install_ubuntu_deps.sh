#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run as root: sudo $0"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "[1/4] Updating apt package index..."
apt-get update

echo "[2/4] Installing core runtime packages for ridge..."
apt-get install -y \
  mininet \
  openvswitch-switch \
  iproute2 \
  iputils-ping \
  iperf3 \
  frr \
  frr-pythontools \
  net-tools

echo "[3/4] Attempting optional D-ITG package install (if available)..."
if apt-cache show d-itg >/dev/null 2>&1; then
  apt-get install -y d-itg
else
  echo "d-itg package not available on this Ubuntu release; ITGSend/ITGRecv may need manual install."
fi

echo "[4/4] Ensuring FRR binaries are in PATH for login shells..."
cat >/etc/profile.d/frr-path.sh <<'EOF'
export PATH="$PATH:/usr/lib/frr"
EOF
chmod 0644 /etc/profile.d/frr-path.sh

echo
echo "Installation complete."
echo "Recommended verification:"
echo "  ./src/scripts/check_system_deps.sh"
echo "  which zebra ospfd vtysh"
