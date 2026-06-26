#!/usr/bin/env bash
# Uninstall the LarkSnap systemd unit.
#
# Usage:
#     sudo ./scripts/uninstall_service.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${LARKSNAP_CONFIG:-${PROJECT_DIR}/config/config.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

UNIT_PATH="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG_PATH}'))['service'].get('systemd_unit_path','/etc/systemd/system/larksnap.service'))")"
SERVICE_NAME="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG_PATH}'))['service'].get('name','larksnap'))")"

if [[ "$EUID" -ne 0 ]]; then
    echo "This script must be run as root. Re-run with sudo:" >&2
    exit 1
fi

if systemctl list-unit-files "${SERVICE_NAME}.service" >/dev/null 2>&1; then
    systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
    systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
fi

if [[ -f "${UNIT_PATH}" ]]; then
    rm -f "${UNIT_PATH}"
    echo "Removed ${UNIT_PATH}"
fi

systemctl daemon-reload
systemctl reset-failed "${SERVICE_NAME}" 2>/dev/null || true
echo "Service '${SERVICE_NAME}' uninstalled."
