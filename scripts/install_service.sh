#!/usr/bin/env bash
# Install the LarkSnap systemd unit.
#
# Usage:
#     sudo ./scripts/install_service.sh [--user larkuser]
#
# After install, start the service with:
#     sudo systemctl start larksnap
#
# The script writes a unit file derived from the project's
# ServiceConfig and reloads systemd. The unit runs as root by
# default; pass --user to run unprivileged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

USER_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)
            USER_OVERRIDE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--user <name>]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# Probe uv availability.
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
else
    UV=""
fi

# Pull the configured unit path / user from the YAML config.
CONFIG_PATH="${LARKSNAP_CONFIG:-${PROJECT_DIR}/config/config.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

UNIT_PATH="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG_PATH}'))['service'].get('systemd_unit_path','/etc/systemd/system/larksnap.service'))")"
DESCRIPTION="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG_PATH}'))['service'].get('description','LarkSnap'))")"
TYPE="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG_PATH}'))['service'].get('systemd_type','notify'))")"
RESTART="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG_PATH}'))['service'].get('systemd_restart','on-failure'))")"
WANTED_BY="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG_PATH}'))['service'].get('systemd_wanted_by','multi-user.target'))")"
if [[ -n "${USER_OVERRIDE}" ]]; then
    USER_LINE="User=${USER_OVERRIDE}"
else
    CONFIG_USER="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG_PATH}'))['service'].get('systemd_user') or '')")"
    if [[ -n "${CONFIG_USER}" ]]; then
        USER_LINE="User=${CONFIG_USER}"
    else
        USER_LINE=""
    fi
fi

if [[ -n "${UV}" ]]; then
    EXEC_START="${UV} --project ${PROJECT_DIR} run python -m larksnap.main service"
else
    PYTHON_BIN="$(command -v python3)"
    EXEC_START="${PYTHON_BIN} -m larksnap.main service"
fi

LOG_FILE="/var/log/larksnap.log"
touch "${LOG_FILE}" 2>/dev/null || LOG_FILE="/tmp/larksnap.log"

UNIT_BODY="[Unit]
Description=${DESCRIPTION}
After=network-online.target
Wants=network-online.target

[Service]
Type=${TYPE}
ExecStart=${EXEC_START}
${USER_LINE}
Restart=${RESTART}
RestartSec=5
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}

[Install]
WantedBy=${WANTED_BY}
"

if [[ "$EUID" -ne 0 ]]; then
    echo "This script must be run as root. Re-run with sudo:" >&2
    echo "    sudo $0 $*" >&2
    exit 1
fi

echo "${UNIT_BODY}" > "${UNIT_PATH}"
echo "Wrote ${UNIT_PATH}"

systemctl daemon-reload
systemctl enable larksnap
echo "Service 'larksnap' installed and enabled."
echo "Start with:    sudo systemctl start larksnap"
echo "View status:   sudo systemctl status larksnap"
echo "View logs:     sudo journalctl -u larksnap -f"
