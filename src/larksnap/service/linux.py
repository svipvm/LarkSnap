"""Linux systemd service integration.

Responsibilities:

- :func:`run_linux_service` – long-running entry point invoked by
  ``python -m larksnap.main service`` on Linux. Sets up signal
  handlers, optionally calls ``sd_notify(READY=1)``, then blocks on
  the shared :class:`ServiceRunner`.
- :func:`install_linux_service` – writes a systemd unit file to the
  path configured in :class:`ServiceConfig` and reloads the systemd
  manager. Requires root (or a sudo prompt).
- :func:`uninstall_linux_service` – stops, disables, and removes the
  unit file.
- :data:`UNIT_TEMPLATE` – the unit file body. Kept in this module
  so it can be rendered / inspected in tests.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from larksnap.config.loader import load_config
from larksnap.service.runner import ServiceRunner, install_posix_stop_handlers

if TYPE_CHECKING:
    pass

_logger = logging.getLogger("larksnap.service.linux")

UNIT_TEMPLATE = """\
[Unit]
Description={description}
After=network-online.target
Wants=network-online.target

[Service]
Type={type}
ExecStart={exec_start}
{user_line}Restart={restart}
RestartSec=5
StandardOutput=append:{log_file}
StandardError=append:{log_file}

[Install]
WantedBy={wanted_by}
"""


def _detect_python() -> str:
    """Return the absolute path to the Python interpreter to use."""
    executable = sys.executable
    return os.path.realpath(executable)


def _detect_uv() -> str | None:
    """Return absolute path to ``uv`` if available, else None.

    When ``uv`` is on ``$PATH`` the generated unit file invokes the
    project via ``uv run`` so deployment follows the project's lock
    file. When ``uv`` is not available we fall back to a direct
    ``python -m larksnap.main service`` invocation.
    """
    uv_path = shutil.which("uv")
    return os.path.realpath(uv_path) if uv_path is not None else None


def _build_exec_start(project_dir: Path) -> str:
    """Build the ``ExecStart=`` line for the unit file."""
    uv = _detect_uv()
    if uv is not None:
        return f"{uv!s} --project {project_dir} run python -m larksnap.main service"
    return f"{_detect_python()} -m larksnap.main service"


def _build_user_line(user: str | None) -> str:
    if not user:
        return ""
    return f"User={user}\n"


def _sd_notify(message: str) -> None:
    """Best-effort ``sd_notify`` call.

    systemd's notification socket is exposed as ``$NOTIFY_SOCKET``. We
    use a tiny AF_UNIX write rather than depending on the
    ``sdnotify`` package – this keeps the Linux extras import-free
    at the stdlib level.
    """
    import socket

    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    try:
        if addr.startswith("@"):
            addr = "\0" + addr[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(message.encode("ascii"))
    except OSError as exc:  # pragma: no cover – depends on systemd
        _logger.debug("sd_notify failed: %s", exc)


def run_linux_service(config_path: str | None = None) -> None:
    """Run the LarkSnap service body on Linux.

    Wire SIGTERM/SIGINT to :meth:`ServiceRunner.request_stop`, emit
    ``READY=1`` once the controller is up, then block. systemd will
    send SIGTERM on ``systemctl stop larksnap``.
    """
    from larksnap.service.runner import run_service_blocking

    def _on_ready(runner: ServiceRunner) -> None:
        _sd_notify("READY=1")
        if runner.logger is not None:
            runner.logger.info("LarkSnap service READY (systemd notified)")

    # The first argument goes through the shared runner; the rest of
    # the function is just the platform glue.
    install_posix_stop_handlers(
        ServiceRunner(config_path=config_path),  # signal target
    )

    run_service_blocking(config_path=config_path, on_ready=_on_ready)


# ---------------------------------------------------------------------------
# Unit file generation / install / uninstall
# ---------------------------------------------------------------------------
def _render_unit(project_dir: Path, log_file: Path) -> str:
    """Render the unit file using the user-supplied ServiceConfig."""
    config = load_config()
    svc = config.service

    user_line = _build_user_line(svc.systemd_user)
    exec_start = _build_exec_start(project_dir)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    return UNIT_TEMPLATE.format(
        description=svc.description,
        type=svc.systemd_type,
        exec_start=exec_start,
        user_line=user_line,
        restart=svc.systemd_restart,
        log_file=str(log_file),
        wanted_by=svc.systemd_wanted_by,
    )


def _run_systemctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``systemctl`` and capture stdout/stderr."""
    return subprocess.run(  # noqa: S603 – we trust the args
        ["systemctl", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def install_linux_service() -> int:
    """Write the systemd unit file and reload the manager.

    Returns the process exit code (0 on success). Common failure
    modes (no root, no systemd) are surfaced as non-zero exit codes
    with a helpful message on stderr.
    """
    if os.geteuid() != 0:
        print(
            "Installing the systemd unit requires root. Re-run with sudo:",
            file=sys.stderr,
        )
        print(f"    sudo -E {' '.join(sys.argv)}", file=sys.stderr)
        return 1

    config = load_config()
    svc = config.service
    project_dir = Path.cwd()
    log_file = Path("/var/log/larksnap.log")

    unit_path = Path(svc.systemd_unit_path)
    unit_path.parent.mkdir(parents=True, exist_ok=True)

    unit_body = _render_unit(project_dir, log_file)
    unit_path.write_text(unit_body, encoding="utf-8")
    print(f"Wrote {unit_path}")

    reload = _run_systemctl(["daemon-reload"])
    if reload.returncode != 0:
        print(
            f"systemctl daemon-reload failed: {reload.stderr.strip()}",
            file=sys.stderr,
        )
        return reload.returncode

    enable = _run_systemctl(["enable", svc.name])
    if enable.returncode != 0:
        print(
            f"systemctl enable failed: {enable.stderr.strip()}",
            file=sys.stderr,
        )
        return enable.returncode

    print(f"Service '{svc.name}' installed. Start with:")
    print(f"    sudo systemctl start {svc.name}")
    return 0


def uninstall_linux_service() -> int:
    """Stop, disable, and remove the systemd unit."""
    if os.geteuid() != 0:
        print(
            "Uninstalling the systemd unit requires root. Re-run with sudo:",
            file=sys.stderr,
        )
        return 1

    config = load_config()
    svc = config.service
    unit_path = Path(svc.systemd_unit_path)

    # Tolerate missing unit file – uninstall should be idempotent.
    if unit_path.exists():
        _run_systemctl(["stop", svc.name])
        _run_systemctl(["disable", svc.name])

    if unit_path.exists():
        unit_path.unlink()
        print(f"Removed {unit_path}")

    _run_systemctl(["daemon-reload"])
    _run_systemctl(["reset-failed", svc.name])
    print(f"Service '{svc.name}' uninstalled.")
    return 0
