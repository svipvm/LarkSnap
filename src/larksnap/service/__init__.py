"""Cross-platform service layer for LarkSnap.

Public API
----------

- :func:`run_service_blocking` – shared service body used by every
  platform. Loads config, brings the :class:`GatewayController` up,
  blocks until a stop signal arrives, and tears it down cleanly.
- :func:`install_service` / :func:`uninstall_service` – dispatch to
  the platform-specific installer (Windows service registration,
  systemd unit file, etc.).
- :func:`current_service_platform` – returns the
  :class:`ServicePlatform` we'd use on this machine.

The Windows / Linux / generic-POSIX wrappers each live in their own
submodule but all delegate to ``runner.run_service_blocking`` so the
business logic is implemented exactly once.
"""

from __future__ import annotations

import sys

from larksnap.service.platform_utils import (
    ServicePlatform,
    current_service_platform,
)
from larksnap.service.runner import (
    ServiceRunner,
    ServiceRunnerConfig,
    run_service_blocking,
)
from larksnap.service.tray import SystemTray

__all__ = [
    "ServicePlatform",
    "ServiceRunner",
    "ServiceRunnerConfig",
    "SystemTray",
    "current_service_platform",
    "install_service",
    "run_service_blocking",
    "uninstall_service",
]


def install_service() -> int:
    """Install LarkSnap as a platform service.

    Returns the platform-specific exit code (0 on success). On a
    platform without first-class service support this prints a hint
    and returns 1 so the shell script can react.
    """
    platform = current_service_platform()

    if platform is ServicePlatform.WINDOWS:
        from larksnap.service.windows import install_windows_service

        return install_windows_service()

    if platform is ServicePlatform.LINUX:
        from larksnap.service.linux import install_linux_service

        return install_linux_service()

    print(
        f"Automatic service installation is not supported on {sys.platform}.",
        file=sys.stderr,
    )
    print(
        "On macOS / other BSDs, run 'larksnap tray' from a user launch "
        "agent or use `larksnap service` from a manual supervisor.",
        file=sys.stderr,
    )
    return 1


def uninstall_service() -> int:
    """Uninstall the LarkSnap platform service.

    Returns the platform-specific exit code (0 on success). On a
    platform without first-class service support this is a no-op that
    returns 0.
    """
    platform = current_service_platform()

    if platform is ServicePlatform.WINDOWS:
        from larksnap.service.windows import uninstall_windows_service

        return uninstall_windows_service()

    if platform is ServicePlatform.LINUX:
        from larksnap.service.linux import uninstall_linux_service

        return uninstall_linux_service()

    return 0
