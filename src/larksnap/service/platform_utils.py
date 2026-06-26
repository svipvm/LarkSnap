"""Platform detection helpers for the service layer.

The values are intentionally coarse – "linux" covers every
distribution that can run systemd (Ubuntu 20.04+, CentOS 8+, Debian
11+, etc.) because we only need the integration shape, not the
distribution identifier.
"""

from __future__ import annotations

import enum
import sys


class ServicePlatform(enum.Enum):
    """Supported service integration targets."""

    WINDOWS = "windows"
    LINUX = "linux"
    OTHER = "other"


def current_service_platform() -> ServicePlatform:
    """Return the service platform for the current interpreter.

    Detection rules:

    - ``sys.platform == "win32"`` → Windows (Win10+ supported).
    - ``sys.platform.startswith("linux")`` → Linux. We assume systemd
      is available; if it isn't, :func:`run_service_blocking` falls
      back to plain POSIX signal handling.
    - Everything else (macOS, BSD, …) → ``OTHER`` – the CLI prints
      a hint but still allows ``larksnap service`` to run as a plain
      foreground process.
    """
    if sys.platform == "win32":
        return ServicePlatform.WINDOWS
    if sys.platform.startswith("linux"):
        return ServicePlatform.LINUX
    return ServicePlatform.OTHER


def is_windows() -> bool:
    """Return True when running on Windows."""
    return sys.platform == "win32"


def is_linux() -> bool:
    """Return True when running on Linux."""
    return sys.platform.startswith("linux")
