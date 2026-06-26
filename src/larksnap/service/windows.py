"""Windows service integration.

The service is implemented with ``pywin32``'s
``win32serviceutil.ServiceFramework``. The class delegates to
:mod:`larksnap.service.runner` so the controller lifecycle is shared
with the Linux implementation – only the SCM plumbing differs.

Two things to be aware of:

1. The SCM dispatches ``SvcDoRun`` on a dedicated thread. We use
   :class:`ServiceRunner` so the controller / logger setup matches
   every other platform.
2. The SCM asks the service to stop via ``SvcStop``; we set the
   internal stop event and rely on the runner's ``stop()`` for
   teardown.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from larksnap.service.runner import ServiceRunner

if TYPE_CHECKING:
    pass

_logger = logging.getLogger("larksnap.service.windows")

try:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil
except ImportError:  # pragma: no cover – Windows only
    servicemanager = None  # type: ignore[assignment]
    win32event = None  # type: ignore[assignment]
    win32service = None  # type: ignore[assignment]
    win32serviceutil = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ServiceFramework subclass – only defined when pywin32 is importable.
# ---------------------------------------------------------------------------
if win32serviceutil is not None:

    class LarkSnapService(win32serviceutil.ServiceFramework):  # type: ignore[misc]
        """Windows service wrapper for LarkSnap.

        The class metadata (``_svc_*_``) is filled in at install time
        from :class:`larksnap.config.models.ServiceConfig`, so the
        same code path serves the default name and any custom name
        the user configured.
        """

        _svc_name_ = "LarkSnap"
        _svc_display_name_ = "LarkSnap Detection Service"
        _svc_description_ = "Gateway-controlled object detection system"

        def __init__(self, args: Any) -> None:
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
            # hWaitStop must outlive SvcDoRun's wait, so we keep a
            # process-level reference to the runner and set the event
            # from SvcStop. We don't store the runner here because
            # SvcDoRun creates it and SvcStop must remain reentrant.
            self._runner: ServiceRunner | None = None

        # ------------------------------------------------------------------
        # SCM callbacks
        # ------------------------------------------------------------------
        def SvcStop(self) -> None:  # noqa: N802 – pywin32 API name
            """Handle the SCM stop request."""
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.hWaitStop)
            if self._runner is not None:
                self._runner.request_stop()
            if self._runner is not None and self._runner.logger is not None:
                self._runner.logger.info("LarkSnap service stop requested")

        def SvcDoRun(self) -> None:  # noqa: N802 – pywin32 API name
            """Bring the service up and block until SvcStop is called."""
            try:
                self._runner = ServiceRunner()
                self._runner.start()

                # Pull service metadata from the loaded config so
                # SCM properties match the user-configured values.
                if self._runner.controller is not None:
                    cfg = self._runner.controller._config  # noqa: SLF001
                    self._svc_name_ = cfg.service.name
                    self._svc_display_name_ = cfg.service.display_name
                    self._svc_description_ = cfg.service.description

                # Tell the SCM we're running, then wait.
                servicemanager.LogMsg(
                    servicemanager.EVENTLOG_INFORMATION_TYPE,
                    servicemanager.PYS_SERVICE_STARTED,
                    (self._svc_name_, ""),
                )

                # Block until SvcStop sets the event OR the runner's
                # internal stop flag fires – whichever comes first.
                wait_timeout_ms = 1000
                while True:
                    rc = win32event.WaitForSingleObject(
                        self.hWaitStop, wait_timeout_ms
                    )
                    if rc == win32event.WAIT_OBJECT_0:
                        break
                    if self._runner.wait_for_stop(timeout=0):
                        break

            except Exception as exc:  # noqa: BLE001
                _logger.exception("Service error: %s", exc)
                servicemanager.LogErrorMsg(str(exc))
                raise

            finally:
                if self._runner is not None:
                    self._runner.stop()


# ---------------------------------------------------------------------------
# Install / uninstall / run helpers (CLI subcommands).
# ---------------------------------------------------------------------------
def install_windows_service() -> int:
    """Register LarkSnap with the Windows SCM."""
    if win32serviceutil is None:
        print(
            "pywin32 is required for Windows service. "
            "Install with: uv pip install pywin32",
            file=sys.stderr,
        )
        return 1
    win32serviceutil.HandleCommandLine(LarkSnapService, argv=["", "install"])
    print("LarkSnap service installed successfully.")
    return 0


def uninstall_windows_service() -> int:
    """Unregister LarkSnap from the Windows SCM."""
    if win32serviceutil is None:
        print(
            "pywin32 is required for Windows service. "
            "Install with: uv pip install pywin32",
            file=sys.stderr,
        )
        return 1
    win32serviceutil.HandleCommandLine(LarkSnapService, argv=["", "remove"])
    print("LarkSnap service uninstalled successfully.")
    return 0


def run_windows_service() -> int:
    """Entry point used by ``python -m larksnap.main service`` on Windows."""
    if win32serviceutil is None:
        print(
            "pywin32 is required for Windows service. "
            "Install with: uv pip install pywin32",
            file=sys.stderr,
        )
        return 1
    win32serviceutil.HandleCommandLine(LarkSnapService)
    return 0
