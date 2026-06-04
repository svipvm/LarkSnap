import logging

from larksnap.config.loader import load_config
from larksnap.gateway.controller import GatewayController
from larksnap.utils.logger import setup_logger

try:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil
except ImportError:
    win32serviceutil = None


if win32serviceutil is not None:

    class LarkSnapService(win32serviceutil.ServiceFramework):
        """Windows service wrapper for LarkSnap detection system."""

        _svc_name_ = "LarkSnap"
        _svc_display_name_ = "LarkSnap Detection Service"
        _svc_description_ = "Gateway-controlled object detection system"

        def __init__(self, args) -> None:
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
            self._controller: GatewayController | None = None
            self._logger: logging.Logger | None = None

        def SvcStop(self) -> None:  # noqa: N802
            """Handle service stop request."""
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.hWaitStop)
            if self._controller is not None:
                self._controller.stop()
            if self._logger:
                self._logger.info("LarkSnap service stopping...")

        def SvcDoRun(self) -> None:  # noqa: N802
            """Handle service run entry point."""
            try:
                config = load_config()
                self._logger = setup_logger(
                    level=config.logging.level,
                    log_format=config.logging.format,
                    file_path=config.logging.file_path,
                    max_bytes=config.logging.max_bytes,
                    backup_count=config.logging.backup_count,
                    console_output=False,
                )
                self._logger.info("LarkSnap service starting...")

                self._svc_name_ = config.service.name
                self._svc_display_name_ = config.service.display_name
                self._svc_description_ = config.service.description

                self._controller = GatewayController(config)
                self._controller.initialize()
                self._controller.start()

                self._logger.info("LarkSnap service started successfully")
                win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)

            except Exception as e:
                if self._logger:
                    self._logger.error("Service error: %s", e)
                servicemanager.LogErrorMsg(str(e))

            finally:
                if self._controller is not None:
                    self._controller.stop()


def install_service() -> None:
    """Install LarkSnap as a Windows service."""
    if win32serviceutil is None:
        print(
            "pywin32 is required for Windows service. Install with: pip install pywin32"
        )
        return
    win32serviceutil.HandleCommandLine(LarkSnapService, argv=["", "install"])
    print("LarkSnap service installed successfully.")


def uninstall_service() -> None:
    """Uninstall the LarkSnap Windows service."""
    if win32serviceutil is None:
        print(
            "pywin32 is required for Windows service. Install with: pip install pywin32"
        )
        return
    win32serviceutil.HandleCommandLine(LarkSnapService, argv=["", "remove"])
    print("LarkSnap service uninstalled successfully.")


def run_service() -> None:
    """Run the LarkSnap Windows service."""
    if win32serviceutil is None:
        print(
            "pywin32 is required for Windows service. Install with: pip install pywin32"
        )
        return
    win32serviceutil.HandleCommandLine(LarkSnapService)
