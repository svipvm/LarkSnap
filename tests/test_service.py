from larksnap.config.models import AppConfig
from larksnap.gateway.controller import GatewayController
from larksnap.service.tray import SystemTray
from larksnap.service.windows_service import install_service, uninstall_service


class TestWindowsService:
    def test_install_service_without_pywin32(self) -> None:
        install_service()

    def test_uninstall_service_without_pywin32(self) -> None:
        uninstall_service()


class TestSystemTray:
    def test_tray_creation(self) -> None:
        config = AppConfig()
        controller = GatewayController(config)
        tray = SystemTray(controller)
        assert tray is not None
