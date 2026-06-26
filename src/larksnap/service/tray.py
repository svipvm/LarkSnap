import logging
import threading
from collections.abc import Callable

from larksnap.gateway.controller import GatewayController

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    pystray = None


class SystemTray:
    """System tray icon for LarkSnap with status display and controls."""

    def __init__(
        self,
        controller: GatewayController,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the system tray with gateway controller."""
        self._controller = controller
        self._on_quit = on_quit
        self._logger = logging.getLogger("larksnap.tray")
        self._icon: pystray.Icon | None = None
        self._shutdown_event = threading.Event()

    def _create_icon_image(self) -> "Image.Image":
        width = 64
        height = 64
        image = Image.new("RGB", (width, height), color=(0, 120, 215))
        dc = ImageDraw.Draw(image)
        dc.rectangle([16, 16, 48, 48], fill=(255, 255, 255))
        dc.rectangle([20, 20, 44, 44], fill=(0, 120, 215))
        return image

    def _get_menu(self) -> tuple:
        status_text = "Running" if self._controller.is_running else "Stopped"

        return (
            pystray.MenuItem(f"Status: {status_text}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit_action),
        )

    def _on_quit_action(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        self._controller.stop()
        icon.stop()
        self._shutdown_event.set()
        if self._on_quit is not None:
            self._on_quit()
        self._logger.info("Application quit from tray")

    def run(self) -> None:
        """Start the system tray icon in a background thread and block
        until shutdown."""
        if pystray is None:
            self._logger.error("pystray and Pillow are required for system tray")
            return

        image = self._create_icon_image()
        menu = self._get_menu()
        self._icon = pystray.Icon("LarkSnap", image, "LarkSnap", menu)

        tray_thread = threading.Thread(target=self._icon.run, daemon=True)
        tray_thread.start()

        # Use short-interval polling instead of infinite wait so that
        # SIGINT (Ctrl+C) signal handler can execute on the main thread.
        while not self._shutdown_event.wait(timeout=0.5):
            pass

    def stop(self) -> None:
        """Stop the system tray icon."""
        self._shutdown_event.set()
        if self._icon is not None:
            self._icon.stop()
