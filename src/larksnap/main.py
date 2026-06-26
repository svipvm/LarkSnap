"""LarkSnap application entry point.

Supports multiple run modes:
  - qt:     PySide6 GUI mode (default)
  - tray:   System tray mode (headless)
  - service: Windows service mode
"""

import argparse
import logging
import signal
import sys

from larksnap.config.loader import load_config
from larksnap.gateway.controller import GatewayController
from larksnap.utils.logger import setup_logger


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="larksnap",
        description="Gateway-controlled object detection system",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("qt", help="Run with PySide6 GUI (default)")
    subparsers.add_parser("tray", help="Run with system tray (headless)")
    subparsers.add_parser("service", help="Run as Windows service")
    subparsers.add_parser("install", help="Install Windows service")
    subparsers.add_parser("uninstall", help="Uninstall Windows service")

    return parser


def run_with_qt(config_path: str | None) -> None:
    """Run the application with PySide6 GUI.

    Startup order is tuned for an "instant launch" feel:

    1. Load config + set up logger
    2. Create ``QApplication`` and ``MainWindow`` (cheap, <50 ms)
    3. ``window.show()`` — user sees the UI immediately with a
       "正在初始化摄像头" overlay
    4. Camera initialization runs on a background daemon thread
       (probing 3 frames + MSMF/DSHOW/ANY backend fallback can take
       1–4 s on Windows; this would otherwise block the event loop)
    5. When the camera is ready the pipeline publishes ``CAMERA_OPENED``;
       the main window hides the loading overlay and starts the preview
    """
    import atexit
    import threading

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from larksnap.ui.main_window import MainWindow

    config = load_config(config_path)
    logger = setup_logger(
        level=config.logging.level,
        log_format=config.logging.format,
        file_path=config.logging.file_path,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        console_output=config.logging.console_output,
    )

    logger.info("Starting LarkSnap (Qt GUI mode)...")

    # Suppress noisy OpenCV/MSMF error logs (they will be translated to
    # user-friendly messages in the camera adapter)
    import logging as _logging
    for _name in ("cv2", "opencv"):
        _logging.getLogger(_name).setLevel(_logging.ERROR)

    app = QApplication(sys.argv)
    app.setApplicationName("LarkSnap")
    app.setQuitOnLastWindowClosed(True)

    controller = GatewayController(config)

    # Ensure controller.stop() runs even if app.exec() is interrupted
    atexit.register(controller.stop)

    window = MainWindow(controller, config, config_path=config_path)
    window.show()  # ← UI is up immediately, loading overlay shown

    # Kick off camera initialization in the background so the event
    # loop stays responsive. ``_async_init_camera`` will publish a
    # ``CAMERA_OPENED`` (or ``CAMERA_FAILED``) event that the main
    # window is already listening for.
    def _async_init_camera() -> None:
        try:
            controller.initialize()
            # start_preview must be called on the Qt main thread
            QTimer.singleShot(0, window.start_preview)
        except Exception as e:
            logger.error("Background camera init failed: %s", e)
            # CAMERA_FAILED event was already published by the
            # controller; the main window will show the error dialog.

    init_thread = threading.Thread(target=_async_init_camera, daemon=True)
    init_thread.start()
    logger.info("LarkSnap UI shown; camera init running in background")

    exit_code = app.exec()

    controller.stop()
    atexit.unregister(controller.stop)
    logger.info("LarkSnap exited (code=%d)", exit_code)
    sys.exit(exit_code)


def run_with_tray(config_path: str | None) -> None:
    """Run the application with system tray UI."""
    from larksnap.service.tray import SystemTray

    config = load_config(config_path)
    logger = setup_logger(
        level=config.logging.level,
        log_format=config.logging.format,
        file_path=config.logging.file_path,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        console_output=config.logging.console_output,
    )

    logger.info("Starting LarkSnap...")

    controller = GatewayController(config)
    controller.initialize()

    tray = SystemTray(controller)

    def _signal_handler(sig: int, frame: object) -> None:
        logger.info("Received shutdown signal, stopping...")
        tray.stop()

    signal.signal(signal.SIGINT, _signal_handler)

    tray.run()

    controller.stop()
    logger.info("LarkSnap exited.")


def run_as_service(config_path: str | None) -> None:
    """Run the application as a Windows service."""
    from larksnap.service.windows_service import run_service

    run_service()


def install_service(config_path: str | None) -> None:
    """Install the application as a Windows service."""
    from larksnap.service.windows_service import install_service

    install_service()


def uninstall_service(config_path: str | None) -> None:
    """Uninstall the Windows service."""
    from larksnap.service.windows_service import uninstall_service

    uninstall_service()


def main() -> None:
    """Application entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    command = args.command or "qt"
    config_path = args.config

    if command == "qt":
        run_with_qt(config_path)
    elif command == "tray":
        run_with_tray(config_path)
    elif command == "service":
        run_as_service(config_path)
    elif command == "install":
        install_service(config_path)
    elif command == "uninstall":
        uninstall_service(config_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
