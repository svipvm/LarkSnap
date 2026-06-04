import argparse
import signal

from larksnap.config.loader import load_config
from larksnap.gateway.controller import GatewayController
from larksnap.service.tray import SystemTray
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

    subparsers.add_parser("run", help="Run the application with system tray")
    subparsers.add_parser("service", help="Run as Windows service")
    subparsers.add_parser("install", help="Install Windows service")
    subparsers.add_parser("uninstall", help="Uninstall Windows service")

    return parser


def run_with_tray(config_path: str | None) -> None:
    """Run the application with system tray UI."""
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
    controller.start()

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

    command = args.command or "run"
    config_path = args.config

    if command == "run":
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
