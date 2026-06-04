import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "larksnap",
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    file_path: str | None = None,
    max_bytes: int = 10485760,
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """Set up and configure the application logger.

    Args:
        name: Logger name.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Log message format string.
        file_path: Optional path to log file. If None, no file logging.
        max_bytes: Maximum size of each log file in bytes.
        backup_count: Number of backup log files to keep.
        console_output: Whether to output logs to console.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(log_format)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_path:
        log_dir = os.path.dirname(file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
