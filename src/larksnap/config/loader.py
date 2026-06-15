from pathlib import Path

import yaml

from larksnap.config.models import AppConfig
from larksnap.utils.exceptions import ConfigError


def load_config(config_path: str | None = None) -> AppConfig:
    """Load and validate application configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. If None, uses default path.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigError: If the file is not found, YAML parsing fails, or validation fails.
    """
    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
        )

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML configuration: {e}") from e

    if raw_config is None:
        raw_config = {}

    try:
        return AppConfig(**raw_config)
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e


def save_config(config: AppConfig, config_path: str) -> None:
    """Save application configuration to a YAML file.

    Args:
        config: AppConfig instance to save.
        config_path: Path to the YAML configuration file.

    Raises:
        ConfigError: If writing fails.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw = config.model_dump()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        raise ConfigError(f"Failed to save configuration: {e}") from e
