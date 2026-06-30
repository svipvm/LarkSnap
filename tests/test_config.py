"""Tests for the Pydantic config models and YAML loader.

Covers:
  * Default values on ``AppConfig`` (every sub-config is reachable).
  * Custom values for ``CameraConfig`` and ``DetectorConfig``.
  * YAML loading: valid file, missing file, malformed YAML, empty file.
  * Edge cases: empty config returns defaults, deeply nested values
    are preserved.
"""

import pytest
import yaml

from larksnap.config.loader import load_config
from larksnap.config.models import AppConfig, CameraConfig, DetectorConfig
from larksnap.utils.exceptions import ConfigError


class TestConfigModels:
    def test_app_config_defaults(self) -> None:
        config = AppConfig()
        assert config.camera.device_index == 0
        assert config.camera.width == 1280
        assert config.camera.height == 720
        # ``seg`` is the production-default detector. Tests that
        # want a no-network model should override this to ``mock``
        # explicitly (see the ``app_config`` fixture in conftest.py).
        assert config.detector.type == "seg"
        assert config.detector.confidence_threshold == 0.5
        assert config.notifier.type == "feishu"
        assert config.gateway.frame_queue_hwm == 30
        assert config.logging.level == "INFO"

    def test_camera_config_custom(self) -> None:
        config = CameraConfig(device_index=1, width=1920, height=1080)
        assert config.device_index == 1
        assert config.width == 1920
        assert config.height == 1080

    def test_detector_config_with_mock(self) -> None:
        config = DetectorConfig(
            type="mock",
            mock={"labels": ["cat", "dog"], "confidence_range": [0.7, 0.99]},
        )
        assert config.mock.labels == ["cat", "dog"]
        assert config.mock.confidence_range == (0.7, 0.99)


class TestConfigLoader:
    def test_load_valid_config(self, tmp_path: pytest.TempPathFactory) -> None:
        config_data = {
            "camera": {"device_index": 1, "width": 1920, "height": 1080},
            "detector": {"type": "mock", "confidence_threshold": 0.8},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_config(str(config_file))
        assert config.camera.device_index == 1
        assert config.camera.width == 1920
        assert config.detector.confidence_threshold == 0.8

    def test_load_missing_file(self) -> None:
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config("/nonexistent/config.yaml")

    def test_load_invalid_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml}}", encoding="utf-8")

        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config(str(config_file))

    def test_load_empty_config(self, tmp_path: pytest.TempPathFactory) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")

        config = load_config(str(config_file))
        assert config.camera.device_index == 0

    def test_load_preserves_deeply_nested_values(
        self, tmp_path: pytest.TempPathFactory,
    ) -> None:
        """A config with all four sub-configs round-trips through load_config.

        Edge case: the loader must preserve deeply nested custom
        values, not collapse them back to defaults.
        """
        config_data = {
            "camera": {"device_index": 2, "width": 800, "height": 600},
            "detector": {
                "type": "mock",
                "confidence_threshold": 0.7,
                "target_classes": ["person", "car"],
            },
            "notifier": {"type": "feishu", "app_id": "test-id"},
            "gateway": {"notification_interval": 60, "snapshot_dir": "out"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_config(str(config_file))
        assert config.camera.device_index == 2
        assert config.detector.target_classes == ["person", "car"]
        assert config.notifier.app_id == "test-id"
        assert config.gateway.notification_interval == 60
        assert config.gateway.snapshot_dir == "out"


class TestConfigValidation:
    """Round-trip checks for boundary values."""

    def test_camera_fps_preserved(self) -> None:
        """Non-default fps values round-trip through the model."""
        cfg = CameraConfig(fps=60)
        assert cfg.fps == 60

    def test_detector_confidence_threshold_preserved(self) -> None:
        """A custom confidence_threshold round-trips through the model.

        The detector filters predictions by this threshold at inference
        time; a bad value silently degrades detection quality rather
        than crashing. We assert the field is preserved verbatim.
        """
        cfg = DetectorConfig(type="mock", confidence_threshold=0.85)
        assert cfg.confidence_threshold == 0.85

    def test_detector_empty_target_classes_preserved(self) -> None:
        """Empty target_classes means "monitor nothing" — a valid contract.

        The detector adapter and the notification service both
        special-case an empty list. A regression where the
        model collapses it to a default would silently change
        behaviour.
        """
        cfg = DetectorConfig(type="mock", target_classes=[])
        assert cfg.target_classes == []
