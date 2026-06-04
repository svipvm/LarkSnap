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
        assert config.detector.type == "mock"
        assert config.detector.confidence_threshold == 0.5
        assert config.notifier.type == "feishu"
        assert config.gateway.event_queue_size == 100
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
