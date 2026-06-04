import numpy as np
import pytest

from larksnap.adapters.base import BaseAdapter
from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.adapters.detector.interface import BBox, DetectionResult, DetectorAdapter
from larksnap.adapters.detector.mock_adapter import MockDetectorAdapter
from larksnap.adapters.notifier.interface import NotificationMessage, NotifierAdapter
from larksnap.config.models import DetectorConfig
from larksnap.utils.exceptions import DetectorError


class TestBaseAdapter:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            BaseAdapter()


class TestCameraAdapter:
    def test_cannot_instantiate_interface(self) -> None:
        with pytest.raises(TypeError):
            CameraAdapter()


class TestDetectorAdapter:
    def test_cannot_instantiate_interface(self) -> None:
        with pytest.raises(TypeError):
            DetectorAdapter()


class TestMockDetector:
    def test_load_model(self) -> None:
        config = DetectorConfig(type="mock")
        detector = MockDetectorAdapter(config)
        detector.load_model()

    def test_detect_returns_results(self) -> None:
        config = DetectorConfig(
            type="mock",
            mock={
                "labels": ["person"],
                "confidence_range": [0.8, 0.9],
                "delay_seconds": 0,
            },
        )
        detector = MockDetectorAdapter(config)
        detector.load_model()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, DetectionResult)
            assert isinstance(r.label, str)
            assert 0.0 <= r.confidence <= 1.0
            assert isinstance(r.bbox, BBox)

    def test_detect_without_load_raises(self) -> None:
        config = DetectorConfig(type="mock")
        detector = MockDetectorAdapter(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(DetectorError):
            detector.detect(frame)

    def test_unload_model(self) -> None:
        config = DetectorConfig(type="mock")
        detector = MockDetectorAdapter(config)
        detector.load_model()
        detector.unload_model()


class TestNotifierAdapter:
    def test_cannot_instantiate_interface(self) -> None:
        with pytest.raises(TypeError):
            NotifierAdapter()


class TestNotificationMessage:
    def test_create_message(self) -> None:
        msg = NotificationMessage(
            title="Test",
            content="Test content",
            label="person",
            confidence=0.95,
            timestamp="2024-01-01 00:00:00",
        )
        assert msg.title == "Test"
        assert msg.label == "person"
        assert msg.confidence == 0.95
        assert msg.snapshot_path is None
