from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    """Camera device configuration."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    capture_interval: float = 1.0


class MockDetectorConfig(BaseModel):
    """Mock detector configuration for testing."""

    labels: list[str] = Field(default_factory=lambda: ["person", "car", "dog"])
    confidence_range: tuple[float, float] = (0.6, 0.95)
    delay_seconds: float = 0.1


class YOLOSegConfig(BaseModel):
    """YOLO Segmentation ONNX Runtime detector configuration."""

    model_path: str = "models/yolo26n-seg.onnx"
    img_size: int = 640
    iou_thres: float = 0.45
    max_det: int = 300
    provider: str = "cpu"


class DetectorConfig(BaseModel):
    """AI detector configuration."""

    type: str = "yolo_seg"
    model_path: str = ""
    confidence_threshold: float = 0.5
    target_classes: list[str] = Field(default_factory=lambda: ["person"])
    mock: MockDetectorConfig = Field(default_factory=MockDetectorConfig)
    yolo_seg: YOLOSegConfig = Field(default_factory=YOLOSegConfig)


class RetryConfig(BaseModel):
    """Notification retry configuration."""

    max_retries: int = 3
    retry_interval: int = 5


class NotifierConfig(BaseModel):
    """Notification service configuration."""

    type: str = "feishu"
    webhook_url: str = ""
    app_id: str = ""
    app_secret: str = ""
    chat_id: str = ""
    message_template: str = (
        "[LarkSnap] 检测到 {label}，置信度: {confidence:.2%}，时间: {timestamp}"
    )
    send_image: bool = True
    retry: RetryConfig = Field(default_factory=RetryConfig)


class GatewayConfig(BaseModel):
    """Gateway controller configuration."""

    event_queue_size: int = 100
    process_interval: float = 5.0
    detection_strategy: str = "interval"
    notification_cooldown: int = 30
    snapshot_dir: str = "snapshots"


class ServiceConfig(BaseModel):
    """Windows service configuration."""

    name: str = "LarkSnap"
    display_name: str = "LarkSnap Detection Service"
    description: str = "Gateway-controlled object detection system"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str | None = "logs/larksnap.log"
    max_bytes: int = 10485760
    backup_count: int = 5
    console_output: bool = True


class AppConfig(BaseModel):
    """Application configuration aggregating all sub-configs."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    notifier: NotifierConfig = Field(default_factory=NotifierConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
