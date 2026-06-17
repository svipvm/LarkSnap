"""Gateway controller — thin orchestrator.

Composes Pipeline, NotificationService, and Recorder.
Delegates all domain logic to specialized modules.
Does NOT directly access scattered config — each module owns its own config.
"""

from __future__ import annotations

import logging

from larksnap.adapters.camera.interface import CameraAdapter
from larksnap.adapters.detector.interface import DetectionResult, DetectorAdapter
from larksnap.adapters.notifier.feishu_adapter import FeishuNotifierAdapter
from larksnap.adapters.notifier.feishu_ws_client import CommandHandler, FeishuWSClient
from larksnap.adapters.notifier.interface import NotifierAdapter
from larksnap.adapters.recorder.video_recorder import VideoRecorderAdapter
from larksnap.adapters.registry import camera_registry, detector_registry, notifier_registry
from larksnap.config.models import AppConfig
from larksnap.gateway.event_bus import Event, EventBus, EventType
from larksnap.gateway.notification_service import NotificationService, NotificationServiceConfig
from larksnap.gateway.pipeline import Pipeline, PipelineConfig
from larksnap.utils.exceptions import CameraError, GatewayError


class GatewayController:
    """Thin orchestrator that composes Pipeline + NotificationService + Recorder.

    Responsibilities:
      - Create adapters via registry
      - Wire modules together
      - Expose public API for UI
      - Handle Feishu commands

    Does NOT:
      - Manage ZMQ infrastructure (delegated to Pipeline)
      - Handle notification cooldown/snapshot (delegated to NotificationService)
      - Filter detection results (delegated to DetectorAdapter)
    """

    def __init__(self, config: AppConfig, event_bus: EventBus | None = None) -> None:
        self._config = config
        self._event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("larksnap.gateway")

        # Adapters (created in initialize())
        self._camera: CameraAdapter | None = None
        self._detector: DetectorAdapter | None = None
        self._notifier: NotifierAdapter | None = None
        self._recorder: VideoRecorderAdapter | None = None
        self._ws_client: FeishuWSClient | None = None

        # Composed services
        self._pipeline: Pipeline | None = None
        self._notification_service: NotificationService | None = None

        # State
        self._running = False
        self._camera_failed = False

    # ── Public Properties ─────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._pipeline.is_paused if self._pipeline else False

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.is_recording

    @property
    def detection_count(self) -> int:
        return self._pipeline.detection_count if self._pipeline else 0

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def producer_fps(self) -> float:
        return self._pipeline.producer_fps if self._pipeline else 0.0

    @property
    def notification_enabled(self) -> bool:
        return self._notification_service.notification_enabled if self._notification_service else False

    def get_latest_frame(self):
        return self._pipeline.get_latest_frame() if self._pipeline else None

    def get_latest_results(self) -> list[DetectionResult]:
        return self._pipeline.get_latest_results() if self._pipeline else []

    # ── Lifecycle ──────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Initialize all adapters and compose services."""
        try:
            self._logger.info("Initializing gateway controller...")

            # Ensure adapter modules are imported so @register decorators execute
            self._ensure_adapters_registered()

            # Create adapters via registry
            self._camera = camera_registry.create(
                self._config.camera.type, self._config.camera
            )
            self._detector = detector_registry.create(
                self._config.detector.type, self._config.detector
            )
            self._notifier = notifier_registry.create(
                self._config.notifier.type, self._config.notifier
            )
            self._recorder = VideoRecorderAdapter(
                output_dir=self._config.recorder.output_dir,
                fps=self._config.recorder.fps,
                codec=self._config.recorder.codec,
                frame_queue_url=Pipeline.FRAME_QUEUE_URL,
            )

            # Initialize adapters
            try:
                self._camera.initialize()
            except CameraError as e:
                self._logger.error("Camera initialization failed: %s", e)
                self._event_bus.publish(Event(
                    type=EventType.CAMERA_FAILED,
                    data={"error": str(e), "device_index": self._config.camera.device_index},
                    source="gateway",
                ))
                self._camera_failed = True
                raise

            self._camera_failed = False
            self._detector.initialize()
            self._notifier.initialize()
            self._recorder.initialize()

            # Compose Pipeline (owns ZMQ + detection flow)
            pipeline_config = PipelineConfig(
                frame_queue_hwm=self._config.gateway.frame_queue_hwm,
                frame_queue_policy=self._config.gateway.frame_queue_policy,
            )
            self._pipeline = Pipeline(
                camera=self._camera,
                detector=self._detector,
                config=pipeline_config,
                event_bus=self._event_bus,
            )

            # Compose NotificationService (owns cooldown + snapshot + dispatch)
            notif_config = NotificationServiceConfig(
                notification_interval=self._config.gateway.notification_interval,
                snapshot_dir=self._config.gateway.snapshot_dir,
                message_template=self._config.notifier.message_template,
            )
            self._notification_service = NotificationService(
                config=notif_config,
                notifier=self._notifier,
                event_bus=self._event_bus,
            )

            # Wire pipeline → notification service
            self._pipeline.set_on_results(self._notification_service.handle_results)

            # Start WebSocket command listener
            if self._config.notifier.app_id and self._config.notifier.app_secret:
                self._init_ws_client()

            # If chat_id is already configured (persisted), publish event for UI
            if self._config.notifier.chat_id:
                self._event_bus.publish(
                    Event(
                        type=EventType.CHAT_ID_OBTAINED,
                        data=self._config.notifier.chat_id,
                        source="gateway",
                    )
                )

            self._logger.info("Gateway controller initialized successfully")
        except Exception as e:
            self._release_all()
            raise GatewayError(f"Failed to initialize gateway: {e}") from e

    def start(self) -> None:
        """Start the gateway pipeline."""
        if self._running:
            self._logger.warning("Gateway is already running")
            return

        self._running = True
        self._pipeline.start()
        self._event_bus.publish(Event(type=EventType.SYSTEM_STARTED, source="gateway"))
        self._logger.info("Gateway controller started")

    def stop(self) -> None:
        """Stop the gateway and release all resources."""
        if not self._running:
            return

        self._running = False

        if self._pipeline is not None:
            self._pipeline.stop()
        if self._ws_client is not None:
            self._ws_client.stop()
            self._ws_client = None

        self._release_all()
        self._event_bus.publish(Event(type=EventType.SYSTEM_STOPPED, source="gateway"))
        self._logger.info("Gateway controller stopped")

    def pause(self) -> None:
        if self._pipeline is not None:
            self._pipeline.pause()

    def resume(self) -> None:
        if self._pipeline is not None:
            self._pipeline.resume()

    def start_recording(self) -> None:
        if self._recorder is not None and self._running:
            self._recorder.start_recording(context=None)
            self._event_bus.publish(
                Event(type=EventType.SYSTEM_STARTED, data="recording", source="recorder")
            )

    def stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.stop_recording()

    # ── Private ────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_adapters_registered() -> None:
        """Import adapter modules to trigger @register decorators."""
        import larksnap.adapters.camera.opencv_adapter  # noqa: F401
        import larksnap.adapters.detector.mock_adapter  # noqa: F401
        import larksnap.adapters.detector.seg_adapter  # noqa: F401
        import larksnap.adapters.notifier.feishu_adapter  # noqa: F401

    def _init_ws_client(self) -> None:
        try:
            self._ws_client = FeishuWSClient(
                config=self._config.notifier,
                on_command=self._handle_command,
            )
            self._ws_client.start()
        except Exception as e:
            self._logger.warning(
                "Feishu WS client failed to start (commands disabled): %s", e
            )
            self._ws_client = None

    def _handle_command(self, cmd: CommandHandler) -> None:
        self._logger.info("Processing command: /%s", cmd.name)

        if cmd.chat_id and isinstance(self._notifier, FeishuNotifierAdapter):
            self._notifier.set_chat_id(cmd.chat_id)

        if cmd.name == "init":
            # /init: just obtain chat_id (already done above)
            self._logger.info("Init command received, chat_id obtained")
            self._event_bus.publish(
                Event(type=EventType.CHAT_ID_OBTAINED, data=cmd.chat_id, source="gateway")
            )
            # Send confirmation message back
            if isinstance(self._notifier, FeishuNotifierAdapter):
                self._notifier.send_text("[LarkSnap] 初始化成功！发送 /start 开始通知，/stop 停止通知")
        elif cmd.name == "start":
            # /start: enable notification dispatch
            if self._notification_service is not None:
                self._notification_service.enable_notification()
            # Send confirmation
            if isinstance(self._notifier, FeishuNotifierAdapter):
                self._notifier.send_text("[LarkSnap] 通知已开启，检测到目标将推送告警")
        elif cmd.name == "stop":
            # /stop: disable notification dispatch
            if self._notification_service is not None:
                self._notification_service.disable_notification()
            # Send confirmation
            if isinstance(self._notifier, FeishuNotifierAdapter):
                self._notifier.send_text("[LarkSnap] 通知已关闭，发送 /start 可重新开启")
        elif cmd.name == "pause":
            if self._running and not self.is_paused:
                self.pause()
        elif cmd.name == "resume":
            if self._running and self.is_paused:
                self.resume()
        elif cmd.name == "status":
            status = "running" if self._running else "stopped"
            if self._running and self.is_paused:
                status = "paused"
            notif = "enabled" if (self._notification_service and self._notification_service.notification_enabled) else "disabled"
            self._logger.info("Gateway status: %s, notification: %s", status, notif)
        elif cmd.name == "help":
            self._logger.info(
                "Available commands: /init, /start, /stop, /pause, /resume, /status, /help"
            )

    def _release_all(self) -> None:
        adapters = [self._notifier, self._detector, self._camera, self._recorder]
        for adapter in adapters:
            if adapter is not None:
                try:
                    adapter.stop()
                    adapter.release()
                except Exception as e:
                    self._logger.error("Error releasing adapter: %s", e)

        self._notifier = None
        self._detector = None
        self._camera = None
        self._recorder = None
        self._pipeline = None
        self._notification_service = None
