"""LarkSnap main window with PySide6 — immersive video-centric UI.

Main page is full video stream only. All controls, stats, and settings
are accessed through the navigation menu bar.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    QSize,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QSystemTrayIcon,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from larksnap.adapters.detector.interface import DetectionResult
from larksnap.config.loader import save_config
from larksnap.config.models import AppConfig
from larksnap.gateway.controller import GatewayController
from larksnap.gateway.event_bus import Event, EventType

# ─── Global Stylesheet ───────────────────────────────────────────────

DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #0f0f1a;
}
QWidget {
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    font-size: 13px;
    color: #8888aa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}
QLabel {
    color: #c0c0d0;
}
QPushButton {
    background-color: #1e1e32;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 8px 18px;
    color: #c0c0d0;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2a2a44;
    border-color: #4a4a6e;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #3a3a5e;
}
QPushButton:disabled {
    background-color: #14141e;
    color: #555566;
    border-color: #1a1a2a;
}
QStatusBar {
    background-color: transparent;
    color: #555570;
    font-size: 11px;
}
QMenuBar {
    background-color: rgba(15, 15, 26, 200);
    border-bottom: 1px solid #1a1a2e;
    spacing: 4px;
    padding: 2px;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 4px;
    color: #8888aa;
}
QMenuBar::item:selected {
    background-color: #1e1e32;
    color: #ffffff;
}
QMenu {
    background-color: #14141e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item {
    padding: 7px 28px 7px 16px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #2a2a44;
}
QMenu::separator {
    height: 1px;
    background: #2a2a3e;
    margin: 4px 8px;
}
QTabWidget::pane {
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    background-color: #0f0f1a;
}
QTabBar::tab {
    background-color: #14141e;
    border: 1px solid #2a2a3e;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #8888aa;
}
QTabBar::tab:selected {
    background-color: #1e1e32;
    color: #ffffff;
    border-bottom-color: #1e1e32;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1a1a2e;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 5px 8px;
    color: #e0e0e0;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #6a6aae;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #3a3a5e;
    background-color: #1a1a2e;
}
QCheckBox::indicator:checked {
    background-color: #6a6aae;
    border-color: #6a6aae;
}
"""


# ─── Video Preview (Full-screen immersive) ────────────────────────────

class VideoPreviewWidget(QWidget):
    """Full-area video preview with detection overlay and floating HUD."""

    _PALETTE = np.random.RandomState(42).randint(50, 255, size=(80, 3), dtype=np.uint8)
    _MASK_ALPHA = 0.45

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._fps: float = 0.0
        self._detection_count: int = 0
        self._is_recording: bool = False
        self._is_running: bool = False
        self._is_paused: bool = False
        self._rec_pulse_phase: float = 0.0
        self._show_hud: bool = True

        # HUD auto-hide timer
        self._hud_timer = QTimer(self)
        self._hud_timer.setInterval(3000)
        self._hud_timer.setSingleShot(True)
        self._hud_timer.timeout.connect(self._hide_hud)

        self.setMouseTracking(True)

    def show_hud_temporarily(self) -> None:
        """Show HUD and auto-hide after timeout."""
        self._show_hud = True
        self._hud_timer.start()
        self.update()

    def _hide_hud(self) -> None:
        self._show_hud = False
        self.update()

    def update_frame(self, frame: np.ndarray, results: list[DetectionResult] | None = None) -> None:
        display = frame.copy()

        if results:
            overlay = display.copy()
            fh, fw = display.shape[:2]
            for result in results:
                if result.mask is not None and result.mask.size > 0:
                    mask = result.mask
                    mh, mw = mask.shape[:2]
                    # Resize mask to match frame if dimensions differ
                    if mh != fh or mw != fw:
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            (fw, fh),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        mask = (mask > 0.5).astype(np.uint8)
                    color = self._get_class_color(result.label)
                    colored_mask = np.zeros_like(display)
                    colored_mask[mask > 0] = color
                    cv2.addWeighted(colored_mask, self._MASK_ALPHA, overlay, 1, 0, overlay)
            cv2.addWeighted(overlay, self._MASK_ALPHA, display, 1 - self._MASK_ALPHA, 0, display)

            for result in results:
                color = self._get_class_color(result.label)
                x, y = int(result.bbox.x), int(result.bbox.y)
                w, h = int(result.bbox.width), int(result.bbox.height)
                cv2.rectangle(display, (x, y), (x + w, y + h), color.tolist(), 2)
                label_text = f"{result.label} {result.confidence:.1%}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (x, y - th - 6), (x + tw, y), color.tolist(), -1)
                cv2.putText(
                    display, label_text, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

        h, w = display.shape[:2]
        channels = display.shape[2] if display.ndim == 3 else 1
        if channels == 3:
            qimage = QImage(display.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        else:
            qimage = QImage(display.data, w, h, w, QImage.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(qimage)
        self.update()

    def update_hud(self, fps: float, detection_count: int, running: bool, paused: bool, recording: bool) -> None:
        self._fps = fps
        self._detection_count = detection_count
        self._is_running = running
        self._is_paused = paused
        self._is_recording = recording
        if recording:
            self._rec_pulse_phase = (self._rec_pulse_phase + 0.1) % (2 * 3.14159)
        self.update()

    @classmethod
    def _get_class_color(cls, label: str) -> np.ndarray:
        idx = hash(label) % len(cls._PALETTE)
        return cls._PALETTE[idx]

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw video frame scaled to fill
        if self._pixmap:
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.fillRect(self.rect(), QColor("#0a0a14"))
            painter.setPen(QColor("#333355"))
            painter.setFont(QFont("Segoe UI", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Signal")

        # Draw floating HUD overlay
        if self._show_hud:
            self._draw_hud(painter)

        painter.end()

    def _draw_hud(self, painter: QPainter) -> None:
        """Draw semi-transparent HUD in corners."""
        w, h = self.width(), self.height()

        # Top-left: status + FPS
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawRoundedRect(12, 12, 200, 50, 8, 8)

        painter.setFont(QFont("Segoe UI", 11))
        if self._is_running:
            status_text = "PAUSED" if self._is_paused else "RUNNING"
            status_color = QColor("#f39c12") if self._is_paused else QColor("#2ecc71")
        else:
            status_text = "STOPPED"
            status_color = QColor("#888899")

        painter.setPen(status_color)
        painter.drawText(22, 34, f"● {status_text}")
        painter.setPen(QColor("#c0c0d0"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(22, 52, f"FPS: {self._fps:.1f}")

        # Top-right: recording indicator
        if self._is_recording:
            import math
            alpha = int(128 + 127 * math.sin(self._rec_pulse_phase))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 120))
            painter.drawRoundedRect(w - 110, 12, 98, 34, 8, 8)
            painter.setBrush(QColor(231, 76, 60, alpha))
            painter.drawEllipse(w - 96, 20, 14, 14)
            painter.setPen(QColor("#e74c3c"))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(w - 78, 34, "REC")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Show HUD on mouse move."""
        self.show_hud_temporarily()
        super().mouseMoveEvent(event)


# ─── Control Dialog ───────────────────────────────────────────────────

class ControlDialog(QDialog):
    """Floating control panel as a dialog, accessible from menu."""

    start_requested = Signal()
    stop_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    start_recording_requested = Signal()
    stop_recording_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Controls")
        self.setFixedSize(280, 240)
        self.setStyleSheet(DARK_STYLE)
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        self._is_running = False
        self._is_paused = False
        self._is_recording = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Detection
        det_group = QGroupBox("Detection")
        det_layout = QHBoxLayout(det_group)
        det_layout.setSpacing(6)

        self._start_btn = QPushButton("Start")
        self._start_btn.setStyleSheet(
            "QPushButton{background:#1a3a2a;border-color:#2ecc71;color:#2ecc71}"
            "QPushButton:hover{background:#2ecc71;color:#fff}"
        )
        self._start_btn.clicked.connect(self.start_requested.emit)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#3a1a1a;border-color:#e74c3c;color:#e74c3c}"
            "QPushButton:hover{background:#e74c3c;color:#fff}"
        )
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        self._stop_btn.setEnabled(False)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setStyleSheet(
            "QPushButton{background:#3a2e1a;border-color:#f39c12;color:#f39c12}"
            "QPushButton:hover{background:#f39c12;color:#fff}"
        )
        self._pause_btn.clicked.connect(self._on_pause)
        self._pause_btn.setEnabled(False)

        det_layout.addWidget(self._start_btn)
        det_layout.addWidget(self._stop_btn)
        det_layout.addWidget(self._pause_btn)

        # Recording
        rec_group = QGroupBox("Recording")
        rec_layout = QHBoxLayout(rec_group)

        self._record_btn = QPushButton("Record")
        self._record_btn.setStyleSheet(
            "QPushButton{background:#3a1a1a;border-color:#e74c3c;color:#e74c3c}"
            "QPushButton:hover{background:#e74c3c;color:#fff}"
        )
        self._record_btn.clicked.connect(self._on_record_toggle)
        self._record_btn.setEnabled(False)

        rec_layout.addWidget(self._record_btn)

        layout.addWidget(det_group)
        layout.addWidget(rec_group)
        layout.addStretch()

    def set_running(self, running: bool) -> None:
        self._is_running = running
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._pause_btn.setEnabled(running)
        self._record_btn.setEnabled(running)
        if not running:
            self._is_paused = False
            self._is_recording = False
            self._pause_btn.setText("Pause")
            self._record_btn.setText("Record")

    def set_paused(self, paused: bool) -> None:
        self._is_paused = paused
        self._pause_btn.setText("Resume" if paused else "Pause")

    def set_recording(self, recording: bool) -> None:
        self._is_recording = recording
        self._record_btn.setText("Stop Rec" if recording else "Record")

    def _on_pause(self) -> None:
        if self._is_paused:
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()

    def _on_record_toggle(self) -> None:
        if self._is_recording:
            self.stop_recording_requested.emit()
        else:
            self.start_recording_requested.emit()


# ─── Stats Dialog ─────────────────────────────────────────────────────

class StatsDialog(QDialog):
    """Statistics display as a floating dialog."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Statistics")
        self.setFixedSize(240, 200)
        self.setStyleSheet(DARK_STYLE)
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        group = QGroupBox("Live Stats")
        grid = QGridLayout(group)
        grid.setSpacing(6)

        self._fps_val = QLabel("0.0")
        self._fps_val.setStyleSheet("font-size:18px;font-weight:bold;color:#fff")
        self._fps_lbl = QLabel("FPS")
        self._fps_lbl.setStyleSheet("font-size:11px;color:#666688")

        self._det_val = QLabel("0")
        self._det_val.setStyleSheet("font-size:18px;font-weight:bold;color:#fff")
        self._det_lbl = QLabel("Detections")
        self._det_lbl.setStyleSheet("font-size:11px;color:#666688")

        self._rec_val = QLabel("Off")
        self._rec_val.setStyleSheet("font-size:18px;font-weight:bold;color:#888899")
        self._rec_lbl = QLabel("Recording")
        self._rec_lbl.setStyleSheet("font-size:11px;color:#666688")

        grid.addWidget(self._fps_val, 0, 0)
        grid.addWidget(self._fps_lbl, 0, 1)
        grid.addWidget(self._det_val, 1, 0)
        grid.addWidget(self._det_lbl, 1, 1)
        grid.addWidget(self._rec_val, 2, 0)
        grid.addWidget(self._rec_lbl, 2, 1)

        layout.addWidget(group)

    def update_stats(self, fps: float, detections: int, recording: bool) -> None:
        self._fps_val.setText(f"{fps:.1f}")
        self._det_val.setText(str(detections))
        self._rec_val.setText("On" if recording else "Off")
        self._rec_val.setStyleSheet(
            f"font-size:18px;font-weight:bold;color:{'#e74c3c' if recording else '#888899'}"
        )


# ─── Settings Dialog ─────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Full-featured settings dialog with tabbed sections."""

    config_saved = Signal()

    def __init__(self, config: AppConfig, config_path: str | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._config_path = config_path
        self._logger = logging.getLogger("larksnap.ui.settings")

        self.setWindowTitle("Settings")
        self.setMinimumSize(520, 480)
        self.setStyleSheet(DARK_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        tabs = QTabWidget()
        tabs.addTab(self._build_camera_tab(), "Camera")
        tabs.addTab(self._build_detector_tab(), "Detector")
        tabs.addTab(self._build_notifier_tab(), "Notifier")
        tabs.addTab(self._build_recorder_tab(), "Recorder")
        tabs.addTab(self._build_gateway_tab(), "Gateway")
        layout.addWidget(tabs)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(
            "QPushButton{background:#1a3a2a;border-color:#2ecc71;color:#2ecc71}"
            "QPushButton:hover{background:#2ecc71;color:#fff}"
        )
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _build_camera_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Device Index:"), 0, 0)
        self._cam_device = QSpinBox()
        self._cam_device.setRange(0, 99)
        self._cam_device.setValue(self._config.camera.device_index)
        layout.addWidget(self._cam_device, 0, 1)

        layout.addWidget(QLabel("Width:"), 1, 0)
        self._cam_width = QSpinBox()
        self._cam_width.setRange(160, 3840)
        self._cam_width.setSingleStep(160)
        self._cam_width.setValue(self._config.camera.width)
        layout.addWidget(self._cam_width, 1, 1)

        layout.addWidget(QLabel("Height:"), 2, 0)
        self._cam_height = QSpinBox()
        self._cam_height.setRange(120, 2160)
        self._cam_height.setSingleStep(120)
        self._cam_height.setValue(self._config.camera.height)
        layout.addWidget(self._cam_height, 2, 1)

        layout.addWidget(QLabel("FPS:"), 3, 0)
        self._cam_fps = QSpinBox()
        self._cam_fps.setRange(1, 120)
        self._cam_fps.setValue(self._config.camera.fps)
        layout.addWidget(self._cam_fps, 3, 1)

        layout.addWidget(QLabel("Capture Interval (s):"), 4, 0)
        self._cam_interval = QDoubleSpinBox()
        self._cam_interval.setRange(0.1, 60.0)
        self._cam_interval.setSingleStep(0.5)
        self._cam_interval.setValue(self._config.camera.capture_interval)
        layout.addWidget(self._cam_interval, 4, 1)

        layout.addWidget(QLabel("Retry Interval (s):"), 5, 0)
        self._cam_retry_interval = QDoubleSpinBox()
        self._cam_retry_interval.setRange(0.5, 60.0)
        self._cam_retry_interval.setSingleStep(0.5)
        self._cam_retry_interval.setValue(self._config.camera.retry_interval)
        layout.addWidget(self._cam_retry_interval, 5, 1)

        layout.addWidget(QLabel("Max Retries:"), 6, 0)
        self._cam_max_retries = QSpinBox()
        self._cam_max_retries.setRange(0, 100)
        self._cam_max_retries.setValue(self._config.camera.max_retries)
        layout.addWidget(self._cam_max_retries, 6, 1)

        layout.setRowStretch(7, 1)
        return w

    def _build_detector_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Detector Type:"), 0, 0)
        self._det_type = QComboBox()
        self._det_type.addItems(["seg", "mock"])
        self._det_type.setCurrentText(self._config.detector.type)
        layout.addWidget(self._det_type, 0, 1)

        layout.addWidget(QLabel("Model Path:"), 1, 0)
        self._det_model = QLineEdit(self._config.detector.seg.model_path)
        layout.addWidget(self._det_model, 1, 1)

        layout.addWidget(QLabel("Confidence Threshold:"), 2, 0)
        self._det_threshold = QDoubleSpinBox()
        self._det_threshold.setRange(0.01, 1.0)
        self._det_threshold.setSingleStep(0.05)
        self._det_threshold.setValue(self._config.detector.confidence_threshold)
        layout.addWidget(self._det_threshold, 2, 1)

        layout.addWidget(QLabel("Target Classes:"), 3, 0)
        self._det_targets = QLineEdit(", ".join(self._config.detector.target_classes))
        self._det_targets.setPlaceholderText("person, car, dog (comma-separated)")
        layout.addWidget(self._det_targets, 3, 1)

        layout.addWidget(QLabel("IoU Threshold:"), 4, 0)
        self._det_iou = QDoubleSpinBox()
        self._det_iou.setRange(0.01, 1.0)
        self._det_iou.setSingleStep(0.05)
        self._det_iou.setValue(self._config.detector.seg.iou_thres)
        layout.addWidget(self._det_iou, 4, 1)

        layout.addWidget(QLabel("Image Size:"), 5, 0)
        self._det_img_size = QSpinBox()
        self._det_img_size.setRange(320, 1280)
        self._det_img_size.setSingleStep(32)
        self._det_img_size.setValue(self._config.detector.seg.img_size)
        layout.addWidget(self._det_img_size, 5, 1)

        layout.addWidget(QLabel("Provider:"), 6, 0)
        self._det_provider = QComboBox()
        self._det_provider.addItems(["cpu", "cuda"])
        self._det_provider.setCurrentText(self._config.detector.seg.provider)
        layout.addWidget(self._det_provider, 6, 1)

        layout.setRowStretch(7, 1)
        return w

    def _build_notifier_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setSpacing(10)

        layout.addWidget(QLabel("App ID:"), 0, 0)
        self._notif_app_id = QLineEdit(self._config.notifier.app_id)
        layout.addWidget(self._notif_app_id, 0, 1)

        layout.addWidget(QLabel("App Secret:"), 1, 0)
        self._notif_app_secret = QLineEdit(self._config.notifier.app_secret)
        self._notif_app_secret.setEchoMode(QLineEdit.Password)
        layout.addWidget(self._notif_app_secret, 1, 1)

        layout.addWidget(QLabel("Chat ID:"), 2, 0)
        self._notif_chat_id = QLineEdit(self._config.notifier.chat_id)
        layout.addWidget(self._notif_chat_id, 2, 1)

        layout.addWidget(QLabel("Message Template:"), 3, 0)
        self._notif_template = QLineEdit(self._config.notifier.message_template)
        self._notif_template.setPlaceholderText("{label}, {confidence}, {timestamp}, {snapshot_path}")
        layout.addWidget(self._notif_template, 3, 1)

        layout.addWidget(QLabel("Send Image:"), 4, 0)
        self._notif_send_image = QCheckBox()
        self._notif_send_image.setChecked(self._config.notifier.send_image)
        layout.addWidget(self._notif_send_image, 4, 1)

        layout.setRowStretch(5, 1)
        return w

    def _build_recorder_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self._rec_dir = QLineEdit(self._config.recorder.output_dir)
        layout.addWidget(self._rec_dir, 0, 1)

        layout.addWidget(QLabel("FPS:"), 1, 0)
        self._rec_fps = QDoubleSpinBox()
        self._rec_fps.setRange(1.0, 120.0)
        self._rec_fps.setValue(self._config.recorder.fps)
        layout.addWidget(self._rec_fps, 1, 1)

        layout.addWidget(QLabel("Codec:"), 2, 0)
        self._rec_codec = QComboBox()
        self._rec_codec.addItems(["mp4v", "xvid", "h264", "avc1"])
        self._rec_codec.setCurrentText(self._config.recorder.codec)
        layout.addWidget(self._rec_codec, 2, 1)

        layout.setRowStretch(3, 1)
        return w

    def _build_gateway_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Notification Interval (s):"), 0, 0)
        self._gw_notif_interval = QSpinBox()
        self._gw_notif_interval.setRange(1, 3600)
        self._gw_notif_interval.setValue(self._config.gateway.notification_interval)
        self._gw_notif_interval.setToolTip("Minimum seconds between same-label notifications to Feishu")
        layout.addWidget(self._gw_notif_interval, 0, 1)

        layout.addWidget(QLabel("Snapshot Directory:"), 1, 0)
        self._gw_snapshot_dir = QLineEdit(self._config.gateway.snapshot_dir)
        layout.addWidget(self._gw_snapshot_dir, 1, 1)

        layout.addWidget(QLabel("Frame Queue HWM:"), 2, 0)
        self._gw_hwm = QSpinBox()
        self._gw_hwm.setRange(1, 1000)
        self._gw_hwm.setValue(self._config.gateway.frame_queue_hwm)
        layout.addWidget(self._gw_hwm, 2, 1)

        layout.addWidget(QLabel("Queue Policy:"), 3, 0)
        self._gw_policy = QComboBox()
        self._gw_policy.addItems(["drop_oldest", "drop_newest", "block"])
        self._gw_policy.setCurrentText(self._config.gateway.frame_queue_policy)
        layout.addWidget(self._gw_policy, 3, 1)

        layout.setRowStretch(4, 1)
        return w

    def _on_save(self) -> None:
        self._config.camera.device_index = self._cam_device.value()
        self._config.camera.width = self._cam_width.value()
        self._config.camera.height = self._cam_height.value()
        self._config.camera.fps = self._cam_fps.value()
        self._config.camera.capture_interval = self._cam_interval.value()
        self._config.camera.retry_interval = self._cam_retry_interval.value()
        self._config.camera.max_retries = self._cam_max_retries.value()

        self._config.detector.type = self._det_type.currentText()
        self._config.detector.seg.model_path = self._det_model.text()
        self._config.detector.confidence_threshold = self._det_threshold.value()
        self._config.detector.target_classes = [
            s.strip() for s in self._det_targets.text().split(",") if s.strip()
        ]
        self._config.detector.seg.iou_thres = self._det_iou.value()
        self._config.detector.seg.img_size = self._det_img_size.value()
        self._config.detector.seg.provider = self._det_provider.currentText()

        self._config.notifier.app_id = self._notif_app_id.text()
        self._config.notifier.app_secret = self._notif_app_secret.text()
        self._config.notifier.chat_id = self._notif_chat_id.text()
        self._config.notifier.message_template = self._notif_template.text()
        self._config.notifier.send_image = self._notif_send_image.isChecked()

        self._config.recorder.output_dir = self._rec_dir.text()
        self._config.recorder.fps = self._rec_fps.value()
        self._config.recorder.codec = self._rec_codec.currentText()

        self._config.gateway.notification_interval = self._gw_notif_interval.value()
        self._config.gateway.snapshot_dir = self._gw_snapshot_dir.text()
        self._config.gateway.frame_queue_hwm = self._gw_hwm.value()
        self._config.gateway.frame_queue_policy = self._gw_policy.currentText()

        if self._config_path:
            try:
                save_config(self._config, self._config_path)
                self._logger.info("Config saved to %s", self._config_path)
            except Exception as e:
                self._logger.error("Failed to save config: %s", e)
                QMessageBox.warning(self, "Save Error", f"Failed to save config:\n{e}")
                return

        self.config_saved.emit()
        self.accept()


# ─── Main Window ──────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Immersive video-centric main window. Controls via menu only."""

    def __init__(
        self,
        controller: GatewayController,
        config: AppConfig,
        config_path: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._config = config
        self._config_path = config_path
        self._logger = logging.getLogger("larksnap.ui.main_window")

        self._control_dialog: ControlDialog | None = None
        self._stats_dialog: StatsDialog | None = None

        self.setStyleSheet(DARK_STYLE)
        self.setWindowTitle("LarkSnap")
        self.setMinimumSize(800, 500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)

        # Full video preview as central widget
        self._preview = VideoPreviewWidget()
        self.setCentralWidget(self._preview)

        # No status bar — HUD is overlaid on video

        # Build menu bar
        self._build_menu()

        # System tray
        self._tray_icon = QSystemTrayIcon(self)
        self._tray_icon.setIcon(self._create_tray_icon())
        self._tray_icon.setToolTip("LarkSnap")
        self._tray_icon.activated.connect(self._on_tray_activated)

        tray_menu = QMenu()
        tray_menu.addAction("Show", self._show_window)
        tray_menu.addSeparator()
        tray_menu.addAction("Start", self._on_start)
        tray_menu.addAction("Stop", self._on_stop)
        tray_menu.addAction("Pause / Resume", self._on_pause_resume)
        tray_menu.addSeparator()
        tray_menu.addAction("Record", self._on_record_toggle)
        tray_menu.addSeparator()
        tray_menu.addAction("Settings", self._show_settings)
        tray_menu.addSeparator()
        tray_menu.addAction("Quit", self._quit_app)
        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.show()

        self._close_to_tray = True

        # Subscribe to gateway events
        self._controller.event_bus.subscribe(EventType.CAMERA_FAILED, self._on_camera_failed)

        # Timers
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.setInterval(33)

        self._hud_timer = QTimer(self)
        self._hud_timer.timeout.connect(self._update_hud)
        self._hud_timer.setInterval(200)

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # ── Detection menu ──
        det_menu = menu_bar.addMenu("Detection")

        self._start_action = det_menu.addAction("Start")
        self._start_action.setShortcut("Ctrl+S")
        self._start_action.triggered.connect(self._on_start)

        self._stop_action = det_menu.addAction("Stop")
        self._stop_action.setShortcut("Ctrl+D")
        self._stop_action.triggered.connect(self._on_stop)
        self._stop_action.setEnabled(False)

        det_menu.addSeparator()

        self._pause_action = det_menu.addAction("Pause")
        self._pause_action.setShortcut("Space")
        self._pause_action.triggered.connect(self._on_pause_resume)
        self._pause_action.setEnabled(False)

        # ── Recording menu ──
        rec_menu = menu_bar.addMenu("Recording")

        self._record_action = rec_menu.addAction("Start Recording")
        self._record_action.setShortcut("Ctrl+R")
        self._record_action.triggered.connect(self._on_record_toggle)
        self._record_action.setEnabled(False)

        # ── View menu ──
        view_menu = menu_bar.addMenu("View")

        view_menu.addAction("Controls", self._show_control_dialog, "Ctrl+L")
        view_menu.addAction("Statistics", self._show_stats_dialog, "Ctrl+I")
        view_menu.addSeparator()
        view_menu.addAction("Fullscreen", self._toggle_fullscreen, "F11")
        view_menu.addSeparator()
        self._hud_action = view_menu.addAction("Show HUD")
        self._hud_action.setCheckable(True)
        self._hud_action.setChecked(True)
        self._hud_action.setShortcut("Ctrl+H")
        self._hud_action.toggled.connect(self._toggle_hud)

        # ── Settings menu ──
        settings_menu = menu_bar.addMenu("Settings")
        settings_menu.addAction("Preferences", self._show_settings, "Ctrl+,")

        # ── Help menu ──
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("About", self._show_about)

    # ── Menu action handlers ──

    def _on_camera_failed(self, event: Event) -> None:
        """Handle camera failure event — show error dialog."""
        data = event.data or {}
        device_index = data.get("device_index", "?")
        error_msg = data.get("error", "Unknown error")
        QMessageBox.critical(
            self,
            "Camera Error",
            f"Failed to open camera (device {device_index}).\n\n"
            f"Error: {error_msg}\n\n"
            f"Please check that the camera is connected and not in use by another application.",
        )

    def _on_start(self) -> None:
        if not self._controller.is_running:
            try:
                self._controller.initialize()
                self._controller.start()
            except Exception as e:
                from larksnap.utils.exceptions import CameraError
                if isinstance(e, (CameraError,)) or "camera" in str(e).lower():
                    # Camera error already handled by event bus
                    self._update_action_states()
                    return
                raise
        self._update_action_states()
        self._preview.show_hud_temporarily()

    def _on_stop(self) -> None:
        self._controller.stop()
        self._update_action_states()
        if self._control_dialog:
            self._control_dialog.set_running(False)
            self._control_dialog.set_recording(False)
        if self._stats_dialog:
            self._stats_dialog.update_stats(0, 0, False)

    def _on_pause_resume(self) -> None:
        if not self._controller.is_running:
            return
        if self._controller.is_paused:
            self._controller.resume()
        else:
            self._controller.pause()
        self._update_action_states()
        if self._control_dialog:
            self._control_dialog.set_paused(self._controller.is_paused)

    def _on_record_toggle(self) -> None:
        if self._controller.is_recording:
            self._controller.stop_recording()
        else:
            self._controller.start_recording()
        self._update_action_states()
        if self._control_dialog:
            self._control_dialog.set_recording(self._controller.is_recording)

    def _update_action_states(self) -> None:
        running = self._controller.is_running
        paused = self._controller.is_paused
        recording = self._controller.is_recording

        self._start_action.setEnabled(not running)
        self._stop_action.setEnabled(running)
        self._pause_action.setEnabled(running)
        self._pause_action.setText("Resume" if paused else "Pause")
        self._record_action.setEnabled(running)
        self._record_action.setText("Stop Recording" if recording else "Start Recording")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.menuBar().show()
        else:
            self.menuBar().hide()
            self.showFullScreen()

    def _toggle_hud(self, checked: bool) -> None:
        self._preview._show_hud = checked
        if not checked:
            self._preview._hud_timer.stop()
        self._preview.update()

    def _show_control_dialog(self) -> None:
        if self._control_dialog is None:
            self._control_dialog = ControlDialog(self)
            self._control_dialog.start_requested.connect(self._on_start)
            self._control_dialog.stop_requested.connect(self._on_stop)
            self._control_dialog.pause_requested.connect(self._on_pause_resume)
            self._control_dialog.resume_requested.connect(self._on_pause_resume)
            self._control_dialog.start_recording_requested.connect(self._on_record_toggle)
            self._control_dialog.stop_recording_requested.connect(self._on_record_toggle)
        self._control_dialog.set_running(self._controller.is_running)
        self._control_dialog.set_paused(self._controller.is_paused)
        self._control_dialog.set_recording(self._controller.is_recording)
        self._control_dialog.show()
        self._control_dialog.raise_()

    def _show_stats_dialog(self) -> None:
        if self._stats_dialog is None:
            self._stats_dialog = StatsDialog(self)
        self._stats_dialog.show()
        self._stats_dialog.raise_()

    def _show_settings(self) -> None:
        dialog = SettingsDialog(self._config, self._config_path, self)
        dialog.config_saved.connect(lambda: None)
        dialog.exec()

    def _show_about(self) -> None:
        QMessageBox.about(self, "About LarkSnap", "LarkSnap v0.1.0\n\nGateway-controlled object detection with ZeroMQ.")

    # ── Timers ──

    def start_preview(self) -> None:
        self._preview_timer.start()
        self._hud_timer.start()

    def stop_preview(self) -> None:
        self._preview_timer.stop()
        self._hud_timer.stop()

    def _update_preview(self) -> None:
        frame = self._controller.get_latest_frame()
        if frame is not None:
            results = self._controller.get_latest_results()
            self._preview.update_frame(frame, results)

    def _update_hud(self) -> None:
        self._preview.update_hud(
            fps=self._controller.producer_fps,
            detection_count=self._controller.detection_count,
            running=self._controller.is_running,
            paused=self._controller.is_paused,
            recording=self._controller.is_recording,
        )
        if self._stats_dialog and self._stats_dialog.isVisible():
            self._stats_dialog.update_stats(
                fps=self._controller.producer_fps,
                detections=self._controller.detection_count,
                recording=self._controller.is_recording,
            )

    # ── Keyboard ──

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Space:
            self._on_pause_resume()
        elif key == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            self.menuBar().show()
        else:
            super().keyPressEvent(event)

    # ── Tray & close ──

    def changeEvent(self, event) -> None:
        if event.type() == event.Type.WindowStateChange:
            if self.windowState() & Qt.WindowMinimized:
                if self._close_to_tray:
                    QTimer.singleShot(0, self.hide)
                    self._tray_icon.showMessage(
                        "LarkSnap", "Running in background. Click tray icon to restore.",
                        QSystemTrayIcon.Information, 2000,
                    )
        super().changeEvent(event)

    def closeEvent(self, event) -> None:
        if self._close_to_tray:
            event.ignore()
            self.hide()
            self._tray_icon.showMessage(
                "LarkSnap", "Minimized to tray. Right-click to quit.",
                QSystemTrayIcon.Information, 2000,
            )
        else:
            self._controller.stop()
            self.stop_preview()
            self._tray_icon.hide()
            super().closeEvent(event)

    def _create_tray_icon(self) -> QIcon:
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(106, 106, 174))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 20, QFont.Bold))
        painter.drawText(QRect(0, 0, 64, 64), Qt.AlignCenter, "LS")
        painter.end()
        return QIcon(pixmap)

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_window()

    def _show_window(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _quit_app(self) -> None:
        self._close_to_tray = False
        self.close()
