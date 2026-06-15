"""LarkSnap main window with PySide6 — dynamic, animated dark-themed UI.

Provides real-time camera preview, detection overlay with masks,
configuration dialog, recording controls, and animated status display.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from larksnap.adapters.detector.interface import DetectionResult
from larksnap.config.loader import save_config
from larksnap.config.models import AppConfig
from larksnap.gateway.controller import GatewayController

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
QPushButton:checked {
    background-color: #3a3a5e;
    border-color: #6a6aae;
}
QStatusBar {
    background-color: #0a0a14;
    color: #666688;
    border-top: 1px solid #1a1a2e;
    font-size: 12px;
}
QMenuBar {
    background-color: #0a0a14;
    border-bottom: 1px solid #1a1a2e;
}
QMenuBar::item:selected {
    background-color: #1e1e32;
}
QMenu {
    background-color: #14141e;
    border: 1px solid #2a2a3e;
}
QMenu::item:selected {
    background-color: #2a2a44;
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
QSlider::groove:horizontal {
    height: 6px;
    background: #1a1a2e;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #6a6aae;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #4a4a8e;
    border-radius: 3px;
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
QScrollArea {
    border: none;
    background-color: transparent;
}
"""


# ─── Animated Pulse Widget ────────────────────────────────────────────

class PulseWidget(QWidget):
    """A widget that draws an animated pulsing circle (for recording indicator)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._opacity = 0.0
        self._target_opacity = 0.0
        self.setFixedSize(20, 20)

    def set_active(self, active: bool) -> None:
        self._target_opacity = 1.0 if active else 0.0
        self.update()

    def set_opacity(self, opacity: float) -> None:
        self._opacity = opacity
        self.update()

    def get_opacity(self) -> float:
        return self._opacity

    opacity = property(get_opacity, set_opacity)

    def paintEvent(self, event: QPaintEvent) -> None:
        if self._opacity < 0.01:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(231, 76, 60, int(self._opacity * 255))
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 16, 16)
        # Glow
        glow = QColor(231, 76, 60, int(self._opacity * 80))
        painter.setBrush(glow)
        painter.drawEllipse(0, 0, 20, 20)
        painter.end()


# ─── Video Preview ────────────────────────────────────────────────────

class VideoPreviewWidget(QWidget):
    """Real-time camera preview with detection bounding box and mask overlay."""

    _PALETTE = np.random.RandomState(42).randint(50, 255, size=(80, 3), dtype=np.uint8)
    _MASK_ALPHA = 0.45

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(640, 480)
        self._image_label.setStyleSheet(
            "background-color: #0a0a14; border-radius: 8px; border: 1px solid #1a1a2e;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_label)

    @classmethod
    def _get_class_color(cls, label: str) -> np.ndarray:
        idx = hash(label) % len(cls._PALETTE)
        return cls._PALETTE[idx]

    def update_frame(self, frame: np.ndarray, results: list[DetectionResult] | None = None) -> None:
        display = frame.copy()

        if results:
            overlay = display.copy()
            for result in results:
                if result.mask is not None and result.mask.size > 0:
                    color = self._get_class_color(result.label)
                    colored_mask = np.zeros_like(display)
                    colored_mask[result.mask > 0] = color
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

        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)


# ─── Animated Button ──────────────────────────────────────────────────

class AnimatedButton(QPushButton):
    """QPushButton with scale animation on click."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._anim = QPropertyAnimation(self, b"geometry")
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def mousePressEvent(self, event) -> None:
        geo = self.geometry()
        shrink = QRect(geo.x() + 2, geo.y() + 2, geo.width() - 4, geo.height() - 4)
        self._anim.setStartValue(shrink)
        self._anim.setEndValue(geo)
        self._anim.start()
        super().mousePressEvent(event)


# ─── Status Badge ─────────────────────────────────────────────────────

class StatusBadge(QLabel):
    """Animated status badge with color transitions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = "stopped"
        self.setFixedHeight(28)
        self.setAlignment(Qt.AlignCenter)
        self._update_style()

    def set_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        # Fade animation
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(300)
        self._anim.setStartValue(0.3)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()
        self._update_style()

    def _update_style(self) -> None:
        colors = {
            "running": ("#2ecc71", "#0f2e1a", "Running"),
            "paused": ("#f39c12", "#2e2a0f", "Paused"),
            "stopped": ("#888899", "#1a1a22", "Stopped"),
            "recording": ("#e74c3c", "#2e0f0f", "Recording"),
        }
        fg, bg, text = colors.get(self._status, colors["stopped"])
        self.setText(f"  {text}  ")
        self.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {fg}; "
            f"border-radius: 14px; font-weight: bold; font-size: 12px; padding: 2px 12px;"
        )


# ─── Control Panel ────────────────────────────────────────────────────

class ControlPanel(QWidget):
    """Control panel with animated buttons and recording pulse indicator."""

    start_requested = Signal()
    stop_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    start_recording_requested = Signal()
    stop_recording_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_running = False
        self._is_paused = False
        self._is_recording = False

        # Detection controls
        detection_group = QGroupBox("Detection")
        detection_layout = QHBoxLayout(detection_group)
        detection_layout.setSpacing(8)

        self._start_btn = AnimatedButton("Start")
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #1a3a2a; border-color: #2ecc71; color: #2ecc71; }"
            "QPushButton:hover { background-color: #2ecc71; color: #ffffff; }"
        )
        self._start_btn.clicked.connect(self._on_start)

        self._stop_btn = AnimatedButton("Stop")
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #3a1a1a; border-color: #e74c3c; color: #e74c3c; }"
            "QPushButton:hover { background-color: #e74c3c; color: #ffffff; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)

        self._pause_btn = AnimatedButton("Pause")
        self._pause_btn.setStyleSheet(
            "QPushButton { background-color: #3a2e1a; border-color: #f39c12; color: #f39c12; }"
            "QPushButton:hover { background-color: #f39c12; color: #ffffff; }"
        )
        self._pause_btn.clicked.connect(self._on_pause)
        self._pause_btn.setEnabled(False)

        detection_layout.addWidget(self._start_btn)
        detection_layout.addWidget(self._stop_btn)
        detection_layout.addWidget(self._pause_btn)

        # Recording controls
        recording_group = QGroupBox("Recording")
        recording_layout = QHBoxLayout(recording_group)
        recording_layout.setSpacing(8)

        self._record_btn = AnimatedButton("Record")
        self._record_btn.setStyleSheet(
            "QPushButton { background-color: #3a1a1a; border-color: #e74c3c; color: #e74c3c; }"
            "QPushButton:hover { background-color: #e74c3c; color: #ffffff; }"
        )
        self._record_btn.clicked.connect(self._on_record_toggle)
        self._record_btn.setEnabled(False)

        self._pulse = PulseWidget()
        self._rec_label = QLabel("")
        self._rec_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 13px;")

        recording_layout.addWidget(self._record_btn)
        recording_layout.addWidget(self._pulse)
        recording_layout.addWidget(self._rec_label)
        recording_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addWidget(detection_group)
        layout.addWidget(recording_group)

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
            self._pulse.set_active(False)
            self._rec_label.setText("")

    def set_paused(self, paused: bool) -> None:
        self._is_paused = paused
        self._pause_btn.setText("Resume" if paused else "Pause")

    def set_recording(self, recording: bool) -> None:
        self._is_recording = recording
        self._record_btn.setText("Stop" if recording else "Record")
        self._pulse.set_active(recording)
        self._rec_label.setText("REC" if recording else "")

    def _on_start(self) -> None:
        self.start_requested.emit()

    def _on_stop(self) -> None:
        self.stop_requested.emit()

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


# ─── Stats Panel ──────────────────────────────────────────────────────

class StatsPanel(QWidget):
    """Animated stats panel with live counters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        group = QGroupBox("Statistics")
        layout = QGridLayout(group)
        layout.setSpacing(6)

        self._status_badge = StatusBadge()
        self._fps_value = self._make_stat_value("0.0")
        self._fps_label = self._make_stat_label("FPS")
        self._det_value = self._make_stat_value("0")
        self._det_label = self._make_stat_label("Detections")
        self._rec_value = self._make_stat_value("Off")
        self._rec_label = self._make_stat_label("Recording")

        layout.addWidget(self._status_badge, 0, 0, 1, 2)
        layout.addWidget(self._fps_value, 1, 0)
        layout.addWidget(self._fps_label, 1, 1)
        layout.addWidget(self._det_value, 2, 0)
        layout.addWidget(self._det_label, 2, 1)
        layout.addWidget(self._rec_value, 3, 0)
        layout.addWidget(self._rec_label, 3, 1)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(group)

    @staticmethod
    def _make_stat_value(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        return lbl

    @staticmethod
    def _make_stat_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 11px; color: #666688;")
        return lbl

    def update_stats(self, status: str, fps: float, detections: int, recording: bool) -> None:
        self._status_badge.set_status(status)
        self._fps_value.setText(f"{fps:.1f}")
        self._det_value.setText(str(detections))
        self._rec_value.setText("On" if recording else "Off")
        self._rec_value.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {'#e74c3c' if recording else '#888899'};"
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

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._build_camera_tab(), "Camera")
        tabs.addTab(self._build_detector_tab(), "Detector")
        tabs.addTab(self._build_notifier_tab(), "Notifier")
        tabs.addTab(self._build_recorder_tab(), "Recorder")
        tabs.addTab(self._build_gateway_tab(), "Gateway")
        layout.addWidget(tabs)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(
            "QPushButton { background-color: #1a3a2a; border-color: #2ecc71; color: #2ecc71; }"
            "QPushButton:hover { background-color: #2ecc71; color: #ffffff; }"
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

        layout.setRowStretch(5, 1)
        return w

    def _build_detector_tab(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Detector Type:"), 0, 0)
        self._det_type = QComboBox()
        self._det_type.addItems(["yolo_seg", "mock"])
        self._det_type.setCurrentText(self._config.detector.type)
        layout.addWidget(self._det_type, 0, 1)

        layout.addWidget(QLabel("Model Path:"), 1, 0)
        self._det_model = QLineEdit(self._config.detector.yolo_seg.model_path)
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
        self._det_iou.setValue(self._config.detector.yolo_seg.iou_thres)
        layout.addWidget(self._det_iou, 4, 1)

        layout.addWidget(QLabel("Image Size:"), 5, 0)
        self._det_img_size = QSpinBox()
        self._det_img_size.setRange(320, 1280)
        self._det_img_size.setSingleStep(32)
        self._det_img_size.setValue(self._config.detector.yolo_seg.img_size)
        layout.addWidget(self._det_img_size, 5, 1)

        layout.addWidget(QLabel("Provider:"), 6, 0)
        self._det_provider = QComboBox()
        self._det_provider.addItems(["cpu", "cuda"])
        self._det_provider.setCurrentText(self._config.detector.yolo_seg.provider)
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

        layout.addWidget(QLabel("Process Interval (s):"), 0, 0)
        self._gw_interval = QDoubleSpinBox()
        self._gw_interval.setRange(0.1, 300.0)
        self._gw_interval.setSingleStep(0.5)
        self._gw_interval.setValue(self._config.gateway.process_interval)
        layout.addWidget(self._gw_interval, 0, 1)

        layout.addWidget(QLabel("Notification Cooldown (s):"), 1, 0)
        self._gw_cooldown = QSpinBox()
        self._gw_cooldown.setRange(1, 3600)
        self._gw_cooldown.setValue(self._config.gateway.notification_cooldown)
        layout.addWidget(self._gw_cooldown, 1, 1)

        layout.addWidget(QLabel("Snapshot Directory:"), 2, 0)
        self._gw_snapshot_dir = QLineEdit(self._config.gateway.snapshot_dir)
        layout.addWidget(self._gw_snapshot_dir, 2, 1)

        layout.addWidget(QLabel("Frame Queue HWM:"), 3, 0)
        self._gw_hwm = QSpinBox()
        self._gw_hwm.setRange(1, 1000)
        self._gw_hwm.setValue(self._config.gateway.frame_queue_hwm)
        layout.addWidget(self._gw_hwm, 3, 1)

        layout.addWidget(QLabel("Queue Policy:"), 4, 0)
        self._gw_policy = QComboBox()
        self._gw_policy.addItems(["drop_oldest", "drop_newest", "block"])
        self._gw_policy.setCurrentText(self._config.gateway.frame_queue_policy)
        layout.addWidget(self._gw_policy, 4, 1)

        layout.setRowStretch(5, 1)
        return w

    def _on_save(self) -> None:
        """Apply dialog values to config and save."""
        self._config.camera.device_index = self._cam_device.value()
        self._config.camera.width = self._cam_width.value()
        self._config.camera.height = self._cam_height.value()
        self._config.camera.fps = self._cam_fps.value()
        self._config.camera.capture_interval = self._cam_interval.value()

        self._config.detector.type = self._det_type.currentText()
        self._config.detector.yolo_seg.model_path = self._det_model.text()
        self._config.detector.confidence_threshold = self._det_threshold.value()
        self._config.detector.target_classes = [
            s.strip() for s in self._det_targets.text().split(",") if s.strip()
        ]
        self._config.detector.yolo_seg.iou_thres = self._det_iou.value()
        self._config.detector.yolo_seg.img_size = self._det_img_size.value()
        self._config.detector.yolo_seg.provider = self._det_provider.currentText()

        self._config.notifier.app_id = self._notif_app_id.text()
        self._config.notifier.app_secret = self._notif_app_secret.text()
        self._config.notifier.chat_id = self._notif_chat_id.text()
        self._config.notifier.message_template = self._notif_template.text()
        self._config.notifier.send_image = self._notif_send_image.isChecked()

        self._config.recorder.output_dir = self._rec_dir.text()
        self._config.recorder.fps = self._rec_fps.value()
        self._config.recorder.codec = self._rec_codec.currentText()

        self._config.gateway.process_interval = self._gw_interval.value()
        self._config.gateway.notification_cooldown = self._gw_cooldown.value()
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
    """Main application window for LarkSnap with animated dark UI."""

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

        self.setStyleSheet(DARK_STYLE)
        self.setWindowTitle("LarkSnap")
        self.setMinimumSize(1020, 660)

        # Menu bar
        self._build_menu()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Left: Video preview
        self._preview = VideoPreviewWidget()
        main_layout.addWidget(self._preview, stretch=3)

        # Right: Panels
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self._control_panel = ControlPanel()
        self._stats_panel = StatsPanel()

        right_layout.addWidget(self._control_panel)
        right_layout.addWidget(self._stats_panel)
        right_layout.addStretch()

        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — Press Start to begin detection")

        # Connect signals
        self._control_panel.start_requested.connect(self._on_start)
        self._control_panel.stop_requested.connect(self._on_stop)
        self._control_panel.pause_requested.connect(self._on_pause)
        self._control_panel.resume_requested.connect(self._on_resume)
        self._control_panel.start_recording_requested.connect(self._on_start_recording)
        self._control_panel.stop_recording_requested.connect(self._on_stop_recording)

        # Timers
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.setInterval(33)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.setInterval(500)

        # Pulse animation for recording indicator
        self._pulse_anim = QPropertyAnimation(self._control_panel._pulse, b"opacity")
        self._pulse_anim.setDuration(800)
        self._pulse_anim.setStartValue(1.0)
        self._pulse_anim.setEndValue(0.2)
        self._pulse_anim.setEasingCurve(QEasingCurve.InOutSine)
        self._pulse_anim.setLoopCount(-1)

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        settings_action = file_menu.addAction("Settings")
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._show_settings)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        # View menu
        view_menu = menu_bar.addMenu("View")

        fullscreen_action = view_menu.addAction("Fullscreen")
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._show_about)

    def _show_settings(self) -> None:
        dialog = SettingsDialog(self._config, self._config_path, self)
        dialog.config_saved.connect(self._on_config_saved)
        dialog.exec()

    def _on_config_saved(self) -> None:
        self._status_bar.showMessage("Settings saved — restart to apply changes", 4000)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _show_about(self) -> None:
        QMessageBox.about(self, "About LarkSnap", "LarkSnap v0.1.0\n\nGateway-controlled object detection system with ZeroMQ message queue.")

    def start_preview(self) -> None:
        self._preview_timer.start()
        self._status_timer.start()

    def stop_preview(self) -> None:
        self._preview_timer.stop()
        self._status_timer.stop()
        self._pulse_anim.stop()

    def _on_start(self) -> None:
        if not self._controller.is_running:
            self._controller.initialize()
            self._controller.start()
        self._control_panel.set_running(True)
        self._status_bar.showMessage("Detection running")

    def _on_stop(self) -> None:
        self._controller.stop()
        self._control_panel.set_running(False)
        self._control_panel.set_recording(False)
        self._status_bar.showMessage("Detection stopped")

    def _on_pause(self) -> None:
        self._controller.pause()
        self._control_panel.set_paused(True)
        self._status_bar.showMessage("Detection paused")

    def _on_resume(self) -> None:
        self._controller.resume()
        self._control_panel.set_paused(False)
        self._status_bar.showMessage("Detection running")

    def _on_start_recording(self) -> None:
        self._controller.start_recording()
        self._control_panel.set_recording(True)
        self._pulse_anim.start()
        self._status_bar.showMessage("Recording started")

    def _on_stop_recording(self) -> None:
        self._controller.stop_recording()
        self._control_panel.set_recording(False)
        self._pulse_anim.stop()
        self._status_bar.showMessage("Recording stopped")

    def _update_preview(self) -> None:
        frame = self._controller.get_latest_frame()
        if frame is not None:
            results = self._controller.get_latest_results()
            self._preview.update_frame(frame, results)

    def _update_status(self) -> None:
        is_running = self._controller.is_running
        is_paused = self._controller.is_paused
        is_recording = self._controller.is_recording

        if is_running:
            status = "paused" if is_paused else "running"
        else:
            status = "stopped"
        if is_recording:
            status = "recording"

        self._stats_panel.update_stats(
            status=status,
            fps=self._controller.producer_fps,
            detections=self._controller.detection_count,
            recording=is_recording,
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space and self._controller.is_running:
            if self._controller.is_paused:
                self._on_resume()
            else:
                self._on_pause()
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._controller.stop()
        self.stop_preview()
        super().closeEvent(event)
