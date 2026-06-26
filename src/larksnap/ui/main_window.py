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
    QObject,
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
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
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
from larksnap.gateway.component_state import (
    ComponentKind,
    ComponentState,
    ComponentStatus,
    SystemStatus,
)
from larksnap.gateway.controller import GatewayController
from larksnap.gateway.event_bus import Event, EventType

# ─── Global Stylesheet ───────────────────────────────────────────────

LIGHT_TECH_STYLE = """
QMainWindow, QDialog {
    background-color: #f5f7fa;
}
QWidget {
    color: #1a1a2e;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #d0d7e2;
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
    color: #3a5a8c;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
}
QLabel {
    color: #2c3e50;
}
QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f0f2f5);
    border: 1px solid #c0c8d4;
    border-radius: 8px;
    padding: 8px 20px;
    color: #2c3e50;
    font-weight: 500;
}
QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e8f0fe, stop:1 #d4e4fc);
    border-color: #0078d4;
    color: #005a9e;
}
QPushButton:pressed {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c8ddf5, stop:1 #b0cceb);
    border-color: #005a9e;
}
QPushButton:disabled {
    background-color: #f0f2f5;
    color: #b0b8c4;
    border-color: #d8dde4;
}
QStatusBar {
    background-color: transparent;
    color: #8899aa;
    font-size: 11px;
}
QMenuBar {
    background-color: rgba(255, 255, 255, 210);
    border-bottom: 1px solid #e0e4ea;
    spacing: 4px;
    padding: 2px;
}
QMenuBar::item {
    padding: 6px 14px;
    border-radius: 6px;
    color: #4a5568;
}
QMenuBar::item:selected {
    background-color: #e8f0fe;
    color: #005a9e;
}
QMenu {
    background-color: #ffffff;
    border: 1px solid #d0d7e2;
    border-radius: 10px;
    padding: 6px;
}
QMenu::item {
    padding: 7px 28px 7px 16px;
    border-radius: 6px;
    color: #2c3e50;
}
QMenu::item:selected {
    background-color: #e8f0fe;
    color: #005a9e;
}
QMenu::separator {
    height: 1px;
    background: #e0e4ea;
    margin: 4px 10px;
}
QTabWidget::pane {
    border: 1px solid #d0d7e2;
    border-radius: 10px;
    background-color: #ffffff;
}
QTabBar::tab {
    background-color: #f0f2f5;
    border: 1px solid #d0d7e2;
    padding: 8px 22px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    color: #6b7c93;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #005a9e;
    border-bottom-color: #ffffff;
    font-weight: 600;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    border: 1px solid #c0c8d4;
    border-radius: 6px;
    padding: 6px 10px;
    color: #2c3e50;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #0078d4;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #b0b8c4;
    background-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #d0d7e2;
    border-radius: 6px;
    selection-background-color: #e8f0fe;
    selection-color: #005a9e;
}
QScrollBar:vertical {
    background: #f0f2f5;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #c0c8d4;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #a0aab8;
}
"""


# ─── UI Event Bridge (cross-thread safe) ──────────────────────────────

class UIEventBridge(QObject):
    """Cross-thread bridge between ``EventBus`` and the main window.

    The ``EventBus`` is synchronous: when ``publish()`` is called from
    any thread, the subscribed handlers run on that *same* thread. If a
    handler is a ``QWidget`` method (e.g. ``MainWindow._on_camera_opened``),
    this violates Qt's thread affinity rules and the UI never updates
    (silently fails or crashes on some platforms).

    This bridge exposes one ``Signal`` per event type. The event bus
    handler calls ``bridge.signal.emit(event)``, which is thread-safe
    and the default connection type is ``QueuedConnection`` — the
    receiving slot runs on the bridge's owning thread (the main thread,
    because the bridge is created in ``MainWindow.__init__``).

    Reference: https://doc.qt.io/qt-6/threads-qobject.html#signals-and-slots-across-threads
    """

    camera_opened = Signal(object)
    camera_closed = Signal(object)
    camera_opening = Signal(object)
    camera_failed = Signal(object)
    camera_read_failed = Signal(object)
    chat_id_obtained = Signal(object)
    notification_enabled = Signal(object)
    notification_disabled = Signal(object)
    # Per-subsystem state change (Camera/Detector/Notifier). Carries
    # an ``Event`` whose ``data`` is a ``ComponentStatus``. Wired to
    # ``MainWindow._on_component_state_changed`` so the status panel
    # and menu checkboxes always reflect the backend in real time.
    component_state_changed = Signal(object)


# ─── Init Overlay (shown when chat_id not obtained) ───────────────────

class InitOverlayWidget(QWidget):
    """Semi-transparent overlay with spinning indicator and status prompt.

    States:
      - ``waiting``: "send /init in Feishu" (legacy behaviour, default)
      - ``loading_camera``: "正在初始化摄像头..." (used during async
        background camera initialisation so the UI can appear instantly
        without waiting for the camera to come up)
      - ``closing_camera``: "正在关闭摄像头..." (shown when the user
        closes the camera, while the non-blocking gateway teardown is
        still in flight on a background thread)
      - ``closed``: hidden
    """

    STATE_WAITING = "waiting"
    STATE_LOADING_CAMERA = "loading_camera"
    STATE_CLOSING_CAMERA = "closing_camera"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._angle = 0
        self._state = self.STATE_WAITING
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._rotate)
        self.hide()

    def show_overlay(self) -> None:
        self.show()
        self.raise_()
        self._timer.start()

    def show_loading_camera(self) -> None:
        """Show the 'initialising camera' state. Used during background init."""
        self._state = self.STATE_LOADING_CAMERA
        self.show_overlay()

    def show_closing_camera(self) -> None:
        """Show the 'closing camera' state. Used while the gateway teardown
        is running in the background after the user clicks the close button.
        """
        self._state = self.STATE_CLOSING_CAMERA
        self.show_overlay()

    def hide_overlay(self) -> None:
        self._timer.stop()
        self.hide()

    def _rotate(self) -> None:
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Frosted glass overlay
        painter.fillRect(self.rect(), QColor(245, 247, 250, 220))

        cx = self.width() // 2
        cy = self.height() // 2 - 30

        # Spinning arc — tech blue
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self._angle)

        pen = QPen(QColor(0, 120, 212), 4, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        rect = QRect(-24, -24, 48, 48)
        painter.drawArc(rect, 0, 270 * 16)

        # Small dot at the leading edge
        painter.setBrush(QColor(0, 120, 212))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(20, -4, 8, 8)

        painter.restore()

        # State-dependent text
        if self._state == self.STATE_LOADING_CAMERA:
            main_text = "正在初始化摄像头…"
            sub_text = "Initialising camera, please wait"
        elif self._state == self.STATE_CLOSING_CAMERA:
            main_text = "正在关闭摄像头…"
            sub_text = "Closing camera, please wait"
        else:
            main_text = "请在聊天软件中发送 /init 命令进行初始化以获取 chat_id"
            sub_text = "Waiting for initialization..."

        painter.setPen(QColor(44, 62, 80))
        painter.setFont(QFont("Segoe UI", 14))
        text_rect = QRect(0, cy + 50, self.width(), 40)
        painter.drawText(text_rect, Qt.AlignCenter, main_text)

        painter.setPen(QColor(107, 124, 147))
        painter.setFont(QFont("Segoe UI", 10))
        sub_rect = QRect(0, cy + 90, self.width(), 30)
        painter.drawText(sub_rect, Qt.AlignCenter, sub_text)

        painter.end()

    def resizeEvent(self, event) -> None:
        if self.parent():
            self.setGeometry(0, 0, self.parent().width(), self.parent().height())
        super().resizeEvent(event)


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
        self._rec_pulse_phase: float = 0.0
        self._show_hud: bool = True
        self._notification_enabled: bool = True
        self._camera_open: bool = False
        # When True, paintEvent renders a solid black background instead
        # of the gray "No Signal" placeholder. Set by clear_frame() after
        # the camera has been closed so the live view area is pure black.
        self._cleared: bool = False

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
        # A new frame has arrived — we're no longer in the "cleared /
        # pure black" post-close state.
        self._cleared = False
        self.update()

    def clear_frame(self) -> None:
        """Reset the preview to a pure-black state with no cached frame.

        Used after the camera is closed so the live view area no longer
        displays the last captured frame. The widget will paint a solid
        black background on the next ``paintEvent``.
        """
        self._pixmap = None
        self._cleared = True
        # The HUD reflects camera state; make sure it is also marked
        # as closed so the floating overlay reads correctly until the
        # init overlay takes over.
        self._camera_open = False
        self.update()

    def update_hud(self, fps: float, detection_count: int, running: bool, recording: bool, notification_enabled: bool = True, camera_open: bool = False) -> None:
        self._fps = fps
        self._detection_count = detection_count
        self._is_running = running
        self._is_recording = recording
        self._notification_enabled = notification_enabled
        self._camera_open = camera_open
        if recording:
            self._rec_pulse_phase = (self._rec_pulse_phase + 0.1) % (2 * 3.14159)
        self.update()

    def update_component_state(self, status: ComponentStatus) -> None:
        """Update HUD-level state derived from a ``ComponentStatus``.

        The persistent top-left ``StatusPanel`` is the primary
        surface for subsystem state. This method is kept for
        backward compatibility with existing call sites, but no
        longer draws any state text — it only flashes the REC-style
        pulse for the recording indicator and forces a repaint.
        """
        # Show the HUD on every state change so the user gets
        # immediate visual feedback even when they're not moving
        # the mouse. ``show_hud_temporarily`` will hide it again
        # after the standard timeout.
        self.show_hud_temporarily()
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
        elif self._cleared:
            # Camera was explicitly closed — render a pure black
            # background so the last frame is not left lingering.
            painter.fillRect(self.rect(), QColor(0, 0, 0))
        else:
            painter.fillRect(self.rect(), QColor("#e8ecf1"))
            painter.setPen(QColor("#8899aa"))
            painter.setFont(QFont("Segoe UI", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Signal")

        # Draw floating HUD overlay
        if self._show_hud:
            self._draw_hud(painter)

        painter.end()

    def _draw_hud(self, painter: QPainter) -> None:
        """Draw semi-transparent HUD in the top-right corner only.

        The camera / detector / notifier status is owned by the
        persistent ``StatusPanel`` widget in the top-left, so the
        HUD no longer duplicates that information here. Only the
        recording indicator (which is transient and only visible
        while recording) is drawn from the HUD, in the top-right
        corner.
        """
        w, h = self.width(), self.height()

        # Top-right: recording indicator — frosted glass card
        if self._is_recording:
            import math
            alpha = int(128 + 127 * math.sin(self._rec_pulse_phase))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.drawRoundedRect(w - 110, 12, 98, 34, 10, 10)
            painter.setPen(QPen(QColor(208, 215, 226, 180), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(w - 110, 12, 98, 34, 10, 10)
            painter.setBrush(QColor(231, 76, 60, alpha))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(w - 96, 20, 14, 14)
            painter.setPen(QColor("#e74c3c"))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(w - 78, 34, "REC")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Show HUD on mouse move."""
        self.show_hud_temporarily()
        super().mouseMoveEvent(event)


# ─── Status Panel (top-left, persistent) ─────────────────────────────

class StatusPanel(QWidget):
    """Persistent top-left panel showing each subsystem's current state.

    Each subsystem (camera, detector, notifier) gets one row: a small
    status dot in the subsystem's state colour, the subsystem's
    English label, and the current state name from
    ``ComponentState.display_name``. The whole card is always visible
    (no auto-hide) so the user can monitor state at a glance.

    A final row shows the live producer FPS so the diagnostics that
    used to live in the (now removed) transient HUD card are
    preserved on a single, always-visible surface.

    Animations:
      - On state change, the row briefly highlights with a colour
        pulse driven by a ``QPropertyAnimation`` on the opacity of
        the new state's colour. This gives the eye a "blink" cue
        that something just changed, but settles to the steady
        state within ~400ms.
      - The status dot colour is animated smoothly to the new
        colour over 250ms rather than snapping, which avoids the
        "flash" effect that would happen on every transition.
    """

    PANEL_MARGIN = 16
    ROW_HEIGHT = 22
    PANEL_PADDING = 12

    # Colour palette for each ComponentState. Single source of
    # truth so the panel stays visually consistent across the app.
    _STATE_COLORS: dict[str, str] = {
        ComponentState.RUNNING.value: "#27ae60",
        ComponentState.STARTING.value: "#0078d4",
        ComponentState.STOPPING.value: "#0078d4",
        ComponentState.FAILED.value: "#e74c3c",
        ComponentState.DISABLED.value: "#7f8c8d",
        ComponentState.STOPPED.value: "#95a5a6",
        ComponentState.IDLE.value: "#95a5a6",
    }

    # English subsystem labels — kept here so the panel and any
    # future log/CSV export stay in lock-step without having to
    # touch multiple files.
    _SUBSYSTEM_LABELS: dict[str, str] = {
        ComponentKind.CAMERA.value: "Camera",
        ComponentKind.DETECTOR.value: "Detector",
        ComponentKind.NOTIFIER.value: "Notifier",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Current and target dot colours per subsystem, used by the
        # animation timer to interpolate. Keys are
        # ``ComponentKind.value``.
        self._current_colors: dict[str, QColor] = {
            ComponentKind.CAMERA.value: QColor(self._STATE_COLORS[ComponentState.IDLE.value]),
            ComponentKind.DETECTOR.value: QColor(self._STATE_COLORS[ComponentState.IDLE.value]),
            ComponentKind.NOTIFIER.value: QColor(self._STATE_COLORS[ComponentState.DISABLED.value]),
        }
        self._target_colors: dict[str, QColor] = dict(self._current_colors)
        self._current_labels: dict[str, str] = {
            ComponentKind.CAMERA.value: ComponentState.IDLE.display_name,
            ComponentKind.DETECTOR.value: ComponentState.IDLE.display_name,
            ComponentKind.NOTIFIER.value: ComponentState.DISABLED.display_name,
        }
        # Live FPS value, refreshed by ``MainWindow._update_hud``.
        self._fps: float = 0.0
        self._anim_step: int = 0  # 0..10, 10 = at target
        self._anim_active: bool = False

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(25)  # ~250ms total
        self._anim_timer.timeout.connect(self._tick_animation)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    def update_fps(self, fps: float) -> None:
        """Update the live FPS readout in the panel."""
        if abs(self._fps - fps) < 0.05:
            return
        self._fps = fps
        self.update()

    def update_state(self, status: ComponentStatus) -> None:
        """Apply a new ``ComponentStatus`` to the panel.

        Triggers a smooth colour and label transition. The label
        changes immediately; the colour is animated to the new
        state over ~250ms.
        """
        kind_key = status.kind.value
        new_label = status.display_name
        new_color_hex = self._STATE_COLORS.get(
            status.state.value, "#95a5a6",
        )
        new_color = QColor(new_color_hex)

        if self._current_labels[kind_key] == new_label and \
                self._target_colors[kind_key].name() == new_color.name():
            return  # no change, skip animation

        self._current_labels[kind_key] = new_label
        self._target_colors[kind_key] = new_color
        self._anim_step = 0
        if not self._anim_active:
            self._anim_active = True
            self._anim_timer.start()
        self.update()

    def _tick_animation(self) -> None:
        """Interpolate colours one step closer to their targets."""
        self._anim_step += 1
        done = True
        for key in self._current_colors:
            cur = self._current_colors[key]
            tgt = self._target_colors[key]
            if cur.red() != tgt.red() or cur.green() != tgt.green() or cur.blue() != tgt.blue():
                t = self._anim_step / 10.0
                nr = int(cur.red() + (tgt.red() - cur.red()) * t)
                ng = int(cur.green() + (tgt.green() - cur.green()) * t)
                nb = int(cur.blue() + (tgt.blue() - cur.blue()) * t)
                self._current_colors[key] = QColor(nr, ng, nb)
                done = False
        if self._anim_step >= 10 or done:
            # Snap to final colours and stop.
            for key in self._current_colors:
                self._current_colors[key] = QColor(self._target_colors[key])
            self._anim_active = False
            self._anim_timer.stop()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(240, 132)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rows = [
            (self._SUBSYSTEM_LABELS[ComponentKind.CAMERA.value], ComponentKind.CAMERA.value),
            (self._SUBSYSTEM_LABELS[ComponentKind.DETECTOR.value], ComponentKind.DETECTOR.value),
            (self._SUBSYSTEM_LABELS[ComponentKind.NOTIFIER.value], ComponentKind.NOTIFIER.value),
        ]

        w = 240
        # 3 subsystem rows + 1 FPS row.
        h = self.PANEL_PADDING * 2 + self.ROW_HEIGHT * (len(rows) + 1)
        # Frosted glass background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 215))
        painter.drawRoundedRect(0, 0, w, h, 10, 10)
        # Subtle border
        painter.setPen(QPen(QColor(208, 215, 226, 200), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(0, 0, w, h, 10, 10)

        font_lbl = QFont("Segoe UI", 10, QFont.DemiBold)
        font_state = QFont("Segoe UI", 10)
        y = self.PANEL_PADDING + 4

        for label, kind_key in rows:
            # Status dot (animated colour)
            color = self._current_colors[kind_key]
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(self.PANEL_PADDING, y - 6, 10, 10)

            # Subsystem label (gray)
            painter.setFont(font_lbl)
            painter.setPen(QColor(74, 85, 104))
            painter.drawText(
                self.PANEL_PADDING + 20, y + 4, label,
            )

            # State name (animated colour)
            painter.setFont(font_state)
            painter.setPen(color)
            state_text = self._current_labels[kind_key]
            painter.drawText(
                self.PANEL_PADDING + 90, y + 4, state_text,
            )

            y += self.ROW_HEIGHT

        # ── FPS row ──
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(74, 85, 104))
        painter.drawEllipse(self.PANEL_PADDING, y - 6, 10, 10)
        painter.setFont(font_lbl)
        painter.setPen(QColor(74, 85, 104))
        painter.drawText(self.PANEL_PADDING + 20, y + 4, "FPS")
        painter.setFont(font_state)
        painter.setPen(QColor(44, 62, 80))
        painter.drawText(self.PANEL_PADDING + 90, y + 4, f"{self._fps:.1f}")

        painter.end()


# ─── Control Dialog ───────────────────────────────────────────────────

class ControlDialog(QDialog):
    """Floating control panel as a dialog, accessible from menu."""

    start_requested = Signal()
    stop_requested = Signal()
    start_recording_requested = Signal()
    stop_recording_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Controls")
        # Height reduced to fit only Start/Stop + Record.
        self.setFixedSize(280, 210)
        self.setStyleSheet(LIGHT_TECH_STYLE)
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        self._is_running = False
        self._is_recording = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Detection
        det_group = QGroupBox("Detection")
        det_layout = QHBoxLayout(det_group)
        det_layout.setSpacing(6)

        self._start_btn = QPushButton("Start")
        self._start_btn.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #e8f8f0,stop:1 #c8f0d8);border-color:#27ae60;color:#1e8449}"
            "QPushButton:hover{background:#27ae60;color:#fff}"
        )
        self._start_btn.clicked.connect(self.start_requested.emit)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #fde8e8,stop:1 #f5c6c6);border-color:#e74c3c;color:#c0392b}"
            "QPushButton:hover{background:#e74c3c;color:#fff}"
        )
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        self._stop_btn.setEnabled(False)

        det_layout.addWidget(self._start_btn)
        det_layout.addWidget(self._stop_btn)

        # Recording
        rec_group = QGroupBox("Recording")
        rec_layout = QHBoxLayout(rec_group)

        self._record_btn = QPushButton("Record")
        self._record_btn.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #fde8e8,stop:1 #f5c6c6);border-color:#e74c3c;color:#c0392b}"
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
        self._record_btn.setEnabled(running)
        if not running:
            self._is_recording = False
            self._record_btn.setText("Record")

    def set_recording(self, recording: bool) -> None:
        self._is_recording = recording
        self._record_btn.setText("Stop Rec" if recording else "Record")

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
        self.setStyleSheet(LIGHT_TECH_STYLE)
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        group = QGroupBox("Live Stats")
        grid = QGridLayout(group)
        grid.setSpacing(6)

        self._fps_val = QLabel("0.0")
        self._fps_val.setStyleSheet("font-size:18px;font-weight:bold;color:#005a9e")
        self._fps_lbl = QLabel("FPS")
        self._fps_lbl.setStyleSheet("font-size:11px;color:#6b7c93")

        self._det_val = QLabel("0")
        self._det_val.setStyleSheet("font-size:18px;font-weight:bold;color:#005a9e")
        self._det_lbl = QLabel("Detections")
        self._det_lbl.setStyleSheet("font-size:11px;color:#6b7c93")

        self._rec_val = QLabel("Off")
        self._rec_val.setStyleSheet("font-size:18px;font-weight:bold;color:#95a5a6")
        self._rec_lbl = QLabel("Recording")
        self._rec_lbl.setStyleSheet("font-size:11px;color:#6b7c93")

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
            f"font-size:18px;font-weight:bold;color:{'#e74c3c' if recording else '#95a5a6'}"
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
        self.setStyleSheet(LIGHT_TECH_STYLE)

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
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #e8f0fe,stop:1 #c8ddf5);border-color:#0078d4;color:#005a9e}"
            "QPushButton:hover{background:#0078d4;color:#fff}"
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

        self.setStyleSheet(LIGHT_TECH_STYLE)
        self.setWindowTitle("LarkSnap")
        self.setMinimumSize(800, 500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)

        # Full video preview as central widget
        self._preview = VideoPreviewWidget()
        self.setCentralWidget(self._preview)

        # Init overlay (shown when chat_id not obtained, or while
        # background camera initialisation is in progress)
        self._init_overlay = InitOverlayWidget(self._preview)
        # Show the "loading camera" overlay immediately so the user sees
        # visual feedback the moment the window appears. It will be
        # replaced by either the /init prompt or hidden once the camera
        # is ready.
        self._init_overlay.show_loading_camera()

        # Persistent top-left status panel showing camera / detector /
        # notifier state in unified ``ComponentState`` wording. Always
        # visible (no auto-hide) so the user can monitor at a glance.
        self._status_panel = StatusPanel(self._preview)
        # Seed the panel with the controller's current state so it
        # doesn't show "Idle" for subsystems that are already
        # running at startup.
        try:
            system_status = self._controller.get_system_status()
            self._status_panel.update_state(system_status.camera)
            self._status_panel.update_state(system_status.detector)
            self._status_panel.update_state(system_status.notifier)
        except Exception:  # noqa: BLE001 — best-effort
            pass

        # No status bar — HUD is overlaid on video

        # Build menu bar
        self._build_menu()

        # System tray
        self._tray_icon = QSystemTrayIcon(self)
        self._tray_icon.setIcon(self._create_tray_icon())
        self._tray_icon.setToolTip("LarkSnap")
        self._tray_icon.activated.connect(self._on_tray_activated)

        tray_menu = QMenu()
        tray_menu.addAction("Show Main Window", self._show_window)
        tray_menu.addSeparator()
        self._tray_open_cam_action = tray_menu.addAction("Open Camera")
        self._tray_open_cam_action.triggered.connect(self._on_open_camera)
        self._tray_close_cam_action = tray_menu.addAction("Close Camera")
        self._tray_close_cam_action.triggered.connect(self._on_close_camera)
        tray_menu.addSeparator()
        self._tray_start_det_action = tray_menu.addAction("Start Detection")
        self._tray_start_det_action.triggered.connect(self._on_start_detection)
        self._tray_stop_det_action = tray_menu.addAction("Stop Detection")
        self._tray_stop_det_action.triggered.connect(self._on_stop_detection)
        tray_menu.addSeparator()
        tray_menu.addAction("Toggle Recording", self._on_record_toggle)
        tray_menu.addSeparator()
        self._tray_enable_notif_action = tray_menu.addAction("Enable Notification")
        self._tray_enable_notif_action.triggered.connect(
            lambda: self._on_notification_toggle(True),
        )
        self._tray_disable_notif_action = tray_menu.addAction("Disable Notification")
        self._tray_disable_notif_action.triggered.connect(
            lambda: self._on_notification_toggle(False),
        )
        tray_menu.addSeparator()
        tray_menu.addAction("Settings", self._show_settings)
        tray_menu.addSeparator()
        tray_menu.addAction("Quit", self._quit_app)
        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.show()

        self._close_to_tray = True

        # Cross-thread UI event bridge. The event bus is synchronous, so
        # handlers run on whatever thread publishes the event. Routing
        # through Qt signals (default QueuedConnection) ensures UI
        # handlers always execute on the main thread.
        self._ui_bridge = UIEventBridge()
        self._ui_bridge.camera_opened.connect(self._on_camera_opened)
        self._ui_bridge.camera_closed.connect(self._on_camera_closed)
        self._ui_bridge.camera_failed.connect(self._on_camera_failed)
        self._ui_bridge.camera_read_failed.connect(self._on_camera_read_failed)
        self._ui_bridge.chat_id_obtained.connect(self._on_chat_id_obtained)
        self._ui_bridge.notification_enabled.connect(self._on_notification_enabled)
        self._ui_bridge.notification_disabled.connect(self._on_notification_disabled)
        # Unified per-subsystem state changes — drives the status
        # panel, the HUD, and the menu enable-state sync.
        self._ui_bridge.component_state_changed.connect(self._on_component_state_changed)

        # Subscribe to gateway events (emit bridge signal as the handler).
        # The emit call is thread-safe; the actual slot runs on the main
        # thread via Qt's queued connection.
        self._controller.event_bus.subscribe(EventType.CAMERA_OPENED, self._ui_bridge.camera_opened.emit)
        self._controller.event_bus.subscribe(EventType.CAMERA_CLOSED, self._ui_bridge.camera_closed.emit)
        self._controller.event_bus.subscribe(EventType.CAMERA_FAILED, self._ui_bridge.camera_failed.emit)
        self._controller.event_bus.subscribe(EventType.CAMERA_READ_FAILED, self._ui_bridge.camera_read_failed.emit)
        self._controller.event_bus.subscribe(EventType.CHAT_ID_OBTAINED, self._ui_bridge.chat_id_obtained.emit)
        self._controller.event_bus.subscribe(EventType.NOTIFICATION_ENABLED, self._ui_bridge.notification_enabled.emit)
        self._controller.event_bus.subscribe(EventType.NOTIFICATION_DISABLED, self._ui_bridge.notification_disabled.emit)
        self._controller.event_bus.subscribe(
            EventType.COMPONENT_STATE_CHANGED,
            self._ui_bridge.component_state_changed.emit,
        )

        # Timers
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.setInterval(33)

        self._hud_timer = QTimer(self)
        self._hud_timer.timeout.connect(self._update_hud)
        self._hud_timer.setInterval(200)

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # ── Camera menu ──
        cam_menu = menu_bar.addMenu("Camera")

        self._cam_device_menu = cam_menu.addMenu("Select Device")
        self._refresh_camera_devices()

        cam_menu.addSeparator()

        self._open_cam_action = cam_menu.addAction("Open Camera")
        self._open_cam_action.setShortcut("Ctrl+O")
        self._open_cam_action.triggered.connect(self._on_open_camera)

        self._close_cam_action = cam_menu.addAction("Close Camera")
        self._close_cam_action.setShortcut("Ctrl+W")
        self._close_cam_action.triggered.connect(self._on_close_camera)
        self._close_cam_action.setEnabled(False)

        cam_menu.addSeparator()

        self._refresh_cam_action = cam_menu.addAction("Refresh Devices")
        self._refresh_cam_action.triggered.connect(self._refresh_camera_devices)

        # ── Detection menu ──
        det_menu = menu_bar.addMenu("Detection")

        self._start_det_action = det_menu.addAction("Start Detection")
        self._start_det_action.setShortcut("Ctrl+S")
        self._start_det_action.triggered.connect(self._on_start_detection)
        self._start_det_action.setEnabled(False)

        self._stop_det_action = det_menu.addAction("Stop Detection")
        self._stop_det_action.setShortcut("Ctrl+D")
        self._stop_det_action.triggered.connect(self._on_stop_detection)
        self._stop_det_action.setEnabled(False)

        det_menu.addSeparator()

        # ── Recording menu ──
        rec_menu = menu_bar.addMenu("Recording")

        self._record_action = rec_menu.addAction("Start Recording")
        self._record_action.setShortcut("Ctrl+R")
        self._record_action.triggered.connect(self._on_record_toggle)
        self._record_action.setEnabled(False)

        # ── Notification menu ──
        notif_menu = menu_bar.addMenu("Notification")

        # Two non-checkable actions so the menu mirrors the Camera
        # menu (separate "Open Camera" / "Close Camera"). The current
        # on/off state is reflected in the StatusPanel / HUD, not in
        # the menu item itself.
        self._enable_notif_action = notif_menu.addAction("Enable Notification")
        self._enable_notif_action.setShortcut("Ctrl+N")
        self._enable_notif_action.triggered.connect(
            lambda: self._on_notification_toggle(True),
        )

        self._disable_notif_action = notif_menu.addAction("Disable Notification")
        self._disable_notif_action.triggered.connect(
            lambda: self._on_notification_toggle(False),
        )

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

    def _refresh_camera_devices(self) -> None:
        """Populate the camera device selection menu."""
        from larksnap.adapters.camera.opencv_adapter import enumerate_cameras

        self._cam_device_menu.clear()
        devices = enumerate_cameras()
        current = self._config.camera.device_index

        if not devices:
            action = self._cam_device_menu.addAction("No cameras found")
            action.setEnabled(False)
            return

        for idx in devices:
            action = self._cam_device_menu.addAction(f"Camera {idx}")
            action.setCheckable(True)
            action.setChecked(idx == current)
            action.triggered.connect(lambda checked, i=idx: self._on_select_camera(i))

    def _on_select_camera(self, device_index: int) -> None:
        """Switch to a different camera device.

        Camera close is non-blocking, so we wait for it to finish
        (bounded) before opening the new device — otherwise the new
        open would race the old close and one of them would be
        rejected by the controller.
        """
        if self._controller.is_busy:
            self._logger.info("Camera operation in progress; ignoring switch")
            return

        if self._controller.is_camera_open:
            self._controller.close_camera()
            # Bounded wait so the UI doesn't freeze if ZMQ teardown
            # is slow. The pipeline's stop() is also bounded, so this
            # returns quickly under normal conditions.
            self._controller.wait_closed(timeout=5.0)
            self.stop_preview()
            # Clear the cached frame so the live view area shows
            # pure black during the device switch instead of the
            # last frame from the previous camera.
            self._preview.clear_frame()

        self._init_overlay.show_loading_camera()
        self._update_action_states()
        try:
            self._controller.open_camera(device_index)
        except Exception as e:
            from larksnap.utils.exceptions import CameraError, GatewayError
            if isinstance(e, (CameraError, GatewayError)):
                # Camera error already handled by event bus
                self._init_overlay.hide_overlay()
            else:
                raise
        self._refresh_camera_devices()
        self._update_action_states()

    def _on_open_camera(self) -> None:
        """Open camera and start preview.

        The gateway is now responsible for starting the pipeline
        (preview) immediately on open. We just trigger the open and
        show the loading overlay. ``_on_camera_opened`` will hide
        the overlay and start the preview timer.
        """
        if self._controller.is_busy:
            self._logger.info("Camera operation in progress; ignoring open request")
            return
        if self._controller.is_camera_open:
            return

        # Show the loading overlay BEFORE the (potentially slow) open
        # call returns, so the user always has visual feedback.
        self._init_overlay.show_loading_camera()
        self._update_action_states()

        try:
            self._controller.open_camera()
        except Exception as e:
            from larksnap.utils.exceptions import CameraError, GatewayError
            if isinstance(e, (CameraError, GatewayError)):
                # Camera error already handled by event bus
                self._init_overlay.hide_overlay()
                self._update_action_states()
                return
            raise
        self._update_action_states()

    def _on_close_camera(self) -> None:
        """Close camera and stop everything.

        Camera close is now non-blocking on the gateway side, so the
        UI thread is never frozen. We hide the preview immediately,
        show a "closing" overlay, and let ``_on_camera_closed`` clean
        up the rest when the background teardown finishes.
        """
        if self._controller.is_busy or not self._controller.is_camera_open:
            self._update_action_states()
            return

        # Show the "closing" overlay so the user sees feedback even
        # though the close happens in the background. This replaces
        # the "initialising camera" loading text with a dedicated
        # "closing camera" message for the close flow.
        self._init_overlay.show_closing_camera()
        # Eagerly clear the preview frame so the live view area
        # switches to pure black immediately, not after the
        # background teardown completes.
        self._preview.clear_frame()
        self.stop_preview()
        self._update_action_states()
        if self._control_dialog:
            self._control_dialog.set_running(False)
            self._control_dialog.set_recording(False)
        if self._stats_dialog:
            self._stats_dialog.update_stats(0, 0, False)

        self._controller.close_camera()

    def _on_start_detection(self) -> None:
        """Start detection (camera must be open)."""
        if not self._controller.is_camera_open:
            return
        self._controller.start_detection()
        self._update_action_states()

    def _on_stop_detection(self) -> None:
        """Stop detection (camera stays open for preview)."""
        self._controller.stop_detection()
        self._update_action_states()
        if self._control_dialog:
            self._control_dialog.set_running(False)

    def _on_camera_failed(self, event: Event) -> None:
        """Handle camera failure event — show error dialog."""
        from larksnap.utils.camera_error_translator import (
            get_solution_hint,
            translate_camera_error,
        )

        data = event.data or {}
        device_index = data.get("device_index", "?")
        raw_error = data.get("error", "未知错误")
        friendly = translate_camera_error(raw_error)
        solution = get_solution_hint(raw_error)

        QMessageBox.critical(
            self,
            "摄像头错误",
            f"无法打开摄像头（设备索引 {device_index}）。\n\n"
            f"原因：{friendly}\n\n"
            f"{solution}",
        )
        self._update_action_states()

    def _on_camera_read_failed(self, event: Event) -> None:
        """Handle camera read failure after max retries — pipeline has stopped."""
        data = event.data or {}
        error_msg = data.get("error", "未知错误")
        QTimer.singleShot(0, lambda: self._handle_camera_read_failed(error_msg))

    def _handle_camera_read_failed(self, error_msg: str) -> None:
        """Show camera read failure dialog on the main thread."""
        from larksnap.utils.camera_error_translator import (
            get_solution_hint,
            translate_camera_error,
        )

        friendly = translate_camera_error(error_msg)
        solution = get_solution_hint(error_msg)

        self._update_action_states()
        self.stop_preview()

        # Dialog with a "重试" option that reopens the camera
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("摄像头读取错误")
        box.setText(
            f"摄像头连续多次读取失败，检测已自动停止。\n\n"
            f"原因：{friendly}\n\n"
            f"{solution}\n\n"
            f"是否立即重新打开摄像头？"
        )
        retry_btn = box.addButton("重新打开摄像头", QMessageBox.AcceptRole)
        box.addButton("稍后再试", QMessageBox.RejectRole)
        box.exec_()

        if box.clickedButton() is retry_btn:
            self._on_open_camera()

    def _on_chat_id_obtained(self, event: Event) -> None:
        """Handle chat_id obtained — hide init overlay."""
        self._logger.info("Chat ID obtained, hiding init overlay")
        self._init_overlay.hide_overlay()

    def _on_camera_opened(self, event: Event) -> None:
        """Handle camera opened — hide the loading overlay and start preview.

        Called via the event bus when the background camera initialisation
        thread (started in ``main.py``) successfully opens the camera.
        """
        self._logger.info("Camera opened event received; hiding loading overlay")
        self._init_overlay.hide_overlay()
        self._update_action_states()
        # Pipeline is now running with detection. Start the preview
        # timer so the user sees frames.
        if not self._preview_timer.isActive():
            self.start_preview()
        # If user has not obtained chat_id, show the init prompt
        if not self._config.notifier.chat_id:
            self._init_overlay.show_overlay()

    def _on_camera_closed(self, event: Event) -> None:
        """Handle camera closed — fired by the background close worker.

        The close itself is non-blocking (returns immediately from
        ``controller.close_camera()``), so this handler runs on the
        main thread some time later when ZMQ teardown is complete.
        """
        self._logger.info("Camera closed event received; finalizing UI")
        # Clear the preview so the live view area shows pure black
        # instead of the last captured frame. The closing overlay
        # covers this for most of the close window, but if the
        # overlay was already hidden (e.g. a fast close path) the
        # black background is what the user sees.
        self._preview.clear_frame()
        self._init_overlay.hide_overlay()
        # Reset diagnostic flags so the next open can re-log first frame
        self._first_frame_logged = False
        self._no_frame_warned = False
        self._update_action_states()

    def _on_notification_enabled(self, event: Event) -> None:
        """Handle notification enabled — sync UI."""
        self._sync_notification_ui(True)

    def _on_notification_disabled(self, event: Event) -> None:
        """Handle notification disabled — sync UI."""
        self._sync_notification_ui(False)

    def _on_component_state_changed(self, event: Event) -> None:
        """Handle a unified component state change from the gateway.

        Updates the persistent status panel, the HUD, and the
        menu/tray checkboxes for the affected subsystem so all
        surfaces stay in lock-step with the backend.

        ``event.data`` is a ``ComponentStatus`` instance (camera,
        detector, or notifier).
        """
        status = event.data
        if not isinstance(status, ComponentStatus):
            return

        # 1. Persistent top-left status panel — primary surface.
        self._status_panel.update_state(status)

        # 2. In-video HUD — secondary surface (transient, auto-hides).
        self._preview.update_component_state(status)

        # 3. Menu / tray enable-state sync.
        if status.kind is ComponentKind.NOTIFIER:
            enabled = status.state is not ComponentState.DISABLED
            self._sync_notification_ui(enabled)

        elif status.kind is ComponentKind.DETECTOR:
            # Detector: menu enable state is driven here so that
            # STARTING/RUNNING/STOPPING/FAILED/STOPPED transitions
            # are all reflected in Start/Stop Detection.
            self._update_action_states()

        elif status.kind is ComponentKind.CAMERA:
            # Camera: menu enable/disable is already driven by
            # _update_action_states, called from the camera-opened/
            # camera-closed/closing-camera handlers. Re-run it here
            # so STARTING/STOPPING/FAILED/STOPPED transitions are
            # reflected too (e.g. disable Open Camera while OPENING).
            self._update_action_states()

    # ── Legacy handlers (for control dialog compatibility) ──

    def _on_start(self) -> None:
        """Legacy: open camera + start detection."""
        self._on_open_camera()
        if self._controller.is_camera_open:
            self._on_start_detection()

    def _on_stop(self) -> None:
        """Legacy: close camera."""
        self._on_close_camera()

    def _on_record_toggle(self) -> None:
        if self._controller.is_recording:
            self._controller.stop_recording()
        else:
            self._controller.start_recording()
        self._update_action_states()
        if self._control_dialog:
            self._control_dialog.set_recording(self._controller.is_recording)

    def _on_notification_toggle(self, enabled: bool) -> None:
        """Enable or disable notifications from menu / tray.

        Called with ``True`` when the user picks "Enable Notification"
        and ``False`` when the user picks "Disable Notification".
        Idempotent: if the requested state already matches the
        backend, the controller call is a no-op.
        """
        if enabled == self._controller.is_notification_enabled:
            return
        if enabled:
            self._controller.enable_notification()
        else:
            self._controller.disable_notification()
        self._sync_notification_ui(enabled)

    def _sync_notification_ui(self, enabled: bool) -> None:
        """Sync notification state across menu, tray, and HUD.

        The menu items themselves are no longer checkable — they
        reflect the on/off state by being enabled or disabled, just
        like the Camera menu's Open/Close pair. Only the action
        representing the *opposite* of the current state is enabled
        (you can disable a running notifier, but not "disable" a
        notifier that's already off).
        """
        self._enable_notif_action.setEnabled(not enabled)
        self._disable_notif_action.setEnabled(enabled)
        self._tray_enable_notif_action.setEnabled(not enabled)
        self._tray_disable_notif_action.setEnabled(enabled)
        self._preview.show_hud_temporarily()

    # ── State machine ─────────────────────────────────────────────────

    def _update_action_states(self) -> None:
        """Update all menu/toolbar action states based on controller state.

        State machine:
          IDLE        → camera off, detection off
          OPENING     → camera init in progress (all actions disabled)
          CLOSING     → camera teardown in progress (all actions disabled)
          CAMERA_ON   → camera on, detection off (preview only)
          DETECTING   → camera on, detection running

        The notification menu uses two separate Enable/Disable actions
        whose enabled state is derived from ``is_notification_enabled``,
        mirroring the Camera menu.
        """
        cam_open = self._controller.is_camera_open
        busy = self._controller.is_busy
        det_active = self._controller.is_detection_active
        recording = self._controller.is_recording
        notif_on = self._controller.is_notification_enabled

        # While open/close is in flight, lock out the camera actions
        # so the user can't queue conflicting requests. Detection and
        # recording actions are also locked out because they depend
        # on a stable camera.
        open_enabled = not cam_open and not busy
        close_enabled = cam_open and not busy

        # Camera actions
        self._open_cam_action.setEnabled(open_enabled)
        self._close_cam_action.setEnabled(close_enabled)
        self._tray_open_cam_action.setEnabled(open_enabled)
        self._tray_close_cam_action.setEnabled(close_enabled)
        self._refresh_cam_action.setEnabled(open_enabled)

        # Detection actions
        self._start_det_action.setEnabled(cam_open and not det_active and not busy)
        self._stop_det_action.setEnabled(det_active and not busy)
        self._tray_start_det_action.setEnabled(cam_open and not det_active and not busy)
        self._tray_stop_det_action.setEnabled(det_active and not busy)

        # Recording (only when detection is active)
        self._record_action.setEnabled(det_active and not busy)
        self._record_action.setText("Stop Recording" if recording else "Start Recording")

        # Notification: enable whichever action represents the
        # transition the user is allowed to make, mirroring the
        # Camera menu (Open ↔ Close).
        self._enable_notif_action.setEnabled(not notif_on)
        self._disable_notif_action.setEnabled(notif_on)
        self._tray_enable_notif_action.setEnabled(not notif_on)
        self._tray_disable_notif_action.setEnabled(notif_on)

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
            self._control_dialog.start_recording_requested.connect(self._on_record_toggle)
            self._control_dialog.stop_recording_requested.connect(self._on_record_toggle)
        self._control_dialog.set_running(self._controller.is_running)
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
        # Reset diagnostic flags so a re-open after close can re-log
        self._first_frame_logged = False
        self._no_frame_warned = False
        self._preview_timer.start()
        self._hud_timer.start()
        self._update_action_states()
        self._sync_notification_ui(self._controller.notification_enabled)

    def stop_preview(self) -> None:
        self._preview_timer.stop()
        self._hud_timer.stop()
        self._update_action_states()

    def _update_preview(self) -> None:
        frame = self._controller.get_latest_frame()
        if frame is not None:
            # First-frame diagnostic — helps catch pipeline issues
            if not self._first_frame_logged:
                self._first_frame_logged = True
                self._logger.info(
                    "First preview frame received: shape=%s, dtype=%s",
                    frame.shape, frame.dtype,
                )
            results = self._controller.get_latest_results()
            self._preview.update_frame(frame, results)
        elif self._controller.is_camera_open and not self._first_frame_logged:
            # Log a warning once if camera is open but no frames yet
            if not hasattr(self, "_no_frame_warned") or not self._no_frame_warned:
                self._no_frame_warned = True
                self._logger.warning(
                    "Camera is open but no frame has been cached yet. "
                    "Pipeline may not be running (camera_open=%s, "
                    "detection_running=%s)",
                    self._controller.is_camera_open,
                    self._controller.is_detection_active,
                )
        # Keep init overlay sized to preview
        if self._init_overlay.isVisible():
            self._init_overlay.setGeometry(0, 0, self._preview.width(), self._preview.height())
        # Always size and position the status panel in the top-left.
        self._status_panel.setGeometry(
            StatusPanel.PANEL_MARGIN,
            StatusPanel.PANEL_MARGIN,
            self._status_panel.sizeHint().width(),
            self._status_panel.sizeHint().height(),
        )

    def _update_hud(self) -> None:
        self._preview.update_hud(
            fps=self._controller.producer_fps,
            detection_count=self._controller.detection_count,
            running=self._controller.is_running,
            recording=self._controller.is_recording,
            notification_enabled=self._controller.notification_enabled,
            camera_open=self._controller.is_camera_open,
        )
        # Mirror FPS into the persistent top-left panel so the
        # diagnostic data lives on a single surface.
        self._status_panel.update_fps(self._controller.producer_fps)
        if self._stats_dialog and self._stats_dialog.isVisible():
            self._stats_dialog.update_stats(
                fps=self._controller.producer_fps,
                detections=self._controller.detection_count,
                recording=self._controller.is_recording,
            )

    # ── Keyboard ──

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Escape and self.isFullScreen():
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
            self._cleanup_and_quit()
            super().closeEvent(event)

    def _cleanup_and_quit(self) -> None:
        """Stop all resources and quit the application."""
        self.stop_preview()
        self._controller.stop()
        self._tray_icon.hide()

        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _create_tray_icon(self) -> QIcon:
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(0, 120, 212))
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
        self._cleanup_and_quit()
