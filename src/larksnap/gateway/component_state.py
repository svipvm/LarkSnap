"""Unified component state model for LarkSnap subsystems.

The system has three independently controllable subsystems:

  - Camera  : captures frames (OpenCV / DSHOW / MSMF)
  - Detector: runs object detection on captured frames
  - Notifier: dispatches alerts to Feishu (Lark)

Each subsystem goes through the same lifecycle vocabulary:

    IDLE ─► STARTING ─► RUNNING ─► STOPPING
       ▲       │           │          │
       │       │           ▼          │
       │       └────► STOPPING ◄──────┘
       │                │
       └────────────────┘ (STOPPED)
       ▲
       └── FAILED (terminal until reset)

The orchestrator (GatewayController) keeps a single ``GatewayState``
enum for *cross-subsystem* state. This module adds per-subsystem
state in the same vocabulary so the UI can render one consistent
"Camera: Disabled" / "Detector: Running" / "Notifier: Disabled"
row for each.

Design notes:
  - ``ComponentState`` is the per-subsystem enum (one value per state).
  - ``ComponentKind`` discriminates which subsystem the state refers to.
  - ``ComponentStatus`` is a snapshot bundle returned by the
    controller's ``get_component_status()`` — useful for the UI to
    fetch the whole picture in a single lock acquisition.
  - ``component_state_from_legacy`` maps the existing mixed legacy
    flags (e.g. ``is_running``, ``is_camera_open``) onto the
    unified enum so we can refactor gradually.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ComponentKind(str, enum.Enum):
    """Identifies which subsystem a state refers to."""

    CAMERA = "camera"
    DETECTOR = "detector"
    NOTIFIER = "notifier"


class ComponentState(str, enum.Enum):
    """Per-subsystem lifecycle state.

    Values are short, stable, lower-snake-case strings so they can be
    serialised to logs and config files without churn.
    """

    IDLE = "idle"            # not initialised, no resources held
    STARTING = "starting"    # async open / load in progress
    RUNNING = "running"      # fully operational
    STOPPING = "stopping"    # async close in progress
    STOPPED = "stopped"      # closed cleanly, ready to be re-started
    FAILED = "failed"        # open/load/connect failed; needs reset
    DISABLED = "disabled"    # disabled by user (e.g. notifier off)

    @property
    def display_name(self) -> str:
        """Human-readable English label for the UI.

        Centralised so the top-left status panel, the HUD, and the
        log messages all use the exact same wording.
        """
        return _DISPLAY_NAMES[self]


_DISPLAY_NAMES: dict[ComponentState, str] = {
    ComponentState.IDLE:     "Idle",
    ComponentState.STARTING: "Starting",
    ComponentState.RUNNING:  "Running",
    ComponentState.STOPPING: "Stopping",
    ComponentState.STOPPED:  "Stopped",
    ComponentState.FAILED:   "Failed",
    ComponentState.DISABLED: "Disabled",
}


# Tuple-of-tuples keeps the transition table immutable + hashable.
# Subsystem start with the lowest level — the camera must be RUNNING
# before the detector can run; the notifier can be enabled regardless
# of the others.
ALLOWED_TRANSITIONS: dict[ComponentState, frozenset[ComponentState]] = {
    ComponentState.IDLE:     frozenset({ComponentState.STARTING, ComponentState.FAILED}),
    ComponentState.STARTING: frozenset({ComponentState.RUNNING, ComponentState.STOPPING, ComponentState.FAILED}),
    ComponentState.RUNNING:  frozenset({ComponentState.STOPPING, ComponentState.FAILED}),
    ComponentState.STOPPING: frozenset({ComponentState.STOPPED, ComponentState.FAILED}),
    ComponentState.STOPPED:  frozenset({ComponentState.IDLE, ComponentState.STARTING}),
    ComponentState.FAILED:   frozenset({ComponentState.STOPPING, ComponentState.IDLE}),
    ComponentState.DISABLED: frozenset({ComponentState.RUNNING, ComponentState.IDLE}),
}


@dataclass(frozen=True)
class ComponentStatus:
    """A single point-in-time snapshot of one subsystem's state."""

    kind: ComponentKind
    state: ComponentState
    # Optional human-readable context, e.g. the last error message
    # when ``state is FAILED``. Always None for the happy path.
    detail: str | None = None

    @property
    def display_name(self) -> str:
        return self.state.display_name

    @property
    def is_active(self) -> bool:
        """True if the subsystem is currently doing work (running)."""
        return self.state in (ComponentState.RUNNING, ComponentState.STARTING)

    @property
    def is_transitioning(self) -> bool:
        """True if the subsystem is mid-transition (starting or stopping)."""
        return self.state in (ComponentState.STARTING, ComponentState.STOPPING)


@dataclass(frozen=True)
class SystemStatus:
    """A complete snapshot of all three subsystems at one point in time."""

    camera: ComponentStatus
    detector: ComponentStatus
    notifier: ComponentStatus


def is_valid_transition(current: ComponentState, target: ComponentState) -> bool:
    """Return True if ``current → target`` is a legal state transition.

    Mirrors the matrix in ``ALLOWED_TRANSITIONS`` but is a function so
    callers don't need to import the table directly.
    """
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())


def component_state_from_legacy(
    *,
    is_open: bool,
    is_busy: bool,
    is_running: bool,
    is_failed: bool = False,
    is_disabled: bool = False,
) -> ComponentState:
    """Map the existing boolean flags onto a unified ``ComponentState``.

    Used by the controller (which already maintains the booleans) to
    derive the new enum without an additional state machine.

    The mapping favours ``is_busy`` and ``is_failed`` as the most
    specific signals. ``is_running`` only applies once the
    subsystem is past the open/load phase.
    """
    if is_disabled:
        return ComponentState.DISABLED
    if is_failed:
        return ComponentState.FAILED
    if is_busy and not is_open:
        return ComponentState.STARTING
    if is_busy and is_open:
        return ComponentState.STOPPING
    if is_running:
        return ComponentState.RUNNING
    if is_open:
        return ComponentState.STOPPED
    return ComponentState.IDLE
