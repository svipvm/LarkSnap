"""Snapshot service — detector-owned local data persistence with
per-class throttling.

Owns the *local* persistence path for detection events: when the
detector produces a result, the snapshot service decides whether
to write a JPEG to disk and, if so, returns the path so the
controller can hand it to the notification service. The notifier
no longer participates in the save flow — it merely receives the
path that the snapshot service already produced.

Responsibility split (the architectural contract this module
enforces):

* **Snapshot saving** belongs to the **detection path**. It is
  triggered by the detector module, runs whenever the detector is
  in the started state, and is *unaffected* by the notifier's
  on/off switch. The user expects the recorded evidence to be
  complete regardless of whether they have silenced the chat
  notifications.

* **Notification dispatch** belongs to the **notifier path**.
  Critically, the notifier is now **driven by the save**: it
  fires only on the frame where this service reports a successful
  save (``SaveOutcome.saved is True``). This makes the
  per-class ``save_interval`` the single source of truth for
  notification frequency — there is no separate cooldown in
  the notifier.

* **Per-class throttling**. To avoid filling the disk with
  near-duplicate images of the same detection (e.g. a person
  standing in front of the camera for an hour), the service
  applies a per-class interval: once an image has been saved
  for a class, subsequent frames containing *only that class*
  within ``save_interval`` seconds are suppressed. Frames that
  introduce a *new* class (not in cooldown) are saved, and the
  cooldown clock for every class present in the frame is
  refreshed. Multi-class frames are coalesced into a single
  image that documents the whole scene.

The two services are stitched together by ``GatewayController``,
which calls the snapshot service first and hands the resulting
``SaveOutcome`` to the notification service. The snapshot
service does not know about the notifier; the notifier does
not know about the snapshot save flow.

Concurrency:

* ``save_snapshot`` is the hot-path entry point. It writes a
  JPEG to disk; ``cv2.imwrite`` for a 1080p JPEG is in the
  single-digit millisecond range, so a synchronous save is
  acceptable on the ZMQ result subscriber thread. The controller
  is responsible for copying the frame out of the live buffer
  before calling this method.
* Per-class cooldown timestamps are mutated under a single
  lock, so concurrent ``save_snapshot`` calls from different
  threads (e.g. multiple producer threads, or the controller's
  worker pool) see a consistent view.
* All disk I/O failures are logged and swallowed — a failed
  save must not stall the detector pipeline.
* The service publishes a ``SNAPSHOT_SAVED`` event on success
  so downstream consumers (e.g. UI status panel) can react.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from larksnap.adapters.detector.interface import DetectionResult
from larksnap.gateway.event_bus import Event, EventBus, EventType


# ``SaveOutcome.reason`` is a small enum-shaped string set
# so tests can assert *why* a save was suppressed. New values
# are backwards-compatible additions; the documented values
# are stable.
_REASON_OK = "ok"
_REASON_DISABLED = "disabled"
_REASON_NO_FRAME = "no_frame"
_REASON_NO_RESULTS = "no_results"
_REASON_COOLDOWN = "cooldown"
_REASON_WRITE_FAILED = "write_failed"


@dataclass(frozen=True)
class SaveOutcome:
    """Result of a :py:meth:`SnapshotService.save_snapshot` call.

    Attributes:
        saved: ``True`` when a JPEG was actually written.
        path: Absolute file path of the saved JPEG, or ``None``
            when ``saved`` is ``False``.
        saved_labels: The subset of ``results`` labels whose
            cooldown had elapsed and therefore triggered this
            save. Always empty when ``saved`` is ``False``.
        reason: A short tag describing why the save did or did
            not happen. One of the ``_REASON_*`` constants
            (``"ok"``, ``"cooldown"``, ``"disabled"``,
            ``"no_frame"``, ``"no_results"``,
            ``"write_failed"``). Useful for telemetry and
            tests; the value is stable across releases.
    """

    saved: bool
    path: str | None
    saved_labels: list[str] = field(default_factory=list)
    reason: str = _REASON_OK


@dataclass
class SnapshotServiceConfig:
    """Configuration owned by :class:`SnapshotService`.

    Only the persistence-side knobs live here. The notification
    template and dispatch logic remain in
    :class:`larksnap.gateway.notification_service.NotificationService`.
    """

    snapshot_dir: str = "snapshots"
    # ``enabled`` mirrors the *detector* state (running / paused).
    # It is the controller's job to flip this — the snapshot
    # service itself does not poll the gateway state, because that
    # would couple the two layers and reintroduce the very
    # dependency the refactor is removing.
    enabled: bool = True
    # Per-class throttle window, in seconds. The same label will
    # not trigger a save more than once per ``save_interval``
    # seconds. Default 30 s — chosen to match the previous
    # notification cooldown so the user-visible behaviour stays
    # the same after the refactor.
    #
    # This is the *only* frequency control in the system now:
    # the notification service has no separate cooldown of its
    # own, so flipping this value changes both the save and
    # the notification cadence together.
    save_interval: float = 30.0
    # When True (default), a frame that introduces a *new* class
    # is always saved even if the throttle window is in effect
    # for other classes in the same frame. The fresh image is
    # the user's only proof that the new class was detected, so
    # we never suppress it. When False, a frame is only saved
    # if *every* class in it is past the throttle.
    save_on_new_class: bool = True


class SnapshotService:
    """Detector-owned local persistence for detection snapshots.

    The service implements a per-class throttle: once a snapshot
    has been saved for a given class label, subsequent frames
    containing *only* that class within ``config.save_interval``
    seconds are suppressed. Frames that introduce a *new* class
    (not in cooldown) are saved, and the cooldown clock for
    every class present in the frame is refreshed.

    Usage::

        cfg = SnapshotServiceConfig(
            snapshot_dir="/var/lib/larksnap/snapshots",
            save_interval=30.0,
        )
        service = SnapshotService(config=cfg, event_bus=bus)
        outcome = service.save_snapshot(frame, results)
        if outcome.saved:
            notifier.send_message(snapshot_path=outcome.path, ...)

    The notifier no longer decides *when* to send; it fires
    only when the snapshot service reports ``outcome.saved``.
    That single bit is the source of truth for "should a
    notification go out for this frame?".
    """

    def __init__(
        self,
        config: SnapshotServiceConfig,
        event_bus: EventBus,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._logger = logging.getLogger("larksnap.snapshot_service")
        # ``_save_count`` is a tiny counter used by tests to assert
        # the service ran. Wrapped in a lock so the rare cross-thread
        # caller (e.g. a unit test that drives ``save_snapshot``
        # from multiple threads) sees a consistent value.
        self._counter_lock = threading.Lock()
        self._save_count: int = 0
        # A monotonic sequence number folded into every filename so
        # two saves within the same microsecond (e.g. on a burst
        # of detections from multiple producer threads) get
        # distinct files. The wall-clock timestamp alone is not
        # fine-grained enough on Windows where ``datetime.now``
        # can have millisecond resolution.
        self._seq_lock = threading.Lock()
        self._seq: int = 0
        # Per-class cooldown state. The dict maps class label →
        # wall-clock seconds of the most recent save. All access
        # goes through ``_cooldown_lock`` so a concurrent
        # ``save_snapshot`` call cannot see a partially-updated
        # state.
        self._cooldown_lock = threading.Lock()
        self._last_save_time: dict[str, float] = {}

    @property
    def config(self) -> SnapshotServiceConfig:
        return self._config

    @property
    def is_enabled(self) -> bool:
        """True when the detector is in the started state.

        Mirrored from the controller: the snapshot service itself
        does not know about the gateway state, so the controller
        flips this flag on ``start_detection`` / ``stop_detection``.
        Treating the flag as authoritative keeps the snapshot
        service free of cross-layer coupling.
        """
        return self._config.enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable snapshot saving (driven by detector state).

        The controller is expected to call this with ``True`` on
        ``start_detection`` and ``False`` on ``stop_detection`` so
        the snapshot service mirrors the detector's on/off
        semantics. Direct callers in tests may flip the flag to
        exercise the gate.

        The per-class cooldown clock is **not** reset by
        ``set_enabled``; the clock represents "time since this
        class was last persisted", which is a property of the
        evidence, not of the detector's on/off cycle.
        """
        self._config.enabled = enabled

    @property
    def save_count(self) -> int:
        """Number of snapshots successfully written since startup.

        Exposed for tests and status panels. Counts only the
        successful writes — a failed ``cv2.imwrite`` does not
        increment the counter.
        """
        with self._counter_lock:
            return self._save_count

    def time_to_next_save(self, label: str, now: float | None = None) -> float:
        """Return the seconds until ``label`` is eligible to save again.

        Used by the UI status panel ("next notification for
        person in 17 s") and by tests. A value of ``0`` means
        the label is already eligible. ``-1`` means the label
        has never been saved.
        """
        if now is None:
            now = time.time()
        with self._cooldown_lock:
            last = self._last_save_time.get(label)
        if last is None:
            return -1.0
        elapsed = now - last
        remaining = self._config.save_interval - elapsed
        return max(0.0, remaining)

    def reset_cooldowns(self) -> None:
        """Forget all per-class cooldowns (test helper / admin command)."""
        with self._cooldown_lock:
            self._last_save_time.clear()

    def save_snapshot(
        self,
        frame: np.ndarray | None,
        results: Iterable[DetectionResult] | None = None,
    ) -> SaveOutcome:
        """Write a JPEG snapshot when the per-class throttle allows.

        The decision tree:

        1. ``enabled`` is False (detector paused) → no save,
           return ``SaveOutcome(saved=False, reason="disabled")``.
        2. ``frame`` is None → no save, return
           ``SaveOutcome(saved=False, reason="no_frame")``.
        3. ``results`` is empty/None → no save, return
           ``SaveOutcome(saved=False, reason="no_results")``.
        4. Compute the subset of labels whose cooldown has
           elapsed.
        5. If that subset is empty → no save, return
           ``SaveOutcome(saved=False, reason="cooldown")``.
        6. Write the JPEG, refresh the cooldown clock for every
           label in the subset (and, with
           ``save_on_new_class=True``, every label present in
           the frame).
        7. Return ``SaveOutcome(saved=True, path=…, saved_labels=…)``.

        The method is non-blocking from the orchestrator's
        point of view — it returns synchronously after the
        JPEG is on disk. ``cv2.imwrite`` for a 1080p JPEG is
        a few milliseconds; the controller's worker pool
        keeps that off the ZMQ result thread.
        """
        if not self._config.enabled:
            return SaveOutcome(saved=False, path=None, reason=_REASON_DISABLED)
        if frame is None:
            return SaveOutcome(saved=False, path=None, reason=_REASON_NO_FRAME)

        # Materialise the iterable once: the cooldown check needs
        # to iterate twice (once for the subset, once for the
        # write-back under ``save_on_new_class``), and we don't
        # want the iterable to be exhausted by the first pass.
        results_list = list(results) if results is not None else []
        if not results_list:
            return SaveOutcome(saved=False, path=None, reason=_REASON_NO_RESULTS)

        now = time.time()
        interval = self._config.save_interval
        present_labels: list[str] = [r.label for r in results_list]

        with self._cooldown_lock:
            eligible: list[str] = []
            for label in present_labels:
                last = self._last_save_time.get(label, 0.0)
                if now - last >= interval:
                    eligible.append(label)

            if not eligible:
                # Every label is in cooldown — no save.
                self._logger.debug(
                    "Snapshot suppressed (cooldown): %s",
                    present_labels,
                )
                return SaveOutcome(
                    saved=False, path=None, saved_labels=[],
                    reason=_REASON_COOLDOWN,
                )

            # In strict mode (``save_on_new_class=False``) we
            # only save when *every* label in the frame is
            # past its cooldown. This is the
            # "all-or-nothing" semantics some deployments
            # prefer: a frame that mixes an in-cooldown
            # class with a fresh one is treated as a
            # duplicate of the previous save and is dropped.
            if (
                not self._config.save_on_new_class
                and len(eligible) != len(present_labels)
            ):
                self._logger.debug(
                    "Snapshot suppressed (strict mode): %s",
                    present_labels,
                )
                return SaveOutcome(
                    saved=False, path=None, saved_labels=[],
                    reason=_REASON_COOLDOWN,
                )

            # Write the file under the same lock so two concurrent
            # ``save_snapshot`` calls cannot both decide they are
            # eligible and double-save. (The lock also protects
            # the per-class cooldown update below.)
            path = self._write_jpeg_locked(frame)
            if path is None:
                return SaveOutcome(
                    saved=False, path=None, saved_labels=[],
                    reason=_REASON_WRITE_FAILED,
                )

            # Refresh the cooldown clock for every label in the
            # frame. The saved image is the user's only record
            # of the whole scene, so refreshing the cooldowns
            # for all labels avoids a follow-up save 100 ms
            # later that would only differ in sub-millisecond
            # details.
            refreshed = list(present_labels)
            for label in refreshed:
                self._last_save_time[label] = now

        # Bump the success counter outside the cooldown lock —
        # the counter is independent of the throttle state and
        # the lock ordering above is enough to guarantee
        # consistency for the throttle.
        with self._counter_lock:
            self._save_count += 1

        # Publish the success event best-effort. The file is
        # on disk; a failed event publish must not undo the save.
        try:
            self._event_bus.publish(
                Event(
                    type=EventType.SNAPSHOT_SAVED,
                    data=path,
                    source="snapshot_service",
                )
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            self._logger.error(
                "Failed to publish SNAPSHOT_SAVED: %s", e
            )

        self._logger.info(
            "Snapshot saved: %s (labels=%s)", path, refreshed,
        )
        return SaveOutcome(
            saved=True, path=path, saved_labels=eligible,
            reason=_REASON_OK,
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _write_jpeg_locked(self, frame: np.ndarray) -> str | None:
        """Write a single JPEG to disk.

        Must be called with ``_cooldown_lock`` held; that is
        the only reason the lock name carries the ``_locked``
        suffix. Splitting the lock out of the per-class state
        would require nested locks and complicate the deadlock
        story.

        Returns the absolute file path on success, ``None`` on
        any I/O failure (the failure is already logged).
        """
        try:
            snapshot_dir = Path(self._config.snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            # Microsecond resolution *and* a per-instance sequence
            # number — together they guarantee a unique filename
            # even when several producer threads call
            # ``save_snapshot`` in the same microsecond (which
            # happens on Windows where ``datetime.now`` has only
            # millisecond resolution).
            with self._seq_lock:
                self._seq += 1
                seq = self._seq
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"snapshot_{timestamp}_{seq:08d}.jpg"
            filepath = snapshot_dir / filename
            cv2.imwrite(str(filepath), frame)
            return str(filepath)
        except Exception as e:
            self._logger.error("Failed to save snapshot: %s", e)
            return None

    def reset(self) -> None:
        """Reset the success counter (test helper)."""
        with self._counter_lock:
            self._save_count = 0
