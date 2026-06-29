from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from larksnap.adapters.base import BaseAdapter


@dataclass
class BBox:
    """Bounding box coordinates for a detection result."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class DetectionResult:
    """Single object detection result."""

    label: str
    confidence: float
    bbox: BBox
    mask: np.ndarray | None = None


def filter_results_by_classes(
    results: list[DetectionResult],
    target_classes: list[str] | None,
) -> list[DetectionResult]:
    """Restrict detection results to the configured monitoring set.

    Implements the *monitoring contract* for the detector adapters:
    ``target_classes`` is the user's explicit declaration of which
    classes the system is actively monitoring. The detector returns
    ONLY detections whose label is in that set — anything else is
    dropped, deterministically. This replaces the legacy behaviour
    where the model output was forwarded verbatim and the
    notification side had to guess what the user actually wanted.

    Contract:

    * ``target_classes`` is ``None`` → the monitoring set is
      *unset*; the function is a no-op and returns the input list
      unchanged. Use this for layers that want to delegate the
      filtering decision to someone else (e.g. the notification
      service that defers to the detector adapter).
    * ``target_classes`` is an empty list ``[]`` → return ``[]``.
      An empty monitoring set means "monitor nothing"; the function
      never falls back to "monitor everything", because that would
      re-introduce the arbitrary behaviour the filter exists to
      prevent.
    * Matching is case-insensitive and whitespace-tolerant. COCO
      class names are lowercase by convention, but a user typing
      ``Person`` in the settings UI should still hit the ``person``
      class.
    * ``None`` entries and empty strings inside ``target_classes``
      are silently ignored (defensive — the UI normally strips them
      but a hand-edited config file might not).
    * The input list is not mutated; a new list is returned.

    This helper is the single source of truth for the filter logic.
    Both the detector adapters (seg / mock) and the notification
    service call it, so the behaviour stays consistent across the
    pipeline.
    """
    # ``None`` is the documented "no filter" sentinel — a layer
    # that has been told not to enforce the contract itself
    # forwards the input unchanged so the upstream gatekeeper
    # remains the single source of truth.
    if target_classes is None:
        return list(results)

    # Build the lookup set once. Lower-cased + stripped so the
    # per-result comparison is O(1) and case-folding is consistent
    # regardless of the caller's input style.
    wanted: set[str] = {
        c.strip().lower() for c in target_classes if c and c.strip()
    }
    if not wanted:
        return []

    return [r for r in results if r.label.lower() in wanted]


class DetectorAdapter(BaseAdapter):
    """Abstract detector adapter interface for object detection."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run detection on a video frame."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the detection model."""
        pass

    def initialize(self) -> None:
        self.load_model()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.unload_model()

    def release(self) -> None:
        pass
