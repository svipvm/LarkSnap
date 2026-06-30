"""Tests for the *monitoring contract* enforced by ``target_classes``.

The contract: when a user lists the classes they want to monitor in
``detector.target_classes``, the system must behave deterministically
and emit ONLY detections whose label is in that set. Anything else
is dropped, never silently forwarded.

These tests cover the helper, both detector adapters, and the
notification service's defense-in-depth filter.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import numpy as np
import pytest

from larksnap.adapters.detector.interface import (
    BBox,
    DetectionResult,
    filter_results_by_classes,
)
from larksnap.adapters.detector.mock_adapter import MockDetectorAdapter
from larksnap.config.models import DetectorConfig
from larksnap.gateway.event_bus import EventType
from larksnap.gateway.notification_service import (
    NotificationService,
    NotificationServiceConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r(label: str, confidence: float = 0.9) -> DetectionResult:
    """Build a minimal DetectionResult for tests."""
    return DetectionResult(
        label=label,
        confidence=confidence,
        bbox=BBox(x=0.0, y=0.0, width=10.0, height=10.0),
    )


def _frame() -> np.ndarray:
    """Standard black 480x640 frame for detector tests."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _has_onnxruntime() -> bool:
    """True when the optional ``onnxruntime`` package is importable.

    The seg detector wrapper imports ``onnxruntime`` at module load
    time, so a missing extra turns the wrapper's tests into a hard
    ``ModuleNotFoundError``. We probe with ``importlib.util`` (a
    no-op when the package is present) and let ``skipif`` handle
    the gate.
    """
    return importlib.util.find_spec("onnxruntime") is not None


def _detector_with(
    *,
    mock_labels: list[str] | None = None,
    target_classes: list[str] | None = None,
    delay: float = 0.0,
) -> MockDetectorAdapter:
    cfg = DetectorConfig(
        type="mock",
        target_classes=target_classes if target_classes is not None else ["person"],
        mock={
            "labels": mock_labels or ["person", "car", "dog"],
            "confidence_range": [0.8, 0.9],
            "delay_seconds": delay,
        },
    )
    detector = MockDetectorAdapter(cfg)
    detector.load_model()
    return detector


# ---------------------------------------------------------------------------
# Helper: filter_results_by_classes
# ---------------------------------------------------------------------------


class TestFilterHelper:
    def test_empty_target_classes_returns_empty(self) -> None:
        assert filter_results_by_classes([_r("person")], []) == []

    def test_none_target_classes_passes_through(self) -> None:
        # ``None`` is the documented "no filter" sentinel: the
        # caller has delegated the contract enforcement elsewhere
        # and expects the input to be returned unchanged (modulo
        # a defensive copy).
        out = filter_results_by_classes([_r("person"), _r("car")], None)
        assert [r.label for r in out] == ["person", "car"]

    def test_keeps_only_matching_labels(self) -> None:
        results = [_r("person"), _r("car"), _r("dog")]
        out = filter_results_by_classes(results, ["person"])
        assert [r.label for r in out] == ["person"]

    def test_keeps_multiple_target_classes(self) -> None:
        results = [_r("person"), _r("car"), _r("dog")]
        out = filter_results_by_classes(results, ["person", "car"])
        assert sorted(r.label for r in out) == ["car", "person"]

    def test_case_insensitive_match(self) -> None:
        # User typed "Person" in the UI; the COCO label is "person".
        out = filter_results_by_classes([_r("person")], ["Person"])
        assert len(out) == 1

    def test_case_insensitive_label_side(self) -> None:
        # Defensive: the model might emit "PERSON" (unusual but possible
        # if a custom adapter capitalises). The filter must still match.
        out = filter_results_by_classes([_r("PERSON")], ["person"])
        assert len(out) == 1

    def test_whitespace_in_target_is_stripped(self) -> None:
        out = filter_results_by_classes([_r("person")], ["  person  "])
        assert len(out) == 1

    def test_blank_entries_ignored(self) -> None:
        # Hand-edited config files can produce empty / whitespace
        # entries; those must not match *every* label and leak data.
        out = filter_results_by_classes(
            [_r("person"), _r("car")], ["", "   ", "person"]
        )
        assert [r.label for r in out] == ["person"]

    def test_all_blank_target_classes_returns_empty(self) -> None:
        out = filter_results_by_classes([_r("person")], ["", "   "])
        assert out == []

    def test_none_entries_ignored(self) -> None:
        out = filter_results_by_classes([_r("person")], [None, "person"])  # type: ignore[list-item]
        assert len(out) == 1

    def test_input_list_not_mutated(self) -> None:
        results = [_r("person"), _r("car")]
        snapshot = list(results)
        filter_results_by_classes(results, ["person"])
        assert results == snapshot

    def test_no_match_returns_empty(self) -> None:
        out = filter_results_by_classes([_r("car")], ["person"])
        assert out == []


# ---------------------------------------------------------------------------
# Mock detector adapter
# ---------------------------------------------------------------------------


class TestMockDetectorContract:
    def test_filters_to_target_classes(self) -> None:
        detector = _detector_with(
            mock_labels=["person", "car", "dog"],
            target_classes=["car"],
        )
        # Force 100% emission by stubbing the probability — easier
        # than re-running until every label is present.
        import random
        random.seed(0)
        random.random = lambda: 0.0  # always pass the < 0.7 gate
        results = detector.detect(_frame())
        assert {r.label for r in results} == {"car"}

    def test_empty_target_classes_emits_nothing(self) -> None:
        detector = _detector_with(
            mock_labels=["person", "car", "dog"],
            target_classes=[],
        )
        import random
        random.random = lambda: 0.0
        results = detector.detect(_frame())
        assert results == []

    def test_default_target_classes_keeps_person(self) -> None:
        # Default target_classes is ["person"]; with mock labels
        # containing "person" the detector must always emit only person.
        detector = _detector_with(
            mock_labels=["person", "car", "dog"],
            target_classes=None,  # use the default
        )
        import random
        random.random = lambda: 0.0
        results = detector.detect(_frame())
        for r in results:
            assert r.label == "person"

    def test_case_insensitive_at_adapter(self) -> None:
        detector = _detector_with(
            mock_labels=["person", "car"],
            target_classes=["CAR"],
        )
        import random
        random.random = lambda: 0.0
        results = detector.detect(_frame())
        assert {r.label for r in results} == {"car"}


# ---------------------------------------------------------------------------
# Seg detector adapter (predict() is exercised via the wrapper, with
# the onnx engine stubbed out — the wrapper is what enforces the
# contract, so testing it in isolation is sufficient).
# ---------------------------------------------------------------------------


class TestSegDetectorContract:
    """Contract tests for the seg detector wrapper.

    The wrapper is the layer that enforces ``target_classes`` on top
    of the onnx engine, so testing the wrapper's ``predict()`` in
    isolation is sufficient — the underlying onnx engine is stubbed
    out and we only need the module to be importable.

    These tests require the optional ``onnxruntime`` dependency
    (it is imported at module load time by ``_seg_ort``). When the
    extra is not installed, the whole class is skipped.
    """

    pytestmark = pytest.mark.skipif(
        not _has_onnxruntime(),
        reason="onnxruntime is not installed",
    )

    def _make_wrapper(self, target_classes: list[str]):
        """Build a ``_SegWrapper`` whose onnx engine is stubbed.

        Lets us drive the wrapper's ``predict()`` deterministically
        without touching onnxruntime or a real model file.
        """
        from larksnap.adapters.detector.seg_adapter import _SegWrapper

        cfg = DetectorConfig(
            type="seg",
            target_classes=target_classes,
            seg={"model_path": "ignored", "provider": "cpu"},
        )
        wrapper = _SegWrapper.__new__(_SegWrapper)
        wrapper._coco_names = ["person", "car", "dog"]
        wrapper._target_classes = list(cfg.target_classes)
        # Stub the underlying onnx engine to return a fixed SegResult.
        wrapper._ort = MagicMock()
        return wrapper

    def _seg_result(self, labels: list[str]):
        from larksnap.adapters.detector._seg_ort import SegResult

        n = len(labels)
        name_to_id = {"person": 0, "car": 1, "dog": 2}
        return SegResult(
            boxes=np.zeros((n, 4), dtype=np.float32),
            scores=np.full((n,), 0.9, dtype=np.float32),
            class_ids=np.array([name_to_id[l] for l in labels], dtype=np.int64),
            masks=None,
        )

    def test_filters_to_target_classes(self) -> None:
        wrapper = self._make_wrapper(["person"])
        wrapper._ort.predict.return_value = self._seg_result(
            ["person", "car", "dog"]
        )
        results = wrapper.predict(_frame())
        assert [r.label for r in results] == ["person"]

    def test_empty_target_classes_emits_nothing(self) -> None:
        wrapper = self._make_wrapper([])
        wrapper._ort.predict.return_value = self._seg_result(
            ["person", "car", "dog"]
        )
        assert wrapper.predict(_frame()) == []

    def test_case_insensitive_at_adapter(self) -> None:
        wrapper = self._make_wrapper(["CAR"])
        wrapper._ort.predict.return_value = self._seg_result(["car", "person"])
        results = wrapper.predict(_frame())
        assert [r.label for r in results] == ["car"]


# ---------------------------------------------------------------------------
# Notification service: defense-in-depth filter
# ---------------------------------------------------------------------------


def _service_with_targets(
    target_classes: list[str] | None,
) -> tuple[NotificationService, MagicMock]:
    cfg = NotificationServiceConfig(
        target_classes=target_classes,
    )
    notifier = MagicMock()
    event_bus = MagicMock()
    svc = NotificationService(config=cfg, notifier=notifier, event_bus=event_bus)
    return svc, notifier


class TestNotificationServiceFilter:
    def test_drops_non_target_classes(self) -> None:
        svc, notifier = _service_with_targets(["person"])
        # ``saved=True`` opts into the dispatch path so the
        # target-class filter is the layer under test (rather
        # than the saved check short-circuiting first).
        svc.handle_results([_r("car")], frame=None, saved=True)
        notifier.send_message.assert_not_called()

    def test_keeps_target_classes(self) -> None:
        svc, notifier = _service_with_targets(["person"])
        svc.handle_results([_r("person")], frame=None, saved=True)
        notifier.send_message.assert_called_once()

    def test_empty_target_drops_everything(self) -> None:
        # An empty monitoring set means "monitor nothing" at every
        # layer, including the defense-in-depth one in the notifier.
        svc, notifier = _service_with_targets([])
        svc.handle_results(
            [_r("person"), _r("car")], frame=None, saved=True,
        )
        notifier.send_message.assert_not_called()

    def test_none_target_disables_filter_at_this_layer(self) -> None:
        # ``None`` means the service trusts the detector adapter's
        # filter and forwards whatever arrives. The detector adapter
        # is still the gatekeeper in the default config.
        svc, notifier = _service_with_targets(None)
        svc.handle_results([_r("person")], frame=None, saved=True)
        notifier.send_message.assert_called_once()

    def test_case_insensitive(self) -> None:
        svc, notifier = _service_with_targets(["PERSON"])
        svc.handle_results([_r("person")], frame=None, saved=True)
        notifier.send_message.assert_called_once()

    def test_mixed_target_keeps_only_matching(self) -> None:
        svc, notifier = _service_with_targets(["person", "car"])
        # ``dog`` is outside the target set and must be dropped.
        # All matching detections in the same batch are coalesced
        # into a single notification (see the multi-class
        # aggregation in ``_dispatch_aggregated``), so ``send_message``
        # is called exactly once with the whole batch.
        svc.handle_results(
            [_r("person"), _r("car"), _r("dog"), _r("person")],
            frame=None,
            saved=True,
        )
        notifier.send_message.assert_called_once()
        message = notifier.send_message.call_args[0][0]
        # The rendered content references the multi-class placeholders
        # from the default template, and the count is 2 (dog filtered
        # out; two person + one car collapses to 2 distinct labels in
        # the labels_summary line).
        assert "person" in message.content
        assert "car" in message.content
        assert "dog" not in message.content
        assert "2" in message.content  # labels_count == 2

    def test_whitespace_in_target_classes_stripped(self) -> None:
        # The detector/notification layer must apply the same
        # whitespace handling as the helper: a user-typed
        # ``"  person  "`` matches ``"person"``.
        svc, notifier = _service_with_targets(["  person  "])
        svc.handle_results([_r("person")], frame=None, saved=True)
        notifier.send_message.assert_called_once()

    def test_blank_target_entries_dropped_not_matched(self) -> None:
        # An empty / whitespace-only entry in the monitoring set
        # must NOT match every label. This is a defense-in-depth
        # check: a hand-edited config with a stray blank line
        # must not silently widen the monitoring set.
        svc, notifier = _service_with_targets(["", "   ", "person"])
        svc.handle_results([_r("person")], frame=None, saved=True)
        notifier.send_message.assert_called_once()
        svc.handle_results([_r("car")], frame=None, saved=True)
        notifier.send_message.assert_called_once()  # only "person" matched

    def test_empty_results_is_noop(self) -> None:
        # An empty results list is the documented "no detection"
        # path: the service must drop it on the floor without
        # touching the notifier. The snapshot service's
        # ``no_results`` outcome produces this input.
        svc, notifier = _service_with_targets(["person"])
        svc.handle_results([], frame=None, saved=True)
        notifier.send_message.assert_not_called()

    def test_disabled_notifier_short_circuits_before_template(self) -> None:
        # When the user has run ``/stop``, the dispatch must
        # short-circuit *before* the template render so a slow
        # custom template cannot keep firing on every frame.
        svc, notifier = _service_with_targets(["person"])
        svc.disable_notification()
        svc.handle_results([_r("person")], frame=None, saved=True)
        notifier.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-class notification aggregation
# ---------------------------------------------------------------------------


class TestMultiClassAggregation:
    """The dispatch path coalesces every detection in one frame into a single
    notification message, rather than sending N near-duplicate messages."""

    def test_multi_class_sends_one_message(self) -> None:
        svc, notifier = _service_with_targets(["person", "car", "dog"])
        svc.handle_results(
            [_r("person", 0.9), _r("car", 0.8), _r("dog", 0.7)],
            frame=None,
            saved=True,
        )
        notifier.send_message.assert_called_once()

    def test_single_class_still_sends_one_message(self) -> None:
        # Backwards-compat: the aggregated dispatcher must not turn
        # a one-result batch into a different shape — single-class
        # users see exactly the behaviour they had before, just
        # routed through the same code path.
        svc, notifier = _service_with_targets(["person"])
        svc.handle_results([_r("person", 0.9)], frame=None, saved=True)
        notifier.send_message.assert_called_once()

    def test_labels_summary_orders_by_confidence_desc(self) -> None:
        svc, notifier = _service_with_targets(["person", "car", "dog"])
        # Deliberately insert in a non-sorted order.
        svc.handle_results(
            [_r("dog", 0.5), _r("car", 0.95), _r("person", 0.7)],
            frame=None,
            saved=True,
        )
        content = notifier.send_message.call_args[0][0].content
        # The most confident detection must appear first.
        car_idx = content.index("car")
        person_idx = content.index("person")
        dog_idx = content.index("dog")
        assert car_idx < person_idx < dog_idx

    def test_labels_summary_format(self) -> None:
        svc, notifier = _service_with_targets(["person", "car"])
        svc.handle_results(
            [_r("person", 0.9), _r("car", 0.75)],
            frame=None,
            saved=True,
        )
        content = notifier.send_message.call_args[0][0].content
        # 90% and 75% are the rounded confidences.
        assert "person (90%)" in content
        assert "car (75%)" in content

    def test_labels_count_in_message(self) -> None:
        svc, notifier = _service_with_targets(["person", "car", "dog"])
        svc.handle_results(
            [_r("person"), _r("car"), _r("dog")],
            frame=None,
            saved=True,
        )
        content = notifier.send_message.call_args[0][0].content
        # The default template uses "{labels_count} 个目标".
        assert "3" in content

    def test_legacy_template_still_works(self) -> None:
        # A template that only uses ``{label}`` / ``{confidence}``
        # — written before the multi-class placeholders existed —
        # must still render without raising, and ``{label}`` must
        # fall back to the top-confidence detection.
        cfg = NotificationServiceConfig(
            target_classes=["person", "car"],
            message_template="[LarkSnap] {label} @ {confidence:.0%}",
        )
        notifier = MagicMock()
        event_bus = MagicMock()
        svc = NotificationService(config=cfg, notifier=notifier, event_bus=event_bus)
        svc.handle_results(
            [_r("car", 0.95), _r("person", 0.80)],
            frame=None,
            saved=True,
        )
        notifier.send_message.assert_called_once()
        content = notifier.send_message.call_args[0][0].content
        # Top confidence is "car" at 95%.
        assert "car" in content
        assert "95%" in content

    def test_legacy_template_unknown_placeholder_falls_back(self) -> None:
        # A template that references an unknown placeholder
        # (``{nonexistent}``) must not crash. The dispatcher catches
        # ``KeyError`` from ``str.format`` and re-renders with the
        # legacy kwarg set, leaving the unknown placeholder blank
        # in the user's template rather than failing the whole
        # notification.
        cfg = NotificationServiceConfig(
            target_classes=["person"],
            message_template="[LarkSnap] {label} | {nonexistent}",
        )
        notifier = MagicMock()
        event_bus = MagicMock()
        svc = NotificationService(config=cfg, notifier=notifier, event_bus=event_bus)
        svc.handle_results([_r("person")], frame=None, saved=True)
        notifier.send_message.assert_called_once()
        content = notifier.send_message.call_args[0][0].content
        assert "person" in content

    def test_aggregated_publishes_one_event_per_label(self) -> None:
        # The aggregated message is a presentation concern; the
        # event bus must still get per-class signals so stats
        # panels / downstream consumers keep working.
        svc, notifier = _service_with_targets(["person", "car"])
        event_bus = MagicMock()
        svc._event_bus = event_bus
        svc.handle_results(
            [_r("person", 0.9), _r("car", 0.8), _r("person", 0.85)],
            frame=None,
            saved=True,
        )
        notifier.send_message.assert_called_once()
        # 3 results → 3 NOTIFICATION_SENT events.
        sent_events = [
            call
            for call in event_bus.publish.call_args_list
            if call.kwargs.get("type") == EventType.NOTIFICATION_SENT
            or (call.args and call.args[0].type == EventType.NOTIFICATION_SENT)
        ]
        assert len(sent_events) == 3

    def test_default_template_uses_multi_class_placeholders(self) -> None:
        # Sanity-check the default template — the project should
        # ship a multi-class-aware default so a fresh install
        # shows the aggregated format out of the box.
        cfg = NotificationServiceConfig()
        assert "{labels_summary}" in cfg.message_template
        assert "{labels_count}" in cfg.message_template
