"""Tests for :class:`larksnap.config.service.ConfigService`.

The service is the runtime source of truth for ``AppConfig`` and is
the only path through which the Feishu ``/config`` command mutates
state. These tests pin down:

* Dotted-path resolution and unknown-path errors
* Type coercion via pydantic ``TypeAdapter`` (string → int, etc.)
* JSON parsing errors
* Disk persistence (``save_config``) after a successful ``set``
* Subscriber pub/sub semantics
* ``needs_restart`` classification
* In-memory-only mode (no ``config_path``) is silent and safe
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from larksnap.config.loader import save_config
from larksnap.config.models import AppConfig
from larksnap.config.service import (
    RESTART_REQUIRED_PREFIXES,
    ConfigService,
)
from larksnap.utils.exceptions import ConfigError


def _reload_config(path: Path) -> AppConfig:
    """Reload an AppConfig from ``path`` using the same loader YAML flavor.

    ``save_config`` uses :py:func:`yaml.dump` (NOT ``safe_dump``), so
    the on-disk file contains a ``!!python/tuple`` tag for
    ``detector.mock.confidence_range``. We re-read with a compatible
    loader so the round-trip works in tests.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.unsafe_load(f)
    return AppConfig(**raw)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> AppConfig:
    """A fresh in-memory AppConfig — no disk side effects from the fixture."""
    return AppConfig()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """A per-test YAML file the ConfigService can persist to.

    Pre-populated with a known-good AppConfig so the test starts from
    a deterministic state. The same path is re-used by the service so
    the on-disk file mirrors the in-memory ``AppConfig`` after each
    ``set``.
    """
    p = tmp_path / "config.yaml"
    save_config(AppConfig(), str(p))
    return p


@pytest.fixture
def service(cfg: AppConfig, config_path: Path) -> ConfigService:
    """A ConfigService bound to a fresh AppConfig and a tmp YAML file."""
    return ConfigService(cfg, config_path=config_path)


# ---------------------------------------------------------------------------
# Dotted-path resolution
# ---------------------------------------------------------------------------


class TestGet:
    def test_returns_scalar_value(self, service: ConfigService) -> None:
        assert service.get("gateway.notification_interval") == 30

    def test_returns_list_value(self, service: ConfigService) -> None:
        assert service.get("detector.target_classes") == ["person"]

    def test_returns_nested_submodel(self, service: ConfigService) -> None:
        cam = service.get("camera")
        # ``cam`` is a ``CameraConfig`` instance — compare against the
        # pydantic tree itself, not ``==`` on the BaseModel, because
        # pydantic v2's ``__eq__`` is field-based but the test stays
        # explicit about the fields it cares about.
        assert cam.device_index == 0
        assert cam.fps == 30

    def test_empty_path_raises(self, service: ConfigService) -> None:
        with pytest.raises(ConfigError, match="路径不能为空"):
            service.get("")

    def test_unknown_path_raises(self, service: ConfigService) -> None:
        with pytest.raises(ConfigError, match="未知路径"):
            service.get("does.not.exist")

    def test_path_with_empty_segment_raises(
        self, service: ConfigService
    ) -> None:
        with pytest.raises(ConfigError, match="路径格式无效"):
            service.get("detector..target_classes")

    def test_path_through_scalar_raises(self, service: ConfigService) -> None:
        # ``detector.target_classes`` is a list[str] — there is no
        # sub-field to descend into, so a further ``.x`` is invalid.
        with pytest.raises(ConfigError):
            service.get("detector.target_classes.x")


# ---------------------------------------------------------------------------
# list_paths
# ---------------------------------------------------------------------------


class TestListPaths:
    def test_includes_every_leaf(self, service: ConfigService) -> None:
        paths = service.list_paths()
        # Spot-check a few well-known leaves.
        assert "camera.device_index" in paths
        assert "detector.confidence_threshold" in paths
        assert "notifier.message_template" in paths
        assert "gateway.notification_interval" in paths

    def test_filter_by_prefix(self, service: ConfigService) -> None:
        camera_paths = service.list_paths("camera.")
        assert all(p.startswith("camera.") for p in camera_paths)
        assert "camera.fps" in camera_paths
        # No detector / notifier leaves should leak into a camera prefix.
        assert not any(p.startswith("detector.") for p in camera_paths)

    def test_no_matches_returns_empty_list(
        self, service: ConfigService
    ) -> None:
        assert service.list_paths("nope.") == []


# ---------------------------------------------------------------------------
# needs_restart
# ---------------------------------------------------------------------------


class TestNeedsRestart:
    @pytest.mark.parametrize(
        "path",
        [
            "camera.device_index",
            "camera.width",
            "detector.type",
            "detector.mock.labels",
            "detector.seg.model_path",
            "recorder.fps",
            "logging.level",
            "service.name",
        ],
    )
    def test_restart_required_paths(self, service: ConfigService, path: str) -> None:
        assert service.needs_restart(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "detector.confidence_threshold",
            "detector.target_classes",
            "notifier.message_template",
            "notifier.send_image",
            "gateway.notification_interval",
            "gateway.snapshot_dir",
        ],
    )
    def test_live_paths(self, service: ConfigService, path: str) -> None:
        assert service.needs_restart(path) is False

    def test_prefix_list_includes_expected_buckets(self) -> None:
        # Sanity check the module-level constant so a future edit
        # doesn't silently demote a hot-swap field.
        for prefix in (
            "camera.",
            "recorder.",
            "logging.",
            "service.",
            "detector.type",
        ):
            assert prefix in RESTART_REQUIRED_PREFIXES


# ---------------------------------------------------------------------------
# set: happy path
# ---------------------------------------------------------------------------


class TestSetHappyPath:
    def test_updates_scalar_field(self, service: ConfigService) -> None:
        old, new, status = service.set(
            "gateway.notification_interval", "60"
        )
        assert old == 30
        assert new == 60
        assert status == "applied"
        # In-memory value reflects the change immediately.
        assert service.get("gateway.notification_interval") == 60

    def test_coerces_string_to_int(self, service: ConfigService) -> None:
        # A user typing ``30`` (not ``"30"``) — JSON ``30`` is an int,
        # but the type-adapter also accepts the string form to match
        # common shell-quoting mistakes.
        old, new, status = service.set(
            "gateway.notification_interval", '"30"'
        )
        assert new == 30
        assert isinstance(new, int)
        assert status == "applied"

    def test_coerces_string_to_float(self, service: ConfigService) -> None:
        old, new, status = service.set(
            "detector.confidence_threshold", "0.7"
        )
        assert new == pytest.approx(0.7)
        assert status == "applied"

    def test_updates_list_field_with_json_array(
        self, service: ConfigService
    ) -> None:
        old, new, status = service.set(
            "detector.target_classes", '["person", "car", "dog"]'
        )
        assert old == ["person"]
        assert new == ["person", "car", "dog"]
        assert status == "applied"

    def test_updates_bool_field(self, service: ConfigService) -> None:
        old, new, status = service.set("notifier.send_image", "false")
        assert old is True
        assert new is False
        assert status == "applied"

    def test_status_restart_required_for_camera(
        self, service: ConfigService
    ) -> None:
        _, _, status = service.set("camera.fps", "60")
        assert status == "restart_required"

    def test_status_restart_required_for_logging(
        self, service: ConfigService
    ) -> None:
        _, _, status = service.set("logging.level", '"DEBUG"')
        assert status == "restart_required"

    def test_persists_to_disk(self, cfg: AppConfig, config_path: Path) -> None:
        service = ConfigService(cfg, config_path=config_path)
        service.set("gateway.notification_interval", "120")
        # Reload the YAML from disk and check the value stuck.
        reloaded = _reload_config(config_path)
        assert reloaded.gateway.notification_interval == 120


# ---------------------------------------------------------------------------
# set: error paths
# ---------------------------------------------------------------------------


class TestSetErrors:
    def test_invalid_json_raises(self, service: ConfigService) -> None:
        with pytest.raises(ConfigError, match="合法 JSON"):
            service.set("gateway.notification_interval", "not a number")

    def test_unknown_path_raises(self, service: ConfigService) -> None:
        with pytest.raises(ConfigError, match="未知路径"):
            service.set("nope.field", "1")

    def test_type_mismatch_raises(self, service: ConfigService) -> None:
        # ``gateway.notification_interval`` is int; passing a dict
        # cannot be coerced.
        with pytest.raises(ConfigError, match="类型校验失败"):
            service.set("gateway.notification_interval", '{"x": 1}')

    def test_validation_failure_raises(self, service: ConfigService) -> None:
        # ``confidence_threshold`` is ``float`` and constrained to
        # reasonable ranges by the model — passing a wildly negative
        # value (after coercion) would normally fail pydantic's own
        # validators. The mock has none here, so the test focuses on
        # the "wrong type" path which is the more common user error.
        with pytest.raises(ConfigError):
            service.set("detector.confidence_threshold", '"not-a-float"')

    def test_empty_path_raises(self, service: ConfigService) -> None:
        with pytest.raises(ConfigError, match="路径不能为空"):
            service.set("", "1")

    def test_failed_set_does_not_persist(
        self, cfg: AppConfig, config_path: Path
    ) -> None:
        service = ConfigService(cfg, config_path=config_path)
        with pytest.raises(ConfigError):
            service.set("nope", "1")
        # The on-disk file should be untouched: reload and compare.
        reloaded = _reload_config(config_path)
        assert reloaded.gateway.notification_interval == 30


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_writes_to_disk(
        self, cfg: AppConfig, config_path: Path
    ) -> None:
        service = ConfigService(cfg, config_path=config_path)
        cfg.gateway.notification_interval = 99
        service.save()
        reloaded = _reload_config(config_path)
        assert reloaded.gateway.notification_interval == 99

    def test_no_path_skips_disk_silently(self, cfg: AppConfig) -> None:
        # ``ConfigService()`` without a path is a valid in-memory
        # service. ``save()`` must not raise; it just logs a debug
        # message and returns.
        service = ConfigService(cfg, config_path=None)
        cfg.gateway.notification_interval = 77
        service.save()
        assert service.get("gateway.notification_interval") == 77

    def test_in_memory_service_set_skips_disk(
        self, cfg: AppConfig
    ) -> None:
        # The same applies to ``set`` — without a backing file, the
        # in-memory mutation succeeds, status is still reported, and
        # nothing blows up.
        service = ConfigService(cfg, config_path=None)
        old, new, status = service.set(
            "gateway.notification_interval", "5"
        )
        assert old == 30
        assert new == 5
        assert status == "applied"


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------


class TestSubscribe:
    def test_subscriber_receives_path_old_new(
        self, service: ConfigService
    ) -> None:
        cb = MagicMock()
        service.subscribe(cb)
        service.set("gateway.notification_interval", "45")
        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "gateway.notification_interval"
        assert args[1] == 30
        assert args[2] == 45

    def test_multiple_subscribers_all_called(
        self, service: ConfigService
    ) -> None:
        cb1 = MagicMock()
        cb2 = MagicMock()
        service.subscribe(cb1)
        service.subscribe(cb2)
        service.set("gateway.notification_interval", "11")
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_unsubscribe_stops_callbacks(
        self, service: ConfigService
    ) -> None:
        cb = MagicMock()
        unsubscribe = service.subscribe(cb)
        service.set("gateway.notification_interval", "5")
        unsubscribe()
        service.set("gateway.notification_interval", "9")
        # Only the first set should have triggered the callback.
        assert cb.call_count == 1

    def test_unsubscribe_unknown_callback_is_noop(
        self, service: ConfigService
    ) -> None:
        cb = MagicMock()
        unsubscribe = service.subscribe(cb)
        unsubscribe()
        # Calling unsubscribe a second time must not raise — defensive
        # code that tears down twice is common in tests.
        unsubscribe()

    def test_subscriber_exception_does_not_abort_chain(
        self, service: ConfigService
    ) -> None:
        bad = MagicMock(side_effect=RuntimeError("boom"))
        good = MagicMock()
        service.subscribe(bad)
        service.subscribe(good)
        # Should not raise; the second subscriber still receives the
        # event and the save still happens.
        service.set("gateway.notification_interval", "7")
        good.assert_called_once()
        assert service.get("gateway.notification_interval") == 7

    def test_subscriber_can_unsubscribe_self(
        self, service: ConfigService
    ) -> None:
        # Mutating the subscriber list from inside a callback is a
        # classic iteration hazard. The service iterates over a copy
        # to keep this safe.
        events: list[int] = []

        def cb(_path: str, _old: Any, _new: Any) -> None:
            events.append(_new)
            unsubscribe()

        unsubscribe = service.subscribe(cb)
        service.set("gateway.notification_interval", "1")
        service.set("gateway.notification_interval", "2")
        # First event captured, then we unsubscribed.
        assert events == [1]


# ---------------------------------------------------------------------------
# Thread-safety sanity check
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_sets_serialize_via_lock(
        self, service: ConfigService
    ) -> None:
        # Stress the lock by hammering ``set`` from a few threads.
        # We don't assert exact end-state (race-free numerical
        # convergence is the OS scheduler's job), but the service
        # must not raise or deadlock.
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                for j in range(20):
                    service.set(
                        "gateway.notification_interval", str(i * 100 + j)
                    )
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert errors == []
        # Whatever the final value is, it's an int — the type
        # coercion held even under contention.
        assert isinstance(
            service.get("gateway.notification_interval"), int
        )


# ---------------------------------------------------------------------------
# Direct property access
# ---------------------------------------------------------------------------


class TestProperties:
    def test_config_returns_same_instance(
        self, cfg: AppConfig, service: ConfigService
    ) -> None:
        # ``service.config`` is the *same* object as the AppConfig the
        # caller injected. The service does not own the model; it
        # only provides a controlled write API on top of it.
        assert service.config is cfg

    def test_config_path_returns_same_path(
        self, service: ConfigService, config_path: Path
    ) -> None:
        assert service.config_path == config_path

    def test_config_path_none_when_unspecified(
        self, cfg: AppConfig
    ) -> None:
        service = ConfigService(cfg, config_path=None)
        assert service.config_path is None
